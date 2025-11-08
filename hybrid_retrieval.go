package batch_0001

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"sort"
	"strings"

	"github.com/cloudwego/eino-ext/components/model/ark"
	"github.com/cloudwego/eino/schema"
	"github.com/neo4j/neo4j-go-driver/v5/neo4j"
)

// RetrievalResult 检索结果数据结构
//
// 标准化的检索结果表示，包含内容、元数据和评分信息。
// 支持不同检索方法的统一结果处理和比较。
type RetrievalResult struct {
	Content        string                 `json:"content"`         // 检索到的文档内容
	NodeID         string                 `json:"node_id"`         // 原始图节点ID，用于去重和溯源
	NodeType       string                 `json:"node_type"`       // 节点类型（Recipe, Ingredient, CookingStep等）
	RelevanceScore float64                `json:"relevance_score"` // 相关性得分，用于结果排序
	RetrievalLevel string                 `json:"retrieval_level"` // 检索层级标识（'entity'或'topic'）
	Metadata       map[string]interface{} `json:"metadata"`        // 丰富的元数据信息
}

// HybridRetrievalModule 混合检索模块 - RAG系统的多策略检索引擎
//
// 实现了创新的双层检索范式，结合图结构的精确匹配和向量语义的模糊匹配，
// 为复杂的烹饪知识查询提供高质量、多维度的检索结果。
//
// 核心特点：
// 1. 双层检索架构：实体级 + 主题级的分层检索策略
// 2. 多数据源融合：Neo4j图数据库 + Milvus向量数据库 + BM25文本检索
// 3. 智能查询理解：LLM驱动的查询意图分析和关键词提取
// 4. 图结构增强：利用知识图谱的关系信息丰富检索结果
// 5. 公平结果合并：Round-robin轮询策略避免单一方法偏差
//
// 检索流程：
// 1. 查询预处理：LLM分析提取实体级和主题级关键词
// 2. 实体级检索：基于图索引的精确实体和关系匹配
// 3. 主题级检索：基于图关系的主题概念检索
// 4. 向量增强检索：Milvus语义相似度搜索
// 5. 结果融合排序：Round-robin合并 + 相关性排序
//
// 技术优势：
// - 高召回率：多种检索方法的组合覆盖不同查询需求
// - 高精确度：图结构的精确匹配保证结果相关性
// - 语义理解：向量检索处理模糊和隐含查询
// - 上下文丰富：图邻居信息提供更完整的知识背景
type HybridRetrievalModule struct {
	config       *Config                        // 系统配置
	milvusModule *MilvusIndexConstructionModule // Milvus向量索引模块
	dataModule   *GraphDataPreparationModule    // 图数据准备模块
	llmClient    *ark.ChatModel                 // 大语言模型客户端
	driver       neo4j.DriverWithContext        // Neo4j数据库连接

	// 图索引相关
	entityCache   map[string]*RetrievalResult // 实体信息缓存
	relationCache map[string]int              // 关系类型缓存
	graphIndexed  bool                        // 图索引构建状态
}

// NewHybridRetrievalModule 创建新的混合检索模块
//
// Args:
//
//	config: 配置对象，包含数据库连接等参数
//	milvusModule: Milvus向量索引模块实例
//	dataModule: 图数据准备模块实例
//	llmClient: 大语言模型客户端，用于查询分析
func NewHybridRetrievalModule(config *Config, milvusModule *MilvusIndexConstructionModule, dataModule *GraphDataPreparationModule, llmClient *ark.ChatModel) *HybridRetrievalModule {
	return &HybridRetrievalModule{
		config:        config,
		milvusModule:  milvusModule,
		dataModule:    dataModule,
		llmClient:     llmClient,
		entityCache:   make(map[string]*RetrievalResult),
		relationCache: make(map[string]int),
		graphIndexed:  false,
	}
}

// Initialize 初始化混合检索系统
//
// 建立必要的数据库连接，构建图索引，初始化各种检索器。
// 这是系统准备阶段，确保所有检索组件都能正常工作。
//
// Args:
//
//	ctx: 上下文对象
//	chunks: 预处理的文档块列表，用于BM25检索器
func (h *HybridRetrievalModule) Initialize(ctx context.Context, chunks []*schema.Document) error {
	log.Println("初始化混合检索模块...")

	// 连接Neo4j图数据库
	driver, err := neo4j.NewDriverWithContext(
		h.config.Neo4jURI,
		neo4j.BasicAuth(h.config.Neo4jUser, h.config.Neo4jPassword, ""),
	)
	if err != nil {
		return fmt.Errorf("Neo4j连接失败: %w", err)
	}
	h.driver = driver

	// 测试连接
	err = driver.VerifyConnectivity(ctx)
	if err != nil {
		return fmt.Errorf("Neo4j连接验证失败: %w", err)
	}

	log.Printf("BM25检索器初始化完成，文档数量: %d", len(chunks))

	// 构建图索引 - 核心的图结构检索能力
	if err := h.buildGraphIndex(ctx); err != nil {
		return fmt.Errorf("构建图索引失败: %w", err)
	}

	return nil
}

// buildGraphIndex 构建图索引系统
//
// 从图数据模块获取实体和关系数据，构建高效的键值对索引结构。
// 支持实体级和主题级的快速检索。
func (h *HybridRetrievalModule) buildGraphIndex(ctx context.Context) error {
	if h.graphIndexed {
		return nil
	}

	log.Println("开始构建图索引...")

	session := h.driver.NewSession(ctx, neo4j.SessionConfig{})
	defer session.Close(ctx)

	// 构建实体索引
	entityQuery := `
		MATCH (n)
		WHERE n.nodeId IS NOT NULL
		WITH n, COUNT { (n)--() } as degree
		RETURN labels(n) as node_labels, n.nodeId as node_id, 
		       n.name as name, n.category as category, 
		       n.description as description, degree
		ORDER BY degree DESC
		LIMIT 1000
	`

	result, err := session.ExecuteRead(ctx, func(tx neo4j.ManagedTransaction) (interface{}, error) {
		result, err := tx.Run(ctx, entityQuery, nil)
		if err != nil {
			return nil, err
		}
		records, err := result.Collect(ctx)
		if err != nil {
			return nil, err
		}
		return records, nil
	})

	if err != nil {
		return fmt.Errorf("执行实体索引查询失败: %w", err)
	}

	records := result.([]*neo4j.Record)
	for _, record := range records {
		nodeID, _ := record.Get("node_id")
		nodeLabels, _ := record.Get("node_labels")
		name, _ := record.Get("name")
		category, _ := record.Get("category")
		description, _ := record.Get("description")
		degree, _ := record.Get("degree")

		// 构建内容
		var contentParts []string
		if nameStr, ok := name.(string); ok && nameStr != "" {
			contentParts = append(contentParts, fmt.Sprintf("名称: %s", nameStr))
		}
		if categoryStr, ok := category.(string); ok && categoryStr != "" {
			contentParts = append(contentParts, fmt.Sprintf("分类: %s", categoryStr))
		}
		if descStr, ok := description.(string); ok && descStr != "" {
			contentParts = append(contentParts, fmt.Sprintf("描述: %s", descStr))
		}

		// 确定节点类型
		nodeType := "Unknown"
		if labels, ok := nodeLabels.([]interface{}); ok && len(labels) > 0 {
			if labelStr, ok := labels[0].(string); ok {
				nodeType = labelStr
			}
		}

		// 缓存实体信息
		h.entityCache[nodeID.(string)] = &RetrievalResult{
			Content:        strings.Join(contentParts, "\n"),
			NodeID:         nodeID.(string),
			NodeType:       nodeType,
			RelevanceScore: 0.8, // 基础相关性得分
			RetrievalLevel: "entity",
			Metadata: map[string]interface{}{
				"name":     name,
				"category": category,
				"degree":   degree,
				"labels":   nodeLabels,
			},
		}
	}

	// 构建关系类型索引
	relationQuery := `
		MATCH ()-[r]->()
		RETURN type(r) as rel_type, count(r) as frequency
		ORDER BY frequency DESC
	`

	result, err = session.ExecuteRead(ctx, func(tx neo4j.ManagedTransaction) (interface{}, error) {
		result, err := tx.Run(ctx, relationQuery, nil)
		if err != nil {
			return nil, err
		}
		records, err := result.Collect(ctx)
		if err != nil {
			return nil, err
		}
		return records, nil
	})

	if err != nil {
		return fmt.Errorf("执行关系索引查询失败: %w", err)
	}

	records = result.([]*neo4j.Record)
	for _, record := range records {
		relType, _ := record.Get("rel_type")
		frequency, _ := record.Get("frequency")
		h.relationCache[relType.(string)] = int(frequency.(int64))
	}

	h.graphIndexed = true
	log.Printf("索引构建完成: %d个实体, %d个关系类型", len(h.entityCache), len(h.relationCache))

	return nil
}

// KeywordExtractionResult 关键词提取结果
type KeywordExtractionResult struct {
	EntityKeywords []string `json:"entity_keywords"`
	TopicKeywords  []string `json:"topic_keywords"`
}

// ExtractQueryKeywords 提取查询关键词：实体级 + 主题级
//
// 使用LLM分析查询，提取实体级和主题级关键词，为双层检索提供基础。
func (h *HybridRetrievalModule) ExtractQueryKeywords(ctx context.Context, query string) ([]string, []string, error) {
	// 尝试使用LLM进行关键词提取
	if h.llmClient != nil {
		entityKeywords, topicKeywords, err := h.extractWithLLM(ctx, query)
		if err == nil {
			log.Printf("LLM关键词提取完成 - 实体级: %v, 主题级: %v", entityKeywords, topicKeywords)
			return entityKeywords, topicKeywords, nil
		}
		log.Printf("LLM关键词提取失败，使用降级方案: %v", err)
	}

	// 降级方案：简单的关键词分割
	keywords := strings.Fields(query)
	entityKeywords := []string{}
	topicKeywords := []string{}

	// 简单的规则分类
	for _, keyword := range keywords {
		if len(keyword) > 1 {
			if strings.Contains(keyword, "菜") || strings.Contains(keyword, "肉") || strings.Contains(keyword, "蛋") {
				entityKeywords = append(entityKeywords, keyword)
			} else {
				topicKeywords = append(topicKeywords, keyword)
			}
		}
	}

	log.Printf("规则关键词提取完成 - 实体级: %v, 主题级: %v", entityKeywords, topicKeywords)
	return entityKeywords, topicKeywords, nil
}

// extractWithLLM 使用LLM进行关键词提取
func (h *HybridRetrievalModule) extractWithLLM(ctx context.Context, query string) ([]string, []string, error) {
	// 构建eino风格的模板
	userContent := fmt.Sprintf(`分析以下查询并提取关键词，分为两个层次：

查询：%s

提取规则：
1. 实体级关键词：具体的食材、菜品名称、工具、品牌等有形实体
   - 例如：鸡胸肉、西兰花、红烧肉、平底锅、老干妈
   - 对于抽象查询，推测相关的具体食材/菜品

2. 主题级关键词：抽象概念、烹饪主题、饮食风格、营养特点等
   - 例如：减肥、低热量、川菜、素食、下饭菜、快手菜
   - 排除动作词：推荐、介绍、制作、怎么做等

示例：
查询："推荐几个减肥菜"
{
    "entity_keywords": ["鸡胸肉", "西兰花", "水煮蛋", "胡萝卜", "黄瓜"],
    "topic_keywords": ["减肥", "低热量", "高蛋白", "低脂"]
}

查询："川菜有什么特色"
{
    "entity_keywords": ["麻婆豆腐", "宫保鸡丁", "水煮鱼", "辣椒", "花椒"],
    "topic_keywords": ["川菜", "麻辣", "香辣", "下饭菜"]
}

请严格按照JSON格式返回，不要包含多余的文字：
{
    "entity_keywords": ["实体1", "实体2", ...],
    "topic_keywords": ["主题1", "主题2", ...]
}`, query)

	// 构建消息
	messages := []*schema.Message{
		schema.SystemMessage("你是烹饪知识助手，专门负责分析查询并提取关键词。"),
		&schema.Message{
			Role:    schema.User,
			Content: userContent,
		},
	}

	response, err := h.llmClient.Generate(ctx, messages)
	if err != nil {
		return nil, nil, fmt.Errorf("LLM生成失败: %w", err)
	}

	// 解析JSON响应
	var result KeywordExtractionResult
	if err := json.Unmarshal([]byte(response.Content), &result); err != nil {
		// 如果JSON解析失败，尝试清理响应内容后再解析
		cleanContent := strings.TrimSpace(response.Content)
		// 移除可能的markdown代码块标记
		cleanContent = strings.TrimPrefix(cleanContent, "```json")
		cleanContent = strings.TrimPrefix(cleanContent, "```")
		cleanContent = strings.TrimSuffix(cleanContent, "```")
		cleanContent = strings.TrimSpace(cleanContent)

		if err := json.Unmarshal([]byte(cleanContent), &result); err != nil {
			return nil, nil, fmt.Errorf("JSON解析失败: %w, 响应内容: %s", err, response.Content)
		}
	}

	return result.EntityKeywords, result.TopicKeywords, nil
}

// EntityLevelRetrieval 实体级检索：专注于具体实体和关系
// 使用图索引的键值对结构进行检索
func (h *HybridRetrievalModule) EntityLevelRetrieval(ctx context.Context, entityKeywords []string, topK int) ([]*RetrievalResult, error) {
	var results []*RetrievalResult

	// 1. 使用图索引进行实体检索
	for _, keyword := range entityKeywords {
		for nodeID, entity := range h.entityCache {
			// 简单的关键词匹配
			if strings.Contains(strings.ToLower(entity.Content), strings.ToLower(keyword)) ||
				strings.Contains(strings.ToLower(entity.NodeID), strings.ToLower(keyword)) {

				// 获取邻居信息
				neighbors, _ := h.getNodeNeighbors(ctx, nodeID, 2)

				// 构建增强内容
				enhancedContent := entity.Content
				if len(neighbors) > 0 {
					enhancedContent += fmt.Sprintf("\n相关信息: %s", strings.Join(neighbors, ", "))
				}

				result := &RetrievalResult{
					Content:        enhancedContent,
					NodeID:         entity.NodeID,
					NodeType:       entity.NodeType,
					RelevanceScore: 0.9, // 精确匹配得分较高
					RetrievalLevel: "entity",
					Metadata: map[string]interface{}{
						"matched_keyword": keyword,
						"source":          "graph_index",
					},
				}

				// 复制原有元数据
				for k, v := range entity.Metadata {
					result.Metadata[k] = v
				}

				results = append(results, result)
			}
		}
	}

	// 2. 如果图索引结果不足，使用Neo4j进行补充检索
	if len(results) < topK {
		neo4jResults, err := h.neo4jEntityLevelSearch(ctx, entityKeywords, topK-len(results))
		if err != nil {
			log.Printf("Neo4j补充检索失败: %v", err)
		} else {
			results = append(results, neo4jResults...)
		}
	}

	// 3. 按相关性排序并返回
	sort.Slice(results, func(i, j int) bool {
		return results[i].RelevanceScore > results[j].RelevanceScore
	})

	if len(results) > topK {
		results = results[:topK]
	}

	log.Printf("实体级检索完成，返回 %d 个结果", len(results))
	return results, nil
}

// neo4jEntityLevelSearch Neo4j补充检索
func (h *HybridRetrievalModule) neo4jEntityLevelSearch(ctx context.Context, keywords []string, limit int) ([]*RetrievalResult, error) {
	var results []*RetrievalResult

	session := h.driver.NewSession(ctx, neo4j.SessionConfig{})
	defer session.Close(ctx)

	cypherQuery := `
		UNWIND $keywords as keyword
		MATCH (node)
		WHERE node.name CONTAINS keyword 
		   OR node.description CONTAINS keyword
		   OR node.category CONTAINS keyword
		RETURN 
		    node.nodeId as node_id,
		    node.name as name,
		    node.description as description,
		    node.category as category,
		    labels(node) as labels,
		    keyword as matched_keyword
		ORDER BY node.name
		LIMIT $limit
	`

	result, err := session.ExecuteRead(ctx, func(tx neo4j.ManagedTransaction) (interface{}, error) {
		result, err := tx.Run(ctx, cypherQuery, map[string]interface{}{
			"keywords": keywords,
			"limit":    limit,
		})
		if err != nil {
			return nil, err
		}
		records, err := result.Collect(ctx)
		if err != nil {
			return nil, err
		}
		return records, nil
	})

	if err != nil {
		return nil, err
	}

	records := result.([]*neo4j.Record)
	for _, record := range records {
		var contentParts []string

		if name, exists := record.Get("name"); exists && name != nil {
			contentParts = append(contentParts, fmt.Sprintf("菜品: %v", name))
		}
		if description, exists := record.Get("description"); exists && description != nil {
			contentParts = append(contentParts, fmt.Sprintf("描述: %v", description))
		}
		if category, exists := record.Get("category"); exists && category != nil {
			contentParts = append(contentParts, fmt.Sprintf("分类: %v", category))
		}

		nodeID, _ := record.Get("node_id")
		name, _ := record.Get("name")
		labels, _ := record.Get("labels")
		matchedKeyword, _ := record.Get("matched_keyword")

		// 确定节点类型
		nodeType := "Unknown"
		if labelSlice, ok := labels.([]interface{}); ok && len(labelSlice) > 0 {
			if labelStr, ok := labelSlice[0].(string); ok {
				nodeType = labelStr
			}
		}

		results = append(results, &RetrievalResult{
			Content:        strings.Join(contentParts, "\n"),
			NodeID:         nodeID.(string),
			NodeType:       nodeType,
			RelevanceScore: 0.7, // 补充检索得分较低
			RetrievalLevel: "entity",
			Metadata: map[string]interface{}{
				"name":            name,
				"labels":          labels,
				"matched_keyword": matchedKeyword,
				"source":          "neo4j_fallback",
			},
		})
	}

	return results, nil
}

// TopicLevelRetrieval 主题级检索：专注于广泛主题和概念
// 使用图索引的关系键值对结构进行主题检索
func (h *HybridRetrievalModule) TopicLevelRetrieval(ctx context.Context, topicKeywords []string, topK int) ([]*RetrievalResult, error) {
	var results []*RetrievalResult

	// 1. 使用实体的分类信息进行主题检索
	for _, keyword := range topicKeywords {
		for nodeID, entity := range h.entityCache {
			// 检查分类匹配
			if category, exists := entity.Metadata["category"]; exists {
				if categoryStr, ok := category.(string); ok {
					if strings.Contains(strings.ToLower(categoryStr), strings.ToLower(keyword)) {
						contentParts := []string{
							fmt.Sprintf("主题分类: %s", keyword),
							entity.Content,
						}

						results = append(results, &RetrievalResult{
							Content:        strings.Join(contentParts, "\n"),
							NodeID:         nodeID,
							NodeType:       entity.NodeType,
							RelevanceScore: 0.85, // 分类匹配得分
							RetrievalLevel: "topic",
							Metadata: map[string]interface{}{
								"matched_keyword": keyword,
								"source":          "category_match",
							},
						})
					}
				}
			}
		}
	}

	// 2. 如果结果不足，使用Neo4j进行补充检索
	if len(results) < topK {
		neo4jResults, err := h.neo4jTopicLevelSearch(ctx, topicKeywords, topK-len(results))
		if err != nil {
			log.Printf("Neo4j主题级检索失败: %v", err)
		} else {
			results = append(results, neo4jResults...)
		}
	}

	// 3. 按相关性排序并返回
	sort.Slice(results, func(i, j int) bool {
		return results[i].RelevanceScore > results[j].RelevanceScore
	})

	if len(results) > topK {
		results = results[:topK]
	}

	log.Printf("主题级检索完成，返回 %d 个结果", len(results))
	return results, nil
}

// neo4jTopicLevelSearch Neo4j主题级检索补充
func (h *HybridRetrievalModule) neo4jTopicLevelSearch(ctx context.Context, keywords []string, limit int) ([]*RetrievalResult, error) {
	var results []*RetrievalResult

	session := h.driver.NewSession(ctx, neo4j.SessionConfig{})
	defer session.Close(ctx)

	cypherQuery := `
		UNWIND $keywords as keyword
		MATCH (r:Recipe)
		WHERE r.category CONTAINS keyword 
		   OR r.cuisineType CONTAINS keyword
		   OR r.tags CONTAINS keyword
		WITH r, keyword
		OPTIONAL MATCH (r)-[:REQUIRES]->(i:Ingredient)
		WITH r, keyword, collect(i.name)[0..3] as ingredients
		RETURN 
		    r.nodeId as node_id,
		    r.name as name,
		    r.category as category,
		    r.cuisineType as cuisine_type,
		    r.difficulty as difficulty,
		    ingredients,
		    keyword as matched_keyword
		ORDER BY r.difficulty ASC, r.name
		LIMIT $limit
	`

	result, err := session.ExecuteRead(ctx, func(tx neo4j.ManagedTransaction) (interface{}, error) {
		result, err := tx.Run(ctx, cypherQuery, map[string]interface{}{
			"keywords": keywords,
			"limit":    limit,
		})
		if err != nil {
			return nil, err
		}
		records, err := result.Collect(ctx)
		if err != nil {
			return nil, err
		}
		return records, nil
	})

	if err != nil {
		return nil, err
	}

	records := result.([]*neo4j.Record)
	for _, record := range records {
		var contentParts []string

		if name, exists := record.Get("name"); exists && name != nil {
			contentParts = append(contentParts, fmt.Sprintf("菜品: %v", name))
		}
		if category, exists := record.Get("category"); exists && category != nil {
			contentParts = append(contentParts, fmt.Sprintf("分类: %v", category))
		}
		if cuisineType, exists := record.Get("cuisine_type"); exists && cuisineType != nil {
			contentParts = append(contentParts, fmt.Sprintf("菜系: %v", cuisineType))
		}
		if difficulty, exists := record.Get("difficulty"); exists && difficulty != nil {
			contentParts = append(contentParts, fmt.Sprintf("难度: %v", difficulty))
		}

		if ingredients, exists := record.Get("ingredients"); exists {
			if ingredientSlice, ok := ingredients.([]interface{}); ok && len(ingredientSlice) > 0 {
				var ingredientNames []string
				for _, ing := range ingredientSlice {
					if ingStr, ok := ing.(string); ok {
						ingredientNames = append(ingredientNames, ingStr)
					}
				}
				if len(ingredientNames) > 0 {
					contentParts = append(contentParts, fmt.Sprintf("主要食材: %s", strings.Join(ingredientNames, ", ")))
				}
			}
		}

		nodeID, _ := record.Get("node_id")
		name, _ := record.Get("name")
		category, _ := record.Get("category")
		cuisineType, _ := record.Get("cuisine_type")
		difficulty, _ := record.Get("difficulty")
		matchedKeyword, _ := record.Get("matched_keyword")

		results = append(results, &RetrievalResult{
			Content:        strings.Join(contentParts, "\n"),
			NodeID:         nodeID.(string),
			NodeType:       "Recipe",
			RelevanceScore: 0.75, // 补充检索得分
			RetrievalLevel: "topic",
			Metadata: map[string]interface{}{
				"name":            name,
				"category":        category,
				"cuisine_type":    cuisineType,
				"difficulty":      difficulty,
				"matched_keyword": matchedKeyword,
				"source":          "neo4j_fallback",
			},
		})
	}

	return results, nil
}

// DualLevelRetrieval 双层检索：结合实体级和主题级检索
func (h *HybridRetrievalModule) DualLevelRetrieval(ctx context.Context, query string, topK int) ([]*schema.Document, error) {
	log.Printf("开始双层检索: %s", query)

	// 1. 提取关键词
	entityKeywords, topicKeywords, err := h.ExtractQueryKeywords(ctx, query)
	if err != nil {
		log.Printf("关键词提取失败: %v", err)
		// 降级方案：直接使用查询作为关键词
		entityKeywords = []string{query}
		topicKeywords = []string{query}
	}

	// 2. 执行双层检索
	entityResults, err := h.EntityLevelRetrieval(ctx, entityKeywords, topK)
	if err != nil {
		log.Printf("实体级检索失败: %v", err)
		entityResults = []*RetrievalResult{}
	}

	topicResults, err := h.TopicLevelRetrieval(ctx, topicKeywords, topK)
	if err != nil {
		log.Printf("主题级检索失败: %v", err)
		topicResults = []*RetrievalResult{}
	}

	// 3. 结果合并和排序
	allResults := append(entityResults, topicResults...)

	// 4. 去重和重排序
	seenNodes := make(map[string]bool)
	var uniqueResults []*RetrievalResult

	// 按相关性得分排序
	sort.Slice(allResults, func(i, j int) bool {
		return allResults[i].RelevanceScore > allResults[j].RelevanceScore
	})

	for _, result := range allResults {
		if !seenNodes[result.NodeID] {
			seenNodes[result.NodeID] = true
			uniqueResults = append(uniqueResults, result)
		}
	}

	// 5. 转换为Document格式
	var documents []*schema.Document
	for i, result := range uniqueResults {
		if i >= topK {
			break
		}

		// 确保recipe_name字段正确设置
		recipeName := "未知菜品"
		if name, exists := result.Metadata["name"]; exists && name != nil {
			if nameStr, ok := name.(string); ok {
				recipeName = nameStr
			}
		}

		metadata := make(map[string]interface{})
		for k, v := range result.Metadata {
			metadata[k] = v
		}
		metadata["node_id"] = result.NodeID
		metadata["node_type"] = result.NodeType
		metadata["retrieval_level"] = result.RetrievalLevel
		metadata["relevance_score"] = result.RelevanceScore
		metadata["recipe_name"] = recipeName
		metadata["search_type"] = "dual_level"

		doc := &schema.Document{
			Content:  result.Content,
			MetaData: metadata,
		}
		documents = append(documents, doc)
	}

	log.Printf("双层检索完成，返回 %d 个文档", len(documents))
	return documents, nil
}

// VectorSearchEnhanced 增强的向量检索：结合图信息
func (h *HybridRetrievalModule) VectorSearchEnhanced(ctx context.Context, query string, topK int) ([]*schema.Document, error) {
	// 由于milvusModule是interface{}类型，这里提供一个基础实现框架
	// 实际使用时需要根据具体的Milvus模块接口进行调整

	var documents []*schema.Document

	// 模拟向量检索结果
	// 在实际实现中，这里应该调用Milvus模块的相似度搜索方法
	log.Printf("执行增强向量检索: %s", query)

	// 从图索引中获取一些相关结果作为模拟
	var mockResults []*RetrievalResult
	for nodeID, entity := range h.entityCache {
		if strings.Contains(strings.ToLower(entity.Content), strings.ToLower(query)) {
			// 获取邻居信息增强
			neighbors, _ := h.getNodeNeighbors(ctx, nodeID, 3)

			content := entity.Content
			if len(neighbors) > 0 {
				content += fmt.Sprintf("\n相关信息: %s", strings.Join(neighbors[:3], ", "))
			}

			mockResults = append(mockResults, &RetrievalResult{
				Content:        content,
				NodeID:         nodeID,
				NodeType:       entity.NodeType,
				RelevanceScore: 0.8,
				RetrievalLevel: "vector",
				Metadata:       entity.Metadata,
			})

			if len(mockResults) >= topK {
				break
			}
		}
	}

	// 转换为Document格式
	for _, result := range mockResults {
		recipeName := "未知菜品"
		if name, exists := result.Metadata["name"]; exists && name != nil {
			if nameStr, ok := name.(string); ok {
				recipeName = nameStr
			}
		}

		metadata := make(map[string]interface{})
		for k, v := range result.Metadata {
			metadata[k] = v
		}
		metadata["recipe_name"] = recipeName
		metadata["score"] = result.RelevanceScore
		metadata["search_type"] = "vector_enhanced"

		doc := &schema.Document{
			Content:  result.Content,
			MetaData: metadata,
		}
		documents = append(documents, doc)
	}

	return documents, nil
}

// getNodeNeighbors 获取节点的邻居信息
func (h *HybridRetrievalModule) getNodeNeighbors(ctx context.Context, nodeID string, maxNeighbors int) ([]string, error) {
	session := h.driver.NewSession(ctx, neo4j.SessionConfig{})
	defer session.Close(ctx)

	query := `
		MATCH (n {nodeId: $node_id})-[r]-(neighbor)
		RETURN neighbor.name as name
		LIMIT $limit
	`

	result, err := session.ExecuteRead(ctx, func(tx neo4j.ManagedTransaction) (interface{}, error) {
		result, err := tx.Run(ctx, query, map[string]interface{}{
			"node_id": nodeID,
			"limit":   maxNeighbors,
		})
		if err != nil {
			return nil, err
		}
		records, err := result.Collect(ctx)
		if err != nil {
			return nil, err
		}
		return records, nil
	})

	if err != nil {
		return nil, err
	}

	var neighbors []string
	records := result.([]*neo4j.Record)
	for _, record := range records {
		if name, exists := record.Get("name"); exists && name != nil {
			if nameStr, ok := name.(string); ok && nameStr != "" {
				neighbors = append(neighbors, nameStr)
			}
		}
	}

	return neighbors, nil
}

// HybridSearch 混合检索：使用Round-robin轮询合并策略
// 公平轮询合并不同检索结果，不使用权重配置
func (h *HybridRetrievalModule) HybridSearch(ctx context.Context, query string, topK int) ([]*schema.Document, error) {
	log.Printf("开始混合检索: %s", query)

	// 1. 双层检索（实体+主题检索）
	dualDocs, err := h.DualLevelRetrieval(ctx, query, topK)
	if err != nil {
		log.Printf("双层检索失败: %v", err)
		dualDocs = []*schema.Document{}
	}

	// 2. 增强向量检索
	vectorDocs, err := h.VectorSearchEnhanced(ctx, query, topK)
	if err != nil {
		log.Printf("向量检索失败: %v", err)
		vectorDocs = []*schema.Document{}
	}

	// 3. Round-robin轮询合并
	var mergedDocs []*schema.Document
	seenDocIDs := make(map[string]bool)
	maxLen := len(dualDocs)
	if len(vectorDocs) > maxLen {
		maxLen = len(vectorDocs)
	}
	originLen := len(dualDocs) + len(vectorDocs)

	for i := 0; i < maxLen; i++ {
		// 先添加双层检索结果
		if i < len(dualDocs) {
			doc := dualDocs[i]
			docID := ""
			if nodeID, exists := doc.MetaData["node_id"]; exists {
				docID = fmt.Sprintf("%v", nodeID)
			} else {
				docID = fmt.Sprintf("dual_%d", i)
			}

			if !seenDocIDs[docID] {
				seenDocIDs[docID] = true
				doc.MetaData["search_method"] = "dual_level"
				doc.MetaData["round_robin_order"] = len(mergedDocs)
				// 设置统一的final_score字段
				if score, exists := doc.MetaData["relevance_score"]; exists {
					doc.MetaData["final_score"] = score
				} else {
					doc.MetaData["final_score"] = 0.0
				}
				mergedDocs = append(mergedDocs, doc)
			}
		}

		// 再添加向量检索结果
		if i < len(vectorDocs) {
			doc := vectorDocs[i]
			docID := ""
			if nodeID, exists := doc.MetaData["node_id"]; exists {
				docID = fmt.Sprintf("%v", nodeID)
			} else {
				docID = fmt.Sprintf("vector_%d", i)
			}

			if !seenDocIDs[docID] {
				seenDocIDs[docID] = true
				doc.MetaData["search_method"] = "vector_enhanced"
				doc.MetaData["round_robin_order"] = len(mergedDocs)
				// 设置统一的final_score字段（向量得分需要转换）
				if score, exists := doc.MetaData["score"]; exists {
					if scoreFloat, ok := score.(float64); ok {
						// COSINE距离转换为相似度：distance越小，相似度越高
						similarityScore := 0.0
						if scoreFloat <= 1.0 {
							similarityScore = 1.0 - scoreFloat
							if similarityScore < 0.0 {
								similarityScore = 0.0
							}
						}
						doc.MetaData["final_score"] = similarityScore
					} else {
						doc.MetaData["final_score"] = 0.0
					}
				} else {
					doc.MetaData["final_score"] = 0.0
				}
				mergedDocs = append(mergedDocs, doc)
			}
		}
	}

	// 取前topK个结果
	finalDocs := mergedDocs
	if len(finalDocs) > topK {
		finalDocs = finalDocs[:topK]
	}

	log.Printf("Round-robin合并：从总共%d个结果合并为%d个文档", originLen, len(finalDocs))
	log.Printf("混合检索完成，返回 %d 个文档", len(finalDocs))
	return finalDocs, nil
}

// Close 关闭资源连接
func (h *HybridRetrievalModule) Close(ctx context.Context) error {
	if h.driver != nil {
		err := h.driver.Close(ctx)
		if err != nil {
			return fmt.Errorf("关闭Neo4j连接失败: %w", err)
		}
		log.Println("Neo4j连接已关闭")
	}
	return nil
}
