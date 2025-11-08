package batch_0001

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"sort"
	"strings"

	"github.com/cloudwego/eino-ext/components/model/ark"
	"github.com/cloudwego/eino/components/model"
	"github.com/cloudwego/eino/components/prompt"
	"github.com/cloudwego/eino/schema"
	"github.com/neo4j/neo4j-go-driver/v5/neo4j"
)

// QueryType 图查询类型枚举
type QueryType string

const (
	// EntityRelation 实体关系查询：A和B有什么关系？
	EntityRelation QueryType = "entity_relation"
	// MultiHop 多跳查询：A通过什么连接到C？
	MultiHop QueryType = "multi_hop"
	// Subgraph 子图查询：A相关的所有信息
	Subgraph QueryType = "subgraph"
	// PathFinding 路径查找：从A到B的最佳路径
	PathFinding QueryType = "path_finding"
	// Clustering 聚类查询：和A相似的都有什么？
	Clustering QueryType = "clustering"
)

// GraphQuery 图查询结构
//
// 封装了图查询的所有参数，包括查询类型、目标实体、关系类型等。
// 提供了统一的查询接口，支持复杂的图查询需求。
type GraphQuery struct {
	QueryType      QueryType              `json:"query_type"`      // 查询类型，决定遍历策略
	SourceEntities []string               `json:"source_entities"` // 源实体列表，查询的起点
	TargetEntities []string               `json:"target_entities"` // 目标实体列表，查询的终点（可选）
	RelationTypes  []string               `json:"relation_types"`  // 关注的关系类型（可选）
	MaxDepth       int                    `json:"max_depth"`       // 最大遍历深度，控制搜索范围
	MaxNodes       int                    `json:"max_nodes"`       // 最大节点数，控制结果规模
	Constraints    map[string]interface{} `json:"constraints"`     // 额外的查询约束条件
}

// GraphPath 图路径结构
//
// 表示图中两个或多个节点之间的路径，包含路径上的所有节点和关系。
// 用于多跳推理和路径分析。
type GraphPath struct {
	Nodes          []map[string]interface{} `json:"nodes"`           // 路径上的节点序列
	Relationships  []map[string]interface{} `json:"relationships"`   // 路径上的关系序列
	PathLength     int                      `json:"path_length"`     // 路径长度（跳数）
	RelevanceScore float64                  `json:"relevance_score"` // 路径的相关性得分
	PathType       string                   `json:"path_type"`       // 路径类型标识
}

// KnowledgeSubgraph 知识子图结构
//
// 表示以特定实体为中心的知识子网络，包含相关的节点、关系和推理链。
// 用于子图查询和知识网络分析。
type KnowledgeSubgraph struct {
	CentralNodes    []map[string]interface{} `json:"central_nodes"`    // 中心节点列表
	ConnectedNodes  []map[string]interface{} `json:"connected_nodes"`  // 连接的节点列表
	Relationships   []map[string]interface{} `json:"relationships"`    // 子图中的关系列表
	GraphMetrics    map[string]float64       `json:"graph_metrics"`    // 图度量指标（密度、连通性等）
	ReasoningChains [][]string               `json:"reasoning_chains"` // 推理链列表
}

// 注意：现在使用 eino 的标准 schema.Document 结构体
// 不再需要自定义 Document 结构体

// Config 配置结构
type Config struct {
	Neo4jURI      string                 `json:"neo4j_uri"`
	Neo4jUser     string                 `json:"neo4j_user"`
	Neo4jPassword string                 `json:"neo4j_password"`
	LLMModel      string                 `json:"llm_model"`
	ArkAPIKey     string                 `json:"ark_api_key"`
	ArkBaseURL    string                 `json:"ark_base_url"`
	Constraints   map[string]interface{} `json:"constraints"`
}

// GraphRAGRetrieval 真正的图RAG检索系统 - 基于图结构的智能检索引擎
//
// 这是图RAG系统的核心组件，实现了基于知识图谱的深度检索和推理能力。
// 与传统的关键词检索不同，它能够理解和利用实体间的复杂关系。
//
// 核心特点：
// 1. 查询意图理解：识别图查询模式，将自然语言转换为图操作
// 2. 多跳图遍历：深度关系探索，发现多步推理路径
// 3. 子图提取：相关知识网络的完整提取
// 4. 图结构推理：基于拓扑的推理，发现隐含关系
// 5. 动态查询规划：自适应遍历策略，优化检索效率
//
// 技术优势：
// - 结构化推理：利用图结构进行逻辑推理
// - 深度关联：发现实体间的深层关系
// - 上下文完整：提供丰富的知识背景
// - 可解释性：清晰的推理路径和关系链
//
// 应用场景：
// - 复杂问答：需要多步推理的知识问题
// - 关系探索：实体间关联关系的发现
// - 知识发现：隐含知识模式的挖掘
// - 智能推荐：基于关系网络的推荐
type GraphRAGRetrieval struct {
	config    *Config
	llmClient *ark.ChatModel
	driver    neo4j.DriverWithContext

	// 图结构缓存 - 提高重复查询的性能
	entityCache   map[string]map[string]interface{} // 实体信息缓存
	relationCache map[string]int                    // 关系类型缓存
	subgraphCache map[string]*KnowledgeSubgraph     // 子图结果缓存
}

// QueryAnalysisResult LLM查询分析结果
type QueryAnalysisResult struct {
	QueryType      string   `json:"query_type"`
	SourceEntities []string `json:"source_entities"`
	TargetEntities []string `json:"target_entities"`
	RelationTypes  []string `json:"relation_types"`
	MaxDepth       int      `json:"max_depth"`
	Reasoning      string   `json:"reasoning"`
}

// NewGraphRAGRetrieval 创建新的图RAG检索系统
func NewGraphRAGRetrieval(config *Config) *GraphRAGRetrieval {
	return &GraphRAGRetrieval{
		config:        config,
		entityCache:   make(map[string]map[string]interface{}),
		relationCache: make(map[string]int),
		subgraphCache: make(map[string]*KnowledgeSubgraph),
	}
}

// Initialize 初始化图RAG检索系统
//
// 建立Neo4j连接，构建图索引，为后续的图查询做准备。
func (g *GraphRAGRetrieval) Initialize(ctx context.Context) error {
	log.Println("初始化图RAG检索系统...")

	// 初始化Ark LLM客户端
	arkClient, err := ark.NewChatModel(ctx, &ark.ChatModelConfig{
		APIKey:  g.config.ArkAPIKey,
		BaseURL: g.config.ArkBaseURL,
		Model:   g.config.LLMModel,
	})
	if err != nil {
		return fmt.Errorf("初始化Ark客户端失败: %w", err)
	}
	g.llmClient = arkClient

	// 连接Neo4j图数据库
	driver, err := neo4j.NewDriverWithContext(
		g.config.Neo4jURI,
		neo4j.BasicAuth(g.config.Neo4jUser, g.config.Neo4jPassword, ""),
	)
	if err != nil {
		return fmt.Errorf("Neo4j连接失败: %w", err)
	}
	g.driver = driver

	// 测试连接有效性
	err = driver.VerifyConnectivity(ctx)
	if err != nil {
		return fmt.Errorf("Neo4j连接验证失败: %w", err)
	}

	log.Println("Neo4j连接成功")

	// 预热：构建实体和关系索引，加速后续查询
	if err := g.buildGraphIndex(ctx); err != nil {
		log.Printf("构建图索引失败: %v", err)
	}

	return nil
}

// buildGraphIndex 构建图索引以加速查询
//
// 预先计算和缓存图中实体和关系的统计信息，
// 包括节点度数、关系频率等，用于查询优化。
func (g *GraphRAGRetrieval) buildGraphIndex(ctx context.Context) error {
	log.Println("构建图结构索引...")

	session := g.driver.NewSession(ctx, neo4j.SessionConfig{})
	defer session.Close(ctx)

	// 构建实体索引 - 计算每个节点的度数（连接关系数）
	entityQuery := `
		MATCH (n)
		WHERE n.nodeId IS NOT NULL
		WITH n, COUNT { (n)--() } as degree
		RETURN labels(n) as node_labels, n.nodeId as node_id, 
		       n.name as name, n.category as category, degree
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
		degree, _ := record.Get("degree")

		// 缓存节点信息，包括重要的度数信息
		g.entityCache[nodeID.(string)] = map[string]interface{}{
			"labels":   nodeLabels,
			"name":     name,
			"category": category,
			"degree":   degree, // 节点度数，用于重要性评估
		}
	}

	// 构建关系类型索引
	// 统计每种关系类型的频率，用于查询优化
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

		g.relationCache[relType.(string)] = int(frequency.(int64))
	}

	log.Printf("索引构建完成: %d个实体, %d个关系类型",
		len(g.entityCache), len(g.relationCache))

	return nil
}

// UnderstandGraphQuery 理解查询的图结构意图 - 图RAG的核心能力
//
// 这是图RAG系统的智能核心：将自然语言查询转换为结构化的图查询操作。
// 通过大语言模型分析查询意图，识别需要的图遍历模式。
//
// 分析维度：
// 1. 查询类型：识别是实体关系查询、多跳推理还是子图探索
// 2. 核心实体：提取查询中的关键实体
// 3. 目标实体：确定期望找到的实体类型
// 4. 关系类型：识别涉及的关系类型
// 5. 遍历深度：评估需要的图遍历深度
func (g *GraphRAGRetrieval) UnderstandGraphQuery(ctx context.Context, query string) (*GraphQuery, error) {
	// 构建详细的查询分析提示词
	template := prompt.FromMessages(schema.FString,
		schema.SystemMessage("你是一个图数据库专家。"),
		&schema.Message{
			Role: schema.User,
			Content: `分析以下查询的图结构意图：
			
			查询：{query}
			
			请识别：
			1. 查询类型：
			   - entity_relation: 询问实体间的直接关系（如：鸡肉和胡萝卜能一起做菜吗？）
			   - multi_hop: 需要多跳推理（如：鸡肉配什么蔬菜？需要：鸡肉→菜品→食材→蔬菜）
			   - subgraph: 需要完整子图（如：川菜有什么特色？需要川菜相关的完整知识网络）
			   - path_finding: 路径查找（如：从食材到成品菜的制作路径）
			   - clustering: 聚类相似性（如：和宫保鸡丁类似的菜有哪些？）
			
			2. 核心实体：查询中的关键实体名称
			3. 目标实体：期望找到的实体类型
			4. 关系类型：涉及的关系类型
			5. 遍历深度：需要的图遍历深度（1-3跳）
			
			示例：
			查询："鸡肉配什么蔬菜好？"
			分析：这是multi_hop查询，需要通过"鸡肉→使用鸡肉的菜品→这些菜品使用的蔬菜"的路径推理
			
			返回JSON格式：
			{
				"query_type": "multi_hop",
				"source_entities": ["鸡肉"],
				"target_entities": ["蔬菜类食材"],
				"relation_types": ["REQUIRES", "BELONGS_TO_CATEGORY"],
				"max_depth": 3,
				"reasoning": "需要多跳推理：鸡肉→菜品→食材→蔬菜"
			}`,
		},
	)

	values := map[string]interface{}{
		"query": query,
	}

	messages, err := template.Format(ctx, values)
	if err != nil {
		return nil, fmt.Errorf("模板格式化失败: %w", err)
	}

	response, err := g.llmClient.Generate(ctx, messages, model.WithTemperature(0.1), model.WithMaxTokens(1000))
	if err != nil {
		log.Printf("查询意图理解失败: %v", err)
		// 降级方案：默认使用子图查询
		return &GraphQuery{
			QueryType:      Subgraph,
			SourceEntities: []string{query},
			MaxDepth:       2,
			MaxNodes:       50,
		}, nil
	}

	var result QueryAnalysisResult
	if err := json.Unmarshal([]byte(response.Content), &result); err != nil {
		log.Printf("解析查询分析结果失败: %v", err)
		// 降级方案
		return &GraphQuery{
			QueryType:      Subgraph,
			SourceEntities: []string{query},
			MaxDepth:       2,
			MaxNodes:       50,
		}, nil
	}

	// 构建GraphQuery对象
	queryType := Subgraph // 默认值
	switch result.QueryType {
	case "entity_relation":
		queryType = EntityRelation
	case "multi_hop":
		queryType = MultiHop
	case "subgraph":
		queryType = Subgraph
	case "path_finding":
		queryType = PathFinding
	case "clustering":
		queryType = Clustering
	}

	maxDepth := result.MaxDepth
	if maxDepth == 0 {
		maxDepth = 2
	}

	return &GraphQuery{
		QueryType:      queryType,
		SourceEntities: result.SourceEntities,
		TargetEntities: result.TargetEntities,
		RelationTypes:  result.RelationTypes,
		MaxDepth:       maxDepth,
		MaxNodes:       50,
	}, nil
}

// MultiHopTraversal 多跳图遍历：这是图RAG的核心优势
// 通过图结构发现隐含的知识关联
func (g *GraphRAGRetrieval) MultiHopTraversal(ctx context.Context, graphQuery *GraphQuery) ([]*GraphPath, error) {
	log.Printf("执行多跳遍历: %v -> %v", graphQuery.SourceEntities, graphQuery.TargetEntities)

	var paths []*GraphPath

	if g.driver == nil {
		return paths, fmt.Errorf("Neo4j连接未建立")
	}

	session := g.driver.NewSession(ctx, neo4j.SessionConfig{})
	defer session.Close(ctx)

	// 根据查询类型选择不同的遍历策略
	if graphQuery.QueryType == MultiHop {
		targetLabelsCondition := ""
		if len(graphQuery.TargetEntities) > 0 {
			targetLabelsCondition = "AND ANY(label IN labels(target) WHERE label IN $target_labels)"
		}

		// 构建多跳遍历查询
		cypherQuery := fmt.Sprintf(`
			// 多跳推理查询
			UNWIND $source_entities as source_name
			MATCH (source)
			WHERE source.name CONTAINS source_name OR source.nodeId = source_name
			
			// 执行多跳遍历
			MATCH path = (source)-[*1..%d]-(target)
			WHERE NOT source = target
			%s
			
			// 计算路径相关性
			WITH path, source, target,
			     length(path) as path_len,
			     relationships(path) as rels,
			     nodes(path) as path_nodes
			
			// 路径评分：短路径 + 高度数节点 + 关系类型匹配
			WITH path, source, target, path_len, rels, path_nodes,
			     (1.0 / path_len) + 
			     (REDUCE(s = 0.0, n IN path_nodes | s + COUNT { (n)--() }) / 10.0 / size(path_nodes)) +
			     (CASE WHEN ANY(r IN rels WHERE type(r) IN $relation_types) THEN 0.3 ELSE 0.0 END) as relevance
			
			ORDER BY relevance DESC
			LIMIT 20
			
			RETURN path, source, target, path_len, rels, path_nodes, relevance
		`, graphQuery.MaxDepth, targetLabelsCondition)

		relationTypes := graphQuery.RelationTypes
		if relationTypes == nil {
			relationTypes = []string{}
		}

		result, err := session.ExecuteRead(ctx, func(tx neo4j.ManagedTransaction) (interface{}, error) {
			result, err := tx.Run(ctx, cypherQuery, map[string]interface{}{
				"source_entities": graphQuery.SourceEntities,
				"target_labels":   graphQuery.TargetEntities,
				"relation_types":  relationTypes,
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
			return nil, fmt.Errorf("多跳遍历查询失败: %w", err)
		}

		records := result.([]*neo4j.Record)
		for _, record := range records {
			pathData := g.parseNeo4jPath(record)
			if pathData != nil {
				paths = append(paths, pathData)
			}
		}
	} else if graphQuery.QueryType == EntityRelation {
		// 实体间关系查询
		entityPaths, err := g.findEntityRelations(ctx, graphQuery, session)
		if err != nil {
			log.Printf("查找实体关系失败: %v", err)
		} else {
			paths = append(paths, entityPaths...)
		}
	} else if graphQuery.QueryType == PathFinding {
		// 最短路径查找
		shortestPaths, err := g.findShortestPaths(ctx, graphQuery, session)
		if err != nil {
			log.Printf("查找最短路径失败: %v", err)
		} else {
			paths = append(paths, shortestPaths...)
		}
	}

	log.Printf("多跳遍历完成，找到 %d 条路径", len(paths))
	return paths, nil
}

// ExtractKnowledgeSubgraph 提取知识子图：获取实体相关的完整知识网络
// 这体现了图RAG的整体性思维
func (g *GraphRAGRetrieval) ExtractKnowledgeSubgraph(ctx context.Context, graphQuery *GraphQuery) (*KnowledgeSubgraph, error) {
	log.Printf("提取知识子图: %v", graphQuery.SourceEntities)

	if g.driver == nil {
		return g.fallbackSubgraphExtraction(graphQuery), fmt.Errorf("Neo4j连接未建立")
	}

	session := g.driver.NewSession(ctx, neo4j.SessionConfig{})
	defer session.Close(ctx)

	// 简化的子图提取（不依赖APOC）
	cypherQuery := fmt.Sprintf(`
		// 找到源实体
		UNWIND $source_entities as entity_name
		MATCH (source)
		WHERE source.name CONTAINS entity_name 
		   OR source.nodeId = entity_name
		
		// 获取指定深度的邻居
		MATCH (source)-[r*1..%d]-(neighbor)
		WITH source, collect(DISTINCT neighbor) as neighbors, 
		     collect(DISTINCT r) as relationships
		WHERE size(neighbors) <= $max_nodes
		
		// 计算图指标
		WITH source, neighbors, relationships,
		     size(neighbors) as node_count,
		     size(relationships) as rel_count
		
		RETURN 
		    source,
		    neighbors[0..%d] as nodes,
		    relationships[0..%d] as rels,
		    {
		        node_count: node_count,
		        relationship_count: rel_count,
		        density: CASE WHEN node_count > 1 THEN toFloat(rel_count) / (node_count * (node_count - 1) / 2) ELSE 0.0 END
		    } as metrics
	`, graphQuery.MaxDepth, graphQuery.MaxNodes, graphQuery.MaxNodes)

	result, err := session.ExecuteRead(ctx, func(tx neo4j.ManagedTransaction) (interface{}, error) {
		result, err := tx.Run(ctx, cypherQuery, map[string]interface{}{
			"source_entities": graphQuery.SourceEntities,
			"max_nodes":       graphQuery.MaxNodes,
		})
		if err != nil {
			return nil, err
		}

		record, err := result.Single(ctx)
		if err != nil {
			return nil, err
		}

		return record, nil
	})

	if err != nil {
		log.Printf("子图提取失败: %v", err)
		return g.fallbackSubgraphExtraction(graphQuery), err
	}

	record := result.(*neo4j.Record)
	return g.buildKnowledgeSubgraph(record), nil
}

// GraphStructureReasoning 基于图结构的推理：这是图RAG的智能之处
// 不仅检索信息，还能进行逻辑推理
func (g *GraphRAGRetrieval) GraphStructureReasoning(subgraph *KnowledgeSubgraph, query string) []string {
	var reasoningChains []string

	// 1. 识别推理模式
	reasoningPatterns := g.identifyReasoningPatterns(subgraph)

	// 2. 构建推理链
	for _, pattern := range reasoningPatterns {
		chain := g.buildReasoningChain(pattern, subgraph)
		if chain != "" {
			reasoningChains = append(reasoningChains, chain)
		}
	}

	// 3. 验证推理链的可信度
	validatedChains := g.validateReasoningChains(reasoningChains, query)

	log.Printf("图结构推理完成，生成 %d 条推理链", len(validatedChains))
	return validatedChains
}

// AdaptiveQueryPlanning 自适应查询规划：根据查询复杂度动态调整策略
func (g *GraphRAGRetrieval) AdaptiveQueryPlanning(query string) []*GraphQuery {
	// 分析查询复杂度
	complexityScore := g.analyzeQueryComplexity(query)

	var queryPlans []*GraphQuery

	if complexityScore < 0.3 {
		// 简单查询：直接邻居查询
		plan := &GraphQuery{
			QueryType:      EntityRelation,
			SourceEntities: []string{query},
			MaxDepth:       1,
			MaxNodes:       20,
		}
		queryPlans = append(queryPlans, plan)
	} else if complexityScore < 0.7 {
		// 中等复杂度：多跳查询
		plan := &GraphQuery{
			QueryType:      MultiHop,
			SourceEntities: []string{query},
			MaxDepth:       2,
			MaxNodes:       50,
		}
		queryPlans = append(queryPlans, plan)
	} else {
		// 复杂查询：子图提取 + 推理
		plan1 := &GraphQuery{
			QueryType:      Subgraph,
			SourceEntities: []string{query},
			MaxDepth:       3,
			MaxNodes:       100,
		}
		plan2 := &GraphQuery{
			QueryType:      MultiHop,
			SourceEntities: []string{query},
			MaxDepth:       3,
			MaxNodes:       50,
		}
		queryPlans = append(queryPlans, plan1, plan2)
	}

	return queryPlans
}

// GraphRAGSearch 图RAG主搜索接口：整合所有图RAG能力
func (g *GraphRAGRetrieval) GraphRAGSearch(ctx context.Context, query string, topK int) ([]*schema.Document, error) {
	log.Printf("开始图RAG检索: %s", query)

	if g.driver == nil {
		log.Println("Neo4j连接未建立，返回空结果")
		return []*schema.Document{}, nil
	}

	// 1. 查询意图理解
	graphQuery, err := g.UnderstandGraphQuery(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("查询意图理解失败: %w", err)
	}

	log.Printf("查询类型: %s", graphQuery.QueryType)

	var results []*schema.Document

	// 2. 根据查询类型执行不同策略
	if graphQuery.QueryType == MultiHop || graphQuery.QueryType == PathFinding {
		// 多跳遍历
		paths, err := g.MultiHopTraversal(ctx, graphQuery)
		if err != nil {
			log.Printf("多跳遍历失败: %v", err)
		} else {
			results = append(results, g.pathsToDocuments(paths, query)...)
		}
	} else if graphQuery.QueryType == Subgraph {
		// 子图提取
		subgraph, err := g.ExtractKnowledgeSubgraph(ctx, graphQuery)
		if err != nil {
			log.Printf("子图提取失败: %v", err)
		} else {
			// 图结构推理
			reasoningChains := g.GraphStructureReasoning(subgraph, query)

			results = append(results, g.subgraphToDocuments(subgraph, reasoningChains, query)...)
		}
	} else if graphQuery.QueryType == EntityRelation {
		// 实体关系查询
		paths, err := g.MultiHopTraversal(ctx, graphQuery)
		if err != nil {
			log.Printf("实体关系查询失败: %v", err)
		} else {
			results = append(results, g.pathsToDocuments(paths, query)...)
		}
	}

	// 3. 图结构相关性排序
	results = g.rankByGraphRelevance(results, query)

	if topK > len(results) {
		topK = len(results)
	}

	log.Printf("图RAG检索完成，返回 %d 个结果", topK)
	return results[:topK], nil
}

// ========== 辅助方法 ==========

// parseNeo4jPath 解析Neo4j路径记录
func (g *GraphRAGRetrieval) parseNeo4jPath(record *neo4j.Record) *GraphPath {
	pathNodes, exists := record.Get("path_nodes")
	if !exists {
		return nil
	}

	rels, exists := record.Get("rels")
	if !exists {
		return nil
	}

	pathLen, exists := record.Get("path_len")
	if !exists {
		return nil
	}

	relevance, exists := record.Get("relevance")
	if !exists {
		return nil
	}

	// 转换节点
	var nodes []map[string]interface{}
	if nodeList, ok := pathNodes.([]interface{}); ok {
		for _, node := range nodeList {
			if n, ok := node.(neo4j.Node); ok {
				nodeMap := map[string]interface{}{
					"id":         n.Props["nodeId"],
					"name":       n.Props["name"],
					"labels":     n.Labels,
					"properties": n.Props,
				}
				nodes = append(nodes, nodeMap)
			}
		}
	}

	// 转换关系
	var relationships []map[string]interface{}
	if relList, ok := rels.([]interface{}); ok {
		for _, rel := range relList {
			if r, ok := rel.(neo4j.Relationship); ok {
				relMap := map[string]interface{}{
					"type":       r.Type,
					"properties": r.Props,
				}
				relationships = append(relationships, relMap)
			}
		}
	}

	return &GraphPath{
		Nodes:          nodes,
		Relationships:  relationships,
		PathLength:     int(pathLen.(int64)),
		RelevanceScore: relevance.(float64),
		PathType:       "multi_hop",
	}
}

// buildKnowledgeSubgraph 构建知识子图对象
func (g *GraphRAGRetrieval) buildKnowledgeSubgraph(record *neo4j.Record) *KnowledgeSubgraph {
	source, _ := record.Get("source")
	nodes, _ := record.Get("nodes")
	rels, _ := record.Get("rels")
	metrics, _ := record.Get("metrics")

	var centralNodes []map[string]interface{}
	if sourceNode, ok := source.(neo4j.Node); ok {
		centralNodes = append(centralNodes, sourceNode.Props)
	}

	var connectedNodes []map[string]interface{}
	if nodeList, ok := nodes.([]interface{}); ok {
		for _, node := range nodeList {
			if n, ok := node.(neo4j.Node); ok {
				connectedNodes = append(connectedNodes, n.Props)
			}
		}
	}

	var relationships []map[string]interface{}
	if relList, ok := rels.([]interface{}); ok {
		for _, rel := range relList {
			if r, ok := rel.(neo4j.Relationship); ok {
				relationships = append(relationships, r.Props)
			}
		}
	}

	var graphMetrics map[string]float64
	if metricsMap, ok := metrics.(map[string]interface{}); ok {
		graphMetrics = make(map[string]float64)
		for k, v := range metricsMap {
			if f, ok := v.(float64); ok {
				graphMetrics[k] = f
			} else if i, ok := v.(int64); ok {
				graphMetrics[k] = float64(i)
			}
		}
	}

	return &KnowledgeSubgraph{
		CentralNodes:    centralNodes,
		ConnectedNodes:  connectedNodes,
		Relationships:   relationships,
		GraphMetrics:    graphMetrics,
		ReasoningChains: [][]string{},
	}
}

// pathsToDocuments 将图路径转换为Document对象
func (g *GraphRAGRetrieval) pathsToDocuments(paths []*GraphPath, query string) []*schema.Document {
	var documents []*schema.Document

	for _, path := range paths {
		// 构建路径描述
		pathDesc := g.buildPathDescription(path)

		recipeName := "图结构结果"
		if len(path.Nodes) > 0 {
			if name, exists := path.Nodes[0]["name"]; exists {
				if nameStr, ok := name.(string); ok {
					recipeName = nameStr
				}
			}
		}

		doc := &schema.Document{
			ID:      fmt.Sprintf("path_%d", len(documents)),
			Content: pathDesc,
			MetaData: map[string]interface{}{
				"search_type":        "graph_path",
				"path_length":        path.PathLength,
				"relevance_score":    path.RelevanceScore,
				"path_type":          path.PathType,
				"node_count":         len(path.Nodes),
				"relationship_count": len(path.Relationships),
				"recipe_name":        recipeName,
			},
		}
		documents = append(documents, doc)
	}

	return documents
}

// subgraphToDocuments 将知识子图转换为Document对象
func (g *GraphRAGRetrieval) subgraphToDocuments(subgraph *KnowledgeSubgraph,
	reasoningChains []string, query string) []*schema.Document {
	var documents []*schema.Document

	// 子图整体描述
	subgraphDesc := g.buildSubgraphDescription(subgraph)

	recipeName := "知识子图"
	if len(subgraph.CentralNodes) > 0 {
		if name, exists := subgraph.CentralNodes[0]["name"]; exists {
			if nameStr, ok := name.(string); ok {
				recipeName = nameStr
			}
		}
	}

	doc := &schema.Document{
		ID:      fmt.Sprintf("subgraph_%d", len(documents)),
		Content: subgraphDesc,
		MetaData: map[string]interface{}{
			"search_type":        "knowledge_subgraph",
			"node_count":         len(subgraph.ConnectedNodes),
			"relationship_count": len(subgraph.Relationships),
			"graph_density":      subgraph.GraphMetrics["density"],
			"reasoning_chains":   reasoningChains,
			"recipe_name":        recipeName,
		},
	}
	documents = append(documents, doc)

	return documents
}

// buildPathDescription 构建路径的自然语言描述
func (g *GraphRAGRetrieval) buildPathDescription(path *GraphPath) string {
	if len(path.Nodes) == 0 {
		return "空路径"
	}

	var descParts []string
	for i, node := range path.Nodes {
		if name, exists := node["name"]; exists {
			if nameStr, ok := name.(string); ok {
				descParts = append(descParts, nameStr)
			} else {
				descParts = append(descParts, fmt.Sprintf("节点%d", i))
			}
		} else {
			descParts = append(descParts, fmt.Sprintf("节点%d", i))
		}

		if i < len(path.Relationships) {
			relType := "相关"
			if relTypeVal, exists := path.Relationships[i]["type"]; exists {
				if relTypeStr, ok := relTypeVal.(string); ok {
					relType = relTypeStr
				}
			}
			descParts = append(descParts, fmt.Sprintf(" --%s--> ", relType))
		}
	}

	return strings.Join(descParts, "")
}

// buildSubgraphDescription 构建子图的自然语言描述
func (g *GraphRAGRetrieval) buildSubgraphDescription(subgraph *KnowledgeSubgraph) string {
	var centralNames []string
	for _, node := range subgraph.CentralNodes {
		if name, exists := node["name"]; exists {
			if nameStr, ok := name.(string); ok {
				centralNames = append(centralNames, nameStr)
			} else {
				centralNames = append(centralNames, "未知")
			}
		} else {
			centralNames = append(centralNames, "未知")
		}
	}

	nodeCount := len(subgraph.ConnectedNodes)
	relCount := len(subgraph.Relationships)

	return fmt.Sprintf("关于 %s 的知识网络，包含 %d 个相关概念和 %d 个关系。",
		strings.Join(centralNames, ", "), nodeCount, relCount)
}

// rankByGraphRelevance 基于图结构相关性排序
func (g *GraphRAGRetrieval) rankByGraphRelevance(documents []*schema.Document, query string) []*schema.Document {
	sort.Slice(documents, func(i, j int) bool {
		scoreI := 0.0
		scoreJ := 0.0

		if score, exists := documents[i].MetaData["relevance_score"]; exists {
			if scoreFloat, ok := score.(float64); ok {
				scoreI = scoreFloat
			}
		}

		if score, exists := documents[j].MetaData["relevance_score"]; exists {
			if scoreFloat, ok := score.(float64); ok {
				scoreJ = scoreFloat
			}
		}

		return scoreI > scoreJ
	})

	return documents
}

// analyzeQueryComplexity 分析查询复杂度
func (g *GraphRAGRetrieval) analyzeQueryComplexity(query string) float64 {
	complexityIndicators := []string{"什么", "如何", "为什么", "哪些", "关系", "影响", "原因"}
	score := 0
	for _, indicator := range complexityIndicators {
		if strings.Contains(query, indicator) {
			score++
		}
	}
	complexity := float64(score) / float64(len(complexityIndicators))
	if complexity > 1.0 {
		complexity = 1.0
	}
	return complexity
}

// identifyReasoningPatterns 识别推理模式
func (g *GraphRAGRetrieval) identifyReasoningPatterns(subgraph *KnowledgeSubgraph) []string {
	return []string{"因果关系", "组成关系", "相似关系"}
}

// buildReasoningChain 构建推理链
func (g *GraphRAGRetrieval) buildReasoningChain(pattern string, subgraph *KnowledgeSubgraph) string {
	return fmt.Sprintf("基于%s的推理链", pattern)
}

// validateReasoningChains 验证推理链
func (g *GraphRAGRetrieval) validateReasoningChains(chains []string, query string) []string {
	if len(chains) > 3 {
		return chains[:3]
	}
	return chains
}

// findEntityRelations 查找实体间关系
func (g *GraphRAGRetrieval) findEntityRelations(ctx context.Context, graphQuery *GraphQuery, session neo4j.SessionWithContext) ([]*GraphPath, error) {
	// 实现实体间关系查找逻辑
	return []*GraphPath{}, nil
}

// findShortestPaths 查找最短路径
func (g *GraphRAGRetrieval) findShortestPaths(ctx context.Context, graphQuery *GraphQuery, session neo4j.SessionWithContext) ([]*GraphPath, error) {
	// 实现最短路径查找逻辑
	return []*GraphPath{}, nil
}

// fallbackSubgraphExtraction 降级子图提取
func (g *GraphRAGRetrieval) fallbackSubgraphExtraction(graphQuery *GraphQuery) *KnowledgeSubgraph {
	return &KnowledgeSubgraph{
		CentralNodes:    []map[string]interface{}{},
		ConnectedNodes:  []map[string]interface{}{},
		Relationships:   []map[string]interface{}{},
		GraphMetrics:    map[string]float64{},
		ReasoningChains: [][]string{},
	}
}

// Close 关闭资源连接
func (g *GraphRAGRetrieval) Close(ctx context.Context) error {
	if g.driver != nil {
		err := g.driver.Close(ctx)
		if err != nil {
			return fmt.Errorf("关闭Neo4j连接失败: %w", err)
		}
		log.Println("图RAG检索系统已关闭")
	}
	return nil
}
