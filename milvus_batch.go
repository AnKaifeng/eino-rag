/*
Milvus向量索引构建模块 - RAG系统的向量化数据存储引擎

这个模块负责将文本文档转换为向量表示并在Milvus向量数据库中建立高效的索引。
它是RAG系统向量检索能力的基础，提供语义相似度搜索功能。

核心功能：
1. 文档向量化：使用预训练的中文embedding模型将文本转换为向量
2. 向量索引构建：在Milvus中创建和管理向量集合
3. 高效检索：基于余弦相似度的快速向量搜索
4. 元数据管理：保留原始文档的丰富元数据信息

技术架构：
- Milvus：高性能向量数据库，支持大规模向量检索
- Ark Embeddings：使用豆包embedding模型进行向量化
- HNSW索引：层次化可导航小世界图，实现亚线性时间复杂度搜索
- 余弦距离：语义相似度度量，适合文本检索任务

数据流程：
1. 文档预处理：文本清洗和格式化
2. 向量生成：batch方式生成embedding向量
3. 索引构建：创建HNSW索引优化检索性能
4. 相似度搜索：基于查询向量检索最相关文档

适用场景：
- 语义相似度搜索：找到与查询语义相近的文档
- 推荐系统：基于内容的相似菜谱推荐
- 聚类分析：发现相似的烹饪主题和模式
*/

package batch_0001

import (
	"context"
	"encoding/binary"
	"fmt"
	"log"
	"math"
	"strings"
	"time"

	"github.com/cloudwego/eino-ext/components/embedding/ark"
	"github.com/cloudwego/eino/schema"
	"github.com/milvus-io/milvus/client/v2/entity"
	"github.com/milvus-io/milvus/client/v2/index"
	"github.com/milvus-io/milvus/client/v2/milvusclient"
)

// 配置常量
const (
	DefaultHost           = "localhost"
	DefaultPort           = "19530"
	DefaultCollectionName = "cooking_knowledge"
	DefaultDimension      = 81920 // 豆包embedding模型的向量维度
	DefaultModelName      = "doubao-embedding-text-240715"
	DefaultAPIKey         = "be0d94ab-5e0f-48ca-87bf-5522007ba28e"
	BatchSize             = 100
	MaxRetries            = 3
)

// 文档实体结构
type DocumentEntity struct {
	ID          string                 `json:"id"`           // 主键，唯一标识
	Vector      []byte                 `json:"vector"`       // 向量表示（二进制格式）
	Text        string                 `json:"text"`         // 原始文本内容
	NodeID      string                 `json:"node_id"`      // 图数据库节点ID
	RecipeName  string                 `json:"recipe_name"`  // 菜谱名称
	NodeType    string                 `json:"node_type"`    // 节点类型
	Category    string                 `json:"category"`     // 菜品分类
	CuisineType string                 `json:"cuisine_type"` // 菜系类型
	Difficulty  int64                  `json:"difficulty"`   // 难度等级
	DocType     string                 `json:"doc_type"`     // 文档类型
	ChunkID     string                 `json:"chunk_id"`     // 分块ID
	ParentID    string                 `json:"parent_id"`    // 父文档ID
	Metadata    map[string]interface{} `json:"metadata"`     // 额外元数据
}

// 搜索结果结构
type SearchResult struct {
	ID       string                 `json:"id"`       // 文档ID
	Score    float32                `json:"score"`    // 相似度分数（余弦距离）
	Text     string                 `json:"text"`     // 文档内容
	Metadata map[string]interface{} `json:"metadata"` // 元数据信息
}

// 集合统计信息
type CollectionStats struct {
	CollectionName        string                 `json:"collection_name"`
	RowCount              int64                  `json:"row_count"`
	IndexBuildingProgress int                    `json:"index_building_progress"`
	Stats                 map[string]interface{} `json:"stats"`
	Error                 string                 `json:"error,omitempty"`
}

// 过滤条件
type SearchFilters map[string]interface{}

// MilvusIndexConstructionModule Milvus向量索引构建模块
type MilvusIndexConstructionModule struct {
	host              string
	port              string
	collectionName    string
	dimension         int64
	modelName         string
	apiKey            string
	client            *milvusclient.Client
	embedder          *ark.Embedder
	collectionCreated bool
	indexCreated      bool
}

// NewMilvusIndexConstructionModule 创建新的Milvus索引构建模块
//
// 初始化Milvus索引构建模块
//
// Args:
//   - host: Milvus服务器地址，默认本地部署
//   - port: Milvus服务器端口，默认19530
//   - collectionName: 向量集合名称，用于组织数据
//   - dimension: 向量维度，需与embedding模型匹配
//   - modelName: Ark embedding模型名称
//   - apiKey: API密钥
//
// Note:
//
//	初始化时会自动连接Milvus服务器并设置embedding模型
func NewMilvusIndexConstructionModule(host, port, collectionName string, dimension int64, modelName, apiKey string) *MilvusIndexConstructionModule {
	if host == "" {
		host = DefaultHost
	}
	if port == "" {
		port = DefaultPort
	}
	if collectionName == "" {
		collectionName = DefaultCollectionName
	}
	if dimension == 0 {
		dimension = DefaultDimension
	}
	if modelName == "" {
		modelName = DefaultModelName
	}
	if apiKey == "" {
		apiKey = DefaultAPIKey
	}

	return &MilvusIndexConstructionModule{
		host:           host,
		port:           port,
		collectionName: collectionName,
		dimension:      dimension,
		modelName:      modelName,
		apiKey:         apiKey,
	}
}

// safeTruncate 安全截取字符串，处理空值和长度限制
//
// Milvus对字符串字段有长度限制，此方法确保数据符合schema要求。
// 同时处理可能的空值，避免运行时错误。
//
// Args:
//   - text: 输入文本，可能为空
//   - maxLength: 最大允许长度
//
// Returns:
//   - string: 安全截取后的字符串
func (m *MilvusIndexConstructionModule) safeTruncate(text string, maxLength int) string {
	if text == "" {
		return ""
	}
	if len(text) <= maxLength {
		return text
	}
	return text[:maxLength]
}

// setupClient 初始化Milvus客户端连接
//
// 建立与Milvus向量数据库的连接，并验证连接有效性。
// 使用TCP协议进行通信，支持远程和本地部署。
//
// Returns:
//   - error: 连接失败时返回错误信息
func (m *MilvusIndexConstructionModule) setupClient(ctx context.Context) error {
	if m.client != nil {
		return nil // 已经连接
	}

	addr := fmt.Sprintf("%s:%s", m.host, m.port)
	client, err := milvusclient.New(ctx, &milvusclient.ClientConfig{
		Address:  addr,
		Username: "root",
		Password: "rootroot",
		DBName:   "default",
	})
	if err != nil {
		return fmt.Errorf("连接Milvus失败 [%s]: %v", addr, err)
	}

	m.client = client
	log.Printf("已连接到Milvus服务器: %s", addr)

	// 测试连接 - 获取集合列表验证连接有效性
	_, err = m.client.ListCollections(ctx, milvusclient.NewListCollectionOption())
	if err != nil {
		return fmt.Errorf("验证Milvus连接失败: %v", err)
	}

	log.Printf("Milvus连接验证成功")
	return nil
}

// setupEmbeddings 初始化文本嵌入模型
//
// 加载预训练的中文embedding模型，用于将文本转换为向量表示。
//
// 模型特点：
// - doubao-embedding-text-240715：优秀的中文语义理解能力
// - 高维向量：强大的表示能力
// - API调用：便于部署和扩展
//
// Returns:
//   - error: 初始化失败时返回错误信息
func (m *MilvusIndexConstructionModule) setupEmbeddings(ctx context.Context) error {
	if m.embedder != nil {
		return nil // 已经初始化
	}

	log.Printf("正在初始化嵌入模型: %s", m.modelName)

	embedder, err := ark.NewEmbedder(ctx, &ark.EmbeddingConfig{
		APIKey: m.apiKey,
		Model:  m.modelName,
	})

	if err != nil {
		return fmt.Errorf("初始化embedding模型失败: %v", err)
	}

	m.embedder = embedder
	log.Printf("嵌入模型初始化完成")
	return nil
}

// createCollectionSchema 创建Milvus集合的数据模式
//
// 定义向量集合的字段结构，包括向量字段和元数据字段。
// 设计考虑了中式烹饪知识的特点，支持丰富的查询和过滤需求。
//
// 字段设计说明：
// - id: 主键，使用菜谱相关的唯一标识
// - vector: 向量字段，存储文档的embedding表示
// - text: 原始文本内容，支持结果展示
// - 元数据字段: 支持分类、难度、菜系等多维度过滤
//
// Returns:
//   - *entity.Schema: Milvus集合模式对象
func (m *MilvusIndexConstructionModule) createCollectionSchema() *entity.Schema {
	schema := entity.NewSchema().WithDynamicFieldEnabled(false).
		// 主键字段：唯一标识每个文档块
		WithField(entity.NewField().WithName("id").WithDataType(entity.FieldTypeVarChar).WithMaxLength(150).WithIsPrimaryKey(true)).
		// 向量字段：存储文档的embedding表示
		WithField(entity.NewField().WithName("vector").WithDataType(entity.FieldTypeBinaryVector).WithDim(m.dimension)).
		// 文本内容字段：原始文档内容
		WithField(entity.NewField().WithName("text").WithDataType(entity.FieldTypeVarChar).WithMaxLength(15000)).
		// 图数据库相关字段
		WithField(entity.NewField().WithName("node_id").WithDataType(entity.FieldTypeVarChar).WithMaxLength(100)).
		WithField(entity.NewField().WithName("recipe_name").WithDataType(entity.FieldTypeVarChar).WithMaxLength(300)).
		WithField(entity.NewField().WithName("node_type").WithDataType(entity.FieldTypeVarChar).WithMaxLength(100)).
		// 菜谱属性字段：支持基于属性的过滤检索
		WithField(entity.NewField().WithName("category").WithDataType(entity.FieldTypeVarChar).WithMaxLength(100)).     // 菜品分类
		WithField(entity.NewField().WithName("cuisine_type").WithDataType(entity.FieldTypeVarChar).WithMaxLength(200)). // 菜系类型
		WithField(entity.NewField().WithName("difficulty").WithDataType(entity.FieldTypeInt64)).                        // 难度等级
		// 文档处理相关字段
		WithField(entity.NewField().WithName("doc_type").WithDataType(entity.FieldTypeVarChar).WithMaxLength(50)).  // 文档类型
		WithField(entity.NewField().WithName("chunk_id").WithDataType(entity.FieldTypeVarChar).WithMaxLength(150)). // 分块ID
		WithField(entity.NewField().WithName("parent_id").WithDataType(entity.FieldTypeVarChar).WithMaxLength(100)) // 父文档ID

	return schema
}

// CreateCollection 创建Milvus集合
//
// Args:
//   - forceRecreate: 是否强制重新创建集合
//
// Returns:
//   - bool: 是否创建成功
//   - error: 错误信息
func (m *MilvusIndexConstructionModule) CreateCollection(ctx context.Context, forceRecreate bool) (bool, error) {
	if err := m.setupClient(ctx); err != nil {
		return false, err
	}

	// 检查集合是否存在
	exists, err := m.client.HasCollection(ctx, milvusclient.NewHasCollectionOption(m.collectionName))
	if err != nil {
		return false, fmt.Errorf("检查集合存在性失败: %v", err)
	}

	if exists {
		if forceRecreate {
			log.Printf("删除已存在的集合: %s", m.collectionName)
			err = m.client.DropCollection(ctx, milvusclient.NewDropCollectionOption(m.collectionName))
			if err != nil {
				return false, fmt.Errorf("删除集合失败: %v", err)
			}
		} else {
			log.Printf("集合 %s 已存在", m.collectionName)
			m.collectionCreated = true
			return true, nil
		}
	}

	// 创建集合
	schema := m.createCollectionSchema()

	// 创建索引配置
	indexOptions := []milvusclient.CreateIndexOption{
		// milvusclient.NewCreateIndexOption(m.collectionName, "vector", index.NewHNSWIndex(entity.HAMMING, 16, 200)),
		milvusclient.NewCreateIndexOption(m.collectionName, "vector", index.NewBinFlatIndex(entity.HAMMING)),
	}

	err = m.client.CreateCollection(ctx, milvusclient.NewCreateCollectionOption(m.collectionName, schema).WithIndexOptions(indexOptions...))
	if err != nil {
		return false, fmt.Errorf("创建集合失败: %v", err)
	}

	log.Printf("成功创建集合: %s", m.collectionName)
	m.collectionCreated = true
	m.indexCreated = true

	return true, nil
}

// vector2Bytes 将float64向量转换为字节数组
//
// 用于Milvus二进制向量存储的数据格式转换
//
// Args:
//   - vector: float64向量
//
// Returns:
//   - []byte: 二进制向量数据
func (m *MilvusIndexConstructionModule) vector2Bytes(vector []float64) []byte {
	float32Arr := make([]float32, len(vector))
	for i, v := range vector {
		float32Arr[i] = float32(v)
	}
	bytes := make([]byte, len(float32Arr)*4)
	for i, v := range float32Arr {
		binary.LittleEndian.PutUint32(bytes[i*4:], math.Float32bits(v))
	}
	return bytes
}

// BuildVectorIndex 构建向量索引
//
// 这是核心方法，负责将文档转换为向量并建立索引
//
// Args:
//   - chunks: 文档块列表
//
// Returns:
//   - error: 构建失败时返回错误信息
func (m *MilvusIndexConstructionModule) BuildVectorIndex(ctx context.Context, chunks []*schema.Document) error {
	log.Printf("正在构建Milvus向量索引，文档数量: %d...", len(chunks))

	if len(chunks) == 0 {
		return fmt.Errorf("文档块列表不能为空")
	}

	// 1. 初始化连接和embedding
	if err := m.setupClient(ctx); err != nil {
		return err
	}
	if err := m.setupEmbeddings(ctx); err != nil {
		return err
	}

	// 2. 创建集合（如果schema不兼容则强制重新创建）
	success, err := m.CreateCollection(ctx, true)
	if err != nil || !success {
		return fmt.Errorf("创建集合失败: %v", err)
	}

	// 3. 准备数据并批量插入
	log.Printf("正在生成向量embeddings...")
	return m.insertDocumentsBatch(ctx, chunks)
}

// insertDocumentsBatch 批量插入文档
func (m *MilvusIndexConstructionModule) insertDocumentsBatch(ctx context.Context, chunks []*schema.Document) error {
	for i := 0; i < len(chunks); i += BatchSize {
		end := i + BatchSize
		if end > len(chunks) {
			end = len(chunks)
		}

		batch := chunks[i:end]
		if err := m.insertSingleBatch(ctx, batch); err != nil {
			return fmt.Errorf("批量插入失败 [%d-%d]: %v", i, end-1, err)
		}

		log.Printf("已插入 %d/%d 条数据", end, len(chunks))
	}

	// 4. 加载集合到内存
	_, err := m.client.LoadCollection(ctx, milvusclient.NewLoadCollectionOption(m.collectionName))
	if err != nil {
		return fmt.Errorf("加载集合到内存失败: %v", err)
	}
	log.Printf("集合已加载到内存")

	// 5. 等待索引构建完成
	log.Printf("等待索引构建完成...")
	time.Sleep(2 * time.Second)

	log.Printf("向量索引构建完成，包含 %d 个向量", len(chunks))
	return nil
}

// insertSingleBatch 插入单个批次
func (m *MilvusIndexConstructionModule) insertSingleBatch(ctx context.Context, batch []*schema.Document) error {
	// 准备数据切片
	ids := make([]string, 0, len(batch))
	texts := make([]string, 0, len(batch))
	nodeIDs := make([]string, 0, len(batch))
	recipeNames := make([]string, 0, len(batch))
	nodeTypes := make([]string, 0, len(batch))
	categories := make([]string, 0, len(batch))
	cuisineTypes := make([]string, 0, len(batch))
	difficulties := make([]int64, 0, len(batch))
	docTypes := make([]string, 0, len(batch))
	chunkIDs := make([]string, 0, len(batch))
	parentIDs := make([]string, 0, len(batch))

	// 提取文本用于向量化
	batchTexts := make([]string, len(batch))
	for i, doc := range batch {
		batchTexts[i] = doc.Content
	}

	// 生成向量
	vectors, err := m.embedder.EmbedStrings(ctx, batchTexts)
	if err != nil {
		return fmt.Errorf("生成向量失败: %v", err)
	}

	// 准备插入数据
	vectorBytes := make([][]byte, 0, len(batch))
	for i, doc := range batch {
		// 安全获取元数据
		getStringMeta := func(key string) string {
			if val, exists := doc.MetaData[key]; exists {
				if str, ok := val.(string); ok {
					return str
				}
			}
			return ""
		}

		getInt64Meta := func(key string) int64 {
			if val, exists := doc.MetaData[key]; exists {
				switch v := val.(type) {
				case int64:
					return v
				case int:
					return int64(v)
				case float64:
					return int64(v)
				}
			}
			return 0
		}

		// 应用长度限制
		ids = append(ids, m.safeTruncate(doc.ID, 150))
		texts = append(texts, m.safeTruncate(doc.Content, 15000))
		nodeIDs = append(nodeIDs, m.safeTruncate(getStringMeta("node_id"), 100))
		recipeNames = append(recipeNames, m.safeTruncate(getStringMeta("recipe_name"), 300))
		nodeTypes = append(nodeTypes, m.safeTruncate(getStringMeta("node_type"), 100))
		categories = append(categories, m.safeTruncate(getStringMeta("category"), 100))
		cuisineTypes = append(cuisineTypes, m.safeTruncate(getStringMeta("cuisine_type"), 200))
		difficulties = append(difficulties, getInt64Meta("difficulty"))
		docTypes = append(docTypes, m.safeTruncate(getStringMeta("doc_type"), 50))
		chunkIDs = append(chunkIDs, m.safeTruncate(getStringMeta("chunk_id"), 150))
		parentIDs = append(parentIDs, m.safeTruncate(getStringMeta("parent_id"), 100))

		// 转换向量为字节
		vectorBytes = append(vectorBytes, m.vector2Bytes(vectors[i]))
	}

	// 执行插入
	_, err = m.client.Insert(ctx, milvusclient.NewColumnBasedInsertOption(m.collectionName).
		WithVarcharColumn("id", ids).
		WithBinaryVectorColumn("vector", int(m.dimension), vectorBytes).
		WithVarcharColumn("text", texts).
		WithVarcharColumn("node_id", nodeIDs).
		WithVarcharColumn("recipe_name", recipeNames).
		WithVarcharColumn("node_type", nodeTypes).
		WithVarcharColumn("category", categories).
		WithVarcharColumn("cuisine_type", cuisineTypes).
		WithInt64Column("difficulty", difficulties).
		WithVarcharColumn("doc_type", docTypes).
		WithVarcharColumn("chunk_id", chunkIDs).
		WithVarcharColumn("parent_id", parentIDs))

	return err
}

// AddDocuments 向现有索引添加新文档
//
// Args:
//   - newChunks: 新的文档块列表
//
// Returns:
//   - error: 添加失败时返回错误信息
func (m *MilvusIndexConstructionModule) AddDocuments(ctx context.Context, newChunks []*schema.Document) error {
	if !m.collectionCreated {
		return fmt.Errorf("请先构建向量索引")
	}

	log.Printf("正在添加 %d 个新文档到索引...", len(newChunks))

	if err := m.setupEmbeddings(ctx); err != nil {
		return err
	}

	return m.insertDocumentsBatch(ctx, newChunks)
}

// SimilaritySearch 相似度搜索
//
// Args:
//   - query: 查询文本
//   - topK: 返回结果数量
//   - filters: 过滤条件
//
// Returns:
//   - []SearchResult: 搜索结果列表
//   - error: 搜索失败时返回错误信息
func (m *MilvusIndexConstructionModule) SimilaritySearch(ctx context.Context, query string, topK int, filters SearchFilters) ([]SearchResult, error) {
	if !m.collectionCreated {
		return nil, fmt.Errorf("请先构建或加载向量索引")
	}

	if err := m.setupEmbeddings(ctx); err != nil {
		return nil, err
	}

	// 生成查询向量
	queryVectors, err := m.embedder.EmbedStrings(ctx, []string{query})
	if err != nil {
		return nil, fmt.Errorf("生成查询向量失败: %v", err)
	}

	queryBytes := m.vector2Bytes(queryVectors[0])

	// 构建过滤表达式
	filterExpr := m.buildFilterExpression(filters)

	// 创建搜索参数
	annParam := index.NewHNSWAnnParam(64)
	searchOption := milvusclient.NewSearchOption(m.collectionName, topK, []entity.Vector{entity.BinaryVector(queryBytes)}).
		WithANNSField("vector").
		WithOutputFields("text", "node_id", "recipe_name", "node_type", "category", "cuisine_type", "difficulty", "doc_type", "chunk_id", "parent_id").
		WithSearchParam("metric_type", "COSINE").
		WithAnnParam(annParam)

	// 添加过滤条件
	if filterExpr != "" {
		searchOption.WithFilter(filterExpr)
	}

	// 执行搜索
	resultSets, err := m.client.Search(ctx, searchOption)
	if err != nil {
		return nil, fmt.Errorf("相似度搜索失败: %v", err)
	}

	// 处理结果
	var results []SearchResult
	if len(resultSets) > 0 {
		res := resultSets[0]
		idCol := res.GetColumn("id")     // 主键列
		textCol := res.GetColumn("text") // 文本列
		nodeCol := res.GetColumn("node_id")
		recipeCol := res.GetColumn("recipe_name")
		typeCol := res.GetColumn("node_type")
		cateCol := res.GetColumn("category")
		cuisineCol := res.GetColumn("cuisine_type")
		diffCol := res.GetColumn("difficulty")
		docCol := res.GetColumn("doc_type")
		chunkCol := res.GetColumn("chunk_id")
		parentCol := res.GetColumn("parent_id")

		ids := idCol.FieldData().GetScalars().GetStringData().GetData()
		texts := textCol.FieldData().GetScalars().GetStringData().GetData()
		nodes := nodeCol.FieldData().GetScalars().GetStringData().GetData()
		recipes := recipeCol.FieldData().GetScalars().GetStringData().GetData()
		types := typeCol.FieldData().GetScalars().GetStringData().GetData()
		cates := cateCol.FieldData().GetScalars().GetStringData().GetData()
		cuisines := cuisineCol.FieldData().GetScalars().GetStringData().GetData()
		diffs := diffCol.FieldData().GetScalars().GetLongData().GetData()
		docs := docCol.FieldData().GetScalars().GetStringData().GetData()
		chunks := chunkCol.FieldData().GetScalars().GetStringData().GetData()
		parents := parentCol.FieldData().GetScalars().GetStringData().GetData()

		count := res.ResultCount

		for i := 0; i < int(count); i++ {
			results = append(results, SearchResult{
				ID:    ids[i],
				Score: res.Scores[i],
				Text:  texts[i],
				Metadata: map[string]interface{}{
					"node_id":      nodes[i],
					"recipe_name":  recipes[i],
					"node_type":    types[i],
					"category":     cates[i],
					"cuisine_type": cuisines[i],
					"difficulty":   diffs[i],
					"doc_type":     docs[i],
					"chunk_id":     chunks[i],
					"parent_id":    parents[i],
				},
			})
		}

	}

	return results, nil
}

// buildFilterExpression 构建过滤表达式
func (m *MilvusIndexConstructionModule) buildFilterExpression(filters SearchFilters) string {
	if len(filters) == 0 {
		return ""
	}

	var conditions []string
	for key, value := range filters {
		switch v := value.(type) {
		case string:
			conditions = append(conditions, fmt.Sprintf(`%s == "%s"`, key, v))
		case int, int64, float64:
			conditions = append(conditions, fmt.Sprintf(`%s == %v`, key, v))
		case []string:
			quotedValues := make([]string, len(v))
			for i, val := range v {
				quotedValues[i] = fmt.Sprintf(`"%s"`, val)
			}
			conditions = append(conditions, fmt.Sprintf(`%s in [%s]`, key, strings.Join(quotedValues, ", ")))
		case []interface{}:
			var quotedValues []string
			for _, val := range v {
				if str, ok := val.(string); ok {
					quotedValues = append(quotedValues, fmt.Sprintf(`"%s"`, str))
				} else {
					quotedValues = append(quotedValues, fmt.Sprintf(`%v`, val))
				}
			}
			conditions = append(conditions, fmt.Sprintf(`%s in [%s]`, key, strings.Join(quotedValues, ", ")))
		}
	}

	return strings.Join(conditions, " and ")
}

// GetCollectionStats 获取集合统计信息
//
// Returns:
//   - *CollectionStats: 统计信息
//   - error: 获取失败时返回错误信息
func (m *MilvusIndexConstructionModule) GetCollectionStats(ctx context.Context) (*CollectionStats, error) {
	if err := m.setupClient(ctx); err != nil {
		return &CollectionStats{Error: err.Error()}, err
	}

	if !m.collectionCreated {
		return &CollectionStats{
			CollectionName: m.collectionName,
			Error:          "集合未创建",
		}, nil
	}

	// 获取集合信息
	stats := &CollectionStats{
		CollectionName: m.collectionName,
		RowCount:       0,
		Stats:          make(map[string]interface{}),
	}

	// 这里可以添加更多统计信息的获取逻辑
	// 目前Milvus Go SDK可能不直接支持获取详细统计信息

	return stats, nil
}

// DeleteCollection 删除集合
//
// Returns:
//   - error: 删除失败时返回错误信息
func (m *MilvusIndexConstructionModule) DeleteCollection(ctx context.Context) error {
	if err := m.setupClient(ctx); err != nil {
		return err
	}

	exists, err := m.client.HasCollection(ctx, milvusclient.NewHasCollectionOption(m.collectionName))
	if err != nil {
		return fmt.Errorf("检查集合存在性失败: %v", err)
	}

	if exists {
		err = m.client.DropCollection(ctx, milvusclient.NewDropCollectionOption(m.collectionName))
		if err != nil {
			return fmt.Errorf("删除集合失败: %v", err)
		}
		log.Printf("集合 %s 已删除", m.collectionName)
		m.collectionCreated = false
		m.indexCreated = false
	} else {
		log.Printf("集合 %s 不存在", m.collectionName)
	}

	return nil
}

// HasCollection 检查集合是否存在
//
// Returns:
//   - bool: 集合是否存在
//   - error: 检查失败时返回错误信息
func (m *MilvusIndexConstructionModule) HasCollection(ctx context.Context) (bool, error) {
	if err := m.setupClient(ctx); err != nil {
		return false, err
	}

	return m.client.HasCollection(ctx, milvusclient.NewHasCollectionOption(m.collectionName))
}

// LoadCollection 加载集合到内存
//
// Returns:
//   - error: 加载失败时返回错误信息
func (m *MilvusIndexConstructionModule) LoadCollection(ctx context.Context) error {
	if err := m.setupClient(ctx); err != nil {
		return err
	}

	exists, err := m.HasCollection(ctx)
	if err != nil {
		return err
	}

	if !exists {
		return fmt.Errorf("集合 %s 不存在", m.collectionName)
	}

	_, err = m.client.LoadCollection(ctx, milvusclient.NewLoadCollectionOption(m.collectionName))
	if err != nil {
		return fmt.Errorf("加载集合失败: %v", err)
	}

	m.collectionCreated = true
	log.Printf("集合 %s 已加载到内存", m.collectionName)
	return nil
}

// Close 关闭连接
func (m *MilvusIndexConstructionModule) Close(ctx context.Context) {
	if m.client != nil {
		m.client.Close(ctx)
		log.Printf("Milvus连接已关闭")
	}
}
