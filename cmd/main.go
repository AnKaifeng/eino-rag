package main

import (
	"bufio"
	"context"
	"fmt"
	"log"
	"os"
	"strings"
	"time"

	batch "batch-0001"

	"github.com/cloudwego/eino-ext/components/model/ark"
)

// GraphRAGConfig 系统配置结构体
type GraphRAGConfig struct {
	// Neo4j配置
	Neo4jURI      string `json:"neo4j_uri"`
	Neo4jUser     string `json:"neo4j_user"`
	Neo4jPassword string `json:"neo4j_password"`
	Neo4jDatabase string `json:"neo4j_database"`

	// Milvus配置
	MilvusHost           string `json:"milvus_host"`
	MilvusPort           string `json:"milvus_port"`
	MilvusCollectionName string `json:"milvus_collection_name"`
	MilvusDimension      int    `json:"milvus_dimension"`

	// LLM配置
	LLMModel       string  `json:"llm_model"`
	EmbeddingModel string  `json:"embedding_model"`
	Temperature    float32 `json:"temperature"`
	MaxTokens      int     `json:"max_tokens"`

	// 检索配置
	TopK         int `json:"top_k"`
	ChunkSize    int `json:"chunk_size"`
	ChunkOverlap int `json:"chunk_overlap"`

	//ark api key
	ApiKey string `json:"api_key"`
}

// DefaultConfig 默认配置
var DefaultConfig = &GraphRAGConfig{
	Neo4jURI:             "bolt://localhost:7687",
	Neo4jUser:            "neo4j",
	Neo4jPassword:        "rootroot",
	Neo4jDatabase:        "neo4j",
	MilvusHost:           "localhost",
	MilvusPort:           "19530",
	MilvusCollectionName: "cooking_recipes",
	MilvusDimension:      1536,
	LLMModel:             "doubao-1-5-pro-256k-250115",
	EmbeddingModel:       "doubao-seed-1-6-thinking-250715",
	ApiKey:               "be0d94ab-5e0f-48ca-87bf-5522007ba28e",
	Temperature:          0.1,
	MaxTokens:            2048,
	TopK:                 5,
	ChunkSize:            512,
	ChunkOverlap:         50,
}

// AdvancedGraphRAGSystem 高级图RAG系统
//
// 核心特性：
// 1. 智能路由：自动选择最适合的检索策略
// 2. 双引擎检索：传统混合检索 + 图RAG检索
// 3. 图结构推理：多跳遍历、子图提取、关系推理
// 4. 查询复杂度分析：深度理解用户意图
// 5. 自适应学习：基于反馈优化系统性能
type AdvancedGraphRAGSystem struct {
	config *GraphRAGConfig

	// 核心模块
	dataModule       *batch.GraphDataPreparationModule
	indexModule      *batch.MilvusIndexConstructionModule
	generationModule *batch.GenerationIntegrationModule

	// 检索引擎
	traditionalRetrieval *batch.HybridRetrievalModule
	graphRAGRetrieval    *batch.GraphRAGRetrieval
	queryRouter          *batch.IntelligentQueryRouter

	//llm
	model *ark.ChatModel

	// 系统状态
	systemReady bool
}

// NewAdvancedGraphRAGSystem 创建高级图RAG系统
func NewAdvancedGraphRAGSystem(config *GraphRAGConfig) *AdvancedGraphRAGSystem {
	if config == nil {
		config = DefaultConfig
	}
	return &AdvancedGraphRAGSystem{
		config:      config,
		systemReady: false,
	}
}

// InitializeSystem 初始化高级图RAG系统
func (s *AdvancedGraphRAGSystem) InitializeSystem(ctx context.Context) error {
	log.Println("启动高级图RAG系统...")

	// 1. 数据准备模块
	fmt.Println("初始化数据准备模块...")
	var err error

	s.model, err = ark.NewChatModel(ctx, &ark.ChatModelConfig{
		Model:       s.config.LLMModel,
		APIKey:      s.config.ApiKey,
		Temperature: &s.config.Temperature,
		MaxTokens:   &s.config.MaxTokens,
	})
	if err != nil {
		return fmt.Errorf("初始化LLM模型失败: %v", err)
	}

	s.dataModule, err = batch.NewGraphDataPreparationModule(
		s.config.Neo4jURI,
		s.config.Neo4jUser,
		s.config.Neo4jPassword,
		s.config.Neo4jDatabase,
	)
	if err != nil {
		return fmt.Errorf("初始化数据准备模块失败: %v", err)
	}

	// 2. 向量索引模块
	fmt.Println("初始化Milvus向量索引...")
	s.indexModule = batch.NewMilvusIndexConstructionModule(
		s.config.MilvusHost,
		s.config.MilvusPort,
		s.config.MilvusCollectionName,
		0,
		s.config.EmbeddingModel,
		s.config.ApiKey,
	)

	// 3. 生成模块
	fmt.Println("初始化生成模块...")
	s.generationModule = batch.NewGenerationIntegrationModule(
		s.config.LLMModel,
		s.config.ApiKey,
		s.config.Temperature,
		s.config.MaxTokens,
	)

	// 初始化生成模块
	if err := s.generationModule.Initialize(ctx); err != nil {
		return fmt.Errorf("初始化生成模块失败: %v", err)
	}

	// 4. 创建系统配置
	systemConfig := &batch.Config{
		Neo4jURI:      s.config.Neo4jURI,
		Neo4jUser:     s.config.Neo4jUser,
		Neo4jPassword: s.config.Neo4jPassword,
		LLMModel:      s.config.LLMModel,
		ArkAPIKey:     os.Getenv("ARK_API_KEY"),
		ArkBaseURL:    os.Getenv("ARK_BASE_URL"),
		Constraints:   make(map[string]interface{}),
	}

	// 5. 传统混合检索模块
	fmt.Println("初始化传统混合检索...")
	s.traditionalRetrieval = batch.NewHybridRetrievalModule(
		systemConfig,
		s.indexModule,
		s.dataModule,
		s.model,
	)

	// 6. 图RAG检索模块
	fmt.Println("初始化图RAG检索引擎...")
	s.graphRAGRetrieval = batch.NewGraphRAGRetrieval(systemConfig)

	// 7. 智能查询路由器
	fmt.Println("初始化智能查询路由器...")
	s.queryRouter = batch.NewIntelligentQueryRouter(
		s.traditionalRetrieval,
		s.graphRAGRetrieval,
		s.generationModule,
		systemConfig,
	)

	fmt.Println("✅ 高级图RAG系统初始化完成！")
	return nil
}

// BuildKnowledgeBase 构建知识库
func (s *AdvancedGraphRAGSystem) BuildKnowledgeBase(ctx context.Context) error {
	fmt.Println("\n检查知识库状态...")

	// 检查Milvus集合是否存在
	exists, err := s.indexModule.HasCollection(ctx)
	if err != nil {
		return fmt.Errorf("检查Milvus集合失败: %v", err)
	}

	if exists {
		fmt.Println("✅ 发现已存在的知识库，尝试加载...")
		if err := s.indexModule.LoadCollection(ctx); err != nil {
			fmt.Println("❌ 知识库加载失败，开始重建...")
		} else {
			fmt.Println("知识库加载成功！")
			return s.initializeRetrievers(ctx)
		}
	}

	fmt.Println("未找到已存在的集合，开始构建新的知识库...")

	// 从Neo4j加载图数据
	fmt.Println("从Neo4j加载图数据...")
	_, err = s.dataModule.LoadGraphData()
	if err != nil {
		return fmt.Errorf("加载图数据失败: %v", err)
	}

	// 构建菜谱文档
	fmt.Println("构建菜谱文档...")
	_, err = s.dataModule.BuildRecipeDocuments()
	if err != nil {
		return fmt.Errorf("构建菜谱文档失败: %v", err)
	}

	// 进行文档分块
	fmt.Println("进行文档分块...")
	chunks, err := s.dataModule.ChunkDocuments(s.config.ChunkSize, s.config.ChunkOverlap)
	if err != nil {
		return fmt.Errorf("文档分块失败: %v", err)
	}

	// 构建Milvus向量索引
	fmt.Println("构建Milvus向量索引...")
	if err := s.indexModule.BuildVectorIndex(ctx, chunks); err != nil {
		return fmt.Errorf("构建向量索引失败: %v", err)
	}

	// 初始化检索器
	if err := s.initializeRetrievers(ctx); err != nil {
		return fmt.Errorf("初始化检索器失败: %v", err)
	}

	// 显示统计信息
	s.showKnowledgeBaseStats(ctx)

	fmt.Println("✅ 知识库构建完成！")
	return nil
}

// initializeRetrievers 初始化检索器
func (s *AdvancedGraphRAGSystem) initializeRetrievers(ctx context.Context) error {
	fmt.Println("初始化检索引擎...")

	// 获取文档块用于初始化传统检索器
	chunks := s.dataModule.Chunks

	// 初始化传统检索器
	if err := s.traditionalRetrieval.Initialize(ctx, chunks); err != nil {
		return fmt.Errorf("初始化传统检索器失败: %v", err)
	}

	// 初始化图RAG检索器
	if err := s.graphRAGRetrieval.Initialize(ctx); err != nil {
		return fmt.Errorf("初始化图RAG检索器失败: %v", err)
	}

	s.systemReady = true
	fmt.Println("✅ 检索引擎初始化完成！")
	return nil
}

// showKnowledgeBaseStats 显示知识库统计信息
func (s *AdvancedGraphRAGSystem) showKnowledgeBaseStats(ctx context.Context) {
	fmt.Println("\n知识库统计:")

	// 数据统计
	stats := s.dataModule.GetStatistics()
	if totalRecipes, ok := stats["total_recipes"].(int); ok {
		fmt.Printf("   菜谱数量: %d\n", totalRecipes)
	}
	if totalIngredients, ok := stats["total_ingredients"].(int); ok {
		fmt.Printf("   食材数量: %d\n", totalIngredients)
	}
	if totalCookingSteps, ok := stats["total_cooking_steps"].(int); ok {
		fmt.Printf("   烹饪步骤: %d\n", totalCookingSteps)
	}
	if totalDocuments, ok := stats["total_documents"].(int); ok {
		fmt.Printf("   文档数量: %d\n", totalDocuments)
	}
	if totalChunks, ok := stats["total_chunks"].(int); ok {
		fmt.Printf("   文本块数: %d\n", totalChunks)
	}

	// Milvus统计
	milvusStats, err := s.indexModule.GetCollectionStats(ctx)
	if err == nil {
		fmt.Printf("   向量索引: %d 条记录\n", milvusStats.RowCount)
	}

	// 路由统计
	routeStats := s.queryRouter.GetRouteStatistics()
	fmt.Printf("   路由统计: 总查询 %d 次\n", routeStats.TotalQueries)
}

// AskQuestionWithRouting 智能问答：自动选择最佳检索策略
func (s *AdvancedGraphRAGSystem) AskQuestionWithRouting(ctx context.Context, question string, stream bool, explainRouting bool) (string, *batch.QueryAnalysis, error) {
	if !s.systemReady {
		return "", nil, fmt.Errorf("系统未就绪，请先构建知识库")
	}

	fmt.Printf("\n❓ 用户问题: %s\n", question)

	startTime := time.Now()

	// 1. 智能路由检索
	fmt.Println("执行智能查询路由...")
	relevantDocs, analysis, err := s.queryRouter.RouteQuery(ctx, question, s.config.TopK)
	if err != nil {
		return "", nil, fmt.Errorf("路由查询失败: %v", err)
	}

	// 2. 显示路由信息
	strategyIcons := map[batch.SearchStrategy]string{
		batch.HybridTraditional: "🔍",
		batch.GraphRAG:          "🕸️",
		batch.Combined:          "🔄",
	}
	strategyIcon := strategyIcons[analysis.RecommendedStrategy]
	fmt.Printf("%s 使用策略: %s\n", strategyIcon, analysis.RecommendedStrategy)
	fmt.Printf("📊 复杂度: %.2f, 关系密集度: %.2f\n", analysis.QueryComplexity, analysis.RelationshipIntensity)

	// 3. 显示检索结果信息
	if len(relevantDocs) > 0 {
		var docInfo []string
		for _, doc := range relevantDocs {
			recipeName := "未知内容"
			if name, ok := doc.MetaData["recipe_name"].(string); ok {
				recipeName = name
			}
			searchType := "unknown"
			if sType, ok := doc.MetaData["search_type"].(string); ok {
				searchType = sType
			}
			score := 0.0
			if s, ok := doc.MetaData["final_score"].(float64); ok {
				score = s
			}
			docInfo = append(docInfo, fmt.Sprintf("%s(%s, %.3f)", recipeName, searchType, score))
		}

		fmt.Printf("📋 找到 %d 个相关文档:\n", len(relevantDocs))
		for i, info := range docInfo {
			fmt.Printf("    %d. %s\n", i+1, info)
		}
	} else {
		return "抱歉，没有找到相关的烹饪信息。请尝试其他问题。", analysis, nil
	}

	// 4. 生成回答
	fmt.Println("🎯 智能生成回答...")

	var result string
	if stream {
		// 流式输出
		resultChan := make(chan string, 100)
		go func() {
			s.generationModule.GenerateAdaptiveAnswerStream(ctx, question, relevantDocs, 3, resultChan)
		}()

		// 实时输出
		var chunks []string
		for chunk := range resultChan {
			fmt.Print(chunk)
			chunks = append(chunks, chunk)
		}
		fmt.Println()
		result = strings.Join(chunks, "")
	} else {
		// 非流式输出
		result, err = s.generationModule.GenerateAdaptiveAnswer(ctx, question, relevantDocs)
		if err != nil {
			return "", analysis, fmt.Errorf("生成回答失败: %v", err)
		}
	}

	// 5. 性能统计
	duration := time.Since(startTime)
	fmt.Printf("\n⏱️ 问答完成，耗时: %.2f秒\n", duration.Seconds())

	return result, analysis, nil
}

// RunInteractive 运行交互式问答
func (s *AdvancedGraphRAGSystem) RunInteractive(ctx context.Context) {
	if !s.systemReady {
		fmt.Println("❌ 系统未就绪，请先构建知识库")
		return
	}

	fmt.Println("\n欢迎使用尝尝咸淡RAG烹饪助手！")
	fmt.Println("可用功能：")
	fmt.Println("   - 'stats' : 查看系统统计")
	fmt.Println("   - 'rebuild' : 重建知识库")
	fmt.Println("   - 'quit' : 退出系统")
	fmt.Println("\n" + strings.Repeat("=", 50))

	scanner := bufio.NewScanner(os.Stdin)

	for {
		fmt.Print("\n您的问题: ")
		if !scanner.Scan() {
			break
		}

		userInput := strings.TrimSpace(scanner.Text())
		if userInput == "" {
			continue
		}

		switch strings.ToLower(userInput) {
		case "quit":
			fmt.Println("\n👋 感谢使用尝尝咸淡RAG烹饪助手！")
			return
		case "stats":
			s.showSystemStats(ctx)
			continue
		case "rebuild":
			s.rebuildKnowledgeBase(ctx)
			continue
		}

		// 普通问答
		useStream := true
		explainRouting := false

		fmt.Println("\n回答:")
		result, analysis, err := s.AskQuestionWithRouting(ctx, userInput, useStream, explainRouting)
		if err != nil {
			fmt.Printf("处理问题时出错: %v\n", err)
			continue
		}

		if !useStream && result != "" {
			fmt.Printf("%s\n", result)
		}

		// 显示分析信息（可选）
		if analysis != nil && explainRouting {
			fmt.Printf("\n📊 分析结果: 置信度 %.2f\n", analysis.Confidence)
		}
	}
}

// showSystemStats 显示系统统计信息
func (s *AdvancedGraphRAGSystem) showSystemStats(ctx context.Context) {
	fmt.Println("\n系统运行统计")
	fmt.Println(strings.Repeat("=", 40))

	// 路由统计
	routeStats := s.queryRouter.GetRouteStatistics()
	totalQueries := routeStats.TotalQueries

	if totalQueries > 0 {
		fmt.Printf("总查询次数: %d\n", totalQueries)
		fmt.Printf("传统检索: %d (%.1f%%)\n", routeStats.TraditionalCount, routeStats.TraditionalRatio*100)
		fmt.Printf("图RAG检索: %d (%.1f%%)\n", routeStats.GraphRAGCount, routeStats.GraphRAGRatio*100)
		fmt.Printf("组合策略: %d (%.1f%%)\n", routeStats.CombinedCount, routeStats.CombinedRatio*100)
	} else {
		fmt.Println("暂无查询记录")
	}

	// 知识库统计
	s.showKnowledgeBaseStats(ctx)
}

// rebuildKnowledgeBase 重建知识库
func (s *AdvancedGraphRAGSystem) rebuildKnowledgeBase(ctx context.Context) {
	fmt.Println("\n准备重建知识库...")

	// 确认操作
	fmt.Print("⚠️  这将删除现有的向量数据并重新构建，是否继续？(y/N): ")
	scanner := bufio.NewScanner(os.Stdin)
	if !scanner.Scan() || strings.ToLower(strings.TrimSpace(scanner.Text())) != "y" {
		fmt.Println("❌ 重建操作已取消")
		return
	}

	fmt.Println("删除现有的Milvus集合...")
	if err := s.indexModule.DeleteCollection(ctx); err != nil {
		fmt.Printf("删除集合时出现问题: %v，继续重建...\n", err)
	} else {
		fmt.Println("✅ 现有集合已删除")
	}

	// 重新构建知识库
	fmt.Println("开始重建知识库...")
	if err := s.BuildKnowledgeBase(ctx); err != nil {
		fmt.Printf("❌ 重建失败: %v\n", err)
		fmt.Println("建议：请检查Milvus服务状态后重试")
		return
	}

	fmt.Println("✅ 知识库重建完成！")
}

// Cleanup 清理资源
func (s *AdvancedGraphRAGSystem) Cleanup(ctx context.Context) {
	if s.dataModule != nil {
		s.dataModule.Close()
	}
	if s.traditionalRetrieval != nil {
		s.traditionalRetrieval.Close(ctx)
	}
	if s.graphRAGRetrieval != nil {
		s.graphRAGRetrieval.Close(ctx)
	}
	if s.indexModule != nil {
		s.indexModule.Close(ctx)
	}
}

func main() {
	ctx := context.Background()

	fmt.Println("启动高级图RAG系统...")

	// 从环境变量加载配置
	config := DefaultConfig
	if uri := os.Getenv("NEO4J_URI"); uri != "" {
		config.Neo4jURI = uri
	}
	if user := os.Getenv("NEO4J_USER"); user != "" {
		config.Neo4jUser = user
	}
	if password := os.Getenv("NEO4J_PASSWORD"); password != "" {
		config.Neo4jPassword = password
	}
	if host := os.Getenv("MILVUS_HOST"); host != "" {
		config.MilvusHost = host
	}
	if model := os.Getenv("LLM_MODEL"); model != "" {
		config.LLMModel = model
	}

	// 创建高级图RAG系统
	ragSystem := NewAdvancedGraphRAGSystem(config)
	defer ragSystem.Cleanup(ctx)

	// 初始化系统
	if err := ragSystem.InitializeSystem(ctx); err != nil {
		log.Fatalf("初始化失败: %v", err)
	}

	// 构建知识库
	if err := ragSystem.BuildKnowledgeBase(ctx); err != nil {
		log.Fatalf("构建知识库失败: %v", err)
	}

	// 运行交互式问答
	ragSystem.RunInteractive(ctx)
}
