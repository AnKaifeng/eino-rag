package batch_0001

import (
	"context"
	"fmt"
	"log"
	"strings"

	"github.com/cloudwego/eino/schema"
)

// SearchStrategy 搜索策略枚举
type SearchStrategy string

const (
	// HybridTraditional 混合传统检索策略
	HybridTraditional SearchStrategy = "hybrid_traditional"
	// GraphRAG 图RAG检索策略
	GraphRAG SearchStrategy = "graph_rag"
	// Combined 组合检索策略
	Combined SearchStrategy = "combined"
)

// QueryAnalysis 查询分析结果
type QueryAnalysis struct {
	QueryComplexity       float64        `json:"query_complexity"`       // 查询复杂度 (0-1)，表示查询的复杂程度
	RelationshipIntensity float64        `json:"relationship_intensity"` // 关系密集度 (0-1)，表示查询涉及实体间关系的密集程度
	ReasoningRequired     bool           `json:"reasoning_required"`     // 是否需要推理，表示查询是否需要多跳推理或因果分析
	EntityCount           int            `json:"entity_count"`           // 实体数量，查询中识别出的实体个数
	RecommendedStrategy   SearchStrategy `json:"recommended_strategy"`   // 推荐的检索策略
	Confidence            float64        `json:"confidence"`             // 推荐置信度 (0-1)，表示对推荐策略的信心程度
	Reasoning             string         `json:"reasoning"`              // 推荐理由，解释为什么选择该策略的原因
}

// RouteStatistics 路由统计信息
type RouteStatistics struct {
	TraditionalCount int     `json:"traditional_count"` // 传统检索使用次数
	GraphRAGCount    int     `json:"graph_rag_count"`   // 图RAG检索使用次数
	CombinedCount    int     `json:"combined_count"`    // 组合检索使用次数
	TotalQueries     int     `json:"total_queries"`     // 总查询次数
	TraditionalRatio float64 `json:"traditional_ratio"` // 传统检索使用比例
	GraphRAGRatio    float64 `json:"graph_rag_ratio"`   // 图RAG检索使用比例
	CombinedRatio    float64 `json:"combined_ratio"`    // 组合检索使用比例
}

// LLMAnalysisResult LLM查询分析结果
type LLMAnalysisResult struct {
	QueryComplexity       float64 `json:"query_complexity"`
	RelationshipIntensity float64 `json:"relationship_intensity"`
	ReasoningRequired     bool    `json:"reasoning_required"`
	EntityCount           int     `json:"entity_count"`
	RecommendedStrategy   string  `json:"recommended_strategy"`
	Confidence            float64 `json:"confidence"`
	Reasoning             string  `json:"reasoning"`
}

// IntelligentQueryRouter 智能查询路由器
//
// 根据查询特征智能选择最适合的检索策略，提升RAG系统的整体性能。
// 通过LLM分析查询复杂度、关系密集度等特征，动态路由到不同的检索方法。
//
// 核心功能：
// 1. 查询特征分析：使用LLM深度分析查询的复杂度和关系特征
// 2. 智能策略选择：根据分析结果选择最优的检索策略
// 3. 组合检索支持：支持多种检索方法的组合使用
// 4. 统计与监控：提供详细的路由统计信息
//
// 路由策略：
// - hybrid_traditional: 适合简单直接的信息查找
// - graph_rag: 适合复杂关系推理和知识发现
// - combined: 需要两种策略结合
type IntelligentQueryRouter struct {
	traditionalRetrieval *HybridRetrievalModule // 传统混合检索模块
	graphRAGRetrieval    *GraphRAGRetrieval     // 图RAG检索模块
	llmClient            interface{}            // 大语言模型客户端
	config               *Config                // 系统配置

	routeStats *RouteStatistics // 路由统计信息
}

// NewIntelligentQueryRouter 创建新的智能查询路由器
//
// Args:
//
//	traditionalRetrieval: 传统混合检索模块实例
//	graphRAGRetrieval: 图RAG检索模块实例
//	llmClient: 大语言模型客户端
//	config: 系统配置
func NewIntelligentQueryRouter(traditionalRetrieval *HybridRetrievalModule, graphRAGRetrieval *GraphRAGRetrieval, llmClient interface{}, config *Config) *IntelligentQueryRouter {
	return &IntelligentQueryRouter{
		traditionalRetrieval: traditionalRetrieval,
		graphRAGRetrieval:    graphRAGRetrieval,
		llmClient:            llmClient,
		config:               config,
		routeStats: &RouteStatistics{
			TraditionalCount: 0,
			GraphRAGCount:    0,
			CombinedCount:    0,
			TotalQueries:     0,
		},
	}
}

// AnalyzeQuery 分析查询特征
//
// 使用LLM深度分析查询的各种特征，为路由决策提供数据支持。
//
// Args:
//
//	ctx: 上下文对象
//	query: 用户查询字符串
//
// Returns:
//
//	*QueryAnalysis: 查询分析结果
//	error: 可能的错误
func (r *IntelligentQueryRouter) AnalyzeQuery(ctx context.Context, query string) (*QueryAnalysis, error) {
	log.Printf("分析查询特征: %s", query)

	// 由于llmClient是interface{}类型，这里需要进行类型断言或者调用相应的方法
	// 为了简化，这里先提供一个基础实现，实际使用时需要根据具体的LLM客户端接口调整

	// 可以在这里添加LLM调用的逻辑，例如：
	// analysisPrompt := fmt.Sprintf(`作为RAG系统的查询分析专家...`, query)
	// 然后调用LLM客户端进行分析

	// 目前降级到基于规则的分析
	return r.ruleBasedAnalysis(query), nil
}

// ruleBasedAnalysis 基于规则的查询分析（降级方案）
func (r *IntelligentQueryRouter) ruleBasedAnalysis(query string) *QueryAnalysis {
	// 复杂度关键词
	complexityKeywords := []string{"为什么", "如何", "关系", "影响", "原因", "比较", "区别", "分析", "推理"}
	// 关系关键词
	relationKeywords := []string{"配", "搭配", "组合", "相关", "联系", "连接", "适合", "匹配"}

	// 计算复杂度得分
	complexityCount := 0
	for _, keyword := range complexityKeywords {
		if strings.Contains(query, keyword) {
			complexityCount++
		}
	}
	complexity := float64(complexityCount) / float64(len(complexityKeywords))

	// 计算关系密集度得分
	relationCount := 0
	for _, keyword := range relationKeywords {
		if strings.Contains(query, keyword) {
			relationCount++
		}
	}
	relationIntensity := float64(relationCount) / float64(len(relationKeywords))

	// 实体数量估算（简单按空格分词计算）
	words := strings.Fields(query)
	entityCount := len(words)

	// 推理需求判断
	reasoningRequired := complexity > 0.3 || relationIntensity > 0.3

	// 策略推荐
	var strategy SearchStrategy
	var confidence float64
	var reasoning string

	if complexity > 0.5 || relationIntensity > 0.5 {
		strategy = GraphRAG
		confidence = 0.8
		reasoning = "查询涉及复杂关系或推理，适合使用图RAG检索"
	} else if complexity > 0.3 || relationIntensity > 0.3 {
		strategy = Combined
		confidence = 0.7
		reasoning = "查询具有中等复杂度，建议组合使用多种检索策略"
	} else {
		strategy = HybridTraditional
		confidence = 0.6
		reasoning = "查询相对简单，使用传统混合检索即可满足需求"
	}

	return &QueryAnalysis{
		QueryComplexity:       complexity,
		RelationshipIntensity: relationIntensity,
		ReasoningRequired:     reasoningRequired,
		EntityCount:           entityCount,
		RecommendedStrategy:   strategy,
		Confidence:            confidence,
		Reasoning:             reasoning,
	}
}

// RouteQuery 智能路由查询
//
// 根据查询分析结果，选择最适合的检索策略并执行检索。
//
// Args:
//
//	ctx: 上下文对象
//	query: 用户查询字符串
//	topK: 返回结果数量
//
// Returns:
//
//	[]*schema.Document: 检索到的文档列表
//	*QueryAnalysis: 查询分析结果
//	error: 可能的错误
func (r *IntelligentQueryRouter) RouteQuery(ctx context.Context, query string, topK int) ([]*schema.Document, *QueryAnalysis, error) {
	log.Printf("开始智能路由: %s", query)

	// 分析查询特征
	analysis, err := r.AnalyzeQuery(ctx, query)
	if err != nil {
		log.Printf("查询分析失败: %v", err)
		// 使用默认分析结果
		analysis = r.ruleBasedAnalysis(query)
	}

	// 更新路由统计
	r.updateRouteStats(analysis.RecommendedStrategy)

	var documents []*schema.Document

	// 根据推荐策略执行检索
	switch analysis.RecommendedStrategy {
	case HybridTraditional:
		log.Println("使用传统混合检索")
		// documents, err = r.executeTraditionalRetrieval(ctx, query, topK)
		documents, err = r.traditionalRetrieval.HybridSearch(ctx, query, topK)

	case GraphRAG:
		log.Println("🕸️ 使用图RAG检索")
		// documents, err = r.executeGraphRAGRetrieval(ctx, query, topK)
		documents, err = r.graphRAGRetrieval.GraphRAGSearch(ctx, query, topK)

	case Combined:
		log.Println("🔄 使用组合检索策略")
		documents, err = r.executeCombinedSearch(ctx, query, topK)

	default:
		log.Printf("未知策略: %s，使用传统检索", analysis.RecommendedStrategy)
		// documents, err = r.executeTraditionalRetrieval(ctx, query, topK)
		documents, err = r.traditionalRetrieval.HybridSearch(ctx, query, topK)
	}

	if err != nil {
		log.Printf("查询路由失败: %v", err)
		// 降级到传统检索
		documents, _ = r.traditionalRetrieval.HybridSearch(ctx, query, topK)
	}

	// 后处理结果
	documents = r.postProcessResults(documents, analysis)

	log.Printf("路由完成，返回 %d 个结果", len(documents))
	return documents, analysis, nil
}

// executeCombinedSearch 执行组合检索
func (r *IntelligentQueryRouter) executeCombinedSearch(ctx context.Context, query string, topK int) ([]*schema.Document, error) {
	log.Printf("执行组合检索: %s", query)

	// 分配检索数量
	traditionalK := topK / 2
	if traditionalK < 1 {
		traditionalK = 1
	}
	graphK := topK - traditionalK

	// 并行执行两种检索
	// traditionalDocs, err1 := r.executeTraditionalRetrieval(ctx, query, traditionalK)
	traditionalDocs, err1 := r.traditionalRetrieval.HybridSearch(ctx, query, traditionalK)
	if err1 != nil {
		log.Printf("传统检索失败: %v", err1)
		traditionalDocs = []*schema.Document{}
	}

	// graphDocs, err2 := r.executeGraphRAGRetrieval(ctx, query, graphK)
	graphDocs, err2 := r.graphRAGRetrieval.GraphRAGSearch(ctx, query, graphK)
	if err2 != nil {
		log.Printf("图RAG检索失败: %v", err2)
		graphDocs = []*schema.Document{}
	}

	// 合并结果，避免重复
	var combinedDocs []*schema.Document
	seenContents := make(map[string]bool)

	maxLen := len(traditionalDocs)
	if len(graphDocs) > maxLen {
		maxLen = len(graphDocs)
	}

	// Round-robin轮询合并
	for i := 0; i < maxLen; i++ {
		// 优先添加图RAG结果（通常质量更高）
		if i < len(graphDocs) {
			doc := graphDocs[i]
			contentHash := hashString(doc.Content[:min(100, len(doc.Content))])
			if !seenContents[contentHash] {
				seenContents[contentHash] = true
				if doc.MetaData == nil {
					doc.MetaData = make(map[string]interface{})
				}
				doc.MetaData["search_source"] = "graph_rag"
				combinedDocs = append(combinedDocs, doc)
			}
		}

		// 再添加传统检索结果
		if i < len(traditionalDocs) {
			doc := traditionalDocs[i]
			contentHash := hashString(doc.Content[:min(100, len(doc.Content))])
			if !seenContents[contentHash] {
				seenContents[contentHash] = true
				if doc.MetaData == nil {
					doc.MetaData = make(map[string]interface{})
				}
				doc.MetaData["search_source"] = "traditional"
				combinedDocs = append(combinedDocs, doc)
			}
		}
	}

	// 限制结果数量
	if len(combinedDocs) > topK {
		combinedDocs = combinedDocs[:topK]
	}

	return combinedDocs, nil
}

// postProcessResults 后处理结果
func (r *IntelligentQueryRouter) postProcessResults(documents []*schema.Document, analysis *QueryAnalysis) []*schema.Document {
	for _, doc := range documents {
		if doc.MetaData == nil {
			doc.MetaData = make(map[string]interface{})
		}

		// 添加路由信息到元数据
		doc.MetaData["route_strategy"] = string(analysis.RecommendedStrategy)
		doc.MetaData["query_complexity"] = analysis.QueryComplexity
		doc.MetaData["route_confidence"] = analysis.Confidence
	}

	return documents
}

// updateRouteStats 更新路由统计信息
func (r *IntelligentQueryRouter) updateRouteStats(strategy SearchStrategy) {
	r.routeStats.TotalQueries++

	switch strategy {
	case HybridTraditional:
		r.routeStats.TraditionalCount++
	case GraphRAG:
		r.routeStats.GraphRAGCount++
	case Combined:
		r.routeStats.CombinedCount++
	}

	// 更新比例
	total := float64(r.routeStats.TotalQueries)
	if total > 0 {
		r.routeStats.TraditionalRatio = float64(r.routeStats.TraditionalCount) / total
		r.routeStats.GraphRAGRatio = float64(r.routeStats.GraphRAGCount) / total
		r.routeStats.CombinedRatio = float64(r.routeStats.CombinedCount) / total
	}
}

// GetRouteStatistics 获取路由统计信息
func (r *IntelligentQueryRouter) GetRouteStatistics() *RouteStatistics {
	return &RouteStatistics{
		TraditionalCount: r.routeStats.TraditionalCount,
		GraphRAGCount:    r.routeStats.GraphRAGCount,
		CombinedCount:    r.routeStats.CombinedCount,
		TotalQueries:     r.routeStats.TotalQueries,
		TraditionalRatio: r.routeStats.TraditionalRatio,
		GraphRAGRatio:    r.routeStats.GraphRAGRatio,
		CombinedRatio:    r.routeStats.CombinedRatio,
	}
}

// ExplainRoutingDecision 解释路由决策
//
// 为用户或开发者提供详细的路由决策解释，帮助理解系统的选择逻辑。
//
// Args:
//
//	ctx: 上下文对象
//	query: 用户查询字符串
//
// Returns:
//
//	string: 详细的路由决策解释
func (r *IntelligentQueryRouter) ExplainRoutingDecision(ctx context.Context, query string) string {
	analysis, _ := r.AnalyzeQuery(ctx, query)

	// 复杂度描述
	var complexityDesc string
	if analysis.QueryComplexity < 0.4 {
		complexityDesc = "简单"
	} else if analysis.QueryComplexity < 0.8 {
		complexityDesc = "中等"
	} else {
		complexityDesc = "复杂"
	}

	// 关系密集度描述
	var relationDesc string
	if analysis.RelationshipIntensity < 0.4 {
		relationDesc = "单一实体"
	} else if analysis.RelationshipIntensity < 0.8 {
		relationDesc = "实体关系"
	} else {
		relationDesc = "复杂关系网络"
	}

	// 推理需求描述
	reasoningDesc := "否"
	if analysis.ReasoningRequired {
		reasoningDesc = "是"
	}

	explanation := fmt.Sprintf(`查询路由分析报告

查询：%s

特征分析：
- 复杂度：%.2f (%s)
- 关系密集度：%.2f (%s)
- 推理需求：%s
- 实体数量：%d

推荐策略：%s
置信度：%.2f

决策理由：%s`,
		query,
		analysis.QueryComplexity, complexityDesc,
		analysis.RelationshipIntensity, relationDesc,
		reasoningDesc,
		analysis.EntityCount,
		analysis.RecommendedStrategy,
		analysis.Confidence,
		analysis.Reasoning)

	return explanation
}

// 辅助函数

// min 返回两个整数中的较小值
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// hashString 简单的字符串哈希函数
func hashString(s string) string {
	// 简单实现，实际项目中可以使用更好的哈希算法
	return fmt.Sprintf("%x", len(s)^int(s[0]))
}
