# Go RAG System - Python to Go 转换项目

这个项目将 Python 版本的 RAG (Retrieval-Augmented Generation) 系统成功转换为 Go 语言实现，保持了原有的核心功能和架构设计。

## 📋 项目概述

本项目包含了一个完整的图RAG系统，结合了知识图谱检索、混合检索、智能路由和答案生成等核心模块：

### 🏗️ 系统架构

```
用户查询 → 智能路由器 → 检索策略选择 → 执行检索 → 答案生成 → 返回结果
          ↓
    ┌─────────────────────────────────────────┐
    │  IntelligentQueryRouter (智能路由器)      │
    │  - 查询复杂度分析                        │
    │  - 策略选择决策                          │
    └─────────────────────────────────────────┘
                        ↓
    ┌─────────────────┐ ┌─────────────────────┐
    │ GraphRAGRetrieval│ │ HybridRetrieval     │
    │ (图RAG检索)      │ │ (混合检索)           │
    │ - 多跳推理        │ │ - 双层检索           │
    │ - 子图提取        │ │ - 向量检索           │
    │ - 图结构推理      │ │ - Round-robin合并    │
    └─────────────────┘ └─────────────────────┘
                        ↓
    ┌─────────────────────────────────────────┐
    │ GenerationIntegration (答案生成)         │
    │ - LightRAG风格统一生成                   │
    │ - 流式答案支持                          │
    │ - 错误重试机制                          │
    └─────────────────────────────────────────┘
```

## 📁 文件结构

```
batch-0001/
├── go.mod                          # Go模块定义
├── go.sum                          # 依赖校验
├── graph_reterval.go              # 图RAG检索模块 (原有)
├── graph_index.go                 # 图索引模块 (原有)
├── milvus_batch.go               # Milvus批处理 (原有)
├── neo4j_batch.go                # Neo4j批处理 (原有)
├── generation_integration.go      # 答案生成模块 (新增)
├── hybrid_retrieval.go           # 混合检索模块 (新增)
├── intelligent_query_router.go   # 智能路由器 (新增)
├── rag_system_integration.go     # 系统集成示例 (新增)
└── README.md                     # 项目说明文档
```

## 🚀 核心模块说明

### 1. GenerationIntegrationModule (答案生成模块)

**文件**: `generation_integration.go`

**功能特点**:
- 🤖 **LightRAG风格统一生成**: 自适应不同查询类型，无需复杂分类
- 🌊 **流式答案支持**: 实时生成答案，提升用户体验
- 🔄 **错误重试机制**: 网络中断时自动重试，确保服务稳定
- 🔗 **OpenAI兼容API**: 支持Moonshot Kimi等多种LLM服务

**主要方法**:
```go
// 同步答案生成
func (g *GenerationIntegrationModule) GenerateAdaptiveAnswer(ctx context.Context, question string, documents []*schema.Document) (string, error)

// 流式答案生成
func (g *GenerationIntegrationModule) GenerateAdaptiveAnswerStream(ctx context.Context, question string, documents []*schema.Document, maxRetries int, resultChan chan<- string)
```

### 2. HybridRetrievalModule (混合检索模块)

**文件**: `hybrid_retrieval.go`

**功能特点**:
- 🔍 **双层检索范式**: 实体级检索 + 主题级检索
- 🕸️ **图向量融合**: 图数据库结构化检索 + 向量数据库语义检索
- 🔄 **Round-robin合并**: 公平轮询策略，避免单一检索方法偏差
- 🧠 **智能关键词提取**: LLM驱动的查询分析和关键词分层提取

**主要方法**:
```go
// 双层检索
func (h *HybridRetrievalModule) DualLevelRetrieval(ctx context.Context, query string, topK int) ([]*schema.Document, error)

// 混合检索
func (h *HybridRetrievalModule) HybridSearch(ctx context.Context, query string, topK int) ([]*schema.Document, error)

// 增强向量检索
func (h *HybridRetrievalModule) VectorSearchEnhanced(ctx context.Context, query string, topK int) ([]*schema.Document, error)
```

### 3. IntelligentQueryRouter (智能查询路由器)

**文件**: `intelligent_query_router.go`

**功能特点**:
- 🧠 **查询特征分析**: 使用LLM深度分析查询复杂度和关系特征
- 🎯 **智能策略选择**: 根据分析结果选择最优检索策略
- 🔄 **组合检索支持**: 支持多种检索方法的组合使用
- 📊 **统计与监控**: 提供详细的路由统计信息

**检索策略**:
- `hybrid_traditional`: 适合简单直接的信息查找
- `graph_rag`: 适合复杂关系推理和知识发现  
- `combined`: 需要两种策略结合

**主要方法**:
```go
// 查询分析
func (r *IntelligentQueryRouter) AnalyzeQuery(ctx context.Context, query string) (*QueryAnalysis, error)

// 智能路由
func (r *IntelligentQueryRouter) RouteQuery(ctx context.Context, query string, topK int) ([]*schema.Document, *QueryAnalysis, error)

// 路由决策解释
func (r *IntelligentQueryRouter) ExplainRoutingDecision(ctx context.Context, query string) string
```

### 4. IntegratedRAGSystem (集成RAG系统)

**文件**: `rag_system_integration.go`

**功能特点**:
- 🎯 **统一查询接口**: 整合所有模块的完整RAG能力
- 🔄 **自动模块编排**: 智能选择和组合不同检索策略
- 🌊 **流式和同步支持**: 支持实时流式答案生成
- 📊 **监控与统计**: 提供系统运行状态监控

## 💻 使用示例

### 基础使用

```go
package main

import (
    "context"
    "fmt"
    "log"
)

func main() {
    // 1. 创建系统配置
    config := &RAGSystemConfig{
        Neo4jURI:       "bolt://localhost:7687",
        Neo4jUser:      "neo4j", 
        Neo4jPassword:  "password",
        LLMModel:       "kimi-k2-0711-preview",
        MoonshotAPIKey: "your-moonshot-api-key",
        DefaultTopK:    5,
        Temperature:    0.1,
        MaxTokens:      2048,
    }

    // 2. 创建并初始化RAG系统
    ctx := context.Background()
    ragSystem := NewIntegratedRAGSystem(config)
    
    if err := ragSystem.Initialize(ctx); err != nil {
        log.Fatalf("初始化失败: %v", err)
    }
    defer ragSystem.Close(ctx)

    // 3. 执行查询
    question := "红烧肉怎么做？"
    response, err := ragSystem.Query(ctx, question)
    if err != nil {
        log.Fatalf("查询失败: %v", err)
    }

    // 4. 输出结果
    fmt.Printf("问题: %s\n", response.Question)
    fmt.Printf("答案: %s\n", response.Answer)
    fmt.Printf("使用策略: %s\n", response.RouteStrategy)
    fmt.Printf("置信度: %.2f\n", response.Confidence)
}
```

### 流式查询示例

```go
// 流式查询
resultChan := make(chan string, 100)
go func() {
    defer close(resultChan)
    _, err := ragSystem.QueryStream(ctx, "川菜有什么特色？", resultChan)
    if err != nil {
        log.Printf("流式查询失败: %v", err)
    }
}()

// 实时接收答案片段
fmt.Println("流式答案:")
for chunk := range resultChan {
    fmt.Print(chunk)
}
```

### 直接图RAG查询

```go
// 绕过路由，直接使用图RAG检索
response, err := ragSystem.DirectGraphRAGQuery(ctx, "鸡肉配什么蔬菜？", 5)
if err != nil {
    log.Printf("图RAG查询失败: %v", err)
} else {
    fmt.Printf("图RAG答案: %s\n", response.Answer)
}
```

## 🔧 环境配置

### 必需的环境变量

```bash
# Moonshot API密钥 (用于答案生成)
export MOONSHOT_API_KEY="your-moonshot-api-key"

# 可选：Ark API配置 (用于查询分析)
export ARK_API_KEY="your-ark-api-key"
```

### 依赖服务

1. **Neo4j数据库**: 用于存储知识图谱
   ```bash
   # Docker启动Neo4j
   docker run -d \
     --name neo4j \
     -p 7474:7474 -p 7687:7687 \
     -e NEO4J_AUTH=neo4j/password \
     neo4j:latest
   ```

2. **Milvus向量数据库** (可选): 用于向量检索
   ```bash
   # Docker启动Milvus
   docker run -d \
     --name milvus \
     -p 19530:19530 \
     milvusdb/milvus:latest
   ```

## 🚦 系统监控

### 获取系统统计

```go
// 获取路由统计信息
stats := ragSystem.GetSystemStatistics()
fmt.Printf("总查询数: %d\n", stats.TotalQueries)
fmt.Printf("图RAG使用率: %.2f%%\n", stats.GraphRAGUsage*100)
fmt.Printf("混合检索使用率: %.2f%%\n", stats.HybridUsage*100)
```

### 查询解释

```go
// 解释查询路由决策
explanation := ragSystem.ExplainQuery(ctx, "为什么川菜用花椒？")
fmt.Println(explanation)
```

## 🔧 自定义配置

### 检索参数调整

```go
config := &RAGSystemConfig{
    DefaultTopK:  10,        // 默认返回结果数
    Temperature:  0.2,       // LLM生成温度
    MaxTokens:    4096,      // 最大生成token数
}
```

### 模块单独使用

```go
// 单独使用图RAG检索
graphConfig := &Config{...}
graphRAG := NewGraphRAGRetrieval(graphConfig)
graphRAG.Initialize(ctx)

documents, err := graphRAG.GraphRAGSearch(ctx, "查询问题", 5)

// 单独使用答案生成
generator := NewGenerationIntegrationModule("kimi-k2-0711-preview", 0.1, 2048)
answer, err := generator.GenerateAdaptiveAnswer(ctx, "问题", documents)
```

## 🧪 测试与调试

### 查询分析测试

```go
// 测试查询路由决策
router := NewIntelligentQueryRouter(nil, nil, nil, config)
analysis, _ := router.AnalyzeQuery(ctx, "鸡肉配什么蔬菜？")

fmt.Printf("查询复杂度: %.2f\n", analysis.QueryComplexity)
fmt.Printf("关系密集度: %.2f\n", analysis.RelationshipIntensity)
fmt.Printf("推荐策略: %s\n", analysis.RecommendedStrategy)
```
