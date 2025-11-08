package batch_0001

import (
	"context"
	"fmt"
	"log"
	"os"
	"strings"
	"time"

	"github.com/cloudwego/eino-ext/components/model/ark"
	"github.com/cloudwego/eino/schema"
	arkModel "github.com/volcengine/volcengine-go-sdk/service/arkruntime/model"
)

// GenerationConfig 生成配置结构
type GenerationConfig struct {
	ModelName   string  `json:"model_name"`  // 使用的大语言模型名称
	Temperature float32 `json:"temperature"` // 生成温度，控制输出的随机性（0.0-1.0）
	MaxTokens   int     `json:"max_tokens"`  // 最大生成token数，控制答案长度
	APIKey      string  `json:"api_key"`     // API密钥
	BaseURL     string  `json:"base_url"`    // API基础URL
}

// GenerationIntegrationModule 生成集成模块 - RAG系统的答案生成引擎
//
// 负责将检索到的相关文档转换为最终的自然语言答案。
// 采用先进的提示工程技术，确保生成的答案准确、实用、符合用户期望。
//
// 核心功能：
// 1. 智能上下文构建：从检索文档中提取和组织关键信息
// 2. 自适应答案生成：根据查询类型自动调整回答风格
// 3. 流式输出支持：实时生成答案，提升交互体验
// 4. 错误处理机制：网络故障时的重试和降级策略
//
// 技术特点：
// - LightRAG风格：统一处理不同类型查询，无需复杂分类
// - 多模式支持：同步生成、流式生成、批量生成
// - 智能提示：根据检索层级优化提示词结构
// - 服务稳定：完善的错误处理和重试机制
type GenerationIntegrationModule struct {
	config    *GenerationConfig
	chatModel *ark.ChatModel
}

func NewGenerationIntegrationModule(modelName string, apiKey string, temperature float32, maxTokens int) *GenerationIntegrationModule {
	if modelName == "" {
		modelName = os.Getenv("ARK_MODEL_ID")
		if modelName == "" {
			modelName = "ep-20241227100717-lhcjj" // 默认模型
		}
	}
	if temperature == 0 {
		temperature = 0.1
	}
	if maxTokens == 0 {
		maxTokens = 2048
	}

	config := &GenerationConfig{
		ModelName:   modelName,
		Temperature: temperature,
		MaxTokens:   maxTokens,
		APIKey:      apiKey,
	}

	log.Printf("生成模块初始化完成，模型: %s", modelName)

	return &GenerationIntegrationModule{
		config: config,
	}
}

// Initialize 初始化生成模块
func (g *GenerationIntegrationModule) Initialize(ctx context.Context) error {
	// 初始化Ark ChatModel
	chatModel, err := ark.NewChatModel(ctx, &ark.ChatModelConfig{
		APIKey: g.config.APIKey,
		Model:  g.config.ModelName,
	})
	if err != nil {
		return fmt.Errorf("初始化Ark ChatModel失败: %w", err)
	}

	g.chatModel = chatModel
	log.Printf("Ark ChatModel初始化完成")
	return nil
}

func (g *GenerationIntegrationModule) GenerateAdaptiveAnswer(ctx context.Context, question string, documents []*schema.Document) (string, error) {
	// 确保模型已初始化
	if g.chatModel == nil {
		if err := g.Initialize(ctx); err != nil {
			return "", fmt.Errorf("初始化生成模块失败: %w", err)
		}
	}

	// 构建上下文 - 整合所有检索到的文档
	var contextParts []string

	for _, doc := range documents {
		content := doc.Content
		if content != "" {
			// 添加检索层级信息（如果有的话）
			// 这有助于LLM理解信息的重要性和可靠性
			if level, exists := doc.MetaData["retrieval_level"]; exists {
				if levelStr, ok := level.(string); ok {
					// 为不同检索层级添加标识，帮助LLM理解信息层次
					contextParts = append(contextParts, fmt.Sprintf("[%s] %s", strings.ToUpper(levelStr), content))
				} else {
					contextParts = append(contextParts, content)
				}
			} else {
				contextParts = append(contextParts, content)
			}
		}
	}

	// 将所有文档内容合并为统一的上下文
	context := strings.Join(contextParts, "\n\n")

	// 优化的烹饪助手提示词 - 专门处理菜谱信息的理解和生成
	prompt := fmt.Sprintf(`你是一位专业的烹饪助手，请基于检索到的信息为用户提供实用的烹饪指导。

检索到的相关信息：
%s

用户问题：%s

重要指导原则：
1. **菜谱识别准确性**：
   - 仔细识别每个菜谱的正确名称，不要混淆不同菜品
   - 如果信息来自多个不同菜谱，请明确区分并分别介绍

2. **推荐数量要求**：
   - 如果用户要求推荐菜品，至少推荐3个不同的菜品
   - 每个菜品都要提供完整的制作步骤和营养特点

3. **烹饪步骤智能补全**：
   - 根据检索到的信息，结合常见烹饪知识，提供完整的制作步骤
   - 如果检索到的步骤不连续，请基于食材和烹饪方法智能推理缺失步骤
   - 不要标注推理信息，直接提供完整的制作步骤，让回答看起来更自然

4. **信息准确性与实用性平衡**：
   - 优先使用检索到的准确信息
   - 当信息不完整时，基于烹饪常识合理补充，让用户能实际操作
   - 所有步骤都应该看起来是专业的烹饪指导

5. **回答格式**：
   - 减肥餐推荐：提供至少3个菜品名称、营养特点、完整制作步骤
   - 制作方法：按步骤顺序清晰列出，标明步骤编号
   - 确保每个菜谱都有可操作的完整步骤

请根据以上原则提供准确、实用的回答：`, context, question)

	// 构建消息
	messages := []*schema.Message{
		{
			Role:    schema.System,
			Content: "你是一位专业的烹饪助手，能够基于提供的信息为用户提供准确、实用的回答。",
		},
		{
			Role:    schema.User,
			Content: prompt,
		},
	}

	// 配置生成选项
	thinking := &arkModel.Thinking{
		Type: arkModel.ThinkingTypeDisabled,
	}

	// 调用ChatModel生成答案
	response, err := g.chatModel.Generate(ctx, messages, ark.WithThinking(thinking))
	if err != nil {
		log.Printf("LightRAG答案生成失败: %v", err)
		return fmt.Sprintf("抱歉，生成回答时出现错误：%v", err), err
	}

	if response == nil || response.Content == "" {
		return "抱歉，未能生成有效回答", fmt.Errorf("API返回空响应")
	}

	// 提取并返回生成的答案
	return strings.TrimSpace(response.Content), nil
}

func (g *GenerationIntegrationModule) GenerateAdaptiveAnswerStream(ctx context.Context, question string, documents []*schema.Document, maxRetries int, resultChan chan<- string) {
	defer close(resultChan)

	// 确保模型已初始化
	if g.chatModel == nil {
		if err := g.Initialize(ctx); err != nil {
			resultChan <- fmt.Sprintf("初始化生成模块失败: %v", err)
			return
		}
	}

	if maxRetries == 0 {
		maxRetries = 3
	}

	// 构建上下文 - 与同步版本相同的逻辑
	var contextParts []string

	for _, doc := range documents {
		content := doc.Content
		if content != "" {
			if level, exists := doc.MetaData["retrieval_level"]; exists {
				if levelStr, ok := level.(string); ok {
					contextParts = append(contextParts, fmt.Sprintf("[%s] %s", strings.ToUpper(levelStr), content))
				} else {
					contextParts = append(contextParts, content)
				}
			} else {
				contextParts = append(contextParts, content)
			}
		}
	}

	context := strings.Join(contextParts, "\n\n")

	// 优化的烹饪助手提示词 - 与同步版本保持一致
	prompt := fmt.Sprintf(`你是一位专业的烹饪助手，请基于检索到的信息为用户提供实用的烹饪指导。

检索到的相关信息：
%s

用户问题：%s

重要指导原则：
1. **菜谱识别准确性**：
   - 仔细识别每个菜谱的正确名称，不要混淆不同菜品
   - 如果信息来自多个不同菜谱，请明确区分并分别介绍

2. **推荐数量要求**：
   - 如果用户要求推荐菜品，至少推荐3个不同的菜品
   - 每个菜品都要提供完整的制作步骤和营养特点

3. **烹饪步骤智能补全**：
   - 根据检索到的信息，结合常见烹饪知识，提供完整的制作步骤
   - 如果检索到的步骤不连续，请基于食材和烹饪方法智能推理缺失步骤
   - 不要标注推理信息，直接提供完整的制作步骤，让回答看起来更自然

4. **信息准确性与实用性平衡**：
   - 优先使用检索到的准确信息
   - 当信息不完整时，基于烹饪常识合理补充，让用户能实际操作
   - 所有步骤都应该看起来是专业的烹饪指导

5. **回答格式**：
   - 减肥餐推荐：提供至少3个菜品名称、营养特点、完整制作步骤
   - 制作方法：按步骤顺序清晰列出，标明步骤编号
   - 确保每个菜谱都有可操作的完整步骤

请根据以上原则提供准确、实用的回答：`, context, question)

	// 构建消息
	messages := []*schema.Message{
		{
			Role:    schema.System,
			Content: "你是一位专业的烹饪助手，能够基于提供的信息为用户提供准确、实用的回答。",
		},
		{
			Role:    schema.User,
			Content: prompt,
		},
	}

	// 配置生成选项
	thinking := &arkModel.Thinking{
		Type: arkModel.ThinkingTypeDisabled,
	}

	// 重试循环 - 最多尝试maxRetries次
	for attempt := 0; attempt < maxRetries; attempt++ {
		// 显示重试信息（如果不是第一次尝试）
		if attempt == 0 {
			resultChan <- "开始流式生成回答...\n\n"
		} else {
			resultChan <- fmt.Sprintf("第%d次尝试流式生成...\n\n", attempt+1)
		}

		// 调用流式生成
		streamReader, err := g.chatModel.Stream(ctx, messages, ark.WithThinking(thinking))
		if err != nil {
			log.Printf("流式生成第%d次尝试失败: %v", attempt+1, err)

			if attempt < maxRetries-1 {
				// 还有重试机会，等待后继续
				waitTime := time.Duration(attempt+1) * 2 * time.Second // 递增等待时间：2s, 4s, 6s...
				resultChan <- fmt.Sprintf("⚠️ 连接中断，%v后重试...\n", waitTime)
				time.Sleep(waitTime)
				continue
			} else {
				// 所有重试都失败，使用非流式作为后备方案
				log.Printf("流式生成完全失败，尝试非流式后备方案")
				resultChan <- "⚠️ 流式生成失败，切换到标准模式...\n"

				// 降级到同步生成
				fallbackResponse, fallbackErr := g.GenerateAdaptiveAnswer(ctx, question, documents)
				if fallbackErr != nil {
					log.Printf("后备生成也失败: %v", fallbackErr)
					resultChan <- fmt.Sprintf("抱歉，生成回答时出现网络错误，请稍后重试。错误信息：%v", err)
				} else {
					resultChan <- fallbackResponse
				}
				return
			}
		}

		// 处理流式响应
		for {
			message, streamErr := streamReader.Recv()
			if streamErr != nil {
				if streamErr.Error() == "EOF" {
					// 正常结束流式生成
					return
				}
				log.Printf("读取流式响应失败: %v", streamErr)
				break // 跳出内层循环，进入重试逻辑
			}

			if message != nil && message.Content != "" {
				resultChan <- message.Content
			}
		}

		// 如果到这里说明流式读取出现了错误，继续重试逻辑
		if attempt < maxRetries-1 {
			waitTime := time.Duration(attempt+1) * 2 * time.Second
			resultChan <- fmt.Sprintf("⚠️ 流式读取中断，%v后重试...\n", waitTime)
			time.Sleep(waitTime)
			continue
		} else {
			// 最后一次重试也失败，使用后备方案
			log.Printf("流式生成完全失败，尝试非流式后备方案")
			resultChan <- "⚠️ 流式生成失败，切换到标准模式...\n"

			fallbackResponse, fallbackErr := g.GenerateAdaptiveAnswer(ctx, question, documents)
			if fallbackErr != nil {
				log.Printf("后备生成也失败: %v", fallbackErr)
				resultChan <- fmt.Sprintf("抱歉，生成回答时出现网络错误，请稍后重试。")
			} else {
				resultChan <- fallbackResponse
			}
			return
		}
	}
}
