package batch_0001

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"strings"

	"github.com/cloudwego/eino/components/model"

	"github.com/cloudwego/eino/components/prompt"

	"github.com/cloudwego/eino-ext/components/model/ark"
	"github.com/cloudwego/eino/schema"
)

// EntityKeyValue 实体键值对数据结构
//
// 将图数据库中的实体节点转换为键值对表示，支持快速的实体检索。
// 每个实体使用其名称作为主要的索引键，值包含实体的完整描述信息。
//
// 设计特点：
// - 唯一索引键：使用实体名称确保检索的精确性
// - 丰富内容：包含实体的所有属性和描述信息
// - 类型识别：保留实体类型便于分类检索
// - 元数据保留：维持与原始图节点的关联关系

type EntityKeyValue struct {
	EntityName   string                 `json:"entity_name"`   // 实体名称，如"宫保鸡丁"、"鸡胸肉"
	IndexKeys    []string               `json:"index_keys"`    // 索引键列表，主要是实体名称
	ValueContent string                 `json:"value_content"` // 实体的详细描述内容
	EntityType   string                 `json:"entity_type"`   // 实体类型 (Recipe, Ingredient, CookingStep)
	Metadata     map[string]interface{} `json:"metadata"`      // 与原始图节点相关的元数据
}

// RelationKeyValue 关系键值对数据结构
//
// 将图数据库中的关系转换为键值对表示，支持主题级和概念级的检索。
// 关系可以有多个索引键，包含从LLM生成的全局主题概念。
//
// 设计特点：
// - 多索引键：支持关系类型、主题概念等多个检索入口
// - 全局主题：通过LLM增强生成抽象的主题概念键
// - 关系语义：保留完整的关系语义和上下文信息
// - 双向关联：连接源实体和目标实体的完整信息
type RelationKeyValue struct {
	RelationID   string                 `json:"relation_id"`   // 关系的唯一标识符
	IndexKeys    []string               `json:"index_keys"`    // 多个索引键（可包含全局主题）
	ValueContent string                 `json:"value_content"` // 关系的描述内容
	RelationType string                 `json:"relation_type"` // 关系类型，如REQUIRES、CONTAINS_STEP
	SourceEntity string                 `json:"source_entity"` // 源实体ID
	TargetEntity string                 `json:"target_entity"` // 目标实体ID
	Metadata     map[string]interface{} `json:"metadata"`      // 关系的元数据信息
}

// GraphEntity 图实体接口
type GraphEntity interface {
	GetNodeID() string
	GetName() string
	GetProperties() map[string]interface{}
}

// Recipe 菜谱实体
type Recipe struct {
	NodeID     string                 `json:"node_id"`
	Name       string                 `json:"name"`
	Properties map[string]interface{} `json:"properties"`
}

func (r *Recipe) GetNodeID() string                     { return r.NodeID }
func (r *Recipe) GetName() string                       { return r.Name }
func (r *Recipe) GetProperties() map[string]interface{} { return r.Properties }

// Ingredient 食材实体
type Ingredient struct {
	NodeID     string                 `json:"node_id"`
	Name       string                 `json:"name"`
	Properties map[string]interface{} `json:"properties"`
}

func (i *Ingredient) GetNodeID() string                     { return i.NodeID }
func (i *Ingredient) GetName() string                       { return i.Name }
func (i *Ingredient) GetProperties() map[string]interface{} { return i.Properties }

// CookingStep 烹饪步骤实体
type CookingStep struct {
	NodeID     string                 `json:"node_id"`
	Name       string                 `json:"name"`
	Properties map[string]interface{} `json:"properties"`
}

func (c *CookingStep) GetNodeID() string                     { return c.NodeID }
func (c *CookingStep) GetName() string                       { return c.Name }
func (c *CookingStep) GetProperties() map[string]interface{} { return c.Properties }

// Relationship 关系结构
type Relationship struct {
	SourceID     string `json:"source_id"`
	RelationType string `json:"relation_type"`
	TargetID     string `json:"target_id"`
}

// GraphIndexingModule 图索引模块 - 图数据的键值对索引系统
//
// 将复杂的图结构转换为高效的键值对索引，支持快速的实体检索和主题检索。
// 这是混合检索系统的重要组成部分，提供了图数据的快速访问能力。
//
// 核心功能：
// 1. 实体索引化：为所有实体创建基于名称的唯一索引
// 2. 关系索引化：为关系创建多维度的主题索引
// 3. 智能去重：识别和合并重复的实体和关系
// 4. LLM增强：可选的智能关系键生成
// 5. 高效检索：O(1)时间复杂度的键值检索
//
// 索引策略：
// - 实体策略：名称作为唯一键，确保精确匹配
// - 关系策略：多键策略，支持关系类型和主题概念检索
// - 去重策略：基于名称和关系签名的智能去重
// - 增强策略：LLM生成的语义丰富的主题键
//
// 技术实现：
// - 内存索引：高性能的内存键值存储
// - 双向映射：键到实体/关系的快速映射
// - 批量处理：高效的大规模图数据处理
// - 增量更新：支持动态的索引更新和维护
type GraphIndexingModule struct {
	config    *Config
	llmClient ark.ChatModel

	// 键值对存储 - 核心的索引数据结构
	entityKVStore   map[string]*EntityKeyValue   // 实体ID -> 实体键值对
	relationKVStore map[string]*RelationKeyValue // 关系ID -> 关系键值对

	// 索引映射：从检索键到实体/关系ID的快速映射
	keyToEntities  map[string][]string // 索引键 -> 实体ID列表
	keyToRelations map[string][]string // 索引键 -> 关系ID列表
}

// LLMKeywordsResponse LLM关键词生成响应
type LLMKeywordsResponse struct {
	Keywords []string `json:"keywords"`
}

// NewGraphIndexingModule 创建新的图索引模块
func NewGraphIndexingModule(config *Config, llmClient ark.ChatModel) *GraphIndexingModule {
	return &GraphIndexingModule{
		config:          config,
		llmClient:       llmClient,
		entityKVStore:   make(map[string]*EntityKeyValue),
		relationKVStore: make(map[string]*RelationKeyValue),
		keyToEntities:   make(map[string][]string),
		keyToRelations:  make(map[string][]string),
	}
}

// CreateEntityKeyValues 为实体创建键值对结构
// 每个实体使用其名称作为唯一索引键
func (g *GraphIndexingModule) CreateEntityKeyValues(recipes []*Recipe, ingredients []*Ingredient, cookingSteps []*CookingStep) map[string]*EntityKeyValue {
	log.Println("开始创建实体键值对...")

	// 处理菜谱实体
	for _, recipe := range recipes {
		entityID := recipe.GetNodeID()
		entityName := recipe.GetName()
		if entityName == "" {
			entityName = fmt.Sprintf("菜谱_%s", entityID)
		}

		// 构建详细内容
		contentParts := []string{fmt.Sprintf("菜品名称: %s", entityName)}

		props := recipe.GetProperties()
		if description, exists := props["description"]; exists {
			contentParts = append(contentParts, fmt.Sprintf("描述: %v", description))
		}
		if category, exists := props["category"]; exists {
			contentParts = append(contentParts, fmt.Sprintf("分类: %v", category))
		}
		if cuisineType, exists := props["cuisineType"]; exists {
			contentParts = append(contentParts, fmt.Sprintf("菜系: %v", cuisineType))
		}
		if difficulty, exists := props["difficulty"]; exists {
			contentParts = append(contentParts, fmt.Sprintf("难度: %v", difficulty))
		}
		if cookingTime, exists := props["cookingTime"]; exists {
			contentParts = append(contentParts, fmt.Sprintf("制作时间: %v", cookingTime))
		}

		// 创建键值对
		entityKV := &EntityKeyValue{
			EntityName:   entityName,
			IndexKeys:    []string{entityName}, // 使用名称作为唯一索引键
			ValueContent: strings.Join(contentParts, "\n"),
			EntityType:   "Recipe",
			Metadata: map[string]interface{}{
				"node_id":    entityID,
				"properties": props,
			},
		}

		g.entityKVStore[entityID] = entityKV
		g.keyToEntities[entityName] = append(g.keyToEntities[entityName], entityID)
	}

	// 处理食材实体
	for _, ingredient := range ingredients {
		entityID := ingredient.GetNodeID()
		entityName := ingredient.GetName()
		if entityName == "" {
			entityName = fmt.Sprintf("食材_%s", entityID)
		}

		contentParts := []string{fmt.Sprintf("食材名称: %s", entityName)}

		props := ingredient.GetProperties()
		if category, exists := props["category"]; exists {
			contentParts = append(contentParts, fmt.Sprintf("类别: %v", category))
		}
		if nutrition, exists := props["nutrition"]; exists {
			contentParts = append(contentParts, fmt.Sprintf("营养信息: %v", nutrition))
		}
		if storage, exists := props["storage"]; exists {
			contentParts = append(contentParts, fmt.Sprintf("储存方式: %v", storage))
		}

		entityKV := &EntityKeyValue{
			EntityName:   entityName,
			IndexKeys:    []string{entityName},
			ValueContent: strings.Join(contentParts, "\n"),
			EntityType:   "Ingredient",
			Metadata: map[string]interface{}{
				"node_id":    entityID,
				"properties": props,
			},
		}

		g.entityKVStore[entityID] = entityKV
		g.keyToEntities[entityName] = append(g.keyToEntities[entityName], entityID)
	}

	// 处理烹饪步骤实体
	for _, step := range cookingSteps {
		entityID := step.GetNodeID()
		entityName := fmt.Sprintf("步骤_%s", entityID)

		contentParts := []string{fmt.Sprintf("烹饪步骤: %s", entityName)}

		props := step.GetProperties()
		if description, exists := props["description"]; exists {
			contentParts = append(contentParts, fmt.Sprintf("步骤描述: %v", description))
		}
		if order, exists := props["order"]; exists {
			contentParts = append(contentParts, fmt.Sprintf("步骤顺序: %v", order))
		}
		if technique, exists := props["technique"]; exists {
			contentParts = append(contentParts, fmt.Sprintf("技巧: %v", technique))
		}
		if time, exists := props["time"]; exists {
			contentParts = append(contentParts, fmt.Sprintf("时间: %v", time))
		}

		entityKV := &EntityKeyValue{
			EntityName:   entityName,
			IndexKeys:    []string{entityName},
			ValueContent: strings.Join(contentParts, "\n"),
			EntityType:   "CookingStep",
			Metadata: map[string]interface{}{
				"node_id":    entityID,
				"properties": props,
			},
		}

		g.entityKVStore[entityID] = entityKV
		g.keyToEntities[entityName] = append(g.keyToEntities[entityName], entityID)
	}

	log.Printf("实体键值对创建完成，共 %d 个实体", len(g.entityKVStore))
	return g.entityKVStore
}

// CreateRelationKeyValues 为关系创建键值对结构
// 关系可能有多个索引键，包含从LLM增强的全局主题
func (g *GraphIndexingModule) CreateRelationKeyValues(ctx context.Context, relationships []*Relationship) map[string]*RelationKeyValue {
	log.Println("开始创建关系键值对...")

	for i, rel := range relationships {
		relationID := fmt.Sprintf("rel_%d_%s_%s", i, rel.SourceID, rel.TargetID)

		// 获取源实体和目标实体信息
		sourceEntity := g.entityKVStore[rel.SourceID]
		targetEntity := g.entityKVStore[rel.TargetID]

		if sourceEntity == nil || targetEntity == nil {
			continue
		}

		// 构建关系描述
		contentParts := []string{
			fmt.Sprintf("关系类型: %s", rel.RelationType),
			fmt.Sprintf("源实体: %s (%s)", sourceEntity.EntityName, sourceEntity.EntityType),
			fmt.Sprintf("目标实体: %s (%s)", targetEntity.EntityName, targetEntity.EntityType),
		}

		// 生成多个索引键（包含全局主题）
		indexKeys := g.generateRelationIndexKeys(ctx, sourceEntity, targetEntity, rel.RelationType)

		// 创建关系键值对
		relationKV := &RelationKeyValue{
			RelationID:   relationID,
			IndexKeys:    indexKeys,
			ValueContent: strings.Join(contentParts, "\n"),
			RelationType: rel.RelationType,
			SourceEntity: rel.SourceID,
			TargetEntity: rel.TargetID,
			Metadata: map[string]interface{}{
				"source_name":        sourceEntity.EntityName,
				"target_name":        targetEntity.EntityName,
				"created_from_graph": true,
			},
		}

		g.relationKVStore[relationID] = relationKV

		// 为每个索引键建立映射
		for _, key := range indexKeys {
			g.keyToRelations[key] = append(g.keyToRelations[key], relationID)
		}
	}

	log.Printf("关系键值对创建完成，共 %d 个关系", len(g.relationKVStore))
	return g.relationKVStore
}

// generateRelationIndexKeys 为关系生成多个索引键，包含全局主题
func (g *GraphIndexingModule) generateRelationIndexKeys(ctx context.Context, sourceEntity *EntityKeyValue, targetEntity *EntityKeyValue, relationType string) []string {
	keys := []string{relationType} // 基础关系类型键

	// 根据关系类型和实体类型生成主题键
	switch relationType {
	case "REQUIRES":
		// 菜谱-食材关系的主题键
		keys = append(keys,
			"食材搭配",
			"烹饪原料",
			fmt.Sprintf("%s_食材", sourceEntity.EntityName),
			targetEntity.EntityName,
		)
	case "HAS_STEP":
		// 菜谱-步骤关系的主题键
		keys = append(keys,
			"制作步骤",
			"烹饪过程",
			fmt.Sprintf("%s_步骤", sourceEntity.EntityName),
			"制作方法",
		)
	case "BELONGS_TO_CATEGORY":
		// 分类关系的主题键
		keys = append(keys,
			"菜品分类",
			"美食类别",
			targetEntity.EntityName,
		)
	}

	// 使用LLM增强关系索引键（可选）
	if g.config != nil {
		if enableLLMKeys, exists := g.config.Constraints["enable_llm_relation_keys"]; exists {
			if enable, ok := enableLLMKeys.(bool); ok && enable {
				enhancedKeys := g.llmEnhanceRelationKeys(ctx, sourceEntity, targetEntity, relationType)
				keys = append(keys, enhancedKeys...)
			}
		}
	}

	// 去重并返回
	return g.uniqueStrings(keys)
}

// llmEnhanceRelationKeys 使用LLM增强关系索引键，生成全局主题
func (g *GraphIndexingModule) llmEnhanceRelationKeys(ctx context.Context, sourceEntity *EntityKeyValue, targetEntity *EntityKeyValue, relationType string) []string {

	template := prompt.FromMessages(schema.FString,
		schema.SystemMessage("你是一个{role}。"),
		&schema.Message{
			Role: schema.User,
			Content: `分析以下实体关系，生成相关的主题关键词：
			源实体: {source_name} ({source_type})
			目标实体: {target_name} ({target_type}) 
			关系类型: {relation_type}

			请生成3-5个相关的主题关键词，用于索引和检索。
			返回JSON格式：{"keywords": ["关键词1", "关键词2", "关键词3"]}`,
		},
	)

	values := map[string]interface{}{
		"source_name":   sourceEntity.EntityName,
		"source_type":   sourceEntity.EntityType,
		"target_name":   targetEntity.EntityName,
		"target_type":   targetEntity.EntityType,
		"relation_type": relationType,
	}

	messages, err := template.Format(context.Background(), values)
	if err != nil {
		log.Printf("LLM增强关系索引键失败: %v", err)
		return []string{}
	}

	response, err := g.llmClient.Generate(context.Background(), messages, model.WithTemperature(0.1), model.WithMaxTokens(200))

	if err != nil {
		log.Printf("LLM增强关系索引键失败: %v", err)
		return []string{}
	}

	var result LLMKeywordsResponse
	if err := json.Unmarshal([]byte(response.Content), &result); err != nil {
		log.Printf("解析LLM响应失败: %v", err)
		return []string{}
	}

	return result.Keywords
}

// DeduplicateEntitiesAndRelations 去重相同的实体和关系，优化图操作
func (g *GraphIndexingModule) DeduplicateEntitiesAndRelations() {
	log.Println("开始去重实体和关系...")

	// 实体去重：基于名称
	nameToEntities := make(map[string][]string)
	for entityID, entityKV := range g.entityKVStore {
		nameToEntities[entityKV.EntityName] = append(nameToEntities[entityKV.EntityName], entityID)
	}

	// 合并重复实体
	var entitiesToRemove []string
	for _, entityIDs := range nameToEntities {
		if len(entityIDs) > 1 {
			// 保留第一个，合并其他的内容
			primaryID := entityIDs[0]
			primaryEntity := g.entityKVStore[primaryID]

			for _, entityID := range entityIDs[1:] {
				duplicateEntity := g.entityKVStore[entityID]
				// 合并内容
				primaryEntity.ValueContent += fmt.Sprintf("\n\n补充信息: %s", duplicateEntity.ValueContent)
				// 标记删除
				entitiesToRemove = append(entitiesToRemove, entityID)
			}
		}
	}

	// 删除重复实体
	for _, entityID := range entitiesToRemove {
		delete(g.entityKVStore, entityID)
	}

	// 关系去重：基于源-目标-类型
	relationSignatureToIDs := make(map[string][]string)
	for relationID, relationKV := range g.relationKVStore {
		signature := fmt.Sprintf("%s_%s_%s", relationKV.SourceEntity, relationKV.TargetEntity, relationKV.RelationType)
		relationSignatureToIDs[signature] = append(relationSignatureToIDs[signature], relationID)
	}

	// 合并重复关系
	var relationsToRemove []string
	for _, relationIDs := range relationSignatureToIDs {
		if len(relationIDs) > 1 {
			// 保留第一个，删除其他
			for _, relationID := range relationIDs[1:] {
				relationsToRemove = append(relationsToRemove, relationID)
			}
		}
	}

	// 删除重复关系
	for _, relationID := range relationsToRemove {
		delete(g.relationKVStore, relationID)
	}

	// 重建索引映射
	g.rebuildKeyMappings()

	log.Printf("去重完成 - 删除了 %d 个重复实体，%d 个重复关系", len(entitiesToRemove), len(relationsToRemove))
}

// rebuildKeyMappings 重建键到实体/关系的映射
func (g *GraphIndexingModule) rebuildKeyMappings() {
	// 清空现有映射
	g.keyToEntities = make(map[string][]string)
	g.keyToRelations = make(map[string][]string)

	// 重建实体映射
	for entityID, entityKV := range g.entityKVStore {
		for _, key := range entityKV.IndexKeys {
			g.keyToEntities[key] = append(g.keyToEntities[key], entityID)
		}
	}

	// 重建关系映射
	for relationID, relationKV := range g.relationKVStore {
		for _, key := range relationKV.IndexKeys {
			g.keyToRelations[key] = append(g.keyToRelations[key], relationID)
		}
	}
}

// GetEntitiesByKey 根据索引键获取实体
func (g *GraphIndexingModule) GetEntitiesByKey(key string) []*EntityKeyValue {
	entityIDs := g.keyToEntities[key]
	var entities []*EntityKeyValue

	for _, entityID := range entityIDs {
		if entity, exists := g.entityKVStore[entityID]; exists {
			entities = append(entities, entity)
		}
	}

	return entities
}

// GetRelationsByKey 根据索引键获取关系
func (g *GraphIndexingModule) GetRelationsByKey(key string) []*RelationKeyValue {
	relationIDs := g.keyToRelations[key]
	var relations []*RelationKeyValue

	for _, relationID := range relationIDs {
		if relation, exists := g.relationKVStore[relationID]; exists {
			relations = append(relations, relation)
		}
	}

	return relations
}

// GetStatistics 获取键值对存储统计信息
func (g *GraphIndexingModule) GetStatistics() map[string]interface{} {
	totalEntityKeys := 0
	for _, entityKV := range g.entityKVStore {
		totalEntityKeys += len(entityKV.IndexKeys)
	}

	totalRelationKeys := 0
	for _, relationKV := range g.relationKVStore {
		totalRelationKeys += len(relationKV.IndexKeys)
	}

	// 统计实体类型
	entityTypes := map[string]int{
		"Recipe":      0,
		"Ingredient":  0,
		"CookingStep": 0,
	}

	for _, entityKV := range g.entityKVStore {
		if count, exists := entityTypes[entityKV.EntityType]; exists {
			entityTypes[entityKV.EntityType] = count + 1
		}
	}

	return map[string]interface{}{
		"total_entities":      len(g.entityKVStore),
		"total_relations":     len(g.relationKVStore),
		"total_entity_keys":   totalEntityKeys,
		"total_relation_keys": totalRelationKeys,
		"entity_types":        entityTypes,
	}
}

// SearchByKeyword 根据关键词搜索实体和关系
func (g *GraphIndexingModule) SearchByKeyword(keyword string) (entities []*EntityKeyValue, relations []*RelationKeyValue) {
	// 精确匹配
	entities = append(entities, g.GetEntitiesByKey(keyword)...)
	relations = append(relations, g.GetRelationsByKey(keyword)...)

	// 模糊匹配（包含关键词的键）
	for key := range g.keyToEntities {
		if strings.Contains(strings.ToLower(key), strings.ToLower(keyword)) && key != keyword {
			entities = append(entities, g.GetEntitiesByKey(key)...)
		}
	}

	for key := range g.keyToRelations {
		if strings.Contains(strings.ToLower(key), strings.ToLower(keyword)) && key != keyword {
			relations = append(relations, g.GetRelationsByKey(key)...)
		}
	}

	// 去重
	entities = g.uniqueEntities(entities)
	relations = g.uniqueRelations(relations)

	return entities, relations
}

// GetAllEntityKeys 获取所有实体索引键
func (g *GraphIndexingModule) GetAllEntityKeys() []string {
	var keys []string
	for key := range g.keyToEntities {
		keys = append(keys, key)
	}
	return keys
}

// GetAllRelationKeys 获取所有关系索引键
func (g *GraphIndexingModule) GetAllRelationKeys() []string {
	var keys []string
	for key := range g.keyToRelations {
		keys = append(keys, key)
	}
	return keys
}

// ========== 辅助方法 ==========

// uniqueStrings 字符串数组去重
func (g *GraphIndexingModule) uniqueStrings(strs []string) []string {
	seen := make(map[string]bool)
	var result []string

	for _, str := range strs {
		if !seen[str] {
			seen[str] = true
			result = append(result, str)
		}
	}

	return result
}

// uniqueEntities 实体去重
func (g *GraphIndexingModule) uniqueEntities(entities []*EntityKeyValue) []*EntityKeyValue {
	seen := make(map[string]bool)
	var result []*EntityKeyValue

	for _, entity := range entities {
		key := fmt.Sprintf("%s_%s", entity.EntityName, entity.EntityType)
		if !seen[key] {
			seen[key] = true
			result = append(result, entity)
		}
	}

	return result
}

// uniqueRelations 关系去重
func (g *GraphIndexingModule) uniqueRelations(relations []*RelationKeyValue) []*RelationKeyValue {
	seen := make(map[string]bool)
	var result []*RelationKeyValue

	for _, relation := range relations {
		if !seen[relation.RelationID] {
			seen[relation.RelationID] = true
			result = append(result, relation)
		}
	}

	return result
}

// ExportToJSON 导出索引数据到JSON
func (g *GraphIndexingModule) ExportToJSON() (map[string]interface{}, error) {
	return map[string]interface{}{
		"entities":   g.entityKVStore,
		"relations":  g.relationKVStore,
		"statistics": g.GetStatistics(),
	}, nil
}

// ImportFromJSON 从JSON导入索引数据
func (g *GraphIndexingModule) ImportFromJSON(data map[string]interface{}) error {
	// 清空现有数据
	g.entityKVStore = make(map[string]*EntityKeyValue)
	g.relationKVStore = make(map[string]*RelationKeyValue)

	// 导入实体数据
	if entitiesData, exists := data["entities"]; exists {
		if entitiesMap, ok := entitiesData.(map[string]interface{}); ok {
			for entityID, entityData := range entitiesMap {
				entityBytes, _ := json.Marshal(entityData)
				var entity EntityKeyValue
				if err := json.Unmarshal(entityBytes, &entity); err == nil {
					g.entityKVStore[entityID] = &entity
				}
			}
		}
	}

	// 导入关系数据
	if relationsData, exists := data["relations"]; exists {
		if relationsMap, ok := relationsData.(map[string]interface{}); ok {
			for relationID, relationData := range relationsMap {
				relationBytes, _ := json.Marshal(relationData)
				var relation RelationKeyValue
				if err := json.Unmarshal(relationBytes, &relation); err == nil {
					g.relationKVStore[relationID] = &relation
				}
			}
		}
	}

	// 重建索引映射
	g.rebuildKeyMappings()

	return nil
}
