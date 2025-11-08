package batch_0001

import (
	"context"
	"fmt"
	"log"
	"strconv"
	"strings"

	"github.com/cloudwego/eino/schema"
	"github.com/neo4j/neo4j-go-driver/v5/neo4j"
)

type GraphNode struct {
	NodeID     string                 `json:"node_id"`    // 节点的唯一标识符
	Labels     []string               `json:"labels"`     // 节点的标签列表，如['Recipe', 'ChineseCuisine']
	Name       string                 `json:"name"`       // 节点的名称，如'宫保鸡丁'
	Properties map[string]interface{} `json:"properties"` // 节点的所有属性，如难度、烹饪时间等
}

type GraphRelation struct {
	StartNodeID  string                 `json:"start_node_id"` // 关系起始节点的ID
	EndNodeID    string                 `json:"end_node_id"`   // 关系结束节点的ID
	RelationType string                 `json:"relation_type"` // 关系类型，如'REQUIRES', 'CONTAINS_STEP'
	Properties   map[string]interface{} `json:"properties"`    // 关系的属性，如分量、顺序等
}

type GraphDataPreparationModule struct {
	// 连接参数
	URI      string `json:"uri"`      // Neo4j数据库连接URI
	User     string `json:"user"`     // 数据库用户名
	Password string `json:"password"` // 数据库密码
	Database string `json:"database"` // 数据库名称

	// 数据库驱动
	Driver neo4j.DriverWithContext `json:"-"`

	// 数据存储容器 - 使用 Eino Schema.Document
	Documents    []*schema.Document `json:"documents"`     // 完整文档列表
	Chunks       []*schema.Document `json:"chunks"`        // 分块文档列表
	Recipes      []GraphNode        `json:"recipes"`       // 菜谱实体列表
	Ingredients  []GraphNode        `json:"ingredients"`   // 食材实体列表
	CookingSteps []GraphNode        `json:"cooking_steps"` // 烹饪步骤实体列表
}

func NewGraphDataPreparationModule(uri, user, password, database string) (*GraphDataPreparationModule, error) {

	if database == "" {
		database = "neo4j"
	}

	module := &GraphDataPreparationModule{
		URI:          uri,
		User:         user,
		Password:     password,
		Database:     database,
		Documents:    make([]*schema.Document, 0),
		Chunks:       make([]*schema.Document, 0),
		Recipes:      make([]GraphNode, 0),
		Ingredients:  make([]GraphNode, 0),
		CookingSteps: make([]GraphNode, 0),
	}

	// 建立数据库连接
	err := module.connect()
	if err != nil {
		return nil, err
	}

	return module, nil
}

func (g *GraphDataPreparationModule) connect() error {
	// 创建Neo4j驱动实例
	driver, err := neo4j.NewDriverWithContext(
		g.URI,
		neo4j.BasicAuth(g.User, g.Password, ""),
	)
	if err != nil {
		log.Printf("连接Neo4j失败: %v", err)
		return err
	}

	g.Driver = driver
	log.Printf("已连接到Neo4j数据库: %s", g.URI)

	// 测试连接 - 执行简单查询验证连接有效性
	ctx := context.Background()
	err = driver.VerifyConnectivity(ctx)
	if err != nil {
		log.Printf("Neo4j连接测试失败: %v", err)
		return err
	}

	log.Println("Neo4j连接测试成功")
	return nil
}

// Close 安全关闭数据库连接
//
// 释放数据库连接资源，避免连接泄露。
// 建议在程序结束或不再需要数据库操作时调用。
func (g *GraphDataPreparationModule) Close() error {
	if g.Driver != nil {
		ctx := context.Background()
		err := g.Driver.Close(ctx)
		if err != nil {
			log.Printf("关闭Neo4j连接失败: %v", err)
			return err
		}
		log.Println("Neo4j连接已关闭")
	}
	return nil
}

func (g *GraphDataPreparationModule) LoadGraphData() (map[string]interface{}, error) {
	log.Println("正在从Neo4j加载图数据...")

	ctx := context.Background()
	session := g.Driver.NewSession(ctx, neo4j.SessionConfig{DatabaseName: g.Database})
	defer session.Close(ctx)

	// 加载所有菜谱节点，同时获取分类信息
	// 使用OPTIONAL MATCH确保即使没有分类关系的菜谱也能被加载
	recipesQuery := `
		MATCH (r:Recipe)
		WHERE r.nodeId >= '200000000'
		OPTIONAL MATCH (r)-[:BELONGS_TO_CATEGORY]->(c:Category)
		WITH r, collect(c.name) as categories
		RETURN r.nodeId as nodeId, labels(r) as labels, r.name as name, 
		       properties(r) as originalProperties,
		       CASE WHEN size(categories) > 0 
		            THEN categories[0] 
		            ELSE COALESCE(r.category, '未知') END as mainCategory,
		       CASE WHEN size(categories) > 0 
		            THEN categories 
		            ELSE [COALESCE(r.category, '未知')] END as allCategories
		ORDER BY r.nodeId
	`

	result, err := session.ExecuteRead(ctx, func(tx neo4j.ManagedTransaction) (interface{}, error) {
		result, err := tx.Run(ctx, recipesQuery, nil)
		if err != nil {
			return nil, err
		}

		var recipes []GraphNode
		for result.Next(ctx) {
			record := result.Record()

			// 获取节点属性
			nodeID, _ := record.Get("nodeId")
			labels, _ := record.Get("labels")
			name, _ := record.Get("name")
			originalProperties, _ := record.Get("originalProperties")
			mainCategory, _ := record.Get("mainCategory")
			allCategories, _ := record.Get("allCategories")

			// 转换属性
			properties := make(map[string]interface{})
			if props, ok := originalProperties.(map[string]interface{}); ok {
				for k, v := range props {
					properties[k] = v
				}
			}
			properties["category"] = mainCategory
			properties["all_categories"] = allCategories

			// 转换标签
			var labelsList []string
			if lbls, ok := labels.([]interface{}); ok {
				for _, lbl := range lbls {
					if str, ok := lbl.(string); ok {
						labelsList = append(labelsList, str)
					}
				}
			}

			node := GraphNode{
				NodeID:     fmt.Sprintf("%v", nodeID),
				Labels:     labelsList,
				Name:       fmt.Sprintf("%v", name),
				Properties: properties,
			}
			recipes = append(recipes, node)
		}

		return recipes, result.Err()
	})

	if err != nil {
		return nil, fmt.Errorf("加载菜谱数据失败: %v", err)
	}

	g.Recipes = result.([]GraphNode)
	log.Printf("加载了 %d 个菜谱节点", len(g.Recipes))

	// 加载所有食材节点
	ingredientsQuery := `
		MATCH (i:Ingredient)
		WHERE i.nodeId >= '200000000'
		RETURN i.nodeId as nodeId, labels(i) as labels, i.name as name,
		       properties(i) as properties
		ORDER BY i.nodeId
	`

	result, err = session.ExecuteRead(ctx, func(tx neo4j.ManagedTransaction) (interface{}, error) {
		result, err := tx.Run(ctx, ingredientsQuery, nil)
		if err != nil {
			return nil, err
		}

		var ingredients []GraphNode
		for result.Next(ctx) {
			record := result.Record()

			nodeID, _ := record.Get("nodeId")
			labels, _ := record.Get("labels")
			name, _ := record.Get("name")
			properties, _ := record.Get("properties")

			// 转换属性
			props := make(map[string]interface{})
			if p, ok := properties.(map[string]interface{}); ok {
				props = p
			}

			// 转换标签
			var labelsList []string
			if lbls, ok := labels.([]interface{}); ok {
				for _, lbl := range lbls {
					if str, ok := lbl.(string); ok {
						labelsList = append(labelsList, str)
					}
				}
			}

			node := GraphNode{
				NodeID:     fmt.Sprintf("%v", nodeID),
				Labels:     labelsList,
				Name:       fmt.Sprintf("%v", name),
				Properties: props,
			}
			ingredients = append(ingredients, node)
		}

		return ingredients, result.Err()
	})

	if err != nil {
		return nil, fmt.Errorf("加载食材数据失败: %v", err)
	}

	g.Ingredients = result.([]GraphNode)
	log.Printf("加载了 %d 个食材节点", len(g.Ingredients))

	// 加载所有烹饪步骤节点
	stepsQuery := `
		MATCH (s:CookingStep)
		WHERE s.nodeId >= '200000000'
		RETURN s.nodeId as nodeId, labels(s) as labels, s.name as name,
		       properties(s) as properties
		ORDER BY s.nodeId
	`

	result, err = session.ExecuteRead(ctx, func(tx neo4j.ManagedTransaction) (interface{}, error) {
		result, err := tx.Run(ctx, stepsQuery, nil)
		if err != nil {
			return nil, err
		}

		var steps []GraphNode
		for result.Next(ctx) {
			record := result.Record()

			nodeID, _ := record.Get("nodeId")
			labels, _ := record.Get("labels")
			name, _ := record.Get("name")
			properties, _ := record.Get("properties")

			// 转换属性
			props := make(map[string]interface{})
			if p, ok := properties.(map[string]interface{}); ok {
				props = p
			}

			// 转换标签
			var labelsList []string
			if lbls, ok := labels.([]interface{}); ok {
				for _, lbl := range lbls {
					if str, ok := lbl.(string); ok {
						labelsList = append(labelsList, str)
					}
				}
			}

			node := GraphNode{
				NodeID:     fmt.Sprintf("%v", nodeID),
				Labels:     labelsList,
				Name:       fmt.Sprintf("%v", name),
				Properties: props,
			}
			steps = append(steps, node)
		}

		return steps, result.Err()
	})

	if err != nil {
		return nil, fmt.Errorf("加载烹饪步骤数据失败: %v", err)
	}

	g.CookingSteps = result.([]GraphNode)
	log.Printf("加载了 %d 个烹饪步骤节点", len(g.CookingSteps))

	// 返回加载统计信息
	return map[string]interface{}{
		"recipes":       len(g.Recipes),
		"ingredients":   len(g.Ingredients),
		"cooking_steps": len(g.CookingSteps),
	}, nil
}

func (g *GraphDataPreparationModule) BuildRecipeDocuments() ([]*schema.Document, error) {
	log.Println("正在构建菜谱文档...")

	ctx := context.Background()
	session := g.Driver.NewSession(ctx, neo4j.SessionConfig{DatabaseName: g.Database})
	defer session.Close(ctx)

	var documents []*schema.Document

	// 遍历所有已加载的菜谱实体，为每个菜谱构建完整文档
	for _, recipe := range g.Recipes {
		recipeID := recipe.NodeID
		recipeName := recipe.Name

		// 第一步：获取菜谱的相关食材信息
		// 通过REQUIRES关系查询菜谱所需的所有食材，包括用量信息
		ingredientsQuery := `
			MATCH (r:Recipe {nodeId: $recipe_id})-[req:REQUIRES]->(i:Ingredient)
			RETURN i.name as name, i.category as category, 
			       req.amount as amount, req.unit as unit,
			       i.description as description
			ORDER BY i.name
		`

		ingredientsResult, err := session.ExecuteRead(ctx, func(tx neo4j.ManagedTransaction) (interface{}, error) {
			result, err := tx.Run(ctx, ingredientsQuery, map[string]interface{}{
				"recipe_id": recipeID,
			})
			if err != nil {
				return nil, err
			}

			var ingredientsInfo []string
			for result.Next(ctx) {
				record := result.Record()

				name, _ := record.Get("name")
				amount, _ := record.Get("amount")
				unit, _ := record.Get("unit")
				description, _ := record.Get("description")

				// 构建食材描述文本：名称 + 用量 + 描述
				ingredientText := fmt.Sprintf("%v", name)

				// 添加用量信息（如果有）
				if amount != nil && unit != nil {
					amountStr := fmt.Sprintf("%v", amount)
					unitStr := fmt.Sprintf("%v", unit)
					if amountStr != "" && unitStr != "" {
						ingredientText += fmt.Sprintf("(%s%s)", amountStr, unitStr)
					}
				}

				// 添加食材描述（如果有）
				if description != nil {
					descStr := fmt.Sprintf("%v", description)
					if descStr != "" {
						ingredientText += fmt.Sprintf(" - %s", descStr)
					}
				}

				ingredientsInfo = append(ingredientsInfo, ingredientText)
			}

			return ingredientsInfo, result.Err()
		})

		if err != nil {
			log.Printf("获取菜谱食材失败 %s (ID: %s): %v", recipeName, recipeID, err)
			continue
		}

		ingredientsInfo := ingredientsResult.([]string)

		// 第二步：获取菜谱的烹饪步骤信息
		// 通过CONTAINS_STEP关系查询菜谱的所有制作步骤，按顺序排列
		stepsQuery := `
			MATCH (r:Recipe {nodeId: $recipe_id})-[c:CONTAINS_STEP]->(s:CookingStep)
			RETURN s.name as name, s.description as description,
			       s.stepNumber as stepNumber, s.methods as methods,
			       s.tools as tools, s.timeEstimate as timeEstimate,
			       c.stepOrder as stepOrder
			ORDER BY COALESCE(c.stepOrder, s.stepNumber, 999)
		`

		stepsResult, err := session.ExecuteRead(ctx, func(tx neo4j.ManagedTransaction) (interface{}, error) {
			result, err := tx.Run(ctx, stepsQuery, map[string]interface{}{
				"recipe_id": recipeID,
			})
			if err != nil {
				return nil, err
			}

			var stepsInfo []string
			for result.Next(ctx) {
				record := result.Record()

				name, _ := record.Get("name")
				description, _ := record.Get("description")
				methods, _ := record.Get("methods")
				tools, _ := record.Get("tools")
				timeEstimate, _ := record.Get("timeEstimate")

				// 构建步骤描述文本：包含步骤名称、详细描述、方法、工具、时间等
				stepText := fmt.Sprintf("步骤: %v", name)

				// 添加详细描述（如果有）
				if description != nil {
					descStr := fmt.Sprintf("%v", description)
					if descStr != "" {
						stepText += fmt.Sprintf("\n描述: %s", descStr)
					}
				}

				// 添加烹饪方法（如果有）
				if methods != nil {
					methodsStr := fmt.Sprintf("%v", methods)
					if methodsStr != "" {
						stepText += fmt.Sprintf("\n方法: %s", methodsStr)
					}
				}

				// 添加所需工具（如果有）
				if tools != nil {
					toolsStr := fmt.Sprintf("%v", tools)
					if toolsStr != "" {
						stepText += fmt.Sprintf("\n工具: %s", toolsStr)
					}
				}

				// 添加时间估计（如果有）
				if timeEstimate != nil {
					timeStr := fmt.Sprintf("%v", timeEstimate)
					if timeStr != "" {
						stepText += fmt.Sprintf("\n时间: %s", timeStr)
					}
				}

				stepsInfo = append(stepsInfo, stepText)
			}

			return stepsInfo, result.Err()
		})

		if err != nil {
			log.Printf("获取菜谱步骤失败 %s (ID: %s): %v", recipeName, recipeID, err)
			continue
		}

		stepsInfo := stepsResult.([]string)

		// 第三步：构建完整的菜谱文档内容
		// 使用Markdown格式，便于阅读和后续处理
		var contentParts []string
		contentParts = append(contentParts, fmt.Sprintf("# %s", recipeName))

		// 添加菜谱基本信息部分
		if description, ok := recipe.Properties["description"]; ok && description != nil {
			descStr := fmt.Sprintf("%v", description)
			if descStr != "" {
				contentParts = append(contentParts, fmt.Sprintf("\n## 菜品描述\n%s", descStr))
			}
		}

		// 添加菜系信息
		if cuisineType, ok := recipe.Properties["cuisineType"]; ok && cuisineType != nil {
			cuisineStr := fmt.Sprintf("%v", cuisineType)
			if cuisineStr != "" {
				contentParts = append(contentParts, fmt.Sprintf("\n菜系: %s", cuisineStr))
			}
		}

		// 添加难度信息
		if difficulty, ok := recipe.Properties["difficulty"]; ok && difficulty != nil {
			difficultyStr := fmt.Sprintf("%v", difficulty)
			if difficultyStr != "" {
				contentParts = append(contentParts, fmt.Sprintf("难度: %s星", difficultyStr))
			}
		}

		// 添加时间信息（准备时间和烹饪时间）
		var timeInfo []string
		if prepTime, ok := recipe.Properties["prepTime"]; ok && prepTime != nil {
			prepTimeStr := fmt.Sprintf("%v", prepTime)
			if prepTimeStr != "" {
				timeInfo = append(timeInfo, fmt.Sprintf("准备时间: %s", prepTimeStr))
			}
		}
		if cookTime, ok := recipe.Properties["cookTime"]; ok && cookTime != nil {
			cookTimeStr := fmt.Sprintf("%v", cookTime)
			if cookTimeStr != "" {
				timeInfo = append(timeInfo, fmt.Sprintf("烹饪时间: %s", cookTimeStr))
			}
		}
		if len(timeInfo) > 0 {
			contentParts = append(contentParts, fmt.Sprintf("\n时间信息: %s", strings.Join(timeInfo, ", ")))
		}

		// 添加份量信息
		if servings, ok := recipe.Properties["servings"]; ok && servings != nil {
			servingsStr := fmt.Sprintf("%v", servings)
			if servingsStr != "" {
				contentParts = append(contentParts, fmt.Sprintf("份量: %s", servingsStr))
			}
		}

		// 添加食材清单部分
		if len(ingredientsInfo) > 0 {
			contentParts = append(contentParts, "\n## 所需食材")
			for i, ingredient := range ingredientsInfo {
				contentParts = append(contentParts, fmt.Sprintf("%d. %s", i+1, ingredient))
			}
		}

		// 添加制作步骤部分
		if len(stepsInfo) > 0 {
			contentParts = append(contentParts, "\n## 制作步骤")
			for i, step := range stepsInfo {
				contentParts = append(contentParts, fmt.Sprintf("\n### 第%d步\n%s", i+1, step))
			}
		}

		// 添加标签信息部分
		if tags, ok := recipe.Properties["tags"]; ok && tags != nil {
			tagsStr := fmt.Sprintf("%v", tags)
			if tagsStr != "" {
				contentParts = append(contentParts, fmt.Sprintf("\n## 标签\n%s", tagsStr))
			}
		}

		// 组合成最终的完整文档内容
		fullContent := strings.Join(contentParts, "\n")

		// 第四步：创建Document对象
		// 包含完整的文档内容和丰富的元数据，支持后续的检索和处理
		metadata := map[string]interface{}{
			// 图数据库相关信息
			"node_id":     recipeID,   // 原始节点ID
			"recipe_name": recipeName, // 菜谱名称
			"node_type":   "Recipe",   // 节点类型

			// 菜谱属性信息
			"category":     getStringFromMap(recipe.Properties, "category", "未知"),    // 菜品分类
			"cuisine_type": getStringFromMap(recipe.Properties, "cuisineType", "未知"), // 菜系类型
			"difficulty":   getIntFromMap(recipe.Properties, "difficulty", 0),        // 难度等级
			"prep_time":    getStringFromMap(recipe.Properties, "prepTime", ""),      // 准备时间
			"cook_time":    getStringFromMap(recipe.Properties, "cookTime", ""),      // 烹饪时间
			"servings":     getStringFromMap(recipe.Properties, "servings", ""),      // 服务份量

			// 文档统计信息
			"ingredients_count": len(ingredientsInfo), // 食材数量
			"steps_count":       len(stepsInfo),       // 步骤数量
			"doc_type":          "recipe",             // 文档类型
			"content_length":    len(fullContent),     // 内容长度
		}

		doc := &schema.Document{
			ID:       recipeID,
			Content:  fullContent,
			MetaData: metadata,
		}

		documents = append(documents, doc)
	}

	g.Documents = documents
	log.Printf("成功构建 %d 个菜谱文档", len(documents))
	return documents, nil
}

func (g *GraphDataPreparationModule) ChunkDocuments(chunkSize, chunkOverlap int) ([]*schema.Document, error) {
	if chunkSize <= 0 {
		chunkSize = 500
	}
	if chunkOverlap < 0 {
		chunkOverlap = 50
	}

	log.Printf("正在进行文档分块，块大小: %d, 重叠: %d", chunkSize, chunkOverlap)

	// 检查是否有可分块的文档
	if len(g.Documents) == 0 {
		return nil, fmt.Errorf("请先构建文档")
	}

	var chunks []*schema.Document
	chunkID := 0

	// 遍历所有文档进行分块处理
	for _, doc := range g.Documents {
		content := doc.Content

		// 分块决策：根据文档长度选择分块策略
		if len(content) <= chunkSize {
			// 情况1：文档较短，无需分块
			// 直接将整个文档作为一个块，保持完整性
			metadata := make(map[string]interface{})
			for k, v := range doc.MetaData {
				metadata[k] = v
			}
			metadata["chunk_id"] = fmt.Sprintf("%v_chunk_%d", doc.MetaData["node_id"], chunkID)
			metadata["parent_id"] = doc.MetaData["node_id"]
			metadata["chunk_index"] = 0
			metadata["total_chunks"] = 1
			metadata["chunk_size"] = len(content)
			metadata["doc_type"] = "chunk"

			chunk := &schema.Document{
				ID:       fmt.Sprintf("%v_chunk_%d", doc.MetaData["node_id"], chunkID),
				Content:  content,
				MetaData: metadata,
			}
			chunks = append(chunks, chunk)
			chunkID++
		} else {
			// 情况2：文档较长，需要分块处理
			// 优先按Markdown章节（## 标题）进行语义分块
			sections := strings.Split(content, "\n## ")
			if len(sections) <= 1 {
				// 情况2a：没有二级标题，按长度强制分块
				// 使用滑动窗口方式，保持重叠以维持上下文
				totalChunks := (len(content)-1)/(chunkSize-chunkOverlap) + 1

				for i := 0; i < totalChunks; i++ {
					start := i * (chunkSize - chunkOverlap)
					end := start + chunkSize
					if end > len(content) {
						end = len(content)
					}

					chunkContent := content[start:end]

					metadata := make(map[string]interface{})
					for k, v := range doc.MetaData {
						metadata[k] = v
					}
					metadata["chunk_id"] = fmt.Sprintf("%v_chunk_%d", doc.MetaData["node_id"], chunkID)
					metadata["parent_id"] = doc.MetaData["node_id"]
					metadata["chunk_index"] = i
					metadata["total_chunks"] = totalChunks
					metadata["chunk_size"] = len(chunkContent)
					metadata["doc_type"] = "chunk"
					metadata["chunking_method"] = "length_based"

					chunk := &schema.Document{
						ID:       fmt.Sprintf("%v_chunk_%d", doc.MetaData["node_id"], chunkID),
						Content:  chunkContent,
						MetaData: metadata,
					}
					chunks = append(chunks, chunk)
					chunkID++
				}
			} else {
				// 情况2b：有二级标题，按章节语义分块
				// 这种方式能更好地保持内容的语义完整性
				totalChunks := len(sections)
				for i, section := range sections {
					var chunkContent string
					var sectionTitle string

					if i == 0 {
						// 第一个部分包含文档标题和开头内容
						chunkContent = section
						sectionTitle = "主标题"
					} else {
						// 其他部分添加章节标题，保持Markdown格式
						chunkContent = fmt.Sprintf("## %s", section)
						lines := strings.Split(section, "\n")
						if len(lines) > 0 {
							sectionTitle = lines[0]
						} else {
							sectionTitle = "未知章节"
						}
					}

					metadata := make(map[string]interface{})
					for k, v := range doc.MetaData {
						metadata[k] = v
					}
					metadata["chunk_id"] = fmt.Sprintf("%v_chunk_%d", doc.MetaData["node_id"], chunkID)
					metadata["parent_id"] = doc.MetaData["node_id"]
					metadata["chunk_index"] = i
					metadata["total_chunks"] = totalChunks
					metadata["chunk_size"] = len(chunkContent)
					metadata["doc_type"] = "chunk"
					metadata["chunking_method"] = "section_based"
					metadata["section_title"] = sectionTitle

					chunk := &schema.Document{
						ID:       fmt.Sprintf("%v_chunk_%d", doc.MetaData["node_id"], chunkID),
						Content:  chunkContent,
						MetaData: metadata,
					}
					chunks = append(chunks, chunk)
					chunkID++
				}
			}
		}
	}

	g.Chunks = chunks
	log.Printf("文档分块完成，共生成 %d 个块", len(chunks))
	return chunks, nil
}

// GetStatistics 获取完整的数据处理统计信息
//
// 提供数据准备过程的详细统计，包括实体数量、文档数量、
// 分块统计以及内容特征分析，用于监控数据质量和处理效果。
//
// Returns:
//
//	map[string]interface{}: 包含以下统计信息的映射：
//	    - 基础统计：实体、文档、分块的数量信息
//	    - 分类统计：按菜系、分类、难度的分布情况
//	    - 质量指标：平均内容长度、分块效果等
func (g *GraphDataPreparationModule) GetStatistics() map[string]interface{} {
	stats := map[string]interface{}{
		"total_recipes":       len(g.Recipes),
		"total_ingredients":   len(g.Ingredients),
		"total_cooking_steps": len(g.CookingSteps),
		"total_documents":     len(g.Documents),
		"total_chunks":        len(g.Chunks),
	}

	if len(g.Documents) > 0 {
		// 分类统计
		categories := make(map[string]int)
		cuisines := make(map[string]int)
		difficulties := make(map[string]int)

		var totalContentLength, totalChunkSize int

		for _, doc := range g.Documents {
			if category, ok := doc.MetaData["category"]; ok {
				categoryStr := fmt.Sprintf("%v", category)
				categories[categoryStr]++
			}

			if cuisine, ok := doc.MetaData["cuisine_type"]; ok {
				cuisineStr := fmt.Sprintf("%v", cuisine)
				cuisines[cuisineStr]++
			}

			if difficulty, ok := doc.MetaData["difficulty"]; ok {
				difficultyStr := fmt.Sprintf("%v", difficulty)
				difficulties[difficultyStr]++
			}

			if contentLength, ok := doc.MetaData["content_length"]; ok {
				if length, ok := contentLength.(int); ok {
					totalContentLength += length
				}
			}
		}

		for _, chunk := range g.Chunks {
			if chunkSize, ok := chunk.MetaData["chunk_size"]; ok {
				if size, ok := chunkSize.(int); ok {
					totalChunkSize += size
				}
			}
		}

		avgContentLength := float64(totalContentLength) / float64(len(g.Documents))
		var avgChunkSize float64
		if len(g.Chunks) > 0 {
			avgChunkSize = float64(totalChunkSize) / float64(len(g.Chunks))
		}

		stats["categories"] = categories
		stats["cuisines"] = cuisines
		stats["difficulties"] = difficulties
		stats["avg_content_length"] = avgContentLength
		stats["avg_chunk_size"] = avgChunkSize
	}

	return stats
}

// 辅助函数：从映射中安全获取字符串值
func getStringFromMap(m map[string]interface{}, key, defaultValue string) string {
	if value, ok := m[key]; ok && value != nil {
		return fmt.Sprintf("%v", value)
	}
	return defaultValue
}

// 辅助函数：从映射中安全获取整数值
func getIntFromMap(m map[string]interface{}, key string, defaultValue int) int {
	if value, ok := m[key]; ok && value != nil {
		if intValue, ok := value.(int64); ok {
			return int(intValue)
		}
		if intValue, ok := value.(int); ok {
			return intValue
		}
		if strValue, ok := value.(string); ok {
			if intValue, err := strconv.Atoi(strValue); err == nil {
				return intValue
			}
		}
	}
	return defaultValue
}
