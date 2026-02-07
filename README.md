# KGUPDATER

基于 DepCache 思路构建的 Graph-RAG 框架雏形。当前仓库只提供**架构设计与模块说明**，不包含可运行的业务代码，便于后续按需实现与扩展。

## 目标

解决传统 Graph-RAG 在知识图谱增量更新时准确率与效率较低的问题：

1. **抽取即缓存**：实体关系抽取完成后不立刻写入图数据库，而是暂存在内存子图缓存中。
2. **检索优先子图**：查询时优先从子图缓存中检索相似实体，再检索图数据库。
3. **访问频次驱动合并**：对子图实体记录访问频率，达到阈值后再合并到图数据库，减少不必要的图合并。

## 推荐技术选型（可按业务调整）

- **图数据库**：
  - Neo4j（生态成熟、查询语言 Cypher）
  - TigerGraph（高性能图计算）
  - NebulaGraph（分布式、开源）
- **向量数据库**：
  - Milvus（高性能、生态完善）
  - Qdrant（轻量、易部署）
  - Weaviate（自带检索/元数据能力）

## 项目结构

```
KGUPDATER/
├── README.md
├── docs/
│   └── architecture.md          # 架构设计与模块交互说明
└── src/
    └── graphrag/
        ├── __init__.py           # 包入口与总体说明
        ├── document_processing/  # 文档处理与实体关系抽取模块
        │   └── README.md
        ├── retrieve/             # 检索与访问频次管理模块
        │   └── README.md
        ├── generate/             # 生成回答模块
        │   └── README.md
        ├── storage/              # 图数据库/向量数据库/子图缓存接口
        │   └── README.md
        └── orchestration/        # 任务编排与数据流控制
            └── README.md
```

## 下一步建议

- 在 `storage/` 中先实现**子图缓存层**和访问频次统计策略，然后再对接图数据库。
- 在 `retrieve/` 模块中引入**分层检索策略**与**阈值驱动合并策略**。
- 在 `document_processing/` 中明确**实体/关系 schema**、抽取模板与 embedding 策略。

详细设计说明见 `docs/architecture.md`。
