# Graph-RAG 框架设计（基于 DepCache 思路）

> 说明：本设计提供模块化分层、数据流、接口边界和关键策略建议，代码暂留空实现。

## 1. 核心目标

- **降低图谱更新成本**：减少频繁图合并造成的写入开销与噪声。
- **提高检索与回答准确率**：子图缓存优先检索，提高增量知识的可用性。
- **支持可控合并策略**：通过访问频次或置信度阈值控制实体入库。

## 2. 总体架构

```
文档 -> 处理与抽取 -> 子图缓存(内存) -> 访问频次管理 -> 达阈值 -> 图数据库
                     \-> 向量数据库(用于实体/关系的 embedding 检索)

query -> 检索层(子图缓存优先) -> 图数据库 -> chunk 回溯 -> 生成回答
```

## 3. 模块设计

### 3.1 文档处理模块（Document Processing）

**输入**：原始文档/知识源

**输出**：
- 结构化实体/关系
- 每个实体/关系的 embedding
- chunk 来源与上下文索引

**关键策略建议**：
- 预定义 entity/relationship schema（避免后续合并歧义）。
- 为每个实体和关系记录来源 chunk 和置信度评分。
- embedding 存入向量数据库，用于相似实体检索。

### 3.2 检索模块（Retrieve）

**输入**：query

**输出**：候选实体 + 对应 chunk

**检索流程建议**：
1. 在子图缓存中优先检索相似实体（embedding + 字符串匹配）。
2. 对命中的子图实体记录访问频次（frequency）。
3. 若子图命中不足，则检索图数据库中的相似实体。
4. 将子图与图数据库结果汇总，输出对应 chunk 与实体关系上下文。

**合并策略建议**：
- **访问频次阈值**：达到固定次数后合并。
- **时间衰减阈值**：频次与最近访问时间组合。
- **置信度阈值**：抽取置信度与频次结合。

### 3.3 生成模块（Generate）

**输入**：query + 上下文 chunk

**输出**：最终回答

**建议**：
- 构建 prompt 时区分子图来源与图数据库来源。
- 可以在回答中保留来源标注，便于追踪。

### 3.4 存储模块（Storage）

**子图缓存**：
- 内存子图结构，支持快速相似检索与访问频次统计。
- 可配置定期清理/淘汰策略。

**图数据库**：
- 持久化存储实体关系与跨文档知识。
- 用于稳定知识查询与推理。

**向量数据库**：
- 管理实体/关系 embedding，用于相似检索。

### 3.5 编排模块（Orchestration）

- 管理文档处理与检索流程的调度。
- 提供统一接口给上层应用调用。

## 4. 数据流与接口

### 4.1 文档处理接口

```
process_document(doc) -> {entities, relations, embeddings, chunks}
```

### 4.2 子图缓存接口

```
subgraph.add(entities, relations)
subgraph.search(query_embedding) -> candidates
subgraph.increment_frequency(entity_id)
subgraph.should_merge(entity_id) -> bool
```

### 4.3 图数据库接口

```
graph_db.merge(entities, relations)
graph_db.search(query_embedding) -> candidates
```

## 5. 技术选型建议

- **图数据库**：Neo4j / TigerGraph / NebulaGraph
- **向量数据库**：Milvus / Qdrant / Weaviate
- **内存缓存**：RedisGraph / 自研内存结构

## 6. 后续扩展

- 引入多级缓存（短期子图缓存 + 中期缓存）。
- 对不同实体类型使用不同阈值策略。
- 增加冲突合并策略（实体消歧、关系冲突处理）。
