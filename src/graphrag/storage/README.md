# 存储模块

**目标**：提供子图缓存、图数据库、向量数据库的统一访问接口。

## 预期能力

- 子图缓存（内存）
  - 添加实体/关系
  - 相似检索
  - 访问频次统计
  - 合并阈值判断
- 图数据库接口
  - 实体/关系持久化
  - 相似检索
- 向量数据库接口
  - embedding 写入与检索

## 对外接口（示意）

```
subgraph.add(...)
subgraph.search(...)
subgraph.increment_frequency(...)
subgraph.should_merge(...)

graph_db.merge(...)
graph_db.search(...)

vector_db.upsert(...)
vector_db.search(...)
```

> 当前仅为功能说明，具体实现留待后续补充。
