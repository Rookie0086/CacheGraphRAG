# 文档处理模块

**目标**：对输入文档进行实体关系抽取与 embedding 生成，并记录来源 chunk。

## 预期能力

- 文本切块与预处理（清洗、去噪、分段）
- LLM 或规则抽取实体/关系
- 生成实体/关系 embedding
- 记录来源 chunk 与置信度

## 对外接口（示意）

```
process_document(doc) -> {
  entities,
  relations,
  embeddings,
  chunks
}
```

> 当前仅为功能说明，具体实现留待后续补充。
