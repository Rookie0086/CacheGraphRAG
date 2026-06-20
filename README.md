<div align="right">

[**中文**] · [**English**](README_EN.md)

</div>

# CacheGraphRAG

基于 **L1/L2 两级缓存** 的 Graph-RAG 框架。L1 用 NetworkX 做内存热缓存，L2 用 NebulaGraph 做持久化存储，访问频率驱动冷热数据晋升。

---

## 环境与部署

### 依赖

- Python >= 3.10
- Docker（Milvus + NebulaGraph）

### 安装

```bash
git clone https://github.com/your-username/CacheGraphRAG.git
cd CacheGraphRAG

# 创建 conda 环境
conda create -n cachegraphrag python=3.10 -y
conda activate cachegraphrag

pip install -r requirements.txt
```

### 启动数据库

```bash
# Milvus 向量数据库
bash database/setup/milvus-install-user.sh start

# NebulaGraph 图数据库
bash database/setup/nebula-install-user.sh install
```

### 配置

编辑 `config/config.yaml`：

```yaml
model:
  backend: openai
  model_name: gpt-4o-mini
  api_key: sk-xxx
  base_url: https://api.openai.com/v1

embedding:
  backend: api
  model_name: BAAI/bge-m3
  api_key: sk-xxx
  base_url: https://api.siliconflow.cn/v1

data:
  dataset: hotpotqa
  start: 0
  end: 600

retrieval:
  mode: hybrid
  agentic: false
```

---

## 使用示例

### 完整流程（索引 + 检索 + 回答）

```python
import asyncio
from src.CacheGraphRAG import CacheGraphRAG

app = CacheGraphRAG(
    dataset="hotpotqa",
    l1_max_chunks=100,
    promotion_threshold=3,
)

async def main():
    await app.index(start=0, end=300)
    app.load_dataset(start=0, end=300)
    results = await app.query(
        questions=app.questions[:5],
        start=0, end=5,
        mode="hybrid",
        answer_topk=6,
    )
    for r in results:
        print(f"Q: {r['query']}\nA: {r['predict']}\n")
    app.shutdown()

asyncio.run(main())
```

### 仅索引

```python
import asyncio
from src.CacheGraphRAG import CacheGraphRAG

app = CacheGraphRAG(dataset="hotpotqa")
asyncio.run(app.index(start=0, end=300))
app.shutdown()
```

### 仅检索（加载已有索引）

```python
import asyncio
from src.CacheGraphRAG import CacheGraphRAG

app = CacheGraphRAG(dataset="hotpotqa")
asyncio.run(app.index_only_qa(start=0, end=300, mode="hybrid", answer_topk=6))
app.shutdown()
```

### Agentic 多步检索

```python
import asyncio
from src.CacheGraphRAG import CacheGraphRAG

app = CacheGraphRAG(dataset="hotpotqa")
asyncio.run(app.index_only_qa(
    start=0, end=300,
    use_agentic=True, agentic_steps=3,
    answer_topk=6,
))
app.shutdown()
```

### 纯图检索（无向量检索）

```python
import asyncio
from src.CacheGraphRAG import CacheGraphRAG

app = CacheGraphRAG(dataset="hotpotqa")
asyncio.run(app.index_only_qa(
    start=0, end=300,
    mode="graph_only",
    entity_extraction="llm",
    answer_topk=6,
))
app.shutdown()
```

### 命令行运行

所有配置在 `config/config.yaml` 中设置后直接运行：

```bash
# 完整流程
python -m src.CacheGraphRAG

# 仅索引 (config.yaml: index_only = true)
python -m src.CacheGraphRAG

# 仅检索 (config.yaml: skip_index = true)
python -m src.CacheGraphRAG
```

---

## 项目结构

```
CacheGraphRAG/
├── src/
│   ├── CacheGraphRAG.py         # 主入口
│   ├── pipeline.py              # 文档入库管线
│   ├── graph/memory_graph.py    # L1 内存图 + L2 NebulaGraph
│   ├── retrieval/
│   │   ├── retriever.py         # 混合检索器
│   │   ├── fusion.py            # 融合策略
│   │   ├── reranker.py          # 重排序
│   │   └── agentic_engine.py    # Agentic 检索
│   ├── entity/resolver.py       # 实体对齐
│   └── llm/env.py               # LLM 环境封装
├── database/                    # Milvus + NebulaGraph
├── data/                        # 数据集加载器
├── config/config.yaml           # 全局配置
└── requirements.txt
```
