# CacheGraphRAG

Graph-RAG framework with **L1/L2 two-tier cache**. L1 uses NetworkX as in-memory hot cache, L2 uses NebulaGraph for persistent storage. Access frequency drives cold/hot data promotion.

---

## Setup & Deployment

### Prerequisites

- Python >= 3.10
- Docker (Milvus + NebulaGraph)

### Installation

```bash
git clone https://github.com/your-username/CacheGraphRAG.git
cd CacheGraphRAG

# Create conda environment
conda create -n cachegraphrag python=3.10 -y
conda activate cachegraphrag

pip install -r requirements.txt
```

### Start Databases

```bash
# Milvus vector database
bash database/setup/milvus-install-user.sh start

# NebulaGraph graph database
bash database/setup/nebula-install-user.sh install
```

### Configuration

Edit `config/config.yaml`:

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

## Usage Examples

### Full Pipeline (Index + Retrieve + Answer)

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

### Index Only

```python
import asyncio
from src.CacheGraphRAG import CacheGraphRAG

app = CacheGraphRAG(dataset="hotpotqa")
asyncio.run(app.index(start=0, end=300))
app.shutdown()
```

### Retrieve Only (with existing index)

```python
import asyncio
from src.CacheGraphRAG import CacheGraphRAG

app = CacheGraphRAG(dataset="hotpotqa")
asyncio.run(app.index_only_qa(start=0, end=300, mode="hybrid", answer_topk=6))
app.shutdown()
```

### Agentic Multi-Step Retrieval

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

### Graph-Only Retrieval (no vector search)

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

### CLI

Set all options in `config/config.yaml` and run:

```bash
# Full pipeline
python -m src.CacheGraphRAG

# Index only (config.yaml: index_only = true)
python -m src.CacheGraphRAG

# Retrieve only (config.yaml: skip_index = true)
python -m src.CacheGraphRAG
```

---

## Project Structure

```
CacheGraphRAG/
├── src/
│   ├── CacheGraphRAG.py         # Main entry
│   ├── pipeline.py              # Document indexing pipeline
│   ├── graph/memory_graph.py    # L1 memory graph + L2 NebulaGraph
│   ├── retrieval/
│   │   ├── retriever.py         # Hybrid retriever
│   │   ├── fusion.py            # Fusion strategies
│   │   ├── reranker.py          # Reranking
│   │   └── agentic_engine.py    # Agentic retrieval
│   ├── entity/resolver.py       # Entity alignment
│   └── llm/env.py               # LLM environment wrapper
├── database/                    # Milvus + NebulaGraph
├── data/                        # Dataset loaders
├── config/config.yaml           # Global config
└── requirements.txt
```
