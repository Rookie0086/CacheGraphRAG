<div align="right">

[**中文**](README.md) · [**English**]

</div>

# CacheGraphRAG

Graph-RAG framework with an **L1/L2 two-tier cache**. L1 is a **capacity-bounded, self-purifying in-memory hot cache** (NetworkX with LRU/TTL + reference-counting eviction), L2 is a persistent NebulaGraph store; access frequency drives cold/hot promotion (subgraphs are asynchronously persisted to L2 once h(c)≥τ_hit). **L1 starts empty by default**: when a query hits a chunk, its topology is lazily rehydrated from the Milvus `graph_meta` backup until the capacity limit is reached; evicted chunks keep their full text and topology in the vector backup store.

---

## Setup & Deployment

### Prerequisites

- Python >= 3.10
- Docker (Milvus + NebulaGraph)
- LLM / Embedding / Rerank services (cloud API or local oMLX)

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

### Credentials (no hard-coded API keys)

`config/config.yaml` and the experiment scripts contain **no real API keys**. Inject them via environment variables before running:

```bash
export CACHEGRAPH_MODEL_API_KEY='sk-xxx'      # LLM (e.g. gptgod.cloud / OpenAI)
export CACHEGRAPH_EMBED_API_KEY='sk-xxx'      # Embedding (cloud siliconflow)
export CACHEGRAPH_RERANK_API_KEY='sk-xxx'     # Rerank (cloud)
```

For local oMLX (127.0.0.1:8000), embedding/rerank keys are read from `.secrets/local_llm_key`.

### Configuration

Edit `config/config.yaml` (key sections):

```yaml
model:
  backend: openai            # Cloud LLM (OpenAI-compatible)
  model_name: gpt-4o-mini
  api_key: ''                # Injected via CACHEGRAPH_MODEL_API_KEY
  base_url: https://gptgod.cloud/v1
  seed: 42                   # Global seed (reproducible splits/sampling)

embedding:
  backend: api               # OpenAI-compatible embedding service
  model_name: bge-m3-mlx-4bit                    # Local oMLX
  base_url: http://127.0.0.1:8000/v1             # or cloud https://api.siliconflow.cn/v1 + BAAI/bge-m3

rerank:
  backend: api               # api = /v1/rerank;local = transformers on-device
  model_name: bge-reranker-v2-m3
  base_url: http://127.0.0.1:8000/v1

data:
  dataset: wikimultihopqa    # 2WikiMultiHopQA (1000 samples); hotpotqa/whoqa/rgb also supported
  start: 0
  end: 600
  save_gexf: false           # Do not write L1 topology snapshot (gexf is observation-only, not storage)

retrieval:
  mode: hybrid               # hybrid = DPR+graph; graph_only = pure graph
  load_gexf: false           # L1 cold start: no gexf preload; chunks rehydrated from Milvus graph_meta on demand
  gamma: 0.5                 # Hop-decay factor (weight = γ^hop)
  max_hops: 3                # Max graph traversal hops
  beam_width: 4              # B: beam-search width in graph retrieval (0 = no pruning)
  alignment_scope: "l1"      # Entity alignment scope: l1 = active node set V_L1; full = whole entity index
  embedding_concurrency: 4   # Query-time embedding concurrency gate
  rerank_concurrency: 2      # Rerank concurrency gate
  graph_entity_parallel: 4   # Per-entity graph retrieval parallelism

hyperparameters:             # Paper symbols ↔ config (see rebuttal R2-6.4)
  C_max: 200                 # L1 capacity limit (indexing.l1_max_chunks)
  tau_hit: 3                 # L2 promotion threshold (indexing.promotion_threshold)
  tau_sim: 0.85              # Strong entity alignment threshold
  tau_desc: 0.6              # Weak semantic alignment threshold
  B: 4                       # Beam width
  gamma: 0.5                 # Hop decay
```

---

## Usage Examples

### Full Pipeline (Index + Retrieve + Answer)

```python
import asyncio
from src.CacheGraphRAG import CacheGraphRAG

app = CacheGraphRAG(
    dataset="wikimultihopqa",
    l1_max_chunks=200,
    promotion_threshold=3,
)

async def main():
    await app.index(start=0, end=300)      # Build index → Milvus vectors/graph_meta + entity index (L1 empty start)
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

app = CacheGraphRAG(dataset="wikimultihopqa")
asyncio.run(app.index(start=0, end=300))
app.shutdown()
```

### Retrieve Only (with existing index; L1 cold start + warmup/test isolation)

```python
import asyncio
from src.CacheGraphRAG import CacheGraphRAG

app = CacheGraphRAG(dataset="wikimultihopqa")
asyncio.run(app.index_only_qa(
    start=0, end=300,
    mode="hybrid", answer_topk=6,
    warmup_ratio=0.15, warmup_seed=42,   # Strict warmup/test isolation (disjoint), addresses R4-W2
    clear_l2=True,                        # Clear L2 before QA (cold start)
))
app.shutdown()
```

### Agentic Multi-Step Retrieval

```python
import asyncio
from src.CacheGraphRAG import CacheGraphRAG

app = CacheGraphRAG(dataset="wikimultihopqa")
asyncio.run(app.index_only_qa(
    start=0, end=300,
    use_agentic=True, agentic_steps=3,
    answer_topk=6,
))
app.shutdown()
```

Agentic mode has three levels of concurrency control: `retrieval.qa_concurrency` bounds
concurrent questions, `retrieval.agentic_parallelism` bounds beam states within a question,
and `model.max_concurrency` is the shared hard cap on in-flight LLM requests (unified
sync/async gate). Planning of sub-questions in the same beam layer is batched into one
request by default (`retrieval.agentic_batch_planning: true`). On Apple Silicon/MLX start
from `2/2/2`; for vLLM or hosted APIs raise the global cap based on server throughput.

### Graph-Only Retrieval (no vector search)

```python
import asyncio
from src.CacheGraphRAG import CacheGraphRAG

app = CacheGraphRAG(dataset="wikimultihopqa")
asyncio.run(app.index_only_qa(
    start=0, end=300,
    mode="graph_only",
    entity_extraction="llm",
    answer_topk=6,
))
app.shutdown()
```

### CLI

Set options in `config/config.yaml` (phase via `retrieval.index_only` / `skip_index`),
inject credentials, and run:

```bash
export CACHEGRAPH_MODEL_API_KEY='sk-xxx'
export CACHEGRAPH_EMBED_API_KEY='sk-xxx'

# Full pipeline
python -m src.CacheGraphRAG

# Index only (config.yaml: index_only = true)
python -m src.CacheGraphRAG

# Retrieve only (config.yaml: skip_index = true)
python -m src.CacheGraphRAG
```

A custom config can be selected via the `CACHEGRAPH_CONFIG` environment variable.

---

## Experiments & Verification

```bash
# Regression verification (all unit/verify scripts)
bash scripts/verify_macos_env.sh

# One-command 2Wiki reproduction (index → warmup/test-isolated QA → storage report → audit)
bash scripts/repro_wikimultihopqa.sh

# Reviewer experiment suite (fairness=leakage check / lru_rehydrate=capacity ablation / latency_cost=latency)
bash scripts/run_local_env_experiments.sh fairness --start 0 --end 100 --qa-concurrency 5
bash scripts/run_local_env_experiments.sh lru_rehydrate --start 0 --end 100 --queries 100
bash scripts/run_local_env_experiments.sh latency_cost --start 0 --end 100 --queries 100

# Evaluate QA results (ACC/Rouge-L/BERTScore, output compatible with external Rouge-LBERTScore.py)
python scripts/eval_rouge_bertscore.py --qa_result output/experiments/.../qa.json
```

---

## Project Structure

```
CacheGraphRAG/
├── src/
│   ├── CacheGraphRAG.py         # Main entry
│   ├── pipeline.py              # Document indexing pipeline (async extract + ingest)
│   ├── graph/memory_graph.py    # L1 memory graph (LRU/TTL/ref-count) + L2 NebulaGraph
│   ├── retrieval/
│   │   ├── retriever.py         # Hybrid retriever (γ-decay beam search + lazy rehydration)
│   │   ├── fusion.py            # Fusion strategies (RRF etc.)
│   │   ├── reranker.py          # Reranking (API / local transformers)
│   │   └── agentic_engine.py    # Agentic retrieval (beam search + semantic loop-breaking)
│   ├── entity/resolver.py       # LLM-free funnel entity alignment (V_L1 scope)
│   └── llm/env.py               # LLM/embedding wrapper (unified concurrency gate)
├── database/                    # Milvus + NebulaGraph clients
├── data/                        # Dataset loaders
├── scripts/                     # Reproduction / experiment / evaluation / verification scripts
├── tests/                       # Verification tests (verify_*.py)
├── config/config.yaml           # Global config
├── acc.py                       # checkanswer accuracy judgment (for evaluation)
├── Rouge-LBERTScore.py          # External eval script (needs server paths; local: scripts/eval_rouge_bertscore.py)
└── requirements.txt
```
