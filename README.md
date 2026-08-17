<div align="right">

[**中文**] · [**English**](README_EN.md)

</div>

# CacheGraphRAG

基于 **L1/L2 两级缓存** 的 Graph-RAG 框架。L1 用 NetworkX 做**容量受限的自净化内存热缓存**（LRU/TTL + 引用计数驱逐），L2 用 NebulaGraph 做持久化存储；访问频率驱动冷热数据晋升（h(c)≥τ_hit 异步固化到 L2）。**L1 默认空启动**：检索命中 chunk 时从 Milvus `graph_meta` 延迟重载（Lazy Rehydration）逐步填充至容量上限，被逐 chunk 的全文与拓扑仍保留在向量后备存储。

---

## 环境与部署

### 依赖

- Python >= 3.10
- Docker（Milvus + NebulaGraph）
- LLM / Embedding / Rerank 服务（云端 API 或本地 oMLX）

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

### 凭据注入（不硬编码 API Key）

`config/config.yaml` 与实验脚本**不含真实 API Key**。运行前通过环境变量注入：

```bash
export CACHEGRAPH_MODEL_API_KEY='sk-xxx'      # LLM(如 gptgod.cloud / OpenAI)
export CACHEGRAPH_EMBED_API_KEY='sk-xxx'      # Embedding(云端 siliconflow 时)
export CACHEGRAPH_RERANK_API_KEY='sk-xxx'     # Rerank(云端时)
```

本地 oMLX（127.0.0.1:8000）时，embedding/rerank 的 key 从 `.secrets/local_llm_key` 读取。

### 配置

编辑 `config/config.yaml`（关键段）：

```yaml
model:
  backend: openai            # 云 LLM(OpenAI 兼容)
  model_name: gpt-4o-mini
  api_key: ''                # 经 CACHEGRAPH_MODEL_API_KEY 注入
  base_url: https://gptgod.cloud/v1
  seed: 42                   # 全局随机种子(数据切分/采样可复现)

embedding:
  backend: api               # OpenAI 兼容嵌入服务
  model_name: bge-m3-mlx-4bit                    # 本地 oMLX
  base_url: http://127.0.0.1:8000/v1             # 或云端 https://api.siliconflow.cn/v1 + BAAI/bge-m3

rerank:
  backend: api               # api = /v1/rerank;local = transformers 本地加载
  model_name: bge-reranker-v2-m3
  base_url: http://127.0.0.1:8000/v1

data:
  dataset: wikimultihopqa    # 2WikiMultiHopQA(1000 条);也支持 hotpotqa/whoqa/rgb 等
  start: 0
  end: 600
  save_gexf: false           # 构建后不写 L1 拓扑快照(gexf 仅观察用,不计入存储审计)

retrieval:
  mode: hybrid               # hybrid=DPR+图;graph_only=纯图检索
  load_gexf: false           # L1 空启动:不加载 gexf,QA 期按 chunk 从 Milvus graph_meta 重载填充
  gamma: 0.5                 # 跳数衰减系数(权重=γ^hop)
  max_hops: 3                # 图遍历最大跳数
  beam_width: 4              # B:图检索束搜索宽度(0=不裁剪)
  alignment_scope: "l1"      # 实体对齐比对空间:l1=活跃节点集 V_L1;full=实体索引全量
  embedding_concurrency: 4   # 查询期嵌入并发门控
  rerank_concurrency: 2      # rerank 并发门控
  graph_entity_parallel: 4   # 实体级图检索并行度

hyperparameters:             # 论文符号 ↔ 配置(见 rebuttal R2-6.4)
  C_max: 200                 # L1 容量上限(indexing.l1_max_chunks)
  tau_hit: 3                 # L2 晋升阈值(indexing.promotion_threshold)
  tau_sim: 0.85              # 实体强对齐阈值
  tau_desc: 0.6              # 实体弱语义阈值
  B: 4                       # 束宽
  gamma: 0.5                 # 跳数衰减
```

---

## 使用示例

### 完整流程（索引 + 检索 + 回答）

```python
import asyncio
from src.CacheGraphRAG import CacheGraphRAG

app = CacheGraphRAG(
    dataset="wikimultihopqa",
    l1_max_chunks=200,
    promotion_threshold=3,
)

async def main():
    await app.index(start=0, end=300)      # 建索引 → Milvus 向量/graph_meta + 实体索引(L1 空启动)
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

app = CacheGraphRAG(dataset="wikimultihopqa")
asyncio.run(app.index(start=0, end=300))
app.shutdown()
```

### 仅检索（加载已有索引；L1 空启动 + 预热/测试隔离）

```python
import asyncio
from src.CacheGraphRAG import CacheGraphRAG

app = CacheGraphRAG(dataset="wikimultihopqa")
asyncio.run(app.index_only_qa(
    start=0, end=300,
    mode="hybrid", answer_topk=6,
    warmup_ratio=0.15, warmup_seed=42,   # 预热/测试严格隔离(disjoint),回应 R4-W2
    clear_l2=True,                        # QA 前清空 L2(冷启动)
))
app.shutdown()
```

### Agentic 多步检索

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

Agentic 模式支持三层并发控制：`retrieval.qa_concurrency` 控制问题间并发，
`retrieval.agentic_parallelism` 控制同一问题内 beam 状态并发，
`model.max_concurrency` 是二者共享的 LLM 在途请求硬上限（sync/async 统一门控）。
同一 beam 层的子问题规划默认合并为一次请求（`retrieval.agentic_batch_planning: true`）。
Apple Silicon/MLX 建议三者从 `2/2/2` 起步；vLLM 或托管 API 可逐步提高全局上限。

### 纯图检索（无向量检索）

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

### 命令行运行

配置在 `config/config.yaml` 中设置（`retrieval.index_only` / `skip_index` 控制阶段），
运行时注入凭据：

```bash
export CACHEGRAPH_MODEL_API_KEY='sk-xxx'
export CACHEGRAPH_EMBED_API_KEY='sk-xxx'

# 完整流程
python -m src.CacheGraphRAG

# 仅索引 (config.yaml: index_only = true)
python -m src.CacheGraphRAG

# 仅检索 (config.yaml: skip_index = true)
python -m src.CacheGraphRAG
```

也可用 `CACHEGRAPH_CONFIG` 指向自定义配置文件。

---

## 实验与验证

```bash
# 回归验证(全部单元/验证脚本)
bash scripts/verify_macos_env.sh

# 一键复现 2Wiki 全流程(建索引 → 预热/测试隔离 QA → 存储报告 → 审计)
bash scripts/repro_wikimultihopqa.sh

# 审稿人实验套件(fairness=泄漏对照 / lru_rehydrate=容量消融 / latency_cost=时延)
bash scripts/run_local_env_experiments.sh fairness --start 0 --end 100 --qa-concurrency 5
bash scripts/run_local_env_experiments.sh lru_rehydrate --start 0 --end 100 --queries 100
bash scripts/run_local_env_experiments.sh latency_cost --start 0 --end 100 --queries 100

# 评估 QA 结果(ACC/Rouge-L/BERTScore,兼容外部 Rouge-LBERTScore.py 输出)
python scripts/eval_rouge_bertscore.py --qa_result output/experiments/.../qa.json
```

---

## 项目结构

```
CacheGraphRAG/
├── src/
│   ├── CacheGraphRAG.py         # 主入口
│   ├── pipeline.py              # 文档入库管线(异步抽取+摄入)
│   ├── graph/memory_graph.py    # L1 内存图(LRU/TTL/引用计数) + L2 NebulaGraph
│   ├── retrieval/
│   │   ├── retriever.py         # 混合检索器(γ 衰减束搜索 + 延迟重载)
│   │   ├── fusion.py            # 融合策略(RRF 等)
│   │   ├── reranker.py          # 重排序(API / 本地 transformers)
│   │   └── agentic_engine.py    # Agentic 检索(束搜索 + 语义循环阻断)
│   ├── entity/resolver.py       # 免 LLM 漏斗式实体对齐(V_L1 比对空间)
│   └── llm/env.py               # LLM/嵌入环境封装(统一并发门控)
├── database/                    # Milvus + NebulaGraph 客户端
├── data/                        # 数据集加载器
├── scripts/                     # 复现/实验/评估/验证脚本
├── tests/                       # 验证测试(verify_*.py)
├── config/config.yaml           # 全局配置
├── acc.py                       # checkanswer 精度判定(评估用)
├── Rouge-LBERTScore.py          # 外部评估脚本(依赖服务器路径,本地用 scripts/eval_rouge_bertscore.py)
└── requirements.txt
```
