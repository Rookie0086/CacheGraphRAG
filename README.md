这是一个为您精心设计的 `README.md` 文件。它整合了项目的核心理念（DepCache思想）、技术架构（NetworkX + NebulaGraph + Milvus）、实体对齐策略以及实验流程。

你可以直接将以下内容复制到你 GitHub 项目的 `README.md` 文件中。

---

# CacheGraphRAG 🚀

**CacheGraphRAG** 是一个基于 **"Write-Back"（写回策略）** 的下一代 Graph-RAG 框架。

它旨在解决传统 Graph-RAG 方法在知识图谱构建过程中面临的**数据污染（Graph Pollution）和低效写入**问题。通过引入类似计算机体系结构中的 "Cache" 机制，本框架实现了**冷热数据分离**：仅将高频访问或高置信度的实体及其关系“晋升”至持久化图数据库（NebulaGraph），而将大量的长尾、一次性或噪声数据保留在轻量级内存图（NetworkX）中。

---

## 📖 背景与动机 (Motivation)

传统的 Graph-RAG 流程通常是：`文档切片 -> 实体抽取 -> 立即写入图数据库`。这种“急切写入”策略带来了显著问题：

1. **噪声爆炸**：LLM 抽取的实体中包含大量无意义的噪声（如“这个”、“某人”），污染了图谱。
2. **写入瓶颈**：频繁的图数据库写入操作导致性能下降。
3. **检索断裂**：缺乏有效的实体对齐机制，导致 "Elon Musk" 和 "E. Musk" 被存储为两个节点。

**CacheGraphRAG** 参考了 **DepCache** 的思想，引入由 `NetworkX` 构建的 **Staging Graph（暂存图）**。实体首先驻留在内存中，只有当其**访问频率（Access Frequency）**达到预设阈值时，才会被异步合并入持久化主图。

---

## 🌟 核心特性 (Key Features)

* **⚡ 基于频率的写回机制 (Frequency-Based Write-Back)**: 只有“热”实体才会进入 NebulaGraph，大幅降低存储成本并提高检索质量。
* **🔗 向量化实体对齐 (Vector-Based Entity Alignment)**: 利用 **Milvus** 存储实体名称向量，通过语义相似度自动合并不同表述的同一实体（如 "Bill Gates" vs "Gates"），避免图谱分裂。
* **🔍 混合检索视图 (Hybrid Retrieval View)**: 查询时动态聚合 `Vector Store` + `Memory Subgraph` + `Persistent Graph` 的信息，确保回答的全面性。
* **🧠 结构化抽取约束**: 采用深度优化的 Prompt 与 JSON Mode，确保 LLM 产出的三元组精准可控。

---

## 🏗️ 系统架构 (Architecture)

### 数据流转逻辑

1. **Ingestion (入库)**:
* LLM 抽取实体关系 -> **Entity Resolver** (Milvus 查重) -> 存入 **Memory Graph** (NetworkX) -> 初始访问计数=0。


2. **Retrieval (检索)**:
* User Query -> **Vector Search** (定位相关 Chunk) -> 提取关键实体。
* 在 Memory Graph 中查找实体 -> **访问计数 +1**。
* **Promotion Check**: 若 `Count >= Threshold` -> 触发异步任务，将该实体子图写入 **NebulaGraph**。
* **Context Assembly**: 聚合 Memory 和 NebulaGraph 中的邻居节点 -> 生成回答。



---

## 🛠️ 技术栈 (Tech Stack)

| 组件 | 技术选型 | 作用 |
| --- | --- | --- |
| **LLM** | GPT-4o / DeepSeek-V3 | 实体抽取、最终问答生成 |
| **Memory Graph** | **NetworkX** (+ Redis) | 内存暂存区，处理高并发读写与计数 |
| **Persistent Graph** | **NebulaGraph** | 存储经过验证的、高质量的“热”知识图谱 |
| **Vector Database** | **Milvus** | 存储文本 Chunk 向量与实体名称向量（用于对齐） |
| **Embedding** | OpenAI / BGE-M3 | 文本与实体的向量化 |

---

## 📂 项目结构 (Structure)

```text
CacheGraphRAG/
├── config/                 # 配置文件 (阈值、数据库连接)
├── data/                   # 实验数据集 (Experiment Stream)
├── src/
│   ├── core/
│   │   ├── entity_resolver.py   # 基于Milvus的实体对齐核心
│   │   ├── memory_graph.py      # NetworkX 内存图与计数逻辑
│   │   └── graph_store.py       # NebulaGraph 交互接口
│   ├── ingestion/
│   │   ├── extractor.py         # LLM 抽取 Prompt 与 Pipeline
│   │   └── processing.py        # 文本切块与清洗
│   ├── retrieval/
│   │   └── searcher.py          # 混合检索器 (Vector + Graph)
│   └── pipeline/
│       └── manager.py           # 主调度器
├── tests/                  # 单元测试
├── main.py                 # 启动入口
└── requirements.txt

```

---

## 🚀 快速开始 (Quick Start)

### 1. 环境准备

需要 Python 3.10+，并确保已部署 NebulaGraph 和 Milvus（可使用 Docker）。

```bash
# 克隆仓库
git clone https://github.com/YourUsername/CacheGraphRAG.git
cd CacheGraphRAG

# 安装依赖
pip install -r requirements.txt

```

### 2. 配置 (`config/settings.yaml`)

```yaml
graph:
  promotion_threshold: 3  # 实体被访问 3 次后写入 Nebula
  ttl_seconds: 86400      # 内存中无用实体的存活时间

nebula:
  address: "127.0.0.1:9669"
  user: "root"
  password: "password"

milvus:
  uri: "http://localhost:19530"
  collection_entity: "entity_index"

```

### 3. 运行实验 (Experiment)

我们提供了一个微型数据集（模拟科技新闻流）来验证“从内存到磁盘”的晋升过程。

```bash
python main.py --mode experiment

```

**预期输出：**

```text
[Step 1] Processing doc_1... Entities stored in Memory (NetworkX).
[Step 2] Processing doc_2... Entities aligned via Milvus.
...
[Step 3] Query: "Who is leading the NebulaX project?"
[System] Entity 'NebulaX' access count reached 3.
[System] 🚀 PROMOTING 'NebulaX' from Memory to NebulaGraph!

```

---

## 💡 核心实现细节

### 实体对齐 (Entity Resolution)

为了解决实体歧义，我们使用 Milvus 维护一个 `Entity Registry`。

```python
# 伪代码逻辑
vector = embed(new_entity_name)
hits = milvus.search(vector, threshold=0.92)

if hits:
    # 认为是同一实体，复用现有 ID (无论在内存还是 Nebula)
    entity_id = hits[0].id
else:
    # 新实体，注册到 Milvus 并存入 NetworkX
    entity_id = create_new_id()

```

### 抽取 Prompt

我们使用 Strict JSON Mode 配合思维链（CoT）来保证抽取质量：

```json
{
  "entities": [{"id": "NebulaX", "type": "PRODUCT", "description": "AI chip..."}],
  "relationships": [{"source": "TechCorp", "target": "NebulaX", "type": "DEVELOPED"}]
}

```

---

## 🤝 贡献 (Contributing)

欢迎提交 Pull Request！目前的 Roadmap 包括：

* [ ] 支持基于 LRU 的内存淘汰策略。
* [ ] 增加可视化面板，实时监控内存图到主图的流动。
* [ ] 支持更多 LLM 后端（Ollama, vLLM）。

## 📄 License

Apache 2.0 License