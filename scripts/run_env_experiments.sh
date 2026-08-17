#!/usr/bin/env bash
# 云端配置运行实验套件(LLM=gptgod.cloud, embedding/rerank=siliconflow)。
#
# 用法:
#   bash scripts/run_env_experiments.sh latency_cost --start 0 --end 2 --queries 2
#   bash scripts/run_env_experiments.sh --all --start 0 --end 600
#
# ⚠️ 凭据不硬编码:运行前 export CACHEGRAPH_MODEL_API_KEY / CACHEGRAPH_EMBED_API_KEY /
#   CACHEGRAPH_RERANK_API_KEY(或由调用方注入)。实验快照(config.yaml)由
#   redact_credentials() 自动清空 api_key,不会外泄。
#
# 说明:embedding/rerank 的 base_url 用裸的 https://api.siliconflow.cn/v1,
# 代码会自行拼接 /embeddings 与 /rerank(见 src/llm/env.py:154、src/retrieval/reranker.py:36)。
set -euo pipefail

# ---- LLM: gptgod.cloud (gpt-4o-mini);key 由调用方注入 ----
export CACHEGRAPH_MODEL_BACKEND=openai
export CACHEGRAPH_MODEL_NAME=gpt-4o-mini
export CACHEGRAPH_MODEL_BASE_URL=https://gptgod.cloud/v1
export CACHEGRAPH_MODEL_API_KEY="${CACHEGRAPH_MODEL_API_KEY:-}"

# ---- Embedding: siliconflow (BAAI/bge-m3) ----
export CACHEGRAPH_EMBED_BACKEND=api
export CACHEGRAPH_EMBED_NAME=BAAI/bge-m3
export CACHEGRAPH_EMBED_BASE_URL=https://api.siliconflow.cn/v1
export CACHEGRAPH_EMBED_API_KEY="${CACHEGRAPH_EMBED_API_KEY:-}"

# ---- Rerank: siliconflow (BAAI/bge-reranker-v2-m3) ----
export CACHEGRAPH_RERANK_BACKEND=api
export CACHEGRAPH_RERANK_NAME=BAAI/bge-reranker-v2-m3
export CACHEGRAPH_RERANK_BASE_URL=https://api.siliconflow.cn/v1
export CACHEGRAPH_RERANK_API_KEY="${CACHEGRAPH_RERANK_API_KEY:-}"

# 透传给 run_reviewer_experiments.sh(后者会优先选择 experiments 解释器)
exec bash scripts/run_reviewer_experiments.sh "$@"
