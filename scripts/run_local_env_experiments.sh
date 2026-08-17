#!/usr/bin/env bash
# 本地 embedding/rerank + 云 LLM 实验入口(siliconflow embedding/rerank 402 欠费期间替代方案):
#   LLM      = gptgod.cloud gpt-4o-mini(云,可用)
#   Embedding= 本地 bge-m3-mlx-4bit(127.0.0.1:8000,.secrets/local_llm_key,1024 维)
#   Rerank   = 本地 bge-reranker-v2-m3(127.0.0.1:8000/v1/rerank,oMLX)
#
# 用法:
#   bash scripts/run_local_env_experiments.sh fairness --start 0 --end 100 --qa-concurrency 2
#   bash scripts/run_local_env_experiments.sh lru_rehydrate --start 0 --end 600 --queries 100
#
# ⚠️ 凭据不硬编码:运行前 export CACHEGRAPH_MODEL_API_KEY(或由调用方注入),
#   本地 embedding/rerank key 从 .secrets/local_llm_key 读取。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# ---- LLM: gptgod.cloud (gpt-4o-mini);key 由调用方注入,防泄露 ----
export CACHEGRAPH_MODEL_BACKEND=openai
export CACHEGRAPH_MODEL_NAME=gpt-4o-mini
export CACHEGRAPH_MODEL_BASE_URL=https://gptgod.cloud/v1
export CACHEGRAPH_MODEL_API_KEY="${CACHEGRAPH_MODEL_API_KEY:-}"

# ---- Embedding/Rerank: 本地 oMLX 8000 端口(模型名/base_url 已由 config 指向本地) ----
LOCAL_KEY="$(cat "$REPO_ROOT/.secrets/local_llm_key" 2>/dev/null || true)"
export CACHEGRAPH_EMBED_API_KEY="$LOCAL_KEY"
export CACHEGRAPH_RERANK_API_KEY="$LOCAL_KEY"
export CACHEGRAPH_RERANK_BACKEND=api   # oMLX /v1/rerank(模型 bge-reranker-v2-m3)

# 本地 MLX 并发敏感:建议显式 --qa-concurrency 2(默认 5 可能 502)
exec bash "$SCRIPT_DIR/run_reviewer_experiments.sh" "$@"
