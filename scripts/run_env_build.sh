#!/usr/bin/env bash
# 用 .env 云端配置驱动 repro 脚本(LLM=gptgod.cloud, embedding/rerank=siliconflow)。
#
# 用法:
#   bash scripts/run_env_build.sh --phase build --start 0 --end 30 --skip-db-check   # 重建索引
#   bash scripts/run_env_build.sh --phase qa    --start 0 --end 30 --skip-db-check   # 预热/测试隔离 QA
#   bash scripts/run_env_build.sh --dry-run
#
# ⚠️ 凭据不硬编码,运行前 export CACHEGRAPH_*_API_KEY(与 /Users/Zhuanz1/Documents/论文/.env 一致),请勿提交到 git。
#
# 说明:
#   - repro 脚本内部用 `python`/`python3` 命令,这里把 experiments 解释器前置到 PATH,
#     避免解析到系统 miniconda/base(缺项目依赖)。
#   - embedding/rerank 的 base_url 用裸的 https://api.siliconflow.cn/v1,代码会自行拼接
#     /embeddings 与 /rerank(见 src/llm/env.py:154、src/retrieval/reranker.py:36)。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

# 前置 experiments 解释器(3.10, 含 pymilvus/nebula3 等依赖)
export PATH="$REPO_ROOT/.conda/cachegraphrag-mac/bin:$PATH"

# ---- LLM: gptgod.cloud (gpt-4o-mini) ----
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

exec bash scripts/repro_wikimultihopqa.sh "$@"
