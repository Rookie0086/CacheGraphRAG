#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MAC_PY="$REPO_ROOT/.conda/cachegraphrag-mac/bin/python"

if [[ ! -x "$MAC_PY" ]]; then
  echo "macOS environment not found: $MAC_PY" >&2
  echo "Create it with: conda create -n cachegraphrag python=3.10 && conda activate cachegraphrag && pip install -r requirements.txt" >&2
  exit 1
fi

cd "$REPO_ROOT"
"$MAC_PY" -c 'import platform, torch, pymilvus; print(f"Python {platform.python_version()} ({platform.machine()})"); print(f"torch={torch.__version__}, MPS built={torch.backends.mps.is_built()}, available={torch.backends.mps.is_available()}"); print(f"pymilvus={pymilvus.__version__}")'
"$MAC_PY" -c 'import grpc; from database.milvus import MilvusDB, myMilvus; from src.CacheGraphRAG import CacheGraphRAG; print(f"grpc={grpc.__version__}; CacheGraphRAG import OK")'

# Legacy verify scripts execute at import time, so isolate every file in its own process.
"$MAC_PY" tests/verify_agentic_block.py
"$MAC_PY" tests/verify_gamma.py
"$MAC_PY" tests/verify_warmup_split.py
"$MAC_PY" tests/verify_token_usage.py
"$MAC_PY" tests/verify_beam_retriever.py
"$MAC_PY" tests/verify_alignment_thresholds.py
"$MAC_PY" tests/verify_alignment_scope.py
"$MAC_PY" tests/verify_l2_async_write.py
"$MAC_PY" tests/verify_llm_unified_gate.py
"$MAC_PY" tests/verify_nebula_session_lock.py
"$MAC_PY" tests/verify_m1_dedicated_executors.py
"$MAC_PY" tests/verify_m2_concurrency_gates.py
"$MAC_PY" tests/verify_m4_snapshot_read.py
"$MAC_PY" tests/verify_l1_cold_start.py
"$MAC_PY" -m unittest \
  tests.verify_beam_search \
  tests.verify_agentic_parallel \
  tests.verify_llm_concurrency \
  tests.verify_nebula_clone \
  tests.verify_topology_rehydrate \
  tests.verify_embedding_batcher

"$MAC_PY" -m py_compile \
  database/milvus.py src/CacheGraphRAG.py src/pipeline.py \
  src/llm/env.py \
  scripts/experiments/nebula_clone.py scripts/experiments/run_experiments.py \
  scripts/experiments/protocol_runner.py \
  src/retrieval/retriever.py src/retrieval/agentic_engine.py

echo "macOS environment verification: PASS"
