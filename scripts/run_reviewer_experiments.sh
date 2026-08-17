#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
PYTHON="$REPO_ROOT/.conda/cachegraphrag-mac/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="${PYTHON_BIN:-python3}"
fi

cd "$REPO_ROOT"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$REPO_ROOT/output/.mplconfig}"
mkdir -p "$MPLCONFIGDIR"

if [[ -n "${CACHEGRAPH_MODEL_API_KEY:-}" ]]; then
  export CACHEGRAPH_EMBED_API_KEY="${CACHEGRAPH_EMBED_API_KEY:-$CACHEGRAPH_MODEL_API_KEY}"
fi

if [[ $# -eq 0 ]]; then
  cat <<'EOF'
Usage:
  bash scripts/run_reviewer_experiments.sh fairness --start 0 --end 100
  bash scripts/run_reviewer_experiments.sh locality lru_rehydrate --queries 100
  bash scripts/run_reviewer_experiments.sh --all --start 0 --end 600
  bash scripts/run_reviewer_experiments.sh --all --dry-run

Local port 8000 users should export CACHEGRAPH_MODEL_* or keep the protected
.secrets/local_llm_key setup described in scripts/experiments/README.md.
EOF
  exit 2
fi

exec "$PYTHON" scripts/experiments/run_experiments.py "$@"
