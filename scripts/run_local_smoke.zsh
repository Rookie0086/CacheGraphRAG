#!/bin/zsh
set -euo pipefail

repo_root="${0:A:h:h}"
key_file="$repo_root/.secrets/local_llm_key"
if [[ ! -r "$key_file" ]]; then
  echo "Missing protected key file: $key_file" >&2
  exit 1
fi

export PATH="$repo_root/.conda/cachegraphrag-mac/bin:$PATH"
export CACHEGRAPH_MODEL_BACKEND="openai"
export CACHEGRAPH_MODEL_NAME="Llama-3.1-8B-Instruct"
export CACHEGRAPH_MODEL_BASE_URL="http://127.0.0.1:8000/v1"
export CACHEGRAPH_MODEL_API_KEY="$(<"$key_file")"

start="${1:-0}"
end="${2:-2}"
shift "$(( $# >= 2 ? 2 : $# ))"
cd "$repo_root"
exec python scripts/smoke_local_llama.py --start "$start" --end "$end" "$@"
