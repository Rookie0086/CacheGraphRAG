#!/usr/bin/env bash
# 热启动 QA + 固定 L2 晋升数据观测(2Wiki 600 构建产物)
#
# 场景:
#   1. L1 热启动 —— 载入 600 构建的 base gexf(全量拓扑进内存图);
#   2. L2 固定 —— 克隆 600 构建的 Nebula `wikimultihopqa` 空间(1583 顶点/1422 边)
#      到独立 exp 空间,原空间不被污染;
#   3. fixed 协议 QA —— 观测每次检索的 L1/L2 命中、chunk 访问计数,
#      以及 h(c) ≥ τ_hit(默认 3)触发的 L2 晋升(promoted_chunks)。
#
# 输出: output/hotqa_promotion/run_<ts>/{config.yaml, artifacts/{summary,qa,latency,l2_seed}.json}
# 用法:
#   bash scripts/run_hotqa_promotion.sh [--start 0] [--end 200] [--queries 100] [--seed 42]
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$REPO/.conda/cachegraphrag-mac/bin/python"
cd "$REPO"

START=0; END=200; QUERIES=100; SEED=42
while [[ $# -gt 0 ]]; do
  case "$1" in
    --start) START="$2"; shift 2;;
    --end) END="$2"; shift 2;;
    --queries) QUERIES="$2"; shift 2;;
    --seed) SEED="$2"; shift 2;;
    *) echo "unknown arg: $1" >&2; exit 1;;
  esac
done

TS=$(date +%Y%m%d_%H%M%S)
RUN="$REPO/output/hotqa_promotion/run_$TS"
TARGET="exp_hotqa_${TS}"
mkdir -p "$RUN/artifacts"
echo "== run: $RUN | L2 target: $TARGET =="

echo "== 1/3 克隆固定 L2(wikimultihopqa -> $TARGET)=="
"$PY" scripts/experiments/nebula_clone.py --source wikimultihopqa --target "$TARGET" \
  --output "$RUN/artifacts/l2_seed.json"

echo "== 2/3 生成热启动配置(L1 载入 600 gexf,L2 指向克隆空间)=="
"$PY" - "$RUN" "$TARGET" <<'EOF'
import sys, yaml, pathlib
run, target = sys.argv[1], sys.argv[2]
cfg = yaml.safe_load(open('config/config.yaml', encoding='utf-8'))
ret = cfg.setdefault('retrieval', {})
ret.update({
    'skip_index': True, 'index_only': False, 'qa_cache': False,
    'nebula_space': target,
    'chunk_collection': 'wikimultihopqa',
    'entity_index_name': 'entity_index_wikimultihopqa',
    'base_gexf': 'subgraph/base/wikimultihopqa_wikimultihopqa_wikimultihopqa_0_600_base.gexf',
    'agentic': False, 'warmup_ratio': 0.0, 'clear_l2': False,
    'qa_concurrency': 5, 'enable_l2': True, 'enable_rehydrate': True,
    'promotion_threshold': 3,
})
idx = cfg.setdefault('indexing', {})
idx.update({'l1_max_chunks': 200, 'promotion_threshold': 3})
out = pathlib.Path(run) / 'config.yaml'
yaml.safe_dump(cfg, open(out, 'w', encoding='utf-8'), allow_unicode=True)
print(out)
EOF

echo "== 3/3 fixed 协议 QA(热 L1 + 固定 L2,记录晋升)=="
CACHEGRAPH_CONFIG="$RUN/config.yaml" "$PY" scripts/experiments/protocol_runner.py fixed \
  --output "$RUN/artifacts" --start "$START" --end "$END" --queries "$QUERIES" \
  --seed "$SEED" --case-label "hot_qa" 2>&1 | tail -20

echo "== L2 晋升后增长核对 =="
"$PY" - "$TARGET" "$RUN" <<'EOF'
import json, pathlib, sys
from nebula3.gclient.net import ConnectionPool
from nebula3.Config import Config
target, run = sys.argv[1], sys.argv[2]
config = Config(); config.max_connection_pool_size = 3
pool = ConnectionPool(); pool.init([('127.0.0.1', 9669)], config)
s = pool.get_session('root', 'nebula')
def count(q):
    r = s.execute(q)
    v = r.column_values('c')[0]
    return v.get_iVal() if hasattr(v, 'get_iVal') else int(str(v))
v = count(f'USE `{target}`; MATCH (n) RETURN count(n) AS c')
e = count(f'USE `{target}`; MATCH ()-[x]->() RETURN count(x) AS c')
seed = json.load(open(f'{run}/artifacts/l2_seed.json'))
after = seed.get('target_after', {})   # 克隆完成后的基线(= 源空间 1583/1422)
base_v, base_e = after.get('vertices', 0), after.get('edges', 0)
print(f'  {target}: 克隆基线 vertices={base_v} edges={base_e}(源 {seed.get("source")})')
print(f'  {target}: QA 后 vertices={v} edges={e}  Δv={v - base_v} Δe={e - base_e}(晋升写入)')
summary = json.load(open(f'{run}/artifacts/summary.json'))
print(f'  promoted_chunks={summary[0].get("promoted_chunks")} chunks_touched={summary[0].get("chunks_touched")}')
print(f'  run 目录: {run}')
EOF
