#!/usr/bin/env bash
# -*- coding: utf-8 -*-
# =====================================================================
# CacheGraphRAG 可复现实验脚本(中档#11,回应 R4-W2 / R4-9.7)
# =====================================================================
# 作用:串起「启动数据库 → 建索引 → 预热/测试隔离 QA → 存储报告 → 汇总」全流程,
#       并把 config 快照 + 随机种子 + git commit 自动记入审计文件,保证结果可复现。
#
# 用法:
#   bash scripts/repro_wikimultihopqa.sh                      # 默认 full:全流程
#   bash scripts/repro_wikimultihopqa.sh --phase build        # 仅建索引
#   bash scripts/repro_wikimultihopqa.sh --phase qa           # 仅预热/测试隔离 QA
#   bash scripts/repro_wikimultihopqa.sh --phase storage      # 仅存储报告
#   bash scripts/repro_wikimultihopqa.sh --dry-run            # 只打印将执行的命令
#
# 关键参数:
#   --dataset wikimultihopqa  数据集名(默认 wikimultihopqa)
#   --start 0 --end 600       数据区间(默认 0~600)
#   --warmup-ratio 0.15       预热集占比(>0 时启用两阶段隔离协议)
#   --warmup-seed 42          预热/测试切分种子(可复现)
#   --skip-db-check           跳过数据库健康检查(已确认在跑时用)
#
# 输出:
#   log/repro_<dataset>_<ts>.output        本脚本运行日志
#   output/repro/repro_<dataset>_<ts>.json 审计文件(config 快照+seed+git commit+产物路径)
#   output/qa/warmup_report_*.json         预热/测试隔离指标
#   output/storage/storage_report_*.json   存储占用字节报告
# =====================================================================

set -euo pipefail

# ── 仓库根目录 ─────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

# ── 参数解析 ─────────────────────────────────────────────
DATASET="wikimultihopqa"
START=0
END=600
WARMUP_RATIO=0.15
WARMUP_SEED=42
PHASE="full"
GEXF_PATTERN="subgraph/base/*_base.gexf"
SKIP_DB_CHECK=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)     DATASET="$2"; shift 2 ;;
    --start)       START="$2";   shift 2 ;;
    --end)         END="$2";     shift 2 ;;
    --warmup-ratio) WARMUP_RATIO="$2"; shift 2 ;;
    --warmup-seed) WARMUP_SEED="$2"; shift 2 ;;
    --phase)       PHASE="$2";   shift 2 ;;
    --gexf)        GEXF_PATTERN="$2"; shift 2 ;;
    --skip-db-check) SKIP_DB_CHECK=1; shift ;;
    --dry-run)     DRY_RUN=1;    shift ;;
    *) echo "未知参数: $1" >&2; exit 2 ;;
  esac
done

# 阶段校验
case "$PHASE" in
  full|build|qa|storage) ;;
  *) echo "非法 --phase: $PHASE (可选 full|build|qa|storage)" >&2; exit 2 ;;
esac

# ── 输出路径与日志 ────────────────────────────────────────
TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="log"
REPRO_DIR="output/repro"
mkdir -p "$LOG_DIR" "$REPRO_DIR" "output/qa" "output/storage"

LOG_FILE="$LOG_DIR/repro_${DATASET}_${TS}.output"
AUDIT_FILE="$REPRO_DIR/repro_${DATASET}_${TS}.json"
CONFIG_TMP="$REPRO_DIR/config_${DATASET}_${PHASE}_${TS}.yaml"

# 所有后续输出同时进终端与日志文件
exec > >(tee -a "$LOG_FILE") 2>&1

echo "======================================================================"
echo "  CacheGraphRAG 复现实验  [${PHASE}]  $DATASET [$START:$END]"
echo "  log:   $LOG_FILE"
echo "  audit: $AUDIT_FILE"
echo "======================================================================"

# 工具函数:打印并执行
run_cmd() {
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "  [dry-run] $*"
  else
    echo "  >>> $*"
    "$@"
  fi
}

# 工具函数:生成临时 config(index_only / skip_index 二选一 + warmup 参数)
# 用法:make_config index_only|skip_index
make_config() {
  local mode="$1"
  run_cmd python3 - "$CONFIG_TMP" "$mode" "$WARMUP_RATIO" "$WARMUP_SEED" "$START" "$END" <<'PY'
import sys, yaml
out, mode, warmup_ratio, warmup_seed = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
start, end = sys.argv[5], sys.argv[6]
with open("config/config.yaml", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
ret = cfg.setdefault("retrieval", {})
if mode == "index_only":
    ret["index_only"] = True
    ret["skip_index"] = False
else:  # skip_index 模式跑 QA
    ret["index_only"] = False
    ret["skip_index"] = True
ret["warmup_ratio"] = float(warmup_ratio)
ret["warmup_seed"] = int(warmup_seed)
# 区间随命令行参数写入临时 config,保证与 audit 一致
cfg.setdefault("data", {})["start"] = int(start)
cfg.setdefault("data", {})["end"] = int(end)
with open(out, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, allow_unicode=True)
print(f"  临时 config 已生成: {out} (start={start} end={end})")
PY
}

# ── 第 1 步:确认数据库在运行 ─────────────────────────────
echo ""
echo "── [1/5] 数据库健康检查 ──────────────────────────────"
if [[ "$SKIP_DB_CHECK" -eq 1 ]] || [[ "$DRY_RUN" -eq 1 ]]; then
  echo "  跳过数据库检查 (--skip-db-check / --dry-run)"
else
  # Milvus standalone
  if docker ps --filter "name=milvus-standalone" --filter "status=running" \
      --format '{{.Names}}' | grep -q "^milvus-standalone$"; then
    echo "  [ok] Milvus standalone 运行中"
  else
    echo "  [..] 启动 Milvus standalone ..."
    bash database/setup/milvus-install-user.sh start
  fi
  # NebulaGraph
  if docker ps --filter "name=nebula-docker-compose-graphd-1" --filter "status=running" \
      --format '{{.Names}}' | grep -q "^nebula-docker-compose-graphd-1$"; then
    echo "  [ok] NebulaGraph 集群运行中"
  else
    echo "  [..] 启动 NebulaGraph 集群 ..."
    bash database/setup/nebula-install-user.sh start "$HOME/.nebula-up"
  fi
fi

# ── 第 2 步:建索引(index_only) ────────────────────────────
echo ""
echo "── [2/5] 建索引 ──────────────────────────────────────"
BASE_GEXF_FOUND="$(ls $GEXF_PATTERN 2>/dev/null | head -1 || true)"
if [[ -n "$BASE_GEXF_FOUND" ]] && [[ "$PHASE" != "build" ]]; then
  echo "  已有基础图快照: $BASE_GEXF_FOUND (跳过建索引)"
  echo "  (如需强制重建: --phase build)"
else
  make_config index_only
  export CACHEGRAPH_CONFIG="$CONFIG_TMP"
  run_cmd python -m src.CacheGraphRAG
  unset CACHEGRAPH_CONFIG
fi

# ── 第 3 步:预热/测试隔离 QA(skip_index) ─────────────────
echo ""
echo "── [3/5] 预热/测试隔离 QA ────────────────────────────"
if [[ "$PHASE" == "build" ]]; then
  echo "  跳过 QA (--phase build 仅建索引)"
else
  make_config skip_index
  export CACHEGRAPH_CONFIG="$CONFIG_TMP"
  run_cmd python -m src.CacheGraphRAG
  unset CACHEGRAPH_CONFIG
fi

# ── 第 4 步:存储报告 ──────────────────────────────────────
echo ""
echo "── [4/5] 存储占用字节报告 ─────────────────────────────"
if [[ "$PHASE" == "build" ]] || [[ "$PHASE" == "qa" ]]; then
  echo "  跳过存储报告 (仅 full/storage 阶段运行)"
else
  STORAGE_OUT="$REPRO_DIR/storage_report_${DATASET}_${TS}.json"
  run_cmd python scripts/storage_report.py \
      --dataset "$DATASET" --start "$START" --end "$END" \
      --nebula-space-map "wikimultihopqa=25,exp_clone_a=50" \
      --nebula-space "$DATASET" --output "$STORAGE_OUT"
fi

# ── 第 5 步:汇总审计文件 ──────────────────────────────────
echo ""
echo "── [5/5] 写入审计文件 ────────────────────────────────"
if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "  [dry-run] 跳过审计文件写入"
  echo ""
  echo "dry-run 结束。移除 --dry-run 执行真实复现。"
  exit 0
fi

python3 - "$AUDIT_FILE" "$DATASET" "$START" "$END" "$PHASE" \
  "$WARMUP_RATIO" "$WARMUP_SEED" "$LOG_FILE" "$CONFIG_TMP" <<'PY'
import json, os, subprocess, sys
audit, dataset, start, end, phase = sys.argv[1:6]
warmup_ratio, warmup_seed, log_file, config_tmp = sys.argv[6:10]

# git commit(仓库是 git repo;失败则降级记录)
git = {"repo": False}
try:
    out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10)
    git = {"repo": True, "commit": out.stdout.strip(),
           "branch": subprocess.run(["git", "branch", "--show-current"],
                                    capture_output=True, text=True).stdout.strip()}
except Exception as e:
    git["error"] = str(e)

# config 快照:保存一份到 output/repro 并记录
snapshot = f"output/repro/config_snapshot_{dataset}_{phase}_{__import__('datetime').datetime.now():%Y%m%d_%H%M%S}.yaml"
os.makedirs("output/repro", exist_ok=True)
with open("config/config.yaml", encoding="utf-8") as f:
    with open(snapshot, "w", encoding="utf-8") as g:
        g.write(f.read())

# 收集各阶段产物路径
warmup_reports = []
for f in sorted(os.listdir("output/qa")):
    if f.startswith(f"warmup_report_{dataset}") or f.startswith(f"qa_{dataset}"):
        warmup_reports.append(f"output/qa/{f}")
storage_reports = []
for f in sorted(os.listdir("output/storage")):
    if f.startswith(f"storage_report_{dataset}") or f.startswith("storage_report_"):
        storage_reports.append(f"output/storage/{f}")

report = {
    "tool": "CacheGraphRAG repro script (中档#11)",
    "dataset": dataset, "start": int(start), "end": int(end),
    "phase": phase,
    "warmup_ratio": float(warmup_ratio), "warmup_seed": int(warmup_seed),
    "git": git,
    "config_snapshot": snapshot,
    "config_tmp_used": config_tmp,
    "log_file": log_file,
    "artifacts": {
        "warmup_reports": warmup_reports,
        "storage_reports": storage_reports,
    },
    "reproduce_command": (
        f"bash scripts/repro_wikimultihopqa.sh --dataset {dataset} "
        f"--start {start} --end {end} --warmup-ratio {warmup_ratio} "
        f"--warmup-seed {warmup_seed}")
}
with open(audit, "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
print(f"  审计文件已写入: {audit}")
print(f"  config 快照: {snapshot}")
print(f"  git: {git}")
PY

echo ""
echo "======================================================================"
echo "  复现完成 ✅  log: $LOG_FILE"
echo "  审计: $AUDIT_FILE"
echo "  → 用以下命令复现本次实验:"
echo "    bash scripts/repro_wikimultihopqa.sh --dataset $DATASET --start $START --end $END --warmup-ratio $WARMUP_RATIO --warmup-seed $WARMUP_SEED"
echo "======================================================================"
