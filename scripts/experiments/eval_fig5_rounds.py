#!/usr/bin/env python
"""图 5(Streaming & Rehydration)按 round 计算 ACC / ROUGE-L / BERTScore。

输入:stream_fig5 case 目录下的 artifacts/
  first_odd.json   1st odd QA(odd 批 ingest 后立即查询,L1 热 + 触发 L2 晋升)
  second_odd.json  2nd odd QA(全部流式完成后,odd 数据靠 L2 留存)
  second_even.json 2nd even QA(even 数据靠向量库 rehydrate 回退)
  summary.json     每轮 em/token_f1/l1_l2_hit_rate/时延/统计

输出:
  <case>/artifacts/standard_metrics.json  逐 round 指标 + 汇总
  <run>/stream_fig5/fig5_rounds_summary.csv 汇总表

用法:
  python scripts/experiments/eval_fig5_rounds.py \
    output/experiments/run_20260815_025512/stream_fig5/fig5
  # 不带参数时默认扫描最新 run 下的 stream_fig5 case
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

ROOT = pathlib.Path(__file__).resolve().parents[2]

# 复用 eval_standard_metrics 的 ACC/ROUGE/BERTScore 计算逻辑(同目录)
# scripts 目录不是包,用 importlib 按文件路径加载,避免 ModuleNotFoundError
import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location(
    "eval_standard_metrics", ROOT / "scripts/experiments/eval_standard_metrics.py")
_eval_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_eval_mod)
compute_metrics = _eval_mod.compute_metrics

ROUND_FILES = [("1st_odd", "first_odd.json"), ("2nd_odd", "second_odd.json"),
               ("2nd_even", "second_even.json")]


def main() -> int:
    ap = argparse.ArgumentParser(description="图 5 按 round 计算标准指标")
    ap.add_argument("case_dirs", nargs="*", help="stream_fig5 case 目录;缺省自动扫描最新 run")
    ap.add_argument("--bert-model", default="sentence-transformers/all-MiniLM-L6-v2",
                    help="BERTScore 编码器(用本地完整缓存的模型)")
    args = ap.parse_args()

    if args.case_dirs:
        dirs = [pathlib.Path(d) for d in args.case_dirs]
    else:
        # 默认扫描最新含 stream_fig5 的 run
        runs = sorted((ROOT / "output" / "experiments").glob("run_*"),
                      key=lambda p: p.name, reverse=True)
        dirs = []
        for run in runs:
            cands = sorted((run / "stream_fig5").glob("*"))
            if cands:
                dirs = [c for c in cands if c.is_dir()]
                break
    if not dirs:
        print("[eval] 未找到任何 stream_fig5 case 目录")
        return 1

    summary_rows = []
    for case_dir in dirs:
        art = case_dir / "artifacts"
        name = case_dir.name
        round_metrics = {}
        for label, fname in ROUND_FILES:
            path = art / fname
            if not path.exists():
                print(f"[eval] {name}: 缺少 {fname},跳过 {label}")
                continue
            rows = json.loads(path.read_text())
            print(f"[eval] {name} {label}: 读取 {len(rows)} 条 QA → 计算中...")
            m = compute_metrics(rows, args.bert_model)
            m["round"] = label
            m["case"] = f"{name}_{label}"
            # 回填 summary.json 里同轮的质量指标(em/token_f1/时延/命中来源)
            sum_path = art / "summary.json"
            if sum_path.exists():
                for agg in json.loads(sum_path.read_text()):
                    if agg.get("case") == label:
                        for k in ("em", "token_f1", "l1_hit_rate", "l2_hit_rate",
                                  "dont_know_rate", "rehydrate_successes", "evicted_chunks"):
                            if k in agg:
                                m[k] = agg[k]
            round_metrics[label] = m
            summary_rows.append(m)

        out = {"case": name, "rounds": round_metrics,
               "diff_l2_vs_rehydrate_acc":
                   round_metrics.get("2nd_odd", {}).get("acc") -
                   round_metrics.get("2nd_even", {}).get("acc")
                   if {"2nd_odd", "2nd_even"} <= set(round_metrics)
                   else None}
        (art / "standard_metrics.json").write_text(
            json.dumps(out, indent=2, ensure_ascii=False))

    # 汇总 CSV(写到 stream_fig5/ 目录下)
    run_dir = dirs[0].parents[0]
    csv_path = run_dir / "fig5_rounds_summary.csv"
    cols = ["case", "round", "count", "acc", "em", "token_f1", "rouge1_f",
            "rouge2_f", "rouge_l_f", "bs_f1", "l1_hit_rate", "l2_hit_rate"]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(summary_rows)
    print(f"[eval] 汇总已写入 → {csv_path}")

    # 终端打印对比(对齐论文图 5:ACC/RL/BS 三行)
    print("\n=== 图 5 三阶段 ACC / ROUGE-L / BERTScore ===")
    for label in ("1st_odd", "2nd_odd", "2nd_even"):
        m = round_metrics.get(label)
        if not m:
            continue
        def fmt(v):
            return "  -  " if v is None else f"{v:6.3f}"
        print(f"{label:<10} ACC={fmt(m['acc'])} RL={fmt(m['rouge_l_f'])} "
              f"BS={fmt(m['bs_f1'])} (n={m['count']}, "
              f"l1_hit={fmt(m.get('l1_hit_rate'))} l2_hit={fmt(m.get('l2_hit_rate'))})")
    if out.get("diff_l2_vs_rehydrate_acc") is not None:
        print(f"\n2nd_odd ACC − 2nd_even ACC = {out['diff_l2_vs_rehydrate_acc']:.3f} "
              f"(论文图 5: 70.83 − 54.33 = 16.50)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
