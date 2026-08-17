#!/usr/bin/env python
"""为标准评估指标计算 lru_rehydrate 各 case 的 ACC / ROUGE-L / BERTScore。

输入:每个 case 目录下的 artifacts/qa.json(含 gt + predict 字段)。
输出:
  - <case>/artifacts/standard_metrics.json  单 case 指标
  - <run>/lru_rehydrate/standard_metrics_summary.csv 汇总表

用法:
  python scripts/experiments/eval_standard_metrics.py \
    output/experiments/run_20260815_025512/lru_rehydrate/capacity_50 \
    output/experiments/run_20260815_025512/lru_rehydrate/rehydrate_on ...
  # 不带参数时默认扫描 output/experiments/run_*/lru_rehydrate/*/
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib
import re
import sys

# BERTScore 一律走本地缓存模型,禁止联网下载
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

ROOT = pathlib.Path(__file__).resolve().parents[2]


def normalize(text) -> str:
    """与 scripts/experiments/common.py 一致:小写、去冠词、去标点、压缩空白。"""
    value = str(text or "").lower().strip()
    value = re.sub(r"\b(a|an|the)\b", " ", value)
    value = re.sub(r"[^\w\s]", " ", value)
    return " ".join(value.split())


def token_contains(pred_toks: list, gt_toks: list) -> bool:
    """GT 的 token 序列是否作为连续子序列出现在预测中(包含式 ACC 判定)。

    论文图 5 的 ACC 定义为"答案是否在输出的预测内",即 GT 是预测输出的
    子串(逐 token 连续匹配),而非严格相等(EM)。用 token 级子序列避免
    子串误判(如 "yes" in "yesterday")。
    """
    if not gt_toks or not pred_toks or len(gt_toks) > len(pred_toks):
        return False
    for i in range(len(pred_toks) - len(gt_toks) + 1):
        if pred_toks[i:i + len(gt_toks)] == gt_toks:
            return True
    return False


def compute_metrics(rows: list[dict], bert_model: str) -> dict:
    """对一组 QA 样本计算 ACC(包含式,GT 在预测内)/ ROUGE-1/2/L / BERTScore。

    ACC 与论文图 5 口径一致:归一化后 GT 的 token 序列作为连续子序列出现在
    预测中即判对(见 token_contains),非严格 EM 匹配。
    """
    preds, gts, accs = [], [], []
    for r in rows:
        p, g = normalize(r.get("predict")), normalize(r.get("gt"))
        accs.append(float(bool(g) and token_contains(p.split(), g.split())))
        # ROUGE 与 BERTScore 不接受空串,空预测/空参考用 "empty" 占位
        # (不能用 ".":rouge 包会把它分词成空假设抛 "Hypothesis is empty")
        preds.append(p or "empty")
        gts.append(g or "empty")

    out = {
        "count": len(rows),
        "acc": sum(accs) / len(accs) if accs else 0.0,
    }

    # ROUGE(含 ROUGE-L)
    try:
        from rouge import Rouge
        scores = Rouge().get_scores(preds, gts, avg=True)
        out["rouge1_f"] = scores["rouge-1"]["f"]
        out["rouge2_f"] = scores["rouge-2"]["f"]
        out["rouge_l_f"] = scores["rouge-l"]["f"]
    except Exception as exc:  # 个别超短答案可能让 rouge 包抛异常,降级为 None
        print(f"[eval] ROUGE 计算失败: {exc}")
        out["rouge1_f"] = out["rouge2_f"] = out["rouge_l_f"] = None

    # BERTScore(本地缓存模型,报告 P/R/F1 均值)
    # 注:本地 bert-base-uncased 缓存残缺(缺 snapshots/权重),
    # 改用完整缓存的 all-MiniLM-L6-v2(BERT 结构,BERTScore 兼容)。
    try:
        from bert_score import score as bs_score
        from bert_score.score import model2layers
        # all-MiniLM-L6-v2(6 层 BERT)不在 bert_score 预置白名单里,
        # 手动注册层数,否则 model2layers[model_type] 抛 KeyError。
        model2layers.setdefault(bert_model, 5)
        P, R, F1 = bs_score(
            preds, gts, model_type=bert_model,
            lang="en", verbose=False, batch_size=64,
        )
        out["bs_precision"] = float(P.mean())
        out["bs_recall"] = float(R.mean())
        out["bs_f1"] = float(F1.mean())
    except Exception as exc:
        print(f"[eval] BERTScore 计算失败: {exc}")
        out["bs_precision"] = out["bs_recall"] = out["bs_f1"] = None

    return out


def iter_case_dirs() -> list[pathlib.Path]:
    """默认收集最新一次 run 目录下 lru_rehydrate 的各 case 目录。"""
    runs = sorted(
        (p for p in (ROOT / "output" / "experiments").glob("run_*")
         if (p / "lru_rehydrate").is_dir()),
        key=lambda p: p.name, reverse=True,
    )
    if not runs:
        return []
    run = runs[0]
    return sorted(
        (p for p in (run / "lru_rehydrate").glob("*")
         if p.is_dir() and (p / "artifacts" / "qa.json").is_file()),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="计算 ACC/ROUGE-L/BERTScore")
    ap.add_argument("case_dirs", nargs="*", help="case 目录;缺省时自动扫描最新 run")
    ap.add_argument("--bert-model", default="sentence-transformers/all-MiniLM-L6-v2",
                    help="BERTScore 编码器(用本地完整缓存的模型)")
    args = ap.parse_args()

    if args.case_dirs:
        dirs = [pathlib.Path(d) for d in args.case_dirs]
    else:
        dirs = iter_case_dirs()
    if not dirs:
        print("[eval] 未找到任何含 qa.json 的 case 目录")
        return 1

    summary_rows = []
    for case_dir in dirs:
        qa_path = case_dir / "artifacts" / "qa.json"
        rows = json.loads(qa_path.read_text())
        name = case_dir.name
        print(f"[eval] {name}: 读取 {len(rows)} 条 QA → 计算中...")
        metrics = compute_metrics(rows, args.bert_model)
        metrics["case"] = name
        # 回填主实验已有的质量指标,便于同表对比
        sum_path = case_dir / "artifacts" / "summary.json"
        if sum_path.exists():
            agg = json.loads(sum_path.read_text())
            if agg:
                for k in ("em", "token_f1", "l1_hit_rate", "dont_know_rate",
                          "rehydrate_successes", "evicted_chunks"):
                    if k in agg[0]:
                        metrics[k] = agg[0][k]
        (case_dir / "artifacts" / "standard_metrics.json").write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False))
        summary_rows.append(metrics)

    # 汇总 CSV(写到 lru_rehydrate/ 目录下)
    run_dir = dirs[0].parents[0]
    csv_path = run_dir / "standard_metrics_summary.csv"
    cols = ["case", "count", "acc", "em", "token_f1", "rouge1_f", "rouge2_f",
            "rouge_l_f", "bs_f1", "bs_precision", "bs_recall",
            "l1_hit_rate", "rehydrate_successes", "evicted_chunks"]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(summary_rows)
    print(f"[eval] 汇总已写入 → {csv_path}")

    # 终端打印对比
    print("\n=== ACC / ROUGE-L / BERTScore 对比 ===")
    header = f"{'case':<14}{'acc':>7}{'em':>7}{'r1_f':>8}{'r2_f':>8}{'rL_f':>8}{'bs_f':>8}{'rehyd':>6}"
    print(header)
    for r in summary_rows:
        def fmt(v):
            return "  -  " if v is None else f"{v:6.3f}"
        print(f"{r['case']:<14}{fmt(r['acc'])}{fmt(r.get('em'))}{fmt(r['rouge1_f'])}{fmt(r['rouge2_f'])}{fmt(r['rouge_l_f'])}{fmt(r['bs_f1'])}{r.get('rehydrate_successes','-')!s:>6}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
