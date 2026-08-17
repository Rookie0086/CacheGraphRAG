#!/usr/bin/env python3
"""本地适配版 Rouge-LBERTScore 评估(复刻外部 Rouge-LBERTScore.py 的接口与输出格式)。

外部脚本依赖 /share3/... 服务器路径(acc.py / Rouge-L&BERTScore.py),本机不可用;
本脚本用本机实现三个指标,输出格式与外部脚本一致:
  - acc      : checkanswer 近似 —— 归一化后 GT token 作为连续子序列包含于预测中
               (与 scripts/experiments/eval_standard_metrics.py 的 token_contains 一致)
  - rouge_l  : Rouge-L-R(recall),来自 rouge 包
  - bert     : BERTScore 余弦,来自 src/entity/bert_sim.py(all-MiniLM-L6-v2)

用法:
  .conda/cachegraphrag-mac/bin/python scripts/eval_rouge_bertscore.py --qa_result <结果.json>...
输出:
  <结果路径>.metrics.json(与外部脚本 *_metrics.json 相同的 qa_items + summary 结构)
"""
import argparse
import json
import re
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.entity.bert_sim import bert as _bert  # noqa: E402
from acc import checkanswer  # noqa: E402  (acc.py,checkanswer:子串包含匹配,与外部 Rouge-LBERTScore.py 一致)


def normalize(text) -> str:
    """与 eval_standard_metrics 一致:小写 + 非字母数字转空格 + 压缩空白。"""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", str(text).lower())).strip()


def checkanswer_acc(pred: str, gt: str) -> bool:
    """ACC 按 acc.py 的 checkanswer 口径:规范化后 GT 是预测的子串(支持嵌套多答案)。"""
    gt_for_check = [[gt]]
    try:
        labels = checkanswer(pred, gt_for_check)
        return bool(labels) and all(labels)
    except Exception:
        return False


def rouge_l_r(pred: str, gt: str) -> float:
    """Rouge-L Recall(对应外部脚本 checkanswer_rougel 的 Rouge-L-R)。"""
    from rouge import Rouge
    p = pred or "empty"
    g = gt or "empty"
    scores = Rouge().get_scores(p, g, avg=True)
    return float(scores["rouge-l"]["r"])


def compute_metrics(items):
    evaluated = []
    for item in items:
        pred = str(item.get("predicted") or item.get("predict") or item.get("raw_answer") or "")
        gt = item.get("ground_truth") or item.get("gt") or ""
        if isinstance(gt, list):
            gt = gt[0] if gt else ""
        gt = str(gt)

        # ACC(acc.py checkanswer 口径:子串包含匹配,支持嵌套多答案)
        try:
            is_correct = checkanswer_acc(pred, gt)
        except Exception:
            is_correct = False

        # Rouge-L-R
        try:
            rouge_l = rouge_l_r(pred, gt)
        except Exception:
            rouge_l = 0.0

        # BERTScore
        try:
            bert_val = float(_bert(pred, gt))
        except Exception:
            bert_val = 0.0

        evaluated.append({
            **item,
            "is_correct": is_correct,
            "rouge_l": round(rouge_l, 4),
            "bert_score": round(bert_val, 4),
        })

    total = len(evaluated)
    correct = sum(1 for r in evaluated if r["is_correct"])
    return evaluated, {
        "total": total,
        "correct": correct,
        "accuracy_pct": round(correct / total * 100, 1) if total else 0.0,
        "avg_rouge_l": round(sum(r["rouge_l"] for r in evaluated) / total, 4) if total else 0.0,
        "avg_bert_score": round(sum(r["bert_score"] for r in evaluated) / total, 4) if total else 0.0,
    }


def build_format(evaluated, summary):
    return {
        "qa_items": [
            {
                "question": r.get("query", ""),
                "ground_truth": str(r.get("ground_truth", r.get("gt", ""))),
                "response": r.get("predicted", r.get("predict", "")),
                "correct": 1 if r["is_correct"] else 0,
                "rouge_l": r["rouge_l"],
                "bert_score": r["bert_score"],
            }
            for r in evaluated
        ],
        "summary": summary,
    }


def process_single(qa_result_path: str) -> bool:
    print(f"[RUN ] {qa_result_path}")
    with open(qa_result_path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        items = data.get("results", data)
        if isinstance(items, dict):
            items = [items]
    else:
        items = [data]
    evaluated, summary = compute_metrics(items)
    print(f"       acc={summary['accuracy_pct']}%  "
          f"rouge_l={summary['avg_rouge_l']:.4f}  "
          f"bert={summary['avg_bert_score']:.4f}  "
          f"({summary['correct']}/{summary['total']})")
    out_path = qa_result_path + ".metrics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(build_format(evaluated, summary), f, ensure_ascii=False, indent=2)
    print(f"       -> {out_path}")
    return True


def main():
    ap = argparse.ArgumentParser(description="本地 Rouge-L/BERTScore 评估(兼容外部 Rouge-LBERTScore.py 输出)")
    ap.add_argument("--qa_result", nargs="+", required=True, help="QA 结果 JSON 文件(可多个)")
    args = ap.parse_args()
    ok = fail = 0
    for p in args.qa_result:
        if process_single(p):
            ok += 1
        else:
            fail += 1
    print(f"\nDone: {ok} succeeded, {fail} failed")


if __name__ == "__main__":
    main()
