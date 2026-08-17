"""Unified metrics runner for all RAG frameworks.

Computes Accuracy (checkanswer), Rouge-L (checkanswer_rougel), and
BERTScore (bert) across all (framework, dataset) combinations.
Metric functions are imported from datasets/Rouge-L&BERTScore.py.

Usage:
    python /share3/home/yangbo/datasets/utils/Rouge-LBERTScore.py
    python /share3/home/yangbo/datasets/utils/Rouge-LBERTScore.py --framework hipporag
    python /share3/home/yangbo/datasets/utils/Rouge-LBERTScore.py --dataset 2wiki
    python /share3/home/yangbo/datasets/utils/Rouge-LBERTScore.py --qa_result path/to/results.json --gpu_id 1
"""
import argparse
import importlib.util
import json
import os
import sys

sys.path.insert(0, "/share3/home/yangbo/datasets/utils")

# ── Constants ──────────────────────────────────────────────────
RESULTS_ROOT = "/share3/home/yangbo/datasets/results"
RAW_QA_DIR = os.path.join(RESULTS_ROOT, "raw_qa_result")
FRAMEWORKS = ["hipporag", "hipporag1", "hypergraphrag", "lightrag", "graphrag", "erarag", "kag"]
DATASETS = ["2wiki", "hotpotqa", "musique", "rgb", "whop600", "hpqa600", "rgb300", "wiki600"]


# ── First pass: parse known args to set GPU before model load ──
_parser = argparse.ArgumentParser(description="Compute metrics for RAG frameworks")
_parser.add_argument("--framework", choices=FRAMEWORKS + ["all"], default="all")
_parser.add_argument("--dataset", choices=DATASETS + ["all"], default="all")
_parser.add_argument("--qa_result", type=str, default=None,
                     help="Path to single QA results JSON file (overrides --framework/--dataset)")
_parser.add_argument("--gpu_id", type=int, default=0,
                     help="GPU device ID to use (default: 0, -1 for CPU)")
_args, _ = _parser.parse_known_args()

if _args.gpu_id >= 0:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(_args.gpu_id)

from acc import checkanswer  # noqa: E402

_metrics_path = "/share3/home/yangbo/datasets/Rouge-L&BERTScore.py"
_metrics_spec = importlib.util.spec_from_file_location("rouge_bert", _metrics_path)
_metrics_mod = importlib.util.module_from_spec(_metrics_spec)
_metrics_spec.loader.exec_module(_metrics_mod)
checkanswer_rougel = _metrics_mod.checkanswer_rougel
bert = _metrics_mod.bert


def _normalize_gt_for_check(gt):
    if isinstance(gt, list) and len(gt) > 0:
        if isinstance(gt[0], list):
            return gt
        return [gt]
    return [[gt]]


def compute_metrics(items):
    evaluated = []
    for item in items:
        pred = item.get("predicted") or item.get("predict") or item.get("raw_answer") or ""
        gt = item.get("ground_truth") or item.get("gt") or ""

        # Accuracy (checkanswer)
        gt_for_check = _normalize_gt_for_check(gt)
        try:
            labels = checkanswer(pred, gt_for_check)
            is_correct = all(labels) if labels else False
        except Exception:
            labels = [0]
            is_correct = False

        # Rouge-L (checkanswer_rougel)
        try:
            rouge_dict = checkanswer_rougel(pred, gt)
            rouge_l = rouge_dict.get("Rouge-L-R", 0.0)
        except Exception:
            rouge_l = 0.0

        # BERTScore (bert)
        try:
            bert_val = bert(pred, gt)
        except Exception:
            bert_val = 0.0

        evaluated.append({
            **item,
            "is_correct": is_correct,
            "labels": labels,
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


def _build_format(evaluated, summary):
    return {
        "qa_items": [
            {
                "question": r.get("query", ""),
                "ground_truth": _normalize_gt_for_str(r.get("ground_truth", r.get("gt", ""))),
                "response": r.get("predicted", r.get("predict", "")),
                "correct": 1 if r["is_correct"] else 0,
                "rouge_l": r["rouge_l"],
                "bert_score": r["bert_score"],
            }
            for r in evaluated
        ],
        "summary": summary,
    }


def _normalize_gt_for_str(gt):
    if isinstance(gt, list):
        return gt
    return str(gt)


def process(framework, dataset):
    results_path = os.path.join(RAW_QA_DIR, f"{framework}_{dataset}_results.json")
    if not os.path.exists(results_path):
        print(f"[MISS] {framework}/{dataset}")
        return False

    print(f"[RUN ] {framework}/{dataset}")
    with open(results_path, encoding="utf-8") as f:
        data = json.load(f)

    evaluated, summary = compute_metrics(data["results"])
    print(f"       acc={summary['accuracy_pct']}%  "
          f"rouge_l={summary['avg_rouge_l']:.4f}  "
          f"bert={summary['avg_bert_score']:.4f}  "
          f"({summary['correct']}/{summary['total']})")

    format_out = _build_format(evaluated, summary)
    out_dir = os.path.join(RESULTS_ROOT, framework)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{framework}_{dataset}_format.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(format_out, f, ensure_ascii=False, indent=2)
    print(f"       -> {out_path}")
    return True


def process_single(qa_result_path: str):
    """Process a single QA results file and print/save metrics."""
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

    format_out = _build_format(evaluated, summary)
    out_path = qa_result_path.replace("_results.json", "_metrics.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(format_out, f, ensure_ascii=False, indent=2)
    print(f"       -> {out_path}")
    return True


def main():
    frameworks = FRAMEWORKS if _args.framework == "all" else [_args.framework]
    datasets = DATASETS if _args.dataset == "all" else [_args.dataset]

    if _args.qa_result:
        process_single(_args.qa_result)
        return

    ok = fail = 0
    for fw in frameworks:
        for ds in datasets:
            if process(fw, ds):
                ok += 1
            else:
                fail += 1

    print(f"\nDone: {ok} succeeded, {fail} skipped")


if __name__ == "__main__":
    main()
