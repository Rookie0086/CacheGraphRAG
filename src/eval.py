"""Evaluation metrics tools: compute QA accuracy, retrieval recall, etc."""

import json
import os
import re
from typing import Dict, List, Optional, Tuple

from src.utils.base import checkanswer, checkanswer_rougel, get_accuracy
from src.entity.bert_sim import bert as _bert_score


# ── Normalization ────────────────────────────────────────────────

def normalize_answer(value: str) -> str:
    """Normalize answer text for exact match comparison."""
    if not value:
        return ""
    value = str(value).lower().strip()
    # Remove articles
    value = re.sub(r"\b(a|an|the)\b", " ", value)
    # Remove punctuation
    value = re.sub(r"[^\w\s]", " ", value)
    return " ".join(value.split())


# ── QA Metrics ──────────────────────────────────────────────

def exact_match(prediction: str, ground_truth: str) -> int:
    """Exact match (after normalization)."""
    return int(normalize_answer(prediction) == normalize_answer(ground_truth))


def rougel_score(prediction: str, ground_truth: str) -> float:
    """ROUGE-L F1 (based on longest common subsequence)."""
    result = checkanswer_rougel(prediction, ground_truth)
    return float(result.get("Rouge-L-F", 0.0))


def bert_score(prediction: str, ground_truth: str) -> float:
    """BERTScore semantic similarity."""
    return float(_bert_score(prediction, ground_truth))


def evaluate_qa(
    predictions: List[str],
    ground_truths: List[str],
    use_rougel: bool = False,
    use_bert: bool = False,
) -> Dict[str, float]:
    """Batch compute QA metrics.

    Args:
        predictions: List of model-generated answers.
        ground_truths: List of ground truth answers.
        use_rougel: Whether to compute ROUGE-L.
        use_bert: Whether to compute BERTScore (slow).

    Returns:
        {"em": exact match rate, "rougel": ROUGE-L F1, "bertscore": BERTScore average}
    """
    assert len(predictions) == len(ground_truths), "Prediction and ground truth count mismatch"

    em_list = [exact_match(p, g) for p, g in zip(predictions, ground_truths)]
    em = sum(em_list) / len(em_list) if em_list else 0.0

    result = {"em": round(em, 4), "count": len(em_list)}

    if use_rougel:
        rg = [rougel_score(p, g) for p, g in zip(predictions, ground_truths)]
        result["rougel"] = round(sum(rg) / len(rg), 4)

    if use_bert:
        bs = [bert_score(p, g) for p, g in zip(predictions, ground_truths)]
        result["bertscore"] = round(sum(bs) / len(bs), 4)

    return result


# ── Retrieval Metrics ─────────────────────────────────────────────

def hit_rate(
    retrieved_chunks: List[List[str]],
    relevant_chunks: List[List[str]],
) -> float:
    """Hit Rate: proportion of queries with at least one relevant chunk retrieved."""
    hits = 0
    for retrieved, relevant in zip(retrieved_chunks, relevant_chunks):
        if any(c in retrieved for c in relevant):
            hits += 1
    return round(hits / len(retrieved_chunks), 4) if retrieved_chunks else 0.0


def mrr(
    retrieved_chunks: List[List[str]],
    relevant_chunks: List[List[str]],
) -> float:
    """Mean Reciprocal Rank: average of reciprocal ranks of first relevant chunk."""
    total = 0.0
    for retrieved, relevant in zip(retrieved_chunks, relevant_chunks):
        for rank, cid in enumerate(retrieved, start=1):
            if cid in relevant:
                total += 1.0 / rank
                break
    return round(total / len(retrieved_chunks), 4) if retrieved_chunks else 0.0


def evaluate_retrieval(
    retrieved_chunks: List[List[str]],
    relevant_chunks: List[List[str]],
) -> Dict[str, float]:
    """Batch compute retrieval metrics.

    Args:
        retrieved_chunks: List of retrieved chunk IDs per query.
        relevant_chunks: List of relevant chunk IDs per query.

    Returns:
        {"hit_rate": Hit Rate, "mrr": MRR}
    """
    return {
        "hit_rate": hit_rate(retrieved_chunks, relevant_chunks),
        "mrr": mrr(retrieved_chunks, relevant_chunks),
        "count": len(retrieved_chunks),
    }


# ── File Evaluation ─────────────────────────────────────────────

def evaluate_from_file(
    qa_file: str,
    use_rougel: bool = False,
    use_bert: bool = False,
) -> Dict:
    """Read results from qa_results_*.json file and evaluate.

    JSON format:
    [{"query": "...", "answer": "...", "chunks": [...]}, ...]
    where answer is model-generated, chunks are retrieved chunk IDs.

    If each item also has a ground_truth field, accuracy will be computed.
    """
    if not os.path.exists(qa_file):
        raise FileNotFoundError(f"File not found: {qa_file}")

    with open(qa_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list) or not data:
        raise ValueError("File must be a non-empty JSON array")

    predictions = []
    ground_truths = []
    retrieved = []

    for item in data:
        pred = item.get("predict") or item.get("answer", "")
        gt = item.get("gt") or item.get("ground_truth") or item.get("answer")
        predictions.append(pred)
        ground_truths.append(gt)
        retrieved.append(item.get("chunk") or item.get("chunks", []))

    results = {}
    if ground_truths:
        results["qa"] = evaluate_qa(predictions, ground_truths, use_rougel, use_bert)
    if retrieved:
        results["retrieval"] = evaluate_retrieval(retrieved, [[]] * len(retrieved))

    return results


def print_report(results: Dict):
    """Print evaluation report in a user-friendly format."""
    print("\n" + "=" * 50)
    print("  Evaluation Report")
    print("=" * 50)

    if "qa" in results:
        qa = results["qa"]
        print(f"\n  QA Metrics ({qa.get('count', 0)} items):")
        print(f"    Exact Match: {qa.get('em', '-'):.2%}")
        if "rougel" in qa:
            print(f"    ROUGE-L:     {qa['rougel']:.2%}")
        if "bertscore" in qa:
            print(f"    BERTScore:   {qa['bertscore']:.4f}")

    if "retrieval" in results:
        ret = results["retrieval"]
        print(f"\n  Retrieval Metrics ({ret.get('count', 0)} items):")
        print(f"    Hit Rate:    {ret.get('hit_rate', '-'):.2%}")
        print(f"    MRR:         {ret.get('mrr', '-'):.4f}")

    print("=" * 50)
