#!/usr/bin/env python
"""Entity-alignment benchmark with labeled pairs and external prediction adapters.

Input JSONL fields: left{name,type,desc}, right{name,type,desc}, is_same.
Built-ins are transparent proxies, not claims of running entire external systems:
  light_name  - normalized exact-name key (LightRAG-style)
  hippo_vec   - embedding cosine threshold (HippoRAG-style synonym edge)
  funnel      - name/type/alias gate followed by embedding threshold
External predictions can be added with --prediction METHOD=FILE.jsonl where each
line contains either {prediction: 0/1} or {is_same: 0/1} in matching order.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import sys
import time
from difflib import SequenceMatcher

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from scripts.experiments.common import save_json, write_csv
from src.llm.env import EmbeddingEnv, APIEmbeddingEnv
from src.utils import get_config


def norm(name):
    value = str(name or "").lower().strip()
    value = re.sub(r"\b(inc|corp|co|ltd|llc|group|foundation|corporation|limited)\b", " ", value)
    return " ".join(re.sub(r"[^a-z0-9\s]", " ", value).split())


def alias(a, b):
    a, b = norm(a), norm(b)
    if not a or not b: return False
    if a == b: return True
    short, long = (a, b) if len(a) < len(b) else (b, a)
    sw, lw = short.split(), long.split()
    if len(sw) == 1 and sw[0] in lw: return True
    if len(sw) == len(lw) and all(y.startswith(x) for x, y in zip(sw, lw)): return True
    return SequenceMatcher(None, a, b).ratio() > .85


def metrics(labels, predictions, elapsed, calls=0):
    tp = sum(y == 1 and p == 1 for y, p in zip(labels, predictions))
    fp = sum(y == 0 and p == 1 for y, p in zip(labels, predictions))
    fn = sum(y == 1 and p == 0 for y, p in zip(labels, predictions))
    tn = len(labels) - tp - fp - fn
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    return {"count": len(labels), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "accuracy": (tp + tn) / len(labels) if labels else 0,
            "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0,
            "elapsed_s": elapsed, "embedding_calls": calls}


def read_jsonl(path):
    with pathlib.Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def make_template(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    examples = [
        {"left": {"name": "International Business Machines", "type": "ORG", "desc": "technology company"},
         "right": {"name": "IBM", "type": "ORG", "desc": "American technology company"}, "is_same": 1},
        {"left": {"name": "Paris", "type": "CITY", "desc": "capital of France"},
         "right": {"name": "Paris", "type": "PERSON", "desc": "mythological prince"}, "is_same": 0},
    ]
    with path.open("w", encoding="utf-8") as handle:
        for row in examples: handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def plot(rows, out):
    os.environ.setdefault("MPLCONFIGDIR", str(out / ".mplconfig"))
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    methods = [r["method"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].bar(methods, [r["f1"] for r in rows]); axes[0].set_ylim(0, 1)
    axes[0].set_title("Entity alignment F1"); axes[0].tick_params(axis="x", rotation=25)
    axes[1].bar(methods, [r["elapsed_s"] for r in rows]); axes[1].set_title("Alignment latency")
    axes[1].set_ylabel("seconds"); axes[1].tick_params(axis="x", rotation=25)
    fig.tight_layout(); fig.savefig(out / "alignment_benchmark.png", dpi=180)
    fig.savefig(out / "alignment_benchmark.pdf"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", required=True); ap.add_argument("--output", required=True, type=pathlib.Path)
    ap.add_argument("--threshold", type=float, default=.85)
    ap.add_argument("--prediction", action="append", default=[], help="METHOD=predictions.jsonl")
    args = ap.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    pair_path = pathlib.Path(args.pairs)
    if not pair_path.exists():
        make_template(pair_path)
        save_json(args.output / "status.json", {"status": "needs_annotation", "template": str(pair_path),
                  "message": "Replace/extend the two examples with labeled entity pairs, then rerun."})
        print(f"Annotation template created: {pair_path}"); return
    pairs = read_jsonl(pair_path); labels = [int(bool(x["is_same"])) for x in pairs]
    rows, detail = [], {}
    started = time.perf_counter()
    pred = [int(norm(x["left"]["name"]) == norm(x["right"]["name"])) for x in pairs]
    rows.append({"method": "light_name_proxy", **metrics(labels, pred, time.perf_counter()-started)})
    detail["light_name_proxy"] = pred

    cfg = get_config().get("embedding", {})
    if cfg.get("backend") == "api":
        embed = APIEmbeddingEnv(model_name=cfg.get("model_name"), api_key=cfg.get("api_key"),
                                base_url=cfg.get("base_url"), batch_size=64)
    else:
        embed = EmbeddingEnv(model_name=cfg.get("model_name", "BAAI/bge-m3"), batch_size=64)
    texts = []
    for pair in pairs:
        for side in ("left", "right"):
            e = pair[side]; texts.append(f"Entity: {e.get('name','')}. Type: {e.get('type','')}. Description: {e.get('desc','')}")
    started = time.perf_counter(); vectors = embed.get_embeddings(texts)
    import numpy as np
    pred_vec, pred_funnel = [], []
    for i, pair in enumerate(pairs):
        a, b = np.asarray(vectors[2*i]), np.asarray(vectors[2*i+1])
        sim = float(a @ b / ((np.linalg.norm(a) * np.linalg.norm(b)) or 1))
        pred_vec.append(int(sim >= args.threshold))
        same_type = norm(pair["left"].get("type")) == norm(pair["right"].get("type"))
        name_gate = alias(pair["left"].get("name"), pair["right"].get("name"))
        pred_funnel.append(int(same_type and (name_gate or sim >= args.threshold)))
    elapsed = time.perf_counter() - started
    rows.append({"method": "hippo_vec_proxy", **metrics(labels, pred_vec, elapsed, 1)})
    rows.append({"method": "cachegraph_funnel", **metrics(labels, pred_funnel, elapsed, 1)})
    detail.update({"hippo_vec_proxy": pred_vec, "cachegraph_funnel": pred_funnel})

    for spec in args.prediction:
        method, file = spec.split("=", 1); external = read_jsonl(file)
        predictions = [int(bool(x.get("prediction", x.get("is_same")))) for x in external]
        if len(predictions) != len(labels): raise ValueError(f"{method}: prediction count mismatch")
        rows.append({"method": method, **metrics(labels, predictions, 0)})
        detail[method] = predictions
    save_json(args.output / "summary.json", rows); save_json(args.output / "predictions.json", detail)
    write_csv(args.output / "summary.csv", rows); plot(rows, args.output)


if __name__ == "__main__": main()
