#!/usr/bin/env python
"""Analyze update frequency, growth, recurrence and drift in streaming datasets."""
from __future__ import annotations
import argparse, collections, json, os, pathlib, re, sys
ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from scripts.experiments.common import save_json, write_csv


def tokens(text):
    return set(re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{2,}", str(text).lower()))


def whoqa_rows():
    path = ROOT / "data/datasets/whoqa/whoqa_experiment_dataset_600.json"
    raw = json.load(path.open(encoding="utf-8")); rows=[]; seen=set(); prev=set()
    for idx, item in enumerate(raw):
        p1, p2 = item.get("phase_1_data", []), item.get("phase_2_data", [])
        p1 = p1 if isinstance(p1, list) else [p1]; p2 = p2 if isinstance(p2, list) else [p2]
        target = str(item.get("target_entity") or item.get("page_id") or f"item_{idx}")
        current = tokens(" ".join(map(str, p2)))
        rows.append({"step": idx+1, "target": target, "phase1_docs": len(p1), "phase2_docs": len(p2),
            "phase1_bytes": sum(len(str(x).encode()) for x in p1),
            "phase2_bytes": sum(len(str(x).encode()) for x in p2),
            "target_repeated": int(target.lower() in seen),
            "topic_jaccard_prev": len(current & prev) / len(current | prev) if current | prev else 0})
        seen.add(target.lower()); prev=current
    return rows


def generic_rows(dataset):
    from src.CacheGraphRAG import _DATASET_LOADERS
    info = _DATASET_LOADERS[dataset](dataset); rows=[]; prev=set(); seen=collections.Counter()
    for idx, text in enumerate(info["texts"]):
        current=tokens(text); repeated=sum(1 for t in current if seen[t])
        rows.append({"step":idx+1,"documents":1,"bytes":len(str(text).encode()),
                     "unique_terms":len(current),"repeated_terms":repeated,
                     "topic_jaccard_prev":len(current&prev)/len(current|prev) if current|prev else 0})
        seen.update(current); prev=current
    return rows


def plot(rows, output):
    os.environ.setdefault("MPLCONFIGDIR", str(output/".mplconfig"))
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xs=[r["step"] for r in rows]; bytes_key="phase2_bytes" if "phase2_bytes" in rows[0] else "bytes"
    cumulative=[]; total=0
    for r in rows: total += r.get(bytes_key,0); cumulative.append(total)
    fig, axes=plt.subplots(1,2,figsize=(11,4.5))
    axes[0].plot(xs,cumulative); axes[0].set_title("Cumulative streamed bytes"); axes[0].set_xlabel("update step")
    axes[1].plot(xs,[r.get("topic_jaccard_prev",0) for r in rows]); axes[1].set_title("Adjacent-step topic similarity")
    axes[1].set_xlabel("update step"); axes[1].set_ylim(0,1)
    fig.tight_layout(); fig.savefig(output/"streaming_characteristics.png",dpi=180)
    fig.savefig(output/"streaming_characteristics.pdf"); plt.close(fig)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--dataset",default="whoqa"); ap.add_argument("--output",required=True,type=pathlib.Path)
    args=ap.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    rows=whoqa_rows() if "whoqa" in args.dataset else generic_rows(args.dataset)
    summary={"case":"dataset_statistics","dataset":args.dataset,"steps":len(rows),
             "total_bytes":sum(r.get("phase1_bytes",0)+r.get("phase2_bytes",r.get("bytes",0)) for r in rows),
             "mean_topic_jaccard":sum(r.get("topic_jaccard_prev",0) for r in rows)/len(rows) if rows else 0,
             "repeated_targets":sum(r.get("target_repeated",0) for r in rows)}
    save_json(args.output/"rows.json",rows); save_json(args.output/"summary.json",[summary]); write_csv(args.output/"rows.csv",rows)
    if rows: plot(rows,args.output)
if __name__=="__main__": main()
