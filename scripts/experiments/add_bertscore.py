#!/usr/bin/env python
"""Add semantic-score statistics to an existing experiment case."""
import argparse, json, pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from src.entity.bert_sim import bert

ap = argparse.ArgumentParser(); ap.add_argument("--qa", required=True, type=pathlib.Path)
ap.add_argument("--summary", required=True, type=pathlib.Path)
args = ap.parse_args()
rows = json.load(args.qa.open(encoding="utf-8"))
scores = [bert(row.get("predict", ""), row.get("gt", "")) for row in rows]
summary = json.load(args.summary.open(encoding="utf-8"))
if not isinstance(summary, list): summary = [summary]
for item in summary:
    item["bertscore"] = sum(scores) / len(scores) if scores else 0.0
    item["bertscore_count"] = len(scores)
json.dump(summary, args.summary.open("w", encoding="utf-8"), ensure_ascii=False, indent=2)
