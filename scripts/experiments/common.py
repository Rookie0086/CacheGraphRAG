#!/usr/bin/env python
"""Shared configuration, execution, collection and plotting helpers."""
from __future__ import annotations

import csv
import json
import os
import pathlib
import shutil
import statistics
import subprocess
import sys
import time
from typing import Any, Dict, Iterable, List

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_PYTHON = ROOT / ".conda/cachegraphrag-mac/bin/python"


def deep_set(data: dict, dotted: str, value: Any) -> None:
    cur = data
    parts = dotted.split(".")
    for part in parts[:-1]:
        cur = cur.setdefault(part, {})
    cur[parts[-1]] = value


def deep_update(base: dict, updates: Dict[str, Any]) -> dict:
    for key, value in updates.items():
        deep_set(base, key, value)
    return base


def load_yaml(path: pathlib.Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def dump_yaml(path: pathlib.Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, allow_unicode=True, sort_keys=False)


def load_json(path: pathlib.Path, default=None):
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return default


def save_json(path: pathlib.Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def percentile(values: Iterable[float], pct: float) -> float:
    values = sorted(float(v) for v in values)
    if not values:
        return 0.0
    idx = min(len(values) - 1, round((len(values) - 1) * pct / 100))
    return values[idx]


def normalize(text: Any) -> str:
    import re
    value = str(text or "").lower().strip()
    value = re.sub(r"\b(a|an|the)\b", " ", value)
    value = re.sub(r"[^\w\s]", " ", value)
    return " ".join(value.split())


def qa_metrics(rows: List[dict]) -> dict:
    if not rows:
        return {"count": 0, "em": 0.0, "token_f1": 0.0, "dont_know_rate": 0.0}
    ems, f1s, unknown = [], [], 0
    for row in rows:
        pred, gt = normalize(row.get("predict")), normalize(row.get("gt"))
        ems.append(float(pred == gt and bool(gt)))
        p, g = pred.split(), gt.split()
        overlap = sum(min(p.count(tok), g.count(tok)) for tok in set(p))
        precision = overlap / len(p) if p else 0.0
        recall = overlap / len(g) if g else 0.0
        f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
        unknown += int("don't know" in pred or "dont know" in pred or not pred)
    return {
        "count": len(rows), "em": sum(ems) / len(ems),
        "token_f1": sum(f1s) / len(f1s),
        "dont_know_rate": unknown / len(rows),
        "avg_agentic_steps": statistics.fmean(
            float(row.get("agentic_steps_count", 0)) for row in rows),
        "total_block_count": sum(int(row.get("block_count", 0)) for row in rows),
        "avg_agentic_planning_calls": statistics.fmean(
            float(row.get("agentic_planning_calls", 0)) for row in rows),
        "total_batched_planning_calls": sum(
            int(row.get("agentic_batched_planning_calls", 0)) for row in rows),
    }


def latency_metrics(rows: List[dict]) -> dict:
    totals = [float(r.get("retrieve_s", 0)) + float(r.get("answer_s", 0)) for r in rows]
    retrieve = [float(r.get("retrieve_s", 0)) for r in rows]
    answer = [float(r.get("answer_s", 0)) for r in rows]
    result = {
        "latency_count": len(rows), "total_p50_s": percentile(totals, 50),
        "total_p95_s": percentile(totals, 95),
        "retrieve_p50_s": percentile(retrieve, 50), "retrieve_p95_s": percentile(retrieve, 95),
        "answer_p50_s": percentile(answer, 50), "answer_p95_s": percentile(answer, 95),
        "l1_hit_rate": sum(int(r.get("l1_hits", 0) > 0) for r in rows) / len(rows) if rows else 0.0,
        "l2_hit_rate": sum(int(r.get("l2_hits", 0) > 0) for r in rows) / len(rows) if rows else 0.0,
        "l2_query_errors": sum(int(r.get("l2_query_errors", 0)) for r in rows),
        "retrieval_calls": sum(int(r.get("retrieval_calls", 1)) for r in rows),
        "l2_measurement_valid": not any(int(r.get("l2_query_errors", 0)) for r in rows),
    }
    stages = sorted({key for row in rows for key in row.get("stages", {})})
    for stage in stages:
        vals = [float(row.get("stages", {}).get(stage, 0)) for row in rows]
        result[f"stage_{stage}_avg_s"] = statistics.fmean(vals) if vals else 0.0
    return result


def newest(directory: pathlib.Path, pattern: str, since: float = 0) -> pathlib.Path | None:
    files = [p for p in directory.glob(pattern) if p.stat().st_mtime >= since]
    return max(files, key=lambda p: p.stat().st_mtime) if files else None


def collect_standard_outputs(case_dir: pathlib.Path, dataset: str, start: int, end: int,
                             since: float) -> dict:
    artifacts = case_dir / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    candidates = {
        "qa": ROOT / f"output/qa/qa_results_{dataset}_{start}_{end}.json",
        "latency": ROOT / f"output/qa/qa_latency_{dataset}_{start}_{end}.json",
    }
    log_json = newest(ROOT / "log", f"pipeline_{dataset}_*.json", since)
    if log_json:
        candidates["pipeline"] = log_json
    copied = {}
    for label, source in candidates.items():
        if source and source.exists() and source.stat().st_mtime >= since:
            target = artifacts / f"{label}.json"
            shutil.copy2(source, target)
            copied[label] = str(target.relative_to(case_dir))
    qa = load_json(artifacts / "qa.json", []) or []
    latency = load_json(artifacts / "latency.json", []) or []
    pipeline = load_json(artifacts / "pipeline.json", {}) or {}
    runtime_keys = ("embedding_api_calls", "embedding_items", "evicted_chunks", "evicted_nodes",
                    "evicted_edges", "rehydrate_attempts", "rehydrate_successes", "rehydrate_failures",
                    "rehydrated_nodes", "rehydrated_edges", "l1_nodes", "l1_edges")
    return {**qa_metrics(qa), **latency_metrics(latency),
            "prompt_tokens": pipeline.get("prompt_tokens", 0),
            "completion_tokens": pipeline.get("completion_tokens", 0),
            "llm_calls": pipeline.get("llm_calls", 0),
            "total_time_s": pipeline.get("total_time_s", 0),
            **{key: pipeline.get(key, 0) for key in runtime_keys}, "artifacts": copied}


def run_command(command: List[str], case_dir: pathlib.Path, env: dict, dry_run: bool) -> int:
    save_json(case_dir / "command.json", {"argv": command, "cwd": str(ROOT)})
    if dry_run:
        print("[dry-run]", " ".join(command))
        return 0
    case_dir.mkdir(parents=True, exist_ok=True)
    with (case_dir / "stdout.log").open("w", encoding="utf-8") as log:
        proc = subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                              stderr=subprocess.STDOUT, check=False)
    return proc.returncode


def write_csv(path: pathlib.Path, rows: List[dict]) -> None:
    if not rows:
        return
    keys = sorted({key for row in rows for key in row if key != "artifacts"})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows({k: row.get(k, "") for k in keys} for row in rows)


def plot_summary(rows: List[dict], out_dir: pathlib.Path, x_key: str = "case") -> None:
    if not rows:
        return
    os.environ.setdefault("MPLCONFIGDIR", str(out_dir / ".mplconfig"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    labels = [str(row.get(x_key, row.get("case", i))) for i, row in enumerate(rows)]
    charts = [
        ("quality", ["em", "token_f1"], "Score", "QA quality"),
        ("latency", ["total_p50_s", "total_p95_s"], "Seconds", "End-to-end latency"),
        ("cache_hits", ["l1_hit_rate", "l2_hit_rate"], "Rate", "Cache hit rate"),
        ("tokens", ["prompt_tokens", "completion_tokens"], "Tokens", "LLM token usage"),
        ("agentic", ["avg_agentic_steps", "total_block_count"], "Count", "Agentic control flow"),
        ("embedding", ["embedding_api_calls", "embedding_items"], "Count", "Embedding batching"),
        ("cache_events", ["evicted_chunks", "rehydrate_attempts", "rehydrate_successes"],
         "Count", "LRU and topology reload events"),
        ("storage", ["total_bytes"], "Bytes", "Total system storage"),
    ]
    for filename, metrics, ylabel, title in charts:
        if not any(any(metric in row for metric in metrics) for row in rows):
            continue
        fig, ax = plt.subplots(figsize=(max(7, len(rows) * 1.15), 4.8))
        width = 0.8 / len(metrics)
        xs = list(range(len(rows)))
        for idx, metric in enumerate(metrics):
            values = [float(row.get(metric, 0) or 0) for row in rows]
            ax.bar([x + (idx - (len(metrics)-1)/2) * width for x in xs], values,
                   width=width, label=metric)
        ax.set_xticks(xs, labels, rotation=30, ha="right")
        ax.set_ylabel(ylabel); ax.set_title(title); ax.legend(); ax.grid(axis="y", alpha=.25)
        fig.tight_layout()
        fig.savefig(out_dir / f"{filename}.png", dpi=180)
        fig.savefig(out_dir / f"{filename}.pdf")
        plt.close(fig)


def python_executable(explicit: str | None = None) -> str:
    if explicit:
        return explicit
    return str(DEFAULT_PYTHON if DEFAULT_PYTHON.exists() else pathlib.Path(sys.executable))
