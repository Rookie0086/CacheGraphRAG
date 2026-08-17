#!/usr/bin/env python
"""Unified reviewer experiment matrix runner and visualizer.

The runner never edits config/config.yaml. Every case gets a config snapshot,
command log, raw artifacts, summary JSON/CSV and PNG/PDF figures.
"""
from __future__ import annotations

import argparse
import itertools
import os
import pathlib
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, List

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.experiments.common import (collect_standard_outputs, deep_update, dump_yaml,
    load_json, load_yaml, plot_summary, python_executable, run_command, save_json, write_csv)


EXPERIMENTS = (
    "fairness", "latency_cost", "locality", "lru_rehydrate", "embedding_batch",
    "beam_gamma", "loop_blocking", "cache_ablation", "storage", "table8_recheck",
    "alignment", "streaming_analysis", "stream_fig5",
)


def safe_name(text: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in text)


def redact_credentials(value):
    """Remove credentials from executable config snapshots; use env vars."""
    if isinstance(value, dict):
        return {key: ("" if key.lower() in {"api_key", "token", "password"}
                      else redact_credentials(item)) for key, item in value.items()}
    if isinstance(value, list):
        return [redact_credentials(item) for item in value]
    return value


def base_overrides(args) -> dict:
    return {
        "data.dataset": args.dataset, "data.start": args.start, "data.end": args.end,
        "model.seed": args.seed, "retrieval.skip_index": True,
        "retrieval.index_only": False, "retrieval.qa_cache": False,
        "retrieval.warmup_ratio": 0.0, "retrieval.clear_l2": args.clear_l2,
        "retrieval.qa_concurrency": args.qa_concurrency,
    }


def experiment_cases(name: str, args) -> List[dict]:
    base = base_overrides(args)
    source_cfg = load_yaml(ROOT / args.config)
    src_ret = source_cfg.get("retrieval", {})
    source_space = src_ret.get("nebula_space", args.dataset)
    source_collection = src_ret.get("chunk_collection", source_space)
    expected = ROOT / (f"subgraph/base/{args.dataset}_{source_collection}_{source_space}_"
                       f"{args.start}_{args.end}_base.gexf")
    if not expected.exists():
        matches = sorted((ROOT / "subgraph/base").glob(
            f"{args.dataset}_{source_collection}_{source_space}_*_base.gexf"))
        expected = matches[0] if matches else expected
    cases = []
    def add(label, updates=None, protocol="fixed", protocol_args=None, action="qa"):
        # 默认 protocol="fixed":普通保持顺序的 held-out QA。注意不要默认
        # "locality" —— 历史 bug 曾让 lru_rehydrate 等实验误跑 Zipf 负载
        # 协议(locality),且 summary 里 case 名被 protocol_runner 写死为
        # "locality",导致容量/重载消融数据标签错乱(见 run_20260815_002015)。
        isolated = {}
        if action == "qa":
            isolated = {"retrieval.nebula_space": f"exp_{safe_name(name)}_{safe_name(label)}",
                        "retrieval.l2_source_space": source_space,
                        "retrieval.l2_seed_mode": "empty" if name in ("fairness", "stream_fig5") else "clone",
                        "retrieval.fail_on_l2_error": True}
            # 图 5 流式复现要求空 L1 启动,不设置 base_gexf(协议不加载)
            if name != "stream_fig5":
                isolated["retrieval.base_gexf"] = str(expected)
        cases.append({"case": label, "updates": {**base, **isolated, **(updates or {})},
                      "protocol": protocol, "protocol_args": protocol_args or [], "action": action})

    if name == "fairness":
        for seed in args.seeds:
            add(f"seed_{seed}", {"model.seed": seed}, "fairness",
                ["--seed", str(seed), "--warmup-ratio", str(args.warmup_ratio)])
    elif name == "latency_cost":
        add("baseline", {"retrieval.agentic": False}, "fixed")
        add("agentic_b1", {"retrieval.agentic": True, "retrieval.beam_width": 1}, "fixed")
        add("agentic_b4", {"retrieval.agentic": True, "retrieval.beam_width": 4}, "fixed")
    elif name == "locality":
        for alpha in (0.0, 0.5, 1.0, 1.5):
            add(f"zipf_{alpha}", protocol="locality", protocol_args=["--zipf-alpha", str(alpha),
                "--repeat-rate", "0.5", "--drift", "0"])
        for repeat in (0.0, 0.25, 0.5, 0.75, 0.9):
            add(f"repeat_{repeat}", protocol="locality", protocol_args=["--zipf-alpha", "1.0",
                "--repeat-rate", str(repeat), "--drift", "0"])
        for drift in (0.0, 0.25, 0.5, 1.0):
            add(f"drift_{drift}", protocol="locality", protocol_args=["--zipf-alpha", "1.0",
                "--repeat-rate", "0.5", "--drift", str(drift)])
    elif name == "lru_rehydrate":
        for capacity in (50, 100, 200, 500):
            add(f"capacity_{capacity}", {"indexing.l1_max_chunks": capacity,
                "retrieval.enable_rehydrate": True})
        add("rehydrate_off", {"indexing.l1_max_chunks": 50, "retrieval.enable_rehydrate": False})
        add("rehydrate_on", {"indexing.l1_max_chunks": 50, "retrieval.enable_rehydrate": True})
    elif name == "embedding_batch":
        for size in (1, 16, 32, 64, 128):
            for wait in ((0, 10, 25) if size == 64 else (10,)):
                label = f"batch_{size}_wait_{wait}"
                unique = f"exp_embed_{safe_name(label)}_{args.start}_{args.end}"
                add(label, {"indexing.embed_batch_size": size, "indexing.embed_batch_wait_ms": wait,
                    "retrieval.index_only": True, "retrieval.skip_index": False,
                    "retrieval.chunk_collection": unique,
                    "retrieval.entity_index_name": f"entity_index_{unique}",
                    "retrieval.nebula_space": unique}, action="build")
    elif name == "beam_gamma":
        for beam in (1, 2, 4):
            add(f"beam_{beam}", {"retrieval.agentic": True, "retrieval.beam_width": beam,
                                 "retrieval.gamma": 0.5}, "fixed")
        for gamma in (0.3, 0.5, 0.7, 1.0):
            add(f"gamma_{gamma}", {"retrieval.agentic": True, "retrieval.beam_width": 4,
                                    "retrieval.gamma": gamma}, "fixed")
        for hops in (2, 3, 4):
            add(f"hops_{hops}", {"retrieval.agentic": True, "retrieval.beam_width": 4,
                                  "retrieval.max_hops": hops}, "fixed")
    elif name == "loop_blocking":
        add("none", {"retrieval.agentic": True, "retrieval.beam_width": 1,
                     "retrieval.enable_string_block": False, "retrieval.enable_semantic_block": False,
                     "retrieval.enable_unknown_block": False}, "fixed")
        add("string", {"retrieval.agentic": True, "retrieval.beam_width": 1,
                       "retrieval.enable_string_block": True, "retrieval.enable_semantic_block": False,
                       "retrieval.enable_unknown_block": False}, "fixed")
        add("semantic", {"retrieval.agentic": True, "retrieval.beam_width": 1,
                         "retrieval.enable_string_block": True, "retrieval.enable_semantic_block": True,
                         "retrieval.enable_unknown_block": False}, "fixed")
        add("semantic_unknown", {"retrieval.agentic": True, "retrieval.beam_width": 1,
                                 "retrieval.enable_string_block": True, "retrieval.enable_semantic_block": True,
                                 "retrieval.enable_unknown_block": True}, "fixed")
    elif name == "cache_ablation":
        add("vector_only", {"retrieval.enable_l1": False, "retrieval.enable_l2": False,
                            "retrieval.enable_rehydrate": False, "retrieval.agentic": False}, "fixed")
        add("l1_only", {"retrieval.enable_l1": True, "retrieval.enable_l2": False,
                        "retrieval.enable_rehydrate": False, "retrieval.agentic": False}, "fixed")
        add("l1_l2", {"retrieval.enable_l1": True, "retrieval.enable_l2": True,
                      "retrieval.enable_rehydrate": True, "retrieval.agentic": False}, "fixed")
        add("agentic_only", {"retrieval.enable_l1": False, "retrieval.enable_l2": False,
                             "retrieval.enable_rehydrate": False, "retrieval.agentic": True}, "fixed")
        add("full", {"retrieval.enable_l1": True, "retrieval.enable_l2": True,
                     "retrieval.enable_rehydrate": True, "retrieval.agentic": True}, "fixed")
    elif name == "table8_recheck":
        for seed in args.seeds:
            add(f"seed_{seed}", {"model.seed": seed}, "fixed")
    elif name == "storage":
        add("full_system", action="storage")
    elif name == "alignment":
        add("alignment_benchmarks", action="analysis")
    elif name == "streaming_analysis":
        add("dataset_statistics", action="analysis")
    elif name == "stream_fig5":
        # 图 5 流式复现:从空 L1/L2 重建,文档流式 ingest,不引入外部数据。
        # 关键:chunk_collection / entity_index_name / nebula_space 三者必须一致且唯一。
        # ingest() 写 Milvus 用 collection_name=self.nebula_space、实体索引用
        # "entity_index_" + self.nebula_space;query() 用 chunk_collection or nebula_space。
        # 若 base config 里的 wikimultihopqa 泄露进来,读写会落到生产 collection,rehydrate
        # 静默失败。这里全部设为 exp_ 前缀统一值,run_tag 会给三者打上同一后缀。
        shared = "exp_stream_fig5"
        add("fig5", {
            "retrieval.nebula_space": shared,
            "retrieval.chunk_collection": shared,
            "retrieval.entity_index_name": f"entity_index_{shared}",
            "retrieval.entity_promotion_threshold": 1,   # 一次命中即晋升,模拟 breaching hit-count
            "retrieval.enable_rehydrate": True,
            "retrieval.agentic": False,
            "retrieval.qa_concurrency": args.qa_concurrency,
            "indexing.l1_max_chunks": args.l1_max_chunks or 200,
        }, protocol="stream_fig5",
           protocol_args=["--n-qa", str(args.queries or 200), "--batch-size", "20"])
    return cases


def case_command(case, cfg_path, case_dir, args, py):
    action = case["action"]
    if action == "qa" and case["protocol"]:
        return [py, "scripts/experiments/protocol_runner.py", case["protocol"],
                "--output", str(case_dir / "artifacts"), "--start", str(args.start),
                "--end", str(args.end), "--seed", str(case["updates"].get("model.seed", args.seed)),
                "--queries", str(args.queries), "--case-label", case["case"],
                *case["protocol_args"]]
    if action == "build":
        return [py, "-m", "src.CacheGraphRAG"]
    if action == "storage":
        return [py, "scripts/storage_report.py", "--dataset", args.dataset,
                "--start", str(args.start), "--end", str(args.end),
                "--output", str(case_dir / "artifacts/storage.json")]
    return []


def run_analysis(name, case_dir, args, py, dry_run):
    if name == "alignment":
        command = [py, "scripts/experiments/alignment_benchmark.py", "--pairs", args.alignment_pairs,
                   "--output", str(case_dir / "artifacts")]
    else:
        command = [py, "scripts/experiments/streaming_dataset_analysis.py", "--dataset", args.dataset,
                   "--output", str(case_dir / "artifacts")]
    return run_command(command, case_dir, os.environ.copy(), dry_run)


def prepare_l2(cfg: dict, case_dir: pathlib.Path, env: dict, py: str,
               dry_run: bool) -> int:
    """Seed and verify an isolated L2 before a QA case starts."""
    ret = cfg.get("retrieval", {})
    if not ret.get("enable_l2", True):
        return 0
    source = str(ret.get("l2_source_space", "")).strip()
    target = str(ret.get("nebula_space", "")).strip()
    if not source or not target:
        raise ValueError("L2-enabled QA cases require l2_source_space and nebula_space")
    command = [py, "scripts/experiments/nebula_clone.py",
               "--source", source, "--target", target,
               "--output", str(case_dir / "artifacts/l2_seed.json")]
    if ret.get("l2_seed_mode", "clone") == "empty":
        command.append("--empty")
    save_json(case_dir / "l2_seed_command.json", {"argv": command, "cwd": str(ROOT)})
    if dry_run:
        print("[dry-run]", " ".join(command))
        return 0
    with (case_dir / "l2_seed.log").open("w", encoding="utf-8") as log:
        proc = subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                              stderr=subprocess.STDOUT, check=False)
    return proc.returncode


def run_experiment(name: str, root_out: pathlib.Path, args) -> List[dict]:
    exp_dir = root_out / name; exp_dir.mkdir(parents=True, exist_ok=True)
    base_cfg = load_yaml(ROOT / args.config)
    py = python_executable(args.python)
    rows = []
    for case in experiment_cases(name, args):
        label = case["case"]; case_dir = exp_dir / safe_name(label); case_dir.mkdir(parents=True, exist_ok=True)
        (case_dir / "artifacts").mkdir(parents=True, exist_ok=True)
        cfg = redact_credentials(deep_update(load_yaml(ROOT / args.config), case["updates"]))
        run_tag = safe_name(root_out.name)[-24:]
        ret = cfg.setdefault("retrieval", {})
        # 凡是含 exp_ 的隔离资源(qa/build 均适用,entity_index_* 前缀也要覆盖)
        # 都加 run_tag,避免重跑时复用旧 collection/index/space 造成数据污染
        for key in ("chunk_collection", "entity_index_name", "nebula_space"):
            val = ret.get(key)
            if val and "exp_" in str(val):
                ret[key] = f"{val}_{run_tag}"[:63]
        cfg_path = case_dir / "config.yaml"; dump_yaml(cfg_path, cfg)
        save_json(case_dir / "case.json", case)
        env = os.environ.copy(); env["CACHEGRAPH_CONFIG"] = str(cfg_path)
        env.setdefault("MPLCONFIGDIR", str(root_out / ".mplconfig"))
        started = time.time()
        if case["action"] == "analysis":
            code = run_analysis(name, case_dir, args, py, args.dry_run)
        else:
            command = case_command(case, cfg_path, case_dir, args, py)
            seed_code = (prepare_l2(cfg, case_dir, env, py, args.dry_run)
                         if case["action"] == "qa" else 0)
            code = (run_command(command, case_dir, env, args.dry_run)
                    if seed_code == 0 else seed_code)
        summary_path = case_dir / "artifacts/summary.json"
        summary = load_json(summary_path, []) or []
        if (not summary and not args.dry_run and code == 0 and
                case["action"] in {"build", "qa"}):
            summary = [collect_standard_outputs(case_dir, args.dataset, args.start, args.end, started)]
        if not summary and case["action"] == "storage":
            storage = load_json(case_dir / "artifacts/storage.json", {}) or {}
            breakdown = storage.get("breakdown", storage)
            summary = [{"case": label, "total_bytes": storage.get("total_bytes", 0),
                        **{f"bytes_{k}": v.get("bytes", v) if isinstance(v, dict) else v
                           for k, v in breakdown.items()}}]
        if (name == "table8_recheck" and not args.dry_run and code == 0 and
                (case_dir / "artifacts/qa.json").exists() and summary_path.exists()):
            bert_cmd = [py, "scripts/experiments/add_bertscore.py", "--qa",
                        str(case_dir / "artifacts/qa.json"), "--summary", str(summary_path)]
            with (case_dir / "bertscore.log").open("w", encoding="utf-8") as log:
                subprocess.run(bert_cmd, cwd=ROOT, env=env, stdout=log,
                               stderr=subprocess.STDOUT, check=True)
            summary = load_json(summary_path, []) or summary
        l2_seed = load_json(case_dir / "artifacts/l2_seed.json", {}) or {}
        if not summary and code:
            summary = [{"case": label}]
        for item in summary:
            item.update({"experiment": name, "case": item.get("case", label),
                         "returncode": code,
                         "l2_seed_mode": l2_seed.get("mode", "disabled"),
                         "l2_seed_verified": l2_seed.get("verified", False),
                         "l2_vertices_before": (l2_seed.get("target_before") or {}).get("vertices", 0),
                         "l2_vertices_seeded": (l2_seed.get("target_after") or {}).get("vertices", 0),
                         "l2_edges_seeded": (l2_seed.get("target_after") or {}).get("edges", 0)})
            rows.append(item)
        if code and not args.keep_going:
            raise SystemExit(f"case failed: {name}/{label}; see {case_dir/'stdout.log'}")
    save_json(exp_dir / "summary.json", rows); write_csv(exp_dir / "summary.csv", rows)
    plot_summary(rows, exp_dir / "figures")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiments", nargs="*", default=[])
    parser.add_argument("--all", action="store_true"); parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-going", action="store_true"); parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--python"); parser.add_argument("--dataset", default="wikimultihopqa")
    parser.add_argument("--start", type=int, default=0); parser.add_argument("--end", type=int, default=100)
    parser.add_argument("--queries", type=int, default=100); parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--warmup-ratio", type=float, default=.15)
    parser.add_argument("--qa-concurrency", type=int, default=1)
    parser.add_argument("--l1-max-chunks", type=int,
                        help="stream_fig5: L1 容量上限(触发 LRU 驱逐)")
    parser.add_argument("--clear-l2", action="store_true")
    parser.add_argument("--alignment-pairs", default="data/annotations/entity_alignment_pairs.jsonl")
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()
    invalid = sorted(set(args.experiments) - set(EXPERIMENTS))
    if invalid:
        parser.error(f"unknown experiments: {', '.join(invalid)}; choices: {', '.join(EXPERIMENTS)}")
    selected = list(EXPERIMENTS) if args.all else args.experiments
    if not selected:
        parser.error("choose one or more experiments, or use --all")
    stamp = time.strftime("%Y%m%d_%H%M%S")
    root_out = args.output or ROOT / f"output/experiments/run_{stamp}"
    root_out.mkdir(parents=True, exist_ok=True)
    save_json(root_out / "run.json", {"experiments": selected, "argv": sys.argv,
              "dataset": args.dataset, "range": [args.start, args.end], "dry_run": args.dry_run})
    all_rows = []
    for name in selected:
        print(f"\n=== {name} ===")
        all_rows.extend(run_experiment(name, root_out, args))
    save_json(root_out / "summary.json", all_rows); write_csv(root_out / "summary.csv", all_rows)
    print(f"\nResults: {root_out}")


if __name__ == "__main__":
    main()
