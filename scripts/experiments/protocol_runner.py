#!/usr/bin/env python
"""Run fair cold/warm and locality workloads against an existing index."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import pathlib
import random
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.experiments.common import latency_metrics, qa_metrics, save_json
from src.CacheGraphRAG import CacheGraphRAG
from src.utils import get_config


def attach_gt(rows, qa_map):
    for row in rows:
        row["gt"] = qa_map.get(row.get("query", ""), row.get("gt", ""))
    return rows


async def run_query(app, questions, cfg):
    ret = cfg.get("retrieval", {})
    before_calls = getattr(app.llm, "total_calls", 0)
    before_prompt = app.llm.total_prompt_tokens
    before_completion = app.llm.total_completion_tokens
    rows = await app.query(
        questions=questions, start=0, end=len(questions),
        use_agentic=ret.get("agentic", False), agentic_steps=ret.get("agentic_steps", 3),
        qa_concurrency=ret.get("qa_concurrency", 1), entity_extraction=ret.get("entity_extraction", "milvus"),
        entity_index_name=ret.get("entity_index_name"), chunk_collection=ret.get("chunk_collection"),
        answer_topk=ret.get("answer_topk", 6), top_chunks=ret.get("top_chunks", 15),
        mode=ret.get("mode", "hybrid"), entity_promotion_threshold=ret.get("entity_promotion_threshold", 3),
    )
    usage = {"llm_calls": getattr(app.llm, "total_calls", 0) - before_calls,
             "prompt_tokens": app.llm.total_prompt_tokens - before_prompt,
             "completion_tokens": app.llm.total_completion_tokens - before_completion}
    return rows, list(app._latency_records), usage


async def fairness(app, cfg, args, output):
    app.load_dataset(args.start, args.end)
    qa_map = dict(zip(app.questions, app.answers))
    indices = list(range(len(app.questions)))
    random.Random(args.seed).shuffle(indices)
    n_warm = max(1, round(len(indices) * args.warmup_ratio))
    warm = [app.questions[i] for i in indices[:n_warm]]
    held = [app.questions[i] for i in indices[n_warm:]]
    if not held:
        raise ValueError("held-out set is empty; lower --warmup-ratio or increase range")

    base_gexf = cfg.get("retrieval", {}).get("base_gexf")
    app.mem_graph.persistent_graph.clear()
    if base_gexf:
        app.mem_graph.load_graph_gexf(base_gexf)
    cold, cold_lat, cold_usage = await run_query(app, held, cfg)
    cold = attach_gt(cold, qa_map)
    app.mem_graph.persistent_graph.clear()
    if base_gexf:
        app.mem_graph.load_graph_gexf(base_gexf)
    warm_rows, _, warm_usage = await run_query(app, warm, cfg)
    post, post_lat, post_usage = await run_query(app, held, cfg)
    post = attach_gt(post, qa_map)
    save_json(output / "cold_results.json", cold)
    save_json(output / "cold_latency.json", cold_lat)
    save_json(output / "warmup_results.json", attach_gt(warm_rows, qa_map))
    save_json(output / "post_warm_results.json", post)
    save_json(output / "post_warm_latency.json", post_lat)
    rows = []
    for name, values, lat, usage in [("cold", cold, cold_lat, cold_usage),
                                     ("post_warm", post, post_lat, post_usage)]:
        rows.append({"case": name, "seed": args.seed, "warmup_ratio": args.warmup_ratio,
                     **qa_metrics(values), **latency_metrics(lat), **usage})
    save_json(output / "summary.json", rows)
    save_json(output / "split.json", {"seed": args.seed, "warmup": warm, "held_out": held,
                                       "disjoint": not bool(set(warm) & set(held)),
                                       "warmup_usage": warm_usage})


def make_locality_sequence(questions, n, alpha, repeat_rate, drift, seed):
    rng = random.Random(seed)
    base = list(questions)
    if not base:
        return []
    split = max(1, len(base) // 2)
    first, second = base[:split], base[split:] or base

    def weighted(pool):
        weights = [1 / ((i + 1) ** alpha) if alpha > 0 else 1 for i in range(len(pool))]
        return rng.choices(pool, weights=weights, k=1)[0]

    seq = []
    for i in range(n):
        if seq and rng.random() < repeat_rate:
            seq.append(rng.choice(seq[max(0, len(seq)-10):]))
            continue
        progress = i / max(1, n - 1)
        use_second = rng.random() < drift * progress
        seq.append(weighted(second if use_second else first))
    return seq


async def locality(app, cfg, args, output):
    app.load_dataset(args.start, args.end)
    qa_map = dict(zip(app.questions, app.answers))
    seq = make_locality_sequence(app.questions, args.queries, args.zipf_alpha,
                                 args.repeat_rate, args.drift, args.seed)
    rows, latency, usage = await run_query(app, seq, cfg)
    rows = attach_gt(rows, qa_map)
    save_json(output / "workload.json", {"questions": seq, "zipf_alpha": args.zipf_alpha,
                                         "repeat_rate": args.repeat_rate, "drift": args.drift})
    save_json(output / "qa.json", rows); save_json(output / "latency.json", latency)
    save_json(output / "summary.json", [{"case": args.case_label or "locality", "seed": args.seed,
        "zipf_alpha": args.zipf_alpha, "repeat_rate": args.repeat_rate, "drift": args.drift,
        **qa_metrics(rows), **latency_metrics(latency), **usage,
        "evicted_chunks": app.mem_graph.evicted_chunks,
        "rehydrate_attempts": app.mem_graph.rehydrate_attempts,
        "rehydrate_successes": app.mem_graph.rehydrate_successes}])


async def stream_fig5(app, cfg, args, output):
    """图 5(Streaming & Rehydration)流式复现。

    从空 L1 + 空 L2 开始,文档按顺序分批流式 ingest 构建图,不加载任何
    预置 gexf、不复用克隆 L2、不引入外部数据。奇数批 ingest 后立即做
    第一轮 QA(1st odd,触发 L2 晋升 + LRU 驱逐),偶数批仅 ingest 不查询;
    全部流式 ingest 完成后显式 prune 模拟 severe churn,再做第二轮 QA:
    奇数批靠 L2(2nd odd),偶数批靠向量库 rehydrate 回退(2nd even)。

    输出:
      sample.json      采样索引/问题/答案(复现依据)
      qa.json          全部 QA(含 round 字段,gt+predict)
      first_odd.json   1st odd QA 结果
      second_odd.json  2nd odd QA 结果
      second_even.json 2nd even QA 结果
      latency.json     时延记录
      summary.json     三组指标的 EM/token_f1/时延/统计
    """
    # ── 1. 随机采样 n_qa 条(按原始顺序排序,保持文档流 chronological) ──
    from data.wikimultihopqa import get_2wikimultihopqa_info
    info = get_2wikimultihopqa_info()
    all_q, all_a, all_t = info["questions"], info["answers"], info["texts"]
    n_total = len(all_q)
    rng = random.Random(args.seed)
    indices = sorted(rng.sample(range(n_total), min(args.n_qa, n_total)))
    questions = [all_q[i] for i in indices]
    answers = [all_a[i] for i in indices]
    texts = [all_t[i] for i in indices]
    qa_map = dict(zip(questions, answers))
    save_json(output / "sample.json", {
        "seed": args.seed, "n_qa": len(questions),
        "indices": indices, "questions": questions, "answers": answers})
    print(f"[fig5] 采样 {len(questions)} 条(seed={args.seed}),分批 ingest")

    # ── 2. 确认空 L1 启动(协议不加载 base_gexf;L2 由 --empty 建空) ──
    print(f"[fig5] 空 L1: chunk_meta={len(app.mem_graph.chunk_meta)}, "
          f"nodes={app.mem_graph.graph.number_of_nodes()}, "
          f"edges={app.mem_graph.graph.number_of_edges()}")

    # ── 3. 流式分批 ingest:奇数批立即 1st odd QA,偶数批仅 ingest ──
    batch_size = max(1, args.batch_size)
    n_batches = max(1, (len(questions) + batch_size - 1) // batch_size)
    first_odd_rows = []
    first_odd_lat = []          # 按轮累计 latency 记录(l1/l2 hit 来源判定)
    usage_before = {"calls": getattr(app.llm, "total_calls", 0),
                    "prompt": app.llm.total_prompt_tokens,
                    "completion": app.llm.total_completion_tokens}
    for b in range(n_batches):
        lo, hi = b * batch_size, min((b + 1) * batch_size, len(questions))
        bq, bt = questions[lo:hi], texts[lo:hi]
        print(f"[fig5] ── 批次 {b}/{n_batches} ingest {len(bt)} 条文档 ──")
        await app.ingest(bt)
        print(f"       L1 chunk_meta={len(app.mem_graph.chunk_meta)}, "
              f"nodes={app.mem_graph.graph.number_of_nodes()}, "
              f"evicted={app.mem_graph.evicted_chunks}")
        if b % 2 == 0:
            rows, lat, _ = await run_query(app, bq, cfg)
            rows = attach_gt(rows, qa_map)
            for r in rows:
                r["round"] = "1st_odd"
                r["batch"] = b
            first_odd_rows.extend(rows)
            first_odd_lat.extend(lat)
            save_json(output / f"first_odd_batch{b}.json", rows)
            print(f"       批次 {b}(odd): 1st odd QA {len(rows)} 条完成")
        else:
            print(f"       批次 {b}(even): 仅 ingest,不查询")

    # ── 4. 二轮前显式 prune,模拟 L1 经历 severe churn ──
    app.mem_graph.prune_if_needed()
    print(f"[fig5] prune 后 L1 chunk_meta={len(app.mem_graph.chunk_meta)}, "
          f"evicted_total={app.mem_graph.evicted_chunks}, "
          f"rehydrate={app.mem_graph.rehydrate_successes}/{app.mem_graph.rehydrate_attempts}")

    # ── 5. 第二轮 QA:奇数批靠 L2,偶数批靠 rehydrate ──
    odd_q = [questions[i] for i in range(len(questions)) if (i // batch_size) % 2 == 0]
    even_q = [questions[i] for i in range(len(questions)) if (i // batch_size) % 2 == 1]
    rows_odd2, lat_odd2, _ = await run_query(app, odd_q, cfg)
    rows_odd2 = attach_gt(rows_odd2, qa_map)
    for r in rows_odd2:
        r["round"] = "2nd_odd"
    rows_even2, lat_even2, _ = await run_query(app, even_q, cfg)
    rows_even2 = attach_gt(rows_even2, qa_map)
    for r in rows_even2:
        r["round"] = "2nd_even"
    print(f"[fig5] 2nd odd {len(rows_odd2)} 条,2nd even {len(rows_even2)} 条完成")

    # ── 6. 汇总 ──
    all_qa = first_odd_rows + rows_odd2 + rows_even2
    all_lat = first_odd_lat + lat_odd2 + lat_even2
    save_json(output / "qa.json", all_qa)
    save_json(output / "first_odd.json", first_odd_rows)
    save_json(output / "second_odd.json", rows_odd2)
    save_json(output / "second_even.json", rows_even2)
    save_json(output / "latency.json", all_lat)
    usage = {"llm_calls": getattr(app.llm, "total_calls", 0) - usage_before["calls"],
             "prompt_tokens": app.llm.total_prompt_tokens - usage_before["prompt"],
             "completion_tokens": app.llm.total_completion_tokens - usage_before["completion"]}
    groups = [("1st_odd", first_odd_rows, first_odd_lat),
              ("2nd_odd", rows_odd2, lat_odd2),
              ("2nd_even", rows_even2, lat_even2)]
    summary = []
    for name, rows, lat in groups:
        summary.append({"case": name, "seed": args.seed, "n_qa": len(rows),
                        "batch_size": batch_size, **qa_metrics(rows),
                        **latency_metrics(lat), **usage,
                        "evicted_chunks": app.mem_graph.evicted_chunks,
                        "rehydrate_attempts": app.mem_graph.rehydrate_attempts,
                        "rehydrate_successes": app.mem_graph.rehydrate_successes})
    save_json(output / "summary.json", summary)
    print(f"[fig5] 完成: evicted={app.mem_graph.evicted_chunks}, "
          f"rehydrate={app.mem_graph.rehydrate_successes}/{app.mem_graph.rehydrate_attempts}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("protocol", choices=["fairness", "locality", "fixed", "stream_fig5"])
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--start", type=int, default=0); parser.add_argument("--end", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42); parser.add_argument("--warmup-ratio", type=float, default=.15)
    parser.add_argument("--queries", type=int, default=100); parser.add_argument("--zipf-alpha", type=float, default=1.0)
    parser.add_argument("--repeat-rate", type=float, default=.5); parser.add_argument("--drift", type=float, default=0.0)
    # 真实 case 名(由 run_experiments.py 传入);缺省时退回协议内建名,
    # 避免 summary 里所有 case 都叫 "fixed"/"locality" 无法区分(历史 bug)。
    parser.add_argument("--case-label", default="")
    # stream_fig5 专用参数
    parser.add_argument("--n-qa", type=int, default=200, help="stream_fig5: 随机采样 QA 数")
    parser.add_argument("--batch-size", type=int, default=20, help="stream_fig5: 每批文档数(批次奇偶交替)")
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    cfg = get_config(); app = CacheGraphRAG.from_config(cfg)
    try:
        # 图 5 流式复现要求空 L1 启动,不加载任何预置 gexf
        if args.protocol != "stream_fig5":
            base_gexf = cfg.get("retrieval", {}).get("base_gexf")
            if base_gexf and pathlib.Path(base_gexf).exists():
                app.mem_graph.load_graph_gexf(base_gexf)
        if args.protocol == "fairness":
            task = fairness(app, cfg, args, args.output)
        elif args.protocol == "stream_fig5":
            task = stream_fig5(app, cfg, args, args.output)
        elif args.protocol == "fixed":
            app.load_dataset(args.start, args.end)
            args.queries = min(args.queries, len(app.questions))
            # alpha=0/repeat=0/drift=0 produces the original held-out order only
            async def fixed_task():
                qa_map = dict(zip(app.questions, app.answers))
                questions = app.questions[:args.queries]
                rows, latency, usage = await run_query(app, questions, cfg)
                rows = attach_gt(rows, qa_map)
                save_json(args.output / "qa.json", rows); save_json(args.output / "latency.json", latency)
                # 晋升/缓存数据(R4-W4-1/R4-W2):本次 QA 触发 L2 晋升的 chunk、
                # 访问计数分布与 QA 结束后的 L1 规模。
                promoted = sorted(app.mem_graph.promoted_chunks)
                access_top = sorted(app.mem_graph.chunk_access_counter.items(),
                                    key=lambda x: x[1], reverse=True)[:20]
                save_json(args.output / "summary.json", [{"case": args.case_label or "fixed", "seed": args.seed,
                    **qa_metrics(rows), **latency_metrics(latency), **usage,
                    "evicted_chunks": app.mem_graph.evicted_chunks,
                    "rehydrate_attempts": app.mem_graph.rehydrate_attempts,
                    "rehydrate_successes": app.mem_graph.rehydrate_successes,
                    "promoted_chunks": len(promoted),
                    "promoted_chunk_ids": promoted[:100],
                    "chunks_touched": len(app.mem_graph.chunk_access_counter),
                    "access_counter_top": [[cid, cnt] for cid, cnt in access_top],
                    "l1_nodes_after": app.mem_graph.graph.number_of_nodes(),
                    "l1_edges_after": app.mem_graph.graph.number_of_edges()}])
            task = fixed_task()
        else:
            task = locality(app, cfg, args, args.output)
        asyncio.run(task)
    finally:
        app.shutdown()


if __name__ == "__main__":
    main()
