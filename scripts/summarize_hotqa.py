#!/usr/bin/env python
"""汇总 hotqa_promotion 运行:晋升数据、命中率、时延、L2 增长。"""
import json, pathlib, sys

run = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "output/hotqa_promotion")
runs = sorted(run.glob("run_*"))
if not runs:
    print("no runs found"); sys.exit(1)
latest = runs[-1]
art = latest / "artifacts"
s = json.load(open(art / "summary.json"))[0]
qa = json.load(open(art / "qa.json"))
lat = json.load(open(art / "latency.json"))
seed = json.load(open(art / "l2_seed.json"))

print(f"== {latest.name} ==")
print(f"QA: {s['count']} 条 | EM={s['em']:.3f} token_f1={s['token_f1']:.3f} dont_know={s['dont_know_rate']:.2f}")
print(f"时延: total p50={s['total_p50_s']:.1f}s p95={s['total_p95_s']:.1f}s | "
      f"retrieve p50={s['retrieve_p50_s']:.1f}s | answer p50={s['answer_p50_s']:.1f}s")
print(f"命中: l1_hit_rate={s['l1_hit_rate']:.2f} l2_hit_rate={s['l2_hit_rate']:.2f} "
      f"l2_errors={s['l2_query_errors']}")
print(f"LLM: calls={s['llm_calls']} prompt_tokens={s['prompt_tokens']} completion_tokens={s['completion_tokens']}")
print(f"缓存: evicted_chunks={s['evicted_chunks']} | rehydrate {s['rehydrate_successes']}/{s['rehydrate_attempts']}")
print(f"晋升: promoted_chunks={s['promoted_chunks']} | chunks_touched={s['chunks_touched']} "
      f"(τ_hit=3)")
print(f"L1 规模(QA 后): nodes={s['l1_nodes_after']} edges={s['l1_edges_after']}")
# access 计数分布
ac = dict(s.get('access_counter_top', []))
if ac:
    print("access 计数 Top10:")
    for cid, cnt in list(ac.items())[:10]:
        print(f"   {cnt}x  {cid}")
# L2 增长
src = seed.get("source", {}); after = seed.get("target_after", {})
print(f"L2(克隆): 基线 {after.get('vertices')}/{after.get('edges')}(源 {src.get('vertices')}/{src.get('edges')})")
print(f"run 目录: {latest}")
