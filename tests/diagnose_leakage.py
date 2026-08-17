#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""诊断实验:测量 query-driven promotion 导致的测试泄漏量(回应审稿人 R4-W2)。

核心问题
--------
论文 Section V-F 的流式实验中,warm-up 查询(odd batches)与评测查询同源:
同一批问题既驱动了 L2 promotion(获得"后见之明"),又被用来评估 compact 图的
ACC,导致 accuracy–footprint 对比不公平。

本脚本在完全相同的 held-out 评测集上跑三个对照场景,量化泄漏量:

  场景 A  冻结图(无 promotion)基线    评测期间 L2 永不写入 → 公平下界
  场景 B  独立 warm-up 驱动 promotion  用与评测集不相交的 warm-up 集填充 L2,
                                      冻结后评测 held-out → 修复后的公平流程
  场景 C  随机触发 promotion 对照      随机选 chunk 触发 promotion 填充 L2,
                                      冻结后评测 held-out → 随机下界

判定规则
--------
  场景A vs 论文 Table III → 论文数字中被泄漏抬高了多少
  场景B vs 场景A         → 独立 warm-up 能否恢复收益
  场景B vs 场景C         → 若 B ≈ C:promotion 主要是架构收益(干净);
                           若 B >> C:收益来自查询情报,需进一步检查覆盖度

用法
----
  # 全量(默认),RGB 数据集,评测 100 题
  python diagnose_leakage.py --eval-n 100 --warmup-n 100

  # 快速冒烟(小规模,先验证链路)
  python diagnose_leakage.py --limit 12 --scenario frozen

  # 指定数据集(需先在 config 中准备好对应 nebula_space / Milvus collection)
  python diagnose_leakage.py --dataset wikimultihopqa --start 0 --end 300

输出
----
  output/diagnose_leakage_{dataset}.json  各场景 ACC/L2 规模 + 论文参照
  控制台对比表
"""

import argparse
import asyncio
import json
import os
import random
import sys
from pathlib import Path

# 允许从仓库根目录直接运行
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.CacheGraphRAG import CacheGraphRAG
from src.utils.base import get_config
from src.eval import evaluate_qa

# 冻结晋升:让 access_chunk 的 `current_count == self.threshold` 永不满足
FROZEN_THRESHOLD = 10 ** 9

# 论文 Table III:CacheGraphRAG 全量 ACC(参照值,非逐场景可比)
PAPER_TABLE3 = {
    "rgb": 97.67,
    "2wiki": 73.17,
    "hotpotqa": 68.30,
}


def parse_args():
    p = argparse.ArgumentParser(description="诊断 query-driven promotion 的测试泄漏量")
    p.add_argument("--dataset", default=None, help="数据集名(默认读 config.yaml 的 data.dataset)")
    p.add_argument("--start", type=int, default=0, help="问题起始下标")
    p.add_argument("--end", type=int, default=-1, help="问题结束下标(-1 = 全量)")
    p.add_argument("--seed", type=int, default=42, help="warm-up / held-out 切分随机种子")
    p.add_argument("--eval-n", type=int, default=100, help="held-out 评测题数")
    p.add_argument("--warmup-n", type=int, default=100, help="warm-up 题数(与评测集不相交)")
    p.add_argument("--random-chunks", type=int, default=60,
                   help="场景 C 随机触发 promotion 的 chunk 数")
    p.add_argument("--scenario", choices=["all", "frozen", "warmup", "random"],
                   default="all", help="要运行的场景(默认全部)")
    p.add_argument("--skip-index", action="store_true",
                   help="跳过建索引(要求 subgraph/base 下已有 base gexf 且 Milvus 已有 collection)")
    p.add_argument("--limit", type=int, default=0,
                   help="快速冒烟:同时覆盖 eval-n 与 warmup-n 的小值")
    p.add_argument("--use-bert", action="store_true", help="额外计算 BERTScore(慢)")
    p.add_argument("--output", default=None, help="输出 JSON 路径(默认 output/diagnose_leakage_{dataset}.json)")
    return p.parse_args()


def build_app(dataset: str):
    """从 config.yaml 构造 CacheGraphRAG 实例。"""
    cfg = get_config()
    if dataset:
        cfg["data"]["dataset"] = dataset
    return CacheGraphRAG.from_config(cfg)


def base_gexf_path(app: CacheGraphRAG, start: int, end: int) -> str:
    """与 index()/__init__ 一致的 base gexf 命名。"""
    cfg = get_config()
    cc = cfg.get("retrieval", {}).get("chunk_collection", app.nebula_space)
    return f"subgraph/base/{app.dataset}_{cc}_{app.nebula_space}_{start}_{end}_base.gexf"


def reset_to_base(app: CacheGraphRAG, gexf: str):
    """场景隔离:清空 L2 + 清空计数器 + 重载干净的 base L1 图。

    注意:load_graph_gexf 内部对计数器用 update() 合并(而非覆盖),
    所以必须先清空 chunk_access_counter / entity_access_counter /
    chunk_lru / promoted_chunks,再重载,避免场景间状态污染。
    """
    if app.mem_graph.persistent_graph:
        try:
            app.mem_graph.persistent_graph.clear()
        except Exception as e:
            print(f"  [warn] L2 clear: {e}")
    app.mem_graph.chunk_access_counter.clear()
    app.mem_graph.entity_access_counter.clear()
    app.mem_graph.chunk_lru.clear()
    app.mem_graph.promoted_chunks.clear()
    app.mem_graph.l2_written_nodes.clear()
    app.mem_graph.l2_written_edges.clear()
    app.mem_graph.l2_written_node_ops = 0
    app.mem_graph.l2_written_edge_ops = 0
    if gexf and os.path.exists(gexf):
        app.mem_graph.load_graph_gexf(gexf)
        print(f"  [reset] 已重载 base L1 图: {gexf}")


def freeze_promotion(app: CacheGraphRAG):
    """冻结 promotion:threshold 设为极大,access_chunk 永不触发写 L2。"""
    app.mem_graph.threshold = FROZEN_THRESHOLD


def unfreeze_promotion(app: CacheGraphRAG, threshold: int = 3):
    """恢复 promotion(诊断用,数值应与 config indexing.promotion_threshold 一致)。"""
    app.mem_graph.threshold = threshold


def build_retriever(app: CacheGraphRAG, ccol: str, mode: str = "hybrid"):
    """按 query() 内部逻辑构造 HybridRetriever + RRFFusion(用于 warm-up 检索)。"""
    from database.milvus import MilvusDB
    from src.retrieval.fusion import RRFFusion
    from src.retrieval.retriever import HybridRetriever

    cfg = get_config()
    eindex = "entity_index_" + app.nebula_space
    common_args = dict(
        vector_store=MilvusDB(db_name=ccol, overwrite=False, embed_model=app.llm.embed_model),
        entity_index_name=eindex,
        memory_graph=app.mem_graph,
        llm=app.llm,
        chunk_registry=app.pipeline.chunk_registry if app.pipeline else {},
        reranker=app.reranker,
    )
    retriever = HybridRetriever(**common_args, entity_extraction="llm", mode=mode)
    retriever.fusion = RRFFusion(retriever, rrf_k=cfg.get("fusion", {}).get("rrf_k", 60))
    return retriever


async def warmup_promote(app: CacheGraphRAG, questions, top_chunks=15, answer_topk=6):
    """场景 B:用独立 warm-up 查询驱动 promotion(只检索,不生成答案,省 LLM)。

    与 query() 内 562-566 行一致:对检索命中的每个 chunk 调 access_chunk,
    触发(达到 threshold)即 write_to_persistent_graph。
    """
    ccol = app.nebula_space
    retriever = build_retriever(app, ccol)
    promoted = 0
    total_hits = 0
    for q in questions:
        res = await asyncio.to_thread(
            retriever.hybrid_retrieve,
            q, topk=10, top_entities=5, top_chunks=top_chunks,
            top_rerank=15, answer_topk=answer_topk, track_promotion=False,
        )
        for cid in res.get("chunks", []):
            total_hits += 1
            triggered, data = app.mem_graph.access_chunk(cid)
            if triggered:
                app.mem_graph.write_to_persistent_graph([data])
                promoted += 1
    print(f"  [warmup] {len(questions)} 题 → {total_hits} 次 chunk 命中 → {promoted} 个 chunk 触发 promotion")
    return promoted


def random_promote(app: CacheGraphRAG, n_chunks: int, seed: int, threshold: int = 3):
    """场景 C:随机选 chunk 并反复 access 到触发 promotion(随机下界)。"""
    rng = random.Random(seed)
    chunks = []
    if app.pipeline and app.pipeline.chunk_registry:
        chunks = list(app.pipeline.chunk_registry.keys())
    if not chunks:
        chunks = list(app.mem_graph.chunk_lru.keys())
    if not chunks:
        raise RuntimeError("没有可用的 chunk 列表:请先建索引(--skip-index 时需要已有数据)")
    chosen = rng.sample(chunks, min(n_chunks, len(chunks)))
    promoted = 0
    for cid in chosen:
        for _ in range(threshold):
            triggered, data = app.mem_graph.access_chunk(cid)
            if triggered:
                app.mem_graph.write_to_persistent_graph([data])
                promoted += 1
                break
    print(f"  [random] 选 {len(chosen)} 个随机 chunk → {promoted} 个触发 promotion")
    return promoted


def get_l2_size(app: CacheGraphRAG) -> dict:
    """L2 规模:优先读 Nebula 中实际节点/边,失败则回退到写入计数器。"""
    result = {
        "nodes": len(app.mem_graph.l2_written_nodes),
        "edges": len(app.mem_graph.l2_written_edges),
        "promoted_chunks": len(app.mem_graph.promoted_chunks),
    }
    if app.mem_graph.persistent_graph:
        try:
            triplets = app.mem_graph.persistent_graph.get_triplets()
            result["nebula_nodes"] = len(triplets.get("source", [])) if isinstance(triplets, dict) else 0
            result["nebula_edges"] = len(triplets) if isinstance(triplets, (list, dict)) else 0
        except Exception as e:
            print(f"  [warn] Nebula L2 统计失败: {e}")
    return result


async def run_eval(app: CacheGraphRAG, questions, answers, start, end,
                   use_bert=False, tag="eval"):
    """冻结状态下评测 held-out 集,返回 (metrics, results)。

    注意:query() 内部 gt 按位置对齐 app.answers[start+idx],而本函数传入的
    questions 是随机采样的 held-out 子集(下标不连续),因此**必须**用传入的
    answers 按 query 重建 ground truth,不能直接用 results 里的 gt 字段。
    """
    freeze_promotion(app)  # 评测期间禁止写 L2(场景 A 全程冻结,场景 B/C 冻结后评测)
    results = await app.query(
        questions=questions, start=start, end=end,
        use_agentic=False, topk=10, top_entities=5, top_chunks=15,
        top_rerank=15, answer_topk=6, answer_aware_promotion=False,
        qa_concurrency=5, entity_extraction="llm",
    )
    # 用传入的 (questions, answers) 建立映射,避免 query() 的位置对齐 gt 错位
    qa_map = dict(zip(questions, answers))
    preds, gts = [], []
    for r in results:
        preds.append(r.get("predict", ""))
        gts.append(qa_map.get(r.get("query", ""), ""))
    metrics = evaluate_qa(preds, gts, use_rougel=True, use_bert=use_bert)
    print(f"  [eval:{tag}] n={len(results)} ACC={metrics['em']:.4f} RougeL={metrics.get('rougel', 0):.4f}")
    return metrics, results


async def run_index(app: CacheGraphRAG, start: int, end: int, gexf: str):
    """建索引(若 base gexf 不存在)。"""
    if os.path.exists(gexf):
        print(f"[index] 复用已有 base gexf: {gexf}")
        app.load_dataset(start, end)
        return
    print("[index] 开始建索引...")
    await app.index(start, end)
    print(f"[index] 完成,已保存 base gexf: {gexf}")
    app.load_dataset(start, end)


async def main():
    args = parse_args()

    # 快速冒烟模式
    if args.limit > 0:
        args.eval_n = args.limit
        args.warmup_n = args.limit
        args.random_chunks = min(args.random_chunks, args.limit)

    dataset = args.dataset or get_config().get("data", {}).get("dataset", "rgb_en_refine")
    start, end = args.start, args.end

    app = build_app(dataset)
    gexf = base_gexf_path(app, start, end)
    if not args.skip_index:
        await run_index(app, start, end, gexf)

    # 若跳过了索引但图未加载,手动 load_dataset
    if not app.questions:
        app.load_dataset(start, end)
    qs, ans = app.questions, app.answers
    if end == -1 or end > len(qs):
        end = len(qs)
    print(f"[data] {dataset} questions={len(qs)} answers={len(ans)} (slice {start}:{end})")

    # ── 用固定 seed 切分 warm-up / held-out(严格不相交) ──
    rng = random.Random(args.seed)
    idx = list(range(len(qs)))
    rng.shuffle(idx)
    warm_idx = idx[:args.warmup_n]
    eval_idx = idx[args.warmup_n:args.warmup_n + args.eval_n]
    warm_q = [qs[i] for i in warm_idx]
    eval_q = [qs[i] for i in eval_idx]
    eval_a = [ans[i] for i in eval_idx]
    print(f"[split] warm-up n={len(warm_q)} | held-out eval n={len(eval_q)} (seed={args.seed})")

    paper_acc = None
    for key, val in PAPER_TABLE3.items():
        if key in dataset.lower():
            paper_acc = val
            break
    paper_note = f"{paper_acc}%" if paper_acc else "(未匹配论文数据集)"

    report = {
        "dataset": dataset,
        "slice": [start, end],
        "seed": args.seed,
        "warmup_n": len(warm_q),
        "eval_n": len(eval_q),
        "paper_table3_acc": paper_acc,
        "scenarios": {},
        "verdicts": {},
    }

    run_all = args.scenario == "all"
    out_path = args.output or f"output/diagnose_leakage_{dataset}.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # ── 场景 A:冻结图(无 promotion)基线 ──
    if run_all or args.scenario == "frozen":
        print("\n======== 场景 A:冻结图(无 promotion)基线 ========")
        reset_to_base(app, gexf)
        freeze_promotion(app)
        metrics, _ = await run_eval(app, eval_q, eval_a, start, end,
                                    args.use_bert, tag="A")
        l2 = get_l2_size(app)
        report["scenarios"]["A_frozen"] = {
            "acc": metrics["em"], "rougel": metrics.get("rougel"),
            "bertscore": metrics.get("bertscore"), "count": metrics["count"],
            "l2": l2,
        }
        print(f"  [A] ACC={metrics['em']:.4f}  (论文参照 {paper_note})")

    # ── 场景 B:独立 warm-up 驱动 promotion 后冻结 ──
    if run_all or args.scenario == "warmup":
        print("\n======== 场景 B:独立 warm-up → 冻结 → 评测 ========")
        reset_to_base(app, gexf)
        unfreeze_promotion(app, threshold=3)
        prom = await warmup_promote(app, warm_q)
        l2_before = get_l2_size(app)
        metrics, _ = await run_eval(app, eval_q, eval_a, start, end,
                                    args.use_bert, tag="B")
        l2 = get_l2_size(app)
        report["scenarios"]["B_warmup"] = {
            "acc": metrics["em"], "rougel": metrics.get("rougel"),
            "bertscore": metrics.get("bertscore"), "count": metrics["count"],
            "promoted_chunks": prom,
            "l2_before_eval": l2_before, "l2_after_eval": l2,
        }
        print(f"  [B] ACC={metrics['em']:.4f}  (promoted={prom})")

    # ── 场景 C:随机触发 promotion 后冻结 ──
    if run_all or args.scenario == "random":
        print("\n======== 场景 C:随机 promotion → 冻结 → 评测 ========")
        reset_to_base(app, gexf)
        unfreeze_promotion(app, threshold=3)
        prom = random_promote(app, args.random_chunks, args.seed + 1)
        metrics, _ = await run_eval(app, eval_q, eval_a, start, end,
                                    args.use_bert, tag="C")
        l2 = get_l2_size(app)
        report["scenarios"]["C_random"] = {
            "acc": metrics["em"], "rougel": metrics.get("rougel"),
            "bertscore": metrics.get("bertscore"), "count": metrics["count"],
            "promoted_chunks": prom, "l2": l2,
        }
        print(f"  [C] ACC={metrics['em']:.4f}  (random promoted={prom})")

    # ── 判定 ──
    s = report["scenarios"]
    if "A_frozen" in s and paper_acc is not None:
        report["verdicts"]["leakage_vs_paper"] = round(paper_acc / 100 - s["A_frozen"]["acc"], 4)
    if "B_warmup" in s and "A_frozen" in s:
        report["verdicts"]["warmup_gain"] = round(s["B_warmup"]["acc"] - s["A_frozen"]["acc"], 4)
    if "B_warmup" in s and "C_random" in s:
        report["verdicts"]["query_intelligence"] = round(s["B_warmup"]["acc"] - s["C_random"]["acc"], 4)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {out_path}")

    # ── 控制台对比表 ──
    print("\n================ 对比表 ================")
    print(f"{'场景':<34} | {'ACC':>8} | {'L2节点':>8} | {'L2边':>8}")
    for key, label in [("A_frozen", "A 冻结图(无晋升)"),
                       ("B_warmup", "B 独立warm-up+冻结"),
                       ("C_random", "C 随机晋升+冻结")]:
        if key in s:
            sc = s[key]
            l2n = sc.get("l2", {}).get("nebula_nodes", sc.get("l2", {}).get("nodes", 0))
            l2e = sc.get("l2", {}).get("nebula_edges", sc.get("l2", {}).get("edges", 0))
            print(f"{label:<34} | {sc['acc']*100:>6.2f}% | {l2n:>8} | {l2e:>8}")
    if paper_acc is not None:
        print(f"{'论文 Table III(全量参照)':<34} | {paper_acc:>7.2f}% |")
    for k, v in report["verdicts"].items():
        print(f"  {k}: {v:+.4f}")


if __name__ == "__main__":
    asyncio.run(main())
