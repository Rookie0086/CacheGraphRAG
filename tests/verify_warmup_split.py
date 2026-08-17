# 验证中档#11:预热/测试隔离协议(_run_split_qa 的 seed 切分 + 两阶段流程)
# 纯内存 mock:绑定 _run_split_qa 到 fake app,monkeypatch evaluate_qa,不连接外部服务。
import sys
import os
import json
import asyncio
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.CacheGraphRAG import CacheGraphRAG
import src.eval as EVAL

ok = True


class FakeMem:
    def __init__(self):
        self.l2_written_nodes = set()
        self.l2_written_edges = set()
        self.promoted_chunks = set()


class FakeApp:
    dataset = "test"
    questions = [f"Q{i:02d}" for i in range(20)]
    answers = [f"A{i:02d}" for i in range(20)]
    mem_graph = FakeMem()

    def __init__(self):
        self.calls = []  # 记录 query 调用

    async def query(self, questions=None, **kw):
        self.calls.append({"questions": list(questions), "kwargs": kw})
        return [{"query": q, "predict": f"A{self.questions.index(q)}"} for q in questions]


# monkeypatch evaluate_qa:固定返回(避免依赖 ROUGE/BERTScore)
def _fake_eval(preds, gts, use_rougel=False, use_bert=False):
    return {"em": 0.7, "rougel": 0.5, "bertscore": 0.0, "count": len(preds)}


EVAL.evaluate_qa = _fake_eval


async def run_case(ratio, seed, label):
    global ok
    app = FakeApp()
    m = CacheGraphRAG._run_split_qa.__get__(app, CacheGraphRAG)
    await m(
        start=0, end=len(app.questions), use_agentic=False, agentic_steps=3,
        answer_aware_promotion=False, entity_promotion_threshold=3,
        promotion_retries=2, promotion_expansion=10, qa_concurrency=5,
        entity_extraction="llm", qa_cache=False, entity_index_name=None,
        chunk_collection=None, answer_topk=6, top_chunks=None, mode="hybrid",
        tag="baseline", warmup_ratio=ratio, warmup_seed=seed)

    # 断言 1:两次 query 调用(warm + eval)
    n_calls = len(app.calls)
    p1 = n_calls == 2
    ok &= p1
    print(f"  [{label}] query 调用次数={n_calls} (期望 2) {'PASS' if p1 else 'FAIL'}")

    # 断言 2:预热与测试集合严格不相交,并集=全量
    warm_set = set(app.calls[0]["questions"])
    eval_set = set(app.calls[1]["questions"])
    disj = warm_set.isdisjoint(eval_set)
    union = warm_set | eval_set
    full = set(app.questions)
    p2 = disj and union == full
    ok &= p2
    print(f"  [{label}] 预热∩测试=∅ {disj}, 并集=全量 {union == full} "
          f"(warm={len(warm_set)}, eval={len(eval_set)}, total={len(full)}) {'PASS' if p2 else 'FAIL'}")

    # 断言 3:预热比例 ≈ ratio
    n_warm = len(warm_set)
    exp = max(1, int(len(full) * ratio))
    p3 = n_warm == exp
    ok &= p3
    print(f"  [{label}] warmup_n={n_warm} (期望 {exp}) {'PASS' if p3 else 'FAIL'}")

    # 断言 4:报告 JSON 生成且含两组指标
    report_path = f"output/qa/warmup_report_test_baseline_{seed}_0_{len(full)}.json"
    p4 = os.path.exists(report_path)
    if p4:
        r = json.load(open(report_path))
        p4 = (r.get("disjoint") is True and "cold_start" in r and "warm_after" in r
              and r["warmup_seed"] == seed)
    ok &= p4
    print(f"  [{label}] 报告生成 disjoint=True + 两组指标: {p4} {'PASS' if p4 else 'FAIL'}")
    return app


app = asyncio.run(run_case(0.5, 42, "ratio=0.5, seed=42"))
asyncio.run(run_case(0.3, 7, "ratio=0.3, seed=7"))

# 断言 5:seed 不同切分不同(可复现性与隔离)
a = FakeApp()
b = FakeApp()
m_a = CacheGraphRAG._run_split_qa.__get__(a, CacheGraphRAG)
m_b = CacheGraphRAG._run_split_qa.__get__(b, CacheGraphRAG)
asyncio.run(m_a(start=0, end=20, use_agentic=False, agentic_steps=3, answer_aware_promotion=False,
                entity_promotion_threshold=3, promotion_retries=2, promotion_expansion=10,
                qa_concurrency=5, entity_extraction="llm", qa_cache=False, entity_index_name=None,
                chunk_collection=None, answer_topk=6, top_chunks=None, mode="hybrid",
                tag="baseline", warmup_ratio=0.5, warmup_seed=42))
asyncio.run(m_b(start=0, end=20, use_agentic=False, agentic_steps=3, answer_aware_promotion=False,
                entity_promotion_threshold=3, promotion_retries=2, promotion_expansion=10,
                qa_concurrency=5, entity_extraction="llm", qa_cache=False, entity_index_name=None,
                chunk_collection=None, answer_topk=6, top_chunks=None, mode="hybrid",
                tag="baseline", warmup_ratio=0.5, warmup_seed=43))
w_a = set(a.calls[0]["questions"])
w_b = set(b.calls[0]["questions"])
p5 = w_a != w_b
ok &= p5
print(f"  [seed 敏感性] seed=42 与 seed=43 的预热集不同: {p5} {'PASS' if p5 else 'FAIL'}")

# 清理测试产物
for f in os.listdir("output/qa"):
    if f.startswith("warmup_report_test_baseline_") or f.startswith("qa_warmup_test_") or f.startswith("qa_test_test_"):
        os.remove(os.path.join("output/qa", f))

print("\n=== 中档#11 预热/测试隔离验证:", "全部通过 ✅" if ok else "存在失败 ❌")
sys.exit(0 if ok else 1)
