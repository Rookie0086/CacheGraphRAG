# 验证中档#9:强化循环阻断(语义级查重 + UNKNOWN 连续计数)
# 纯内存 mock,不连接任何外部服务/API。
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import src.retrieval.agentic_engine as AE


class FakeEmbed:
    """按语义维度构造稀疏向量:同主题句子(换措辞)向量相同 → 余弦=1.0,不同主题正交 → 余弦≈0"""

    def get_embedding(self, text):
        t = text.lower()
        v = np.zeros(256)
        if "founder" in t or "founded" in t or "created" in t:
            v[1] += 1.0          # "创立者"语义维度
        if "apple" in t:
            v[2] += 1.0
        if "microsoft" in t:
            v[3] += 1.0
        if "ceo" in t:
            v[4] += 1.0
        n = np.linalg.norm(v)
        return (v / n).tolist() if n else v.tolist()


class FakeLLM:
    embed_model = FakeEmbed()


class FakeRetriever:
    def __init__(self):
        self.llm = FakeLLM()

    def hybrid_retrieve(self, q, topk=2, top_entities=3, top_chunks=3, top_rerank=6):
        return {"chunks": [f"c_{abs(hash(q)) % 100}"]}


def make_engine():
    AE.get_config = lambda: {"retrieval": {"semantic_dup_threshold": 0.8, "unknown_tolerance": 2}}
    return AE.IterativeAgenticEngine(
        llm=FakeLLM(), dataset="test", retriever=FakeRetriever(), max_steps=4)


ok = True

# ── 场景 1:语义查重拦截(LLM 换措辞重复同一问题)──
eng = make_engine()
calls = {"plan": 0}
REWARDS = ["Who was the founder of Apple?", "Which individual created Apple Inc.?"]


def fake_first(q):
    return {"need_multihop": True, "subquestion": "Who founded Apple?"}


def fake_next(query, history):
    if calls["plan"] < len(REWARDS):
        q = REWARDS[calls["plan"]]
        calls["plan"] += 1
        return {"is_final": False, "subquestion": q}
    return {"is_final": True, "final_answer": "done"}


def fake_answer(q, cids):
    return ("Apple was founded by Steve Jobs.", "p")


eng._plan_first_step = fake_first
eng._plan_next_step = fake_next
eng._answer_with_chunks = fake_answer
res = eng.run("Who founded Apple?")
sc1 = (res["block_count"] == 1 and len(res["agentic_steps"]) == 1)
ok &= sc1
print(f"  [语义查重] block_count={res['block_count']} steps={len(res['agentic_steps'])} "
      f"(期望 block=1, steps=1, 换措辞问题被拦截不执行) {'PASS' if sc1 else 'FAIL'}")
print(f"            被拦截问题(未执行): '{REWARDS[0]}'")

# ── 场景 2:UNKNOWN 连续计数阻断(连续 2 次 I don't know → 提前结束)──
eng = make_engine()
ci = [0]
TURNS = ["Q2", "Q3"]


def fake_first2(q):
    return {"need_multihop": True, "subquestion": "Q1"}


def fake_next2(query, history):
    if ci[0] < len(TURNS):
        q = TURNS[ci[0]]
        ci[0] += 1
        return {"is_final": False, "subquestion": q}
    return {"is_final": True, "final_answer": "done"}


def fake_answer2(q, cids):
    return ("I don't know.", "p")


eng._plan_first_step = fake_first2
eng._plan_next_step = fake_next2
eng._answer_with_chunks = fake_answer2
res2 = eng.run("Q?")
sc2 = (res2["block_count"] == 1 and len(res2["agentic_steps"]) == 2)
ok &= sc2
print(f"  [UNKNOWN 计数] block_count={res2['block_count']} steps={len(res2['agentic_steps'])} "
      f"(期望 block=1, steps=2, 连续 2 次后提前结束) {'PASS' if sc2 else 'FAIL'}")

# ── 场景 3:字符串查重仍生效(精确重复)──
eng = make_engine()


def fake_first3(q):
    return {"need_multihop": True, "subquestion": "SameQ?"}


def fake_next3(query, history):
    return {"is_final": False, "subquestion": "sameq"}  # 规范化后与 asked_set 完全重复


eng._plan_first_step = fake_first3
eng._plan_next_step = fake_next3
eng._answer_with_chunks = fake_answer
res3 = eng.run("SameQ?")
sc3 = res3["block_count"] == 1 and len(res3["agentic_steps"]) == 1
ok &= sc3
print(f"  [字符串查重] block_count={res3['block_count']} steps={len(res3['agentic_steps'])} "
      f"(期望 block=1, steps=1) {'PASS' if sc3 else 'FAIL'}")

# ── 场景 4:对照——真正不同方向的问题链不被误拦截 ──
eng = make_engine()
cj = [0]
TURNS4 = ["Who is CEO of Microsoft?", "What is the capital of France?"]


def fake_first4(q):
    return {"need_multihop": True, "subquestion": "Who founded Apple?"}


def fake_next4(query, history):
    if cj[0] < len(TURNS4):
        q = TURNS4[cj[0]]
        cj[0] += 1
        return {"is_final": False, "subquestion": q}
    return {"is_final": True, "final_answer": "done"}


eng._plan_first_step = fake_first4
eng._plan_next_step = fake_next4
eng._answer_with_chunks = fake_answer
res4 = eng.run("Who founded Apple?")
sc4 = (res4["block_count"] == 0 and len(res4["agentic_steps"]) >= 3)
ok &= sc4
print(f"  [对照:不同方向] block_count={res4['block_count']} steps={len(res4['agentic_steps'])} "
      f"(期望 block=0, 3 个问题全部执行) {'PASS' if sc4 else 'FAIL'}")

print("\n=== 中档#9 循环阻断验证:", "全部通过 ✅" if ok else "存在失败 ❌")
sys.exit(0 if ok else 1)
