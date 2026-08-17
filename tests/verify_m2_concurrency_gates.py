# 验证 M2(2026-08-15):查询期嵌入/重排并发门控 + 实体级图检索并行
#   - _embed_text 与束剪枝批量嵌入受 embedding_concurrency 门控
#   - APIReranker.rerank 受 rerank_concurrency 门控
#   - _graph_retrieve 多实体并发执行(L1/L2 各实体并行,降查询时延)
# 纯内存 mock,不连接任何 API。
import sys
import os
import time
import threading
from types import SimpleNamespace
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest import mock
import src.retrieval.retriever as R
import src.retrieval.reranker as RK

ok = True
results = []


def report(name, cond):
    global ok
    ok &= cond
    results.append(cond)
    print(f"  [{name}] {'PASS' if cond else 'FAIL'}")


# 注入最小配置:embedding_concurrency=2 / rerank_concurrency=2 / graph_entity_parallel=2
CFG = {"retrieval": {"embedding_concurrency": 2, "rerank_concurrency": 2,
                     "graph_entity_parallel": 2, "gamma": 0.5, "max_hops": 3}}
R.get_config = lambda: CFG


class ConcurrencyTracker:
    """记录最大并发度的嵌入后端。"""
    def __init__(self, delay=0.2):
        self.delay = delay
        self.active = 0
        self.max_active = 0
        self.lock = threading.Lock()

    def _enter(self):
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)

    def _leave(self):
        with self.lock:
            self.active -= 1

    def get_embedding(self, text):
        self._enter()
        time.sleep(self.delay)
        self._leave()
        return [1.0, 0.0]

    def get_embeddings(self, texts):
        self._enter()
        time.sleep(self.delay)
        self._leave()
        return [[1.0, 0.0] for _ in texts]


class FakeLLM:
    def __init__(self, embed):
        self.embed_model = embed


# ── 场景 1:嵌入门控 —— 4 线程并发 _embed_text,最大并发 ≤ 2 ──
tracker = ConcurrencyTracker()
r = R.BaseRetriever(vector_store=None, memory_graph=None, llm=FakeLLM(tracker))
threads = [threading.Thread(target=r._embed_text, args=(f"t{i}",)) for i in range(4)]
for t in threads:
    t.start()
for t in threads:
    t.join()
s1 = tracker.max_active <= 2
report("嵌入门控 max_active ≤ 2", s1)
print(f"        (实测 max_active={tracker.max_active})")

# ── 场景 2:rerank 门控 —— 4 线程并发 rerank,最大并发 ≤ 2 ──
rerank_tracker = ConcurrencyTracker()


def fake_post(url, headers=None, json=None, timeout=None):
    rerank_tracker._enter()
    time.sleep(0.2)
    rerank_tracker._leave()
    resp = SimpleNamespace()
    resp.raise_for_status = lambda: None
    resp.json = lambda: {"results": [{"index": 0, "score": 0.9, "text": "x"}]}
    return resp


with mock.patch.object(RK.requests, "post", fake_post):
    rr = RK.APIReranker("bge-reranker", "key", "http://localhost:1")
    rthreads = [threading.Thread(target=rr.rerank, args=("q", ["doc"])) for _ in range(4)]
    for t in rthreads:
        t.start()
    for t in rthreads:
        t.join()
s2 = rerank_tracker.max_active <= 2
report("rerank 门控 max_active ≤ 2", s2)
print(f"        (实测 max_active={rerank_tracker.max_active})")

# ── 场景 3:实体级图检索并行 —— 2 实体在 2 个线程并发执行(墙钟 < 串行和)──
r2 = R.BaseRetriever(vector_store=None, memory_graph=None, llm=FakeLLM(tracker))
r2.graph_entity_parallel = 2
r2.enable_l1 = r2.enable_l2 = True
thread_ids = []


def fake_memory(self, query_emb, ent, top_entities, top_chunks, relations, ew):
    thread_ids.append(threading.get_ident())
    time.sleep(0.25)
    # 每实体返回独立 chunk,便于验证 chunk_entity_coverage 归并
    return {"chunks": [], "chunk_scores": {f"chunk_{ent['id']}": 1.0},
            "node_scores": {}, "node_chunks": {}, "matched_entities": []}


r2._retrieve_memory = fake_memory.__get__(r2, R.BaseRetriever)
r2._retrieve_persistent = fake_memory.__get__(r2, R.BaseRetriever)
entities = [{"id": "e1"}, {"id": "e2"}]
t0 = time.time()
m_res, p_res, cov = r2._graph_retrieve(None, entities, [], {"e1": 1.0, "e2": 1.0}, 5, 5, True)
wall = time.time() - t0
s3 = (len(thread_ids) == 4 and len(set(thread_ids)) >= 2 and wall < 0.75
      and len(cov) == 2)  # 串行需 4×0.25=1.0s
report("实体级图检索并行(墙钟<串行)", s3)
print(f"        (线程数={len(set(thread_ids))}, 墙钟={wall:.2f}s, 串行预计≈1.0s)")

# ── 场景 4:graph_entity_parallel=1 时退化为串行(结果一致)──
r3 = R.BaseRetriever(vector_store=None, memory_graph=None, llm=FakeLLM(tracker))
r3.graph_entity_parallel = 1
r3.enable_l1 = r3.enable_l2 = True
r3._retrieve_memory = fake_memory.__get__(r3, R.BaseRetriever)
r3._retrieve_persistent = fake_memory.__get__(r3, R.BaseRetriever)
t0 = time.time()
m_res, p_res, cov = r3._graph_retrieve(None, entities, [], {"e1": 1.0, "e2": 1.0}, 5, 5, True)
wall3 = time.time() - t0
s4 = len(m_res) == 2 and len(p_res) == 2 and len(cov) == 2 and wall3 >= 0.9
report("parallel=1 退化为串行且结果一致", s4)
print(f"        (墙钟={wall3:.2f}s)")

print("\n=== M2 并发门控/实体并行验证:", "全部通过 ✅" if ok else "存在失败 ❌")
sys.exit(0 if ok else 1)
