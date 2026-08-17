# 验证中档#8:跳数衰减 γ 参数化
# 通过纯内存 mock(不连接 NebulaGraph/Milvus),对比 γ=0.5 与 γ=0.3 两档的 hop 权重分布,
# 并验证 max_hops 控制遍历深度。
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import networkx as nx
import src.retrieval.retriever as R


class FakeEmbed:
    """固定向量 embed,使相似度计算稳定(所有候选相似度相同)"""

    def get_embedding(self, text):
        return [0.1] * 64

    def get_embeddings(self, texts):
        return [[0.1] * 64 for _ in texts]


class FakeStore:
    embed_model = FakeEmbed()

    def search(self, vector, sp, limit, output_fields=None):
        class Hit:
            pass
        h = Hit()
        h.entity = {"uid": "1", "name": "A", "type": "person", "desc": ""}
        h.distance = 1.0
        return [[h]]


class FakeGraph:
    """L1 链式图:A --r1--> B --r2--> C,验证 hop0/1/2 权重衰减"""

    def __init__(self):
        self.persistent_graph = None  # L2 未接入,纯 L1 验证
        g = nx.DiGraph()
        g.add_node("A", source_chunks="chunk_A")
        g.add_node("B", source_chunks="chunk_B")
        g.add_node("C", source_chunks="chunk_C")
        g.add_edge("A", "B", relationship="r1", source_chunk="chunk_AB")
        g.add_edge("B", "C", relationship="r2", source_chunk="chunk_BC")
        self.graph = g

    def _rebuild_id_index(self):
        pass

    def get_nodes_by_id(self, uid):
        return ["A"]

    # M4:检索 BFS 改走快照读,测试夹具补齐
    def snapshot_node_data(self, uid):
        d = self.graph.nodes.get(uid)
        if d is None:
            return None
        sc = d.get("source_chunks")
        return {"_uid": uid, "name": d.get("name", ""), "type": d.get("type", ""),
                "desc": d.get("desc", ""), "source_chunk": d.get("source_chunk", ""),
                "source_chunks": set(sc) if isinstance(sc, (set, list, tuple)) else (sc or set()),
                "ref_count": d.get("ref_count", 0), "last_access_time": d.get("last_access_time", 0)}

    def snapshot_edges(self, uid, direction="out", keys=False):
        if not self.graph.has_node(uid):
            return []
        edges = (self.graph.in_edges(uid, data=True) if direction == "in"
                 else self.graph.out_edges(uid, data=True))
        if keys:  # DiGraph 无多边 key,用序号占位
            return [(u, v, i, d) for i, (u, v, d) in enumerate(edges)]
        return list(edges)

    def snapshot_all_edges(self):
        return list(self.graph.edges(data=True, keys=True))

    def is_chunk_in_l1(self, c):
        return True

    def is_chunk_in_l2(self, c):
        return False

    def touch_chunk(self, c):
        pass

    def rehydrate_chunk_from_milvus(self, c):
        pass


def run(gamma, max_hops):
    # 注入配置:monkeypatch 模块内的 get_config
    R.get_config = lambda: {"retrieval": {"gamma": gamma, "max_hops": max_hops}}
    # 用 BaseRetriever 避免 HybridRetriever.__init__ 连接真实 Milvus,
    # 手动绑定 retrieve_from_memory_graph(其只依赖 memory_graph/_entity_store/gamma/max_hops)
    r = R.BaseRetriever(vector_store=FakeStore(), memory_graph=FakeGraph())
    r._entity_store = FakeStore()
    r._search_milvus = R.HybridRetriever._search_milvus.__get__(r)
    r.retrieve_from_memory_graph = R.HybridRetriever.retrieve_from_memory_graph.__get__(r)
    # 束搜索接入后 retrieve_from_memory_graph 内部调用 _beam_prune_frontier,需一并绑定
    r._beam_prune_frontier = R.HybridRetriever._beam_prune_frontier.__get__(r)
    out = r.retrieve_from_memory_graph(
        query_emb=np.array([0.1] * 64), entity_name="A", entity_type="person",
        entity_desc="", top_entities=5, top_chunks=5, entity_weight=1.0)
    return out["chunk_scores"], r


def close(a, b, eps=1e-6):
    return abs(a - b) < eps


ok = True

# 1) γ=0.5,max_hops=3:节点权重 hop0=1.0,hop1=0.5,hop2=0.25
#    边双向计分(起点出边 + 终点入边):chunk_AB=hop0(1.0)+hop1(0.5)=1.5, chunk_BC=hop1(0.5)+hop2(0.25)=0.75
s5, r5 = run(0.5, 3)
exp5 = {"chunk_A": 1.0, "chunk_AB": 1.5, "chunk_B": 0.5, "chunk_BC": 0.75, "chunk_C": 0.25}
for cid, w in exp5.items():
    got = s5.get(cid)
    passed = got is not None and close(got, w)
    ok &= passed
    print(f"  [γ=0.5] {cid}: got={got} expect={w} {'PASS' if passed else 'FAIL'}")
print(f"  [γ=0.5] retriever.gamma={r5.gamma} max_hops={r5.max_hops}")

# 2) γ=0.3,max_hops=3:节点 hop1=0.3,hop2=0.09;边 chunk_BC=0.3+0.09=0.39
s3, r3 = run(0.3, 3)
exp3 = {"chunk_B": 0.3, "chunk_BC": 0.39, "chunk_C": 0.09}
for cid, w in exp3.items():
    got = s3.get(cid)
    passed = got is not None and close(got, w)
    ok &= passed
    print(f"  [γ=0.3] {cid}: got={got} expect={w} {'PASS' if passed else 'FAIL'}")
print(f"  [γ=0.3] retriever.gamma={r3.gamma} max_hops={r3.max_hops}")

# 3) 两档差异显著:hop2 权重 0.25 vs 0.09
diff = s5["chunk_C"] - s3["chunk_C"]
passed = diff > 0.1
ok &= passed
print(f"  [跨档差异] chunk_C γ=0.5({s5['chunk_C']}) vs γ=0.3({s3['chunk_C']}) 差={diff:.4f} {'PASS' if passed else 'FAIL'}")

# 4) max_hops=2:hop2 节点 C 不被访问,chunk_C 应为 0
s2, r2 = run(0.5, 2)
passed = not s2.get("chunk_C") or close(s2["chunk_C"], 0.0)
ok &= passed
print(f"  [max_hops=2] chunk_C: got={s2.get('chunk_C')} expect=0 {'PASS' if passed else 'FAIL'}")
print(f"  [max_hops=2] retriever.max_hops={r2.max_hops}")

print("\n=== 中档#8 γ 参数化验证:", "全部通过 ✅" if ok else "存在失败 ❌")
sys.exit(0 if ok else 1)
