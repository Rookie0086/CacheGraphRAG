# 验证 P0-1:图检索器 Decay-Guided Beam Search(论文 IV-E-1)
# 每跳扩展后按 节点-查询余弦 × γ^hop 打分,保留 top-B 候选路径;束宽可配置。
# 纯内存 mock(不连接 NebulaGraph/Milvus),复用 verify_gamma 的链式图确保无回归。
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import networkx as nx
import src.retrieval.retriever as R


class NameVectorEmbed:
    """按节点名返回不同向量:best 与查询同向(cos=1.0),mid 半向(cos=0.5),worst 正交(cos=0)"""

    VECTORS = {
        "best": [1.0, 0.0, 0.0],
        "mid": [0.5, 0.8660254, 0.0],
        "worst": [0.0, 1.0, 0.0],
    }

    def _vec(self, text):
        for name in self.VECTORS:
            if f"Entity: {name}." in text:
                return self.VECTORS[name]
        return [0.0, 0.0, 1.0]

    def get_embedding(self, text):
        return self._vec(text)

    def get_embeddings(self, texts):
        return [self._vec(t) for t in texts]


class FlatEmbed:
    """固定向量:所有候选余弦相同(用于链式图无回归验证)"""

    def get_embedding(self, text):
        return [0.1] * 64

    def get_embeddings(self, texts):
        return [[0.1] * 64 for _ in texts]


class FakeStore:
    def __init__(self, embed):
        self.embed_model = embed

    def search(self, vector, sp, limit, output_fields=None):
        class Hit:
            pass
        h = Hit()
        h.entity = {"uid": "1", "name": "A", "type": "person", "desc": ""}
        h.distance = 1.0
        return [[h]]


class BeamGraph:
    """含 best/mid/worst 三个候选节点的图,用于束剪枝顺序验证"""

    def __init__(self):
        self.persistent_graph = None
        g = nx.DiGraph()
        g.add_node("best", name="best", type="t", source_chunks="chunk_best")
        g.add_node("mid", name="mid", type="t", source_chunks="chunk_mid")
        g.add_node("worst", name="worst", type="t", source_chunks="chunk_worst")
        self.graph = g

    # M4:检索 BFS/束剪枝改走快照读,测试夹具补齐
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
        if keys:
            return [(u, v, i, d) for i, (u, v, d) in enumerate(edges)]
        return list(edges)

    def snapshot_all_edges(self):
        return list(self.graph.edges(data=True, keys=True))


class ChainGraph:
    """L1 链式图:A --r1--> B --r2--> C(与 verify_gamma 相同,验证束搜索无回归)"""

    def __init__(self):
        self.persistent_graph = None
        g = nx.DiGraph()
        g.add_node("A", name="A", type="person", source_chunks="chunk_A")
        g.add_node("B", name="B", type="person", source_chunks="chunk_B")
        g.add_node("C", name="C", type="person", source_chunks="chunk_C")
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
        if keys:
            return [(u, v, i, d) for i, (u, v, d) in enumerate(edges)]
        return list(edges)

    def snapshot_all_edges(self):
        return list(self.graph.edges(data=True, keys=True))


def make_retriever(embed, memory_graph, gamma=0.5, max_hops=3, beam_width=4):
    r = R.BaseRetriever(vector_store=FakeStore(embed), memory_graph=memory_graph)
    r._entity_store = FakeStore(embed)
    r.gamma = gamma
    r.max_hops = max_hops
    r.beam_width = beam_width
    r._search_milvus = R.HybridRetriever._search_milvus.__get__(r)
    r._beam_prune_frontier = R.HybridRetriever._beam_prune_frontier.__get__(r)
    r.retrieve_from_memory_graph = R.HybridRetriever.retrieve_from_memory_graph.__get__(r)
    return r


ok = True
q = np.array([1.0, 0.0, 0.0])

# ── 场景 1:束剪枝按 余弦×γ^hop 保留 top-B ──
r = make_retriever(NameVectorEmbed(), BeamGraph(), gamma=0.5, beam_width=2)
out = r._beam_prune_frontier(["worst", "best", "mid"], 2, 1, q)
# hop=1 → decay=0.5:best=0.5, mid=0.25, worst=0 → 恰好保留 top-2:best,mid
s1 = (out == ["best", "mid"])
ok &= s1
print(f"  [束剪枝 top-B] out={out} (期望恰好保留 ['best','mid']) {'PASS' if s1 else 'FAIL'}")

# ── 场景 2:深度衰减注入打分(hop=2,γ=0.5 → decay=0.25,排序不变但分数按公式)──
r2 = make_retriever(NameVectorEmbed(), BeamGraph(), gamma=0.5, beam_width=2)
out2 = r2._beam_prune_frontier(["worst", "best", "mid"], 2, 2, q)
s2 = (out2[:2] == ["best", "mid"])
ok &= s2
print(f"  [hop 衰减] out={out2} (期望 best,mid) {'PASS' if s2 else 'FAIL'}")

# ── 场景 3:束宽=1 → 仅保留最高分候选 ──
r3 = make_retriever(NameVectorEmbed(), BeamGraph(), gamma=0.5, beam_width=1)
out3 = r3._beam_prune_frontier(["worst", "best", "mid"], 1, 1, q)
s3 = (out3[0] == "best")
ok &= s3
print(f"  [束宽=1] out={out3} (期望 best 唯一入选) {'PASS' if s3 else 'FAIL'}")

# ── 场景 4:候选数 ≤ 束宽 → 不裁剪(全量保留)──
r4 = make_retriever(NameVectorEmbed(), BeamGraph(), gamma=0.5, beam_width=5)
out4 = r4._beam_prune_frontier(["worst", "best", "mid"], 5, 1, q)
s4 = (set(out4) == {"worst", "best", "mid"})
ok &= s4
print(f"  [候选≤束宽] out={out4} (期望全量保留) {'PASS' if s4 else 'FAIL'}")

# ── 场景 5:链式图 + 束宽 4 —— 与 verify_gamma 权重完全一致(无回归)──
r5 = make_retriever(FlatEmbed(), ChainGraph(), gamma=0.5, max_hops=3, beam_width=4)
res = r5.retrieve_from_memory_graph(
    query_emb=np.array([0.1] * 64), entity_name="A", entity_type="person",
    entity_desc="", top_entities=5, top_chunks=5, entity_weight=1.0)
cs = res["chunk_scores"]
exp5 = {"chunk_A": 1.0, "chunk_AB": 1.5, "chunk_B": 0.5, "chunk_BC": 0.75, "chunk_C": 0.25}
s5 = all(abs(cs.get(cid, -1) - w) < 1e-6 for cid, w in exp5.items())
ok &= s5
print(f"  [链式图无回归] chunk_scores={cs} (期望 {exp5}) {'PASS' if s5 else 'FAIL'}")

# ── 场景 6:束宽从配置读取(beam_width / hyperparameters.B 兜底)──
R.get_config = lambda: {"retrieval": {"gamma": 0.5, "max_hops": 3, "beam_width": 2},
                        "hyperparameters": {"B": 4}}
r6 = R.BaseRetriever(vector_store=FakeStore(FlatEmbed()), memory_graph=ChainGraph())
s6a = (r6.beam_width == 2)
R.get_config = lambda: {"retrieval": {}, "hyperparameters": {"B": 3}}
r6b = R.BaseRetriever(vector_store=FakeStore(FlatEmbed()), memory_graph=ChainGraph())
s6b = (r6b.beam_width == 3)
R.get_config = lambda: {"retrieval": {}}
r6c = R.BaseRetriever(vector_store=FakeStore(FlatEmbed()), memory_graph=ChainGraph())
s6c = (r6c.beam_width == 4)  # 默认 4
s6 = s6a and s6b and s6c
ok &= s6
print(f"  [束宽配置] beam_width=2({r6.beam_width}) / hyper B=3({r6b.beam_width}) / 默认4({r6c.beam_width}) "
      f"{'PASS' if s6 else 'FAIL'}")

print("\n=== 图检索束搜索验证:", "全部通过 ✅" if ok else "存在失败 ❌")
sys.exit(0 if ok else 1)
