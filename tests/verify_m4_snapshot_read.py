# 验证 M4(2026-08-15):图读快照 —— 锁内拷贝节点属性/边,消除检索 BFS、晋升子图提取
# 与写方(add_node/add_edge/pruner 驱逐)并发时的 NetworkX 迭代竞态。
# 纯内存构造 MemoryGraphManager 最小实例(object.__new__ 跳过 Nebula/Milvus 连接)。
import sys
import os
import time
import threading
import networkx as nx
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.graph.memory_graph import MemoryGraphManager

ok = True
results = []


def report(name, cond):
    global ok
    ok &= cond
    results.append(cond)
    print(f"  [{name}] {'PASS' if cond else 'FAIL'}")


def make_mg():
    mg = object.__new__(MemoryGraphManager)
    mg.graph = nx.MultiDiGraph()
    mg._lock = threading.RLock()
    mg._id_index = {}
    mg._id_index_dirty = True
    return mg


# ── 场景 1:snapshot_node_data 返回拷贝,后续写方修改不影响已取快照 ──
mg = make_mg()
mg.graph.add_node("A", name="Apple", type="org", source_chunks={"c1"}, ref_count=1)
snap = mg.snapshot_node_data("A")
mg.graph.nodes["A"]["name"] = "Apple Inc."
mg.graph.nodes["A"]["source_chunks"].add("c2")
s1 = snap["name"] == "Apple" and snap["source_chunks"] == {"c1"} and snap["type"] == "org"
s1 &= mg.snapshot_node_data("A")["name"] == "Apple Inc."  # 新快照看到新值
s1 &= mg.snapshot_node_data("NOPE") is None
report("snapshot_node_data 拷贝语义", s1)

# ── 场景 2:snapshot_edges 返回列表(非视图),修改后快照不变;in/out 正确 ──
mg = make_mg()
mg.graph.add_edge("A", "B", key="r1", relation="rel1", source_chunk="c1")
mg.graph.add_edge("C", "A", key="r2", relation="rel2", source_chunk="c2")
out_snap = mg.snapshot_edges("A", "out")
in_snap = mg.snapshot_edges("A", "in")
s2 = ([(u, v, d["relation"]) for u, v, d in out_snap] == [("A", "B", "rel1")]
      and [(u, v, d["relation"]) for u, v, d in in_snap] == [("C", "A", "rel2")]
      and mg.snapshot_edges("ZZZ", "out") == []
      and isinstance(out_snap, list))
mg.graph.remove_edge("A", "B", key="r1")
s2 &= len(out_snap) == 1  # 已取快照不受删除影响
report("snapshot_edges 方向/拷贝语义", s2)

# ── 场景 3:_id_index 脏标记:仅在有变更时重建,重建后索引正确 ──
mg = make_mg()
mg.graph.add_node(1, name="x", Id="x")
mg.graph.add_node(2, name="y", Id="y")
mg._rebuild_id_index()
s3 = mg._id_index_dirty is False and mg.get_nodes_by_id("x") == {1} and mg.get_nodes_by_id("y") == {2}
mg._rebuild_id_index()  # 未变更 → 不应重建
s3 &= mg._id_index_dirty is False
mg.graph.add_node(3, name="z", Id="z")
mg._id_index_dirty = True  # add_node 会置脏;此处模拟图结构已变更
mg._rebuild_id_index()
s3 &= mg.get_nodes_by_id("z") == {3}
report("_id_index 脏标记按需重建", s3)

# ── 场景 4:并发写 + 快照读 压力测试(写方增删节点/边,读方反复快照,不抛异常)──
mg = make_mg()
for i in range(20):
    mg.graph.add_node(i, name=f"n{i}", source_chunks={f"c{i}"})
    mg.graph.add_edge(i, (i + 1) % 20, key="r", relation="rel", source_chunk=f"c{i}")

stop = threading.Event()
errors = []


def writer():
    # 模拟生产写路径:add_node/add_edge/驱逐均在 _lock 内完成
    i = 1000
    while not stop.is_set():
        with mg._lock:
            mg.graph.add_node(i, name=f"w{i}", source_chunks={f"wc{i}"})
            mg.graph.add_edge(i, (i + 1), key="r", relation="rel", source_chunk=f"wc{i}")
            mg.graph.remove_node(i)
        i += 1
        time.sleep(0.0005)


def reader():
    i = 0
    # 只对固定 uid 集合做快照读(避免测试自身直接迭代 graph 造成竞态;
    # 生产代码的读均通过快照助手)
    fixed_uids = list(range(20))
    while not stop.is_set():
        try:
            for uid in fixed_uids:
                d = mg.snapshot_node_data(uid)
                _ = mg.snapshot_edges(uid, "out")
                _ = mg.snapshot_edges(uid, "in")
            _ = mg.snapshot_all_edges()
        except Exception as exc:  # noqa: BLE001
            errors.append(repr(exc))
            break
        i += 1


tw = threading.Thread(target=writer, daemon=True)
tr = threading.Thread(target=reader, daemon=True)
tw.start(); tr.start()
time.sleep(1.5)
stop.set()
tw.join(); tr.join()
s4 = not errors
report("并发写+快照读无竞态异常", s4)
if errors:
    print(f"        (errors: {errors[:2]})")

print("\n=== M4 图读快照验证:", "全部通过 ✅" if ok else "存在失败 ❌")
sys.exit(0 if ok else 1)
