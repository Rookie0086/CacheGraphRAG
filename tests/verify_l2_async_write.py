# 验证论文靠齐修复(2026-08-15 第三轮)+ 并发加固(H2/M3,同轮):
#   - L2 晋升"异步固化"(论文 IV-C):写盘在后台线程,断言非主线程执行
#   - H2:晋升触发(access_chunk + 子图提取,含 Milvus 查询)整体后台完成
#   - M3:有界背压,积压超阈值退化同步执行;executor 关闭兜底;shutdown flush
import sys
import os
import time
import threading
import concurrent.futures
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.CacheGraphRAG import CacheGraphRAG

ok = True
results = []


def report(name, cond):
    global ok
    ok &= cond
    results.append(cond)
    print(f"  [{name}] {'PASS' if cond else 'FAIL'}")


class DummyMemGraph:
    """记录 write_to_persistent_graph / promote_subgraph / access_chunk 调用与线程。"""
    def __init__(self, trigger=True):
        self.calls = []
        self.threads = []
        self.trigger = trigger

    def write_to_persistent_graph(self, data):
        self.calls.append(("persist", data))
        self.threads.append(threading.get_ident())

    def promote_subgraph(self, ents, edges):
        self.calls.append(("promote", ents, edges))
        self.threads.append(threading.get_ident())

    def access_chunk(self, cid, increment=True):
        if not self.trigger:
            return False, {}
        return True, {"chunk_id": cid, "nodes": {}, "edges": []}


class Dummy:
    """只含 _submit_l2* 依赖的最小对象,直接复用 CacheGraphRAG 的助手方法。"""
    def __init__(self, max_pending=64):
        self._l2_writer = concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="l2-writer")
        self._l2_max_pending = max_pending
        self._l2_pending = 0
        self._l2_pending_lock = threading.Lock()
        self.mem_graph = DummyMemGraph()


# ── 场景 1:_submit_l2_persist 提交后在后台线程写盘(非阻塞主线程)──
d = Dummy()
d._submit_l2 = CacheGraphRAG._submit_l2.__get__(d, Dummy)
d._submit_l2_persist = CacheGraphRAG._submit_l2_persist.__get__(d, Dummy)
d._submit_l2_persist([{"chunk_id": "c1", "nodes": {}, "edges": []}])
time.sleep(0.3)
s1 = (len(d.mem_graph.calls) == 1
      and d.mem_graph.threads[0] != threading.get_ident()
      and d.mem_graph.calls[0][1] == [{"chunk_id": "c1", "nodes": {}, "edges": []}]
      and d._l2_pending == 0)
d._l2_writer.shutdown(wait=True)
report("async persist 后台线程写盘", s1)

# ── 场景 2:_submit_l2_promote 同样后台异步 ──
d = Dummy()
d._submit_l2 = CacheGraphRAG._submit_l2.__get__(d, Dummy)
d._submit_l2_promote = CacheGraphRAG._submit_l2_promote.__get__(d, Dummy)
d._submit_l2_promote({"1": {"name": "A"}}, [{"src": 1, "tgt": 2, "relation": "r"}])
time.sleep(0.3)
s2 = (len(d.mem_graph.calls) == 1
      and d.mem_graph.threads[0] != threading.get_ident()
      and d.mem_graph.calls[0] == ("promote", {"1": {"name": "A"}}, [{"src": 1, "tgt": 2, "relation": "r"}]))
d._l2_writer.shutdown(wait=True)
report("async promote 后台线程写盘", s2)

# ── 场景 3:H2 晋升触发整体后台(access_chunk + 写盘都在 worker 线程)──
d = Dummy()
d._submit_l2 = CacheGraphRAG._submit_l2.__get__(d, Dummy)
d._promote_chunk_in_background = CacheGraphRAG._promote_chunk_in_background.__get__(d, Dummy)
d._submit_l2(d._promote_chunk_in_background, "chunk_hot")
time.sleep(0.3)
s3 = (len(d.mem_graph.calls) == 1
      and d.mem_graph.threads[0] != threading.get_ident()
      and d.mem_graph.calls[0][1] == [{"chunk_id": "chunk_hot", "nodes": {}, "edges": []}])
d._l2_writer.shutdown(wait=True)
report("H2 晋升触发+写盘后台完成", s3)

# ── 场景 4:H2 未达晋升阈值(access_chunk=False)→ 不写盘 ──
d = Dummy()
d.mem_graph = DummyMemGraph(trigger=False)
d._submit_l2 = CacheGraphRAG._submit_l2.__get__(d, Dummy)
d._promote_chunk_in_background = CacheGraphRAG._promote_chunk_in_background.__get__(d, Dummy)
d._submit_l2(d._promote_chunk_in_background, "chunk_cold")
time.sleep(0.3)
s4 = d.mem_graph.calls == [] and d._l2_pending == 0
d._l2_writer.shutdown(wait=True)
report("H2 未触发不写盘", s4)

# ── 场景 5:M3 有界背压:积压达阈值后退化同步执行 ──
d = Dummy(max_pending=1)
d._submit_l2 = CacheGraphRAG._submit_l2.__get__(d, Dummy)
d._submit_l2_persist = CacheGraphRAG._submit_l2_persist.__get__(d, Dummy)
# 任务 A:正常入队(pending=1)
d._submit_l2_persist([{"chunk_id": "a", "nodes": {}, "edges": []}])
# 任务 B:积压已达上限 → 退化同步执行(立即写盘,主线程)
d._submit_l2_persist([{"chunk_id": "b", "nodes": {}, "edges": []}])
s5 = d._l2_pending == 1  # B 未入队
time.sleep(0.3)           # A 由后台线程完成
s5 &= len(d.mem_graph.calls) == 2 and d._l2_pending == 0
d._l2_writer.shutdown(wait=True)
report("M3 有界背压退化同步", s5)

# ── 场景 6:后台写盘异常被捕获,不崩溃 ──
class FailingDummy(Dummy):
    def __init__(self):
        super().__init__()
        self.mem_graph = DummyMemGraph()
        self.mem_graph.write_to_persistent_graph = lambda data: (_ for _ in ()).throw(RuntimeError("nebula down"))

d = FailingDummy()
d._submit_l2 = CacheGraphRAG._submit_l2.__get__(d, FailingDummy)
d._submit_l2_persist = CacheGraphRAG._submit_l2_persist.__get__(d, FailingDummy)
d._submit_l2_persist([{"chunk_id": "c2", "nodes": {}, "edges": []}])
time.sleep(0.3)  # 异常应被内部捕获,进程不崩溃
s6 = d._l2_pending == 0
d._l2_writer.shutdown(wait=True)
report("后台写盘异常不崩溃", s6)

# ── 场景 7:executor 已关闭(RuntimeError)→ 退化同步写,数据不丢 ──
d = Dummy()
d._submit_l2 = CacheGraphRAG._submit_l2.__get__(d, Dummy)
d._submit_l2_persist = CacheGraphRAG._submit_l2_persist.__get__(d, Dummy)
d._l2_writer.shutdown(wait=True)
d._submit_l2_persist([{"chunk_id": "c3", "nodes": {}, "edges": []}])
s7 = d.mem_graph.calls == [("persist", [{"chunk_id": "c3", "nodes": {}, "edges": []}])] and d._l2_pending == 0
report("executor 关闭后退化同步写", s7)

# ── 场景 8:shutdown() flush 语义 —— 提交后立即 shutdown 仍能全部落盘 ──
d = Dummy()
d._submit_l2 = CacheGraphRAG._submit_l2.__get__(d, Dummy)
d._submit_l2_persist = CacheGraphRAG._submit_l2_persist.__get__(d, Dummy)
d._submit_l2_persist([{"chunk_id": "c4", "nodes": {}, "edges": []}])
d._submit_l2_persist([{"chunk_id": "c5", "nodes": {}, "edges": []}])
d._l2_writer.shutdown(wait=True)  # 对应 CacheGraphRAG.shutdown() 中的 flush
s8 = len(d.mem_graph.calls) == 2 and d._l2_pending == 0
report("shutdown flush 队列落盘", s8)

print("\n=== L2 异步固化 + H2/M3 并发加固验证:", "全部通过 ✅" if ok else "存在失败 ❌")
sys.exit(0 if ok else 1)
