# 验证 M1(2026-08-15):专用线程池 —— QA 检索/引擎(engine.run、hybrid_retrieve)与
# 构建期 Milvus 插入不再占用默认 asyncio.to_thread 池(避免与阻塞型任务互抢)。
# 源码断言 + 配置断言(纯静态检查,不构造重型对象)。
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ok = True
results = []


def report(name, cond):
    global ok
    ok &= cond
    results.append(cond)
    print(f"  [{name}] {'PASS' if cond else 'FAIL'}")


repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

with open(os.path.join(repo, "src", "CacheGraphRAG.py"), encoding="utf-8") as f:
    cgr = f.read()
with open(os.path.join(repo, "src", "pipeline.py"), encoding="utf-8") as f:
    pl = f.read()
with open(os.path.join(repo, "config", "config.yaml"), encoding="utf-8") as f:
    cfg = f.read()

# ── 场景 1:CacheGraphRAG 不再用 asyncio.to_thread(调用形式),QA 检索/引擎走 _qa_executor ──
s1 = ("asyncio.to_thread(" not in cgr
      and "self._qa_executor" in cgr
      and "run_in_executor(\n                        self._qa_executor, engine.run" in cgr
      and "run_in_executor(\n                            self._qa_executor, retriever.hybrid_retrieve" in cgr
      and 'thread_name_prefix="qa-worker"' in cgr)
report("QA 检索/引擎走专用 _qa_executor", s1)

# ── 场景 2:shutdown 回收 _qa_executor ──
s2 = "self._qa_executor.shutdown(wait=True)" in cgr
report("shutdown 回收 _qa_executor", s2)

# ── 场景 3:pipeline 构建期 Milvus 插入走 _io_executor,不再用 asyncio.to_thread(调用形式)──
s3 = ("asyncio.to_thread(" not in pl
      and "self._io_executor" in pl
      and "run_in_executor(\n            self._io_executor, self.vector_store.insert_chunk" in pl
      and 'thread_name_prefix="pipeline-io"' in pl)
report("构建期插入走专用 _io_executor", s3)

# ── 场景 4:M2 配置键已落 config.yaml ──
s4 = ("embedding_concurrency" in cfg and "rerank_concurrency" in cfg
      and "graph_entity_parallel" in cfg)
report("config 含 M2 并发配置键", s4)

# ── 场景 5:其余 to_thread 调用点(如有)已收敛(仅注释可含字样)──
for fname, content in [("src/CacheGraphRAG.py", cgr), ("src/pipeline.py", pl)]:
    assert "asyncio.to_thread(" not in content, fname

print("\n=== M1 专用线程池验证:", "全部通过 ✅" if ok else "存在失败 ❌")
sys.exit(0 if ok else 1)
