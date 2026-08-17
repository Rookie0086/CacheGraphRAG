# 验证 L1 空启动模型(2026-08-15 第六轮):
#   默认 retrieval.load_gexf=false:启动/QA 不加载 base gexf,L1 初始为空,
#   QA 期由 DPR 命中触发 rehydrate_chunk_from_milvus 从 Milvus graph_meta 按 chunk
#   重载填充至容量上限(论文 IV-C Lazy Rehydration,与 fig5 冷启动一致)。
#   默认 data.save_gexf=false:构建后不写 gexf 快照(仅观察用,不计入存储审计)。
#   纯源码/config 断言(不构造重型对象)。
import sys
import os
import yaml
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
with open(os.path.join(repo, "config", "config.yaml"), encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

# ── 场景 1:config 默认 load_gexf=false / save_gexf=false ──
s1 = (cfg.get("retrieval", {}).get("load_gexf", True) is False
      and cfg.get("data", {}).get("save_gexf", True) is False)
report("config 默认 load_gexf=false / save_gexf=false", s1)

# ── 场景 2:__init__ 的 gexf 加载被 load_gexf 条件包住(空启动默认分支)──
init_zone = cgr.split("def __init__")[1].split("def ingest")[0]
s2 = ("load_gexf = bool(cfg.get(\"retrieval\", {}).get(\"load_gexf\", False))" in init_zone
      and init_zone.count("if load_gexf:") >= 1
      and "空启动(lazy rehydration)" in init_zone)
report("__init__ gexf 加载受 load_gexf 开关控制", s2)

# ── 场景 3:index() 的 gexf 快照写入被 save_gexf 条件包住 ──
index_zone = cgr.split("async def index(")[1].split("async def index_only_qa")[0]
s3 = ("save_gexf" in index_zone
      and index_zone.count("save_graph_gexf") >= 1
      and "if bool(get_config().get(\"data\", {}).get(\"save_gexf\", False)):" in index_zone)
report("index() 写 gexf 受 save_gexf 开关控制", s3)

# ── 场景 4:index_only_qa 基线模式不加载 base gexf(空 L1 启动)──
qa_zone = cgr.split("async def index_only_qa")[1].split("async def _run_split_qa")[0]
s4 = ("cfg.get(\"retrieval\", {}).get(\"load_gexf\", False)" in qa_zone
      and "L1 空启动(lazy rehydration)" in qa_zone)
report("index_only_qa 基线模式空 L1 启动", s4)

# ── 场景 5:resume_gexf(增量/断点续跑)保留,不被默认关闭 ──
s5 = "if resume_gexf and os.path.exists(resume_gexf):" in qa_zone
report("resume_gexf 增量模式保留", s5)

# ── 场景 6:空 L1 的重载数据源(graph_meta rehydrate)链路仍在 ──
with open(os.path.join(repo, "src", "graph", "memory_graph.py"), encoding="utf-8") as f:
    mg = f.read()
s6 = "def rehydrate_chunk_from_milvus" in mg and "graph_meta" in mg
report("Milvus graph_meta 延迟重载链路仍在(空 L1 数据源)", s6)

print("\n=== L1 空启动模型验证:", "全部通过 ✅" if ok else "存在失败 ❌")
sys.exit(0 if ok else 1)
