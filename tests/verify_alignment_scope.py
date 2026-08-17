# 验证论文靠齐修复(2026-08-15 第三轮):
#   1) 对齐比对空间限定活跃节点集 V_L1(resolver.scope="l1",论文式7/算法1)
#   2) desc 相似度 = 纯余弦 cos(d_new,d_old)(算法1 行9/12/15,替换 Jaccard+余弦 0.5/0.5 混合)
#   3) 同名异型 + cos(d)≥τ_desc → 合并(算法1 行15-16,容忍 LLM 类型偏差,替换原"新实体 type 为空"窄条件)
# 纯内存 mock,不连接 Milvus。embedding_func 必须为 async 函数(gather 语义)。
import sys
import os
import asyncio
from types import SimpleNamespace
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest import mock
import src.entity.resolver as RE


class FakeMilvusDB:
    def __init__(self, db_name=None, overwrite=False, embed_model=None):
        self.db_name = db_name
        self.inserted = []

    def create_entity_collection(self):
        pass

    def load(self):
        pass

    def insert(self, data):
        self.inserted.extend(data)


class FakeMilvusClient:
    def list_collections(self):
        return []


class FakeGraph:
    def __init__(self, nodes):
        self._nodes = set(nodes)

    def has_node(self, uid):
        return uid in self._nodes


class FakeMemoryGraph:
    def __init__(self, nodes):
        self.graph = FakeGraph(nodes)

    def add_node(self, *a, **k):
        pass

    def add_edge(self, *a, **k):
        pass


async def emb_same(t):
    """任意输入 → 同一向量:cos=1.0。"""
    return [1.0, 0.0]


async def emb_orth(t):
    """"x-" 前缀 → [1,0];其余 → [0,1]:前缀内外正交。"""
    return [1.0, 0.0] if t.startswith("x-") else [0.0, 1.0]


async def emb_by_word(t):
    """含 "president" → [1,0];否则 → [0,1]。"""
    return [1.0, 0.0] if "president" in t else [0.0, 1.0]


async def failing_embed(t):
    raise RuntimeError("embed api down")


def make_resolver(**kw):
    defaults = dict(embedding_func=emb_same, memory_graph=FakeMemoryGraph({123}))
    defaults.update(kw)
    with mock.patch.object(RE, "MilvusDB", FakeMilvusDB), \
         mock.patch.object(RE, "myMilvus", lambda: FakeMilvusClient()):
        return RE.AsyncEntityResolver(
            collection_name="entity_index_test",
            **defaults)


def hit(uid, name, etype, desc, distance):
    return SimpleNamespace(
        entity={"uid": uid, "name": name, "type": etype, "desc": desc},
        distance=distance,
    )


async def _resolve(r, name, etype, desc, hits, vector=(0.9, 0.1)):
    r._search_milvus = lambda v, params, limit: [hits]
    return await r._resolve_with_vector(
        name, etype, desc, list(vector),
        cache_key=f"{name}[{etype}]:{desc}", name_key=f"{name}|{etype}")


ok = True
results = []


def report(name, cond):
    global ok
    ok &= cond
    results.append(cond)
    print(f"  [{name}] {'PASS' if cond else 'FAIL'}")


# ── 场景 1:scope="l1",命中实体在 L1(V_L1)内 → τ_sim 合并返回原 uid ──
async def s1():
    r = make_resolver()
    uid = await _resolve(r, "Apple", "org", "tech company", [hit(123, "Apple", "org", "tech company", 0.9)])
    return uid == 123 and 123 in r._active_uids
report("V_L1 活跃节点 τ_sim 合并", asyncio.run(s1()))

# ── 场景 2:scope="l1",命中实体不在 V_L1 → 视为未命中,生成新 uid ──
async def s2():
    r = make_resolver()
    uid = await _resolve(r, "Apple", "org", "tech company", [hit(456, "Apple", "org", "tech company", 0.9)])
    return uid != 456
report("V_L1 非活跃节点被过滤", asyncio.run(s2()))

# ── 场景 3:scope="full"(旧行为)→ 不限制比对空间,命中 456 直接合并 ──
async def s3():
    r = make_resolver(scope="full")
    uid = await _resolve(r, "Apple", "org", "tech company", [hit(456, "Apple", "org", "tech company", 0.9)])
    return uid == 456
report("V_L1 scope=full 退化全量", asyncio.run(s3()))

# ── 场景 4:_active_uids 机制:本次会话先新建的实体,后续实体可与之合并 ──
async def s4():
    r = make_resolver()
    # 第一次:Milvus 无命中 → 孤立节点,生成新 uid X
    first = await _resolve(r, "John Smith", "person", "writer from 1800s", [])
    # 第二次:同名同型,desc 相似(emb_same → cos=1.0 ≥ τ_desc=0.6),
    # Milvus 命中第一次的 uid(在 _active_uids 中,视为活跃)→ 合并回 X
    second = await _resolve(r, "John Smith", "person", "writer from 1900s",
                            [hit(first, "John Smith", "person", "writer from 1800s", 0.6)])
    return second == first and first in r._active_uids
report("V_L1 会话内实体可合并(_active_uids)", asyncio.run(s4()))

# ── 场景 5:_in_active_set 直接断言 ──
r = make_resolver()
r._active_uids.add(999)
s5 = (r._in_active_set(123) and r._in_active_set("123")
      and r._in_active_set(999) and not r._in_active_set(777)
      and r._in_active_set(None) is False)
report("_in_active_set 判定", s5)

# ── 场景 6:desc 相似度 = 纯余弦(替换 Jaccard+余弦混合)──
async def s6():
    r = make_resolver(embedding_func=emb_orth)
    same = await r._desc_similarity("x-same one", "x-same two")   # 同向量 → cos=1.0
    orth = await r._desc_similarity("x-orth a", "y-orth b")       # 正交向量 → cos=0.0
    return abs(same - 1.0) < 1e-9 and abs(orth - 0.0) < 1e-9
report("desc 纯余弦", asyncio.run(s6()))

# ── 场景 7:嵌入失败回退 token Jaccard ──
async def s7():
    r = make_resolver(embedding_func=failing_embed)
    v = await r._desc_similarity("alpha beta", "alpha gamma")
    return abs(v - 1.0 / 3.0) < 1e-9  # {alpha,beta}∩{alpha,gamma} / 并集 = 1/3
report("desc 嵌入失败回退", asyncio.run(s7()))

# ── 场景 8:同名异型 + cos(d)≥τ_desc → 合并(算法1 行15-16,容忍类型偏差)──
async def s8():
    r = make_resolver(memory_graph=FakeMemoryGraph({321}))
    # 名称相同("bush"),类型不同(person vs organization),desc 相似(cos=1.0 ≥ 0.6)
    uid = await _resolve(r, "Bush", "person", "US president",
                         [hit(321, "Bush", "organization", "US president", 0.5)])
    return uid == 321  # 类型偏差被容忍 → 合并
report("同名异型+cos≥τ_desc 合并", asyncio.run(s8()))

# ── 场景 9:同名同型 + cos(d)<τ_desc → possible same as 软边(算法1 行12-14)──
async def s9():
    r = make_resolver(embedding_func=emb_by_word, memory_graph=FakeMemoryGraph({321}))
    # 同名同型但 desc 正交(cos=0.0 < 0.6)→ 新 uid + possible_same_as 边
    uid = await _resolve(r, "Bush", "person", "a plant",
                         [hit(321, "Bush", "person", "US president", 0.5)])
    return uid != 321
report("同名同型+cos<τ_desc 新建 uid", asyncio.run(s9()))

# ── 场景 10:resolve_async 的 name_cache 复用阈值改为 τ_desc(0.3 残留检查)──
src_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "src", "entity", "resolver.py")
with open(src_path, encoding="utf-8") as f:
    src = f.read()
s10 = ("desc_sim >= 0.3" not in src
       and "desc_sim >= self.desc_threshold" in src.split("def _resolve_with_vector")[1])
report("name_cache 阈值对齐 τ_desc", s10)

print("\n=== 论文靠齐(比对空间/desc 余弦/类型偏差)验证:", "全部通过 ✅" if ok else "存在失败 ❌")
sys.exit(0 if ok else 1)
