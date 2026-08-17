#!/usr/bin/env python
"""验证修复:query() 接线 chunk_vector_store 后,rehydrate 对真实 Milvus graph_meta 能成功。
用法:
  CACHEGRAPH_CONFIG=output/experiments/run_20260815_002015/lru_rehydrate/capacity_50/config.yaml \
  python scripts/experiments/diag_rehydrate_fixed.py
"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.CacheGraphRAG import CacheGraphRAG
from src.utils import get_config

CacheGraphRAG._check_databases = staticmethod(lambda: None)

cfg = get_config()
app = CacheGraphRAG.from_config(cfg)
ret_cfg = cfg.get("retrieval", {})
ccol = ret_cfg.get("chunk_collection") or app.nebula_space

# 复刻 query() 的接线逻辑(修复后的代码)
from database.milvus import MilvusDB
vector_store = MilvusDB(db_name=ccol, overwrite=False, embed_model=app.llm.embed_model)
app.mem_graph.chunk_vector_store = vector_store
print(f"[diag] chunk_vector_store 已接线 -> {app.mem_graph.chunk_vector_store is not None}")

# 从 Milvus 拿一个带 graph_meta 的真实 chunk_id
col = vector_store._get_db()
try:
    results = col.query(expr="chunk_id != ''", output_fields=["chunk_id"], limit=30)
except Exception as exc:
    print(f"[diag] 查询 collection 失败: {exc}")
    sys.exit(1)
cids = [r.get("chunk_id") for r in results]
print(f"[diag] 拿到 {len(cids)} 个 chunk_id: {[str(c)[:20] for c in cids[:5]]}")

success = 0
for cid in cids:
    meta = vector_store.get_chunk_graph_meta(cid)
    if not (meta and meta.get("entities")):
        continue
    # 模拟该 chunk 已在 L1(不重复 rehydrate)时跳过;只测不在 L1 的
    if app.mem_graph.is_chunk_in_l1(cid):
        app.mem_graph._evict_chunk_locked(cid) if cid in app.mem_graph.chunk_meta else None
    ok = app.mem_graph.rehydrate_chunk_from_milvus(cid)
    print(f"[diag] rehydrate({str(cid)[:22]}) = {ok} "
          f"(entities={len(meta.get('entities', []))})")
    success += int(ok)

print(f"\n[diag] 成功 {success}/{len(cids)} 个样本")
print(f"[diag] 统计: attempts={app.mem_graph.rehydrate_attempts} "
      f"successes={app.mem_graph.rehydrate_successes} "
      f"failures={app.mem_graph.rehydrate_failures} "
      f"restored_nodes={app.mem_graph.rehydrated_nodes}")
