#!/usr/bin/env python3
"""小规模全链路冒烟:2Wiki 前 3 条 QA —— 验证 L1 空启动 + 本地 embedding(oMLX bge-m3-mlx-4bit)
+ 本地 rerank(oMLX bge-reranker-v2-m3) + 云 LLM(gpt-4o-mini) + L2 晋升 整链。

用法(conda python):
  .conda/cachegraphrag-mac/bin/python scripts/smoke_local_chain.py
"""
import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("CACHEGRAPH_EMBED_API_KEY", open(".secrets/local_llm_key").read().strip())
os.environ.setdefault("CACHEGRAPH_RERANK_API_KEY", open(".secrets/local_llm_key").read().strip())

from src.CacheGraphRAG import CacheGraphRAG
from src.utils import get_config


async def main():
    cfg = get_config()
    print(f"embedding: {cfg['embedding']['model_name']} @ {cfg['embedding']['base_url']}")
    print(f"rerank:    {cfg['rerank']['model_name']} @ {cfg['rerank']['base_url']} (backend={cfg['rerank']['backend']})")
    print(f"load_gexf: {cfg['retrieval'].get('load_gexf', False)} (L1 空启动)")
    t0 = time.time()
    app = CacheGraphRAG.from_config(cfg)
    print(f"初始化耗时: {time.time()-t0:.1f}s")
    # 冒烟:3 条 QA,clear_l2=False 不清主 space L2,qa_concurrency=1 稳
    await app.index_only_qa(start=0, end=3, qa_concurrency=1,
                            clear_l2=False, use_agentic=False)
    app.shutdown()
    print("\n[SMOKE] 全链路通过 ✅")


if __name__ == "__main__":
    asyncio.run(main())
