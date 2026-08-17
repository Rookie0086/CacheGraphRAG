#!/usr/bin/env python
"""Small full-pipeline smoke test against local OpenAI-compatible models.

Model credentials are read through CACHEGRAPH_MODEL_* environment variables.
The script builds a small index, then queries it in the same process.
"""
import argparse
import asyncio
import getpass
import os
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.CacheGraphRAG import CacheGraphRAG
from src.utils import get_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=2)
    parser.add_argument("--agentic", action="store_true")
    parser.add_argument("--qa-concurrency", type=int, default=1)
    parser.add_argument("--agentic-parallelism", type=int, default=2)
    parser.add_argument("--skip-build", action="store_true",
                        help="Reuse the existing Milvus index and graph snapshot.")
    args = parser.parse_args()

    if not os.environ.get("CACHEGRAPH_MODEL_API_KEY"):
        os.environ["CACHEGRAPH_MODEL_API_KEY"] = getpass.getpass(
            "Local LLM API key: ")

    cfg = get_config()
    data_cfg = cfg.setdefault("data", {})
    ret_cfg = cfg.setdefault("retrieval", {})
    # Keep the smoke test fully local. The embedding endpoint shares the
    # OpenAI-compatible server and credential with the chat model.
    cfg.setdefault("embedding", {}).update({
        "backend": "api",
        "model_name": "bge-m3-mlx-fp16",
        "api_key": os.environ["CACHEGRAPH_MODEL_API_KEY"],
        "base_url": os.environ.get("CACHEGRAPH_MODEL_BASE_URL", "http://127.0.0.1:8000/v1"),
    })
    cfg.setdefault("rerank", {})["backend"] = "none"
    data_cfg["start"], data_cfg["end"] = args.start, args.end
    ret_cfg.update({
        "index_only": False,
        "skip_index": False,
        "clear_l2": False,
        "warmup_ratio": 0.0,
        "qa_concurrency": args.qa_concurrency,
        "agentic": args.agentic,
        "agentic_parallelism": args.agentic_parallelism,
    })

    app = CacheGraphRAG.from_config(cfg)
    try:
        common = dict(
            start=args.start,
            end=args.end,
            qa_concurrency=args.qa_concurrency,
            entity_extraction=ret_cfg.get("entity_extraction", "milvus"),
            qa_cache=False,
            use_agentic=args.agentic,
            agentic_steps=ret_cfg.get("agentic_steps", 3),
            entity_index_name=ret_cfg.get("entity_index_name"),
            chunk_collection=ret_cfg.get("chunk_collection"),
            answer_topk=ret_cfg.get("answer_topk", 6),
            top_chunks=ret_cfg.get("top_chunks"),
            mode=ret_cfg.get("mode", "hybrid"),
        )
        if args.skip_build:
            asyncio.run(app.index_only_qa(clear_l2=False, **common))
        else:
            asyncio.run(app.run(stream_mode=False, **common))
    finally:
        app.shutdown()


if __name__ == "__main__":
    main()
