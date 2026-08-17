"""Offline tests for cross-chunk delayed embedding batching."""
import asyncio
import importlib.util
import pathlib
import sys
import types
import unittest


for name in ("database", "database.milvus"):
    sys.modules.setdefault(name, types.ModuleType(name))
sys.modules["database.milvus"].MilvusDB = object
sys.modules["database.milvus"].myMilvus = object
fake_env = types.ModuleType("src.llm.env")
fake_env.LLMEnv = object
sys.modules["src.llm.env"] = fake_env

module_path = pathlib.Path(__file__).parents[1] / "src" / "pipeline.py"
spec = importlib.util.spec_from_file_location("batch_pipeline", module_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class FakeEmbed:
    def __init__(self):
        self.calls = []

    async def get_embeddings_async(self, texts):
        self.calls.append(list(texts))
        return [[float(len(text))] for text in texts]


def make_pipeline(batch_size=4, wait_ms=5):
    obj = module.DocumentIngestionPipeline.__new__(module.DocumentIngestionPipeline)
    obj.llm = types.SimpleNamespace(embed_model=FakeEmbed())
    obj._embed_batch_size = batch_size
    obj._embed_batch_wait_ms = wait_ms
    obj._embed_queue = None
    obj._embed_worker_task = None
    obj.embedding_api_calls = 0
    obj.embedding_items = 0
    obj.embedding_batches = 0
    return obj


class EmbeddingBatcherTest(unittest.IsolatedAsyncioTestCase):
    async def test_requests_from_multiple_chunks_are_coalesced(self):
        pipeline = make_pipeline(batch_size=8)
        await pipeline._start_embed_batcher()
        results = await asyncio.gather(
            pipeline._get_embeddings_batched(["a", "bb"]),
            pipeline._get_embeddings_batched(["ccc"]),
        )
        await pipeline._stop_embed_batcher()
        self.assertEqual(results, [[[1.0], [2.0]], [[3.0]]])
        self.assertEqual(pipeline.embedding_api_calls, 1)
        self.assertEqual(pipeline.embedding_items, 3)

    async def test_api_sub_batches_respect_text_limit(self):
        pipeline = make_pipeline(batch_size=2)
        await pipeline._start_embed_batcher()
        result = await pipeline._get_embeddings_batched(["a", "bb", "ccc", "dddd", "eeeee"])
        await pipeline._stop_embed_batcher()
        self.assertEqual(len(result), 5)
        self.assertEqual([len(call) for call in pipeline.llm.embed_model.calls], [2, 2, 1])
        self.assertEqual(pipeline.embedding_api_calls, 3)

    async def test_direct_call_fallback_is_supported(self):
        pipeline = make_pipeline(batch_size=4)
        result = await pipeline._get_embeddings_batched(["abc"])
        self.assertEqual(result, [[3.0]])
        self.assertEqual(pipeline.embedding_api_calls, 1)


if __name__ == "__main__":
    unittest.main()
