"""Offline verification for the L1 topology rehydration trigger."""
import importlib.util
import pathlib
import sys
import types
import unittest


# Avoid importing the optional database runtime through retriever.py.
fake_milvus = types.ModuleType("database.milvus")
fake_milvus.MilvusDB = object
sys.modules.setdefault("database", types.ModuleType("database"))
sys.modules["database.milvus"] = fake_milvus

module_path = pathlib.Path(__file__).parents[1] / "src" / "retrieval" / "retriever.py"
spec = importlib.util.spec_from_file_location("rehydrate_retriever", module_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class FakeGraph:
    def __init__(self, l1=(), l2=(), succeeds=()):
        self.l1 = set(l1)
        self.l2 = set(l2)
        self.succeeds = set(succeeds)
        self.touched = []
        self.attempted = []

    def is_chunk_in_l1(self, chunk_id):
        return chunk_id in self.l1

    def is_chunk_in_l2(self, chunk_id):
        return chunk_id in self.l2

    def touch_chunk(self, chunk_id):
        self.touched.append(chunk_id)

    def rehydrate_chunk_from_milvus(self, chunk_id):
        self.attempted.append(chunk_id)
        if chunk_id in self.succeeds:
            self.l1.add(chunk_id)
            return True
        return False


class TopologyRehydrateTest(unittest.TestCase):
    def make_retriever(self, graph):
        obj = module.BaseRetriever.__new__(module.BaseRetriever)
        obj.memory_graph = graph
        obj.last_rehydrate = {}
        return obj

    def test_l2_chunk_evicted_from_l1_is_rehydrated(self):
        graph = FakeGraph(l2={"chunk_cold"}, succeeds={"chunk_cold"})
        stats = self.make_retriever(graph)._rehydrate_vector_hits([{"id": "chunk_cold"}])
        self.assertEqual(graph.attempted, ["chunk_cold"])
        self.assertEqual(stats["succeeded"], 1)

    def test_existing_l1_chunk_is_only_touched(self):
        graph = FakeGraph(l1={"chunk_hot"})
        stats = self.make_retriever(graph)._rehydrate_vector_hits([{"id": "chunk_hot"}])
        self.assertEqual(graph.touched, ["chunk_hot"])
        self.assertEqual(graph.attempted, [])
        self.assertEqual(stats["attempted"], 0)

    def test_non_chunk_hits_and_failures_are_reported(self):
        graph = FakeGraph()
        stats = self.make_retriever(graph)._rehydrate_vector_hits(
            [{"id": "entity_1"}, {"id": "chunk_missing"}])
        self.assertEqual(stats, {"attempted": 1, "succeeded": 0, "chunk_ids": []})


if __name__ == "__main__":
    unittest.main()
