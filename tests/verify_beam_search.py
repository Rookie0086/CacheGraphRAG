"""Offline checks for configurable agentic beam search."""
import importlib.util
import pathlib
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np

# Load the module directly so importing src.retrieval does not eagerly import
# the optional Milvus/Nebula runtime in an offline unit test.
module_path = pathlib.Path(__file__).parents[1] / "src" / "retrieval" / "agentic_engine.py"
spec = importlib.util.spec_from_file_location("beam_agentic_engine", module_path)
agentic_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agentic_module)
IterativeAgenticEngine = agentic_module.IterativeAgenticEngine


class FakeEmbed:
    def get_embedding(self, text):
        return np.array([1.0, float(len(text) % 7 + 1)])


class FakeLLM:
    embed_model = FakeEmbed()

    def complete(self, prompt):
        return '{}'


class FakeVectorStore:
    def get_chunk_text(self, chunk_id):
        return {"text": f"evidence {chunk_id}", "ts": "2026"}


class FakeRetriever:
    llm = FakeLLM()
    vector_store = FakeVectorStore()

    def hybrid_retrieve(self, question, **kwargs):
        count = {"root": 1, "strong": 3, "weak": 1}.get(question, 0)
        return {"chunks": [f"{question}-{i}" for i in range(count)]}


class BeamSearchTest(unittest.TestCase):
    def make_engine(self, width=2):
        config = {"retrieval": {"beam_width": width, "unknown_tolerance": 2},
                  "hyperparameters": {"B": width}}
        with patch.object(agentic_module, "get_config", return_value=config):
            engine = IterativeAgenticEngine(FakeLLM(), "test", FakeRetriever(), max_steps=2)
        engine._plan_first_step = lambda query: {"need_multihop": True, "subquestion": "root"}
        engine._answer_with_chunks = lambda question, chunks: (f"answer {question}", "")
        return engine

    def test_beam_prefers_path_with_more_evidence(self):
        engine = self.make_engine(2)
        engine._plan_candidates = lambda query, history, width: (["strong", "weak"] if len(history) == 1 else [], "")
        result = engine.run("original")
        self.assertEqual(result["beam_width"], 2)
        self.assertEqual(result["agentic_steps"][-1]["question"], "strong")
        self.assertEqual(len(result["chunks"]), 4)

    def test_width_one_uses_legacy_greedy_path(self):
        engine = self.make_engine(1)
        sentinel = {"chunks": ["legacy"]}
        with patch.object(engine, "_run_beam", side_effect=AssertionError("beam path used")):
            engine.retriever.hybrid_retrieve = lambda *args, **kwargs: {"chunks": ["legacy"]}
            engine._plan_next_step = lambda *args: {"is_final": True, "final_answer": "done"}
            result = engine.run("original")
        self.assertEqual(result["chunks"], sentinel["chunks"])

    def test_beam_is_capped_to_configured_width(self):
        engine = self.make_engine(2)
        engine.max_steps = 1
        engine._plan_candidates = lambda query, history, width: (["a", "b", "c", "d"], "")
        result = engine.run("original")
        self.assertEqual(result["beam_width"], 2)


if __name__ == "__main__":
    unittest.main()
