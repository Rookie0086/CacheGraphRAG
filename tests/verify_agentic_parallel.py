"""Offline verification for parallel beam evaluation and batched planning."""
import json
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np

import src.retrieval.agentic_engine as module
from src.retrieval.agentic_engine import IterativeAgenticEngine


class Embed:
    def get_embedding(self, text):
        return np.array([1.0, float(len(text) + 1)])


class BatchLLM:
    embed_model = Embed()

    def __init__(self):
        self.calls = 0

    def complete(self, prompt):
        self.calls += 1
        if "Paths:" in prompt:
            return json.dumps({"paths": [
                {"id": 0, "is_final": False, "subquestions": ["a-next"], "final_answer": ""},
                {"id": 1, "is_final": True, "subquestions": [], "final_answer": "done"},
            ]})
        return "{}"


class ParallelRetriever:
    def __init__(self, llm):
        self.llm = llm
        self.vector_store = type("Store", (), {"get_chunk_text": lambda *_: {"text": "x", "ts": "t"}})()
        self.lock = threading.Lock()
        self.active = 0
        self.max_active = 0

    def hybrid_retrieve(self, question, **kwargs):
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        if question in {"a", "b"}:
            time.sleep(0.08)
        with self.lock:
            self.active -= 1
        return {
            "chunks": [f"chunk-{question}"],
            "memory": [{"chunk_scores": {f"l1-{question}": 1.0}}],
            "persistent": [{"chunk_scores": {f"l2-{question}": 1.0}}],
            "stats": {"l2_query_errors": 0, "latency": {"graph": 0.01}},
        }


def make_engine(batch=True):
    llm = BatchLLM(); retriever = ParallelRetriever(llm)
    cfg = {"retrieval": {"beam_width": 2, "unknown_tolerance": 2,
                         "agentic_parallelism": 2, "agentic_batch_planning": batch,
                         "enable_semantic_block": False}}
    with patch.object(module, "get_config", return_value=cfg):
        engine = IterativeAgenticEngine(llm, "test", retriever, max_steps=2)
    engine._plan_first_step = lambda _: {"need_multihop": True, "subquestion": "root"}
    engine._answer_with_chunks = lambda question, chunks: (f"answer-{question}", "")
    return engine, llm, retriever


class AgenticParallelTest(unittest.TestCase):
    def test_beam_states_execute_concurrently(self):
        engine, _, retriever = make_engine(batch=False)
        engine._plan_candidates = lambda q, h, w: ((["a", "b"], "") if len(h) == 1 else ([], ""))
        result = engine.run("original")
        self.assertGreaterEqual(retriever.max_active, 2)
        self.assertEqual(result["retrieval_calls"], 3)
        self.assertEqual(result["l1_hits"], 3)
        self.assertEqual(result["l2_hits"], 3)
        self.assertEqual(result["l2_query_errors"], 0)
        self.assertAlmostEqual(result["stage_latency"]["graph"], 0.03)

    def test_multiple_paths_share_one_planning_call(self):
        engine, llm, _ = make_engine(batch=True)
        paths = [
            {"path_id": 0, "base": {"history": [{"question": "q0", "answer": "x"}]}},
            {"path_id": 1, "base": {"history": [{"question": "q1", "answer": "y"}]}},
        ]
        result = engine._plan_candidates_batch("original", paths, 2)
        self.assertEqual(llm.calls, 1)
        self.assertEqual(result[0][0], ["a-next"])
        self.assertEqual(result[1][1], "done")
        self.assertEqual(engine.batched_planning_calls, 1)


if __name__ == "__main__":
    unittest.main()
