"""Offline regression test for agentic beam final_answer handling.

背景(bug 根因):修复前 `_plan_candidates` 在 is_final=true 时直接 `return []`,
把 LLM 针对原始问题生成的 final_answer 丢弃,导致 `_run_beam` 兜底使用
"子问题"的回答 → 答非所问(b4 束宽=4 时 EM=0)。

本测试锁定修复后的契约:
  1. `_plan_candidates` 返回 (候选子问题列表, final_answer) 二元组;
  2. is_final=true 时 final_answer 必须被透传,不被丢弃;
  3. `_run_beam` / `run` 的最终答案优先使用 LLM 综合答案,而非子问题回答。
"""
import importlib.util
import pathlib
import sys
import unittest
from unittest.mock import patch

import numpy as np

# 直接加载模块,避免离线单测引入 Milvus/Nebula 运行时。
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

    def __init__(self, responses):
        # responses: list[str],每次 complete 依序弹出;耗尽后返回 '{}'
        self._responses = list(responses)

    def complete(self, prompt):
        return self._responses.pop(0) if self._responses else '{}'


class FakeVectorStore:
    def get_chunk_text(self, chunk_id):
        return {"text": f"evidence {chunk_id}", "ts": "2026"}


class FakeRetriever:
    llm = FakeLLM(["{}"])

    def __init__(self, llm=None):
        if llm is not None:
            self.llm = llm

    def hybrid_retrieve(self, question, **kwargs):
        return {"chunks": [f"{question}-0", f"{question}-1"]}


class AgenticFinalAnswerTest(unittest.TestCase):
    def make_engine(self, width=2, llm=None):
        config = {"retrieval": {"beam_width": width, "unknown_tolerance": 2},
                  "hyperparameters": {"B": width}}
        with patch.object(agentic_module, "get_config", return_value=config):
            engine = IterativeAgenticEngine(llm or FakeLLM([]), "test",
                                            FakeRetriever(llm), max_steps=3)
        # 固定首跳,避免依赖网络
        engine._plan_first_step = lambda query: {"need_multihop": True, "subquestion": "root"}
        engine._answer_with_chunks = lambda question, chunks: (f"sub_ans:{question}", "")
        return engine

    def test_plan_candidates_is_final_returns_final_answer(self):
        """修复 A:is_final 时必须透传 LLM 的 final_answer,而非丢弃。"""
        llm = FakeLLM(['{"is_final": true, "subquestions": [], '
                       '"final_answer": "Port of Spain"}'])
        engine = self.make_engine(width=2, llm=llm)
        candidates, final_answer = engine._plan_candidates("q", [{"question": "r", "answer": "x"}], 2)
        self.assertEqual(candidates, [])
        self.assertEqual(final_answer, "Port of Spain")

    def test_plan_candidates_returns_candidates_when_not_final(self):
        """非 final 时返回候选子问题, final_answer 为空。"""
        llm = FakeLLM(['{"is_final": false, "subquestions": ["who", "when"], '
                       '"final_answer": ""}'])
        engine = self.make_engine(width=2, llm=llm)
        candidates, final_answer = engine._plan_candidates("q", [{"question": "r", "answer": "x"}], 2)
        self.assertEqual(sorted(candidates), ["when", "who"])
        self.assertEqual(final_answer, "")

    def test_plan_candidates_width1_is_final_returns_final_answer(self):
        """束宽 1 走 _plan_next_step,is_final 时同样透传 final_answer。"""
        llm = FakeLLM(['{"is_final": true, "subquestion": "", '
                       '"final_answer": "20 March 851"}'])
        engine = self.make_engine(width=1, llm=llm)
        candidates, final_answer = engine._plan_candidates("q", [{"question": "r", "answer": "x"}], 1)
        self.assertEqual(candidates, [])
        self.assertEqual(final_answer, "20 March 851")

    def test_run_beam_uses_synthesized_final_answer(self):
        """修复 A:beam 收敛时最终答案应为 LLM 综合答案,而非子问题回答。"""
        engine = self.make_engine(width=2)
        # 第一轮:is_final 且给出针对原始问题的简洁答案
        engine._plan_candidates = lambda query, history, width: ([], "God'S Gift To Women")
        result = engine.run("Which film has the older director?")
        self.assertEqual(result["final_answer"], "God'S Gift To Women")
        # 修复前 _plan_candidates 返回 [] → _run_beam 兜底用子问题回答
        # "sub_ans:root",必然断言失败。


if __name__ == "__main__":
    unittest.main()
