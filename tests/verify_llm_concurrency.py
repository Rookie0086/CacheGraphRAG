"""Verify that sync LLM calls share the configured global in-flight limit."""
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor

from src.llm.env import LLMEnv


class SlowBackend:
    def __init__(self):
        self.lock = threading.Lock(); self.active = 0; self.max_active = 0

    def complete(self, prompt, **kwargs):
        with self.lock:
            self.active += 1; self.max_active = max(self.max_active, self.active)
        time.sleep(.03)
        with self.lock:
            self.active -= 1
        return prompt


class LLMConcurrencyTest(unittest.TestCase):
    def test_global_request_gate_caps_parallel_calls(self):
        env = LLMEnv.__new__(LLMEnv)
        env.llm = SlowBackend(); env.backend = "fake"; env.total_calls = 0
        env._counter_lock = threading.Lock(); env.max_concurrency = 2
        env._request_gate = threading.BoundedSemaphore(2)
        env._inflight = 0; env.max_inflight_observed = 0
        with ThreadPoolExecutor(max_workers=6) as pool:
            values = list(pool.map(env.complete, [str(i) for i in range(6)]))
        self.assertEqual(values, [str(i) for i in range(6)])
        self.assertEqual(env.llm.max_active, 2)
        self.assertEqual(env.max_inflight_observed, 2)
        self.assertEqual(env.total_calls, 6)


if __name__ == "__main__":
    unittest.main()
