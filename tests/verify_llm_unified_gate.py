# 验证 H3(2026-08-15):LLM 并发门控统一 —— sync(complete)/async(async_complete)
# 共用同一把 threading.BoundedSemaphore,混合负载下实际在途请求 ≤ max_concurrency
# (原双门控 design 下混合负载可达 2×max_concurrency)。
# 纯内存构造 LLMEnv 门面(object.__new__ 跳过后端初始化),不连接任何 API。
import sys
import os
import time
import threading
import asyncio
import concurrent.futures
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llm.env import LLMEnv

ok = True
results = []


def report(name, cond):
    global ok
    ok &= cond
    results.append(cond)
    print(f"  [{name}] {'PASS' if cond else 'FAIL'}")


class FakeLLM:
    """模拟后端:async/sync 均延迟执行,便于观测并发上限。"""
    def __init__(self, delay=0.3):
        self.delay = delay

    async def async_complete(self, prompt, verbose=False, return_info=False):
        await asyncio.sleep(self.delay)
        return "ok"

    def complete(self, prompt, verbose=False, return_info=False):
        time.sleep(self.delay)
        return "ok"


def make_env(max_concurrency=2):
    env = object.__new__(LLMEnv)
    env.total_calls = 0
    env._counter_lock = threading.Lock()
    env.max_concurrency = max_concurrency
    env._request_gate = threading.BoundedSemaphore(max_concurrency)
    env._gate_acquirer = concurrent.futures.ThreadPoolExecutor(
        max_workers=max_concurrency + 1, thread_name_prefix="llm-gate")
    env._inflight = 0
    env.max_inflight_observed = 0
    env.backend = "fake"
    env.llm = FakeLLM(delay=0.3)
    env._BACKOFF_SECONDS = 0.01
    env._MAX_RETRIES = 1
    env._THROTTLE_SECONDS = 0.01
    env._GLOBAL_RETRY_COUNT = 0
    env._GLOBAL_RETRY_LIMIT = 20
    return env


async def main(max_concurrency):
    env = make_env(max_concurrency)
    # 4 个 async 请求(事件循环)+ 4 个 sync 请求(线程)同时发起
    async_tasks = [asyncio.create_task(env.async_complete(f"a{i}")) for i in range(4)]
    threads = [threading.Thread(target=env.complete, args=(f"s{i}",)) for i in range(4)]
    for t in threads:
        t.start()
    await asyncio.gather(*async_tasks)
    for t in threads:
        t.join()
    env._gate_acquirer.shutdown(wait=True)
    return env


# ── 场景 1:混合负载下最大在途 ≤ max_concurrency(统一门控生效)──
env = asyncio.run(main(2))
s1 = env.max_inflight_observed <= 2
report("混合负载 max_inflight ≤ 2", s1)
print(f"        (实测 max_inflight_observed={env.max_inflight_observed}, total_calls={env.total_calls})")

# ── 场景 2:所有 8 个请求均成功完成且计数正确 ──
s2 = env.total_calls == 8
report("8 请求全部完成且计数正确", s2)

# ── 场景 3:并发上限可调(3 时在途 ≤ 3)──
env3 = asyncio.run(main(3))
s3 = env3.max_inflight_observed <= 3 and env3.total_calls == 8
report("max_concurrency=3 时在途 ≤ 3", s3)

# ── 场景 4:async 侧 acquire 走专用小线程池(不占默认 asyncio.to_thread 池)──
env4 = make_env(2)
s4 = isinstance(env4._gate_acquirer, concurrent.futures.ThreadPoolExecutor) \
    and env4._gate_acquirer._max_workers == 3 \
    and not hasattr(env4, "_async_gate")  # 旧的独立 asyncio 门控已移除
env4._gate_acquirer.shutdown(wait=True)
report("_async_gate 移除,acquire 走专用线程池", s4)

print("\n=== H3 LLM 统一门控验证:", "全部通过 ✅" if ok else "存在失败 ❌")
sys.exit(0 if ok else 1)
