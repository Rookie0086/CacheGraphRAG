# 验证 H1(2026-08-15):NebulaClient 单 session 线程安全收口
#   - 所有 session.execute 调用改走 _execute(锁内串行化)
#   - 并发 execute 严格串行(max_active == 1)
#   - init_session_pool 重建有锁保护
# 纯内存 mock ConnectionPool/Session,不连接真实 Nebula。
import sys
import os
import time
import threading
from types import SimpleNamespace
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest import mock
import database.nebulagraph as NG

ok = True
results = []


def report(name, cond):
    global ok
    ok &= cond
    results.append(cond)
    print(f"  [{name}] {'PASS' if cond else 'FAIL'}")


class FakeSession:
    """记录并发度与语句;并发度由 execute 内的 active 计数观测。"""
    def __init__(self):
        self.active = 0
        self.max_active = 0
        self.lock = threading.Lock()
        self.statements = []

    def execute(self, stmt):
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(0.05)
        with self.lock:
            self.active -= 1
        self.statements.append(stmt)
        return SimpleNamespace(is_succeeded=lambda: True)

    def release(self):
        pass


class FakePool:
    def __init__(self):
        self.session = FakeSession()

    def init(self, addrs, config):
        return True

    def get_session(self, user, password):
        return self.session

    def close(self):
        pass


with mock.patch.object(NG, "ConnectionPool", FakePool), \
     mock.patch.object(NG, "Config", lambda: SimpleNamespace()):
    client = NG.NebulaClient()

# ── 场景 1:8 个线程并发 _execute,严格串行(max_active == 1)──
threads = [threading.Thread(target=lambda i=i: client._execute(f"q{i}")) for i in range(8)]
for t in threads:
    t.start()
for t in threads:
    t.join()
s1 = client.session.max_active == 1 and len(client.session.statements) == 8
report("并发 execute 严格串行(max_active=1)", s1)

# ── 场景 2:所有 NebulaClient 方法均走 _execute,裸 session.execute 仅存在于 _execute 内部 ──
src_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "database", "nebulagraph.py")
with open(src_path, encoding="utf-8") as f:
    src = f.read()
# _execute 之外的区域不得出现裸 session.execute(全局应仅剩 _execute 内部这一处)
outside = src.split("def _execute(")[1].split("def _atexit_close")[0]
s2 = (src.count("self.session.execute(") == 1
      and "self.session.execute(" in outside
      and "self._execute(" in src)
report("裸 session.execute 仅限 _execute 内部", s2)

# ── 场景 3:_execute 在 session 为 None 时明确报错(不静默)──
client2 = client.__class__.__new__(client.__class__)
client2._session_lock = threading.RLock()
client2.session = None
try:
    client2._execute("SHOW SPACES;")
    s3 = False
except RuntimeError:
    s3 = True
report("session=None 时 _execute 明确报错", s3)

# ── 场景 4:init_session_pool 有锁保护(源码断言)──
s4 = "with self._pool_lock:" in src and "_pool_lock = threading.Lock()" in src
report("init_session_pool 重建有锁保护", s4)

print("\n=== H1 Nebula 会话锁验证:", "全部通过 ✅" if ok else "存在失败 ❌")
sys.exit(0 if ok else 1)
