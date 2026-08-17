# 验证 token 统计抽到 UsageRecordingMixin 层:openai/deepseek 后端都能累计
# prompt/completion tokens(原实现仅 OpenAIEnv 记账,deepseek 等后端恒为 0,
# 影响 R3-W4 的 token/成本上报)。纯内存 mock,不连接任何外部服务/API。
import asyncio
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llm.env import DeepSeekEnv, LLMEnv, OpenAIEnv, UsageRecordingMixin


class _Usage:
    def __init__(self, prompt, completion):
        self.prompt_tokens = prompt
        self.completion_tokens = completion


class _Resp:
    """对象形式 usage(openai SDK 风格,无 model_dump)。"""

    def __init__(self, prompt, completion, content="ok"):
        self.usage = _Usage(prompt, completion)
        self.choices = [type("C", (), {"message": type("M", (), {"content": content})})()]


class _RespDictUsage:
    """dict 形式 usage(部分代理/本地服务直接返回 JSON)。"""

    def __init__(self, prompt, completion):
        self.usage = {"prompt_tokens": prompt, "completion_tokens": completion}


class _PydanticUsage:
    """带 model_dump 的 pydantic 风格 usage(真实 bound method)。"""

    def __init__(self, prompt, completion):
        self.prompt_tokens = prompt
        self.completion_tokens = completion

    def model_dump(self):
        return {"prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens}


class _RespPydantic:
    def __init__(self, prompt, completion):
        self.usage = _PydanticUsage(prompt, completion)


ok = True

# ── 场景 1:mixin 对象形式 usage 累计 ──
m = UsageRecordingMixin(); m._init_usage()
m._record_usage(_Resp(100, 25))
m._record_usage(_Resp(50, 10))
s1 = (m.total_prompt_tokens == 150 and m.total_completion_tokens == 35)
ok &= s1
print(f"  [对象形式 usage] prompt={m.total_prompt_tokens} completion={m.total_completion_tokens} "
      f"(期望 150/35) {'PASS' if s1 else 'FAIL'}")

# ── 场景 2:dict 形式 usage 累计 ──
m2 = UsageRecordingMixin(); m2._init_usage()
m2._record_usage(_RespDictUsage(10, 3))
m2._record_usage(_RespDictUsage(7, 2))
s2 = (m2.total_prompt_tokens == 17 and m2.total_completion_tokens == 5)
ok &= s2
print(f"  [dict 形式 usage] prompt={m2.total_prompt_tokens} completion={m2.total_completion_tokens} "
      f"(期望 17/5) {'PASS' if s2 else 'FAIL'}")

# ── 场景 3:pydantic 风格 usage(model_dump)──
m3 = UsageRecordingMixin(); m3._init_usage()
m3._record_usage(_RespPydantic(200, 40))
s3 = (m3.total_prompt_tokens == 200 and m3.total_completion_tokens == 40)
ok &= s3
print(f"  [pydantic usage] prompt={m3.total_prompt_tokens} completion={m3.total_completion_tokens} "
      f"(期望 200/40) {'PASS' if s3 else 'FAIL'}")

# ── 场景 4:OpenAIEnv 实例记账(不发起网络)──
o = OpenAIEnv(model="gpt-4o-mini", api_key="sk-dummy", base_url="https://api.openai.com/v1")
o._record_usage(_Resp(300, 60))
s4 = (o.total_prompt_tokens == 300 and o.total_completion_tokens == 60)
ok &= s4
print(f"  [OpenAIEnv] prompt={o.total_prompt_tokens} completion={o.total_completion_tokens} "
      f"(期望 300/60) {'PASS' if s4 else 'FAIL'}")

# ── 场景 5:DeepSeekEnv.complete 成功返回后记账(mock client)──
d = DeepSeekEnv(model="deepseek-chat", api_key="sk-dummy",
                base_url="https://api.deepseek.com")
d.client.chat.completions.create = lambda **kw: _Resp(500, 80, content="  answer  ")
out = d.complete("q")
s5 = (out == "answer" and d.total_prompt_tokens == 500 and d.total_completion_tokens == 80)
ok &= s5
print(f"  [DeepSeekEnv.complete] out={out!r} prompt={d.total_prompt_tokens} "
      f"completion={d.total_completion_tokens} (期望 'answer'/500/80) {'PASS' if s5 else 'FAIL'}")

# ── 场景 6:DeepSeekEnv.async_complete 记账(mock async client)──
async def _case6():
    d6 = DeepSeekEnv(model="deepseek-chat", api_key="sk-dummy",
                     base_url="https://api.deepseek.com")
    async def fake_create(**kw):
        return _Resp(700, 90, content="async answer")
    d6.asyclient.chat.completions.create = fake_create
    out = await d6.async_complete("q")
    return out, d6.total_prompt_tokens, d6.total_completion_tokens

out6, p6, c6 = asyncio.run(_case6())
s6 = (out6 == "async answer" and p6 == 700 and c6 == 90)
ok &= s6
print(f"  [DeepSeekEnv.async_complete] out={out6!r} prompt={p6} completion={c6} "
      f"(期望 'async answer'/700/90) {'PASS' if s6 else 'FAIL'}")

# ── 场景 7:LLMEnv 门面属性委托(不构造 embed 模型,跳过 __init__)──
f = LLMEnv.__new__(LLMEnv)
f.llm = type("B", (), {"total_prompt_tokens": 1234, "total_completion_tokens": 567})()
s7 = (f.total_prompt_tokens == 1234 and f.total_completion_tokens == 567)
ok &= s7
print(f"  [门面委托] prompt={f.total_prompt_tokens} completion={f.total_completion_tokens} "
      f"(期望 1234/567) {'PASS' if s7 else 'FAIL'}")

# ── 场景 8:无 usage 的后端返回 0(不崩)──
f8 = LLMEnv.__new__(LLMEnv)
f8.llm = type("B", (), {})()
s8 = (f8.total_prompt_tokens == 0 and f8.total_completion_tokens == 0)
ok &= s8
print(f"  [无 usage 后端] prompt={f8.total_prompt_tokens} completion={f8.total_completion_tokens} "
      f"(期望 0/0) {'PASS' if s8 else 'FAIL'}")

print("\n=== token 统计门面化验证:", "全部通过 ✅" if ok else "存在失败 ❌")
sys.exit(0 if ok else 1)
