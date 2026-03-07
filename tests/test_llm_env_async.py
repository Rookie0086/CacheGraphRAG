import unittest
from types import SimpleNamespace

from utils.llm_env import LLMEnv, OpenAIEnv


class FakeChatCompletions:
    def __init__(self, response):
        self._response = response

    async def create(self, **kwargs):
        return self._response


class FakeAsyncClient:
    def __init__(self, response):
        self.chat = SimpleNamespace(completions=FakeChatCompletions(response))


class OpenAIEnvAsyncCompleteTests(unittest.IsolatedAsyncioTestCase):
    async def test_async_complete_returns_content(self):
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=" ok "))]
        )
        env = OpenAIEnv(model="gpt-4o-mini", api_key="test", base_url="http://x")
        env.asyclient = FakeAsyncClient(response)

        result = await env.async_complete("ping")

        self.assertEqual(result, "ok")

    async def test_async_complete_returns_none_on_empty_choices(self):
        response = SimpleNamespace(choices=[])
        env = OpenAIEnv(model="gpt-4o-mini", api_key="test", base_url="http://x")
        env.asyclient = FakeAsyncClient(response)

        result = await env.async_complete("ping")

        self.assertIsNone(result)


class LLMEnvAsyncCompleteTests(unittest.IsolatedAsyncioTestCase):
    async def test_async_complete_raises_when_backend_missing(self):
        env = LLMEnv.__new__(LLMEnv)
        env.backend = "mock"
        env.llm = object()

        with self.assertRaises(NotImplementedError):
            await env.async_complete("ping")


if __name__ == "__main__":
    unittest.main()