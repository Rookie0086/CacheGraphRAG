"""LLM and Embedding environment wrappers.

Supports OpenAI-compatible API, DeepSeek, and Ollama backends.
Embedding supports local (HuggingFace), API (OpenAI-compatible), and Ollama.
"""

import asyncio
import os
import time
from abc import ABC, abstractmethod
from typing import List, Optional, Union

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import numpy as np
import torch
import yaml
from openai import OpenAI, AsyncOpenAI
from transformers import AutoModel, AutoTokenizer

from src.utils import get_config, get_project_dir
from src.utils.base import print_text

EMBEDD_DIMS = {
    "BAAI/bge-large-en-v1.5": 1024,
    "BAAI/bge-base-en-v1.5": 768,
    "BAAI/bge-small-en-v1.5": 384,
    "text-embedding-ada-002": 1536,
    "nomic-embed-text": 768,
    "BAAI/bge-m3": 1024,
}


# ── Embedding Environments ──────────────────────────────────────

class EmbeddingEnv:
    """Local HuggingFace embedding model."""

    def __init__(self, model_name="BAAI/bge-small-en-v1.5", device=None, normalize=True, batch_size=-1):
        self.model_name = model_name
        self.normalize = normalize
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        with torch.no_grad():
            dummy = self.tokenizer("test", return_tensors="pt").to(self.device)
            dummy_out = self.model(**dummy)[0][:, 0]
        self.dim = dummy_out.shape[-1]
        print(f"EmbeddingEnv init -> model={self.model_name}, dim={self.dim}, device={self.device}")

    def _encode(self, texts):
        single = isinstance(texts, str)
        texts = [texts] if single else texts
        all_embeddings = []
        batch_size = self.batch_size if self.batch_size > 0 else len(texts)
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            encoded = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(self.device)
            with torch.no_grad():
                output = self.model(**encoded)
                embeddings = output[0][:, 0]  # CLS pooling
            if self.normalize:
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(embeddings.cpu())
        all_embeddings = torch.cat(all_embeddings, dim=0).numpy()
        return all_embeddings[0] if single else all_embeddings

    def get_embedding(self, text):
        return self._encode(text)

    def get_embeddings(self, texts):
        return self._encode(texts)

    async def get_embedding_async(self, text):
        return self._encode(text)

    async def get_embeddings_async(self, texts):
        return await asyncio.to_thread(self.get_embeddings, texts)

    def calculate_similarity(self, text1, text2):
        e1 = self.get_embedding(text1)
        e2 = self.get_embedding(text2)
        return round(float(np.dot(e1, e2)), 6)


class APIEmbeddingEnv:
    """OpenAI-compatible API embedding (e.g., SiliconFlow, OpenAI)."""

    def __init__(self, model_name="BAAI/bge-m3", api_key=None, base_url=None, normalize=True):
        self.model_name = model_name
        self.normalize = normalize
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        test_emb = self._call_api("test")
        self.dim = len(test_emb)
        print(f"APIEmbeddingEnv init -> model={self.model_name}, dim={self.dim}, base_url={base_url}")

    def _call_api(self, text_or_texts, _retries=5):
        delay = 1.0
        single = isinstance(text_or_texts, str)
        for attempt in range(_retries):
            try:
                resp = self.client.embeddings.create(model=self.model_name, input=text_or_texts)
                if single:
                    return resp.data[0].embedding
                return [d.embedding for d in resp.data]
            except Exception as e:
                if attempt >= _retries - 1:
                    raise
                time.sleep(delay)
                delay *= 2

    def get_embedding(self, text):
        emb = self._call_api(text)
        vec = np.array(emb, dtype=np.float32)
        if self.normalize:
            vec = vec / np.linalg.norm(vec)
        return vec

    async def get_embedding_async(self, text):
        return await asyncio.to_thread(self.get_embedding, text)

    def get_embeddings(self, texts):
        embs = self._call_api(texts)
        vecs = np.array(embs, dtype=np.float32)
        if self.normalize:
            vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs

    async def get_embeddings_async(self, texts):
        return await asyncio.to_thread(self.get_embeddings, texts)

    def calculate_similarity(self, text1, text2):
        e1 = self.get_embedding(text1)
        e2 = self.get_embedding(text2)
        return round(float(np.dot(e1, e2)), 6)


class OllamaEmbeddingEnv:
    """Ollama local API embedding."""

    def __init__(self, model_name="nomic-embed-text", base_url="http://localhost:11434", normalize=True):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.normalize = normalize
        test_emb = self._call_api("test")
        self.dim = len(test_emb)
        print(f"OllamaEmbeddingEnv init -> model={self.model_name}, dim={self.dim}, base_url={self.base_url}")

    def _call_api(self, text_or_texts, _retries=5):
        import requests
        single = isinstance(text_or_texts, str)
        texts = [text_or_texts] if single else text_or_texts
        embeddings = []
        for t in texts:
            for attempt in range(_retries):
                try:
                    resp = requests.post(
                        f"{self.base_url}/api/embeddings",
                        json={"model": self.model_name, "prompt": t},
                        timeout=30,
                    )
                    resp.raise_for_status()
                    embeddings.append(resp.json()["embedding"])
                    break
                except Exception:
                    if attempt >= _retries - 1:
                        raise
                    time.sleep(1)
        return embeddings[0] if single else embeddings

    def get_embedding(self, text):
        emb = self._call_api(text)
        vec = np.array(emb, dtype=np.float32)
        if self.normalize:
            vec = vec / np.linalg.norm(vec)
        return vec

    async def get_embedding_async(self, text):
        return await asyncio.to_thread(self.get_embedding, text)

    def get_embeddings(self, texts):
        embs = self._call_api(texts)
        vecs = np.array(embs, dtype=np.float32)
        if self.normalize:
            vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs

    async def get_embeddings_async(self, texts):
        return await asyncio.to_thread(self.get_embeddings, texts)

    def calculate_similarity(self, text1, text2):
        e1 = self.get_embedding(text1)
        e2 = self.get_embedding(text2)
        return round(float(np.dot(e1, e2)), 6)


# ── LLM Environments ────────────────────────────────────────────

class BaseLLMEnv(ABC):
    @abstractmethod
    def complete(self, prompt, verbose=False, return_info=False):
        pass


class OpenAIEnv(BaseLLMEnv):
    """OpenAI-compatible API LLM (supports token usage tracking)."""

    def __init__(self, model="gpt-4o-mini", api_key=None, base_url=None, temperature=0):
        self.model = model
        self.temperature = temperature
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        base_url = base_url or os.getenv("OPENAI_API_BASE")
        if not api_key or not base_url:
            raise ValueError("OpenAI api_key and base_url must be provided")
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.asyclient = AsyncOpenAI(
            base_url=base_url, api_key=api_key, timeout=300,
            default_headers={'RITS_API_KEY': os.environ["RITS_API_KEY"]} if os.environ.get("RITS_API_KEY") else None
        )
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        print(f"Initialized OpenAIEnv with model={self.model}, base_url={base_url}, temperature={self.temperature}")

    def _record_usage(self, response):
        usage = getattr(response, "usage", None)
        if usage is not None:
            self.total_prompt_tokens += (usage.prompt_tokens or 0)
            self.total_completion_tokens += (usage.completion_tokens or 0)

    async def async_complete(self, prompt, verbose=False, return_info=False):
        try:
            response = await self.asyclient.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                stream=False,
            )
            self._record_usage(response)
            if getattr(response, "choices", None):
                message = response.choices[0].message
                if message and message.content:
                    return message.content.strip()
            return None
        except Exception as e:
            from src.utils.logger import get_logger
            log = get_logger()
            if log:
                log.warn(f"LLM API async call failed: {e}")
            return None

    def complete(self, prompt, verbose=False, return_info=False):
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                stream=False,
            )
            self._record_usage(response)
            if getattr(response, "choices", None):
                message = response.choices[0].message
                if message and message.content:
                    return message.content.strip()
            raise RuntimeError("API response missing valid content")
        except Exception as e:
            from src.utils.logger import get_logger
            log = get_logger()
            if log:
                log.warn(f"LLM API sync call failed: {e}")
            return None


class DeepSeekEnv(BaseLLMEnv):
    """DeepSeek API LLM."""

    def __init__(self, model="deepseek-chat", api_key=None, base_url=None, temperature=0):
        self.model = model
        self.temperature = temperature
        api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        base_url = base_url or os.getenv("DEEPSEEK_API_BASE")
        if not api_key or not base_url:
            raise ValueError("DeepSeek api_key and base_url must be provided")
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def complete(self, prompt, verbose=False, return_info=False):
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                stream=False,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error in LLM API call: {e}")
            return None


class OllamaEnv(BaseLLMEnv):
    """Ollama local LLM."""

    def __init__(self, model="llama3.1:8b", timeout=300, temperature=0,
                 base_url="http://localhost:11434", max_tokens=500):
        if base_url and base_url.endswith("/v1"):
            base_url = base_url[:-3]
        from llama_index.llms.ollama import Ollama
        self.llm = Ollama(
            model=model, request_timeout=timeout, base_url=base_url,
            temperature=temperature,
            additional_kwargs={"num_predict": max_tokens},
        )

    async def async_complete(self, prompt, verbose=False, return_info=False):
        return await asyncio.to_thread(self.complete, prompt, verbose=verbose, return_info=return_info)

    def complete(self, prompt, verbose=False, return_info=False):
        output = self.llm.complete(prompt)
        return output.text


# ── Unified LLM Environment ────────────────────────────────────

class LLMEnv:
    """Unified LLM+Embedding environment.

    Args:
        backend: "openai" | "deepseek" | "ollama"
        model: Model name for the chosen backend
        embed_model_name: Embedding model name
        embed_backend: "local" | "api" | "ollama"
    """

    _MAX_RETRIES = 5
    _BACKOFF_SECONDS = 1

    def __init__(
        self,
        backend: Literal["openai", "deepseek", "ollama"] = "ollama",
        model="llama3.1:8b",
        embed_model_name="BAAI/bge-small-en-v1.5",
        embed_backend: Literal["local", "api", "ollama"] = "local",
        embed_api_key=None,
        embed_base_url=None,
        api_key=None,
        base_url=None,
        timeout=300,
        max_tokens=2048,
        temperature=0,
    ):
        self.backend = backend
        self.model_name = model
        self.max_tokens = max_tokens

        # Initialize embedding model
        if embed_backend == "api":
            self.embed_model = APIEmbeddingEnv(
                model_name=embed_model_name, api_key=embed_api_key, base_url=embed_base_url)
        elif embed_backend == "ollama":
            self.embed_model = OllamaEmbeddingEnv(
                model_name=embed_model_name, base_url=embed_base_url or "http://localhost:11434")
        else:
            self.embed_model = EmbeddingEnv(model_name=embed_model_name)

        # Initialize LLM
        if backend == "ollama":
            self.llm = OllamaEnv(
                model=model, timeout=timeout, base_url=base_url,
                temperature=temperature, max_tokens=max_tokens)
        elif backend == "openai":
            self.llm = OpenAIEnv(
                model=model, api_key=api_key, base_url=base_url, temperature=temperature)
        elif backend == "deepseek":
            self.llm = DeepSeekEnv(
                model=model, api_key=api_key, base_url=base_url, temperature=temperature)
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    def complete(self, prompt, verbose=False, return_info=False):
        delay = self._BACKOFF_SECONDS
        for attempt in range(self._MAX_RETRIES):
            try:
                return self.llm.complete(prompt, verbose=verbose, return_info=return_info)
            except Exception as exc:
                if attempt >= self._MAX_RETRIES - 1:
                    raise
                time.sleep(delay)
                delay *= 2
        return ""

    async def async_complete(self, prompt, verbose=False, return_info=False):
        if not hasattr(self.llm, "async_complete"):
            return await asyncio.to_thread(self.complete, prompt, verbose, return_info)
        delay = self._BACKOFF_SECONDS
        for attempt in range(self._MAX_RETRIES):
            try:
                return await self.llm.async_complete(prompt, verbose=verbose, return_info=return_info)
            except Exception as exc:
                print_text(f"Error in async LLM call (attempt {attempt+1}/{self._MAX_RETRIES}): {exc}", color="red")
                if attempt >= self._MAX_RETRIES - 1:
                    raise
                await asyncio.sleep(delay)
                delay *= 2
        return ""

    @property
    def total_prompt_tokens(self):
        if hasattr(self.llm, "total_prompt_tokens"):
            return self.llm.total_prompt_tokens
        return 0

    @property
    def total_completion_tokens(self):
        if hasattr(self.llm, "total_completion_tokens"):
            return self.llm.total_completion_tokens
        return 0

    def hello_world(self):
        response = self.complete("who are you?")
        print_text("Q: who are you?\n", color="red")
        print_text(f"A: {response}\n", color="green")
