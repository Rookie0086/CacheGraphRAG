"""LLM response cache: uses input text hash as key to avoid duplicate API calls on restart."""

import hashlib
import json
import os
from typing import Optional


class LLMCache:
    """File-level LLM cache, keyed by MD5 of prompt text.

    Cache file format is JSONL, each line: {"key": "md5", "response": "..."}.
    """

    def __init__(self, cache_path: str = "output/llm_cache/llm_cache.jsonl"):
        self._path = cache_path
        os.makedirs(os.path.dirname(self._path), exist_ok=True)
        self._cache: dict[str, str] = {}
        self._load()

    def _hash(self, text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def _load(self):
        if not os.path.exists(self._path):
            return
        with open(self._path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    self._cache[record["key"]] = record["response"]
                except (json.JSONDecodeError, KeyError):
                    continue

    def get(self, prompt: str) -> Optional[str]:
        """Return cached response if hit, otherwise None."""
        key = self._hash(prompt)
        return self._cache.get(key)

    def put(self, prompt: str, response: str):
        """Write to cache."""
        key = self._hash(prompt)
        if key in self._cache:
            return  # Already exists
        self._cache[key] = response
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"key": key, "response": response}, ensure_ascii=False) + "\n")

    @property
    def size(self) -> int:
        return len(self._cache)
