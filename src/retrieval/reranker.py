import os
import threading
import requests
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class APIReranker:
    """API-based reranker (compatible with SiliconFlow / Jina / Cohere etc.)."""

    def __init__(self, model_name: str, api_key: str, base_url: str):
        self.model_name = model_name
        # 允许环境变量补位(实验框架会 redact config 里的 api_key):
        # 运行时设置 CACHEGRAPH_RERANK_API_KEY 提供真实 key,避免 401 导致检索结果被丢弃。
        self.api_key = (api_key or os.getenv("CACHEGRAPH_RERANK_API_KEY")
                        or os.getenv("CACHEGRAPH_MODEL_API_KEY") or os.getenv("OPENAI_API_KEY"))
        self.base_url = base_url.rstrip("/")
        # M2(2026-08-15):rerank 并发门控,防止并发查询无上限打爆 rerank API。
        from src.utils import get_config
        _ret_cfg = get_config().get("retrieval", {})
        self._rerank_gate = threading.BoundedSemaphore(
            max(1, int(_ret_cfg.get("rerank_concurrency", 2))))

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _payload(self, query: str, documents: list[str], top_n: int = None) -> dict:
        payload = {"model": self.model_name, "query": query, "documents": documents}
        if top_n is not None:
            payload["top_n"] = top_n
        return payload

    def score(self, query: str, passage: str) -> float:
        """Score a single pair, compatible with legacy interface."""
        results = self.rerank(query, [passage], top_n=1)
        return results[0]["score"] if results else 0.0

    def rerank(self, query: str, documents: list[str], top_n: int = None) -> list[dict]:
        """Batch rerank, returns [{index, score, text}, ...] sorted by score descending."""
        if not query or not documents:
            return []
        url = f"{self.base_url}/rerank"
        # M2:rerank 并发门控(同步 HTTP 调用)
        with self._rerank_gate:
            try:
                resp = requests.post(
                    url, headers=self._headers(),
                    json=self._payload(query, documents, top_n),
                    timeout=60,
                )
                resp.raise_for_status()
                results = resp.json().get("results", [])
            except requests.RequestException as e:
                print(f"[APIReranker] Request failed: {e}")
                return []
        # 响应归一化:兼容不同服务端字段(SiliconFlow: score/text;oMLX: relevance_score/document.text)
        norm = []
        for r in results:
            if not isinstance(r, dict):
                continue
            score = r.get("score", r.get("relevance_score", 0.0))
            text = r.get("text")
            if text is None and isinstance(r.get("document"), dict):
                text = r["document"].get("text", "")
            norm.append({"index": r.get("index"), "score": score, "text": text})
        return norm


class LocalReranker:
    def __init__(self, model_path: str, device: str = None, max_length: int = 512):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Local reranker model not found: {model_path}")
        self.model_path = model_path
        # 默认设备:cuda > mps(Mac MLX 可用时) > cpu
        self.device = device or (
            "cuda" if torch.cuda.is_available()
            else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
                  else "cpu"))
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

    def score(self, query: str, passage: str) -> float:
        if not query or not passage:
            return 0.0
        inputs = self.tokenizer(
            query,
            passage,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits.squeeze()
        score = torch.sigmoid(logits).item() if logits.numel() == 1 else torch.sigmoid(logits[0]).item()
        return float(max(0.0, min(1.0, score)))

    def rerank(self, query: str, documents: list[str], top_n: int = None) -> list[dict]:
        """Batch rerank for unified interface."""
        scores = [self.score(query, d) for d in documents]
        indexed = sorted(
            [{"index": i, "score": s, "text": documents[i]} for i, s in enumerate(scores)],
            key=lambda x: x["score"], reverse=True,
        )
        if top_n is not None:
            indexed = indexed[:top_n]
        return indexed
