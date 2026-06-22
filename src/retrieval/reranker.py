import os
import requests
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class APIReranker:
    """API-based reranker (compatible with SiliconFlow / Jina / Cohere etc.)."""

    def __init__(self, model_name: str, api_key: str, base_url: str):
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

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
        try:
            resp = requests.post(
                url, headers=self._headers(),
                json=self._payload(query, documents, top_n),
                timeout=60,
            )
            resp.raise_for_status()
            return resp.json().get("results", [])
        except requests.RequestException as e:
            print(f"[APIReranker] Request failed: {e}")
            return []


class LocalReranker:
    def __init__(self, model_path: str, device: str = None, max_length: int = 512):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Local reranker model not found: {model_path}")
        self.model_path = model_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
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
