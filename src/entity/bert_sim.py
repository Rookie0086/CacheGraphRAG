"""BERTScore semantic similarity using sentence-transformers.

Model is lazily loaded on first call to avoid import-time overhead.
"""

import os
from typing import Optional

_model = None


def _get_model():
    """Lazily load the sentence-transformer model."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _model


def _normalize_texts(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        normalized = []
        for item in value:
            if isinstance(item, (list, tuple)):
                parts = [str(v) for v in item if str(v)]
                parts = sorted(parts)
                normalized.append(" ".join(parts))
            else:
                normalized.append(str(item))
        normalized = [v for v in normalized if v]
        return sorted(normalized)
    return [str(value)]


def bert(response, ground_truth):
    """
    @param response: the response from LLM
    @param ground_truth: the ground truth of the question
    @return: the cosine similarity
    """
    from sentence_transformers import util
    import torch

    response_texts = _normalize_texts(response)
    ground_truth_texts = _normalize_texts(ground_truth)
    if not response_texts or not ground_truth_texts:
        return 0.0

    model = _get_model()
    query_embedding = model.encode(response_texts, convert_to_tensor=True)

    max_score = 0.0
    for gt in ground_truth_texts:
        text_embedding = model.encode([gt], convert_to_tensor=True)
        cosine_score = util.pytorch_cos_sim(query_embedding, text_embedding)
        score = float(cosine_score.max().item())
        if score > max_score:
            max_score = score

    return max_score
