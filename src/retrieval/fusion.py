"""Fusion algorithm base class and implementations: RRF / Weighted fusion."""

from typing import List, Dict


class BaseFusion:
    """Fusion base class: subclasses override fuse() to implement different fusion strategies."""

    def __init__(self, retriever):
        self.retriever = retriever

    def fuse(self, query: str, vector_hits: List[Dict],
             graph_ranked_chunks: List[tuple], chunk_entity_coverage: Dict,
             top_rerank: int, answer_topk: int) -> List[str]:
        """Returns final_chunks. Subclasses override."""
        raise NotImplementedError

    def handle_promotion(self, chunk_entity_coverage, entity_chunks, track_promotion):
        """Promotion hook. Subclasses may optionally override."""
        pass

    def _rerank(self, query: str, candidates: List[str], top_n: int,
                 skip_set: set = None) -> List[str]:
        """Cross-encoder reranking. Chunks in skip_set are not reranked, original order preserved."""
        r = self.retriever
        if not candidates:
            return []
        skip_set = skip_set or set()

        # Separate: those needing reranking vs those kept
        to_rerank = [c for c in candidates if c not in skip_set]
        kept = [c for c in candidates if c in skip_set]

        if not to_rerank:
            return kept[:top_n]

        texts = r._get_chunk_text(to_rerank)
        if r.reranker and hasattr(r.reranker, "rerank"):
            reranked = r.reranker.rerank(query, texts, top_n=max(1, top_n - len(kept)))
            reranked_ids = [to_rerank[item["index"]] for item in reranked]
        else:
            scored = [(c, r._rerank_score(query, t)) for c, t in zip(to_rerank, texts)]
            scored.sort(key=lambda x: x[1], reverse=True)
            reranked_ids = [c for c, _ in scored[:max(1, top_n - len(kept))]]

        return reranked_ids + kept[:top_n - len(reranked_ids)]


class RRFFusion(BaseFusion):
    """RRF fusion: only considers rankings, equal-weight fusion (current default behavior)."""

    def __init__(self, retriever, rrf_k: int = 60):
        super().__init__(retriever)
        self.rrf_k = rrf_k

    def fuse(self, query, vector_hits, graph_ranked_chunks, chunk_entity_coverage,
             top_rerank, answer_topk):
        rrf_scores = {}
        for rank, hit in enumerate(vector_hits):
            rrf_scores[hit["id"]] = rrf_scores.get(hit["id"], 0.0) + 1.0 / (self.rrf_k + rank + 1)
        for rank, (cid, _) in enumerate(graph_ranked_chunks):
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (self.rrf_k + rank + 1)

        pool_size = 20
        fused = [cid for cid, _ in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:pool_size]]
        return self._rerank(query, fused, answer_topk)

    def handle_promotion(self, chunk_entity_coverage, entity_chunks, track_promotion):
        """Traditional access_chunk promotion."""
        if not track_promotion:
            return
        g = self.retriever.memory_graph
        for cid in chunk_entity_coverage:
            triggered, data = g.access_chunk(cid)
            if triggered:
                g.write_to_persistent_graph([data])


class WeightedFusion(BaseFusion):
    """Weighted fusion: preserves original scores + overlap bonus + convergence bonus."""

    def __init__(self, retriever, dpr_weight: float = 0.5,
                 overlap_boost: float = 3.0,
                 convergence_boost: float = 1.3, convergence_threshold: int = 2):
        super().__init__(retriever)
        self.dpr_weight = dpr_weight
        self.overlap_boost = overlap_boost
        self.convergence_boost = convergence_boost
        self.convergence_threshold = convergence_threshold

    def fuse(self, query, vector_hits, graph_ranked_chunks, chunk_entity_coverage,
             top_rerank, answer_topk):
        # Normalize DPR scores
        dpr_scores = {}
        dpr_raw = [h["score"] for h in vector_hits if h["score"] > 0]
        dpr_max = max(dpr_raw) if dpr_raw else 1.0
        dpr_min = min(dpr_raw) if dpr_raw else 0.0
        for h in vector_hits:
            norm = (h["score"] - dpr_min) / (dpr_max - dpr_min) if dpr_max > dpr_min else 0.5
            dpr_scores[h["id"]] = norm

        # Normalize graph scores
        graph_scores = {}
        graph_raw = [s for _, s in graph_ranked_chunks if s > 0]
        g_max = max(graph_raw) if graph_raw else 1.0
        g_min = min(graph_raw) if graph_raw else 0.0
        for cid, score in graph_ranked_chunks:
            norm = (score - g_min) / (g_max - g_min) if g_max > g_min else 0.5
            graph_scores[cid] = norm

        # Overlap set: chunks in DPR ∩ Graph
        dpr_set = set(dpr_scores.keys())
        graph_set = set(graph_scores.keys())
        overlap = dpr_set & graph_set

        # Independent path fusion: graph-only doesn't get zero score (min_dpr as baseline)
        min_dpr = min(dpr_scores.values()) if dpr_scores else 0.0
        fused = {}
        for cid in dpr_set | graph_set:
            dpr = dpr_scores.get(cid, min_dpr)  # graph-only gets the lowest DPR score
            g = graph_scores.get(cid, 0.0)
            if cid in overlap:
                # Overlap chunk: take max of both + boost
                score = max(dpr, g) * self.overlap_boost
            else:
                score = self.dpr_weight * dpr + (1 - self.dpr_weight) * g
            n_entities = len(chunk_entity_coverage.get(cid, set()))
            if n_entities >= self.convergence_threshold:
                score *= self.convergence_boost ** (n_entities - self.convergence_threshold + 1)
            fused[cid] = score

        pool_size = top_rerank * 2
        ranked = [cid for cid, _ in sorted(fused.items(), key=lambda x: x[1], reverse=True)[:pool_size]]
        return self._rerank(query, ranked, answer_topk)

    def handle_promotion(self, chunk_entity_coverage, entity_chunks, track_promotion):
        """Weighted convergence promotion."""
        if not track_promotion or not hasattr(self, '_entity_weights'):
            return
        ew = self._entity_weights
        if not ew:
            return
        sorted_weights = sorted(ew.values())
        n = len(sorted_weights)
        if n == 0: return
        median_weight = sorted_weights[n // 2]
        g = self.retriever.memory_graph
        for cid, entity_set in chunk_entity_coverage.items():
            weighted_coverage = sum(ew.get(uid, 0) for uid in entity_set)
            if weighted_coverage >= median_weight:
                subgraph_data = g._extract_subgraph_by_chunk(cid)
                if subgraph_data.get("nodes"):
                    g.write_to_persistent_graph([subgraph_data])
                    g.promoted_chunks.add(cid)


class DualFusion(BaseFusion):
    """Overlap priority + interleaved fairness + RRF fill-in fusion.

    1. Overlap chunks (DPR ∩ Graph) → directly selected (higher rank)
    2. Alternating DPR / Graph take one each → ensure fair contribution from both paths
    3. RRF fill-in when answer_topk not reached"""

    def __init__(self, retriever, rrf_k: int = 60):
        super().__init__(retriever)
        self.rrf_k = rrf_k

    def fuse(self, query, vector_hits, graph_ranked_chunks, chunk_entity_coverage,
             top_rerank, answer_topk):
        # DPR path rerank
        dpr_ids = [h["id"] for h in vector_hits]
        dpr_reranked = self._rerank(query, dpr_ids, len(dpr_ids))

        dpr_set = set(dpr_reranked)
        graph_list = [cid for cid, _ in graph_ranked_chunks]
        graph_set = set(graph_list)

        # 1. Overlap priority
        overlap = [cid for cid in graph_list if cid in dpr_set]
        used = set(overlap)

        # 2. Alternating selection
        dpr_only = [cid for cid in dpr_reranked if cid not in used]
        graph_only = [cid for cid in graph_list if cid not in used]

        interleaved = []
        di, gi = 0, 0
        while di < len(dpr_only) or gi < len(graph_only):
            if di < len(dpr_only):
                interleaved.append(dpr_only[di]); di += 1
            if gi < len(graph_only):
                interleaved.append(graph_only[gi]); gi += 1

        # 3. Merge: overlap + interleaved
        result = overlap + interleaved

        # 4. RRF fill when insufficient
        if len(result) < answer_topk:
            rrf_scores = {}
            for rank, cid in enumerate(dpr_reranked):
                rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (self.rrf_k + rank + 1)
            for rank, cid in enumerate(graph_list):
                rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (self.rrf_k + rank + 1)
            extra = [cid for cid, _ in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
                     if cid not in set(result)]
            result.extend(extra)

        return result[:answer_topk]
