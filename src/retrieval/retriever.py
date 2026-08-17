"""Retrievers: BaseRetriever + HybridRetriever + FactRetriever"""

import json
import re
import time
import threading
import concurrent.futures
import numpy as np
from typing import List, Dict
import networkx as nx
from src.utils.prompts import prompt_extract_entities_str
from src.utils import get_config
from database.milvus import MilvusDB


class BaseRetriever:
    """Base retriever class: shared utilities + hybrid_retrieve template method."""

    def __init__(self, vector_store=None, memory_graph=None, llm=None,
                 reranker=None, chunk_registry=None, fusion=None,
                 mode="hybrid"):
        self.vector_store = vector_store
        self.memory_graph = memory_graph
        self.persistent_graph = memory_graph.persistent_graph if memory_graph else None
        self.llm = llm
        self.reranker = reranker
        self.chunk_registry = chunk_registry or {}
        self.fusion = fusion
        self.mode = mode  # "hybrid" or "graph_only"
        self.last_latency = {}  # 最近一次 hybrid_retrieve 的分阶段耗时(秒)
        self.last_rehydrate = {"attempted": 0, "succeeded": 0, "chunk_ids": []}
        # 跳数衰减 γ 与最大跳数 max_hops(可配置,默认 γ=0.5 / max_hops=3)
        _cfg_all = get_config()
        _ret_cfg = _cfg_all.get("retrieval", {})
        _hyper_cfg = _cfg_all.get("hyperparameters", {})
        self.gamma = float(_ret_cfg.get("gamma", 0.5))
        self.max_hops = max(1, int(_ret_cfg.get("max_hops", 3)))
        # 束搜索宽度 B(论文 IV-E-1:图检索器 Decay-Guided Beam Search):
        # 每跳扩展后保留 top-B 高分候选路径。0 = 不裁剪(全量扩展)。
        self.beam_width = max(0, int(_ret_cfg.get("beam_width", _hyper_cfg.get("B", 4))))
        # Experiment switches. Defaults preserve the paper implementation.
        self.enable_l1 = bool(_ret_cfg.get("enable_l1", True))
        self.enable_l2 = bool(_ret_cfg.get("enable_l2", True))
        self.enable_rehydrate = bool(_ret_cfg.get("enable_rehydrate", True))
        self.fail_on_l2_error = bool(_ret_cfg.get("fail_on_l2_error", False))
        # M2(2026-08-15):查询期嵌入并发门控,防止并发查询无上限打爆 embedding API。
        self._embed_gate = threading.BoundedSemaphore(
            max(1, int(_ret_cfg.get("embedding_concurrency", 4))))
        # M2:实体级检索并行度(L1/L2 各实体并发执行,降查询时延)。
        self.graph_entity_parallel = max(
            1, int(_ret_cfg.get("graph_entity_parallel", 4)))

    # ── Shared Utilities ──────────────────────────────────────────

    def _embed_text(self, text: str) -> np.ndarray:
        if not self.llm:
            raise ValueError("LLM env is required for embeddings.")
        # M2:嵌入并发门控(单次同步 HTTP,门控防并发查询无上限打爆 API)
        with self._embed_gate:
            return np.array(self.llm.embed_model.get_embedding(text), dtype=float)

    def _cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    def _get_chunk_text(self, chunk_ids: List[str]) -> List[str]:
        texts = []
        for cid in chunk_ids:
            info = self.vector_store.get_chunk_text(cid)
            texts.append(info.get("text", "") if info else "")
        return texts

    def _rerank_score(self, query: str, passage: str) -> float:
        if not self.reranker or not passage:
            return 0.0
        if hasattr(self.reranker, "score"):
            return float(self.reranker.score(query, passage))
        return 0.0

    def _rehydrate_vector_hits(self, hits: List[Dict]) -> Dict:
        stats = {"attempted": 0, "succeeded": 0, "chunk_ids": []}
        if not self.memory_graph or not getattr(self, "enable_rehydrate", True):
            return stats
        for hit in hits:
            chunk_id = hit.get("id")
            if not chunk_id or not str(chunk_id).startswith("chunk_"):
                continue
            if self.memory_graph.is_chunk_in_l1(chunk_id):
                self.memory_graph.touch_chunk(chunk_id)
                continue
            # 即使 chunk 已在 L2，也要在 L1 淘汰后从 graph_meta 恢复拓扑。
            stats["attempted"] += 1
            if self.memory_graph.rehydrate_chunk_from_milvus(chunk_id):
                stats["succeeded"] += 1
                stats["chunk_ids"].append(chunk_id)
        self.last_rehydrate = stats
        return stats

    # ── DPR ─────────────────────────────────────────────

    def retrieve_from_embedding(self, query_emb: np.ndarray, topk: int = 5):
        try:
            search_params = {"metric_type": self.vector_store.metric, "params": {"nprobe": 10}}
            output_fields = ["chunk_id"] if self.vector_store._has_field("chunk_id") else []
            result = self.vector_store.search(
                query_emb.tolist(), search_params, topk, output_fields=output_fields)
        except Exception:
            result = []
        hits = []
        for hits_group in result:
            for hit in hits_group:
                chunk_id = None
                if output_fields:
                    chunk_id = hit.entity.get("chunk_id")
                hits.append({"id": str(chunk_id) if chunk_id else str(hit.pk),
                             "score": float(hit.distance)})
        return {"embedding_hits": hits}

    # ── Template Method ─────────────────────────────────────────

    def hybrid_retrieve(self, query: str, topk: int = 10, top_entities: int = 5,
                        top_chunks: int = 15, top_rerank: int = 15, answer_topk: int = 6,
                        track_promotion: bool = True):
        """Template method: subclasses override _extract_entities / _handle_promotion."""
        # 分阶段计时:embed/dpr/entity/graph/aggregate/fusion,供查询时延分析(R2-6.2/R4-W3)
        _t = time.time
        _latency = {}
        self.last_rehydrate = {"attempted": 0, "succeeded": 0, "chunk_ids": []}

        _t0 = _t()
        query_emb = self._embed_text(query)
        _latency["embed"] = _t() - _t0

        # 1. DPR (skipped in graph_only mode)
        vector_hits = []
        rehydrate_stats = {"attempted": 0, "succeeded": 0, "chunk_ids": []}
        if self.mode != "graph_only":
            _t1 = _t()
            embedding_res = self.retrieve_from_embedding(query_emb, topk=topk)
            vector_hits = embedding_res.get("embedding_hits", [])
            rehydrate_stats = self._rehydrate_vector_hits(vector_hits)
            _latency["dpr"] = _t() - _t1
        else:
            _latency["dpr"] = 0.0

        # 2. Entity extraction + weights (implemented by subclasses)
        _t2 = _t()
        extracted_entities, query_relations, entity_weights, entity_chunks = \
            self._extract_entities(query_emb, query, top_entities)
        _latency["entity"] = _t() - _t2

        # 3. Graph retrieval (L1 + L2)
        _t3 = _t()
        memory_res, persistent_res, chunk_entity_coverage = \
            self._graph_retrieve(query_emb, extracted_entities, query_relations,
                                 entity_weights, top_entities, top_chunks, track_promotion)
        _latency["graph"] = _t() - _t3

        # 4. Score aggregation
        _t4 = _t()
        graph_ranked_chunks = self._aggregate_graph_scores(
            memory_res, persistent_res, top_chunks, query)
        _latency["aggregate"] = _t() - _t4

        # 5. Fusion / graph-only mode
        _t5 = _t()
        if self.mode == "graph_only":
            # Graph-only retrieval: directly use graph-ranked chunks, skip DPR fusion
            final_chunks = [cid for cid, _ in graph_ranked_chunks[:answer_topk]]
        elif self.fusion:
            final_chunks = self.fusion.fuse(query, vector_hits, graph_ranked_chunks,
                                            chunk_entity_coverage, top_rerank, answer_topk)
            self.fusion.handle_promotion(
                chunk_entity_coverage, entity_chunks, track_promotion,
                entity_weights=entity_weights)
        else:
            final_chunks = []
        _latency["fusion"] = _t() - _t5
        _latency["total"] = _t() - _t0
        self.last_latency = _latency

        # Track sources
        dpr_cids = {h["id"] for h in vector_hits}
        graph_cids = set()
        for m in memory_res + persistent_res:
            graph_cids.update(m.get("chunk_scores", {}).keys())
        final_cids = set(final_chunks)
        graph_in_final = final_cids & graph_cids
        dpr_in_final = final_cids & dpr_cids

        return {
            "chunks": final_chunks,
            "embedding": {"embedding_hits": vector_hits},
            "memory": memory_res,
            "persistent": persistent_res,
            "chunk_entity_coverage": chunk_entity_coverage,
            "stats": {
                "dpr_total": len(dpr_cids),
                "graph_total": len(graph_cids),
                "graph_in_final": len(graph_in_final),
                "dpr_in_final": len(dpr_in_final),
                "overlap": len(graph_cids & dpr_cids),
                "n_entities": len(extracted_entities),
                "n_chunks_covered": len(chunk_entity_coverage),
                "rehydrate_attempted": rehydrate_stats["attempted"],
                "rehydrate_succeeded": rehydrate_stats["succeeded"],
                "rehydrated_chunks": rehydrate_stats["chunk_ids"],
                "l2_query_errors": sum(int(item.get("l2_query_errors", 0))
                                       for item in persistent_res),
                "latency": _latency,
            },
        }

    # ── Subclasses Must Override ──────────────────────────────────────

    def _extract_entities(self, query_emb: np.ndarray, query: str, top_entities: int):
        """Returns (entities, relations, entity_weights, entity_chunks)"""
        raise NotImplementedError

    def _handle_promotion(self, chunk_entity_coverage, entity_chunks, track_promotion):
        """Subclasses may optionally override"""
        pass

    # ── Graph Retrieval + Score Aggregation (Shared) ─────────────────────────

    def _graph_retrieve(self, query_emb, entities, relations, entity_weights,
                        top_entities, top_chunks, track_promotion):
        memory_res, persistent_res = [], []
        chunk_entity_coverage = {}

        def _one_entity(ent):
            """单个实体的 L1+L2 检索(M2:多实体并发执行,降查询时延)。"""
            ew = entity_weights.get(ent["id"], 0)
            m_res = (self._retrieve_memory(query_emb, ent, top_entities, top_chunks,
                                           relations, ew) if self.enable_l1 else
                     {"chunks": [], "chunk_scores": {}, "node_scores": {},
                      "node_chunks": {}, "matched_entities": []})
            p_res = (self._retrieve_persistent(query_emb, ent, top_entities, top_chunks,
                                               relations, ew) if self.enable_l2 else
                     {"chunks": [], "chunk_scores": {}, "node_scores": {},
                      "node_chunks": {}, "matched_entities": []})
            return ent["id"], m_res, p_res

        if len(entities) > 1 and self.graph_entity_parallel > 1:
            with concurrent.futures.ThreadPoolExecutor(
                    max_workers=min(len(entities), self.graph_entity_parallel),
                    thread_name_prefix="graph-entity") as ex:
                for ent_id, m_res, p_res in ex.map(_one_entity, entities):
                    memory_res.append(m_res)
                    persistent_res.append(p_res)
                    for cid in set(list(m_res.get("chunk_scores", {}).keys()) +
                                   list(p_res.get("chunk_scores", {}).keys())):
                        chunk_entity_coverage.setdefault(cid, set()).add(ent_id)
        else:
            for ent in entities:
                ent_id, m_res, p_res = _one_entity(ent)
                memory_res.append(m_res)
                persistent_res.append(p_res)
                for cid in set(list(m_res.get("chunk_scores", {}).keys()) +
                               list(p_res.get("chunk_scores", {}).keys())):
                    chunk_entity_coverage.setdefault(cid, set()).add(ent_id)
        return memory_res, persistent_res, chunk_entity_coverage

    def _retrieve_memory(self, query_emb, ent, top_entities, top_chunks,
                         relations, entity_weight):
        """Subclasses override this method to provide L1 retrieval. Empty by default."""
        return {"chunks": [], "chunk_scores": {}, "node_scores": {}, "node_chunks": {},
                "matched_entities": []}

    def _retrieve_persistent(self, query_emb, ent, top_entities, top_chunks,
                             relations, entity_weight):
        """Subclasses override this method to provide L2 retrieval. Empty by default."""
        return {"chunks": [], "chunk_scores": {}, "node_scores": {}, "node_chunks": {},
                "matched_entities": []}

    def _aggregate_graph_scores(self, memory_res, persistent_res, top_chunks, query=""):
        graph_chunk_scores_aggregated = {}
        all_node_scores = {}
        all_node_chunks = {}
        for res in memory_res + persistent_res:
            for cid, score in res.get("chunk_scores", {}).items():
                graph_chunk_scores_aggregated[cid] = graph_chunk_scores_aggregated.get(cid, 0.0) + score
            for nid, score in res.get("node_scores", {}).items():
                all_node_scores[nid] = all_node_scores.get(nid, 0) + score
            for nid, chunks in res.get("node_chunks", {}).items():
                all_node_chunks.setdefault(nid, set()).update(chunks)
        top_nodes = sorted(all_node_scores.items(), key=lambda x: x[1], reverse=True)[:top_chunks]

        # Entity rerank: reorder top-N entities by semantic similarity between description text and query
        if top_nodes and self.reranker:
            entity_texts = []
            for nid, _ in top_nodes:
                # Lookup in L1
                d = self.memory_graph.graph.nodes.get(nid, {})
                name = d.get('name', '')
                etype = d.get('type', '')
                desc = d.get('desc', '')
                # Not in L1 → fetch from L2 (NebulaGraph)
                if not name and self.persistent_graph:
                    try:
                        vid = self.persistent_graph._format_vid(nid)
                        rows = self.persistent_graph.query(
                            f"FETCH PROP ON `entity` {vid} "
                            "YIELD properties(vertex).name AS name, "
                            "properties(vertex).type AS type;")
                        name = str(rows.get('name', [''])[0] or '') if rows else ''
                        etype = str(rows.get('type', [''])[0] or '') if rows else ''
                    except Exception:
                        pass
                entity_texts.append(f"{name} {etype}")
            if entity_texts:
                reranked = self.reranker.rerank(query, entity_texts, top_n=len(top_nodes))
                top_nodes = [(top_nodes[item["index"]][0], item["index"]) for item in reranked]

        # Chunks ordered by entity rerank: entities with higher rank have their associated chunks listed first
        entity_rank = {nid: rank for rank, (nid, _) in enumerate(top_nodes)}
        entity_chunk_set = set()
        for nid, _ in top_nodes:
            entity_chunk_set.update(all_node_chunks.get(nid, set()))
        graph_ranked = []
        for cid, score in graph_chunk_scores_aggregated.items():
            if cid not in entity_chunk_set:
                continue
            # Take the best rerank among entities associated with this chunk as the main sorting criterion
            best_entity_rank = min(
                (entity_rank.get(nid, 999) for nid in all_node_chunks if cid in all_node_chunks[nid]),
                default=999)
            graph_ranked.append((cid, -best_entity_rank, score))  # -rank makes smaller rank come first
        graph_ranked.sort(key=lambda x: (x[1], x[2]), reverse=True)
        return [(cid, score) for cid, _, score in graph_ranked]


# ================================================================
#  HybridRetriever — Traditional Entity Extraction + Graph Traversal
# ================================================================

class HybridRetriever(BaseRetriever):
    """Traditional retrieval: extract entities via milvus or LLM → 2-hop graph traversal → equal weighting."""

    def __init__(self, vector_store=None, entity_index_name="entity_index_example",
                 memory_graph=None, llm=None, reranker=None, chunk_registry=None,
                 entity_extraction: str = "milvus", mode="hybrid"):
        super().__init__(vector_store=vector_store, memory_graph=memory_graph,
                         llm=llm, reranker=reranker, chunk_registry=chunk_registry,
                         mode=mode)
        self.entity_index_name = entity_index_name
        self.entity_extraction = entity_extraction
        self._entity_store = MilvusDB(
            db_name=entity_index_name, overwrite=False,
            embed_model=llm.embed_model if llm else None)

    # ── Entity Extraction ─────────────────────────────────────────

    def _extract_entities_heuristic(self, query, max_entities=6):
        """Heuristically extract entities from query text (uppercase words, quoted phrases)."""
        entities = []
        seen = set()
        # Quoted phrases first
        for token in query.split('"'):
            token = token.strip()
            if token and token not in seen and len(entities) < max_entities:
                seen.add(token)
                entities.append({"id": token, "type": "", "desc": ""})
        # Uppercase words
        words = [w.strip(".,:;!?()[]{}\"") for w in query.split()]
        for word in words:
            if len(entities) >= max_entities:
                break
            if not word:
                continue
            if word[0].isupper() or word.isupper():
                if word not in seen:
                    seen.add(word)
                    entities.append({"id": word, "type": "", "desc": ""})
        # Fallback: use all lowercase words
        if not entities:
            for word in words[:max_entities]:
                if word not in seen:
                    seen.add(word)
                    entities.append({"id": word, "type": "", "desc": ""})
        return entities

    def _extract_entities(self, query_emb, query, top_entities):
        if self.mode == "graph_only":
            entities = self._extract_entities_heuristic(query, top_entities)
            weights = {e["id"]: 1.0 for e in entities}
            return entities, [], weights, {}
        if self.entity_extraction == "milvus":
            entities = self._extract_entities_from_milvus(query_emb, top_entities)
            relations = []
        else:
            raw = self.llm.complete(prompt=prompt_extract_entities_str.format(context=query))
            parsed = self._parse_entities_from_llm(raw)
            entities = parsed.get("entities", [])
            relations = parsed.get("relations", [])
        weights = {e["id"]: 1.0 for e in entities}
        return entities, relations, weights, {}

    def _extract_entities_from_milvus(self, query_emb, topk=5):
        search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
        results = self._search_milvus(self._entity_store, query_emb, search_params, limit=topk)
        entities = []
        if results and len(results) > 0:
            for hit in results[0]:
                ent = hit.entity
                name = ent.get("name") or ""
                etype = ent.get("type") or ""
                edesc = ent.get("desc") or ""
                if name:
                    entities.append({"id": str(name), "type": str(etype), "desc": str(edesc)})
        return entities

    def _parse_entities_from_llm(self, raw_text: str) -> Dict[str, List]:
        if not raw_text:
            return {"entities": [], "relations": []}
        raw_text = raw_text.strip()
        if raw_text.startswith("```"):
            raw_text = raw_text.strip("`")
            if raw_text.lower().startswith("json"):
                raw_text = raw_text[4:]
        start, end = raw_text.find("{"), raw_text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return {"entities": [], "relations": []}
        try:
            payload = json.loads(raw_text[start:end + 1])
        except json.JSONDecodeError:
            return {"entities": [], "relations": []}
        entities = payload.get("entities", []) or []
        relations_raw = (payload.get("relations") or payload.get("relationship")
                         or payload.get("relationships") or [])
        relations = []
        if isinstance(relations_raw, str):
            relations = [r.strip() for r in re.split(r"[;,]", relations_raw) if r.strip()]
        elif isinstance(relations_raw, list):
            for rel in relations_raw:
                if isinstance(rel, str) and rel.strip():
                    relations.append(rel.strip())
                elif isinstance(rel, dict):
                    v = rel.get("rel") or rel.get("relation") or rel.get("text")
                    if v: relations.append(str(v).strip())
        return {"entities": entities, "relations": relations}

    # ── Promotion ────────────────────────────────────────────

    def _handle_promotion(self, chunk_entity_coverage, entity_chunks, track_promotion):
        if not track_promotion:
            return
        for cid in chunk_entity_coverage:
            triggered, data = self.memory_graph.access_chunk(cid)
            if triggered:
                self.memory_graph.write_to_persistent_graph([data])

    # ── Graph Traversal ──────────────────────────────────────────

    def _search_milvus(self, milvus_db, vector, search_params, limit):
        return milvus_db.search(vector, search_params, limit,
                                       output_fields=["uid", "name", "type", "desc"])

    def _retrieve_memory(self, query_emb, ent, top_entities, top_chunks,
                         relations, entity_weight):
        return self.retrieve_from_memory_graph(
            query_emb, ent["id"], ent["type"], ent["desc"],
            top_entities, top_chunks, relations, entity_weight)

    def _retrieve_persistent(self, query_emb, ent, top_entities, top_chunks,
                             relations, entity_weight):
        return self.retrieve_from_persistent_graph(
            query_emb, ent["id"], ent["type"], ent["desc"],
            top_entities, top_chunks, relations, entity_weight)

    def _beam_prune_frontier(self, nodes, k, hop, similarity_emb):
        """束搜索剪枝(论文 IV-E-1):每跳扩展后保留 top-B 高分候选路径。

        打分 = 节点名与查询的余弦相似度 × γ^hop —— 深度衰减因子注入路由打分,
        与 hop 权重(HOP_WEIGHTS)配合抑制长尾结构噪声。返回顺序:top-B 高分节点
        在前,其余节点补位在后(保持与原 top-k 剪枝一致的返回契约)。
        """
        if k <= 0 or len(nodes) <= k:
            return nodes
        graph = self.memory_graph.graph if self.memory_graph is not None else None
        vs = self._entity_store
        texts, valid = [], []
        for n in nodes:
            # M4:锁内快照读节点属性
            d = self.memory_graph.snapshot_node_data(n) if self.memory_graph is not None else None
            d = d or {}
            name = d.get('name', '') or ''
            if name.strip():
                texts.append(f"Entity: {name}. Type: {d.get('type', '') or ''}.")
                valid.append(n)
        if not valid:
            return list(nodes)[:k]
        # M2:批量嵌入走并发门控
        with self._embed_gate:
            embs = vs.embed_model.get_embeddings(texts)
        decay = self.gamma ** max(0, hop)
        scored = [(n, self._cosine(similarity_emb, np.array(e, dtype=float)) * decay)
                  for n, e in zip(valid, embs)]
        scored.sort(key=lambda x: x[1], reverse=True)
        top = [n for n, _ in scored[:k]]
        top_set = set(top)
        rest = [n for n in nodes if n not in top_set]
        return top + rest[:max(0, k - len(top))]

    def retrieve_from_memory_graph(self, query_emb, entity_name, entity_type,
                                   entity_desc, top_entities=5, top_chunks=5,
                                   query_relations=None, entity_weight=1.0):
        if not self.memory_graph or (not entity_name and not entity_desc):
            return {"chunks": [], "chunk_scores": {}, "node_scores": {}, "node_chunks": {},
                    "matched_entities": []}

        graph = self.memory_graph.graph
        vs = self._entity_store
        text = f"Entity: {entity_name}. Type: {entity_type}. Description: {entity_desc}"
        # M2:嵌入并发门控(实体嵌入为同步 HTTP)
        with self._embed_gate:
            entity_emb = np.array(vs.embed_model.get_embedding(text), dtype=float)
        similarity_emb = query_emb if query_emb is not None else entity_emb
        sp = {"metric_type": "COSINE", "params": {"nprobe": 10}}
        results = self._search_milvus(vs, entity_emb, sp, limit=top_entities)

        matched, matched_ids = [], []
        if results and len(results) > 0:
            for hit in results[0]:
                uid = hit.entity.get("uid")
                name = hit.entity.get("name")
                matched.append({"uid": uid, "name": name, "score": float(hit.distance)})
                matched_ids.append(uid)
        matched_id_set = {str(uid) for uid in matched_ids if uid is not None}

        node_set = set()
        self.memory_graph._rebuild_id_index()
        for mid in matched_id_set:
            node_set.update(self.memory_graph.get_nodes_by_id(mid))

        # BFS 跳数衰减:权重 γ^l(可配置,默认 γ=0.5),遍历深度受 max_hops 约束
        HOP_WEIGHTS = [self.gamma ** l for l in range(self.max_hops)]
        chunk_scores, node_scores, node_chunks = {}, {}, {}

        def add_score(chunks_data, weight):
            if isinstance(chunks_data, str):
                for c in [x.strip() for x in chunks_data.split(",") if x.strip()]:
                    chunk_scores[c] = chunk_scores.get(c, 0.0) + weight
            elif isinstance(chunks_data, (set, list, tuple)):
                for c in chunks_data:
                    c = str(c).strip()
                    if c: chunk_scores[c] = chunk_scores.get(c, 0.0) + weight

        # hop 0: seed entity
        frontier = set(node_set)
        visited = set()
        for hop in range(self.max_hops):  # 遍历深度受 max_hops 约束(默认 3)
            if not frontier:
                break
            hop_weight = entity_weight * HOP_WEIGHTS[min(hop, len(HOP_WEIGHTS) - 1)]
            next_frontier = set()
            for uid in frontier:
                if uid in visited:
                    continue
                visited.add(uid)
                # M4:锁内快照读节点属性/边,避免与写方(add/evict)并发迭代竞态
                d = self.memory_graph.snapshot_node_data(uid) or {}
                # Node chunk weighting
                add_score(d.get("source_chunks"), hop_weight)
                node_scores[uid] = node_scores.get(uid, 0) + hop_weight
                node_chunks.setdefault(uid, set()).update(d.get("source_chunks") or [])
                # Traverse out-edges, collect next-hop nodes + current edge's chunk
                for _, neighbor, data in self.memory_graph.snapshot_edges(uid, "out"):
                    if neighbor not in visited:
                        next_frontier.add(neighbor)
                    chunk = data.get("source_chunk")
                    if chunk:
                        add_score([chunk], hop_weight)
                for neighbor, _, data in self.memory_graph.snapshot_edges(uid, "in"):
                    if neighbor not in visited:
                        next_frontier.add(neighbor)
                    chunk = data.get("source_chunk")
                    if chunk:
                        add_score([chunk], hop_weight)
            # 束搜索剪枝(论文 IV-E-1):下一跳保留 top-B 高分候选路径,
            # 打分 = 节点-查询余弦相似度 × γ^(hop+1)(深度衰减注入路由打分)。
            if self.beam_width > 0:
                frontier = set(self._beam_prune_frontier(
                    list(next_frontier), self.beam_width, hop + 1, similarity_emb))
            else:
                frontier = next_frontier

        sorted_chunks = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)[:top_chunks]
        return {"chunks": [c for c, _ in sorted_chunks], "chunk_scores": dict(sorted_chunks),
                "node_scores": node_scores, "node_chunks": {k: list(v) for k, v in node_chunks.items()},
                "matched_entities": matched}

    def retrieve_from_persistent_graph(self, query_emb, entity_name, entity_type,
                                       entity_desc, top_entities=5, top_chunks=5,
                                       query_relations=None, entity_weight=1.0):
        if not self.persistent_graph or (not entity_name and not entity_desc):
            return {"chunks": [], "chunk_scores": {}, "node_scores": {}, "node_chunks": {},
                    "matched_entities": []}

        g = self.persistent_graph
        vs = self._entity_store
        query_errors = []

        def _on_l2_error(operation, exc):
            message = f"L2 query failed during {operation}: {exc}"
            if self.fail_on_l2_error:
                raise RuntimeError(message) from exc
            query_errors.append(message)
            print(f"[L2 warning] {message}")

        text = f"Entity: {entity_name}. Type: {entity_type}. Description: {entity_desc}"
        # M2:嵌入并发门控(实体嵌入为同步 HTTP)
        with self._embed_gate:
            entity_emb = np.array(vs.embed_model.get_embedding(text), dtype=float)
        similarity_emb = query_emb if query_emb is not None else entity_emb
        sp = {"metric_type": "COSINE", "params": {"nprobe": 10}}
        results = self._search_milvus(vs, entity_emb, sp, limit=top_entities)

        matched, matched_ids = [], []
        if results and len(results) > 0:
            for hit in results[0]:
                matched.append({"uid": hit.entity.get("uid"),
                                "name": hit.entity.get("name"),
                                "score": float(hit.distance)})
                matched_ids.append(hit.entity.get("uid"))
        if not matched_ids:
            return {"chunks": [], "chunk_scores": {}, "node_scores": {}, "node_chunks": {},
                    "matched_entities": []}

        vid_list = [g._format_vid(uid) for uid in matched_ids if uid is not None]
        if not vid_list:
            return {"chunks": [], "chunk_scores": {}, "node_scores": {}, "node_chunks": {},
                    "matched_entities": []}

        vids_str = ", ".join(vid_list)
        try:
            node_rows = g.query(
                f"FETCH PROP ON `entity` {vids_str} "
                "YIELD id(vertex) AS vid, properties(vertex).name AS name, "
                "properties(vertex).source_chunk AS source_chunk;")
        except Exception as exc:
            _on_l2_error("seed entity fetch", exc)
            return {"chunks": [], "chunk_scores": {}, "node_scores": {}, "node_chunks": {},
                    "matched_entities": [], "l2_query_errors": len(query_errors)}

        existing_vids = set(str(v) for v in node_rows.get("vid", []) if v is not None)
        if not existing_vids:
            return {"chunks": [], "chunk_scores": {}, "node_scores": {}, "node_chunks": {},
                    "matched_entities": []}
        matched = [m for m in matched if str(m.get("uid")) in existing_vids]
        if not matched:
            return {"chunks": [], "chunk_scores": {}, "node_scores": {}, "node_chunks": {},
                    "matched_entities": []}

        def _fetch_node_props(vids):
            if not vids: return {}
            vs_str = ", ".join([g._format_vid(v) for v in vids])
            try:
                rows = g.query(
                    f"FETCH PROP ON `entity` {vs_str} "
                    "YIELD id(vertex) AS vid, properties(vertex).name AS name, "
                    "properties(vertex).type AS type, "
                    "properties(vertex).source_chunk AS source_chunk;")
            except Exception as exc:
                _on_l2_error("node property fetch", exc)
                return {}
            props = {}
            for idx, vid in enumerate(rows.get("vid", [])):
                if vid is not None:
                    props[str(vid)] = {
                        "name": rows["name"][idx] if idx < len(rows.get("name", [])) else "",
                        "type": rows["type"][idx] if idx < len(rows.get("type", [])) else "",
                        "source_chunk": rows["source_chunk"][idx] if idx < len(rows.get("source_chunk", [])) else "",
                    }
            return props

        def _run_go(vids):
            if not vids: return {}
            vs_str = ", ".join([g._format_vid(v) for v in vids])
            try:
                return g.query(
                    f"GO 1 STEPS FROM {vs_str} OVER `relationship` BIDIRECT "
                    "YIELD src(edge) AS src, dst(edge) AS dst, "
                    "properties(edge).relationship AS relation, "
                    "properties(edge).source_chunk AS source_chunk;")
            except Exception as exc:
                _on_l2_error("graph traversal", exc)
                return {}

        def _node_text(row):
            return f"Entity: {row.get('name', '')}. Type: {row.get('type', '')}."

        def _topk_sim(vids, k=5):
            props = _fetch_node_props(vids)
            if not props: return []
            texts = [(_node_text(r), vid) for vid, r in props.items()]
            embs = vs.embed_model.get_embeddings([t for t, _ in texts])
            scored = [(vid, self._cosine(similarity_emb, np.array(e, dtype=float)))
                      for (_, vid), e in zip(texts, embs)]
            scored.sort(key=lambda x: x[1], reverse=True)
            return [vid for vid, _ in scored[:k]]

        base_vids = set(str(v) for v in existing_vids)
        _collect = lambda rows: {str(v) for k in ("src", "dst") for v in rows.get(k, []) if v is not None}

        # 按 max_hops 逐跳展开:每跳取 topk 相似节点继续前推,边/节点权重 = γ^hop
        hop_edges = []              # 每跳的 GO 边结果(边 chunk 贡献)
        hop_props = {}              # hop -> 该跳 topk 节点属性
        frontier = set(base_vids)
        visited = set(base_vids)
        for hop in range(1, self.max_hops):
            edge_rows = _run_go(frontier)
            hop_edges.append(edge_rows)
            next_hop = _collect(edge_rows) - visited
            top_hop = _topk_sim(next_hop, k=5)
            if not top_hop:
                break
            hop_props[hop] = _fetch_node_props(set(top_hop))
            visited.update(top_hop)
            frontier = set(top_hop)

        chunk_scores, node_scores, node_chunks = {}, {}, {}
        def add_score(chunks_data, weight):
            if isinstance(chunks_data, str):
                for c in [x.strip() for x in chunks_data.split(",") if x.strip()]:
                    chunk_scores[c] = chunk_scores.get(c, 0.0) + weight
            elif isinstance(chunks_data, list):
                for c in chunks_data:
                    c = str(c).strip()
                    if c: chunk_scores[c] = chunk_scores.get(c, 0.0) + weight

        node_vids = node_rows.get("vid", [])
        node_chunk_list = node_rows.get("source_chunk", [])

        # hop 0:seed 实体,权重 entity_weight
        for idx, chunks in enumerate(node_chunk_list):
            add_score(chunks, entity_weight)
            if idx < len(node_vids) and node_vids[idx] is not None:
                vid = str(node_vids[idx])
                node_scores[vid] = node_scores.get(vid, 0) + entity_weight
                node_chunks.setdefault(vid, set()).update(
                    [c.strip() for c in str(chunks).split(",") if c.strip()] if chunks else [])
        # hop ≥1:节点权重 = γ^hop * entity_weight
        for hop, props in hop_props.items():
            hop_weight = (self.gamma ** hop) * entity_weight
            for vid, row in props.items():
                add_score(row.get("source_chunk", ""), hop_weight)
                node_scores[vid] = node_scores.get(vid, 0) + hop_weight
                node_chunks.setdefault(vid, set()).update(
                    [c.strip() for c in str(row.get("source_chunk", "")).split(",") if c.strip()]
                    if row.get("source_chunk") else [])
        # 边 chunk 贡献:第 hop_idx 跳的边权重 = γ^hop_idx * entity_weight
        for hop_idx, edge_rows in enumerate(hop_edges, start=1):
            hop_weight = (self.gamma ** hop_idx) * entity_weight
            for chunk in edge_rows.get("source_chunk", []):
                if chunk: add_score([chunk], hop_weight)

        sorted_chunks = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)
        return {"chunks": [c for c, _ in sorted_chunks], "chunk_scores": chunk_scores,
                "node_scores": node_scores, "node_chunks": {k: list(v) for k, v in node_chunks.items()},
                "matched_entities": matched, "l2_query_errors": len(query_errors)}


# ================================================================
#  FactRetriever — Fact Embedding Retrieval + Convergence Promotion
# ================================================================

class FactRetriever(BaseRetriever):
    """Fact retrieval: Query → fact embedding → entities+weights → graph traversal → convergence promotion."""

    def __init__(self, vector_store=None, entity_index_name="entity_index_example",
                 memory_graph=None, llm=None, reranker=None, chunk_registry=None,
                 fact_store=None, fact_config: dict = None):
        super().__init__(vector_store=vector_store, memory_graph=memory_graph,
                         llm=llm, reranker=reranker, chunk_registry=chunk_registry)
        self.fact_store = fact_store
        self.fact_config = fact_config or {}
        # Reuse HybridRetriever's graph traversal, create internal instance
        self._graph = HybridRetriever(
            vector_store=vector_store, entity_index_name=entity_index_name,
            memory_graph=memory_graph, llm=llm, chunk_registry=chunk_registry,
            entity_extraction="milvus")

    # ── Entity Extraction ─────────────────────────────────────────

    def _extract_entities(self, query_emb, query, top_entities):
        if not self.fact_store or not self.llm:
            return [], [], {}, {}
        topk = self.fact_config.get("topk_facts", 100)
        max_entities = self.fact_config.get("topk_entities", 10)
        facts = self.fact_store.search_facts(query_emb.tolist(), topk=topk)
        if not facts:
            return [], [], {}, {}

        entity_scores, entity_names, entity_chunks = {}, {}, {}
        for f in facts:
            score = float(f.get("score", 0))
            chunk_id = f.get("chunk_id", "")
            for uk, nk in [("subj_uid", "subj_name"), ("obj_uid", "obj_name")]:
                uid = str(f.get(uk, ""))
                name = f.get(nk, "")
                if not uid or uid == "0": continue
                entity_scores[uid] = entity_scores.get(uid, 0.0) + score
                entity_names[uid] = name or entity_names.get(uid, "")
                entity_chunks.setdefault(uid, set()).add(chunk_id)

        entity_weights = {}
        for uid, total_score in entity_scores.items():
            if not self.memory_graph or not self.memory_graph.graph.has_node(uid):
                continue  # Skip entities not in the graph
            ref_count = max(1, int(self.memory_graph.graph.nodes[uid].get("ref_count", 1)))
            entity_weights[uid] = total_score / ref_count

        if not entity_weights:
            return [], [], {}, {}

        sorted_ents = sorted(entity_weights.items(), key=lambda x: x[1], reverse=True)[:max_entities]
        entities = [{"id": uid, "type": "", "desc": entity_names.get(uid, "")}
                    for uid, _ in sorted_ents]
        entity_weights = {uid: w for uid, w in sorted_ents}
        total = sum(entity_weights.values()) or 1.0
        entity_weights = {k: v / total for k, v in entity_weights.items()}

        return entities, [], entity_weights, entity_chunks

    # ── Graph Retrieval (directly locate nodes by UID, no re-embedding search) ──

    def _retrieve_memory(self, query_emb, ent, top_entities, top_chunks,
                         relations, entity_weight):
        return self._retrieve_by_uid(query_emb, ent, top_chunks, entity_weight,
                                     use_persistent=False)

    def _retrieve_persistent(self, query_emb, ent, top_entities, top_chunks,
                             relations, entity_weight):
        uid = str(ent["id"])
        if self.persistent_graph:
            return self._retrieve_from_l2_by_uid(uid, top_chunks, entity_weight)
        return {"chunks": [], "chunk_scores": {}, "node_scores": {}, "node_chunks": {},
                "matched_entities": []}

    def _retrieve_by_uid(self, query_emb, ent, top_chunks, entity_weight, use_persistent):
        """Given entity UID, directly start BFS traversal from graph node, skip entity_index search."""
        uid = str(ent["id"])

        if use_persistent and self.persistent_graph:
            return self._retrieve_from_l2_by_uid(uid, top_chunks, entity_weight)

        if not self.memory_graph or not self.memory_graph.graph.has_node(uid):
            return {"chunks": [], "chunk_scores": {}, "node_scores": {}, "node_chunks": {},
                    "matched_entities": []}

        return self._bfs_from_uid(uid, entity_weight, top_chunks)

    def _bfs_from_uid(self, uid, entity_weight, top_chunks):
        graph = self.memory_graph.graph
        chunk_scores, node_scores, node_chunks = {}, {}, {}

        def add_score(chunks_data, weight):
            if isinstance(chunks_data, str):
                for c in [x.strip() for x in chunks_data.split(",") if x.strip()]:
                    chunk_scores[c] = chunk_scores.get(c, 0.0) + weight
            elif isinstance(chunks_data, (set, list, tuple)):
                for c in chunks_data:
                    c = str(c).strip()
                    if c: chunk_scores[c] = chunk_scores.get(c, 0.0) + weight

        # BFS starting from UID
        DECAY = 0.5
        visited = set()
        frontier = {uid}
        hop_weight = entity_weight
        for _ in range(10):
            if not frontier: break
            next_frontier = set()
            for n in frontier:
                if n in visited: continue
                visited.add(n)
                d = graph.nodes.get(n, {})
                add_score(d.get("source_chunks"), hop_weight)
                node_scores[n] = node_scores.get(n, 0) + hop_weight
                node_chunks.setdefault(n, set()).update(d.get("source_chunks") or [])
                for _, nb, data in graph.out_edges(n, data=True):
                    if nb not in visited:
                        next_frontier.add(nb)
                    c = data.get("source_chunk")
                    if c: add_score([c], hop_weight * DECAY)
                for nb, _, data in graph.in_edges(n, data=True):
                    if nb not in visited:
                        next_frontier.add(nb)
                    c = data.get("source_chunk")
                    if c: add_score([c], hop_weight * DECAY)
            frontier = next_frontier
            hop_weight *= DECAY

        sorted_chunks = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)[:top_chunks]
        return {"chunks": [c for c, _ in sorted_chunks], "chunk_scores": dict(sorted_chunks),
                "node_scores": node_scores, "node_chunks": {k: list(v) for k, v in node_chunks.items()},
                "matched_entities": [{"uid": uid, "name": "", "score": 1.0}]}

    def _retrieve_from_l2_by_uid(self, uid, top_chunks, entity_weight):
        """Retrieve chunk of entity and its neighbors from L2 (NebulaGraph) by UID."""
        g = self.persistent_graph
        vid = g._format_vid(uid)
        chunk_scores = {}

        def add_score(chunks_data, weight):
            if isinstance(chunks_data, str):
                for c in [x.strip() for x in chunks_data.split(",") if x.strip()]:
                    chunk_scores[c] = chunk_scores.get(c, 0.0) + weight
            elif isinstance(chunks_data, list):
                for c in chunks_data:
                    c = str(c).strip()
                    if c: chunk_scores[c] = chunk_scores.get(c, 0.0) + weight

        try:
            rows = g.query(
                f"FETCH PROP ON `entity` {vid} "
                "YIELD properties(vertex).source_chunk AS source_chunk;")
            if rows:
                for sc in rows.get("source_chunk", []):
                    add_score(sc, entity_weight)
        except Exception:
            pass

        try:
            rows = g.query(
                f"GO 1 STEPS FROM {vid} OVER `relationship` BIDIRECT "
                "YIELD properties(edge).source_chunk AS source_chunk;")
            if rows:
                for sc in rows.get("source_chunk", []):
                    add_score(sc, 0.5 * entity_weight)
        except Exception:
            pass

        sorted_chunks = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)[:top_chunks]
        return {"chunks": [c for c, _ in sorted_chunks], "chunk_scores": dict(sorted_chunks),
                "node_scores": {}, "node_chunks": {},
                "matched_entities": [{"uid": uid, "name": "", "score": 1.0}]}

    # ── Quality Promotion (called by CacheGraphRAG after answer generation) ──

    def _handle_promotion(self, chunk_entity_coverage, entity_chunks, track_promotion):
        pass  # Promotion deferred to after answer generation, triggered by promote_after_qa

    def promote_after_qa(self, chunk_entity_coverage, quality_ok):
        """Called after answer generation, skips promotion when quality_ok=False."""
        if not quality_ok:
            return
        threshold = self.fact_config.get("entity_convergence_threshold", 3)
        self.memory_graph.promote_by_convergence(chunk_entity_coverage, threshold=threshold)
