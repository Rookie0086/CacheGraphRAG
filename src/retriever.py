import asyncio
import time
import json
import math
import os
import re
import sys
import uuid
from collections import Counter
import networkx as nx
import numpy as np
from typing import List, Dict, Tuple
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pymilvus.exceptions import MilvusException
from tqdm import tqdm
from difflib import SequenceMatcher
from utils import get_config
from utils.base import read_json, save_to_json
from utils.prompts import prompt_extract_triplest_str, prompt_extract_entities_str, prompt_answer_with_chunks_str
from utils.llm_env import LLMEnv
from database.milvus import MilvusDB, myMilvus
from database.nebulagraph import NebulaClient, NebulaDB
from langchain_text_splitters import RecursiveCharacterTextSplitter
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


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

class HybridRetriever:
    def __init__(
        self, 
        vector_store=MilvusDB(db_name="example", overwrite=False), 
        entity_index_name="entity_index_example",
        memory_graph=None, 
        persistent_graph=None, 
        llm = None, 
        reranker=None,
        chunk_registry=None
        ):
        self.vector_store = vector_store
        self.entity_index_name = entity_index_name
        self.memory_graph = memory_graph
        self.persistent_graph = self.memory_graph.persistent_graph 
        self.llm = llm
        self.reranker = reranker or LocalReranker(
            model_path="/home/shuyurui/model/bge-reranker-v2-m3",
        )
        self.chunk_registry = chunk_registry or {}

    def _embed_text(self, text: str) -> np.ndarray:
        if not self.llm:
            raise ValueError("LLM env is required for embeddings.")
        embedding = self.llm.embed_model.get_embedding(text)
        return np.array(embedding, dtype=float)

    def _search_milvus(self, milvus_db, vector, search_params, limit):
        if not milvus_db.db:
            milvus_db.load()
        return milvus_db.db.search(
            data=[vector],
            anns_field="vec",
            param=search_params,
            limit=limit,
            output_fields=["uid", "name"],
            consistency_level="Strong",
        )
    
    def retrieve_from_embedding(self, query_emb: np.ndarray, topk: int = 5):
        try:
            if not self.vector_store.db:
                self.vector_store.load()
            search_params = {
                "metric_type": self.vector_store.metric,
                "params": {"nprobe": 10},
            }
            output_fields = ["chunk_id"] if self.vector_store._has_field("chunk_id") else []
            result = self.vector_store.db.search(
                data=[query_emb.tolist()],
                anns_field="vec",
                param=search_params,
                limit=topk,
                output_fields=output_fields,
                consistency_level="Strong",
            )
        except Exception:
            result = []

        hits = []
        for hits_group in result:
            for hit in hits_group:
                chunk_id = None
                if output_fields:
                    chunk_id = hit.entity.get("chunk_id")
                if chunk_id:
                    hit_id = str(chunk_id)
                else:
                    hit_id = str(hit.pk)
                hits.append({"id": hit_id, "score": float(hit.distance)})
        return {"embedding_hits": hits}
    
    def save_graph_gexf(self, path: str, g: nx.MultiDiGraph):
        export_g = g.copy()
        for _, data in export_g.nodes(data=True):
            if "source_chunks" in data and isinstance(data["source_chunks"], (set, list)):
                data["source_chunks"] = ",".join(str(v) for v in data["source_chunks"])
        for _, _, _, data in export_g.edges(data=True, keys=True):
            if "source_chunk" in data and isinstance(data["source_chunk"], (set, list)):
                data["source_chunk"] = ",".join(str(v) for v in data["source_chunk"])
        nx.write_gexf(export_g, path)

    def retrieve_from_memory_graph(
        self,
        query_emb: np.ndarray,
        entity_name: str,
        entity_type: str,
        entity_desc: str,
        top_entities: int = 5,
        top_chunks: int = 5,
        query_relations: List[str] = None,
    ):
        if not self.memory_graph:
            return {"chunks": [], "matched_entities": []}
        if not entity_name and not entity_desc:
            return {"chunks": [], "matched_entities": []}

        graph = self.memory_graph.graph
        vector_store = MilvusDB(db_name=self.entity_index_name, overwrite=False)
        text_to_embed = f"Entity: {entity_name}. Type: {entity_type}. Description: {entity_desc}"
        entity_emb = np.array(vector_store.embed_model.get_embedding(text_to_embed), dtype=float)
        similarity_emb = query_emb if query_emb is not None else entity_emb
        search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
        results = self._search_milvus(vector_store, entity_emb, search_params, limit=top_entities)

        matched = []
        matched_ids = []
        if results and len(results) > 0:
            for hit in results[0]:
                uid = hit.entity.get("uid")
                name = hit.entity.get("name")
                matched.append({"uid": uid, "name": name, "score": float(hit.distance)})
                matched_ids.append(uid)
        matched_id_set = {str(uid) for uid in matched_ids if uid is not None}
        # print(f"Matched entity UIDs: {matched_id_set}")
        node_set = set()
        self.memory_graph._rebuild_id_index()  # Ensure ID index is up-to-date before fetching nodes
        for mid in matched_id_set:
            node_set.update(self.memory_graph.get_nodes_by_id(mid))
        # print(f"Initial matched nodes in graph: {node_set}")
        base_nodes = set(node_set)
        
        def _cosine(a: np.ndarray, b: np.ndarray) -> float:
            denom = (np.linalg.norm(a) * np.linalg.norm(b))
            if denom == 0:
                return 0.0
            return float(np.dot(a, b) / denom)

        relation_texts = [str(r).strip() for r in (query_relations or []) if str(r).strip()]
        relation_emb_cache = {}
        query_rel_embs = [
            np.array(vector_store.embed_model.get_embedding(r), dtype=float)
            for r in relation_texts
        ]

        def _relation_match(rel_text: str) -> bool:
            if not rel_text or not query_rel_embs:
                return False
            rel_text = str(rel_text).strip()
            if not rel_text:
                return False
            if rel_text in relation_emb_cache:
                rel_emb = relation_emb_cache[rel_text]
            else:
                rel_emb = np.array(vector_store.embed_model.get_embedding(rel_text), dtype=float)
                relation_emb_cache[rel_text] = rel_emb
            for q_emb in query_rel_embs:
                if _cosine(rel_emb, q_emb) > 0.8:
                    return True
            return False

        def _node_text(uid) -> str:
            data = graph.nodes.get(uid, {})
            name = data.get("name") or str(uid)
            ent_type = data.get("type", "")
            desc = data.get("desc", "")
            return f"Entity: {name}. Type: {ent_type}. Description: {desc}"

        def _topk_by_similarity(node_ids: set, k: int = 5) -> List:
            scored = []
            for uid in node_ids:
                text = _node_text(uid)
                emb = np.array(vector_store.embed_model.get_embedding(text), dtype=float)
                scored.append((uid, _cosine(similarity_emb, emb)))
            scored.sort(key=lambda x: x[1], reverse=True)
            return [uid for uid, _ in scored[:k]]
            
        first_hop = set()
        for uid in list(node_set):
            first_hop.update(graph.predecessors(uid))
            first_hop.update(graph.successors(uid))
        node_set.update(first_hop)
        # 只选取一跳中最相似的节点进行扩展
        top_first = _topk_by_similarity(first_hop, k=5)
        second_hop = set()
        for uid in top_first:
            second_hop.update(graph.predecessors(uid))
            second_hop.update(graph.successors(uid))
        node_set.update(second_hop)

        # 再从二跳中选最相似的节点扩展三跳
        # top_second = _topk_by_similarity(second_hop, k=5)
        # third_hop = set()
        # for uid in top_second:
        #     third_hop.update(graph.predecessors(uid))
        #     third_hop.update(graph.successors(uid))
        # node_set.update(third_hop)
        # print(f"Expanded node set with neighbors: {node_set}")
        subgraph = graph.subgraph(node_set).copy()
        # self.save_graph_gexf(f"subgraph/temp_subgraph_{entity_name}.gexf", subgraph)  # Debug: export subgraph to inspect structure

        chunk_scores = {}
        def add_score(chunks_data, weight):
            if isinstance(chunks_data, str):
                parts = [c.strip() for c in chunks_data.split(",") if c.strip()]
                for c in parts:
                    chunk_scores[c] = chunk_scores.get(c, 0.0) + weight
            elif isinstance(chunks_data, (set, list, tuple)):
                for c in chunks_data:
                    c = str(c).strip()
                    if c:
                        chunk_scores[c] = chunk_scores.get(c, 0.0) + weight

        # 0-hop nodes: highest weight
        for uid in base_nodes:
            data = graph.nodes.get(uid, {})
            add_score(data.get("source_chunks"), 1.0)

        # 1-hop nodes: medium weight
        for uid in first_hop:
            data = graph.nodes.get(uid, {})
            add_score(data.get("source_chunks"), 0.5)

        # 2-hop nodes: low weight
        for uid in second_hop:
            data = graph.nodes.get(uid, {})
            add_score(data.get("source_chunks"), 0.3)

        # 3-hop nodes: minimal weight
        # for uid in third_hop:
        #     data = graph.nodes.get(uid, {})
        #     add_score(data.get("source_chunks"), 0.2)

        # Edges: weight by hop proximity
        for u, v, _, data in subgraph.edges(data=True, keys=True):
            chunk = data.get("source_chunk")
            if not chunk:
                continue
            rel_text = data.get("relation_type") or data.get("relation")
            if rel_text and _relation_match(rel_text):
                add_score([chunk], 1.0)
            if u in base_nodes or v in base_nodes:
                add_score([chunk], 0.5)
            elif u in first_hop or v in first_hop:
                add_score([chunk], 0.3)
            elif u in second_hop or v in second_hop:
                add_score([chunk], 0.2)
            # elif u in third_hop or v in third_hop:
            #     add_score([chunk], 0.1)

        sorted_chunks = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)
        top_chunk_ids = [c for c, _ in sorted_chunks[:top_chunks]]
        return {
            "chunks": top_chunk_ids,
            "chunk_scores": dict(sorted_chunks[:top_chunks]),
            "matched_entities": matched,
        }

    def retrieve_from_persistent_graph(
        self,
        query_emb: np.ndarray,
        entity_name: str,
        entity_type: str,
        entity_desc: str,
        top_entities: int = 5,
        top_chunks: int = 5,
        query_relations: List[str] = None,
    ):
        if not self.persistent_graph:
            return {"chunks": [], "matched_entities": []}
        if not entity_name and not entity_desc:
            return {"chunks": [], "matched_entities": []}

        graph = self.persistent_graph
        vector_store = MilvusDB(db_name=self.entity_index_name, overwrite=False)
        text_to_embed = f"Entity: {entity_name}. Type: {entity_type}. Description: {entity_desc}"
        entity_emb = np.array(vector_store.embed_model.get_embedding(text_to_embed), dtype=float)
        similarity_emb = query_emb if query_emb is not None else entity_emb
        search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
        results = self._search_milvus(vector_store, entity_emb, search_params, limit=top_entities)

        matched = []
        matched_ids = []
        if results and len(results) > 0:
            for hit in results[0]:
                uid = hit.entity.get("uid")
                name = hit.entity.get("name")
                matched.append({"uid": uid, "name": name, "score": float(hit.distance)})
                matched_ids.append(uid)

        if not matched_ids:
            return {"chunks": [], "matched_entities": []}
        # print(f"Matched entity UIDs in persistent graph: {matched_ids}")
        vid_list = [graph._format_vid(uid) for uid in matched_ids if uid is not None]
        if not vid_list:
            return {"chunks": [], "matched_entities": []}
        # print(f"Formatted VIDs for query: {vid_list}")
        # FETCH 过滤 NebulaGraph 中存在的实体
        vids_str = ", ".join(vid_list)
        fetch_nodes_query = (
            "FETCH PROP ON `entity` "
            f"{vids_str} "
            "YIELD id(vertex) AS vid, properties(vertex).name AS name, "
            "properties(vertex).source_chunk AS source_chunk;"
        )
        try:
            node_rows = graph.query(fetch_nodes_query)
        except Exception:
            print("Error occurred while fetching nodes.")
            return {"chunks": [], "matched_entities": []}

        existing_vids = set(str(v) for v in node_rows.get("vid", []) if v is not None)
        if not existing_vids:
            return {"chunks": [], "matched_entities": []}

        matched = [m for m in matched if str(m.get("uid")) in existing_vids]
        if not matched:
            return {"chunks": [], "matched_entities": []}

        def _cosine(a: np.ndarray, b: np.ndarray) -> float:
            denom = (np.linalg.norm(a) * np.linalg.norm(b))
            if denom == 0:
                return 0.0
            return float(np.dot(a, b) / denom)

        relation_texts = [str(r).strip() for r in (query_relations or []) if str(r).strip()]
        relation_emb_cache = {}
        query_rel_embs = [
            np.array(vector_store.embed_model.get_embedding(r), dtype=float)
            for r in relation_texts
        ]

        def _relation_match(rel_text: str) -> bool:
            if not rel_text or not query_rel_embs:
                return False
            rel_text = str(rel_text).strip()
            if not rel_text:
                return False
            if rel_text in relation_emb_cache:
                rel_emb = relation_emb_cache[rel_text]
            else:
                rel_emb = np.array(vector_store.embed_model.get_embedding(rel_text), dtype=float)
                relation_emb_cache[rel_text] = rel_emb
            for q_emb in query_rel_embs:
                if _cosine(rel_emb, q_emb) > 0.8:
                    return True
            return False

        def _node_text(row: dict) -> str:
            name = row.get("name") or ""
            ent_type = row.get("type", "")
            return f"Entity: {name}. Type: {ent_type}."

        def _fetch_node_props(vids: set) -> Dict[str, Dict[str, str]]:
            if not vids:
                return {}
            vids_str = ", ".join([graph._format_vid(v) for v in vids])
            fetch_query = (
                "FETCH PROP ON `entity` "
                f"{vids_str} "
                "YIELD id(vertex) AS vid, properties(vertex).name AS name, "
                "properties(vertex).type AS type, "
                "properties(vertex).source_chunk AS source_chunk;"
            )
            try:
                rows = graph.query(fetch_query)
            except Exception:
                return {}
            vids_list = rows.get("vid", [])
            names = rows.get("name", [])
            types = rows.get("type", [])
            chunks = rows.get("source_chunk", [])
            props = {}
            for idx, vid in enumerate(vids_list):
                if vid is None:
                    continue
                props[str(vid)] = {
                    "name": names[idx] if idx < len(names) else "",
                    "type": types[idx] if idx < len(types) else "",
                    "source_chunk": chunks[idx] if idx < len(chunks) else "",
                }
            return props

        def _topk_by_similarity(vids: set, k: int = 5) -> List[str]:
            props = _fetch_node_props(vids)
            scored = []
            for vid, row in props.items():
                text = _node_text(row)
                emb = np.array(vector_store.embed_model.get_embedding(text), dtype=float)
                scored.append((vid, _cosine(similarity_emb, emb)))
            scored.sort(key=lambda x: x[1], reverse=True)
            return [vid for vid, _ in scored[:k]]

        def _run_go_query(vids: set) -> Dict:
            if not vids:
                return {}
            vids_str = ", ".join([graph._format_vid(v) for v in vids])
            go_query = (
                "GO 1 STEPS FROM "
                f"{vids_str} "
                "OVER `relationship` BIDIRECT "
                "YIELD src(edge) AS src, dst(edge) AS dst, "
                "properties(edge).relationship AS relation, "
                "properties(edge).source_chunk AS source_chunk;"
            )
            try:
                return graph.query(go_query)
            except Exception:
                return {}

        def _add_relation_match_scores(rows: Dict):
            rels = rows.get("relation", [])
            chunks = rows.get("source_chunk", [])
            for idx, rel in enumerate(rels):
                if rel is None:
                    continue
                if not _relation_match(rel):
                    continue
                chunk = chunks[idx] if idx < len(chunks) else None
                if chunk:
                    add_score([chunk], 1.0)

        def _collect_vids(rows: Dict) -> set:
            vids = set()
            for key in ("src", "dst"):
                for vid in rows.get(key, []):
                    if vid is None:
                        continue
                    vids.add(str(vid))
            return vids

        base_vids = set(str(v) for v in existing_vids)
        edge_rows = _run_go_query(base_vids)
        first_hop = _collect_vids(edge_rows) - base_vids
        top_first = _topk_by_similarity(first_hop, k=5)

        edge_rows_2 = _run_go_query(set(top_first))
        second_hop = _collect_vids(edge_rows_2) - base_vids - set(top_first)
        top_second = _topk_by_similarity(second_hop, k=5)

        # edge_rows_3 = _run_go_query(set(top_second))
        # third_hop = _collect_vids(edge_rows_3) - base_vids - set(top_first) - set(top_second)

        first_props = _fetch_node_props(set(top_first))
        second_props = _fetch_node_props(set(top_second))
        # third_props = _fetch_node_props(set(third_hop))

        chunk_scores = {}
        def add_score(chunks_data, weight):
            """辅助函数：处理字符串列表并加权累计分数"""
            if isinstance(chunks_data, str):
                parts = [c.strip() for c in chunks_data.split(",") if c.strip()]
                for c in parts:
                    chunk_scores[c] = chunk_scores.get(c, 0.0) + weight
            elif isinstance(chunks_data, list):
                for c in chunks_data:
                    c = c.strip()
                    if c:
                        chunk_scores[c] = chunk_scores.get(c, 0.0) + weight

        # 1. 源实体本身的 Chunk (0-hop)：核心依据，给予最高权重 1.0
        for chunks in node_rows.get("source_chunk", []):
            add_score(chunks, 1.0)

        # 2. 1-hop 相似节点 Chunk：次级依据，给予权重 0.5
        for row in first_props.values():
            add_score(row.get("source_chunk", ""), 0.5)

        # 3. 2-hop 相似节点 Chunk：中等权重 0.3
        for row in second_props.values():
            add_score(row.get("source_chunk", ""), 0.3)

        # 4. 3-hop 节点 Chunk：较低权重 0.2
        # for row in third_props.values():
        #     add_score(row.get("source_chunk", ""), 0.2)

        # 边的 source_chunk 评分
        for chunk in edge_rows.get("source_chunk", []):
            if chunk:
                add_score([chunk], 0.5)
        for chunk in edge_rows_2.get("source_chunk", []):
            if chunk:
                add_score([chunk], 0.3)
        _add_relation_match_scores(edge_rows)
        _add_relation_match_scores(edge_rows_2)
        # for chunk in edge_rows_3.get("source_chunk", []):
        #     if chunk:
        #         add_score([chunk], 0.2)

        # 按分数降序排列提取 Top N
        sorted_chunks = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)
        top_chunk_ids = [c for c, _ in sorted_chunks[:top_chunks]]
        
        return {
            "chunks": top_chunk_ids,
            "chunk_scores": dict(sorted_chunks[:top_chunks]), # 返回分数供后续 RRF 使用
            "matched_entities": matched,
        }

    def _parse_entities_from_llm(self, raw_text: str) -> Dict[str, List]:
        if not raw_text:
            return {"entities": [], "relations": []}
        raw_text = raw_text.strip()
        if raw_text.startswith("```"):
            raw_text = raw_text.strip("`")
            if raw_text.lower().startswith("json"):
                raw_text = raw_text[4:]
        start = raw_text.find("{")
        end = raw_text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return {"entities": [], "relations": []}
        try:
            payload = json.loads(raw_text[start : end + 1])
        except json.JSONDecodeError:
            return {"entities": [], "relations": []}
        entities = payload.get("entities", []) or []
        relations_raw = (
            payload.get("relations")
            or payload.get("relationship")
            or payload.get("relationships")
            or []
        )
        relations: List[str] = []
        if isinstance(relations_raw, str):
            relations = [r.strip() for r in re.split(r"[;,]", relations_raw) if r.strip()]
        elif isinstance(relations_raw, list):
            for rel in relations_raw:
                if isinstance(rel, str):
                    if rel.strip():
                        relations.append(rel.strip())
                elif isinstance(rel, dict):
                    value = rel.get("rel") or rel.get("relation") or rel.get("text")
                    if value:
                        relations.append(str(value).strip())
        return {"entities": entities, "relations": relations}

    def _get_chunk_text(
        self, chunk_ids: List[str]
    ) -> List[str]:
        chunk_text = []
        for chunk_id in chunk_ids:
            text_info = self.vector_store.get_chunk_text(chunk_id)
            if not text_info or not text_info.get("text"):
                print(f"Warning: no chunk_text found for chunk_id={chunk_id}")
                chunk_text.append("")
                continue
            chunk_text.append(text_info.get("text", ""))
        return chunk_text

    def _rerank_score(self, query: str, passage: str) -> float:
        if not self.reranker or not passage:
            return 0.0
        if hasattr(self.reranker, "score"):
            return float(self.reranker.score(query, passage))
        return 0.0

    def hybrid_retrieve(self, query: str, topk: int = 5, top_entities: int = 5, top_chunks: int = 10, top_rerank: int = 10):
        # ==========================================
        # 1. 向量检索路 (Vector Retrieval)
        # ==========================================
        query_emb = self._embed_text(query)
        embedding_res = self.retrieve_from_embedding(query_emb, topk=topk)
        vector_hits = embedding_res.get("embedding_hits", []) # [{"id": "chunk_xxx", "score": 0.8}, ...]

        # ==========================================
        # 2. 图检索路 (Graph Retrieval)
        # ==========================================
        raw_response = self.llm.complete(prompt=prompt_extract_entities_str.format(context=query))
        parsed = self._parse_entities_from_llm(raw_response)
        extracted_entities = parsed.get("entities", [])
        query_relations = parsed.get("relations", [])
        
        memory_res = []
        persistent_res = []
        graph_chunk_scores_aggregated = {} # 用于汇总图检索分数
        # chunk_entity_coverage = {} # 新增：记录 Chunk 被几个不同的实体检索到

        for ent in extracted_entities:
            # 检索内存图 
            m_res = self.retrieve_from_memory_graph(
                query_emb,
                ent["id"],
                ent["type"],
                ent["desc"],
                top_entities,
                top_chunks,
                query_relations=query_relations,
            )
            memory_res.append(m_res)           
            
            # 检索持久化图
            p_res = self.retrieve_from_persistent_graph(
                query_emb,
                ent["id"],
                ent["type"],
                ent["desc"],
                top_entities,
                top_chunks,
                query_relations=query_relations,
            )
            persistent_res.append(p_res)

            # 汇总图的chunk分数
            for cid, score in m_res.get("chunk_scores", {}).items():
                graph_chunk_scores_aggregated[cid] = graph_chunk_scores_aggregated.get(cid, 0.0) + score
                # chunk_entity_coverage.setdefault(cid, set()).add(ent["id"]) # 记录该 chunk 关联的实体

            for cid, score in p_res.get("chunk_scores", {}).items():
                graph_chunk_scores_aggregated[cid] = graph_chunk_scores_aggregated.get(cid, 0.0) + score
                # chunk_entity_coverage.setdefault(cid, set()).add(ent["id"])
            
            # [保持原有的晋升逻辑触发...]
            for cid in m_res.get("chunks", []):
                triggered, data = self.memory_graph.access_chunk(cid)
                if triggered:
                    self.memory_graph.write_to_persistent_graph([data])
        
        # # 相交奖励计算
        # for cid in graph_chunk_scores_aggregated:
        #     coverage_count = len(chunk_entity_coverage[cid])
        #     if coverage_count > 1:
        #         # 如果一个 chunk 被 2 个及以上 Query 实体共同引出，说明路径相交，给予强烈提权
        #         graph_chunk_scores_aggregated[cid] *= (1.5 ** (coverage_count - 1))
        
        # 将图检索结果按分数排序，得到图检索 Rank
        graph_ranked_chunks = sorted(graph_chunk_scores_aggregated.items(), key=lambda x: x[1], reverse=True)

        # ==========================================
        # 3. RRF 融合打分 (Reciprocal Rank Fusion)
        # ==========================================
        rrf_scores = {}
        k = 60 # RRF 的平滑常数，业界通常设为 60

        # 加入向量检索排名
        for rank, hit in enumerate(vector_hits):
            cid = hit["id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k + rank + 1)

        # 加入图检索排名
        for rank, (cid, _) in enumerate(graph_ranked_chunks):
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k + rank + 1)

        # 得到融合后的候选池 (选取 top_rerank 的 2~3 倍送给 Re-ranker)
        candidate_pool_size = top_rerank * 2
        fused_candidates = [cid for cid, score in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:candidate_pool_size]]

        # ==========================================
        # 4. Re-ranker 重排层 (最后防线)
        # ==========================================
        if not fused_candidates:
            final_chunks = []
        else:
            # 准备待重排的文本
            # 根据 chunk_id 获取原始文本：self._get_chunk_text(cid)
            candidate_texts = self._get_chunk_text(fused_candidates)

            # [模型调用点]：调用轻量级 Cross-Encoder 进行重排
            reranker_scores = [
                self._rerank_score(query, text) for text in candidate_texts
            ]

            # 根据 Re-ranker 分数倒序排列
            reranked_pairs = sorted(zip(fused_candidates, reranker_scores), key=lambda x: x[1], reverse=True)
            
            # 截取最终的 Top-K 交给大模型生成
            final_chunks = [cid for cid, score in reranked_pairs[:top_rerank]]

        return {
            "chunks": final_chunks,
            "embedding": embedding_res,
            "memory": memory_res,
            "persistent": persistent_res,
        }
