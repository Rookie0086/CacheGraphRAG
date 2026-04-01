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

class HybridRetriever:
    def __init__(
        self, 
        vector_store=MilvusDB(db_name="example", overwrite=False), 
        memory_graph=None, 
        persistent_graph=None, 
        llm=None, 
        chunk_registry=None
        ):
        self.vector_store = vector_store
        self.memory_graph = memory_graph
        self.persistent_graph = self.memory_graph.persistent_graph 
        self.llm = llm
        self.chunk_registry = chunk_registry or {}

    def _embed_text(self, text: str) -> np.ndarray:
        if not self.llm:
            raise ValueError("LLM env is required for embeddings.")
        embedding = self.llm.embed_model.get_embedding(text)
        return np.array(embedding, dtype=float)

    def _cosine_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    def _extract_query_entities(self, query: str, max_entities: int = 6) -> List[str]:
        # Simple heuristic: use capitalized tokens and quoted phrases.
        entities = []
        for token in query.split("\""):
            if token.strip() and token.strip() not in entities and len(entities) < max_entities:
                entities.append(token.strip())

        if len(entities) >= max_entities:
            return entities[:max_entities]

        words = [w.strip(".,:;!?()[]{}\"") for w in query.split()]
        for word in words:
            if len(entities) >= max_entities:
                break
            if not word:
                continue
            if word[0].isupper() or word.isupper():
                if word not in entities:
                    entities.append(word)
        if not entities:
            entities = words[:max_entities]
        return entities[:max_entities]

        if not candidates:
            return []
        query_emb = self._embed_text(query)
        cand_embs = self.llm.embed_model.get_embeddings(candidates)
        cand_embs = np.array(cand_embs, dtype=float)
        sims = [self._cosine_sim(query_emb, emb) for emb in cand_embs]
        ranked = sorted(zip(candidates, sims), key=lambda x: x[1], reverse=True)
        return ranked[:topk]

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
    
    def retrieve_from_embedding(self, query: str, topk: int = 5):
        query_emb = self._embed_text(query)
        try:
            ids, distances = self.vector_store.search([query_emb.tolist()], limit=topk)
        except Exception:
            ids, distances = [], []
        hits = [
            {"id": str(pk), "score": float(dist)}
            for pk, dist in zip(ids, distances)
        ]
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

    def retrieve_from_memory_graph(self, entity_name: str, entity_type: str, entity_desc: str, top_entities: int = 5, top_chunks: int = 5):
        if not self.memory_graph:
            return {"chunks": [], "matched_entities": []}
        if not entity_name and not entity_desc:
            return {"chunks": [], "matched_entities": []}

        graph = self.memory_graph.graph
        vector_store = MilvusDB(db_name="entity_index", overwrite=False)
        text_to_embed = f"Entity: {entity_name}. Type: {entity_type}. Description: {entity_desc}"
        entity_emb = vector_store.embed_model.get_embedding(text_to_embed)
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
        for uid in list(node_set):
            node_set.update(graph.predecessors(uid))
            node_set.update(graph.successors(uid))
        # print(f"Expanded node set with neighbors: {node_set}")
        subgraph = graph.subgraph(node_set).copy()
        # self.save_graph_gexf(f"subgraph/temp_subgraph_{entity_name}.gexf", subgraph)  # Debug: export subgraph to inspect structure 
        
        counter = Counter()
        for _, data in subgraph.nodes(data=True):
            chunks = data.get("source_chunks")
            if isinstance(chunks, set):
                counter.update(chunks)
            elif isinstance(chunks, list):
                counter.update(chunks)

        for _, _, _, data in subgraph.edges(data=True, keys=True):
            chunk = data.get("source_chunk")
            if isinstance(chunk, str):
                counter.update([chunk])
            elif isinstance(chunk, (set, list)):
                counter.update(chunk)

        top_chunk_ids = [c for c, _ in counter.most_common(top_chunks)]
        return {
            "chunks": top_chunk_ids,
            "matched_entities": matched,
        }

    def retrieve_from_persistent_graph(self, entity_name: str, entity_type: str, entity_desc: str, top_entities: int = 5, top_chunks: int = 5):
        if not self.persistent_graph:
            return {"chunks": [], "matched_entities": []}
        if not entity_name and not entity_desc:
            return {"chunks": [], "matched_entities": []}

        graph = self.persistent_graph
        vector_store = MilvusDB(db_name="entity_index", overwrite=False)
        text_to_embed = f"Entity: {entity_name}. Type: {entity_type}. Description: {entity_desc}"
        entity_emb = vector_store.embed_model.get_embedding(text_to_embed)
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
            "properties(vertex).source_chunks AS source_chunks;"
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

        valid_vids_str = ", ".join([graph._format_vid(v) for v in existing_vids])
        # GO 1 STEPS 拉 1-hop 边，统计节点及相关联的边（不包含邻居节点）的 source_chunk 频次
        go_query = (
            "GO 1 STEPS FROM "
            f"{valid_vids_str} "
            "OVER `relationship` BIDIRECT "
            "YIELD src(edge) AS src, dst(edge) AS dst, "
            "properties(edge).relationship AS relation, "
            "properties(edge).source_chunk AS source_chunk;"
        )
        try:
            edge_rows = graph.query(go_query)
        except Exception:
            print("Error occurred while fetching edges.")
            edge_rows = {}

        counter = Counter()
        for chunks in node_rows.get("source_chunks", []):
            if isinstance(chunks, str):
                parts = [c.strip() for c in chunks.split(",") if c.strip()]
                counter.update(parts)

        for chunk in edge_rows.get("source_chunk", []):
            if isinstance(chunk, str) and chunk:
                counter.update([chunk])

        top_chunk_ids = [c for c, _ in counter.most_common(top_chunks)]
        return {
            "chunks": top_chunk_ids,
            "matched_entities": matched,
        }

    def _parse_entities_from_llm(self, raw_text: str) -> List[str]:
        if not raw_text:
            return []
        raw_text = raw_text.strip()
        if raw_text.startswith("```"):
            raw_text = raw_text.strip("`")
            if raw_text.lower().startswith("json"):
                raw_text = raw_text[4:]
        start = raw_text.find("{")
        end = raw_text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return []
        try:
            payload = json.loads(raw_text[start : end + 1])
        except json.JSONDecodeError:
            return []
        entities = payload.get("entities", [])
        return entities

    def hybrid_retrieve(self, query: str, topk: int = 5, top_entities: int = 5, top_chunks: int = 5):
        # embedding_res = self.retrieve_from_embedding(query, topk=topk)
        raw_response = self.llm.complete(prompt=prompt_extract_entities_str.format(context=query))
        extracted_entities = self._parse_entities_from_llm(raw_response)
        memory_res = []
        persistent_res = []
        # TODO: 考虑异步并发地检索多个实体以提升效率
        for ent in extracted_entities:
            memory_res.append(self.retrieve_from_memory_graph(ent["id"], ent["type"], ent["desc"], top_entities=top_entities, top_chunks=top_chunks))
            # TODO: 内存图中找到了相关实体后，是否还需要继续在持久化图中检索？(冲突解决策略)
            persistent_res.append(self.retrieve_from_persistent_graph(ent["id"], ent["type"], ent["desc"], top_entities=top_entities, top_chunks=top_chunks))

        chunk_ids = []
        for res in memory_res:
            candidate_chunk_ids = res.get("chunks", [])
            chunk_ids.extend(candidate_chunk_ids)
            # TODO: 晋升改为离线托管
            # 根据内存图检索结果晋升达到门槛的 chunk 相关的子图
            
            access_results = []
            for cid in candidate_chunk_ids:
                triggered, data = self.memory_graph.access_chunk(cid)
                if triggered:
                    access_results.append(data)
            if access_results:
                self.memory_graph.write_to_persistent_graph(access_results)
                # print("已晋升的 chunk_id 列表:", [data["chunk_id"] for data in access_results])

        for res in persistent_res:
            chunk_ids.extend(res.get("chunks", []))

        # Deduplicate while preserving order.
        seen = set()
        unique_chunks = []
        for cid in chunk_ids:
            if cid in seen:
                continue
            seen.add(cid)
            unique_chunks.append(cid)

        return {
            "chunks": unique_chunks,
            # "embedding": embedding_res,
            "memory": memory_res,
            "persistent": persistent_res,
        }
        
