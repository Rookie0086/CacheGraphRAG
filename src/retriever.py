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
from utils.llm_env import LLMEnv, OllamaEnv  
from database.milvus import MilvusDB, myMilvus
from database.nebulagraph import NebulaClient, NebulaDB
from langchain_text_splitters import RecursiveCharacterTextSplitter

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
        self.reranker = reranker or OllamaEnv(
            model="dengcao/bge-reranker-v2-m3:latest",
            base_url="http://localhost:11434",
            timeout=1000,
            temperature=0,
            max_tokens=16,
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
    
    def retrieve_from_embedding(self, query: str, topk: int = 5):
        query_emb = self._embed_text(query)
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

    def retrieve_from_memory_graph(self, entity_name: str, entity_type: str, entity_desc: str, top_entities: int = 5, top_chunks: int = 5):
        if not self.memory_graph:
            return {"chunks": [], "matched_entities": []}
        if not entity_name and not entity_desc:
            return {"chunks": [], "matched_entities": []}

        graph = self.memory_graph.graph
        vector_store = MilvusDB(db_name=self.entity_index_name, overwrite=False)
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
        base_nodes = set(node_set)
        first_hop = set()
        for uid in list(node_set):
            first_hop.update(graph.predecessors(uid))
            first_hop.update(graph.successors(uid))
        node_set.update(first_hop)
        # 扩展成2-hop邻居集合，统计相关 chunk 频次
        second_hop = set()
        for uid in list(first_hop):
            second_hop.update(graph.predecessors(uid))
            second_hop.update(graph.successors(uid))
        node_set.update(second_hop)
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
            add_score(data.get("source_chunks"), 0.1)

        # Edges: weight by hop proximity
        for u, v, _, data in subgraph.edges(data=True, keys=True):
            chunk = data.get("source_chunk")
            if not chunk:
                continue
            if u in base_nodes or v in base_nodes:
                add_score([chunk], 0.5)
            elif u in first_hop or v in first_hop:
                add_score([chunk], 0.1)

        sorted_chunks = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)
        top_chunk_ids = [c for c, _ in sorted_chunks[:top_chunks]]
        return {
            "chunks": top_chunk_ids,
            "chunk_scores": dict(sorted_chunks[:top_chunks]),
            "matched_entities": matched,
        }

    def retrieve_from_persistent_graph(self, entity_name: str, entity_type: str, entity_desc: str, top_entities: int = 5, top_chunks: int = 5):
        if not self.persistent_graph:
            return {"chunks": [], "matched_entities": []}
        if not entity_name and not entity_desc:
            return {"chunks": [], "matched_entities": []}

        graph = self.persistent_graph
        vector_store = MilvusDB(db_name=self.entity_index_name, overwrite=False)
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

        valid_vids_str = ", ".join([graph._format_vid(v) for v in existing_vids])
        # GO 1 STEPS 拉 1-hop 边与邻居节点，统计 source_chunk 频次
        go_query = (
            "GO 1 STEPS FROM "
            f"{valid_vids_str} "
            "OVER `relationship` BIDIRECT "
            "YIELD src(edge) AS src, dst(edge) AS dst, "
            "properties(edge).relationship AS relation, "
            "properties(edge).source_chunk AS source_chunk, "
            "properties($$).source_chunk AS dst_source_chunk;"
        )
        try:
            edge_rows = graph.query(go_query)
        except Exception:
            print("Error occurred while fetching edges.")
            edge_rows = {}

        # GO 2 STEPS 拉 2-hop 邻居边与节点，扩展统计 source_chunk 频次
        go_query_2 = (
            "GO 2 STEPS FROM "
            f"{valid_vids_str} "
            "OVER `relationship` BIDIRECT "
            "YIELD src(edge) AS src, dst(edge) AS dst, "
            "properties(edge).relationship AS relation, "
            "properties(edge).source_chunk AS source_chunk, "
            "properties($$).source_chunk AS dst_source_chunk;"
        )
        try:
            edge_rows_2 = graph.query(go_query_2)
        except Exception:
            print("Error occurred while fetching 2-hop edges.")
            edge_rows_2 = {}

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

        # 2. 1-hop 关联的边与目标节点 Chunk：次级依据，给予权重 0.5
        for chunk in edge_rows.get("source_chunk", []):
            if chunk: add_score([chunk], 0.5)
        for chunks in edge_rows.get("dst_source_chunk", []):
            add_score(chunks, 0.5)

        # 3. 2-hop 关联的边与目标节点 Chunk：极易引入噪声，给予极低权重 0.1
        for chunk in edge_rows_2.get("source_chunk", []):
            if chunk: add_score([chunk], 0.1)
        for chunks in edge_rows_2.get("dst_source_chunk", []):
            add_score(chunks, 0.1)

        # 按分数降序排列提取 Top N
        sorted_chunks = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)
        top_chunk_ids = [c for c, _ in sorted_chunks[:top_chunks]]
        
        return {
            "chunks": top_chunk_ids,
            "chunk_scores": dict(sorted_chunks[:top_chunks]), # 返回分数供后续 RRF 使用
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
        if not passage:
            return 0.0
        prompt = (
            "You are a reranker. Given a query and a passage, output a single floating point "
            "relevance score between 0 and 1. Output only the number.\n"
            f"Query: {query}\n"
            f"Passage: {passage}\n"
            "Score:"
        )
        response = self.reranker.complete(prompt) if self.reranker else ""
        if not response:
            return 0.0
        match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", response)
        if not match:
            return 0.0
        try:
            score = float(match.group(0))
        except ValueError:
            return 0.0
        return max(0.0, min(1.0, score))

    def hybrid_retrieve(self, query: str, topk: int = 5, top_entities: int = 5, top_chunks: int = 10, top_rerank: int = 10):
        # ==========================================
        # 1. 向量检索路 (Vector Retrieval)
        # ==========================================
        embedding_res = self.retrieve_from_embedding(query, topk=topk)
        vector_hits = embedding_res.get("embedding_hits", []) # [{"id": "chunk_xxx", "score": 0.8}, ...]

        # ==========================================
        # 2. 图检索路 (Graph Retrieval)
        # ==========================================
        raw_response = self.llm.complete(prompt=prompt_extract_entities_str.format(context=query))
        extracted_entities = self._parse_entities_from_llm(raw_response)
        
        memory_res = []
        persistent_res = []
        graph_chunk_scores_aggregated = {} # 用于汇总图检索分数

        for ent in extracted_entities:
            # 检索内存图 
            m_res = self.retrieve_from_memory_graph(ent["id"], ent["type"], ent["desc"], top_entities, top_chunks)
            memory_res.append(m_res)
            
            # 检索持久化图
            p_res = self.retrieve_from_persistent_graph(ent["id"], ent["type"], ent["desc"], top_entities, top_chunks)
            persistent_res.append(p_res)

            # 汇总图的chunk分数
            for cid, score in m_res.get("chunk_scores", {}).items():
                graph_chunk_scores_aggregated[cid] = graph_chunk_scores_aggregated.get(cid, 0.0) + score
            for cid, score in p_res.get("chunk_scores", {}).items():
                graph_chunk_scores_aggregated[cid] = graph_chunk_scores_aggregated.get(cid, 0.0) + score

            # [保持原有的晋升逻辑触发...]
            for cid in m_res.get("chunks", []):
                triggered, data = self.memory_graph.access_chunk(cid)
                if triggered:
                    self.memory_graph.write_to_persistent_graph([data])

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
