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
class MemoryGraphManager:
    def __init__(self,
    space_name="rgb_en", 
    promotion_threshold=3, 
    chunk_vector_store=None
    ):
        # 使用 MultiDiGraph 支持同节点间的多种/多来源关系
        self.graph = nx.MultiDiGraph()
        if space_name not in NebulaClient().show_space():
            NebulaClient().create_graph_space(space_name)
        self.persistent_graph = NebulaDB(space_name=space_name) # 连接 NebulaGraph，持久化存储内存图数据
        # 记录每个 chunk 的被检索/访问次数: {chunk_id: count}
        self.chunk_access_counter = {}
        self.threshold = promotion_threshold
        self._id_index = {}
        self.chunk_vector_store = chunk_vector_store
        self.nebula_IO_count = 0

    def _rebuild_id_index(self):
        self._id_index = {}
        for node_id, data in self.graph.nodes(data=True):
            node_attr_id = data.get("Id")
            # 用节点 key 回填 Id 建立索引表
            if node_attr_id is None:
                data["Id"] = node_id
                node_attr_id = node_id
            key = str(node_attr_id)
            self._id_index.setdefault(key, set()).add(node_id)

    def get_nodes_by_id(self, node_id_value: str):
        # print(self._id_index)  # Debug: 打印当前 ID 索引状态
        return self._id_index.get(str(node_id_value), set())

    def _normalize_attrs_for_export(self, g: nx.MultiDiGraph) -> nx.MultiDiGraph:
        export_g = g.copy()
        for _, data in export_g.nodes(data=True):
            if "source_chunks" in data and isinstance(data["source_chunks"], set):
                data["source_chunks"] = list(data["source_chunks"])
        for _, _, _, data in export_g.edges(data=True, keys=True):
            if "source_chunk" in data and isinstance(data["source_chunk"], set):
                data["source_chunk"] = list(data["source_chunk"])
        return export_g

    def _normalize_attrs_for_graphml(self, g: nx.MultiDiGraph) -> nx.MultiDiGraph:
        export_g = g.copy()
        for _, data in export_g.nodes(data=True):
            if "source_chunks" in data and isinstance(data["source_chunks"], (set, list)):
                data["source_chunks"] = ",".join(str(v) for v in data["source_chunks"])
        for _, _, _, data in export_g.edges(data=True, keys=True):
            if "source_chunk" in data and isinstance(data["source_chunk"], (set, list)):
                data["source_chunk"] = ",".join(str(v) for v in data["source_chunk"])
        return export_g

    def _ensure_parent_dir(self, path: str):
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)

    def _restore_attrs_after_import(self, g: nx.MultiDiGraph) -> nx.MultiDiGraph:
        for _, data in g.nodes(data=True):
            if "source_chunks" in data and isinstance(data["source_chunks"], list):
                data["source_chunks"] = set(data["source_chunks"])
        for _, _, _, data in g.edges(data=True, keys=True):
            if "source_chunk" in data and isinstance(data["source_chunk"], list):
                data["source_chunk"] = set(data["source_chunk"])
        return g

    def save_graph_graphml(self, path: str):
        from networkx.readwrite.graphml import write_graphml_xml

        self._ensure_parent_dir(path)
        export_g = self._normalize_attrs_for_graphml(self.graph)
        write_graphml_xml(export_g, path)

    def load_graph_graphml(self, path: str):
        g = nx.read_graphml(path)
        if not isinstance(g, nx.MultiDiGraph):
            g = nx.MultiDiGraph(g)
        self.graph = self._restore_attrs_after_import(g)
        self._rebuild_id_index()

    def save_graph_gexf(self, path: str):
        self._ensure_parent_dir(path)
        export_g = self._normalize_attrs_for_graphml(self.graph)
        nx.write_gexf(export_g, path)

    def load_graph_gexf(self, path: str):
        g = nx.read_gexf(path)
        if not isinstance(g, nx.MultiDiGraph):
            g = nx.MultiDiGraph(g)
        self.graph = self._restore_attrs_after_import(g)
        self._rebuild_id_index()

    def add_node(self, uid: str, name: str, type: str, source_chunk: str):
        """添加节点。如果已存在，则追加 source_chunk"""
        if self.graph.has_node(uid):
            # 节点已存在属性不缺失，更新来源集合
            # 节点存在但属性缺失时，补全属性
            if "source_chunks" not in self.graph.nodes[uid]:
                self.graph.nodes[uid]["source_chunks"] = set()
            self.graph.nodes[uid]["source_chunks"].add(source_chunk)
            if not self.graph.nodes[uid].get("name") and name:
                self.graph.nodes[uid]["name"] = name
            if not self.graph.nodes[uid].get("type") and type:
                self.graph.nodes[uid]["type"] = type
        else:
            # 新节点
            self.graph.add_node(
                uid, 
                name=name, 
                type=type, 
                source_chunks={source_chunk} # 使用集合存储
            )
        node_attr_id = self.graph.nodes[uid].get("Id")
        if node_attr_id is not None:
            key = str(node_attr_id)
            self._id_index.setdefault(key, set()).add(uid)

    def add_edge(self, src_uid: str, tgt_uid: str, relation_type: str, source_chunk: str):
        """添加关系。每条边强绑定一个 source_chunk"""
        self.graph.add_edge(
            src_uid, 
            tgt_uid, 
            key=f"{relation_type}_{source_chunk}", # 确保多重边的唯一性
            relation=relation_type,
            source_chunk=source_chunk
        )

    def access_chunk(self, chunk_id: str) -> Tuple[bool, Dict]:
        """
        当检索命中该 chunk 时调用。
        返回: (是否触发晋升, 晋升的子图数据)
        """
        # 1. 计数累加
        self.chunk_access_counter[chunk_id] = self.chunk_access_counter.get(chunk_id, 0) + 1
        current_count = self.chunk_access_counter[chunk_id]

        # 2. 检查阈值
        if current_count == self.threshold:
            # 达到阈值，提取子图
            subgraph_data = self._extract_subgraph_by_chunk(chunk_id)
            return True, subgraph_data
            
        return False, {}

    def _extract_subgraph_by_chunk(self, chunk_id: str) -> Dict:
        """根据 chunk_id 提取相关联的实体和关系，准备写入 NebulaGraph"""
        promoted_edges = []
        promoted_nodes_dict = {}

        def _normalize_uid(uid):
            if uid is None:
                return None
            return str(uid).strip()

        def _coerce_graph_uid(uid):
            if uid is None:
                return None
            if self.graph.has_node(uid):
                return uid
            uid_str = str(uid)
            if self.graph.has_node(uid_str):
                return uid_str
            try:
                uid_int = int(uid)
            except (ValueError, TypeError):
                return None
            if self.graph.has_node(uid_int):
                return uid_int
            return None

        def _get_chunk_entity_ids():
            if not self.chunk_vector_store:
                return []
            try:
                entity_ids = self.chunk_vector_store.get_chunk_entities(chunk_id)
            except Exception:
                return []
            return [_normalize_uid(uid) for uid in entity_ids]

        entity_ids = _get_chunk_entity_ids()
        if entity_ids:
            node_set = set()
            for uid in entity_ids:
                uid_key = _coerce_graph_uid(uid)
                if uid_key is not None:
                    node_set.add(uid_key)
                    promoted_nodes_dict[uid_key] = self.graph.nodes[uid_key]

            seen_edges = set()
            for uid in list(node_set):
                for u, v, key, data in self.graph.out_edges(uid, data=True, keys=True):
                    if data.get("source_chunk") != chunk_id:
                        continue
                    edge_key = (u, v, key)
                    if edge_key in seen_edges:
                        continue
                    seen_edges.add(edge_key)
                    promoted_edges.append({"src": u, "tgt": v, "relation": data["relation"]})
                    if u not in promoted_nodes_dict:
                        promoted_nodes_dict[u] = self.graph.nodes[u]
                    if v not in promoted_nodes_dict:
                        promoted_nodes_dict[v] = self.graph.nodes[v]
                for u, v, key, data in self.graph.in_edges(uid, data=True, keys=True):
                    if data.get("source_chunk") != chunk_id:
                        continue
                    edge_key = (u, v, key)
                    if edge_key in seen_edges:
                        continue
                    seen_edges.add(edge_key)
                    promoted_edges.append({"src": u, "tgt": v, "relation": data["relation"]})
                    if u not in promoted_nodes_dict:
                        promoted_nodes_dict[u] = self.graph.nodes[u]
                    if v not in promoted_nodes_dict:
                        promoted_nodes_dict[v] = self.graph.nodes[v]
            # print(f"Chunk {chunk_id} has {len(promoted_nodes_dict)} nodes and {len(promoted_edges)} edges promoted.")
            return {
                "chunk_id": chunk_id,
                "nodes": promoted_nodes_dict,
                "edges": promoted_edges,
            }

        # 遍历所有边，筛选属于该 chunk 的边
        for u, v, key, data in self.graph.edges(data=True, keys=True):
            if data.get("source_chunk") == chunk_id:
                # 记录边
                promoted_edges.append({
                    "src": u,
                    "tgt": v,
                    "relation": data["relation"]
                })
                # 记录关联的节点 (防止重复)
                if u not in promoted_nodes_dict:
                    promoted_nodes_dict[u] = self.graph.nodes[u]
                if v not in promoted_nodes_dict:
                    promoted_nodes_dict[v] = self.graph.nodes[v]
        # print("Use edge-based promotion for chunk", chunk_id)
        return {
            "chunk_id": chunk_id,
            "nodes": promoted_nodes_dict,
            "edges": promoted_edges
        }

    def write_to_persistent_graph(self, subgraph_data: List[Dict]):
        """将晋升的子图写入 NebulaGraph"""
        for data in subgraph_data:
            self.nebula_IO_count += 1
            chunk_id = data["chunk_id"]
            nodes = data["nodes"]
            edges = data["edges"]

            def _format_chunks(source_chunks):
                if isinstance(source_chunks, str):
                    return source_chunks
                if isinstance(source_chunks, (set, list, tuple)):
                    parts = [str(c) for c in source_chunks if str(c)]
                    return ",".join(parts)
                return ""

            # 1. 写入节点
            for uid, data in nodes.items():
                # print(f"Upserting node {uid} to NebulaGraph with name: {data.get('name', '')}, type: {data.get('type', '')}, source_chunks: {_format_chunks(data.get('source_chunks', ''))}")
                source_chunks = data.get("source_chunks") or data.get("source_chunk")
                self.persistent_graph.upsert_vertex(
                    vertex_id=uid,
                    properties={
                        "name": data.get("name", ""),
                        "type": data.get("type", ""),
                        "source_chunk": _format_chunks(source_chunks)
                    }
                )

            # 2. 写入边
            for edge in edges:
                # print(f"Upserting edge from {edge['src']} to {edge['tgt']} with relation: {edge['relation']} and source_chunk: {chunk_id}")
                self.persistent_graph.upsert_edge(
                    src_id=edge["src"],
                    tgt_id=edge["tgt"],
                    relation=edge["relation"],
                    properties={
                        "source_chunk": chunk_id
                    }
                )
    
    def show_status(self):
        """展示逻辑：在控制台优雅地打印当前内存图的状态"""
        print("\n" + "="*40)
        print("🧠 [Memory Graph Status]")
        print("="*40)
        
        node_count = self.graph.number_of_nodes()
        edge_count = self.graph.number_of_edges()
        print(f"📊 规模: {node_count} 节点 | {edge_count} 边\n")
        
        print("🔥 [Chunk 访问频率排行]")
        # 按访问次数降序排序
        sorted_chunks = sorted(self.chunk_access_counter.items(), key=lambda x: x[1], reverse=True)
        if not sorted_chunks:
            print("   暂无数据")
        else:
            for cid, count in sorted_chunks[:5]: # 只展前5
                status = "✅ 已晋升" if count >= self.threshold else "⏳ 暂存中"
                print(f"   - {cid}: {count} 次 ({status})")

        print("\n🧩 [最新驻留实体示例 (Top 3)]")
        sample_nodes = list(self.graph.nodes(data=True))[:3]
        for uid, data in sample_nodes:
            chunks_str = ", ".join(list(data['source_chunks'])[:2])
            print(f"   - {data['name']} ({data['type']}) | 来源: [{chunks_str}...]")
        print("="*40 + "\n")    
