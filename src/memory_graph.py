"""L1 memory graph manager.

This module implements the L1 cache layer referenced in the paper:
- L1CachePolicy (LRU + TTL eviction): capacity bound C_max, TTL expiry, and
  LRU-ordered eviction of cold chunks (prune_if_needed / _evict_chunk_locked).
- rehydrate_chunk_from_milvus(): Lazy Topology Rehydration — rebuilds a chunk's
  subgraph topology from vector-store metadata without LLM calls.
- L2 promotion: subgraphs passing h(c) >= tau_hit are written to NebulaGraph.
"""

import asyncio
import time
import json
import math
import os
import re
import sys
import uuid
import threading
from collections import Counter, OrderedDict
import networkx as nx
import numpy as np
from typing import List, Dict, Tuple
from pymilvus.exceptions import MilvusException
from tqdm import tqdm
from difflib import SequenceMatcher
from src.utils import get_config
from src.utils.base import read_json, save_to_json
from src.utils.prompts import prompt_extract_triplest_str, prompt_extract_entities_str, prompt_answer_with_chunks_str
from src.llm.env import LLMEnv
from database.milvus import MilvusDB, myMilvus
from database.nebulagraph import NebulaClient, NebulaDB
from langchain_text_splitters import RecursiveCharacterTextSplitter


class MemoryGraphManager:
    def __init__(
    self,
    space_name="rgb_en", 
    promotion_threshold=3, 
    chunk_vector_store=None,
    capacity_limit=100,
    max_nodes=None,
    ttl_seconds=3600,
    prune_interval=5,
    enable_pruner=True,
    ):
        # Use MultiDiGraph to support multiple/multi-source relations between same nodes
        self.graph = nx.MultiDiGraph()
        if space_name not in NebulaClient().show_space():
            NebulaClient().create_graph_space(space_name)
        self.persistent_graph = NebulaDB(space_name=space_name) # Connect to NebulaGraph for persistent storage
        # Track retrieval/access count per chunk: {chunk_id: count}
        self.chunk_access_counter = {}
        self.entity_access_counter = {}
        self.threshold = promotion_threshold
        self._id_index = {}
        self.chunk_vector_store = chunk_vector_store
        self.nebula_IO_count = 0
        self.capacity_limit = capacity_limit
        self.max_nodes = max_nodes
        self.ttl_seconds = ttl_seconds
        self.prune_interval = prune_interval
        self.chunk_meta = {}
        self.chunk_nodes = {}
        self.chunk_edges = {}
        self.chunk_lru = OrderedDict()
        self.promoted_chunks = set()
        self.direct_write_nodes = set()
        self.direct_write_edges = set()
        self.direct_write_node_ops = 0
        self.direct_write_edge_ops = 0
        self.l2_written_nodes = set()
        self.l2_written_edges = set()
        self.l2_written_node_ops = 0
        self.l2_written_edge_ops = 0
        self.timing = {
            "extract": 0.0,
            "resolve": 0.0,
            "graph_write": 0.0,
            "milvus_insert": 0.0,
        }
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._pruner_thread = None
        if enable_pruner:
            self._start_pruner()

    def _start_pruner(self):
        if self._pruner_thread:
            return
        self._pruner_thread = threading.Thread(target=self._prune_loop, daemon=True)
        self._pruner_thread.start()

    def shutdown(self):
        if not self._pruner_thread:
            return
        self._stop_event.set()
        self._pruner_thread.join(timeout=2)

    def _prune_loop(self):
        while not self._stop_event.is_set():
            try:
                self.prune_if_needed()
            except Exception as exc:
                print(f"[MemoryGraph] prune error: {exc}")
            self._stop_event.wait(self.prune_interval)

    def _over_capacity_locked(self) -> bool:
        if self.capacity_limit and len(self.chunk_meta) > self.capacity_limit:
            return True
        if self.max_nodes and self.graph.number_of_nodes() > self.max_nodes:
            return True
        return False

    def _expired_chunks_locked(self, now_ts: float) -> List[str]:
        if not self.ttl_seconds:
            return []
        expired = []
        for chunk_id, meta in self.chunk_meta.items():
            last_ts = meta.get("last_access_time", 0)
            if last_ts and (now_ts - last_ts) > self.ttl_seconds:
                expired.append(chunk_id)
        return expired

    def _ensure_chunk_entry_locked(self, chunk_id: str, now_ts: float):
        if chunk_id not in self.chunk_meta:
            self.chunk_meta[chunk_id] = {
                "created_at": now_ts,
                "last_access_time": now_ts,
            }
        else:
            self.chunk_meta[chunk_id]["last_access_time"] = now_ts
        self.chunk_lru[chunk_id] = now_ts
        self.chunk_lru.move_to_end(chunk_id)

    def _touch_chunk_entities_locked(self, chunk_id: str, now_ts: float):
        for uid in self.chunk_nodes.get(chunk_id, set()):
            if self.graph.has_node(uid):
                self.graph.nodes[uid]["last_access_time"] = now_ts
        for u, v, key in self.chunk_edges.get(chunk_id, set()):
            if self.graph.has_edge(u, v, key=key):
                self.graph[u][v][key]["last_access_time"] = now_ts

    def touch_chunk(self, chunk_id: str):
        if not chunk_id:
            return
        now_ts = time.time()
        with self._lock:
            self._ensure_chunk_entry_locked(chunk_id, now_ts)
            self._touch_chunk_entities_locked(chunk_id, now_ts)

    def record_timing(self, key: str, seconds: float):
        if not key or seconds <= 0:
            return
        with self._lock:
            self.timing[key] = self.timing.get(key, 0.0) + seconds

    def is_chunk_in_l1(self, chunk_id: str) -> bool:
        return chunk_id in self.chunk_meta

    def is_chunk_in_l2(self, chunk_id: str) -> bool:
        return chunk_id in self.promoted_chunks

    def _select_eviction_candidate_locked(self) -> str:
        for chunk_id in self.chunk_lru.keys():
            if self.chunk_access_counter.get(chunk_id, 0) == 0:
                return chunk_id
        for chunk_id in self.chunk_lru.keys():
            return chunk_id
        return ""

    def prune_if_needed(self):
        now_ts = time.time()
        with self._lock:
            expired = self._expired_chunks_locked(now_ts)
            for chunk_id in expired:
                self._evict_chunk_locked(chunk_id)
            while self._over_capacity_locked():
                candidate = self._select_eviction_candidate_locked()
                if not candidate:
                    break
                self._evict_chunk_locked(candidate)

    def _evict_chunk_locked(self, chunk_id: str):
        node_ids = self.chunk_nodes.get(chunk_id, set())
        for uid in node_ids:
            if not self.graph.has_node(uid):
                continue
            data = self.graph.nodes[uid]
            source_chunks = data.get("source_chunks", set())
            if isinstance(source_chunks, set) and chunk_id in source_chunks:
                source_chunks.discard(chunk_id)
            data["ref_count"] = max(0, int(data.get("ref_count", 0)) - 1)
            if data.get("ref_count", 0) <= 0 or not data.get("source_chunks"):
                node_attr_id = data.get("Id")
                if node_attr_id is not None:
                    key = str(node_attr_id)
                    if key in self._id_index:
                        self._id_index[key].discard(uid)
                        if not self._id_index[key]:
                            del self._id_index[key]
                self.graph.remove_node(uid)

        edge_keys = self.chunk_edges.get(chunk_id, set())
        for u, v, key in edge_keys:
            if not self.graph.has_edge(u, v, key=key):
                continue
            data = self.graph[u][v][key]
            data["ref_count"] = max(0, int(data.get("ref_count", 0)) - 1)
            if data.get("ref_count", 0) <= 0:
                self.graph.remove_edge(u, v, key=key)

        self.chunk_nodes.pop(chunk_id, None)
        self.chunk_edges.pop(chunk_id, None)
        self.chunk_meta.pop(chunk_id, None)
        if chunk_id in self.chunk_lru:
            self.chunk_lru.pop(chunk_id, None)

    def rehydrate_chunk_from_milvus(self, chunk_id: str) -> bool:
        if not self.chunk_vector_store or not chunk_id:
            return False
        graph_meta = self.chunk_vector_store.get_chunk_graph_meta(chunk_id)
        if not graph_meta:
            return False
        entities = graph_meta.get("entities") or []
        relations = graph_meta.get("relations") or []
        for ent in entities:
            uid = ent.get("uid")
            if uid is None:
                continue
            self.add_node(
                uid,
                name=ent.get("name", ""),
                type=ent.get("type", ""),
                source_chunk=chunk_id,
                desc=ent.get("desc", ""),
            )
        for rel in relations:
            src_uid = rel.get("src_uid") or rel.get("src")
            tgt_uid = rel.get("tgt_uid") or rel.get("tgt")
            rel_type = rel.get("relation") or rel.get("rel")
            if not src_uid or not tgt_uid or not rel_type:
                continue
            self.add_edge(
                src_uid,
                tgt_uid,
                relation_type=rel_type,
                source_chunk=chunk_id,
            )
        self.touch_chunk(chunk_id)
        self.prune_if_needed()
        return True

    def _rebuild_id_index(self):
        self._id_index = {}
        with self._lock:
            nodes = list(self.graph.nodes(data=True))
        for node_id, data in nodes:
            node_attr_id = data.get("Id")
            # Backfill Id from node key to build index table
            if node_attr_id is None:
                data["Id"] = node_id
                node_attr_id = node_id
            key = str(node_attr_id)
            self._id_index.setdefault(key, set()).add(node_id)

    def get_nodes_by_id(self, node_id_value: str):
        # print(self._id_index)  # Debug: print current ID index state
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
        # Backup existing file with timestamp if it exists
        if os.path.exists(path):
            from datetime import datetime
            base, ext = os.path.splitext(path)
            backup = f"{base}_{datetime.now():%Y%m%d_%H%M%S}{ext}"
            os.rename(path, backup)
            print(f"Backed up old graph snapshot: {backup}")
        self._ensure_parent_dir(path)
        self.graph.graph["chunk_access_counter"] = json.dumps(self.chunk_access_counter)
        self.graph.graph["entity_access_counter"] = json.dumps(self.entity_access_counter)
        export_g = self._normalize_attrs_for_graphml(self.graph)
        nx.write_gexf(export_g, path)

    def load_graph_gexf(self, path: str):
        g = nx.read_gexf(path)
        if not isinstance(g, nx.MultiDiGraph):
            g = nx.MultiDiGraph(g)
        self.graph = self._restore_attrs_after_import(g)
        self._rebuild_id_index()
        raw = self.graph.graph.get("chunk_access_counter", "{}")
        self.chunk_access_counter.update(json.loads(raw))
        raw = self.graph.graph.get("entity_access_counter", "{}")
        self.entity_access_counter.update(json.loads(raw))

    def add_node(self, uid: str, name: str, type: str, source_chunk: str, desc: str = ""):
        """Add a node. Appends source_chunk if node already exists."""
        now_ts = time.time()
        with self._lock:
            if source_chunk:
                self._ensure_chunk_entry_locked(source_chunk, now_ts)
                self.direct_write_node_ops += 1
                self.direct_write_nodes.add(uid)
            if self.graph.has_node(uid):
                if "source_chunks" not in self.graph.nodes[uid]:
                    self.graph.nodes[uid]["source_chunks"] = set()
                source_chunks = self.graph.nodes[uid]["source_chunks"]
                if source_chunk and source_chunk not in source_chunks:
                    source_chunks.add(source_chunk)
                    self.graph.nodes[uid]["ref_count"] = int(self.graph.nodes[uid].get("ref_count", 0)) + 1
                if not self.graph.nodes[uid].get("name") and name:
                    self.graph.nodes[uid]["name"] = name
                if not self.graph.nodes[uid].get("type") and type:
                    self.graph.nodes[uid]["type"] = type
                if not self.graph.nodes[uid].get("desc") and desc:
                    self.graph.nodes[uid]["desc"] = desc
                self.graph.nodes[uid]["last_access_time"] = now_ts
            else:
                self.graph.add_node(
                    uid,
                    name=name,
                    type=type,
                    desc=desc,
                    source_chunks={source_chunk} if source_chunk else set(),
                    ref_count=1 if source_chunk else 0,
                    last_access_time=now_ts,
                )
            if source_chunk:
                self.chunk_nodes.setdefault(source_chunk, set()).add(uid)
            node_attr_id = self.graph.nodes[uid].get("Id")
            if node_attr_id is not None:
                key = str(node_attr_id)
                self._id_index.setdefault(key, set()).add(uid)

    def add_edge(self, src_uid: str, tgt_uid: str, relation_type: str, source_chunk: str):
        """Add an edge. Each edge is bound to a source_chunk."""
        now_ts = time.time()
        key = f"{relation_type}_{source_chunk}" if source_chunk else f"{relation_type}_adhoc"
        with self._lock:
            if source_chunk:
                self._ensure_chunk_entry_locked(source_chunk, now_ts)
                self.direct_write_edge_ops += 1
                self.direct_write_edges.add((src_uid, tgt_uid, relation_type))
            edge_key = (src_uid, tgt_uid, key)
            edge_set = self.chunk_edges.setdefault(source_chunk, set()) if source_chunk else None
            if edge_set is not None and edge_key in edge_set:
                if self.graph.has_edge(src_uid, tgt_uid, key=key):
                    self.graph[src_uid][tgt_uid][key]["last_access_time"] = now_ts
                return
            if self.graph.has_edge(src_uid, tgt_uid, key=key):
                data = self.graph[src_uid][tgt_uid][key]
                data["ref_count"] = int(data.get("ref_count", 0)) + (1 if source_chunk else 0)
                data["last_access_time"] = now_ts
            else:
                self.graph.add_edge(
                    src_uid,
                    tgt_uid,
                    key=key,
                    relation=relation_type,
                    source_chunk=source_chunk,
                    ref_count=1 if source_chunk else 0,
                    last_access_time=now_ts,
                )
            if edge_set is not None:
                edge_set.add(edge_key)

    def promote_by_convergence(self, chunk_entity_coverage: dict, threshold: int = 3):
        """Convergence promotion: chunk hit by >= threshold different query entities -> promote to L2.

        Args:
            chunk_entity_coverage: {chunk_id: set(entity_uids)}
            threshold: minimum number of entities that must hit simultaneously for promotion
        """
        for cid, entity_set in chunk_entity_coverage.items():
            if len(entity_set) >= threshold:
                subgraph_data = self._extract_subgraph_by_chunk(cid)
                if subgraph_data.get("nodes"):
                    self.write_to_persistent_graph([subgraph_data])
                    self.promoted_chunks.add(cid)

    def access_chunk(self, chunk_id: str, increment: bool = True) -> Tuple[bool, Dict]:
        """
        Called when retrieval hits this chunk.
        Returns: (whether to trigger promotion, promoted subgraph data)
        """
        now_ts = time.time()
        with self._lock:
            self._ensure_chunk_entry_locked(chunk_id, now_ts)
            self._touch_chunk_entities_locked(chunk_id, now_ts)
            if increment:
                self.chunk_access_counter[chunk_id] = self.chunk_access_counter.get(chunk_id, 0) + 1
            current_count = self.chunk_access_counter.get(chunk_id, 0)

        if increment and current_count == self.threshold:
            subgraph_data = self._extract_subgraph_by_chunk(chunk_id)
            self.prune_if_needed()
            return True, subgraph_data

        self.prune_if_needed()
        return False, {}

    def _extract_subgraph_by_chunk(self, chunk_id: str) -> Dict:
        """Extract related entities and relations by chunk_id, prepare for NebulaGraph write"""
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

        # Iterate all edges, filter those belonging to this chunk
        for u, v, key, data in self.graph.edges(data=True, keys=True):
            if data.get("source_chunk") == chunk_id:
                # Record edge
                promoted_edges.append({
                    "src": u,
                    "tgt": v,
                    "relation": data["relation"]
                })
                # Record associated nodes (prevent duplicates)
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

    def promote_subgraph(self, nodes: dict, edges: list):
        """Directly promote specified entities and relations to NebulaGraph, bypassing chunk counter."""
        data = {"chunk_id": "", "nodes": nodes, "edges": edges}
        self.write_to_persistent_graph([data])

    def write_to_persistent_graph(self, subgraph_data: List[Dict]):
        """Write promoted subgraph to NebulaGraph"""
        for data in subgraph_data:
            self.nebula_IO_count += 1
            chunk_id = data["chunk_id"]
            if chunk_id:
                self.promoted_chunks.add(chunk_id)
            nodes = data["nodes"]
            edges = data["edges"]

            def _format_chunks(source_chunks):
                if isinstance(source_chunks, str):
                    return source_chunks
                if isinstance(source_chunks, (set, list, tuple)):
                    parts = [str(c) for c in source_chunks if str(c)]
                    return ",".join(parts)
                return ""

            # 1. Write nodes
            for uid, data in nodes.items():
                with self._lock:
                    self.l2_written_node_ops += 1
                    self.l2_written_nodes.add(uid)
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

            # 2. Write edges
            for edge in edges:
                with self._lock:
                    self.l2_written_edge_ops += 1
                    self.l2_written_edges.add(
                        (edge["src"], edge["tgt"], edge["relation"], chunk_id)
                    )
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
        """Display logic: gracefully print current memory graph state to console."""
        print("\n" + "="*40)
        print("🧠 [Memory Graph Status]")
        print("="*40)
        
        node_count = self.graph.number_of_nodes()
        edge_count = self.graph.number_of_edges()
        print(f"📊 Size: {node_count} nodes | {edge_count} edges\n")
        
        print("🔥 [Chunk Access Frequency Ranking]")
        # Sort by access count descending
        sorted_chunks = sorted(self.chunk_access_counter.items(), key=lambda x: x[1], reverse=True)
        if not sorted_chunks:
            print("   No data")
        else:
            for cid, count in sorted_chunks[:5]: # Show top 5
                status = "✅ Promoted" if count >= self.threshold else "⏳ Pending"
                print(f"   - {cid}: {count} hits ({status})")

        print("\n🧩 [Recent Resident Entities (Top 3)]")
        sample_nodes = list(self.graph.nodes(data=True))[:3]
        for uid, data in sample_nodes:
            source_chunks = data.get("source_chunks", [])
            if isinstance(source_chunks, set):
                source_chunks = list(source_chunks)
            chunks_str = ", ".join(list(source_chunks)[:2])
            print(f"   - {data['name']} ({data['type']}) | source: [{chunks_str}...]")
        print("="*40 + "\n")    
