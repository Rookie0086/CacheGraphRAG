import asyncio
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

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils import get_config
from utils.base import read_json, save_to_json
from triplet.prompts import prompt_extract_triplest_str, prompt_extract_entities_str
from utils.llm_env import LLMEnv  
from database.milvus import MilvusDB, myMilvus
from database.nebulagraph import NebulaClient, NebulaDB
from langchain_text_splitters import RecursiveCharacterTextSplitter

os.environ["MILVUS_FORCE_FLUSH"] = "1"
config = get_config()
model_name = "gpt-4o-mini"
api_key = config["model"]["OPENAI_API_KEY"]
base_url = config["model"]["OPENAI_BASE_URL"]
llm = LLMEnv(
    backend="openai", 
    model=model_name, 
    api_key=api_key, 
    base_url=base_url
    )


class MemoryGraphManager:
    def __init__(self, promotion_threshold=3, chunk_vector_store=None):
        # 使用 MultiDiGraph 支持同节点间的多种/多来源关系
        self.graph = nx.MultiDiGraph()
        self.persistent_graph = NebulaDB(space_name="example") # 连接 NebulaGraph，持久化存储内存图数据
        # 记录每个 chunk 的被检索/访问次数: {chunk_id: count}
        self.chunk_access_counter = {}
        self.threshold = promotion_threshold
        self._id_index = {}
        self.chunk_vector_store = chunk_vector_store

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
            if isinstance(uid, int):
                return uid
            try:
                return int(uid)
            except (ValueError, TypeError):
                return uid

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
            node_set = set(entity_ids)
            for uid in list(node_set):
                if self.graph.has_node(uid):
                    promoted_nodes_dict[uid] = self.graph.nodes[uid]

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

        return {
            "chunk_id": chunk_id,
            "nodes": promoted_nodes_dict,
            "edges": promoted_edges
        }

    def write_to_persistent_graph(self, subgraph_data: List[Dict]):
        """将晋升的子图写入 NebulaGraph"""
        for data in subgraph_data:
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
                self.persistent_graph.upsert_vertex(
                    vertex_id=uid,
                    properties={
                        "name": data.get("name", ""),
                        "type": data.get("type", ""),
                        "source_chunks": _format_chunks(data.get("source_chunks"))
                    }
                )

            # 2. 写入边
            for edge in edges:
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


class AsyncEntityResolver:
    def __init__(
        self,   
        embedding_func,
        collection_name="entity_index", 
        threshold=0.85,
        memory_graph=None,
    ):
        self.milvus_db = MilvusDB(db_name=collection_name, overwrite=False)
        self.milvus_client = myMilvus()
        self.embed = embedding_func
        self.collection_name = collection_name
        self.threshold = threshold
        self.memory_graph = memory_graph
        
        if self.collection_name not in self.milvus_client.list_collections():
            self.milvus_db.create_entity_collection()
        else:
            self.milvus_db.load()
        # 本地级联缓存：减少对相同实体的重复 Embedding 和 数据库查询
        # 结构: { "entity_name:entity_desc": "global_uid" }
        self.local_cache = {}
        self._pending_tasks = []

    def _normalize_name(self, name: str) -> str:
        """
        1. 统一小写，去除首尾空格
        2. 剔除常见的商业组织后缀
        """
        if not name:
            return ""
        
        # 1. 基础清洗
        name = name.lower().strip()
        
        # 2. 定义无意义后缀正则 (注意处理边界 \b)
        # 覆盖：Inc, Corp, Co, Ltd, LLC, Group, Foundation 等
        suffixes = r'\b(inc|corp|co|ltd|llc|group|foundation|incorporated|corporation|limited)\b'
        
        # 移除后缀及可能存在的标点（如 Apple, Inc. -> apple）
        name = re.sub(rf'[,\.\s]+({suffixes})[,\.\s]*', '', name)
        name = re.sub(rf'[,\.\s]+$', '', name) # 清理结尾残余标点
        
        return name.strip()

    def _is_abbreviation_or_nickname(self, name1: str, name2: str) -> bool:
        """
        判断两个字符串是否互为缩写、简称或高相似度变形
        """
        n1 = self._normalize_name(name1)
        n2 = self._normalize_name(name2)
        
        if not n1 or not n2: return False
        if n1 == n2: return True

        # 1. 首字母缩写判断 (例如: "IBM" vs "International Business Machines")
        def check_acronym(short_str, long_str):
            # 提取长字符串中每个单词的首字母
            words = [w for w in re.split(r'[\s\-]+', long_str) if w]
            acronym = "".join([w[0] for w in words])
            return short_str == acronym

        short, long = (n1, n2) if len(n1) < len(n2) else (n2, n1)
        if check_acronym(short, long):
            return True

        # 2. 子串判断 (例如: "Tim Cook" vs "Timothy Cook")
        # 如果短字符串的所有 token 都是长字符串对应 token 的前缀
        short_tokens = short.split()
        long_tokens = long.split()
        if len(short_tokens) == len(long_tokens):
            if all(l_t.startswith(s_t) for s_t, l_t in zip(short_tokens, long_tokens)):
                return True

        # 3. 编辑距离相似度 (模糊匹配)
        # SequenceMatcher 计算 ratio: 2.0*M / T (M是匹配数, T是总长度)
        similarity = SequenceMatcher(None, n1, n2).ratio()
        if similarity > 0.85: # 高相似度阈值
            return True

        return False

    # def _normalize_name(self, name: str) -> List[str]:
    #     if not name:
    #         return []
    #     cleaned = re.sub(r"[^a-zA-Z0-9\s]", " ", str(name)).lower()
    #     return [token for token in cleaned.split() if token]

    def _normalize_type(self, entity_type: str) -> str:
        if not entity_type:
            return ""
        return str(entity_type).strip().lower()

    async def _desc_similarity(self, left: str, right: str) -> float:
        left_norm = self._normalize_name(left)
        right_norm = self._normalize_name(right)
        left_tokens = set(left_norm.split())
        right_tokens = set(right_norm.split())
        if not left_tokens or not right_tokens:
            return 0.0

        token_sim = len(left_tokens & right_tokens) / len(left_tokens | right_tokens)

        try:
            left_vec, right_vec = await asyncio.gather(
                self.embed(left),
                self.embed(right),
            )
        except Exception:
            return token_sim

        if hasattr(left_vec, "tolist"):
            left_vec = left_vec.tolist()
        if hasattr(right_vec, "tolist"):
            right_vec = right_vec.tolist()

        left_arr = np.array(left_vec, dtype=float)
        right_arr = np.array(right_vec, dtype=float)
        denom = np.linalg.norm(left_arr) * np.linalg.norm(right_arr)
        if denom == 0:
            return token_sim
        cos_sim = float(np.dot(left_arr, right_arr) / denom)

        return 0.5 * token_sim + 0.5 * cos_sim

    # def _is_alias_pair(self, short_name: str, long_name: str) -> bool:
    #     short_tokens = self._normalize_name(short_name)
    #     long_tokens = self._normalize_name(long_name)
    #     if not short_tokens or not long_tokens:
    #         return False
    #     if short_tokens == long_tokens:
    #         return False
    #     if len(short_tokens) == 1:
    #         token = short_tokens[0]
    #         return any(t.startswith(token) and t != token for t in long_tokens)
    #     if len(short_tokens) != len(long_tokens):
    #         return False
    #     for idx, token in enumerate(short_tokens):
    #         if not long_tokens[idx].startswith(token):
    #             return False
    #     return True

    def _maybe_add_edge(self, src_uid: int, dst_uid: int, src_name: str = "", dst_name: str = "", relation_type: str = "") -> None:
        if not self.memory_graph:
            return
        if src_uid is None or dst_uid is None:
            return
        if src_name:
            self.memory_graph.add_node(src_uid, name=src_name, type="", source_chunk=relation_type)
        if dst_name:
            self.memory_graph.add_node(dst_uid, name=dst_name, type="", source_chunk=relation_type)
        self.memory_graph.add_edge(
            src_uid,
            dst_uid,
            relation_type=relation_type,
            source_chunk=relation_type,
        )

    def _make_uid(self) -> int:
        # Keep uid consistent with INT64 schema.
        return uuid.uuid4().int % (2**63 - 1)

    def _search_milvus(self, vector, search_params):
        if not self.milvus_db.db:
            self.milvus_db.load()
        return self.milvus_db.db.search(
            data=[vector],
            anns_field="vec",
            param=search_params,
            limit=1,
            output_fields=["uid", "name", "type", "desc"],
            consistency_level="Strong",
        )

    def _log_task_error(self, task: asyncio.Task):
        try:
            task.result()
        except asyncio.CancelledError:
            # Task cancelled during shutdown; ignore.
            return
        except Exception as exc:
            print(f"Error in background entity insert: {exc}")

    async def wait_pending(self):
        if not self._pending_tasks:
            return
        await asyncio.gather(*self._pending_tasks, return_exceptions=True)
        self._pending_tasks.clear()

    async def resolve_async(self, entity_name: str,entity_type: str, entity_desc: str) -> str:
        """
        核心对齐方法：输入实体名称、类型和描述，返回全局唯一的 UID。
        """
        cache_key = f"{entity_name}[{entity_type}]:{entity_desc}"
        
        # 1. 检查本地缓存 (O(1) 命中，极速返回)
        if cache_key in self.local_cache:
            return self.local_cache[cache_key]

        # 2. 异步获取向量 (不阻塞主线程)
        text_to_embed = f"Entity: {entity_name}. Type: {entity_type}. Description: {entity_desc}"
        vector = await self.embed(text_to_embed)

        # 3. 在 Milvus 中进行向量检索 (使用 to_thread 防止同步阻塞)
        search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
        
        results = await asyncio.to_thread(self._search_milvus, vector, search_params)

        # 4. 判定逻辑
        if results and len(results[0]) > 0:
            top_match = results[0][0]
            exist_uid = top_match.entity.get("uid")
            exist_name = top_match.entity.get("name")
            exist_type = top_match.entity.get("type")
            exist_desc = top_match.entity.get("desc")
            name_match = self._normalize_name(entity_name) == self._normalize_name(exist_name)
            type_match = self._normalize_type(entity_type) == self._normalize_type(exist_type)
            desc_sim = await self._desc_similarity(entity_desc, exist_desc)

            # 直接基于向量距离的高相似度判定
            if top_match.distance >= self.threshold:
                self.local_cache[cache_key] = exist_uid
                return exist_uid
            # name,type相同且描述相似度较高，认为是同一实体
            elif name_match and type_match and (desc_sim >= 0.5):
                self.local_cache[cache_key] = exist_uid
                return exist_uid
            # name，type相同且描述相似度较低，可能是描述侧重不同，生成新 UID 并建立 possible_same_as 关系
            elif name_match and type_match and desc_sim < 0.5:
                new_uid = self._make_uid()
                self.local_cache[cache_key] = new_uid
                self._maybe_add_edge(new_uid, exist_uid, src_name=entity_name, dst_name=exist_name, relation_type="possible_same_as")
                task = asyncio.create_task(self._register_new_entity(new_uid, entity_name, entity_type, entity_desc, vector))
                task.add_done_callback(self._log_task_error)
                self._pending_tasks.append(task)
                return new_uid
            # name相同但 type 不同，且描述相似度较高，认为是同一实体type提取偏差
            elif name_match and (not entity_type) and (not type_match) and desc_sim >= 0.5:
                self.local_cache[cache_key] = exist_uid
                return exist_uid
            # 简称或缩写判定
            elif type_match and (self._is_abbreviation_or_nickname(entity_name, exist_name) or self._is_abbreviation_or_nickname(exist_name, entity_name)):
                new_uid = self._make_uid()
                self.local_cache[cache_key] = new_uid
                self._maybe_add_edge(new_uid, exist_uid, src_name=entity_name, dst_name=exist_name, relation_type="alias_of")
                task = asyncio.create_task(self._register_new_entity(new_uid, entity_name, entity_type, entity_desc, vector))
                task.add_done_callback(self._log_task_error)
                self._pending_tasks.append(task)
                return new_uid

        # 5. 未命中：生成新实体 UID，并异步注册到 Milvus
        new_uid = self._make_uid()
        self.local_cache[cache_key] = new_uid
        
        # 触发异步写入，不等待其完成即可返回
        task = asyncio.create_task(self._register_new_entity(new_uid, entity_name, entity_type, entity_desc, vector))
        task.add_done_callback(self._log_task_error)
        self._pending_tasks.append(task)
        
        return new_uid

    async def _register_new_entity(self, uid: str, name: str, entity_type: str, entity_desc: str, vector: list):
        """后台异步将新实体写入 Milvus"""
        if hasattr(vector, "tolist"):
            vector = vector.tolist()
        data = [
            {"uid": uid, "name": name, "type": entity_type, "desc": entity_desc, "vec": vector}
        ]
        await asyncio.to_thread(
            self.milvus_db.insert,
            data
        )


class HybridRetriever:
    def __init__(self, vector_store=None, memory_graph=None, persistent_graph=None, llm=None, chunk_registry=None):
        self.vector_store = vector_store or MilvusDB(db_name="example", overwrite=False)
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

    def retrieve_from_memory_graph(self, entity_name: str, entity_desc: str, top_entities: int = 5, top_chunks: int = 5):
        if not self.memory_graph:
            return {"chunks": [], "matched_entities": []}
        if not entity_name and not entity_desc:
            return {"chunks": [], "matched_entities": []}

        graph = self.memory_graph.graph
        vector_store = MilvusDB(db_name="entity_index", overwrite=False)
        text_to_embed = f"Entity: {entity_name}. Description: {entity_desc}"
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
        print(f"Matched entity UIDs: {matched_id_set}")
        node_set = set()
        for mid in matched_id_set:
            node_set.update(self.memory_graph.get_nodes_by_id(mid))
        print(f"Initial matched nodes in graph: {node_set}")
        for uid in list(node_set):
            node_set.update(graph.predecessors(uid))
            node_set.update(graph.successors(uid))
        print(f"Expanded node set with neighbors: {node_set}")
        subgraph = graph.subgraph(node_set).copy()
        self.save_graph_gexf(f"subgraph/temp_subgraph_{entity_name}.gexf", subgraph)  # Debug: export subgraph to inspect structure 
        
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

    def retrieve_from_persistent_graph(self, entity_name: str, entity_desc: str, top_entities: int = 5, top_chunks: int = 5):
        if not self.persistent_graph:
            return {"chunks": [], "matched_entities": []}
        if not entity_name and not entity_desc:
            return {"chunks": [], "matched_entities": []}

        graph = self.persistent_graph
        vector_store = MilvusDB(db_name="entity_index", overwrite=False)
        text_to_embed = f"Entity: {entity_name}. Description: {entity_desc}"
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
        print(f"Matched entity UIDs in persistent graph: {matched_ids}")
        vid_list = [graph._format_vid(uid) for uid in matched_ids if uid is not None]
        if not vid_list:
            return {"chunks": [], "matched_entities": []}
        print(f"Formatted VIDs for query: {vid_list}")
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
        embedding_res = self.retrieve_from_embedding(query, topk=topk)
        raw_response = self.llm.complete(prompt=prompt_extract_entities_str.format(context=query))
        extracted_entities = self._parse_entities_from_llm(raw_response)
        memory_res = []
        persistent_res = []
        # TODO: 考虑异步并发地检索多个实体以提升效率
        for ent in extracted_entities:
            memory_res.append(self.retrieve_from_memory_graph(ent["id"], ent["desc"], top_entities=top_entities, top_chunks=top_chunks))
            # TODO: 内存图中找到了相关实体后，是否还需要继续在持久化图中检索？
            persistent_res.append(self.retrieve_from_persistent_graph(ent["id"], ent["desc"], top_entities=top_entities, top_chunks=top_chunks))

        chunk_ids = []
        for res in memory_res:
            candidate_chunk_ids = res.get("chunks", [])
            chunk_ids.extend(candidate_chunk_ids)
            # TODO: 晋升改为离线托管
            # 根据内存图检索结果晋升达到门槛的 chunk 相关的子图
            
            # access_results = []
            # for cid in candidate_chunk_ids:
            #     triggered, data = self.memory_graph.access_chunk(cid)
            #     if triggered:
            #         access_results.append(data)
            # if access_results:
            #     self.memory_graph.write_to_persistent_graph(access_results)
            #     print("已晋升的 chunk_id 列表:", [data["chunk_id"] for data in access_results])

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
            "embedding": embedding_res,
            "memory": memory_res,
            "persistent": persistent_res,
        }
        


class DocumentIngestionPipeline:
    def __init__(
        self, 
        llm_client: LLMEnv,   # LLM 环境 (负责抽取和 Embedding)
        memory_graph,      # NetworkX 管理器
        entity_resolver,   # 负责与 Milvus 交互进行实体对齐
        collection_name="example",    # Milvus 客户端 (负责存储 Chunk)
        max_concurrency=10 # LLM API 并发限制
    ):
        self.llm = llm_client
        self.collection_name = collection_name
        self.vector_store = MilvusDB(db_name=collection_name, overwrite=False) # 直接在 Pipeline 内部管理 MilvusDB 实例
        self.vector_client = myMilvus() 
        self.memory_graph = memory_graph
        self.entity_resolver = entity_resolver
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.chunk_registry = {}

        if self.collection_name not in self.vector_client.list_collections():
            self.vector_store.create_chunk_collection()
        else:
            self.vector_store.load()
        
        # 初始化分块器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            length_function=len,
            separators=["\n\n", "\n", "。", ".", " ", ""]
        )

    async def process_document(self, doc_text: str, source_file: str):
        """处理单篇长文档的入口"""
        print(f"开始处理文档: {source_file}")
        
        # 1. 文本分块
        chunks = self.text_splitter.split_text(doc_text)
        print(f"文档被切分为 {len(chunks)} 个 Chunk.")

        # 2. 构建异步任务列表
        tasks = []
        for text in chunks:
            chunk_id = f"chunk_{uuid.uuid4().hex[:12]}"
            self.chunk_registry[chunk_id] = text
            tasks.append(self.process_single_chunk(chunk_id, text))

        # 3. 并发执行所有 Chunk 处理任务
        # asyncio.gather 会等待所有任务完成，并返回结果列表
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 4. 错误处理与统计
        success_count = sum(1 for r in results if r is True)
        print(f"文档 {source_file} 处理完成。成功: {success_count}/{len(chunks)}")
        # self.vector_store.db.flush()  # 确保所有数据写入 Milvus
        # print(f"已 flush 到 Milvus，文档 {source_file} 的数据现在可见。")

    async def process_single_chunk(self, chunk_id: str, text: str) -> bool:
        """核心处理逻辑：受 Semaphore 控制的异步方法"""
        async with self.semaphore:
            try:
                # ==========================================
                # 链路 A: LLM 抽取 -> 实体对齐 -> 存入 NetworkX
                # ==========================================
                # 1. 调用 LLM 进行抽取 (使用之前设计的严格 Schema)
                raw_json_str = await self.llm.async_complete(prompt=prompt_extract_triplest_str.format(context=text))
                print(f"Chunk {chunk_id} 的原始抽取结果: {raw_json_str[:200]}...") # 只打印前200字符预览
                entities, relations = self._clean_and_validate(raw_json_str)

                if not entities:
                    return True # 抽取为空，无需入图，但不算失败              

                # 2. 实体对齐 (Entity Resolution)
                # 将 LLM 抽取的临时 ID 转换为全局唯一 ID
                aligned_entities = {}
                for ent in entities:
                    # resolver 会去 Milvus 查重，返回全局 uid
                    uid = await self.entity_resolver.resolve_async(ent["id"], ent["type"], ent["desc"])
                    aligned_entities[ent["id"]] = {
                        "uid": uid, 
                        "name": ent["id"], 
                        "type": ent["type"], 
                        "desc": ent["desc"],
                    }

                # 3. 写入 NetworkX (Memory Graph)
                self._write_to_memory_graph(chunk_id, aligned_entities, relations)

                # ==========================================
                # 链路 B: 存入 Milvus (Chunk 向量检索库)
                # ==========================================
                # 1. 获取文本的 Embedding
                chunk_vector = await self.llm.embed_model.get_embedding_async(text)
                if hasattr(chunk_vector, "tolist"):
                    chunk_vector = chunk_vector.tolist()
                print(f"Chunk {chunk_id} 的向量维度: {len(chunk_vector)}")

                # 2. 存入 Milvus
                await self.vector_store.insert_chunk_async(
                    chunk_id=chunk_id, 
                    vector=chunk_vector, 
                    entity_uids=[ent["uid"] for ent in aligned_entities.values()],
                )
                print(f"Chunk {chunk_id} 已加入 Milvus 数据流。")
                # self.vector_store.db.flush()  # 确保数据可见
                # print(f"Chunk {chunk_id} 已 flush 到 Milvus。")
                return True

            except Exception as e:
                print(f"处理 Chunk {chunk_id} 时出错: {str(e)}")
                return False

    def _clean_and_validate(self, raw_str: str):
        """清洗 LLM 输出，过滤悬空关系"""
        try:
            if not raw_str:
                return [], []
            raw_str = raw_str.strip()
            if raw_str.startswith("```"):
                raw_str = raw_str.strip("`")
                if raw_str.lower().startswith("json"):
                    raw_str = raw_str[4:]
            start = raw_str.find("{")
            end = raw_str.rfind("}")
            if start == -1 or end == -1 or end <= start:
                return [], []
            data = json.loads(raw_str[start : end + 1])
            entities = data.get("entities", [])
            relations = data.get("relations", [])
            
            valid_ids = {e["id"] for e in entities}
            valid_relations = [
                r for r in relations 
                if r.get("src") in valid_ids and r.get("tgt") in valid_ids
            ]
            return entities, valid_relations
        except json.JSONDecodeError:
            return [], []

    def _write_to_memory_graph(self, chunk_id: str, aligned_entities: Dict, relations: List[Dict]):
        """将对齐后的实体和关系写入 NetworkX，并强制绑定 chunk_id"""
        # 注意：NetworkX 操作是内存操作，速度极快，无需 async
        
        # 1. 写入节点
        for original_id, ent_data in aligned_entities.items():
            uid = ent_data["uid"]
            # 如果节点已存在，只需追加 source_chunk；这里为了简化，每次覆盖属性或累加
            self.memory_graph.add_node(
                uid, 
                name=ent_data["name"], 
                type=ent_data["type"], 
                source_chunk=chunk_id # 核心绑定
            )

        # 2. 写入边
        for rel in relations:
            src_original = rel["src"]
            tgt_original = rel["tgt"]
            
            # 获取全局对齐后的 ID
            src_uid = aligned_entities[src_original]["uid"]
            tgt_uid = aligned_entities[tgt_original]["uid"]
            
            self.memory_graph.add_edge(
                src_uid, 
                tgt_uid, 
                relation_type=rel["rel"],
                source_chunk=chunk_id # 核心绑定
            )

# ==========================================
# 2. 运行测试流程
# ==========================================

async def run_tests():
    # 初始化你的文档文本 (使用你提供的 example.txt 的片段)
    # with open("data/example.txt", "r") as f:
    #     document_text = f.read()

    # print("🚀 --- [阶段 1: 文档入库处理 (Ingestion)] ---")
    mem_graph = MemoryGraphManager(promotion_threshold=2) # 测试环境阈值设低一点：2次
    resolver = AsyncEntityResolver(
        embedding_func=llm.embed_model.get_embedding_async,
        memory_graph=mem_graph,
    )

    pipeline = DocumentIngestionPipeline(
        llm_client=llm, 
        memory_graph=mem_graph, 
        entity_resolver=resolver,
    )
    mem_graph.chunk_vector_store = pipeline.vector_store
    
    await pipeline.process_document(document_text, source_file="data/example.txt")
    await resolver.wait_pending()
    print("文档入库处理完成！")
    # try:
    #     resolver.milvus_db.flush()
    # except MilvusException as e:
    #     print(f"Warning: entity_index flush failed: {e}")

    try:
        print(resolver.milvus_client.get_collection_stats("entity_index"))
    except MilvusException as e:
        print(f"Warning: Failed to read collection stats: {e}")

    mem_graph.show_status()
    mem_graph.save_graph_graphml("subgraph/memory_graph_3.graphml")
    mem_graph.save_graph_gexf("subgraph/memory_graph_3.gexf")
    # mem_graph.load_graph_gexf("subgraph/memory_graph.gexf")
    # mem_graph.show_status()
    # print("\n🚀 --- [阶段 2: 检索与子图晋升 (Retrieval & Promotion)] ---")
    
    # qa_file = "data/example_qa.json"
    # if os.path.exists(qa_file):
    #     data = read_json(qa_file)
    #     print(f"Loaded {len(data)} QA pairs from {qa_file}.")
    # else:
    #     data = []
    
    # querys = set([item["query"] for item in data])
    # answers = set([item["answer"] for item in data])
    # print(f"Unique Queries: {len(querys)}, Unique Answers: {len(answers)}")
    # assert len(querys) == len(answers), "QA pairs should be 1-to-1."
    # retriever = HybridRetriever(
    #     # vector_store=resolver.milvus_db, 
    #     memory_graph=mem_graph, 
    #     llm=llm, 
    #     chunk_registry=pipeline.chunk_registry
    # )
    # data = []
    # for i,(query, answer) in tqdm(enumerate(zip(querys, answers)), total=len(querys)):
    #     print(f"\n🔍 Query {i+1}: {query}")
    #     retrieval_res = retriever.hybrid_retrieve(query, topk=2, top_entities=3, top_chunks=3)
    #     data.append({
    #         "query": query,
    #         "answer": answer,
    #         "retrieval": retrieval_res,
    #     })
    #     save_to_json(file_path="data/retrieval_results.json", data=data, indent=2, info=False)

    # mem_graph.show_status()

if __name__ == "__main__":
    asyncio.run(run_tests())

    # python -m tests.test_framework