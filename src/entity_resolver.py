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
        name = re.sub(rf'[,.\s]+({suffixes})[,.\s]*', '', name)
        name = re.sub(rf'[,.\s]+$', '', name) # 清理结尾残余标点

        # 标准化标点为空格并折叠空白
        name = re.sub(r"[^a-z0-9\s]", " ", name)
        name = re.sub(r"\s+", " ", name).strip()
        return name

    # 在 _normalize_name 的基础上再“切分成 token 列表”，并去掉称谓
    def _tokenize_name(self, name: str) -> List[str]:
        if not name:
            return []
        honorifics = {"dr", "mr", "mrs", "ms", "prof", "sir", "madam"}
        tokens = self._normalize_name(name).split()
        tokens = [t for t in tokens if t and t not in honorifics and len(t) > 1]
        return tokens

    def _is_abbreviation_or_nickname(self, name1: str, name2: str) -> bool:
        """
        判断两个字符串是否互为缩写、简称或高相似度变形
        """
        n1 = " ".join(self._tokenize_name(name1))
        n2 = " ".join(self._tokenize_name(name2))
        
        if not n1 or not n2:
            return False
        if n1 == n2:
            return True

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

        # 2.1 单个姓氏或名字匹配 (例如: "Bush" vs "Sophia Bush")
        if len(short_tokens) == 1 and len(long_tokens) >= 2:
            token = short_tokens[0]
            if token == long_tokens[-1]:
                return True
            if token in long_tokens and len(token) >= 3:
                return True

        # 2.2 子序列匹配 (例如: "Rob Griffith" vs "Dr Rob Griffith")
        if len(short_tokens) < len(long_tokens) and short_tokens:
            pos = 0
            for tok in long_tokens:
                if pos < len(short_tokens) and tok == short_tokens[pos]:
                    pos += 1
            if pos == len(short_tokens):
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

    def _is_alias_pair(self, short_name: str, long_name: str) -> bool:
        short_tokens = self._normalize_name(short_name)
        long_tokens = self._normalize_name(long_name)
        if not short_tokens or not long_tokens:
            return False
        if short_tokens == long_tokens:
            return False
        if len(short_tokens) == 1:
            token = short_tokens[0]
            return any(t.startswith(token) and t != token for t in long_tokens)
        if len(short_tokens) != len(long_tokens):
            return False
        for idx, token in enumerate(short_tokens):
            if not long_tokens[idx].startswith(token):
                return False
        return True

    def _maybe_add_edge(self, src_uid: int, dst_uid: int, src_name: str = "", dst_name: str = "", relation_type: str = "") -> None:
        if not self.memory_graph:
            return
        if src_uid is None or dst_uid is None:
            return
        if src_name:
            self.memory_graph.add_node(src_uid, name=src_name, type="", source_chunk="")
        if dst_name:
            self.memory_graph.add_node(dst_uid, name=dst_name, type="", source_chunk="")
        self.memory_graph.add_edge(
            src_uid,
            dst_uid,
            relation_type=relation_type,
            source_chunk="",
        )

    def _make_uid(self) -> int:
        # Keep uid consistent with INT64 schema.
        return uuid.uuid4().int % (2**63 - 1)

    def _search_milvus(self, vector, search_params, limit=5):
        if not self.milvus_db.db:
            self.milvus_db.load()
        return self.milvus_db.db.search(
            data=[vector],
            anns_field="vec",
            param=search_params,
            limit=limit,
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
        
        # 4. 判定逻辑
        results = await asyncio.to_thread(self._search_milvus, vector, search_params, 5)
        if results and len(results[0]) > 0:
            hits = results[0]
            # top‑k 命中后优先选“同名+同类型”的候选
            def _name_type_match(hit) -> bool:
                return (
                    self._normalize_name(entity_name) == self._normalize_name(hit.entity.get("name"))
                    and self._normalize_type(entity_type) == self._normalize_type(hit.entity.get("type"))
                )

            exact_hit = next((hit for hit in hits if _name_type_match(hit)), None)
            alias_hit = next(
                (
                    hit
                    for hit in hits
                    if self._normalize_type(entity_type) == self._normalize_type(hit.entity.get("type"))
                    and (
                        self._is_abbreviation_or_nickname(entity_name, hit.entity.get("name"))
                        or self._is_abbreviation_or_nickname(hit.entity.get("name"), entity_name)
                    )
                ),
                None,
            )

            top_match = exact_hit or alias_hit or hits[0]
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
