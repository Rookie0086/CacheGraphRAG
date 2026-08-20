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
from typing import List, Dict, Optional, Tuple
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

class AsyncEntityResolver:
    def __init__(
        self,
        embedding_func,
        collection_name="entity_index_example",
        threshold=0.85,
        memory_graph=None,
        embed_model=None,
        embedding_concurrency=5,
        tau_desc=0.5,
    ):
        self.milvus_db = MilvusDB(db_name=collection_name, overwrite=False, embed_model=embed_model)
        self.milvus_client = myMilvus()
        self.embed = embedding_func
        self.collection_name = collection_name
        self.threshold = threshold          # tau_sim: vector similarity threshold
        self.tau_desc = tau_desc             # tau_desc: description similarity threshold
        self.memory_graph = memory_graph
        self._embed_semaphore = asyncio.Semaphore(embedding_concurrency)
        self.embed_model = embed_model
        # Dedicated large thread pool: avoid asyncio.to_thread default pool (12 threads) becoming a bottleneck
        import concurrent.futures
        self._search_executor = concurrent.futures.ThreadPoolExecutor(max_workers=50)

        if self.collection_name not in self.milvus_client.list_collections():
            self.milvus_db.create_entity_collection()
        else:
            self.milvus_db.load()  # trigger connection + load
        # Local cascade cache
        self.local_cache = {}
        # Name cache: {norm_name|norm_type: (uid, desc)}
        # Only reuse when desc similarity is sufficient, to prevent blind merging of same-name different entities
        self._name_cache = {}
        self._pending_tasks = []
        

    def _normalize_name(self, name: str) -> str:
        """
        1. Convert to lowercase, strip leading/trailing whitespace
        2. Remove common business organization suffixes
        """
        if not name:
            return ""
        
        # 1. Basic cleaning
        name = name.lower().strip()
        
        # 2. Define insignificant suffix regex (handle boundary \b)
        # Covers: Inc, Corp, Co, Ltd, LLC, Group, Foundation, etc.
        suffixes = r'\b(inc|corp|co|ltd|llc|group|foundation|incorporated|corporation|limited)\b'
        
        # Remove suffixes and possible punctuation (e.g., Apple, Inc. -> apple)
        name = re.sub(rf'[,.\s]+({suffixes})[,.\s]*', '', name)
        name = re.sub(rf'[,.\s]+$', '', name) # Clean up trailing punctuation

        # Normalize punctuation to spaces and collapse whitespace
        name = re.sub(r"[^a-z0-9\s]", " ", name)
        name = re.sub(r"\s+", " ", name).strip()
        return name

    # On top of _normalize_name, further "split into token list" and remove honorifics
    def _tokenize_name(self, name: str) -> List[str]:
        if not name:
            return []
        honorifics = {"dr", "mr", "mrs", "ms", "prof", "sir", "madam"}
        tokens = self._normalize_name(name).split()
        tokens = [t for t in tokens if t and t not in honorifics and len(t) > 1]
        return tokens

    def _is_abbreviation_or_nickname(self, name1: str, name2: str) -> bool:
        """
        Determine if two strings are abbreviations, nicknames, or high-similarity variants of each other
        """
        n1 = " ".join(self._tokenize_name(name1))
        n2 = " ".join(self._tokenize_name(name2))
        
        if not n1 or not n2:
            return False
        if n1 == n2:
            return True

        # 1. Acronym check (e.g., "IBM" vs "International Business Machines")
        def check_acronym(short_str, long_str):
            # Extract the first letter of each word in the long string
            words = [w for w in re.split(r'[\s\-]+', long_str) if w]
            acronym = "".join([w[0] for w in words])
            return short_str == acronym

        short, long = (n1, n2) if len(n1) < len(n2) else (n2, n1)
        if check_acronym(short, long):
            return True

        # 2. Substring check (e.g., "Tim Cook" vs "Timothy Cook")
        # If all tokens of the short string are prefixes of corresponding tokens in the long string
        short_tokens = short.split()
        long_tokens = long.split()
        if len(short_tokens) == len(long_tokens):
            if all(l_t.startswith(s_t) for s_t, l_t in zip(short_tokens, long_tokens)):
                return True

        # 2.1 Single surname or given name match (e.g., "Bush" vs "Sophia Bush")
        if len(short_tokens) == 1 and len(long_tokens) >= 2:
            token = short_tokens[0]
            if token == long_tokens[-1]:
                return True
            if token in long_tokens and len(token) >= 3:
                return True

        # 2.2 Subsequence match (e.g., "Rob Griffith" vs "Dr Rob Griffith")
        if len(short_tokens) < len(long_tokens) and short_tokens:
            pos = 0
            for tok in long_tokens:
                if pos < len(short_tokens) and tok == short_tokens[pos]:
                    pos += 1
            if pos == len(short_tokens):
                return True

        # 3. Edit distance similarity (fuzzy matching)
        # SequenceMatcher calculates ratio: 2.0*M / T (M is matches, T is total length)
        similarity = SequenceMatcher(None, n1, n2).ratio()
        if similarity > 0.85: # High similarity threshold
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

        # Use token_sim for initial filter to avoid unnecessary embedding API calls
        if token_sim < 0.15:  # Clearly different
            return token_sim
        if token_sim > 0.65:  # Clearly same
            return 0.5 * token_sim + 0.5 * 1.0

        try:
            async with self._embed_semaphore:
                left_vec, right_vec = await asyncio.gather(
                    self.embed(left), self.embed(right))
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
        return self.milvus_db.search(vector, search_params, limit,
                                     output_fields=["uid", "name", "type", "desc"])

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

    async def resolve_async(self, entity_name: str, entity_type: str, entity_desc: str) -> str:
        cache_key = f"{entity_name}[{entity_type}]:{entity_desc}"

        # 1. Exact cache hit (identical name+type+desc)
        if cache_key in self.local_cache:
            return self.local_cache[cache_key]

        # 1.5 Name cache: reuse only if same name+type + desc similarity >= 0.3
        name_key = f"{self._normalize_name(entity_name)}|{self._normalize_type(entity_type)}"
        if name_key in self._name_cache:
            cached_uid, cached_desc = self._name_cache[name_key]
            desc_sim = await self._desc_similarity(entity_desc, cached_desc)
            if desc_sim >= 0.3:
                self.local_cache[cache_key] = cached_uid
                return cached_uid

        # 2. Async vector fetch
        text_to_embed = f"Entity: {entity_name}. Type: {entity_type}. Description: {entity_desc}"
        async with self._embed_semaphore:
            vector = await self.embed(text_to_embed)

        return await self._resolve_with_vector(entity_name, entity_type, entity_desc, vector, cache_key, name_key)

    async def _resolve_with_vector(self, entity_name: str, entity_type: str, entity_desc: str,
                                   vector, cache_key: str, name_key: str) -> str:
        search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}

        # 4. Decision logic
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(self._search_executor, self._search_milvus, vector, search_params, 50)
        if results and len(results[0]) > 0:
            hits = results[0]

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

            # Vector distance >= 0.92: high confidence same entity
            if top_match.distance >= 0.92:
                self.local_cache[cache_key] = exist_uid
                self._name_cache[name_key] = (exist_uid, exist_desc)
                return exist_uid
            # Same name+type + desc similarity >= 0.6: same entity
            elif name_match and type_match and desc_sim >= 0.6:
                self.local_cache[cache_key] = exist_uid
                self._name_cache[name_key] = (exist_uid, exist_desc)
                return exist_uid
            # Same name+type but large desc difference -> different entity, generate new UID
            elif name_match and type_match and desc_sim < 0.6:
                new_uid = self._make_uid()
                self.local_cache[cache_key] = new_uid
                # Do not update _name_cache, keep first entity as "primary entity"
                self._maybe_add_edge(new_uid, exist_uid, src_name=entity_name, dst_name=exist_name, relation_type="possible_same_as")
                task = asyncio.create_task(self._register_new_entity(new_uid, entity_name, entity_type, entity_desc, vector))
                task.add_done_callback(self._log_task_error)
                self._pending_tasks.append(task)
                return new_uid
            # Same name, different type + similar desc: type extraction deviation, treat as same entity
            elif name_match and (not entity_type) and (not type_match) and desc_sim >= self.tau_desc:
                self.local_cache[cache_key] = exist_uid
                return exist_uid
            # Abbreviation/short form: create new UID + alias_of edge
            elif type_match and (self._is_abbreviation_or_nickname(entity_name, exist_name) or self._is_abbreviation_or_nickname(exist_name, entity_name)):
                new_uid = self._make_uid()
                self.local_cache[cache_key] = new_uid
                self._maybe_add_edge(new_uid, exist_uid, src_name=entity_name, dst_name=exist_name, relation_type="alias_of")
                task = asyncio.create_task(self._register_new_entity(new_uid, entity_name, entity_type, entity_desc, vector))
                task.add_done_callback(self._log_task_error)
                self._pending_tasks.append(task)
                return new_uid

        # 5. Cache miss: generate new entity UID and asynchronously register in Milvus
        new_uid = self._make_uid()
        self.local_cache[cache_key] = new_uid
        self._name_cache[name_key] = (new_uid, entity_desc)

        task = asyncio.create_task(self._register_new_entity(new_uid, entity_name, entity_type, entity_desc, vector))
        task.add_done_callback(self._log_task_error)
        self._pending_tasks.append(task)

        return new_uid

    async def resolve_batch_async(self, entities: list, pre_vectors: Optional[list] = None) -> dict:
        """Batch alignment: one API call for all entity embeddings, then Milvus search + match individually.

        Args:
            entities: list of dicts with keys "id", "type", "desc".
            pre_vectors: Pre-computed entity vectors (corresponding to entities order),
                         skips internal embedding API call when provided.

        Returns:
            dict mapping original entity id -> {"uid": ..., "name": ..., "type": ..., "desc": ...}
        """
        cache_keys = {}
        name_keys = {}
        uncached = []  # (index, entity)

        for i, ent in enumerate(entities):
            name, etype, desc = ent["id"], ent["type"], ent["desc"]
            ck = f"{name}[{etype}]:{desc}"
            nk = f"{self._normalize_name(name)}|{self._normalize_type(etype)}"
            cache_keys[i] = ck
            name_keys[i] = nk

            if ck in self.local_cache:
                continue
            if nk in self._name_cache:
                self.local_cache[ck] = self._name_cache[nk][0]  # Get uid
                continue

            uncached.append((i, ent))

        # Batch embedding (prefer pre-computed vectors)
        if uncached:
            if pre_vectors is not None:
                vectors = [pre_vectors[i] for (i, _) in uncached]
            else:
                texts = [f"Entity: {e['id']}. Type: {e['type']}. Description: {e['desc']}" for _, e in uncached]
                async with self._embed_semaphore:
                    vectors = await self.embed_model.get_embeddings_async(texts)

            # Concurrent Milvus search: parallel search all entities after batch embedding
            tasks = [
                self._resolve_with_vector(
                    ent["id"], ent["type"], ent["desc"], vector,
                    cache_keys[i], name_keys[i],
                )
                for (i, ent), vector in zip(uncached, vectors)
            ]
            await asyncio.gather(*tasks)

        # Assemble results
        aligned = {}
        for i, ent in enumerate(entities):
            ck = cache_keys[i]
            cached = self._name_cache.get(name_keys[i])
            uid = self.local_cache.get(ck) or (cached[0] if cached else None)
            aligned[ent["id"]] = {"uid": uid, "name": ent["id"], "type": ent["type"], "desc": ent["desc"]}
        return aligned

    async def _register_new_entity(self, uid: str, name: str, entity_type: str, entity_desc: str, vector: list):
        """Asynchronously write new entities to Milvus in background"""
        if hasattr(vector, "tolist"):
            vector = vector.tolist()
        data = [
            {"uid": uid, "name": name, "type": entity_type, "desc": entity_desc, "vec": vector}
        ]
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(self._search_executor, self.milvus_db.insert, data)
