import asyncio
import re
import uuid
from typing import Callable, Optional, Tuple

class AsyncEntityResolver:
    def __init__(
        self, 
        milvus_client, 
        embedding_func, 
        collection_name="entity_index", 
        threshold=0.92,
        alias_callback: Optional[Callable[[str, str], None]] = None,
    ):
        self.milvus = milvus_client
        self.embed = embedding_func
        self.collection_name = collection_name
        self.threshold = threshold
        self.alias_callback = alias_callback
        
        # 本地级联缓存：减少对相同实体的重复 Embedding 和 数据库查询
        # 结构: { "entity_name:entity_desc": "global_uid" }
        self.local_cache = {}

    def _normalize_name(self, name: str):
        if not name:
            return []
        cleaned = re.sub(r"[^a-zA-Z0-9\s]", " ", str(name)).lower()
        return [token for token in cleaned.split() if token]

    def _normalize_type(self, entity_type: str) -> str:
        if not entity_type:
            return ""
        return str(entity_type).strip().lower()

    def _desc_similarity(self, left: str, right: str) -> float:
        left_tokens = set(self._normalize_name(left))
        right_tokens = set(self._normalize_name(right))
        if not left_tokens or not right_tokens:
            return 0.0
        return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)

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

    async def resolve_async(self, entity_name: str, entity_desc: str, entity_type: Optional[str] = None) -> str:
        """
        核心对齐方法：输入实体名称和描述，返回全局唯一的 UID。
        """
        cache_key = f"{entity_name}:{entity_desc}"
        
        # 1. 检查本地缓存 (O(1) 命中，极速返回)
        if cache_key in self.local_cache:
            return self.local_cache[cache_key]

        # 2. 异步获取向量 (不阻塞主线程)
        text_to_embed = f"Entity: {entity_name}. Description: {entity_desc}"
        vector = await self.embed(text_to_embed)

        # 3. 在 Milvus 中进行向量检索 (使用 to_thread 防止同步阻塞)
        search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
        
        results = await asyncio.to_thread(
            self.milvus.search,
            collection_name=self.collection_name,
            data=[vector],
            limit=1,
            output_fields=["uid", "name", "type", "desc"],
            search_params=search_params
        )

        # 4. 判定逻辑
        if results and len(results[0]) > 0:
            top_match = results[0][0]
            # 距离/相似度大于阈值，判定为同一实体
            if top_match.distance >= self.threshold:
                exist_uid = top_match.entity.get("uid")
                exist_name = top_match.entity.get("name")
                exist_type = top_match.entity.get("type")
                exist_desc = top_match.entity.get("desc")
                name_match = self._normalize_name(entity_name) == self._normalize_name(exist_name)
                type_match = self._normalize_type(entity_type) == self._normalize_type(exist_type)
                alias_type_match = type_match or not self._normalize_type(exist_type)
                desc_sim = self._desc_similarity(entity_desc, exist_desc)
                if name_match and (not entity_type or type_match) and (desc_sim >= 0.5 or top_match.distance >= self.threshold):
                    self.local_cache[cache_key] = exist_uid
                    return exist_uid
                if alias_type_match and (self._is_alias_pair(entity_name, exist_name) or self._is_alias_pair(exist_name, entity_name)):
                    new_uid = f"ent_{uuid.uuid4().hex[:10]}"
                    self.local_cache[cache_key] = new_uid
                    if self.alias_callback:
                        try:
                            self.alias_callback(new_uid, exist_uid)
                        except Exception:
                            pass
                    asyncio.create_task(self._register_new_entity(new_uid, entity_name, entity_type or "", entity_desc, vector))
                    return new_uid
                self.local_cache[cache_key] = exist_uid
                return exist_uid

        # 5. 未命中：生成新实体 UID，并异步注册到 Milvus
        new_uid = f"ent_{uuid.uuid4().hex[:10]}"
        self.local_cache[cache_key] = new_uid
        
        # 触发异步写入，不等待其完成即可返回
        asyncio.create_task(self._register_new_entity(new_uid, entity_name, entity_type or "", entity_desc, vector))
        
        return new_uid

    async def _register_new_entity(self, uid: str, name: str, entity_type: str, entity_desc: str, vector: list):
        """后台异步将新实体写入 Milvus"""
        data = [
            {"uid": uid, "name": name, "type": entity_type, "desc": entity_desc, "embedding": vector}
        ]
        await asyncio.to_thread(
            self.milvus.insert,
            collection_name=self.collection_name,
            data=data
        )