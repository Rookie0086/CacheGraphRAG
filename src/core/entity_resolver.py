import uuid
import asyncio
from typing import Tuple

class AsyncEntityResolver:
    def __init__(
        self, 
        milvus_client, 
        embedding_func, 
        collection_name="entity_index", 
        threshold=0.92
    ):
        self.milvus = milvus_client
        self.embed = embedding_func
        self.collection_name = collection_name
        self.threshold = threshold
        
        # 本地级联缓存：减少对相同实体的重复 Embedding 和 数据库查询
        # 结构: { "entity_name:entity_desc": "global_uid" }
        self.local_cache = {}

    async def resolve_async(self, entity_name: str, entity_desc: str) -> str:
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
            output_fields=["uid", "name"],
            search_params=search_params
        )

        # 4. 判定逻辑
        if results and len(results[0]) > 0:
            top_match = results[0][0]
            # 距离/相似度大于阈值，判定为同一实体
            if top_match.distance >= self.threshold:
                exist_uid = top_match.entity.get("uid")
                self.local_cache[cache_key] = exist_uid
                return exist_uid

        # 5. 未命中：生成新实体 UID，并异步注册到 Milvus
        new_uid = f"ent_{uuid.uuid4().hex[:10]}"
        self.local_cache[cache_key] = new_uid
        
        # 触发异步写入，不等待其完成即可返回
        asyncio.create_task(self._register_new_entity(new_uid, entity_name, vector))
        
        return new_uid

    async def _register_new_entity(self, uid: str, name: str, vector: list):
        """后台异步将新实体写入 Milvus"""
        data = [
            {"uid": uid, "name": name, "embedding": vector}
        ]
        await asyncio.to_thread(
            self.milvus.insert,
            collection_name=self.collection_name,
            data=data
        )