import asyncio
import uuid
import json
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 假设这些是你已经封装好的客户端工具类
# from src.core.llm_client import AsyncLLMClient
# from src.core.embedding import get_embedding_async
# from src.core.entity_resolver import AsyncEntityResolver

class DocumentIngestionPipeline:
    def __init__(
        self, 
        llm_client, 
        vector_store,      # Milvus 客户端 (负责存储 Chunk)
        memory_graph,      # NetworkX 管理器
        entity_resolver,   # 负责与 Milvus 交互进行实体对齐
        max_concurrency=10 # LLM API 并发限制
    ):
        self.llm = llm_client
        self.vector_store = vector_store
        self.memory_graph = memory_graph
        self.entity_resolver = entity_resolver
        self.semaphore = asyncio.Semaphore(max_concurrency)
        
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
            tasks.append(self.process_single_chunk(chunk_id, text, source_file))

        # 3. 并发执行所有 Chunk 处理任务
        # asyncio.gather 会等待所有任务完成，并返回结果列表
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 4. 错误处理与统计
        success_count = sum(1 for r in results if r is True)
        print(f"文档 {source_file} 处理完成。成功: {success_count}/{len(chunks)}")

    async def process_single_chunk(self, chunk_id: str, text: str, source_file: str) -> bool:
        """核心处理逻辑：受 Semaphore 控制的异步方法"""
        async with self.semaphore:
            try:
                # ==========================================
                # 链路 A: 存入 Milvus (Chunk 向量检索库)
                # ==========================================
                # 1. 获取文本的 Embedding
                chunk_vector = await self.llm.get_embedding_async(text)
                
                # 2. 存入 Milvus
                await self.vector_store.insert_chunk_async(
                    chunk_id=chunk_id, 
                    vector=chunk_vector, 
                    payload={"text": text, "source": source_file}
                )

                # ==========================================
                # 链路 B: LLM 抽取 -> 实体对齐 -> 存入 NetworkX
                # ==========================================
                # 1. 调用 LLM 进行抽取 (使用之前设计的严格 Schema)
                raw_json_str = await self.llm.extract_knowledge_async(text)
                entities, relations = self._clean_and_validate(raw_json_str)

                if not entities:
                    return True # 抽取为空，无需入图，但不算失败

                # 2. 实体对齐 (Entity Resolution)
                # 将 LLM 抽取的临时 ID 转换为全局唯一 ID
                aligned_entities = {}
                for ent in entities:
                    # resolver 会去 Milvus 查重，返回全局 uid
                    uid = await self.entity_resolver.resolve_async(ent["id"], ent["desc"])
                    aligned_entities[ent["id"]] = {
                        "uid": uid, 
                        "name": ent["id"], 
                        "type": ent["type"], 
                        "desc": ent["desc"]
                    }

                # 3. 写入 NetworkX (Memory Graph)
                self._write_to_memory_graph(chunk_id, aligned_entities, relations)

                return True

            except Exception as e:
                print(f"处理 Chunk {chunk_id} 时出错: {str(e)}")
                return False

    def _clean_and_validate(self, raw_str: str):
        """清洗 LLM 输出，过滤悬空关系"""
        try:
            data = json.loads(raw_str)
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