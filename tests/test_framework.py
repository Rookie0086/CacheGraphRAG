import asyncio
import json
import math
import os
import sys
import uuid
import networkx as nx
from typing import List, Dict, Tuple
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pymilvus.exceptions import MilvusException

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils import get_config
from triplet.prompts import prompt_extract_triplest_str
from utils.llm_env import LLMEnv  
from database.milvus import MilvusDB, myMilvus
from database.nebulagraph import NebulaClient
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
    def __init__(self, promotion_threshold=3):
        # 使用 MultiDiGraph 支持同节点间的多种/多来源关系
        self.graph = nx.MultiDiGraph()
        # 记录每个 chunk 的被检索/访问次数: {chunk_id: count}
        self.chunk_access_counter = {}
        self.threshold = promotion_threshold

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

    def save_graph_gexf(self, path: str):
        self._ensure_parent_dir(path)
        export_g = self._normalize_attrs_for_graphml(self.graph)
        nx.write_gexf(export_g, path)

    def load_graph_gexf(self, path: str):
        g = nx.read_gexf(path)
        if not isinstance(g, nx.MultiDiGraph):
            g = nx.MultiDiGraph(g)
        self.graph = self._restore_attrs_after_import(g)

    def add_node(self, uid: str, name: str, type: str, source_chunk: str):
        """添加节点。如果已存在，则追加 source_chunk"""
        if self.graph.has_node(uid):
            # 节点已存在，更新来源集合
            self.graph.nodes[uid]["source_chunks"].add(source_chunk)
        else:
            # 新节点
            self.graph.add_node(
                uid, 
                name=name, 
                type=type, 
                source_chunks={source_chunk} # 使用集合存储
            )

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
        threshold=0.92
    ):
        self.milvus_db = MilvusDB(db_name=collection_name, overwrite=False)
        self.milvus_client = myMilvus()
        self.embed = embedding_func
        self.collection_name = collection_name
        self.threshold = threshold
        
        if self.collection_name not in self.milvus_client.list_collections():
            self.milvus_db.create_entity_collection()
        else:
            self.milvus_db.load()
        # 本地级联缓存：减少对相同实体的重复 Embedding 和 数据库查询
        # 结构: { "entity_name:entity_desc": "global_uid" }
        self.local_cache = {}
        self._pending_tasks = []

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
            output_fields=["uid", "name"],
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
        
        results = await asyncio.to_thread(self._search_milvus, vector, search_params)

        # 4. 判定逻辑
        if results and len(results[0]) > 0:
            top_match = results[0][0]
            # 距离/相似度大于阈值，判定为同一实体
            if top_match.distance >= self.threshold:
                exist_uid = top_match.entity.get("uid")
                self.local_cache[cache_key] = exist_uid
                return exist_uid

        # 5. 未命中：生成新实体 UID，并异步注册到 Milvus
        new_uid = self._make_uid()
        self.local_cache[cache_key] = new_uid
        
        # 触发异步写入，不等待其完成即可返回
        task = asyncio.create_task(self._register_new_entity(new_uid, entity_name, vector))
        task.add_done_callback(self._log_task_error)
        self._pending_tasks.append(task)
        
        return new_uid

    async def _register_new_entity(self, uid: str, name: str, vector: list):
        """后台异步将新实体写入 Milvus"""
        if hasattr(vector, "tolist"):
            vector = vector.tolist()
        data = [
            {"uid": uid, "name": name, "vec": vector}
        ]
        await asyncio.to_thread(
            self.milvus_db.insert,
            data
        )


class DocumentIngestionPipeline:
    def __init__(
        self, 
        llm_client: LLMEnv,   # LLM 环境 (负责抽取和 Embedding)
        # vector_store,      # Milvus 客户端 (负责存储 Chunk)
        memory_graph,      # NetworkX 管理器
        entity_resolver,   # 负责与 Milvus 交互进行实体对齐
        max_concurrency=10 # LLM API 并发限制
    ):
        self.llm = llm_client
        self.vector_store = MilvusDB(db_name="example", overwrite=False) # 直接在 Pipeline 内部管理 MilvusDB 实例
        self.memory_graph = memory_graph
        self.entity_resolver = entity_resolver
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.chunk_registry = {}
        
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

    async def process_single_chunk(self, chunk_id: str, text: str) -> bool:
        """核心处理逻辑：受 Semaphore 控制的异步方法"""
        async with self.semaphore:
            try:
                # ==========================================
                # 链路 A: 存入 Milvus (Chunk 向量检索库)
                # ==========================================
                # 1. 获取文本的 Embedding
                # chunk_vector = await self.llm.embed_model.get_embedding_async(text)
                # if hasattr(chunk_vector, "tolist"):
                #     chunk_vector = chunk_vector.tolist()
                # print(f"Chunk {chunk_id} 的向量维度: {len(chunk_vector)}")

                # # 2. 存入 Milvus
                # await self.vector_store.insert_chunk_async(
                #     chunk_id=chunk_id, 
                #     vector=chunk_vector, 
                # )
                # print(f"Chunk {chunk_id} 已加入 Milvus 数据流。")
                # self.vector_store.db.flush()  # 确保数据可见
                # print(f"Chunk {chunk_id} 已 flush 到 Milvus。")
                # ==========================================
                # 链路 B: LLM 抽取 -> 实体对齐 -> 存入 NetworkX
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
                    uid = await self.entity_resolver.resolve_async(ent["id"], ent["desc"])
                    aligned_entities[ent["id"]] = {
                        "uid": uid, 
                        "name": ent["id"], 
                        "type": ent["type"], 
                        "desc": ent["desc"],
                    }

                # # 3. 写入 NetworkX (Memory Graph)
                self._write_to_memory_graph(chunk_id, aligned_entities, relations)

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
    with open("data/example.txt", "r") as f:
        document_text = f.read()

    print("🚀 --- [阶段 1: 文档入库处理 (Ingestion)] ---")
    mem_graph = MemoryGraphManager(promotion_threshold=2) # 测试环境阈值设低一点：2次
    resolver = AsyncEntityResolver(
        embedding_func=llm.embed_model.get_embedding_async,
    )

    pipeline = DocumentIngestionPipeline(
        llm_client=llm, 
        memory_graph=mem_graph, 
        entity_resolver=resolver,
    )
    
    await pipeline.process_document(document_text, source_file="data/example.txt")
    await resolver.wait_pending()
    print("文档入库处理完成！")
    if os.getenv("MILVUS_FORCE_FLUSH") == "1":
        flush_ok = False
        for attempt in range(3):
            try:
                resolver.milvus_db.flush()
                flush_ok = True
                break
            except MilvusException as e:
                print(f"Warning: Milvus flush failed on attempt {attempt + 1}: {e}")
                await asyncio.sleep(2)

        if not flush_ok:
            print("Warning: Milvus flush failed after retries; continue without hard fail.")

    try:
        print(resolver.milvus_client.get_collection_stats("entity_index"))
    except MilvusException as e:
        print(f"Warning: Failed to read collection stats: {e}")

    mem_graph.show_status()
    mem_graph.save_graph_graphml("subgraph/memory_graph.graphml")
    mem_graph.save_graph_gexf("subgraph/memory_graph.gexf")
    # print("\n🚀 --- [阶段 2: 模拟检索与子图晋升 (Retrieval & Promotion)] ---")
    
    # # 我们模拟来了 3 个相关的 Query，通过 VectorDB 定位到了特定的 Chunk
    # # 假设 query 定位到了包含 "Sam" 和 "Rob" 的 Chunk
    # target_chunks = [cid for cid, text in pipeline.chunk_registry.items() if "Sam" in text]
    
    # if not target_chunks:
    #     print("未找到相关的 Chunk")
    #     return

    # test_chunk_id = target_chunks[0]
    
    # for i in range(1, 4):
    #     print(f"\n💬 模拟收到用户 Query {i}...")
    #     print(f"   => 向量检索命中了 Chunk: {test_chunk_id}")
        
    #     # 触发内存图访问逻辑
    #     is_promoted, data = mem_graph.access_chunk(test_chunk_id)
        
    #     if is_promoted:
    #         print(f"   🎉 触发阈值！执行写入操作...")
    #         print(f"   => 将 {data} 异步写入 NebulaGraph！")
    #     else:
    #         print(f"   => Chunk 访问计数 +1，暂未达到晋升标准。")

    # mem_graph.show_status()

if __name__ == "__main__":
    asyncio.run(run_tests())

    # python -m tests.test_framework