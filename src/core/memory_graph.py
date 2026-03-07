import networkx as nx
from typing import Tuple, List, Dict

class MemoryGraphManager:
    def __init__(self, promotion_threshold=3):
        # 使用 MultiDiGraph 支持同节点间的多种/多来源关系
        self.graph = nx.MultiDiGraph()
        # 记录每个 chunk 的被检索/访问次数: {chunk_id: count}
        self.chunk_access_counter = {}
        self.threshold = promotion_threshold

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