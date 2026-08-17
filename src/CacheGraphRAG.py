import asyncio
import faulthandler
import json
import signal
import threading
import time
import os
import sys
import warnings
import concurrent.futures
from typing import List, Optional

# 诊断用:kill -USR1 <pid> 可 dump 各线程 Python 栈(定位并发卡死)
faulthandler.register(signal.SIGUSR1)

# Suppress noisy warnings from third-party libraries
warnings.filterwarnings("ignore", category=UserWarning, module="pymilvus")
warnings.filterwarnings("ignore", category=UserWarning, message="pkg_resources")
warnings.filterwarnings("ignore", message=".*validate_default.*")
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "OFF"
import logging
logging.getLogger("pymilvus").setLevel(logging.ERROR)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils import get_config
from src.utils.base import read_json, save_to_json, set_global_seed
from src.utils.prompts import prompt_answer_with_chunks_str
from src.utils.logger import PipelineLogger, set_logger
from src.llm.env import LLMEnv
from database.milvus import MilvusDB
from database.nebulagraph import NebulaClient
from src.pipeline import DocumentIngestionPipeline
from src.graph.memory_graph import MemoryGraphManager
from src.entity.resolver import AsyncEntityResolver
from src.retrieval.retriever import HybridRetriever, FactRetriever
from src.retrieval.fusion import RRFFusion, WeightedFusion, DualFusion
from src.retrieval.reranker import APIReranker, LocalReranker
from src.retrieval.agentic_engine import IterativeAgenticEngine
from data.rgb import get_rgb_info
from data.multihop import get_multihop_info
from data.dragonball import get_dragonball_info
from data.squad import get_squad_info
from data.hotpotqa import get_hotpotqa_info, get_hotpotqa_corpus
from data.ectqa import get_ectqa_info
from data.whoqa import get_whoqa_ex_info
from data.cond import get_cond_info
from data.wikimultihopqa import get_2wikimultihopqa_info
from data.musique import get_musique_info


_DATASET_LOADERS = {
    "rgb": lambda ds: get_rgb_info(file=ds[4:]),
    "dragonball": lambda _: get_dragonball_info("en", "Factual Question"),
    "wikimultihopqa": lambda _: get_2wikimultihopqa_info(),  # Must be before "multihop" to avoid partial match
    "multihop": lambda _: get_multihop_info(),
    "squad": lambda _: get_squad_info(file="dev"),
    "hotpotqa": lambda _: get_hotpotqa_info(file="hotpot_dev_distractor_v1", num=600),
    "ectqa": lambda _: get_ectqa_info(corpus_file="new.jsonl.gz"),
    "whoqa": lambda _: get_whoqa_ex_info(limit=600, update=True),
    "cond": lambda _: get_cond_info(file="cond"),
    "musique": lambda _: get_musique_info(limit=300),
}

_CORPUS_LOADERS: dict = {
    "hotpotqa": get_hotpotqa_corpus,
}


class CacheGraphRAG:
    def __init__(
        self,
        dataset: str,
        backend: str = "openai",
        llm_config: Optional[dict] = None,
        embed_config: Optional[dict] = None,
        l1_max_chunks: int = 100,
        l1_max_nodes: Optional[int] = None,
        l1_ttl_seconds: int = 3600,
        prune_interval: int = 5,
        promotion_threshold: int = 3,
        log_dir: str = "log",
        llm_concurrency: int = 10,
        chunk_concurrency: int = 5,
        build_concurrency: int = 3,
        nebula_space: Optional[str] = None,
        chunk_size: int = 800,
        chunk_overlap: int = 100,
        reranker=None,
    ):
        self.dataset = dataset
        self.reranker = reranker
        self.nebula_space = nebula_space or dataset
        self.backend = backend
        self._llm_concurrency = llm_concurrency
        self._chunk_concurrency = chunk_concurrency
        self._build_concurrency = build_concurrency
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self.logger = PipelineLogger(log_dir=log_dir, dataset=dataset)
        set_logger(self.logger)
        self.llm = self._init_llm(llm_config, embed_config)
        self.mem_graph = MemoryGraphManager(
            space_name=nebula_space,
            promotion_threshold=promotion_threshold,
            capacity_limit=l1_max_chunks,
            max_nodes=l1_max_nodes,
            ttl_seconds=l1_ttl_seconds,
            prune_interval=prune_interval,
        )
        # L1 初始化策略(2026-08-15 第六轮):
        #   默认空启动(load_gexf=false)——L1 启动为空,QA 期由 DPR 命中触发
        #   rehydrate_chunk_from_milvus 从 Milvus graph_meta 按 chunk 重载填充,
        #   逐步达到容量上限 C_max(论文 IV-C Lazy Rehydration,与 fig5 冷启动一致)。
        #   仅当 retrieval.load_gexf=true 时从构建期 gexf 快照预填 L1(旧行为/观察用)。
        cfg = get_config()
        load_gexf = bool(cfg.get("retrieval", {}).get("load_gexf", False))
        if load_gexf:
            cc = cfg.get("retrieval", {}).get("chunk_collection", nebula_space)
            s, e = cfg.get("data", {}).get("start", 0), cfg.get("data", {}).get("end", -1)
            gexf_path = f"subgraph/base/{dataset}_{cc}_{nebula_space}_{s}_{e}_base.gexf"
            # fallback: legacy naming format
            if not os.path.exists(gexf_path):
                gexf_path = f"subgraph/base/memory_graph_{nebula_space}.gexf"
            if os.path.exists(gexf_path):
                try:
                    self.mem_graph.load_graph_gexf(gexf_path)
                    print(f"Loaded graph snapshot: {gexf_path}")
                except Exception as e:
                    print(f"Failed to load graph snapshot: {e}")
            else:
                print("[L1] load_gexf=true 但未找到 gexf 快照,以空 L1 启动")
        else:
            print("[L1] 空启动(lazy rehydration):L1 初始为空,QA 期按 chunk 从 Milvus graph_meta 重载填充")
        self.pipeline: Optional[DocumentIngestionPipeline] = None
        self.retriever: Optional[HybridRetriever] = None
        self.questions: List[str] = []
        self.answers: List[str] = []
        self.texts: List[str] = []
        self.build_phase: str = "base"
        self._llm_answer_cache: dict = {}  # question+chunks_hash → (answer, raw)

        self._ingestion_time: List[float] = []
        # 论文 IV-C:L2 晋升子图"异步固化"——后台线程写 Nebula,不阻塞 QA 检索路径。
        # H2:晋升触发(命中计数+子图提取,含 Milvus 查询)也整体放入后台线程。
        # M3:有界背压,积压超阈值退化同步执行,避免无界队列与 L2 写滞后。
        # 退出前由 shutdown() 统一 flush,保证晋升数据落盘。
        self._l2_writer = concurrent.futures.ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="l2-writer")
        self._l2_max_pending = 64
        self._l2_pending = 0
        self._l2_pending_lock = threading.Lock()
        # M1(2026-08-15):QA 检索/引擎专用线程池——engine.run 与 hybrid_retrieve 不再
        # 占用默认 asyncio.to_thread 池(与构建期 Milvus 插入等共享),命名便于观测。
        _qa_workers = max(4, int(get_config().get("retrieval", {}).get("qa_concurrency", 5)))
        self._qa_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=_qa_workers, thread_name_prefix="qa-worker")
        self._retrieval_time: List[float] = []
        self._total_io_count: int = 0

        llm_model_name = (llm_config or {}).get("model_name", "gpt-4o-mini")
        embed_model_name = (embed_config or {}).get("model_name", "BAAI/bge-m3")
        embed_backend = (embed_config or {}).get("backend", "local")
        self.logger.log_config({
            "dataset": dataset,
            "backend": backend,
            "llm_model": llm_model_name,
            "embed_model": embed_model_name,
            "embed_backend": embed_backend,
            "l1_max_chunks": l1_max_chunks,
            "l1_max_nodes": l1_max_nodes or "unlimited",
            "l1_ttl_seconds": l1_ttl_seconds,
            "promotion_threshold": promotion_threshold,
            "prune_interval": prune_interval,
            "llm_concurrency": llm_concurrency,
            "chunk_concurrency": chunk_concurrency,
            "build_concurrency": build_concurrency,
        })
        self._check_databases()

    # ── Initialization ──────────────────────────────────────────────

    @staticmethod
    def _init_llm(llm_config: Optional[dict], embed_config: Optional[dict]) -> LLMEnv:
        config = llm_config or get_config()
        return LLMEnv(
            backend=config.get("backend", "openai"),
            model=config.get("model_name", "gpt-4o-mini"),
            api_key=config.get("api_key"),
            base_url=config.get("base_url"),
            embed_model_name=embed_config.get("model_name", "BAAI/bge-m3") if embed_config else "BAAI/bge-m3",
            embed_backend=embed_config.get("backend", "local") if embed_config else "local",
            embed_api_key=embed_config.get("api_key") if embed_config and embed_config.get("backend") == "api" else None,
            embed_base_url=embed_config.get("base_url") if embed_config else None,
        )

    @classmethod
    def from_config(cls, custom_cfg: Optional[dict] = None) -> "CacheGraphRAG":
        """Build instance from config.yaml."""
        cfg = custom_cfg or get_config()
        model_cfg = cfg.get("model", {})
        # 全局随机种子(只影响数据切分/采样,不控制 LLM API 输出)
        set_global_seed(model_cfg.get("seed", 42))
        embed_cfg = dict(cfg.get("embedding", {}))
        # Keep credentials out of reproducibility configs; experiments may
        # select a local/OpenAI-compatible embedding endpoint via environment.
        embed_env = {
            "backend": os.environ.get("CACHEGRAPH_EMBED_BACKEND"),
            "model_name": os.environ.get("CACHEGRAPH_EMBED_NAME"),
            "base_url": os.environ.get("CACHEGRAPH_EMBED_BASE_URL"),
            "api_key": os.environ.get("CACHEGRAPH_EMBED_API_KEY"),
        }
        embed_cfg.update({key: value for key, value in embed_env.items() if value})
        data_cfg = cfg.get("data", {})
        idx_cfg = cfg.get("indexing", {})
        ret_cfg = cfg.get("retrieval", {})
        rerank_cfg = dict(cfg.get("rerank", {}))
        rerank_env = {
            "backend": os.environ.get("CACHEGRAPH_RERANK_BACKEND"),
            "model_name": os.environ.get("CACHEGRAPH_RERANK_NAME"),
            "base_url": os.environ.get("CACHEGRAPH_RERANK_BASE_URL"),
            "api_key": os.environ.get("CACHEGRAPH_RERANK_API_KEY"),
        }
        rerank_cfg.update({key: value for key, value in rerank_env.items() if value})

        # Build reranker
        rerank_backend = rerank_cfg.get("backend", "none")
        if rerank_backend == "api":
            reranker = APIReranker(
                model_name=rerank_cfg.get("model_name", "BAAI/bge-reranker-v2-m3"),
                api_key=rerank_cfg.get("api_key", ""),
                base_url=rerank_cfg.get("base_url", "https://api.siliconflow.cn/v1"),
            )
        elif rerank_backend == "local":
            local_path = rerank_cfg.get("local_path", "")
            if local_path:
                reranker = LocalReranker(model_path=local_path)
            else:
                print("[WARN] rerank local_path not set, skipping reranker")
                reranker = None
        else:
            reranker = None

        backend = model_cfg.get("backend", "openai")
        model_name = model_cfg.get("model_name", "")
        if backend == "openai":
            llm_cfg = {"backend": "openai",
                       "model_name": model_name or "gpt-4o-mini",
                       "api_key": model_cfg.get("api_key"), "base_url": model_cfg.get("base_url")}
        elif backend == "deepseek":
            llm_cfg = {"backend": "deepseek",
                       "model_name": model_name or "deepseek-chat",
                       "api_key": model_cfg.get("api_key"), "base_url": model_cfg.get("base_url")}
        elif backend == "ollama":
            llm_cfg = {"backend": "ollama",
                       "model_name": model_name or "llama3.1:8b",
                       "api_key": None, "base_url": model_cfg.get("base_url")}
        else:
            raise ValueError(f"Unsupported backend: {backend}")

        l1_max_nodes = idx_cfg.get("l1_max_nodes", 0)
        l1_max_nodes = l1_max_nodes if l1_max_nodes > 0 else None

        return cls(
            dataset=data_cfg.get("dataset", "rgb_en"),
            backend=backend,
            llm_config=llm_cfg,
            embed_config=embed_cfg,
            l1_max_chunks=idx_cfg.get("l1_max_chunks", 100),
            l1_max_nodes=l1_max_nodes,
            l1_ttl_seconds=idx_cfg.get("l1_ttl_seconds", 3600),
            prune_interval=idx_cfg.get("prune_interval", 5),
            promotion_threshold=idx_cfg.get("promotion_threshold", 3),
            llm_concurrency=idx_cfg.get("llm_concurrency", 10),
            chunk_concurrency=idx_cfg.get("chunk_concurrency", 5),
            build_concurrency=idx_cfg.get("build_concurrency", 3),
            chunk_size=idx_cfg.get("chunk_size", 800),
            chunk_overlap=idx_cfg.get("chunk_overlap", 100),
            nebula_space=ret_cfg.get("nebula_space", None),
            reranker=reranker,
        )

    # ── Dataset Loading ──────────────────────────────────────────

    def load_dataset(self, start: int = 0, end: int = -1):
        """Load and slice dataset."""
        if "whoqa" in self.dataset:
            update = self.build_phase == "incremental"
            data_info = get_whoqa_ex_info(limit=600, update=update)
            print(f"[whoqa] build_phase={self.build_phase}, update={update}, loading phase_{'2' if update else '1'}_data")
        else:
            loader = self._resolve_loader()
            data_info = loader(self.dataset)

        self.questions = data_info["questions"]
        self.answers = data_info["answers"]
        self.texts = data_info["texts"]

        assert len(self.questions) == len(self.answers), "Questions and answers must have the same length."
        mismatch_texts = len(self.questions) != len(self.texts)

        if end == -1:
            end = len(self.questions)
        self.questions = self.questions[start:end]
        self.answers = self.answers[start:end]

        if not mismatch_texts:
            self.texts = self.texts[start:end]
        elif self.dataset == "squad":
            self.texts = self.texts[start:(end // 5)]
        elif self.dataset == "ectqa":
            self.texts = self.texts[start:100]

        print(f"questions: {len(self.questions)}, answers: {len(self.answers)}, texts: {len(self.texts)}")
        return self

    def load_corpus(self, limit: int = 0) -> List[str]:
        """Load corpus (deduplicated unique document list), not QA-aligned.

        Args:
            limit: Max corpus docs (0 = no limit, use default).
        """
        func = _CORPUS_LOADERS.get(self.dataset)
        if func is None:
            for key in _CORPUS_LOADERS:
                if key in self.dataset:
                    func = _CORPUS_LOADERS[key]
                    break
        if func is None:
            raise ValueError(f"Corpus loader not available for dataset: {self.dataset}")
        num = limit if limit > 0 else 300
        corpus = func(file="hotpot_dev_distractor_v1", num=num)
        print(f"Corpus: {len(corpus)} unique docs (limit={num})")
        return corpus

    def _resolve_loader(self):
        for key, loader in _DATASET_LOADERS.items():
            if key in self.dataset:
                return loader
        raise ValueError(f"Unknown dataset: {self.dataset}")

    # ── Document Ingestion ────────────────────────────────────────────

    async def ingest(self, texts: List[str]) -> "CacheGraphRAG":
        print("\n--- [Ingestion] ---")
        fact_cfg = get_config().get("fact_retrieval", {})
        ret_cfg = get_config().get("retrieval", {})
        resolver = AsyncEntityResolver(
            collection_name="entity_index_" + self.nebula_space,
            embedding_func=self.llm.embed_model.get_embedding_async,
            memory_graph=self.mem_graph,
            embed_model=self.llm.embed_model,
            embedding_concurrency=20,
            threshold=ret_cfg.get("alignment_threshold", 0.85),      # τ_sim:强对齐阈值(论文式5)
            desc_threshold=ret_cfg.get("alignment_desc_threshold", 0.6),  # τ_desc:弱语义阈值(算法1行9/12/15)
            scope=ret_cfg.get("alignment_scope", "l1"),              # 比对空间:l1=活跃节点集 V_L1(论文式7);full=全量索引
        )
        idx_cfg = get_config().get("indexing", {})
        self.pipeline = DocumentIngestionPipeline(
            collection_name=self.nebula_space,
            llm_client=self.llm,
            memory_graph=self.mem_graph,
            entity_resolver=resolver,
            embed_model=self.llm.embed_model,
            llm_concurrency=self._llm_concurrency,
            chunk_concurrency=self._chunk_concurrency,
            build_concurrency=self._build_concurrency,
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
            fact_enabled=fact_cfg.get("enabled", False),
            fact_collection_prefix=fact_cfg.get("fact_collection_prefix", "fact_index"),
        )
        self.pipeline._offline_alignment = idx_cfg.get("offline_alignment", False)
        self.mem_graph.chunk_vector_store = self.pipeline.vector_store

        start = time.time()

        print("Pipeline: extracting and building in parallel...")
        total_chunks, built_count = await self.pipeline.extract_and_build(texts)

        # L1 容量保持用户设定值(l1_max_chunks),不再自动抬升。
        # 若显式设为 0 则视为无容量上限(LRU/TTL 仍按 TTL 生效)。
        if self.mem_graph.capacity_limit == 0:
            print("  [L1] l1_max_chunks=0 → L1 无容量上限(LRU 淘汰不触发),仅 TTL 驱逐生效")

        print(f"  Pipeline done: {built_count}/{total_chunks} chunks ingested")

        elapsed = time.time() - start
        self._ingestion_time.append(elapsed)
        print(f"ingestion done: {elapsed:.2f}s")
        return self

    # ── Retrieval + Answering ────────────────────────────────────────

    async def _answer_with_chunks(self, question: str, chunk_ids: List[str], chunk_collection: Optional[str] = None) -> tuple:
        """Three-step CoT: extract candidates → analyze → answer. Returns (parsed_answer, raw_response)."""
        if not chunk_ids:
            return "", ""

        # LLM cache: skip reading when use_llm_cache=false, but always save
        cache_key = hash(question + "|" + ",".join(sorted(chunk_ids)))
        use_cache = get_config().get("retrieval", {}).get("use_llm_cache", True)
        if use_cache and cache_key in self._llm_answer_cache:
            return self._llm_answer_cache[cache_key]

        cname = chunk_collection or self.nebula_space
        milvus_db = MilvusDB(db_name=cname, overwrite=False, embed_model=self.llm.embed_model)
        context_parts = []
        for cid in chunk_ids:
            text_info = milvus_db.get_chunk_text(cid)
            text = (text_info or {}).get("text", "")
            if text:
                context_parts.append(f"[{cid}] {text}")
        context = "\n\n".join(context_parts)
        if not context:
            return "I don't know.", ""
        prompt = prompt_answer_with_chunks_str.format(query=question, context=context)
        raw = await self.llm.async_complete(prompt=prompt) or ""

        import re, json
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:].strip()
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end > start:
            try:
                payload = json.loads(cleaned[start:end + 1])
                candidates = payload.get("step1_candidates", "")
                analysis = payload.get("step2_analysis", "")
                # verbose output disabled
                answer = payload.get("final_answer") or payload.get("answer")
                if answer:
                    return str(answer).strip(), raw.strip()
            except json.JSONDecodeError:
                pass
        match = re.search(r"(?:final_)?answer\s*[:：]\s*(.+)", raw, re.IGNORECASE)
        parsed = match.group(1).strip() if match else raw.strip()
        result = (parsed, raw.strip())
        # 只缓存有内容的回答:LLM API 偶发故障返回空/None 时,缓存空结果会
        # 让同 (query, chunks) 组合的后续查询全部命中空答案,放大一次故障的影响。
        if result[0]:
            self._llm_answer_cache[cache_key] = result
        return result

    async def query(
        self,
        questions: List[str],
        start: int = 0,
        end: int = 0,
        use_agentic: bool = True,
        agentic_steps: int = 3,
        topk: int = 10,
        top_entities: int = 5,
        top_chunks: Optional[int] = None,
        top_rerank: int = 15,
        answer_topk: int = 6,
        answer_aware_promotion: bool = False,
        entity_promotion_threshold: int = 3,
        promotion_retries: int = 2,
        promotion_expansion: int = 10,
        qa_concurrency: int = 5,
        entity_extraction: str = "llm",
        qa_cache_file: Optional[str] = None,
        entity_index_name: Optional[str] = None,
        chunk_collection: Optional[str] = None,
        mode: str = "hybrid",
    ) -> List[dict]:
        """Retrieve and generate answers in one pass."""
        print(f"\n--- [Query] (concurrency={qa_concurrency}, entity_extraction={entity_extraction}) ---")
        self.mem_graph.nebula_IO_count = 0

        ccol = chunk_collection or self.nebula_space
        print(f"  chunk_collection: {ccol}")

        # Read params from config if not explicitly passed
        ret_cfg = get_config().get("retrieval", {})
        if top_chunks is None:
            top_chunks = ret_cfg.get("top_chunks", 15)

        # ── Load QA Cache (resume support) ────────────────────────
        results: List[dict] = []
        latency_records: List[dict] = []  # 每查询时延记录(检索/作答/阶段/命中来源)
        completed_questions: set = set()
        if qa_cache_file and os.path.exists(qa_cache_file):
            try:
                with open(qa_cache_file) as _f:
                    cached = json.load(_f)
                results.extend(cached)
                completed_questions = {item["query"] for item in cached}
                print(f"Loaded QA cache: {len(cached)} completed, skipping")
            except Exception as e:
                print(f"QA cache load failed, restarting: {e}")

        eindex_name = entity_index_name or "entity_index_" + self.nebula_space
        print(f"  entity_index: {eindex_name}")

        # ── Fact Retrieval vs Traditional ──
        cfg = get_config()
        fact_cfg = cfg.get("fact_retrieval", {})
        fact_enabled = fact_cfg.get("enabled", False)
        quality_cfg = cfg.get("quality_promotion", {})
        quality_enabled = quality_cfg.get("enabled", False)
        conv_threshold = fact_cfg.get("entity_convergence_threshold", 3)

        # 接线 chunk_vector_store:让 rehydrate 能从向量库恢复被 L1 淘汰 chunk 的拓扑。
        # ingest() 里也会赋一次 pipeline.vector_store;这里保证纯 QA(skip_index/实验)路径
        # 同样可用,否则 _rehydrate_vector_hits 因 store 为 None 全部静默失败。
        vector_store = MilvusDB(db_name=ccol, overwrite=False, embed_model=self.llm.embed_model)
        self.mem_graph.chunk_vector_store = vector_store
        common_args = dict(
            vector_store=vector_store,
            entity_index_name=eindex_name,
            memory_graph=self.mem_graph,
            llm=self.llm,
            chunk_registry=self.pipeline.chunk_registry if self.pipeline else {},
            reranker=self.reranker,
        )

        if fact_enabled:
            fact_coll = f"{fact_cfg.get('fact_collection_prefix', 'fact_index')}_{ccol}"
            fact_store = MilvusDB(db_name=fact_coll, overwrite=False, embed_model=self.llm.embed_model)
            retriever = FactRetriever(
                **common_args,
                fact_store=fact_store,
                fact_config=fact_cfg,
            )
            print(f"  Fact retrieval mode: {fact_coll}")
        else:
            retriever = HybridRetriever(
                **common_args,
                entity_extraction=entity_extraction,
                mode=mode,
            )

        # ── Fusion Strategy (skipped in graph_only mode) ──
        if mode == "graph_only":
            retriever.fusion = None
        else:
            fusion_cfg = cfg.get("fusion", {})
            method = fusion_cfg.get("method", "rrf")
            if method == "weighted":
                fusion = WeightedFusion(
                    retriever,
                    dpr_weight=fusion_cfg.get("dpr_weight", 0.5),
                    overlap_boost=fusion_cfg.get("overlap_boost", 3.0),
                    convergence_boost=fusion_cfg.get("convergence_boost", 1.3),
                    convergence_threshold=fusion_cfg.get("convergence_threshold", 2),
                )
            elif method == "dual":
                fusion = DualFusion(
                    retriever,
                    rrf_k=fusion_cfg.get("rrf_k", 60),
                )
            else:
                fusion = RRFFusion(retriever, rrf_k=fusion_cfg.get("rrf_k", 60))
            retriever.fusion = fusion

        start_time = time.time()
        sem = asyncio.Semaphore(qa_concurrency)
        _rlock = threading.Lock()

        rewrite_enabled = fact_cfg.get("query_rewrite", False)

        async def _process_one(idx: int, question: str):
            async with sem:
                if qa_cache_file and question in completed_questions:
                    self.logger.qa_progress()
                    return
                if qa_concurrency == 1:
                    self.logger.buffer_on()

                # Query rewriting
                q = question
                if rewrite_enabled:
                    rewrite_prompt = f"Rewrite this question to highlight key disambiguating clues (names, titles, dates, places, relations). Output ONLY the rewritten query, no explanation.\\n\\nQuestion: {question}\\nRewritten:"
                    try:
                        rewritten = await self.llm.async_complete(prompt=rewrite_prompt)
                        if rewritten and len(rewritten.strip()) > 3:
                            q = rewritten.strip()
                    except Exception:
                        pass

                self.logger.query_start(idx + 1, question)

                retry_count = 0
                ret_ms = 0.0  # 本查询检索阶段耗时(秒,记录最后一次成功轮)
                ans_ms = 0.0  # 本查询 LLM 作答耗时(秒)
                stage_lat = {}  # retriever 内部分阶段耗时(embed/dpr/entity/graph/aggregate/fusion)
                agentic_metrics = {}
                if use_agentic:
                    engine = IterativeAgenticEngine(
                        llm=self.llm, dataset=self.dataset, retriever=retriever,
                        max_steps=agentic_steps, topk=topk, top_entities=top_entities,
                        top_chunks=top_chunks, top_rerank=top_rerank,
                    )
                    _t0 = time.time()
                    # engine.run performs synchronous retrieval and several LLM calls.
                    # Offload it so qa_concurrency is effective for Agentic QA too.
                    # M1:使用专用 qa-worker 线程池,不占默认 asyncio.to_thread 池
                    result = await asyncio.get_running_loop().run_in_executor(
                        self._qa_executor, engine.run, question)
                    ret_ms = time.time() - _t0
                    ra_answer = result["final_answer"]
                    chunk_ids, raw_answer = result.get("chunks", []), ""
                    final_retrieval_res = None
                    agentic_metrics = result
                    stage_lat = result.get("stage_latency", {}) or {}
                else:
                    current_topk = topk
                    final_retrieval_res = None
                    for attempt in range(promotion_retries + 1):
                        track = False  # Promotion handled uniformly outside retrieval
                        _t0 = time.time()
                        # M1:使用专用 qa-worker 线程池,不占默认 asyncio.to_thread 池
                        retrieval_res = await asyncio.get_running_loop().run_in_executor(
                            self._qa_executor, retriever.hybrid_retrieve,
                            q, current_topk, top_entities, top_chunks, top_rerank, answer_topk, track)
                        ret_ms = time.time() - _t0
                        chunk_ids = retrieval_res.get("chunks", [])
                        if not chunk_ids:
                            self.logger.warn(f"Query {idx+1} retrieved no chunks, answer may be inaccurate")
                        _t0 = time.time()
                        ra_answer, raw_answer = await self._answer_with_chunks(question, chunk_ids, chunk_collection)
                        ans_ms = time.time() - _t0

                        if answer_aware_promotion and attempt < promotion_retries and self._is_dont_know(ra_answer):
                            current_topk += promotion_expansion
                            retry_count = attempt + 1
                            self.logger._print(f"  Query {idx+1} retry {attempt+1}/{promotion_retries}, topk={current_topk}")
                            continue
                        final_retrieval_res = retrieval_res
                        stage_lat = (final_retrieval_res.get("stats", {}) or {}).get("latency", {})
                        break

                    # Promotion: increment chunk access count for final answer_topk
                    # 论文 IV-C:命中计数 h(c)≥τ_hit 触发晋升后,子图"异步"固化提交到 G_L2
                    # H2:触发检查+子图提取(含 Milvus 查询)+写盘全部在后台线程,不阻塞事件循环
                    if not answer_aware_promotion and chunk_ids:
                        for cid in chunk_ids:
                            self._submit_l2(self._promote_chunk_in_background, cid)

                    # answer_aware mode: successful answer triggers entity-level promotion
                    # H2:实体级晋升(含 Nebula FETCH 查询)整体后台执行,不阻塞事件循环
                    if answer_aware_promotion and final_retrieval_res and not self._is_dont_know(ra_answer):
                        self._submit_l2(self._promote_entities_after_answer,
                                        final_retrieval_res, ra_answer, entity_promotion_threshold)

                n_chunks = len(chunk_ids)
                self.logger.query_done(idx + 1, n_chunks, 0)
                gt = self.answers[start + idx] if start + idx < len(self.answers) else ""

                # Collect retrieval stats
                if final_retrieval_res and "stats" in final_retrieval_res:
                    retrieval_stats.append(final_retrieval_res["stats"])

                # 查询时延记录:供 P50/P95 与分阶段占比分析(R2-6.2 / R4-W3)
                l1_hits = l2_hits = l2_query_errors = 0
                retrieval_calls = 1
                if final_retrieval_res:
                    l1_hits = sum(len(m.get("chunk_scores", {})) for m in final_retrieval_res.get("memory", []))
                    l2_hits = sum(len(p.get("chunk_scores", {})) for p in final_retrieval_res.get("persistent", []))
                    l2_query_errors = int((final_retrieval_res.get("stats", {}) or {}).get("l2_query_errors", 0))
                elif use_agentic:
                    l1_hits = int(agentic_metrics.get("l1_hits", 0))
                    l2_hits = int(agentic_metrics.get("l2_hits", 0))
                    l2_query_errors = int(agentic_metrics.get("l2_query_errors", 0))
                    retrieval_calls = int(agentic_metrics.get("retrieval_calls", 0))
                lat_rec = {
                    "query": question,
                    "mode": "agentic" if use_agentic else mode,
                    "retrieve_s": round(ret_ms, 4),
                    "answer_s": round(ans_ms, 4),
                    "l1_hits": l1_hits,
                    "l2_hits": l2_hits,
                    "l2_query_errors": l2_query_errors,
                    "retrieval_calls": retrieval_calls,
                    "stages": {k: round(v, 4) for k, v in stage_lat.items()},
                }
                item = {
                    "query": question,
                    "gt": gt,
                    "raw_answer": raw_answer,
                    "predict": ra_answer,
                    "chunk": chunk_ids,
                    "retry": retry_count,
                }
                if use_agentic:
                    item.update({
                        "agentic_steps_count": len(result.get("agentic_steps", [])),
                        "block_count": result.get("block_count", 0),
                        "beam_width": result.get("beam_width", 1),
                        "beam_score": result.get("beam_score", 0),
                        "agentic_parallel_workers": result.get("parallel_workers", 1),
                        "agentic_planning_calls": result.get("planning_calls", 0),
                        "agentic_batched_planning_calls": result.get("batched_planning_calls", 0),
                        "agentic_retrieval_calls": result.get("retrieval_calls", 0),
                        "agentic_l1_hits": result.get("l1_hits", 0),
                        "agentic_l2_hits": result.get("l2_hits", 0),
                        "agentic_l2_query_errors": result.get("l2_query_errors", 0),
                    })
                with _rlock:
                    results.append(item)
                    latency_records.append(lat_rec)
                    if qa_cache_file:
                        results.sort(key=lambda x: questions.index(x["query"]))
                        save_to_json(qa_cache_file, results, indent=2, info=False)
                if qa_concurrency == 1:
                    self.logger.buffer_off()
                self.logger.qa_progress()

        if qa_concurrency > 1:
            self.logger.buffer_off()
        self.logger.set_qa_mode(len(questions))
        retrieval_stats = []  # Collect retrieval stats
        tasks = [_process_one(i, q) for i, q in enumerate(questions)]
        await asyncio.gather(*tasks)
        self.logger.qa_done()

        # Retrieval source stats
        if retrieval_stats:
            avg_graph = sum(s["graph_in_final"] for s in retrieval_stats) / len(retrieval_stats)
            avg_dpr = sum(s["dpr_in_final"] for s in retrieval_stats) / len(retrieval_stats)
            avg_entities = sum(s["n_entities"] for s in retrieval_stats) / len(retrieval_stats)
            avg_coverage = sum(s["n_chunks_covered"] for s in retrieval_stats) / len(retrieval_stats)
            graph_zero = sum(1 for s in retrieval_stats if s["graph_in_final"] == 0)
            print(f"\n  [Retrieval Stats] avg entity={avg_entities:.1f} | "
                  f"graph chunks={avg_graph:.1f} | DPR chunks={avg_dpr:.1f} | "
                  f"zero-graph queries={graph_zero}/{len(retrieval_stats)}")

        # Sort by original order
        results.sort(key=lambda x: questions.index(x["query"]))

        os.makedirs("output/qa", exist_ok=True)
        save_to_json(f"output/qa/qa_results_{self.dataset}_{start}_{end}.json", results, indent=2, info=False)
        # 保存查询时延明细(含分阶段耗时),供 P50/P95 与阶段占比分析
        if latency_records:
            save_to_json(f"output/qa/qa_latency_{self.dataset}_{start}_{end}.json", latency_records, indent=2, info=False)

        elapsed = time.time() - start_time
        self._retrieval_time.append(elapsed)
        self._latency_records = latency_records
        self._total_io_count += self.mem_graph.nebula_IO_count
        print(f"query done: {elapsed:.2f}s, nebula_IO: {self.mem_graph.nebula_IO_count}")
        self.mem_graph.show_status()
        return results

    # ── Index Only ─────────────────────────────────────────

    async def index(self, start: int = 0, end: int = -1, build_phase: str = "base"):
        """Document ingestion and graph index building only (no retrieval).

        Args:
            start: Start index
            end: End index
            build_phase: "base" or "incremental", controls whoqa dataset phase_1/phase_2 loading
        """
        self.build_phase = build_phase
        try:
            limit = end if end > 0 else 0
            texts = self.load_corpus(limit=limit)
        except ValueError:
            self.load_dataset(start, end)
            texts = self.texts
        await self.ingest(texts)
        # 构建后 L1 拓扑快照(gexf)仅为观察/调试用,不计入存储审计(R4-W2 口径)。
        # 默认 data.save_gexf=false 不写;需要观察 L1 常驻拓扑时设 true。
        if bool(get_config().get("data", {}).get("save_gexf", False)):
            cc = get_config().get("retrieval", {}).get("chunk_collection", self.nebula_space)
            prefix = f"subgraph/base/{self.dataset}_{cc}_{self.nebula_space}_{start}_{end}"
            suffix = "update" if build_phase == "incremental" else build_phase
            gexf_name = f"{prefix}_{suffix}.gexf"
            self.mem_graph.save_graph_gexf(gexf_name)
            # Save phase1 snapshot separately, not overwritten by phase2
            if build_phase == "base":
                phase1_name = f"{prefix}_phase1.gexf"
                self.mem_graph.save_graph_gexf(phase1_name)
            # Phase2 saves as _base.gexf for downstream index_only_qa
            if build_phase == "incremental":
                base_name = f"{prefix}_base.gexf"
                self.mem_graph.save_graph_gexf(base_name)
        else:
            print("  [gexf] data.save_gexf=false:跳过 L1 拓扑快照(仅观察用,不影响检索/晋升/L2)")
        print(f"\nIndex build done ({build_phase}), processed {len(texts)} docs.")

    async def index_only_qa(self, start: int = 0, end: int = -1,
                            answer_aware_promotion: bool = False,
                            entity_promotion_threshold: int = 3,
                            promotion_retries: int = 2,
                            promotion_expansion: int = 10,
                            qa_concurrency: int = 5,
                            entity_extraction: str = "llm",
                            qa_cache: bool = False,
                            use_agentic: bool = True,
                            agentic_steps: int = 3,
                            entity_index_name: Optional[str] = None,
                            chunk_collection: Optional[str] = None,
                            answer_topk: int = 6,
                            top_chunks: Optional[int] = None,
                            mode: str = "hybrid",
                            clear_l2: bool = True,
                            resume_gexf: Optional[str] = None,
                            warmup_ratio: float = 0.0,
                            warmup_seed: int = 42):
        """Skip indexing, load existing data and run QA.

        Default: copy a base gexf, run QA on the copy to avoid polluting the base.
        If clear_l2=False and resume_gexf is set, load the gexf and keep L2 (incremental mode).

        warmup_ratio>0(中档#11):按 seed 把查询切分为不相交的预热集/测试集,
        先用预热集(允许晋升 L2,代表冷启动)再用测试集评估(预热后),输出两组指标。
        """
        if clear_l2 and self.mem_graph.persistent_graph:
            # Clear L2 (Nebula) promoted vertices and edges to avoid contamination
            try:
                self.mem_graph.persistent_graph.clear()
                print(f"Cleared NebulaSpace: {self.nebula_space}")
            except Exception as e:
                print(f"Nebula clear failed (may already be empty): {e}")

        cfg = get_config()
        cc = cfg.get("retrieval", {}).get("chunk_collection", self.nebula_space)

        if resume_gexf and os.path.exists(resume_gexf):
            # Incremental mode: load specified gexf (v1 final state), keep L2
            from shutil import copy2
            from datetime import datetime
            qa_gexf = f"subgraph/test/{self.dataset}_{cc}_{self.nebula_space}_{start}_{end}_qa_{datetime.now():%Y%m%d_%H%M%S}.gexf"
            os.makedirs(os.path.dirname(qa_gexf), exist_ok=True)
            copy2(resume_gexf, qa_gexf)
            print(f"Incremental load gexf: {resume_gexf}")
            print(f"QA using copy: {qa_gexf}")
            try:
                self.mem_graph.load_graph_gexf(qa_gexf)
            except Exception as e:
                    print(f"Failed to load graph snapshot: {e}")
        else:
            # Baseline mode:默认 L1 空启动(load_gexf=false)——不 copy base gexf,
            # QA 期由 DPR 命中触发 rehydrate_chunk_from_milvus 从 Milvus graph_meta
            # 按 chunk 重载填充 L1(论文 IV-C Lazy Rehydration,冷启动→热化模型)。
            # 仅 load_gexf=true(旧行为/观察)时才 copy 并加载 base gexf 作为 QA 初始 L1。
            if cfg.get("retrieval", {}).get("load_gexf", False):
                base_gexf = f"subgraph/base/{self.dataset}_{cc}_{self.nebula_space}_{start}_{end}_base.gexf"
                if not os.path.exists(base_gexf):
                    import glob
                    candidates = glob.glob(f"subgraph/base/{self.dataset}_{cc}_{self.nebula_space}_*_base.gexf")
                    if candidates:
                        base_gexf = candidates[0]
                        target = f"subgraph/base/{self.dataset}_{cc}_{self.nebula_space}_{start}_{end}_base.gexf"
                        from shutil import copy2
                        copy2(base_gexf, target)
                        base_gexf = target
                if not os.path.exists(base_gexf):
                    base_gexf = f"subgraph/base/memory_graph_{self.nebula_space}.gexf"
                if os.path.exists(base_gexf):
                    from shutil import copy2
                    from datetime import datetime
                    qa_gexf = f"subgraph/test/{self.dataset}_{cc}_{self.nebula_space}_{start}_{end}_qa_{datetime.now():%Y%m%d_%H%M%S}.gexf"
                    os.makedirs(os.path.dirname(qa_gexf), exist_ok=True)
                    copy2(base_gexf, qa_gexf)
                    print(f"Base graph snapshot: {base_gexf}")
                    print(f"QA using copy: {qa_gexf}")
                    try:
                        self.mem_graph.load_graph_gexf(qa_gexf)
                    except Exception as e:
                        print(f"Failed to load graph snapshot: {e}")
            else:
                print("[QA] L1 空启动(lazy rehydration):不加载 base gexf,首个查询起按 chunk 从 Milvus graph_meta 重载填充")
        self.load_dataset(start, end)
        tag = "agentic" if use_agentic else "baseline"
        if warmup_ratio > 0:
            # 中档#11:预热/测试隔离协议,输出冷启动 vs 预热后两组指标
            return await self._run_split_qa(
                start=start, end=end, use_agentic=use_agentic,
                agentic_steps=agentic_steps,
                answer_aware_promotion=answer_aware_promotion,
                entity_promotion_threshold=entity_promotion_threshold,
                promotion_retries=promotion_retries,
                promotion_expansion=promotion_expansion,
                qa_concurrency=qa_concurrency,
                entity_extraction=entity_extraction,
                qa_cache=qa_cache,
                entity_index_name=entity_index_name,
                chunk_collection=chunk_collection,
                answer_topk=answer_topk, top_chunks=top_chunks,
                mode=mode, tag=tag,
                warmup_ratio=warmup_ratio, warmup_seed=warmup_seed)
        cache_file = f"output/qa/qa_results_{self.dataset}_{tag}_{start}_{end}.json" if qa_cache else None
        await self.query(
            questions=self.questions,
            start=start, end=end,
            use_agentic=use_agentic,
            agentic_steps=agentic_steps,
            answer_aware_promotion=answer_aware_promotion,
            entity_promotion_threshold=entity_promotion_threshold,
            promotion_retries=promotion_retries,
            promotion_expansion=promotion_expansion,
            qa_concurrency=qa_concurrency,
            entity_extraction=entity_extraction,
            qa_cache_file=cache_file,
            entity_index_name=entity_index_name,
            chunk_collection=chunk_collection,
            answer_topk=answer_topk,
            top_chunks=top_chunks,
            mode=mode,
        )

    async def _run_split_qa(self, start, end, use_agentic, agentic_steps,
                            answer_aware_promotion, entity_promotion_threshold,
                            promotion_retries, promotion_expansion, qa_concurrency,
                            entity_extraction, qa_cache, entity_index_name, chunk_collection,
                            answer_topk, top_chunks, mode, tag,
                            warmup_ratio, warmup_seed):
        """中档#11:预热/测试隔离协议(回应 R4-W2 / R4-9.7)。

        按 warmup_seed 把查询切分为严格不相交的预热集与测试集:
          阶段1 预热集(冷启动,L2 从空开始;允许晋升写 L2)
          阶段2 测试集(预热后,评估时 L2 已由预热集填充)
        分别输出 ACC/RougeL + L2 规模,并保存切分明细,保证可审计。
        """
        import random
        from src.eval import evaluate_qa

        os.makedirs("output/qa", exist_ok=True)
        qs = self.questions
        end = len(qs) if end == -1 or end > len(qs) else end
        idx = list(range(start, end))
        rng = random.Random(warmup_seed)
        rng.shuffle(idx)
        n_warm = max(1, int(len(idx) * warmup_ratio))
        warm_idx = idx[:n_warm]
        eval_idx = idx[n_warm:]

        def _subset(ids):
            return [qs[i] for i in ids], [self.answers[i] for i in ids]

        warm_q, warm_a = _subset(warm_idx)
        eval_q, eval_a = _subset(eval_idx)
        print(f"  [warmup-split] seed={warmup_seed} warmup_n={len(warm_q)} "
              f"({100 * len(warm_q) / max(1, len(idx)):.0f}%) | "
              f"eval_n={len(eval_q)} | 预热∩测试=∅")

        def _l2_size():
            try:
                return {"nodes": len(self.mem_graph.l2_written_nodes),
                        "edges": len(self.mem_graph.l2_written_edges),
                        "promoted_chunks": len(self.mem_graph.promoted_chunks)}
            except Exception:
                return {}

        def _metric(results, questions, answers):
            # query() 内部 gt 按位置对齐 answers[start+idx],对随机子集会错位,
            # 这里用传入 (questions, answers) 重建 ground truth(与 diagnose_leakage 一致)
            qa_map = dict(zip(questions, answers))
            preds, gts = [], []
            for r in results:
                preds.append(r.get("predict", ""))
                gts.append(qa_map.get(r.get("query", ""), ""))
            m = evaluate_qa(preds, gts, use_rougel=True, use_bert=False)
            return m

        common = dict(
            start=start, end=end, use_agentic=use_agentic,
            agentic_steps=agentic_steps,
            answer_aware_promotion=answer_aware_promotion,
            entity_promotion_threshold=entity_promotion_threshold,
            promotion_retries=promotion_retries,
            promotion_expansion=promotion_expansion,
            qa_concurrency=qa_concurrency,
            entity_extraction=entity_extraction,
            entity_index_name=entity_index_name,
            chunk_collection=chunk_collection,
            answer_topk=answer_topk, top_chunks=top_chunks, mode=mode)

        # ── 阶段 1:预热集(冷启动,允许晋升)──
        l2_before_warm = _l2_size()
        warm_cache = (f"output/qa/qa_cache_{self.dataset}_{tag}_warmup_{warmup_seed}_{start}_{end}.json"
                      if qa_cache else None)
        warm_res = await self.query(questions=warm_q, qa_cache_file=warm_cache, **common)
        l2_after_warm = _l2_size()
        warm_metrics = _metric(warm_res, warm_q, warm_a)
        save_to_json(f"output/qa/qa_warmup_{self.dataset}_{tag}_{warmup_seed}_{start}_{end}.json",
                     warm_res, indent=2, info=False)

        # ── 阶段 2:测试集(预热后)──
        l2_before_eval = _l2_size()
        eval_cache = (f"output/qa/qa_cache_{self.dataset}_{tag}_eval_{warmup_seed}_{start}_{end}.json"
                      if qa_cache else None)
        eval_res = await self.query(questions=eval_q, qa_cache_file=eval_cache, **common)
        l2_after_eval = _l2_size()
        eval_metrics = _metric(eval_res, eval_q, eval_a)
        save_to_json(f"output/qa/qa_test_{self.dataset}_{tag}_{warmup_seed}_{start}_{end}.json",
                     eval_res, indent=2, info=False)

        report = {
            "dataset": self.dataset, "slice": [start, end],
            "warmup_seed": warmup_seed, "warmup_ratio": warmup_ratio,
            "warmup_n": len(warm_q), "eval_n": len(eval_q),
            "disjoint": True,  # seed 切分保证预热∩测试=∅
            "cold_start": {"acc": warm_metrics["em"], "rougel": warm_metrics.get("rougel"),
                           "count": warm_metrics.get("count"),
                           "l2_before": l2_before_warm, "l2_after": l2_after_warm},
            "warm_after": {"acc": eval_metrics["em"], "rougel": eval_metrics.get("rougel"),
                           "count": eval_metrics.get("count"),
                           "l2_before": l2_before_eval, "l2_after": l2_after_eval},
        }
        out_path = f"output/qa/warmup_report_{self.dataset}_{tag}_{warmup_seed}_{start}_{end}.json"
        save_to_json(out_path, report, indent=2, info=False)
        print(f"\n  [warmup-report] 冷启动(cold) ACC={report['cold_start']['acc']:.4f} | "
              f"预热后(warm) ACC={report['warm_after']['acc']:.4f} | "
              f"Δ={report['warm_after']['acc'] - report['cold_start']['acc']:+.4f}")
        print(f"  [warmup-report] 保存: {out_path}")
        return eval_res


    # ── Full Pipeline ────────────────────────────────────────────

    async def run(
        self,
        start: int = 0,
        end: int = -1,
        use_agentic: bool = True,
        agentic_steps: int = 3,
        stream_mode: bool = False,
        batch_size: int = 20,
        noise_flush_docs: int = 0,
        answer_aware_promotion: bool = False,
        entity_promotion_threshold: int = 3,
        promotion_retries: int = 2,
        promotion_expansion: int = 10,
        qa_concurrency: int = 5,
        entity_extraction: str = "llm",
        qa_cache: bool = False,
        entity_index_name: Optional[str] = None,
        chunk_collection: Optional[str] = None,
        answer_topk: int = 6,
        top_chunks: Optional[int] = None,
        mode: str = "hybrid",
    ):
        # Build index from corpus (same as index()), not QA text
        await self.index(start=start, end=end)

        # Load QA data and run queries
        self.load_dataset(start, end)
        self.mem_graph.show_status()
        cache_file = f"output/qa/qa_cache_{self.dataset}_{start}_{end}_{entity_extraction}.json" if qa_cache else None
        await self.query(self.questions, start, end, use_agentic, agentic_steps,
                         answer_aware_promotion=answer_aware_promotion,
            entity_promotion_threshold=entity_promotion_threshold,
                         promotion_retries=promotion_retries,
                         promotion_expansion=promotion_expansion,
                         qa_concurrency=qa_concurrency,
                         entity_extraction=entity_extraction,
                         qa_cache_file=cache_file,
                         entity_index_name=entity_index_name,
                         chunk_collection=chunk_collection,
                         answer_topk=answer_topk,
                         top_chunks=top_chunks,
                         mode=mode)

        self._print_summary()

    async def _run_batched(self, start, end, use_agentic, agentic_steps, stream_mode, batch_size, noise_flush_docs,
                           answer_aware_promotion=False, promotion_retries=2, promotion_expansion=10,
                           qa_concurrency=5, top_chunks=None):
        batch_files = []
        merged_output = f"data/qa_results_{self.dataset}_{start}_{end}.json"
        batch_index = 0

        for offset in range(0, len(self.texts), batch_size):
            batch_start = start + offset
            batch_end = min(start + offset + batch_size, end)
            batch_no = batch_index + 1
            run_qa = not (stream_mode and batch_no % 2 == 0)

            await self.ingest(self.texts[offset:offset + batch_size])

            if run_qa:
                q_slice = self.questions[offset:offset + batch_size]
                await self.query(q_slice, batch_start, batch_end, use_agentic, agentic_steps,
                                 answer_aware_promotion=answer_aware_promotion,
                                 promotion_retries=promotion_retries,
                                 promotion_expansion=promotion_expansion,
                                 qa_concurrency=qa_concurrency,
                                 top_chunks=top_chunks)
                batch_files.append(f"data/qa_results_{self.dataset}_{batch_start}_{batch_end}.json")
                total = self._merge_retrieval_files(batch_files, merged_output)
                print(f"merged {total} items -> {merged_output}")
            elif stream_mode and noise_flush_docs > 0:
                noise = [f"noise doc {i} " + ("lorem " * 120) for i in range(noise_flush_docs)]
                await self.ingest(noise)

            batch_index += 1

    # ── Utility Methods ────────────────────────────────────────────

    @staticmethod
    def _merge_retrieval_files(input_files: List[str], output_file: str) -> int:
        merged = []
        for path in input_files:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing: {path}")
            data = read_json(path)
            if isinstance(data, list):
                merged.extend(data)
        save_to_json(output_file, merged, indent=2, info=False)
        return len(merged)

    @staticmethod
    def _is_dont_know(answer: str) -> bool:
        if not answer or answer == "I don't know.":
            return True
        lowered = answer.lower()
        return "i don't know" in lowered or "cannot answer" in lowered or "not enough information" in lowered

    @staticmethod
    def _check_databases():
        """Check Milvus and NebulaGraph connectivity on startup."""
        import sys as _sys
        ok = True

        # ── Check Milvus ──
        try:
            from database.milvus import myMilvus
            m = myMilvus()
            m.list_collections()
            print("[✓] Milvus connected")
        except Exception as e:
            print(f"[✗] Milvus connection failed: {e}")
            ok = False

        # ── Check NebulaGraph ──
        try:
            nc = NebulaClient()
            nc.show_space()
            print("[✓] NebulaGraph connected")
        except Exception as e:
            print(f"[✗] NebulaGraph connection failed: {e}")
            ok = False

        if not ok:
            print("\n⚠ Database check failed. Ensure Docker containers are running:")
            print("   bash database/setup/milvus-install-user.sh start")
            print("   bash database/setup/nebula-install-user.sh install")
            _sys.exit(1)

    def _submit_l2(self, fn, *args) -> None:
        """统一 L2 后台任务提交(M3 有界背压 + executor 关闭兜底)。

        - 积压 pending ≥ _l2_max_pending 时退化为同步执行,防止无界队列与写滞后;
        - executor 已关闭(RuntimeError)时同步兜底,保证晋升数据不丢。
        """
        with self._l2_pending_lock:
            if self._l2_pending >= self._l2_max_pending:
                overloaded = True
            else:
                self._l2_pending += 1
                overloaded = False
        if overloaded:
            try:
                fn(*args)
            except Exception as exc:
                print(f"[L2-writer] sync fallback failed: {exc}")
            return

        def _run():
            try:
                fn(*args)
            except Exception as exc:
                print(f"[L2-writer] async task failed: {exc}")
            finally:
                with self._l2_pending_lock:
                    self._l2_pending -= 1

        try:
            self._l2_writer.submit(_run)
        except RuntimeError:
            # executor 已关闭(进程退出竞态):退化为同步写,保证晋升数据不丢失
            with self._l2_pending_lock:
                self._l2_pending -= 1
            try:
                fn(*args)
            except Exception as exc:
                print(f"[L2-writer] fallback persist failed: {exc}")

    def _submit_l2_persist(self, subgraph_data: list) -> None:
        """论文 IV-C:将 chunk 晋升子图异步固化到 G_L2(后台线程,不阻塞 QA)。"""
        self._submit_l2(self.mem_graph.write_to_persistent_graph, subgraph_data)

    def _submit_l2_promote(self, ents: dict, edges: list) -> None:
        """论文 IV-C:实体级晋升同样异步固化到 G_L2。"""
        self._submit_l2(self.mem_graph.promote_subgraph, ents, edges)

    def _promote_chunk_in_background(self, cid: str) -> None:
        """H2:chunk 晋升触发(命中计数 + 子图提取,含 Milvus 查询)与 L2 固化整体后台完成。

        原先 access_chunk 在事件循环内同步调用,_extract_subgraph_by_chunk 内部的
        Milvus 查询会阻塞所有并发查询;现整体挪入 L2 worker 线程。
        """
        try:
            triggered, data = self.mem_graph.access_chunk(cid)
            if triggered:
                self.mem_graph.write_to_persistent_graph([data])
        except Exception as exc:
            print(f"[L2-writer] promote chunk {cid} failed: {exc}")

    def _promote_entities_after_answer(self, final_retrieval_res, ra_answer: str,
                                       entity_promotion_threshold: int) -> None:
        """H2:answer_aware 模式实体级晋升(含 Nebula FETCH 查询)整体后台执行。"""
        try:
            if not final_retrieval_res or self._is_dont_know(ra_answer):
                return
            ents = {}
            edges = []
            promote_uids = set()
            for m in final_retrieval_res.get("memory", []):
                for e in m.get("matched_entities", []):
                    uid = e.get("uid")
                    if uid is not None and self.mem_graph.graph.has_node(uid):
                        str_uid = str(uid)
                        self.mem_graph.entity_access_counter[str_uid] = self.mem_graph.entity_access_counter.get(str_uid, 0) + 1
                        if self.mem_graph.entity_access_counter[str_uid] >= entity_promotion_threshold:
                            promote_uids.add(uid)
                            del self.mem_graph.entity_access_counter[str_uid]
            for uid in promote_uids:
                # M4:锁内快照读节点数据
                ents[uid] = self.mem_graph.snapshot_node_data(uid)
            seen_edge_keys = set()
            for uid in promote_uids:
                for u, v, key, data in self.mem_graph.snapshot_edges(uid, "out", keys=True):
                    ek = (u, v, key)
                    if ek not in seen_edge_keys:
                        seen_edge_keys.add(ek)
                        edges.append({"src": u, "tgt": v, "relation": data.get("relation_type") or data.get("relation", "")})
                for u, v, key, data in self.mem_graph.snapshot_edges(uid, "in", keys=True):
                    ek = (u, v, key)
                    if ek not in seen_edge_keys:
                        seen_edge_keys.add(ek)
                        edges.append({"src": u, "tgt": v, "relation": data.get("relation_type") or data.get("relation", "")})
            # Include neighbor entities in L2 (NebulaGraph edges need both endpoints)
            for e in edges:
                for nuid in (e["src"], e["tgt"]):
                    if nuid not in ents and self.mem_graph.graph.has_node(nuid):
                        ents[nuid] = self.mem_graph.snapshot_node_data(nuid)
            # Check which neighbors of promoted entities are in L2, add missing edges
            neighbor_uids = set()
            for uid in promote_uids:
                for u, v, _k, _d in self.mem_graph.snapshot_edges(uid, "out", keys=True):
                    if v not in ents:
                        neighbor_uids.add(v)
                for u, v, _k, _d in self.mem_graph.snapshot_edges(uid, "in", keys=True):
                    if u not in ents:
                        neighbor_uids.add(u)
            if neighbor_uids:
                nebula = self.mem_graph.persistent_graph
                vid_list = [nebula._format_vid(n) for n in neighbor_uids]
                fetch_q = (
                    "FETCH PROP ON `entity` "
                    f"{', '.join(vid_list)} "
                    "YIELD id(vertex) AS vid;"
                )
                try:
                    rows = nebula.query(fetch_q)
                    existing_l2 = set(str(v) for v in rows.get("vid", []) if v)
                except Exception:
                    existing_l2 = set()
                for uid in promote_uids:
                    for u, v, key, data in self.mem_graph.snapshot_edges(uid, "out", keys=True):
                        if str(v) in existing_l2 and (u, v, key) not in seen_edge_keys:
                            seen_edge_keys.add((u, v, key))
                            edges.append({"src": u, "tgt": v, "relation": data.get("relation_type") or data.get("relation", "")})
                    for u, v, key, data in self.mem_graph.snapshot_edges(uid, "in", keys=True):
                        if str(u) in existing_l2 and (u, v, key) not in seen_edge_keys:
                            seen_edge_keys.add((u, v, key))
                            edges.append({"src": u, "tgt": v, "relation": data.get("relation_type") or data.get("relation", "")})
            if ents:
                self.mem_graph.promote_subgraph(ents, edges)
                self.logger._print(f"  L2 promoted {len(ents)} entities, {len(edges)} edges")
        except Exception as exc:
            print(f"[L2-writer] answer-aware promote failed: {exc}")

    def shutdown(self):
        if getattr(self, "_shutdown_complete", False):
            return
        self._shutdown_complete = True
        # flush 异步 L2 固化队列(论文 IV-C:异步提交,退出前必须全部落盘)
        try:
            self._l2_writer.shutdown(wait=True)
        except Exception as exc:
            print(f"[L2-writer] shutdown error: {exc}")
        # M1:回收 QA 专用线程池
        try:
            self._qa_executor.shutdown(wait=True)
        except Exception as exc:
            print(f"[qa-executor] shutdown error: {exc}")
        self.mem_graph.shutdown()
        self.logger.set_token_usage(self.llm.total_prompt_tokens, self.llm.total_completion_tokens)
        self.logger.set_runtime_metrics(
            llm_calls=getattr(self.llm, "total_calls", 0),
            llm_max_concurrency=getattr(self.llm, "max_concurrency", 1),
            llm_max_inflight_observed=getattr(self.llm, "max_inflight_observed", 0),
            embedding_api_calls=getattr(self.pipeline, "embedding_api_calls", 0),
            embedding_items=getattr(self.pipeline, "embedding_items", 0),
            evicted_chunks=self.mem_graph.evicted_chunks,
            evicted_nodes=self.mem_graph.evicted_nodes,
            evicted_edges=self.mem_graph.evicted_edges,
            rehydrate_attempts=self.mem_graph.rehydrate_attempts,
            rehydrate_successes=self.mem_graph.rehydrate_successes,
            rehydrate_failures=self.mem_graph.rehydrate_failures,
            rehydrated_nodes=self.mem_graph.rehydrated_nodes,
            rehydrated_edges=self.mem_graph.rehydrated_edges,
            l1_nodes=self.mem_graph.graph.number_of_nodes(),
            l1_edges=self.mem_graph.graph.number_of_edges(),
        )
        self.logger.summary()
        build_info =(
            f"Build Summary | dataset={self.dataset} "
            f"time={round(sum(self._ingestion_time), 1)}s "
            f"docs={len(self.texts) if self.texts else 0} "
            f"chunks={self.logger._chunk_count} "
            f"entities={self.logger._entity_count} "
            f"triplets={self.logger._triplet_count} "
            f"tokens={self.llm.total_prompt_tokens + self.llm.total_completion_tokens} "
            f"warnings={len(self.logger.metrics.get('warnings', []))}"
        )
        self.logger._print(build_info)
        self.logger.close()

    def _print_summary(self):
        print("\n--- [Summary] ---")
        ing_total = sum(self._ingestion_time)
        ret_total = sum(self._retrieval_time)
        prompt_tok = self.llm.total_prompt_tokens
        completion_tok = self.llm.total_completion_tokens
        llm_calls = getattr(self.llm, "total_calls", 0)
        print(f"  total ingestion: {ing_total:.2f}s  (avg: {(ing_total / len(self._ingestion_time)) if self._ingestion_time else 0:.2f}s)")
        print(f"  total retrieval: {ret_total:.2f}s  (avg: {(ret_total / len(self._retrieval_time)) if self._retrieval_time else 0:.2f}s)")
        print(f"  total NebulaGraph IO: {self._total_io_count}")
        if self.mem_graph is not None:
            print(f"  L1 evicted: {self.mem_graph.evicted_chunks} chunks, "
                  f"{self.mem_graph.evicted_nodes} nodes, {self.mem_graph.evicted_edges} edges")
            print(f"  topology rehydration: {self.mem_graph.rehydrate_successes}/"
                  f"{self.mem_graph.rehydrate_attempts} succeeded "
                  f"({self.mem_graph.rehydrate_failures} failed), "
                  f"restored {self.mem_graph.rehydrated_nodes} nodes / "
                  f"{self.mem_graph.rehydrated_edges} edges")
        if self.pipeline is not None and getattr(self.pipeline, "embedding_api_calls", 0):
            embed_calls = self.pipeline.embedding_api_calls
            embed_items = self.pipeline.embedding_items
            print(f"  embedding batches: {embed_calls} API calls, {embed_items} texts, "
                  f"avg batch size={embed_items / embed_calls:.2f}")
        if prompt_tok or completion_tok:
            print(f"  LLM token usage: {prompt_tok} input + {completion_tok} output = {prompt_tok + completion_tok} total")
            print(f"  LLM calls: {llm_calls}")
            # 成本估算(按 gpt-4o-mini 官价 $0.15/M input + $0.60/M output)
            cost = prompt_tok * 0.15 / 1e6 + completion_tok * 0.60 / 1e6
            print(f"  LLM cost estimate: ${cost:.4f} (gpt-4o-mini pricing)")

        # 查询时延分位统计(R2-6.2 / R4-W3)
        recs = getattr(self, "_latency_records", None)
        if recs:
            import statistics
            tots = [r["retrieve_s"] + r["answer_s"] for r in recs]
            rets = [r["retrieve_s"] for r in recs]
            ans = [r["answer_s"] for r in recs]

            def _pct(arr, p):
                if not arr:
                    return 0.0
                s = sorted(arr)
                k = min(len(s) - 1, int(round(p / 100.0 * (len(s) - 1))))
                return s[k]

            print(f"\n  [Query Latency] ({len(tots)} queries)")
            print(f"    P50={_pct(tots, 50):.3f}s  P95={_pct(tots, 95):.3f}s  avg={sum(tots) / len(tots):.3f}s  "
                  f"(retrieve avg={sum(rets) / len(rets):.3f}s + answer avg={sum(ans) / len(ans):.3f}s)")
            # 冷启动(L1 未命中)vs 稳态(L1 命中)分组
            hot = [r for r in recs if r["l1_hits"] > 0]
            cold = [r for r in recs if r["l1_hits"] == 0]
            if hot:
                h_tots = [r["retrieve_s"] + r["answer_s"] for r in hot]
                print(f"    L1-hit (hot)  queries={len(hot)}  P50={_pct(h_tots, 50):.3f}s  P95={_pct(h_tots, 95):.3f}s")
            if cold:
                c_tots = [r["retrieve_s"] + r["answer_s"] for r in cold]
                print(f"    L1-miss(cold) queries={len(cold)}  P50={_pct(c_tots, 50):.3f}s  P95={_pct(c_tots, 95):.3f}s")
            # 检索内部分阶段占比
            stage_keys = ["embed", "dpr", "entity", "graph", "aggregate", "fusion"]
            active = [k for k in stage_keys if any(r.get("stages", {}).get(k, 0) for r in recs)]
            if active:
                total_ret = sum(rets)
                parts = "  ".join(
                    f"{k}={sum(r.get('stages', {}).get(k, 0) for r in recs) / total_ret * 100:.1f}%"
                    for k in active
                ) if total_ret else ""
                if parts:
                    print(f"    检索阶段占比: {parts}")


# ── CLI Entry (all config from config.yaml)────────────────

if __name__ == "__main__":
    cfg = get_config()
    data_cfg = cfg.get("data", {})
    ret_cfg = cfg.get("retrieval", {})
    # 入口处显式设置一次全局种子,覆盖手动实例化(不走 from_config)的场景
    set_global_seed(cfg.get("model", {}).get("seed", 42))

    app = CacheGraphRAG.from_config(cfg)

    if ret_cfg.get("index_only", False):
        # Incremental experiment: Phase 1 (base) → Phase 2 (incremental)
        inc_exp = cfg.get("incremental_experiment", {})
        if inc_exp.get("enabled") and "whoqa" in data_cfg.get("dataset", ""):
            # Clear Nebula L2 to start from scratch
            if app.mem_graph.persistent_graph:
                try:
                    app.mem_graph.persistent_graph.clear()
                    print("Cleared Nebula L2, starting incremental build from scratch")
                except Exception as e:
                    print(f"Nebula clear failed: {e}")
            print("=" * 60)
            print("  Incremental Phase 1: build base graph with phase_1_data (distractors)")
            print("=" * 60)
            asyncio.run(app.index(
                start=data_cfg.get("start", 0),
                end=data_cfg.get("end", -1),
                build_phase="base",
            ))
            print("\n" + "=" * 60)
            print("  Phase 1 done, starting Phase 2: incrementally add phase_2_data (target)")
            print("=" * 60 + "\n")
            asyncio.run(app.index(
                start=data_cfg.get("start", 0),
                end=data_cfg.get("end", -1),
                build_phase="incremental",
            ))
        else:
            asyncio.run(app.index(
                start=data_cfg.get("start", 0),
                end=data_cfg.get("end", -1),
            ))
    elif ret_cfg.get("skip_index", False):
        start = data_cfg.get("start", 0)
        end = data_cfg.get("end", -1)
        entity_extraction = ret_cfg.get("entity_extraction", "llm")
        entity_index_name = ret_cfg.get("entity_index_name", None)
        chunk_collection = ret_cfg.get("chunk_collection", None)
        asyncio.run(app.index_only_qa(
            start=start, end=end,
            answer_aware_promotion=ret_cfg.get("answer_aware_promotion", False),
            entity_promotion_threshold=ret_cfg.get("entity_promotion_threshold", 3),
            promotion_retries=ret_cfg.get("promotion_retries", 2),
            promotion_expansion=ret_cfg.get("promotion_expansion", 10),
            qa_concurrency=ret_cfg.get("qa_concurrency", 5),
            entity_extraction=entity_extraction,
            qa_cache=ret_cfg.get("qa_cache", False),
            use_agentic=ret_cfg.get("agentic", True),
            agentic_steps=ret_cfg.get("agentic_steps", 3),
            entity_index_name=entity_index_name,
            chunk_collection=chunk_collection,
            answer_topk=ret_cfg.get("answer_topk", 6),
            top_chunks=ret_cfg.get("top_chunks", None),
            mode=ret_cfg.get("mode", "hybrid"),
            clear_l2=ret_cfg.get("clear_l2", True),
            resume_gexf=ret_cfg.get("resume_gexf", None),
            warmup_ratio=ret_cfg.get("warmup_ratio", 0.0),
            warmup_seed=ret_cfg.get("warmup_seed", 42),
        ))
    else:
        qa_concurrency = ret_cfg.get("qa_concurrency", 5)
        entity_extraction = ret_cfg.get("entity_extraction", "llm")
        entity_index_name = ret_cfg.get("entity_index_name", None)
        chunk_collection = ret_cfg.get("chunk_collection", None)
        asyncio.run(app.run(
            start=data_cfg.get("start", 0),
            end=data_cfg.get("end", -1),
            use_agentic=ret_cfg.get("agentic", True),
            agentic_steps=ret_cfg.get("agentic_steps", 3),
            stream_mode=ret_cfg.get("stream_mode", False),
            batch_size=ret_cfg.get("batch_size", 20),
            noise_flush_docs=ret_cfg.get("noise_flush_docs", 0),
            answer_aware_promotion=ret_cfg.get("answer_aware_promotion", False),
            promotion_retries=ret_cfg.get("promotion_retries", 2),
            promotion_expansion=ret_cfg.get("promotion_expansion", 10),
            qa_concurrency=qa_concurrency,
            entity_promotion_threshold=ret_cfg.get("entity_promotion_threshold", 3),
            entity_extraction=entity_extraction,
            qa_cache=ret_cfg.get("qa_cache", False),
            entity_index_name=entity_index_name,
            chunk_collection=chunk_collection,
            answer_topk=ret_cfg.get("answer_topk", 6),
            top_chunks=ret_cfg.get("top_chunks", None),
            mode=ret_cfg.get("mode", "hybrid"),
        ))
    app.shutdown()
