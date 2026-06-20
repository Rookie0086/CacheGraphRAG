"""Two-stage build pipeline: Stage 1 Extraction -> Stage 2 Ingestion (entity alignment + graph write + Embedding + vector store write)."""

import asyncio
import json
import os
import time
import uuid
from typing import Dict, List, Optional, Tuple

from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm.asyncio import tqdm as atqdm

from src.utils.prompts import prompt_extract_triplest_str
from src.utils.logger import get_logger
from src.utils.llm_cache import LLMCache
from src.utils.base import save_to_json
from src.llm.env import LLMEnv
from database.milvus import MilvusDB, myMilvus


class DocumentIngestionPipeline:
    def __init__(
        self,
        llm_client: LLMEnv,
        memory_graph,
        entity_resolver,
        collection_name="example",
        llm_concurrency=10,
        chunk_concurrency=5,
        build_concurrency=3,
        embed_model=None,
        chunk_size=800,
        chunk_overlap=100,
        fact_enabled=False,
        fact_collection_prefix="fact_index",
    ):
        self.llm = llm_client
        self.memory_graph = memory_graph
        self.entity_resolver = entity_resolver
        self.llm_semaphore = asyncio.Semaphore(llm_concurrency)
        self._chunk_semaphore = asyncio.Semaphore(chunk_concurrency) if chunk_concurrency > 0 else None
        self._build_semaphore = asyncio.Semaphore(build_concurrency) if build_concurrency > 0 else None
        self._build_concurrency = build_concurrency
        self.chunk_registry = {}
        self.llm_cache = LLMCache()
        self.fact_enabled = fact_enabled

        # Vector store (used in stage 2)
        self.vector_store = MilvusDB(db_name=collection_name, overwrite=False, embed_model=embed_model)
        self.vector_client = myMilvus()
        if collection_name not in self.vector_client.list_collections():
            self.vector_store.create_chunk_collection()
        else:
            self.vector_store.load()

        # Fact collection
        self.fact_store = None
        if fact_enabled:
            fact_coll_name = f"{fact_collection_prefix}_{collection_name}"
            self.fact_store = MilvusDB(db_name=fact_coll_name, overwrite=False, embed_model=embed_model)
            if fact_coll_name not in self.vector_client.list_collections():
                self.fact_store.create_fact_collection()
            else:
                self.fact_store.load()
            print(f"  Fact collection: {fact_coll_name}")

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", "。", ".", " ", ""]
        )

    # ── Stage 1: Extraction ──────────────────────────────────────

    async def extract(self, texts: List[str]) -> List[dict]:
        """Stage 1: Chunk all documents and perform LLM extraction, return raw extraction data."""
        log = get_logger()
        all_chunks = []
        for i, doc_text in enumerate(texts):
            chunks = self.text_splitter.split_text(doc_text)
            if log:
                log.doc_start(i, f"document_{i}.txt", len(chunks))
            tasks = []
            for text in chunks:
                chunk_id = f"chunk_{uuid.uuid4().hex[:12]}"
                self.chunk_registry[chunk_id] = text
                tasks.append(self._extract_single(chunk_id, text))

            if self._chunk_semaphore:
                async def limited(coro):
                    async with self._chunk_semaphore:
                        return await coro
                tasks = [limited(t) for t in tasks]

            doc_results = []
            with atqdm(total=len(tasks), desc=f"Extract doc {i}", unit="chunk", leave=False) as pbar:
                for coro in asyncio.as_completed(tasks):
                    try:
                        result = await coro
                        if result:
                            doc_results.append(result)
                            all_chunks.append(result)
                    except Exception:
                        pass
                    pbar.update(1)

            if log:
                log.doc_done(i, len(doc_results), len(chunks), 0)

        # Backup chunk mapping to JSON
        os.makedirs("data/chunk", exist_ok=True)
        save_to_json("data/chunk/chunks.json", self.chunk_registry, indent=1)

        if self.llm_cache.size > 0:
            print(f"  Cache stats: LLM cache has {self.llm_cache.size} entries")

        return all_chunks

    async def _extract_single(self, chunk_id: str, text: str) -> Optional[dict]:
        log = get_logger()
        prompt = prompt_extract_triplest_str.format(context=text)

        for attempt in range(10):
            try:
                async with self.llm_semaphore:
                    cached_str = self.llm_cache.get(prompt)
                    is_cached = cached_str is not None
                    if is_cached:
                        raw_json_str = cached_str
                    else:
                        raw_json_str = await self.llm.async_complete(prompt=prompt)
                        if raw_json_str is None:
                            raise RuntimeError("LLM API returned empty (rate limit/connection error)")
                        if raw_json_str:
                            self.llm_cache.put(prompt, raw_json_str)

                    entities, relations = self._clean_and_validate(raw_json_str)
                    if not entities:
                        if log:
                            log.warn(f"Chunk {chunk_id[:16]}... extraction empty (LLM returned no entities)")
                        return None
                    if log:
                        log.chunk_extracted(chunk_id, len(entities), len(relations), 0, cached=is_cached)
                    return {"chunk_id": chunk_id, "text": text, "entities": entities, "relations": relations}
            except Exception as e:
                if log:
                    log.warn(f"Chunk {chunk_id[:16]}... extraction error (attempt {attempt+1}/10): {e}")
                if attempt < 9:
                    await asyncio.sleep(3)

        if log:
            log.warn(f"Chunk {chunk_id[:16]}... all 10 retries failed, giving up")
        return None

    # ── Stage 2: Ingestion ──────────────────────────────────────

    async def build(self, extracted: List[dict]) -> int:
        """Stage 2: Entity alignment -> Graph write -> Embedding -> Vector store write.

        Args:
            extracted: List of raw extraction data returned by extract().

        Returns:
            Number of chunks successfully ingested.
        """
        if not extracted:
            return 0

        log = get_logger()
        sem = self._build_semaphore or asyncio.Semaphore(5)

        async def _build_limited(item):
            async with sem:
                await self._build_single(item)

        tasks = [_build_limited(item) for item in extracted]
        success = 0
        with atqdm(total=len(tasks), desc="Ingest", unit="chunk") as pbar:
            for coro in asyncio.as_completed(tasks):
                try:
                    await coro
                    success += 1
                except Exception as e:
                    if log:
                        log.warn(f"Ingest chunk failed: {e}")
                pbar.update(1)

        await self.entity_resolver.wait_pending()
        return success

    async def extract_and_build(self, texts: List[str]) -> Tuple[int, int]:
        """Pipeline parallel: extraction and ingestion run concurrently.

        Returns:
            (total_chunks, built_count)
        """
        log = get_logger()
        if not texts:
            return 0, 0

        # Phase 0: Pre-split all documents, register chunk_registry, get total chunk count
        all_chunk_infos = []  # [(doc_index, chunk_id, text)]
        for i, doc_text in enumerate(texts):
            chunks = self.text_splitter.split_text(doc_text)
            if log:
                log.doc_start(i, f"document_{i}.txt", len(chunks))
            for text in chunks:
                chunk_id = f"chunk_{uuid.uuid4().hex[:12]}"
                self.chunk_registry[chunk_id] = text
                all_chunk_infos.append((i, chunk_id, text))

        total = len(all_chunk_infos)
        if total == 0:
            return 0, 0

        from tqdm import tqdm
        pbar_extract = tqdm(total=total, desc="Extract", unit="chunk", position=0)
        pbar_build = tqdm(total=total, desc="Ingest", unit="chunk", position=1)

        queue: asyncio.Queue = asyncio.Queue()
        _SENTINEL = object()
        built_count = 0

        # Build consumer: take extraction results from queue, execute ingestion
        build_sem = self._build_semaphore or asyncio.Semaphore(5)

        async def _build_consumer():
            nonlocal built_count
            while True:
                item = await queue.get()
                if item is _SENTINEL:
                    queue.task_done()
                    break
                if item is None:
                    # LLM returned no entities, skip ingestion
                    queue.task_done()
                    pbar_build.update(1)
                    continue

                # Retry loop: semaphore held within each attempt, released during wait
                for attempt in range(10):
                    try:
                        async with build_sem:
                            await self._build_single(item)
                        built_count += 1
                        break  # Success, exit retry loop
                    except Exception as e:
                        err_msg = str(e)
                        if log:
                            log.warn(f"Ingest failed (attempt {attempt+1}/10) {item.get('chunk_id','')[:12]}: {err_msg[:80]}")
                        # collection not loaded -> try reloading
                        if "collection not loaded" in err_msg or "collection not found" in err_msg:
                            try:
                                self._reload_milvus_collections()
                            except Exception:
                                pass
                        if attempt < 9:
                            await asyncio.sleep(3)

                queue.task_done()
                pbar_build.update(1)

        # Extraction worker: LLM extracts one chunk, pushes into queue (success or skip)
        async def _extract_worker(doc_idx, chunk_id, text):
            if self._chunk_semaphore:
                async with self._chunk_semaphore:
                    result = await self._extract_single(chunk_id, text)
            else:
                result = await self._extract_single(chunk_id, text)
            await queue.put(result)  # None = skip (no entities)
            pbar_extract.update(1)

        num_consumers = max(1, self._build_concurrency or 5)
        offline_align = getattr(self, '_offline_alignment', False)

        if offline_align:
            # ── Offline batch alignment: extract all -> batch embed+align -> then ingest ──
            extract_tasks = [
                asyncio.create_task(_extract_worker(di, cid, txt))
                for di, cid, txt in all_chunk_infos
            ]
            await asyncio.gather(*extract_tasks, return_exceptions=True)
            pbar_extract.close()

            # Collect all entities
            all_entities = []
            while not queue.empty():
                item = queue.get_nowait()
                if item:
                    all_entities.append(item)
                    queue.task_done()

            if all_entities:
                pbar_build.set_description("Align")
                await self._batch_align_entities(all_entities)
                pbar_build.set_description("Ingest")
                # Put aligned results back into the queue for build to use
                for item in all_entities:
                    await queue.put(item)

            consumers = [asyncio.create_task(_build_consumer()) for _ in range(num_consumers)]
            for _ in range(num_consumers):
                await queue.put(_SENTINEL)
            await asyncio.gather(*consumers)
        else:
            consumers = [asyncio.create_task(_build_consumer()) for _ in range(num_consumers)]
            extract_tasks = [
                asyncio.create_task(_extract_worker(di, cid, txt))
                for di, cid, txt in all_chunk_infos
            ]
            await asyncio.gather(*extract_tasks, return_exceptions=True)
            pbar_extract.close()
            for _ in range(num_consumers):
                await queue.put(_SENTINEL)
            await asyncio.gather(*consumers)

        pbar_build.close()

        # Save chunk backup + wait for background registration to complete
        os.makedirs("data/chunk", exist_ok=True)
        save_to_json("data/chunk/chunks.json", self.chunk_registry, indent=1)
        await self.entity_resolver.wait_pending()

        if self.llm_cache.size > 0:
            print(f"  Cache stats: LLM cache has {self.llm_cache.size} entries")

        return total, built_count

    async def _batch_align_entities(self, extracted_items: list):
        """Offline batch entity alignment: collect all entities -> single embed -> batch align -> cache."""
        seen = set()
        all_entities = []
        for item in extracted_items:
            for ent in item["entities"]:
                key = f"{ent['id']}|{ent['type']}|{ent['desc']}"
                if key not in seen:
                    seen.add(key)
                    all_entities.append(ent)

        if not all_entities:
            return

        from tqdm import tqdm
        n = len(all_entities)
        texts = [f"Entity: {e['id']}. Type: {e['type']}. Description: {e['desc']}" for e in all_entities]

        with tqdm(total=n, desc="Align", unit="ent", position=1, leave=False) as pbar:
            pbar.set_postfix_str("Embedding...")
            vectors = await self.llm.embed_model.get_embeddings_async(texts)
            if hasattr(vectors, "tolist"):
                vectors = vectors.tolist()

            pbar.set_postfix_str("Aligning...")
            # Batch concurrent alignment, update progress
            batch = 200
            for i in range(0, n, batch):
                chunk = all_entities[i:i + batch]
                vecs = vectors[i:i + batch]
                await self.entity_resolver.resolve_batch_async(chunk, pre_vectors=vecs)
                pbar.update(len(chunk))
                pbar.set_postfix_str(f"{min(i+batch, n)}/{n}")

    async def _build_single(self, item: dict):
        """Single chunk ingestion: entity alignment -> graph write -> Embed -> vector store write."""
        import time as _time
        t0 = _time.time()
        chunk_id = item["chunk_id"]
        text = item["text"]
        entities = item["entities"]
        relations = item["relations"]

        # Merge chunk + entity + fact embeddings into a single API call
        entity_texts = [f"Entity: {e['id']}. Type: {e['type']}. Description: {e['desc']}" for e in entities]
        fact_texts = []
        fact_raw = []
        if self.fact_enabled:
            for rel in relations:
                _r, _s, _t = rel.get("rel", ""), rel.get("src", ""), rel.get("tgt", "")
                if _r and _s and _t:
                    _ft = f"{_s} {_r} {_t}"
                    fact_texts.append(_ft)
                    fact_raw.append({"src_name": _s, "tgt_name": _t, "relation": _r, "text": _ft})
        all_texts = [text] + entity_texts + fact_texts
        all_vectors = await self.llm.embed_model.get_embeddings_async(all_texts)
        if hasattr(all_vectors, "tolist"):
            all_vectors = all_vectors.tolist()
        chunk_vector = all_vectors[0]
        entity_vectors = all_vectors[1:1 + len(entities)]
        fact_vectors = all_vectors[1 + len(entities):] if self.fact_enabled else []
        t1 = _time.time()

        # 1. Entity alignment (use pre-computed vectors, skip internal embedding)
        aligned = await self.entity_resolver.resolve_batch_async(entities, pre_vectors=entity_vectors)
        t2 = _time.time()

        # 2. Write to graph
        self._write_to_memory_graph(chunk_id, aligned, relations)
        self.memory_graph.touch_chunk(chunk_id)
        t3 = _time.time()

        # 3. Build metadata
        entities_meta = [{"uid": v["uid"], "name": v["name"], "type": v["type"], "desc": v["desc"]}
                         for v in aligned.values()]
        relations_meta = []
        for rel in relations:
            src_uid = aligned.get(rel["src"], {}).get("uid")
            tgt_uid = aligned.get(rel["tgt"], {}).get("uid")
            if src_uid is not None and tgt_uid is not None and rel.get("rel"):
                relations_meta.append({"src_uid": src_uid, "tgt_uid": tgt_uid, "relation": rel["rel"]})

        # 4. Write vector store + fact concurrently
        import asyncio as _asyncio
        insert_tasks = [_asyncio.to_thread(self.vector_store.insert_chunk,
            chunk_id, chunk_vector,
            [str(v["uid"]) for v in aligned.values()],
            text, {"entities": entities_meta, "relations": relations_meta})]

        if self.fact_enabled and fact_raw and fact_vectors:
            fact_datas, fact_vecs = [], []
            for fi, fr in enumerate(fact_raw):
                src_uid = aligned.get(fr["src_name"], {}).get("uid")
                tgt_uid = aligned.get(fr["tgt_name"], {}).get("uid")
                if src_uid is not None and tgt_uid is not None:
                    fact_datas.append({
                        "fact_text": fr["text"], "subj_name": fr["src_name"],
                        "obj_name": fr["tgt_name"], "relation": fr["relation"],
                        "chunk_id": chunk_id, "subj_uid": src_uid, "obj_uid": tgt_uid,
                    })
                    fact_vecs.append(fact_vectors[fi])
            if fact_datas:
                insert_tasks.append(_asyncio.to_thread(
                    self.fact_store.insert_facts_batch, fact_datas, fact_vecs))

        await _asyncio.gather(*insert_tasks)
        t4 = _time.time()

        self._build_counter = getattr(self, "_build_counter", 0) + 1
        if self._build_counter <= 20 or self._build_counter % 500 == 0:
            log = get_logger()
            log._print(f"  [cost#{self._build_counter}] {chunk_id[:12]} embed={t1-t0:.2f}s resolve={t2-t1:.2f}s graph={t3-t2:.2f}s milvus={t4-t3:.2f}s")

    def _reload_milvus_collections(self):
        """Reload Milvus collections after restart (sync call, for retry use)."""
        from pymilvus import Collection
        for name in [self.vector_store.db_name, self.entity_resolver.collection_name]:
            try:
                col = Collection(name)
                col.load()
            except Exception:
                pass

    # ── Utilities ──────────────────────────────────────────────

    def _clean_and_validate(self, raw_str: str) -> Tuple[List[dict], List[dict]]:
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
            data = json.loads(raw_str[start: end + 1])
            entities = data.get("entities", [])
            relations = data.get("relations", [])
            valid_ids = {e["id"] for e in entities}
            valid_relations = [r for r in relations if r.get("src") in valid_ids and r.get("tgt") in valid_ids]
            return entities, valid_relations
        except json.JSONDecodeError:
            return [], []

    def _write_to_memory_graph(self, chunk_id: str, aligned: Dict, relations: List[Dict]):
        for original_id, ent_data in aligned.items():
            self.memory_graph.add_node(
                ent_data["uid"],
                name=ent_data["name"],
                type=ent_data["type"],
                source_chunk=chunk_id,
                desc=ent_data.get("desc", ""),
            )
        for rel in relations:
            src_uid = aligned[rel["src"]]["uid"]
            tgt_uid = aligned[rel["tgt"]]["uid"]
            self.memory_graph.add_edge(src_uid, tgt_uid, relation_type=rel["rel"], source_chunk=chunk_id)
