import asyncio
import time
import json
import math
import os
import re
import sys
import uuid
import argparse
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
from utils.prompts import prompt_extract_triplest_str, prompt_extract_entities_str, prompt_answer_with_chunks_str
from utils.llm_env import LLMEnv  
from database.milvus import MilvusDB, myMilvus
from database.nebulagraph import NebulaClient, NebulaDB
from src.pipeline import DocumentIngestionPipeline
from src.memory_graph import MemoryGraphManager
from src.entity_resolver import AsyncEntityResolver
from src.retriever import HybridRetriever
from data.rgb import get_rgb_info

# os.environ["MILVUS_FORCE_FLUSH"] = "1"
ingestion_time = []
retrieval_time = []

async def run_ingestion(llm: LLMEnv, dataset: str, questions: List[str], answers: List[str], texts: List[str]):

    print("🚀 --- [阶段 1: 文档入库处理 (Ingestion)] ---")
    mem_graph = MemoryGraphManager(
        space_name=dataset, # NebulaGraph 持久化存储
        promotion_threshold=5 # 晋升阈值
        ) 
    resolver = AsyncEntityResolver(
        collection_name="entity_index",
        embedding_func=llm.embed_model.get_embedding_async,
        memory_graph=mem_graph,
    )

    pipeline = DocumentIngestionPipeline(
        collection_name=dataset,
        llm_client=llm, 
        memory_graph=mem_graph, 
        entity_resolver=resolver,
    )
    mem_graph.chunk_vector_store = pipeline.vector_store

    start_time = time.time()
    tasks = []
    for i, doc_text in enumerate(texts):
        tasks.append(pipeline.process_document(doc_text, source_file=f"document_{i}.txt"))
    await asyncio.gather(*tasks)
    await resolver.wait_pending()
    print("文档入库处理完成！")
    print(f"process docs has taken: {time.time() - start_time} seconds")
    ingestion_time.append(time.time() - start_time)
    return mem_graph, pipeline

async def run_retrieval(
    llm: LLMEnv,
    dataset: str,
    questions: List[str],
    answers: List[str],
    start: int,
    end: int,
    mem_graph: MemoryGraphManager,
    pipeline: DocumentIngestionPipeline,
):
    print("\n🚀 --- [阶段 2: 检索与子图晋升 (Retrieval & Promotion)] ---")
    
    querys = questions
    answers = answers
    print(f"Unique Queries: {len(querys)}, Unique Answers: {len(answers)}")
    assert len(querys) == len(answers), "QA pairs should be 1-to-1."
    start_time = time.time()
    retriever = HybridRetriever(
        vector_store = MilvusDB(db_name=dataset, overwrite=False), 
        memory_graph=mem_graph, 
        llm=llm, 
        chunk_registry=pipeline.chunk_registry
    )
    data = []
    for i,(query, answer) in tqdm(enumerate(zip(querys, answers)), total=len(querys)):
        print(f"\n🔍 Query {i+1}: {query}")
        retrieval_res = retriever.hybrid_retrieve(query, topk=2, top_entities=3, top_chunks=3)
        data.append({
            "query": query,
            "answer": answer,
            "retrieval": retrieval_res,
        })
        save_to_json(file_path=f"data/retrieval_results_{dataset}_{start}_{end}.json", data=data, indent=2, info=False)
    print(f"检索与晋升阶段完成！")
    print(f"retrieval & promotion has taken: {time.time() - start_time} seconds")
    mem_graph.show_status()
    retrieval_time.append(time.time() - start_time)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Process some entities and triplets for knowledge extraction."
    )

    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=-1)
    parser.add_argument("--dataset", type=str, default="rgb_en")
    parser.add_argument("--backend", type=str, default="openai")

    args = parser.parse_args()
    print(args)

    config = get_config()
    print(config)

    if args.backend == "openai":
        model_name = "gpt-4o-mini"
        api_key = config["model"]["OPENAI_API_KEY"]
        base_url = config["model"]["OPENAI_BASE_URL"]
    elif args.backend == "deepseek":
        model_name = "deepseek-chat"
        api_key = config["model"]["DEEPSEEK_API_KEY"]
        base_url = config["model"]["DEEPSEEK_BASE_URL"]
    elif args.backend == "ollama":
        # print("***********************************************")
        model_name = "llama3.1:8b"
        api_key = None
        base_url = config["model"]["LLAMA3_8B_URL"]
    else:
        raise ValueError(f"Unsupported backend: {args.backend}")

    llm = LLMEnv(
        backend=args.backend,
        model=model_name,
        api_key=api_key,
        base_url=base_url,
    )

    if "rgb" in args.dataset:
        # data_info = get_rgb_info(f"{args.dataset[4:]}", chunk_size=2048)
        data_info = get_rgb_info()

    elif "dragonball" == args.dataset:
        data_info = get_dragonball_info("en", "Summary Question")
        texts = data_info["texts"]
        # Summary Question: 415 questions
        # "questions": questions,
        # "answers": answers,
        # "languages": languages,
        # "domains": domains,
        # "question_types": question_types,
        # "texts": texts,
        # print(len(data_info["questions"]))
        # print(len(texts), type(texts[0]))

    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    questions, answers, texts = (
        data_info["questions"],
        data_info["answers"],
        data_info["texts"],
    )

    print("number of questions:", len(questions))
    print("number of answers:", len(answers))
    print("number of texts:", len(texts))
    # exit(0)
    if args.end == -1:
        args.end = len(questions)

    questions = questions[args.start : args.end]
    answers = answers[args.start : args.end]
    texts = texts[args.start : args.end]
    # save_to_json(f"data/qa_pairs_{args.dataset}_{args.backend}_{args.start}_{args.end}_1.json", {
    #     "questions": questions,
    #     "answers": answers,
    #     "texts": texts,
    # }, indent=2, info=False)

    async def main(
        batch_questions: List[str],
        batch_answers: List[str],
        batch_texts: List[str],
        batch_start: int,
        batch_end: int,
    ):
        mem_graph, pipeline = await run_ingestion(
            llm, args.dataset, batch_questions, batch_answers, batch_texts
        )
        mem_graph.show_status()
        await run_retrieval(
            llm,
            args.dataset,
            batch_questions,
            batch_answers,
            batch_start,
            batch_end,
            mem_graph,
            pipeline,
        )
        mem_graph.save_graph_graphml(f"subgraph/memory_graph_{args.dataset}.graphml")
        mem_graph.save_graph_gexf(f"subgraph/memory_graph_{args.dataset}.gexf")
        mem_graph.graph.clear()
        mem_graph.chunk_access_counter.clear()
        mem_graph._id_index.clear()

    async def run_batches():
        if len(texts) > 30 and args.end > 30:
            batch_size = 30
            for offset in range(0, len(texts), batch_size):
                batch_start = args.start + offset
                batch_end = min(args.start + offset + batch_size, args.end)
                await main(
                    questions[offset : offset + batch_size],
                    answers[offset : offset + batch_size],
                    texts[offset : offset + batch_size],
                    batch_start,
                    batch_end,
                )
        else:
            await main(questions, answers, texts, args.start, args.end)

    asyncio.run(run_batches())

    print("\n📊 --- [总结] ---")
    print(f" average ingestion time: {sum(ingestion_time) / len(ingestion_time) if ingestion_time else 0}")
    print(f" retrieval time: {sum(retrieval_time) / len(retrieval_time) if retrieval_time else 0}")
    # mem_graph.load_graph_gexf(f"subgraph/memory_graph_{args.dataset}.gexf")
    # mem_graph.show_status()
    # asyncio.run(run_retrieval(llm, args.dataset, questions, answers, texts))

# python -m database.db-tool --db rgb_en --clear vector
# python -m database.db-tool --db entity_index --clear vector
# python -m src.graphrag --start 0 --end 10 --dataset rgb_en --backend openai
