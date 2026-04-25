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
from data.multihop import get_multihop_info
from data.dragonball import get_dragonball_info
from data.squad import get_squad_info
from data.hotpotqa import get_hotpotqa_info
from data.ectqa import get_ectqa_info
from data.whoqa import get_whoqa_info, get_whoqa_ex_info
from data.cond import get_cond_info
from data.wikimultihopqa import get_2wikimultihopqa_info
from data.musique import get_musique_info

# os.environ["MILVUS_FORCE_FLUSH"] = "1"
ingestion_time = []
retrieval_time = []

async def run_ingestion(llm: LLMEnv, mem_graph: MemoryGraphManager, dataset: str, questions: List[str], answers: List[str], texts: List[str]):

    print("🚀 --- [阶段 1: 文档入库处理 (Ingestion)] ---")
    mem_graph = mem_graph or MemoryGraphManager(
        space_name=dataset, # NebulaGraph 持久化存储
        promotion_threshold=5 # 晋升阈值
        ) 
    resolver = AsyncEntityResolver(
        collection_name="entity_index_" + dataset,
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
    pipeline: DocumentIngestionPipeline = None,
):
    print("\n🚀 --- [阶段 2: 检索与子图晋升 (Retrieval & Promotion)] ---")
    
    querys = questions
    answers = answers
    print(f"Unique Queries: {len(querys)}, Unique Answers: {len(answers)}")
    assert len(querys) == len(answers), "QA pairs should be 1-to-1."
    start_time = time.time()
    retriever = HybridRetriever(
        vector_store = MilvusDB(db_name=dataset, overwrite=False),
        entity_index_name = "entity_index_" + dataset, 
        memory_graph=mem_graph, 
        llm=llm, 
        chunk_registry=pipeline.chunk_registry # 直接从持久化的图加载状态这里就不需要 pipeline 了
    )
    data = []
    for i,(query, answer) in tqdm(enumerate(zip(querys, answers)), total=len(querys)):
        print(f"\n🔍 Query {i+1}: {query}")
        retrieval_res = retriever.hybrid_retrieve(query, topk=2, top_entities=3, top_chunks=3, top_rerank=6)
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
        data_info = get_rgb_info(file=args.dataset[4:])
        # texts:300, questions:300, answers:300

    elif "dragonball" == args.dataset:
        data_info = get_dragonball_info("en", "Factual Question")
        # texts:216
        # "Factual Question":535, short answer
        # "Multi-hop Reasoning Question":532, long answer
        # "Irrelevant Unsolvable Question":233, answer: "Unable to answer"
        # "Summary Question": 415 questions, long answer, summary of the context   

    elif "multihop" == args.dataset:
        data_info = get_multihop_info()
        # texts:609, question:2255, answer:2255, Insufficient information question:301.

    elif "squad" == args.dataset:
        data_info = get_squad_info(file="dev")
        # texts:2067, question:10570, answer:10570

    elif "hotpotqa" == args.dataset:
        data_info = get_hotpotqa_info(file="hotpot_dev_distractor_v1", num=300)
        # texts:7405, question:7405, answer:7405  

    elif "ectqa" == args.dataset:
        data_info = get_ectqa_info(corpus_file="new.jsonl.gz")
        # base_texts:384, new_text:96, question:248, answer:248
    
    elif "whoqa" == args.dataset:
        data_info = get_whoqa_ex_info(limit=120, update=True)
        # texts: 120, questions: 120, answers: 120

    elif "cond" == args.dataset:
        data_info = get_cond_info(file="cond")
        # texts: 334, questions: 334, answers: 334
        
    elif "wikimultihopqa" == args.dataset:
        data_info = get_2wikimultihopqa_info()
        # texts: 1000, questions: 1000, answers: 1000

    elif "musique" == args.dataset:
        data_info = get_musique_info(limit=300)
        # texts: 1000, questions: 1000, answers: 1000
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
    assert len(questions) == len(answers), "Questions, answers should have the same length."
    mismatch_texts = len(questions) == len(answers) and len(questions) != len(texts)
    # exit(0)
    if args.end == -1:
        args.end = len(questions)
    questions = questions[args.start : args.end]
    answers = answers[args.start : args.end]
    if not mismatch_texts:
        texts = texts[args.start : args.end]
    elif args.dataset == "squad":
        texts = texts[args.start : (args.end//5)]  # SQuAD 每个文本对应多个QA对，简单起见按比例截取文本
    elif args.dataset == "ectqa":
        texts = texts[args.start : 100]  

    # save_to_json(f"data/data_format/qa_pairs_{args.dataset}_{args.backend}_{args.start}_{args.end}.json", {
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
        mem_graph = MemoryGraphManager(
        space_name=args.dataset, # NebulaGraph 持久化存储
        promotion_threshold=5 # 晋升阈值
        )
        # mem_graph.load_graph_gexf(f"subgraph/memory_graph_{args.dataset}.gexf")
        # batch_texts = []
        mem_graph.show_status()
        mem_graph, pipeline = await run_ingestion(
            llm, mem_graph, args.dataset, batch_questions, batch_answers, batch_texts
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
            pipeline,  # 直接从持久化的图加载状态这里就不需要 pipeline 了
        )
        # mem_graph.save_graph_graphml(f"subgraph/memory_graph_{args.dataset}.graphml")
        mem_graph.save_graph_gexf(f"subgraph/memory_graph_{args.dataset}.gexf")
        mem_graph.graph.clear()
        mem_graph.chunk_access_counter.clear()
        mem_graph._id_index.clear()

    async def run_batches():
        if mismatch_texts or args.dataset in ["whoqa","ectqa"]:
            await main(questions, answers, texts, args.start, args.end)
        elif len(texts) > 30:
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

# python -m database.db-tool --db wikimultihopqa --clear vector
# python -m database.db-tool --db entity_index_wikimultihopqa --clear vector
# CUDA_VISIBLE_DEVICES="1" python -m src.graphrag --start 0 --end 300 --dataset wikimultihopqa --backend openai > log/wikimultihopqa_0_300_openai.log
# tmux new -s wikimultihopqa -d bash -lc 'CUDA_VISIBLE_DEVICES="" python -m src.graphrag --start 0 --end 300 --dataset wikimultihopqa --backend openai > log/wikimultihopqa_0_300_openai.log'