import os
import sys
import argparse
import json
import re
import time
import numpy as np
from tqdm import tqdm
from typing import Dict, List
from utils.bert_sim import bert


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
	sys.path.insert(0, PROJECT_ROOT)

from database.milvus import MilvusDB
from utils.prompts import prompt_answer_with_chunks_str
from utils.base import read_json, save_to_json, checkanswer, checkanswer_rougel, get_accuracy, get_accuracy_rougel
from utils.llm_env import LLMEnv


def build_chunk_text_map(
	chunk_ids: List[str],
	milvus_db: MilvusDB,
) -> Dict[str, Dict[str, str]]:
	chunk_map: Dict[str, Dict[str, str]] = {}
	for chunk_id in chunk_ids:
		text_info = milvus_db.get_chunk_text(chunk_id)
		if not text_info or not text_info.get("text"):
			print(f"Warning: no chunk_text found for chunk_id={chunk_id}")
			continue
		chunk_map[chunk_id] = {
			"text": text_info.get("text", ""),
			"ts": text_info.get("ts", ""),
		}
	return chunk_map


def run_generate(dataset: str,start: int, end: int, backend: str):
	data_path = os.path.join(PROJECT_ROOT, "data", "retrieval_results_" + dataset + "_"+ str(start) +"_"+ str(end) +".json")
	if not os.path.exists(data_path):
		raise FileNotFoundError(f"Missing retrieval results: {data_path}")

	retrieval_data = read_json(data_path)
	if not retrieval_data:
		print("No retrieval results found.")
		return

	chunk_ids = sorted(
		{cid for item in retrieval_data for cid in item.get("retrieval", {}).get("chunks", [])}
	)

	# model_path = "/home/shuyurui/model/Llama-3.1-8B"  # Update this path to your local model"
	# llm = LLMEnv(
    #         backend="ollama",
    #         model="llama3:latest",
    #         base_url="http://localhost:11434",
    #         max_tokens=1024,
    #         timeout=1000,
    #     )
	llm = LLMEnv(
		backend="openai", 
		model="gpt-4o-mini", 
		api_key="sk-of-OUbbhtqucYYpKtappAhGLNhviBsIiNEgiwzxwiwOpiEgxgtNXMzrPRAVauVBvalD",
		base_url="https://api.ofox.ai/v1",
		max_tokens=1024, 
		timeout=1000
		)

	milvus_db = MilvusDB(db_name=dataset, overwrite=False)
	milvus_db.load()

	chunk_text_map = build_chunk_text_map(chunk_ids, milvus_db)
	all_labels = []
	
	all_time = []
	all_bert_scores = []
	outputs = []
	for i, item in tqdm(enumerate(retrieval_data), total=len(retrieval_data)):
		query = item.get("query", "")
		gold = item.get("answer", "")
		# if not isinstance(gold[0], list):
		# 	gold = [gold]
		prompt = ""
		pred = ""
		pred_str = ""
		chunk_list = item.get("retrieval", {}).get("chunks", [])
		agentic_answer = item.get("retrieval", {}).get("final_answer", "")
		if agentic_answer:
			pred_str = str(agentic_answer).strip()
			subquestions = item.get("retrieval", {}).get("agentic_steps", [])
			subqa = [f"subquestions:{item.get('question', '')}" + "\n" + f"subanswer:{item.get('answer', '')}"+ "\n" for item in subquestions]

		else:
			context_parts = []
			for cid in chunk_list:
				text_info = chunk_text_map.get(cid)
				if not text_info:
					continue
				ts_value = text_info.get("ts") or "UNKNOWN"
				context_parts.append(f"[{cid} | ts={ts_value}] {text_info['text']}")
			context = "\n\n".join(context_parts)
			generate_time = -time.time()
			if not context:
				pred = "I don't know."
				pred_str = pred
			else:
				prompt = prompt_answer_with_chunks_str.format(
					query=query,
					context=context,
				)
				pred = llm.complete(prompt=prompt) or ""
				pred_str = _extract_final_answer(pred)
			generate_time += time.time()
			all_time.append(generate_time)
		
		label = checkanswer(pred_str, gold)
		all_labels.append(label)
		accuracy = get_accuracy(all_labels)
		bert_score = bert(pred_str, gold)
		all_bert_scores.append(bert_score)
		avg_bert_score = np.average(all_bert_scores)

		outputs.append(
			{
				"query": query,
				"answer": gold,
				"subqa": subqa if agentic_answer else [],
                "prompt": prompt,
				"pred": pred,
				"prediction_str": pred_str,
				"label": label,
				"accuracy_so_far": accuracy,
				"bert_score": bert_score,
				"bert_score_so_far": avg_bert_score,
				# "avg_generate_time": np.average(all_time),
				"chunk_ids": chunk_list,
			}
		)

	output_path = os.path.join(PROJECT_ROOT, "data", f"generated_answers_{dataset}_{start}_{end}_{backend}.json")
	save_to_json(output_path, outputs, indent=2, info=False)
	print(f"Saved generated answers to {output_path}")


def _extract_final_answer(pred: str) -> str:
	if not pred:
		return ""
	def _clean_answer(value: str) -> str:
		val = value.strip()
		val = re.sub(r"^[\s\*`\[\(\{\"]+", "", val)
		val = re.sub(r"[\s\*`\]\)\}\"]+$", "", val)
		return val.strip()

	def _get_final_answer(obj: dict) -> str:
		for key in ("Final Answer", "final_answer", "final", "answer"):
			if key in obj:
				value = obj.get(key, "")
				return value.strip() if isinstance(value, str) else str(value)
		return ""

	try:
		data = json.loads(pred)
		if isinstance(data, dict):
			final_answer = _get_final_answer(data)
			return final_answer or pred.strip()
		return pred.strip()
	except Exception:
		pass

	start = pred.find("{")
	end = pred.rfind("}")
	if start != -1 and end != -1 and end > start:
		try:
			data = json.loads(pred[start : end + 1])
			if isinstance(data, dict):
				final_answer = _get_final_answer(data)
				return final_answer or pred.strip()
		except Exception:
			pass

	for pattern in (
		r"Final Answer\s*[:：]\s*(.+)",
		r"\*\*Final Answer\*\*\s*[:：]\s*(.+)",
		r"\"Final Answer\"\s*:\s*([^\n\r]+)",
	):
		match = re.search(pattern, pred, flags=re.IGNORECASE)
		if match:
			return _clean_answer(match.group(1))

	return pred.strip()


if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		description="Process some entities and triplets for knowledge extraction."
	)
	parser.add_argument("--dataset", type=str, default="rgb_en")
	parser.add_argument("--start", type=int, default=0)
	parser.add_argument("--end", type=int, default=-1)
	parser.add_argument("--backend", type=str, default="openai")
	args = parser.parse_args()
	run_generate(args.dataset, args.start, args.end, args.backend)
# CUDA_VISIBLE_DEVICES="1" python -m tests.test_generate --dataset whoqa --start 0 --end 600 --backend openai
# python -m scripts.merge_retrieval_results \
#   --inputs data/retrieval_results_musique_0_30.json \
#            data/retrieval_results_musique_30_60.json \
#            data/retrieval_results_musique_60_90.json \
#            data/retrieval_results_musique_90_120.json \
# 		   data/retrieval_results_musique_120_150.json \
# 		   data/retrieval_results_musique_150_180.json \
# 		   data/retrieval_results_musique_180_210.json \
# 		   data/retrieval_results_musique_210_240.json \
# 		   data/retrieval_results_musique_240_270.json \
# 		   data/retrieval_results_musique_270_300.json \
#   --output data/retrieval_results_musique_0_300.json