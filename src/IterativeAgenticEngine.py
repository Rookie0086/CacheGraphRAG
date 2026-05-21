import json
import re
from typing import Dict, List, Optional
from utils.prompts import prompt_answer_with_chunks_str, prompt_sub_answer_lite_str, prompt_plan_first_step_str, prompt_plan_next_step_str
from database.milvus import MilvusDB


class IterativeAgenticEngine:
	def __init__(
		self,
		llm,
		dataset,
		retriever,
		max_steps: int = 2,
		topk: int = 2,
		top_entities: int = 3,
		top_chunks: int = 3,
		top_rerank: int = 6,
	):
		self.llm = llm
		self.dataset = dataset
		self.retriever = retriever
		self.max_steps = max_steps
		self.topk = topk
		self.top_entities = top_entities
		self.top_chunks = top_chunks
		self.top_rerank = top_rerank
		self.milvus_db = MilvusDB(db_name=dataset, overwrite=False)
		self.milvus_db.load()

	def _extract_json(self, text: str) -> Optional[dict]:
		if not text:
			return None
		cleaned = text.strip()
		if cleaned.startswith("```"):
			cleaned = cleaned.strip("`")
			if cleaned.lower().startswith("json"):
				cleaned = cleaned[4:].strip()
		start = cleaned.find("{")
		end = cleaned.rfind("}")
		if start == -1 or end == -1 or end <= start:
			return None
		try:
			return json.loads(cleaned[start : end + 1])
		except json.JSONDecodeError:
			return None

	def _as_bool(self, value, default: bool = False) -> bool:
		if isinstance(value, bool):
			return value
		if isinstance(value, str):
			val = value.strip().lower()
			if val in ("true", "yes", "y", "1"):
				return True
			if val in ("false", "no", "n", "0"):
				return False
		return default

	def _dedupe(self, items: List[str]) -> List[str]:
		seen = set()
		result = []
		for item in items:
			if item in seen:
				continue
			seen.add(item)
			result.append(item)
		return result

	def _plan_first_step(self, query: str) -> Dict[str, str]:
		prompt = (
			"You are a decomposition assistant. Decide whether the question needs multi-hop reasoning. "
			"If yes, provide the first subquestion needed to find an intermediate entity or fact. "
			"If no, set subquestion to the original question.\n"
			"Output JSON only: {\"need_multihop\": true/false, \"subquestion\": \"...\"}.\n"
			f"Question: {query}\n"
			"JSON:"
		)
		# prompt = prompt_plan_first_step_str.format(query=query)
		response = self.llm.complete(prompt=prompt) or ""
		payload = self._extract_json(response) or {}
		need_multihop = self._as_bool(payload.get("need_multihop"), default=False)
		subquestion = str(payload.get("subquestion", "")).strip() or query
		return {"need_multihop": need_multihop, "subquestion": subquestion}

	def _answer_with_chunks(self, question: str, chunk_ids: List[str]) -> str:
		if not chunk_ids:
			return ""
		if not hasattr(self.retriever, "_get_chunk_text"):
			return ""
		context_parts = []
		for chunk_id in chunk_ids:
			text_info = self.milvus_db.get_chunk_text(chunk_id)
			if not text_info or not text_info.get("text"):
				print(f"Warning: no chunk_text found for chunk_id={chunk_id}")
				continue
			ts_value = text_info.get("ts") or "UNKNOWN"
			text_value = text_info.get("text", "")
			context_parts.append(f"[{chunk_id}] | ts=[{ts_value}] {text_value}")
		context = "\n\n".join(context_parts)
		if not context:
			pred = "I don't know."
			pred_str = pred
		else:
			prompt = prompt_answer_with_chunks_str.format(
				query=question,
				context=context,
			)
			response = self.llm.complete(prompt=prompt) or ""
			payload = self._extract_json(response) or {}

		if isinstance(payload, dict):
			# Compatible with prompt keys 'final_answer' and 'answer'.
			answer = payload.get("final_answer") or payload.get("answer")
			if answer is not None:
				return str(answer).strip(),prompt

		match = re.search(r"(?:final_)?answer\s*[:：]\s*(.+)", response, flags=re.IGNORECASE)
		if match:
			return match.group(1).strip(),prompt
		return response.strip(),prompt

	def _plan_next_step(self, query: str, history: List[Dict[str, str]]) -> Dict[str, str]:
		# 1. 构建全局推理记忆
		history_str = "\n".join([f"Step {i+1} - Asked: {h['question']}\nResult: {h['answer']}" for i, h in enumerate(history)])
		
		prompt = (
			"You are an expert reasoning agent. Your goal is to answer the Original Question step by step.\n"
			"Here is the history of your investigation so far:\n"
			"-------------------\n"
			f"{history_str}\n"
			"-------------------\n"
			f"Original Question: {query}\n\n"
			"### INSTRUCTIONS:\n"
			"1. Look at the history. Do you have enough combined information to answer the Original Question?\n"
			"2. If YES: set \"is_final\" to true and provide the \"final_answer\".\n"
			"3. If NO: provide the next \"subquestion\".\n"
			"4. CRITICAL RULES: \n"
			"   - DO NOT ask a question you have already asked in the history.\n"
			"   - If a previous result was 'UNKNOWN' or 'I don't know', DO NOT ask about it again. Set \"is_final\" to true and conclude with the information you have.\n\n"
			"Output JSON only: {\"is_final\": true/false, \"subquestion\": \"...\", \"final_answer\": \"...\"}.\n"
			"JSON:"
		)
		# prompt = prompt_plan_next_step_str.format(history_str=history_str, query=query)
		response = self.llm.complete(prompt=prompt) or ""
		payload = self._extract_json(response) or {}
		
		# 容错处理
		is_final = self._as_bool(payload.get("is_final"), default=False)
		subquestion = str(payload.get("subquestion", "")).strip()
		final_answer = str(payload.get("final_answer", "")).strip()
		
		return {"is_final": is_final, "subquestion": subquestion, "final_answer": final_answer}

	def run(self, query: str) -> Dict[str, object]:
		plan = self._plan_first_step(query)
		need_multihop = plan.get("need_multihop", False)
		current_question = plan.get("subquestion") or query

		steps = []
		collected_chunks: List[str] = []
		history_for_llm = [] # 用于记录 Q&A 轨迹
		asked_questions = set() # 用于代码级强制查重
		
		final_answer = ""

		for step_index in range(self.max_steps):
			# 记录已问过的问题（转小写去标点，增加查重鲁棒性）
			normalized_q = re.sub(r'[^\w\s]', '', current_question.lower().strip())
			asked_questions.add(normalized_q)

			# 检索与回答
			retrieval_res = self.retriever.hybrid_retrieve(
				current_question,
				topk=self.topk,
				top_entities=self.top_entities,
				top_chunks=self.top_chunks,
				top_rerank=self.top_rerank,
			)
			chunks = retrieval_res.get("chunks", [])
			collected_chunks.extend(chunks)
			answer,prompt= self._answer_with_chunks(current_question, chunks)

			# 更新状态
			steps.append({
				"question": current_question,
				"prompt": prompt,
				"answer": answer,
				"retrieval": retrieval_res,
			})
			history_for_llm.append({
				"question": current_question, 
				"answer": answer
			})

			if not need_multihop:
				final_answer = answer
				break

			# 传入全局历史记录规划下一步
			next_plan = self._plan_next_step(query, history_for_llm)
			
			if next_plan.get("is_final") or step_index == self.max_steps - 1:
				final_answer = next_plan.get("final_answer") or answer
				steps.append({
					"question": next_plan.get("subquestion") or "",
					"prompt": "",
					"answer": final_answer,
					"retrieval": "",
				})
				break

			next_question = next_plan.get("subquestion") or ""
			
			# ==========================================
			# 代码级容错：强制阻断 LLM 的死循环行为
			# ==========================================
			# normalized_next_q = re.sub(r'[^\w\s]', '', next_question.lower().strip())
			# if not next_question or normalized_next_q in asked_questions:
			# 	print(f"⚠️ [Agent 阻断] LLM 试图重复提问: {next_question}。强制结束迭代。")
			# 	# 如果 LLM 陷入死循环且没有给出 final_answer，我们强制它用现有历史做一次总结
			# 	final_answer = next_plan.get("final_answer") 
			# 	if not final_answer:
			# 		final_answer = self._answer_with_chunks(query, collected_chunks)
			# 	break
				
			current_question = next_question

		return {
			"chunks": self._dedupe([str(c) for c in collected_chunks if str(c)]),
			"agentic_steps": steps,
			"final_answer": final_answer,
		}
