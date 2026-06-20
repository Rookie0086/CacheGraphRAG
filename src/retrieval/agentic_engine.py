import json
import re
from typing import Dict, List, Optional
from src.utils.prompts import prompt_answer_with_chunks_str, prompt_sub_answer_lite_str, prompt_plan_first_step_str, prompt_plan_next_step_str


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
		print(f"\n========== [Agentic] _plan_first_step ==========")
		print(f"[Agentic] Original question: {query}")
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
		print(f"[Agentic] Step1 raw response: {response[:300]}")
		payload = self._extract_json(response) or {}
		need_multihop = self._as_bool(payload.get("need_multihop"), default=False)
		subquestion = str(payload.get("subquestion", "")).strip() or query
		print(f"[Agentic] Step1 result: need_multihop={need_multihop}, subquestion={subquestion[:100]}")
		return {"need_multihop": need_multihop, "subquestion": subquestion}

	def _answer_with_chunks(self, question: str, chunk_ids: List[str]) -> str:
		if not chunk_ids:
			return "", ""
		if not hasattr(self.retriever, "_get_chunk_text"):
			return "", ""
		context_parts = []
		for chunk_id in chunk_ids:
			text_info = self.retriever.vector_store.get_chunk_text(chunk_id)
			if not text_info or not text_info.get("text"):
				print(f"Warning: no chunk_text found for chunk_id={chunk_id}")
				continue
			ts_value = text_info.get("ts") or "UNKNOWN"
			text_value = text_info.get("text", "")
			context_parts.append(f"[{chunk_id}] | ts=[{ts_value}] {text_value}")
		context = "\n\n".join(context_parts)
		if not context:
			return "I don't know.", ""

		prompt = prompt_answer_with_chunks_str.format(
			query=question,
			context=context,
		)
		response = self.llm.complete(prompt=prompt) or ""
		payload = self._extract_json(response) or {}

		if isinstance(payload, dict):
			# Three-step CoT output
			candidates = payload.get("step1_candidates", "")
			analysis = payload.get("step2_analysis", "")
			if candidates and analysis:
				print(f"  Step1 candidates: {candidates[:200]}")
				print(f"  Step2 analysis: {analysis[:200]}")
			answer = payload.get("final_answer") or payload.get("answer")
			if answer is not None:
				return str(answer).strip(), prompt

		match = re.search(r"(?:final_)?answer\s*[:：]\s*(.+)", response, re.IGNORECASE)
		if match:
			return match.group(1).strip(), prompt
		return response.strip(), prompt

	def _plan_next_step(self, query: str, history: List[Dict[str, str]]) -> Dict[str, str]:
		print(f"\n========== [Agentic] _plan_next_step ==========")
		print(f"[Agentic] Original question: {query}")
		print(f"[Agentic] History ({len(history)} steps):")
		for i, h in enumerate(history):
			print(f"  Step {i+1}: Q={h['question'][:80]} | A={h['answer'][:80]}")
		# 1. Build global reasoning memory
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
			"4. CRITICAL RULES:\n"
			"   - DO NOT ask a question you have already asked in the history.\n"
			"   - If a previous result was 'UNKNOWN' or 'I don't know', DO NOT ask about it again. "
			"Set \"is_final\" to true and conclude with the information you have.\n\n"
			"Output JSON only: {\"is_final\": true/false, \"subquestion\": \"...\", \"final_answer\": \"...\"}.\n"
			"JSON:"
		)
		response = self.llm.complete(prompt=prompt) or ""
		print(f"[Agentic] NextStep raw response: {response[:300]}")
		payload = self._extract_json(response) or {}
		result = {
			"is_final": self._as_bool(payload.get("is_final"), default=False),
			"subquestion": str(payload.get("subquestion", "")).strip(),
			"final_answer": str(payload.get("final_answer", "")).strip(),
		}
		print(f"[Agentic] NextStep result: is_final={result['is_final']}, "
		      f"subquestion={result['subquestion'][:100]}, "
		      f"final_answer={result['final_answer'][:100]}")
		return result

	def run(self, query: str) -> Dict:
		# Step 1: Determine if multi-hop is needed, get the first subquestion
		plan = self._plan_first_step(query)
		need_multihop = plan["need_multihop"]
		history_for_llm: List[Dict[str, str]] = []
		all_chunks: List[str] = []
		asked_set: set = set()

		current_question = plan["subquestion"]
		stop_tokens = ["unknown", "i don't know", "not found"]
		final_answer = ""
		steps = []

		for step_idx in range(self.max_steps):
			norm_q = current_question.strip().rstrip("?").lower()
			asked_set.add(norm_q)

			# 2. Retrieve
			retrieval_res = self.retriever.hybrid_retrieve(
				current_question, topk=self.topk, top_entities=self.top_entities,
				top_chunks=self.top_chunks, top_rerank=self.top_rerank,
			)
			chunk_ids = retrieval_res.get("chunks", [])
			all_chunks.extend(chunk_ids)

			# 3. Answer
			answer, prompt_used = self._answer_with_chunks(current_question, chunk_ids)

			steps.append({"question": current_question, "answer": answer, "chunks": chunk_ids})
			history_for_llm.append({"question": current_question, "answer": answer})

			# 4. Single-hop question, return directly
			if not need_multihop:
				final_answer = answer
				break

			# 5. Check for terminating answers (unknown/not found)
			stop = any(t in answer.lower() for t in stop_tokens)
			if stop:
				final_answer = answer
				break

			# 6. Plan next step
			plan = self._plan_next_step(query, history_for_llm)
			if plan.get("is_final"):
				final_answer = plan.get("final_answer") or answer
				break

			next_q = plan.get("subquestion", "")
			if not next_q or next_q.strip().rstrip("?").lower() in asked_set:
				final_answer = answer
				break

			current_question = next_q

		if not final_answer:
			final_answer = answer if "answer" in dir() else ""

		return {
			"chunks": list(set(all_chunks)),
			"agentic_steps": steps,
			"final_answer": final_answer,
		}
