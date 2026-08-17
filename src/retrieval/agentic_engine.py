import json
import re
import threading
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional
from src.utils.prompts import prompt_answer_with_chunks_str, prompt_sub_answer_lite_str, prompt_plan_first_step_str, prompt_plan_next_step_str
from src.utils import get_config


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
		# 中档#9:循环阻断参数(语义查重阈值 / UNKNOWN 连续容忍次数)
		_ret_cfg = get_config().get("retrieval", {})
		self.semantic_dup_threshold = float(_ret_cfg.get("semantic_dup_threshold", 0.95))
		self.unknown_tolerance = max(1, int(_ret_cfg.get("unknown_tolerance", 2)))
		self.enable_string_block = bool(_ret_cfg.get("enable_string_block", True))
		self.enable_semantic_block = bool(_ret_cfg.get("enable_semantic_block", True))
		self.enable_unknown_block = bool(_ret_cfg.get("enable_unknown_block", True))
		_hyper_cfg = get_config().get("hyperparameters", {})
		self.beam_width = max(1, int(_ret_cfg.get("beam_width", _hyper_cfg.get("B", 1))))
		self.parallelism = max(1, int(_ret_cfg.get("agentic_parallelism", self.beam_width)))
		self.batch_planning = bool(_ret_cfg.get("agentic_batch_planning", True))
		self._embedding_cache = {}
		self._embedding_cache_lock = threading.Lock()
		self.planning_calls = 0
		self.batched_planning_calls = 0
		self._metrics_lock = threading.Lock()
		self.retrieval_calls = 0
		self.l1_hits = 0
		self.l2_hits = 0
		self.l2_query_errors = 0
		self.stage_latency = {}

	def _embed(self, text: str) -> Optional[np.ndarray]:
		"""获取文本 embedding(优先 retriever 的 embed 通道,失败返回 None)"""
		cache_key = str(text or "").strip().lower()
		with self._embedding_cache_lock:
			cached = self._embedding_cache.get(cache_key)
		if cached is not None:
			return cached
		embed_model = getattr(getattr(self.retriever, "llm", None), "embed_model", None)
		if embed_model is None:
			embed_model = getattr(self.llm, "embed_model", None)
		if embed_model is None:
			return None
		try:
			value = np.array(embed_model.get_embedding(text), dtype=float)
			with self._embedding_cache_lock:
				self._embedding_cache[cache_key] = value
			return value
		except Exception:
			return None

	def _is_semantic_dup(self, next_q: str, asked_embs: Dict[str, np.ndarray]) -> bool:
		"""语义级查重:next_q 与任一已问问题的余弦相似度 ≥ threshold 视为重复"""
		if not asked_embs or not next_q:
			return False
		emb = self._embed(next_q)
		if emb is None:
			return False
		for asked, a_emb in asked_embs.items():
			if a_emb is None:
				continue
			norm = np.linalg.norm(emb) * np.linalg.norm(a_emb)
			if norm == 0:
				continue
			sim = float(np.dot(emb, a_emb) / norm)
			if sim >= self.semantic_dup_threshold:
				return True
		return False

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

	def _record_retrieval(self, result: Dict) -> Dict[str, int]:
		"""Aggregate every internal retrieval, including discarded beam paths."""
		l1_hits = sum(len(item.get("chunk_scores", {})) for item in result.get("memory", []))
		l2_hits = sum(len(item.get("chunk_scores", {})) for item in result.get("persistent", []))
		stats = result.get("stats", {}) or {}
		l2_errors = int(stats.get("l2_query_errors", 0))
		with self._metrics_lock:
			self.retrieval_calls += 1
			self.l1_hits += l1_hits
			self.l2_hits += l2_hits
			self.l2_query_errors += l2_errors
			for key, value in (stats.get("latency", {}) or {}).items():
				self.stage_latency[key] = self.stage_latency.get(key, 0.0) + float(value)
		return {"l1_hits": l1_hits, "l2_hits": l2_hits, "l2_query_errors": l2_errors}

	def _retrieval_metrics(self) -> Dict:
		with self._metrics_lock:
			return {
				"retrieval_calls": self.retrieval_calls,
				"l1_hits": self.l1_hits,
				"l2_hits": self.l2_hits,
				"l2_query_errors": self.l2_query_errors,
				"stage_latency": dict(self.stage_latency),
			}

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
			"Set \"is_final\" to true and conclude with the information you have.\n"
			"   - When is_final is true, final_answer must be a concise direct answer to the "
			"Original Question (typically 1-8 words, e.g. 'Port of Spain'). No full sentences or "
			"explanation.\n\n"
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

	def _plan_candidates(self, query: str, history: List[Dict[str, str]], width: int) -> tuple:
		"""生成下一跳候选子问题。

		返回 (候选子问题列表, 最终答案) 二元组:
		- is_final=true 时:候选为空,并透传 LLM 针对原始问题的简洁 final_answer;
		- 否则:候选为去重后的子问题列表, final_answer 为空串。
		修复前 is_final 时直接 `return []` 会丢弃 LLM 生成的 final_answer,
		导致 _run_beam 兜底使用子问题的回答 → 答非所问(beam 宽度>1 时 EM=0)。
		"""
		self.planning_calls += 1
		if width <= 1:
			plan = self._plan_next_step(query, history)
			if plan.get("is_final"):
				return [], plan.get("final_answer", "")
			return [plan.get("subquestion", "")], ""
		history_str = "\n".join(
			f"Step {i + 1} - Asked: {item['question']}\nResult: {item['answer']}"
			for i, item in enumerate(history)
		)
		prompt = (
			"You are an expert reasoning agent exploring alternative graph-retrieval paths.\n"
			f"Original Question: {query}\nHistory:\n{history_str}\n\n"
			f"Return up to {width} distinct next subquestions. They must pursue different useful facts, "
			"must not repeat the history, and must be answerable by retrieval. "
			"If the history is already sufficient, set is_final=true and return no subquestions, "
			"and give a concise final_answer to the Original Question (typically 1-8 words, "
			"direct answer, no explanation).\n"
			'Output JSON only: {"is_final": true/false, "subquestions": ["..."], "final_answer": "..."}.\nJSON:'
		)
		payload = self._extract_json(self.llm.complete(prompt=prompt) or "") or {}
		if self._as_bool(payload.get("is_final"), default=False):
			return [], str(payload.get("final_answer", "")).strip()
		candidates = payload.get("subquestions", [])
		if isinstance(candidates, str):
			candidates = [candidates]
		return self._dedupe([str(item).strip() for item in candidates if str(item).strip()])[:width], ""

	def _plan_candidates_batch(self, query: str, paths: List[Dict], width: int) -> Dict[int, tuple]:
		"""Plan the next step for all active beam paths in one LLM request."""
		if len(paths) <= 1 or not self.batch_planning:
			return {
				item["path_id"]: self._plan_candidates(query, item["base"]["history"], width)
				for item in paths
			}
		payload_paths = []
		for item in paths:
			history = [
				{"question": step.get("question", ""), "answer": step.get("answer", "")}
				for step in item["base"]["history"]
			]
			payload_paths.append({"id": item["path_id"], "history": history})
		prompt = (
			"You are planning several independent graph-retrieval reasoning paths for one original question.\n"
			f"Original Question: {query}\n"
			f"For EACH path return up to {width} distinct useful next subquestions, or mark it final and give "
			"a concise answer to the Original Question. Do not repeat questions already present in that path.\n"
			"Paths:\n" + json.dumps(payload_paths, ensure_ascii=False) + "\n"
			'Output JSON only: {"paths":[{"id":0,"is_final":false,'
			'"subquestions":["..."],"final_answer":""}]}.\nJSON:'
		)
		self.planning_calls += 1
		self.batched_planning_calls += 1
		parsed = self._extract_json(self.llm.complete(prompt=prompt) or "") or {}
		rows = parsed.get("paths", [])
		if not isinstance(rows, list):
			rows = []
		result = {}
		for row in rows:
			try:
				path_id = int(row.get("id"))
			except (TypeError, ValueError):
				continue
			if self._as_bool(row.get("is_final"), default=False):
				result[path_id] = ([], str(row.get("final_answer", "")).strip())
				continue
			candidates = row.get("subquestions", [])
			if isinstance(candidates, str):
				candidates = [candidates]
			result[path_id] = (
				self._dedupe([str(value).strip() for value in candidates if str(value).strip()])[:width], "")
		# A malformed/missing path falls back independently instead of killing the beam.
		for item in paths:
			path_id = item["path_id"]
			if path_id not in result:
				result[path_id] = self._plan_candidates(query, item["base"]["history"], width)
		return result

	def _evaluate_beam_state(self, state: Dict) -> Dict:
		"""Retrieve and answer one beam state; safe to run in a worker thread."""
		question = state["question"]
		result = self.retriever.hybrid_retrieve(
			question, topk=self.topk, top_entities=self.top_entities,
			top_chunks=self.top_chunks, top_rerank=self.top_rerank,
		)
		retrieval_metrics = self._record_retrieval(result)
		chunk_ids = result.get("chunks", [])
		answer, _ = self._answer_with_chunks(question, chunk_ids)
		history = state["history"] + [{"question": question, "answer": answer,
		                                  "chunks": chunk_ids, **retrieval_metrics}]
		chunks = self._dedupe(state["chunks"] + chunk_ids)
		stop_tokens = ("unknown", "i don't know", "not found")
		unknown = any(token in answer.lower() for token in stop_tokens) or not answer.strip()
		unknown_streak = state["unknown_streak"] + 1 if unknown else 0
		score = state["score"] + len(set(chunk_ids)) + (0.0 if unknown else 1.0)
		return {"state": state, "answer": answer, "question": question,
		        "base": {**state, "history": history, "chunks": chunks, "score": score,
		                 "unknown_streak": unknown_streak}}

	def _run_beam(self, query: str, first_question: str, need_multihop: bool) -> Dict:
		"""Run beam search with parallel path evaluation and batched planning."""
		beam = [{"question": first_question, "history": [], "chunks": [], "score": 0.0,
		         "unknown_streak": 0, "blocked": 0, "final_answer": ""}]
		completed = []
		with ThreadPoolExecutor(max_workers=self.parallelism, thread_name_prefix="agent-beam") as executor:
			for _ in range(self.max_steps):
				expanded = []
				if self.parallelism > 1 and len(beam) > 1:
					evaluated = list(executor.map(self._evaluate_beam_state, beam))
				else:
					evaluated = [self._evaluate_beam_state(state) for state in beam]

				pending = []
				for item in evaluated:
					state, base, answer = item["state"], item["base"], item["answer"]
					unknown_limit = (self.enable_unknown_block and
					                 base["unknown_streak"] >= self.unknown_tolerance)
					if not need_multihop or unknown_limit:
						completed.append({**base, "final_answer": answer,
						                  "blocked": state["blocked"] + int(unknown_limit)})
					else:
						pending.append({**item, "path_id": len(pending)})

				plans = self._plan_candidates_batch(query, pending, self.beam_width) if pending else {}
				for item in pending:
					state, base, answer, question = item["state"], item["base"], item["answer"], item["question"]
					candidates, final_answer = plans.get(item["path_id"], ([], ""))
					if not candidates:
						completed.append({**base, "final_answer": final_answer or answer})
						continue
					norm_asked = {step["question"].strip().rstrip("?").lower() for step in state["history"]}
					asked_embs = ({step["question"]: self._embed(step["question"]) for step in state["history"]}
					              if self.enable_semantic_block else {})
					for next_q in candidates:
						norm_q = next_q.strip().rstrip("?").lower()
						if ((self.enable_string_block and
						     (norm_q in norm_asked or norm_q == question.strip().rstrip("?").lower()))
								or (self.enable_semantic_block and self._is_semantic_dup(next_q, asked_embs))):
							continue
						expanded.append({**base, "question": next_q})
				if not expanded:
					break
				beam = sorted(expanded, key=lambda item: (-item["score"], item["question"]))[:self.beam_width]

		pool = completed or beam
		best = max(pool, key=lambda item: (item["score"], len(item["chunks"])))
		final_answer = best.get("final_answer") or (best["history"][-1]["answer"] if best["history"] else "")
		return {"chunks": best["chunks"], "agentic_steps": best["history"],
		        "final_answer": final_answer, "block_count": best["blocked"],
		        "beam_width": self.beam_width, "beam_score": best["score"],
		        "parallel_workers": self.parallelism, "planning_calls": self.planning_calls,
		        "batched_planning_calls": self.batched_planning_calls,
		        **self._retrieval_metrics()}

	def run(self, query: str) -> Dict:
		# Step 1: Determine if multi-hop is needed, get the first subquestion
		plan = self._plan_first_step(query)
		need_multihop = plan["need_multihop"]
		if self.beam_width > 1:
			return self._run_beam(query, plan["subquestion"], need_multihop)
		history_for_llm: List[Dict[str, str]] = []
		all_chunks: List[str] = []
		asked_set: set = set()
		asked_embs: Dict[str, np.ndarray] = {}  # 语义查重缓存:norm_q -> embedding
		block_count = 0          # 被代码级循环阻断打断的次数(字符串 + 语义 + UNKNOWN)
		unknown_streak = 0       # 连续 UNKNOWN/无结果计数(触发换角度或提前结束)

		current_question = plan["subquestion"]
		stop_tokens = ["unknown", "i don't know", "not found"]
		final_answer = ""
		steps = []

		for step_idx in range(self.max_steps):
			norm_q = current_question.strip().rstrip("?").lower()
			asked_set.add(norm_q)
			asked_embs[norm_q] = self._embed(current_question)

			# 2. Retrieve
			retrieval_res = self.retriever.hybrid_retrieve(
				current_question, topk=self.topk, top_entities=self.top_entities,
				top_chunks=self.top_chunks, top_rerank=self.top_rerank,
			)
			retrieval_metrics = self._record_retrieval(retrieval_res)
			chunk_ids = retrieval_res.get("chunks", [])
			all_chunks.extend(chunk_ids)

			# 3. Answer
			answer, prompt_used = self._answer_with_chunks(current_question, chunk_ids)

			steps.append({"question": current_question, "answer": answer,
			              "chunks": chunk_ids, **retrieval_metrics})
			history_for_llm.append({"question": current_question, "answer": answer})

			# 4. Single-hop question, return directly
			if not need_multihop:
				final_answer = answer
				break

			# 5. UNKNOWN 计数阻断:连续 unknown_tolerance 次无结果 → 提前结束
			stop = any(t in answer.lower() for t in stop_tokens)
			if stop and self.enable_unknown_block:
				unknown_streak += 1
				if unknown_streak >= self.unknown_tolerance:
					block_count += 1
					print(f"[Agentic] UNKNOWN 连续 {unknown_streak} 次,提前结束 (tolerance={self.unknown_tolerance})")
					final_answer = answer
					break
				# 未达阈值:换角度——让 LLM 重新规划一个不同方向的子问题
				plan = self._plan_next_step(query, history_for_llm)
				if plan.get("is_final"):
					final_answer = plan.get("final_answer") or answer
					break
				next_q = plan.get("subquestion", "")
				if not next_q:
					final_answer = answer
					break
				current_question = next_q
				continue
			unknown_streak = 0

			# 6. Plan next step
			plan = self._plan_next_step(query, history_for_llm)
			if plan.get("is_final"):
				final_answer = plan.get("final_answer") or answer
				break

			next_q = plan.get("subquestion", "")
			# 7. 字符串查重(规范化后完全重复)
			if not next_q or (self.enable_string_block and next_q.strip().rstrip("?").lower() in asked_set):
				block_count += 1
				final_answer = answer
				break
			# 8. 语义查重(LLM 换措辞重复同一问题)
			if self.enable_semantic_block and self._is_semantic_dup(next_q, asked_embs):
				block_count += 1
				print(f"[Agentic] 语义查重阻断: next_q='{next_q[:60]}' (threshold={self.semantic_dup_threshold})")
				final_answer = answer
				break

			current_question = next_q

		if not final_answer:
			final_answer = answer if "answer" in dir() else ""

		print(f"[Agentic] loop-break: steps={len(steps)} blocked={block_count} unknown_streak={unknown_streak}")
		return {
			"chunks": list(set(all_chunks)),
			"agentic_steps": steps,
			"final_answer": final_answer,
			"block_count": block_count,
			**self._retrieval_metrics(),
		}
