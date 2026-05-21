import json
import os
from typing import Dict, List

from data.paths import WHOQA_DATAPATH
from utils.base import file_exist,save_to_json


def _join_candidate_texts(candidate_texts) -> str:
	if isinstance(candidate_texts, list):
		return "\n\n".join(str(t) for t in candidate_texts if str(t))
	if candidate_texts is None:
		return ""
	return str(candidate_texts)

def get_whoqa_ex_info(limit: int = 10, update: bool = False) -> Dict[str, List]:
	"""
	获取 WhoQA 数据的结构化信息，适用于增量更新实验。
	每个样本包含：
	- target_entity: 目标实体名称（从 page_id 提取）
	- specific_question: 针对目标实体生成的特指问题
	- phase_1_data: 用于第一阶段建图的干扰项文本列表
	- phase_2_data: 用于第二阶段增量更新的目标项文本列表
	- ground_truth: 目标实体的正确答案列表
	"""
	data_file = "data/whoqa_experiment_dataset_600.json"
	assert file_exist(data_file), f"{data_file} not exist!"

	texts = []
	questions = []
	answers = []
	with open(data_file, "r", encoding="utf-8") as f:
		data_list = json.load(f)
	for item in data_list[:limit]:
		phase_1_data = item.get("phase_1_data", [])
		phase_2_data = item.get("phase_2_data", [])
		if update:
			texts.append(_join_candidate_texts(phase_2_data))
		else:
			texts.append(_join_candidate_texts(phase_1_data))
		questions.append(item.get("specific_question", ""))
		answers.append(item.get("ground_truth", []))

	data_info = {
		"texts": texts,
		"questions": questions,
		"answers": answers,
	}

	return data_info

def get_whoqa_info(limit: int = 120) -> Dict[str, List]:
	data_file = os.path.join(WHOQA_DATAPATH, "WhoQA.json")
	assert file_exist(data_file), f"{data_file} not exist!"

	texts = []
	questions = []
	answers = []

	with open(data_file, "r", encoding="utf-8") as f:
		data_list = json.load(f)

	for item in data_list[:limit]:
		contexts = item.get("contexts", [])
		if contexts:
			candidate_texts = contexts[0].get("candidate_texts", [])
			if len(contexts) > 1:
				candidate_texts = candidate_texts + "\n\n" + contexts[1].get("candidate_texts", [])
			texts.append(_join_candidate_texts(candidate_texts))
		else:
			texts.append("")

		question_list = item.get("questions", [])
		question = question_list[0] if question_list else ""
		if len(contexts) > 1:
			page_id = contexts[1].get("page_id")
			if page_id:
				page_id_str = str(page_id)
				abbr = page_id_str.split("(", 1)[0].strip()
				if abbr:
					question = question.replace(abbr, page_id_str)
		questions.append(question)

		answer_by_context = item.get("answer_by_context", {})
		answers.append(answer_by_context.get("1", []))

	data_info = {
		"texts": texts,
		"questions": questions,
		"answers": answers,
	}

	return data_info


if __name__ == "__main__":
	data_info = get_whoqa_ex_info(limit=600, update=False)
	print(
		f"questions {len(data_info['questions'])} ",
		f"answers {len(data_info['answers'])} ",
		f"texts {len(data_info['texts'])} ",
	)
	questions, answers, texts = (
        data_info["questions"],
        data_info["answers"],
        data_info["texts"],
    )
	save_to_json(
		"data/data_format/qa_pairs_whoqa.json",
		{
			"questions": questions[:10],
			"answers": answers[:10],
			"texts": texts[:10],
		},
		indent=2,
		info=False
	)