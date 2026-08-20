import json
import os
from typing import Dict, List

from data.paths import SPECIFICQA_DATAPATH
from src.utils.base import file_exist, save_to_json


def _join_candidate_texts(candidate_texts) -> str:
	if isinstance(candidate_texts, list):
		return "\n\n".join(str(t) for t in candidate_texts if str(t))
	if candidate_texts is None:
		return ""
	return str(candidate_texts)

def get_specificqa_info(limit: int = 10, update: bool = False) -> Dict[str, List]:
	"""
	Get structured information from the SpecificQA experiment dataset (same-name
	entity disambiguation, two-phase streaming), suitable for incremental update experiments.
	Each sample contains:
	- target_entity: target entity name (extracted from page_id)
	- specific_question: specific question generated for the target entity
	- phase_1_data: distractor text list used for first-stage graph construction
	- phase_2_data: target text list used for second-stage incremental update
	- ground_truth: correct answer list for the target entity
	"""
	data_file = os.path.join(SPECIFICQA_DATAPATH, "specificqa_experiment_dataset_600.json")
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


if __name__ == "__main__":
	data_info = get_specificqa_info(limit=600, update=False)
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
		"data/data_format/qa_pairs_specificqa.json",
		{
			"questions": questions[:10],
			"answers": answers[:10],
			"texts": texts[:10],
		},
		indent=2,
		info=False
	)
