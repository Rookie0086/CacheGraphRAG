import json
import os
import random
from typing import Dict, List

from data.paths import HOTPOTQA_DATAPATH
from utils import file_exist
from utils.base import save_to_json


def _format_context(entry) -> str:
    if not isinstance(entry, list) or len(entry) != 2:
        return ""
    title, sentences = entry
    if not isinstance(title, str):
        title = str(title)
    if isinstance(sentences, list):
        body = " ".join(str(s) for s in sentences if str(s))
    else:
        body = str(sentences)
    return f"{title}:{body}".strip()


def get_hotpotqa_info(file: str = "hotpot_dev_distractor_v1",num: int = 300) -> Dict[str, List]:
    data_file = os.path.join(HOTPOTQA_DATAPATH, f"{file}.json")
    assert file_exist(data_file), f"{data_file} not exist!"
    texts = []
    questions = []
    answers = []

    with open(data_file, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    data_list = data if isinstance(data, list) else [data]
    sample_size = min(num, len(data_list))
    sampled_items = random.sample(data_list, sample_size) if sample_size > 0 else []
    for item in sampled_items:
        questions.append(item.get("question", ""))
        answers.append(item.get("answer", ""))
        context_parts = []
        for context_entry in item.get("context", []):
            text = _format_context(context_entry)
            if text:
                context_parts.append(text)
        texts.append("\n\n".join(context_parts))

    data_info = {
        "texts": texts,
        "questions": questions,
        "answers": answers,
    }

    return data_info


if __name__ == "__main__":
    data_info = get_hotpotqa_info("hotpot_dev_distractor_v1")
    print(
        len(data_info["texts"]),
        len(data_info["questions"]),
        len(data_info["answers"]),
    )
    questions, answers, texts = (
        data_info["questions"],
        data_info["answers"],
        data_info["texts"],
    )
    save_to_json(f"data/data_format/qa_pairs_hotpotqa.json", {
        "questions": questions[:10],
        "answers": answers[:10],
        "texts": texts[:10],
    }, indent=2, info=False)
