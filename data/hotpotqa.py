import json
import os
from typing import Dict, List

from data.paths import HOTPOTQA_DATAPATH
from utils import file_exist


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
    return f"{title} {body}".strip()


def get_hotpotqa_info(file: str = "hotpot_dev_fullwiki_v1") -> Dict[str, List]:
    data_file = os.path.join(HOTPOTQA_DATAPATH, f"{file}.json")
    assert file_exist(data_file), f"{data_file} not exist!"

    texts = []
    questions = []
    answers = []

    with open(data_file, "r", encoding="utf-8") as handle:
        data_list = json.load(handle)

    for item in data_list:
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
    hotpot_info = get_hotpotqa_info("hotpot_dev_fullwiki_v1")
    print(
        len(hotpot_info["texts"]),
        len(hotpot_info["questions"]),
        len(hotpot_info["answers"]),
    )
