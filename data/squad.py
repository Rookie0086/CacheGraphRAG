import json
import os
from typing import Dict, List

from data.paths import SQUAD_DATAPATH
from src.utils import file_exist

def _unique_preserve(items: List[str]) -> List[str]:
    seen = set()
    result = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def get_squad_info(file: str = "train") -> Dict:
    data_file = os.path.join(SQUAD_DATAPATH, f"{file}-v1.1.json")
    assert file_exist(data_file), f"{data_file} not exist!"

    texts = []
    questions = []
    answers = []

    with open(data_file, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    for article in data.get("data", []):
        for paragraph in article.get("paragraphs", []):
            context = paragraph.get("context", "")
            texts.append(context)
            for qa in paragraph.get("qas", []):
                question = qa.get("question", "")
                answer_items = qa.get("answers", [])
                answer_texts = [a.get("text", "") for a in answer_items if a.get("text")]
                answer_texts = _unique_preserve(answer_texts)

                questions.append(question)
                answers.append(answer_texts[0] if answer_texts else "")

    data_info = {
        "texts": texts,
        "questions": questions,
        "answers": answers,
    }

    return data_info


if __name__ == "__main__":
    squad_info = get_squad_info("train")
    print(len(squad_info["texts"]), len(squad_info["questions"]), len(squad_info["answers"]))