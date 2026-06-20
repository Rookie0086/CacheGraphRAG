import json
import os
from typing import Dict, List

from data.paths import COND_DATAPATH
from src.utils import file_exist


def _resolve_cond_path(file: str) -> str:
    if os.path.isabs(file):
        return file
    filename = file if file.endswith(".json") else f"{file}.json"
    return os.path.join(COND_DATAPATH, filename)


def get_cond_info(file: str = "cond", limit: int = -1) -> Dict[str, List]:
    data_file = _resolve_cond_path(file)
    if not file_exist(data_file):
        repo_fallback = os.path.join(os.path.dirname(__file__), "cond.json")
        if file_exist(repo_fallback):
            data_file = repo_fallback
        else:
            raise FileNotFoundError(f"{data_file} not exist!")

    texts: List[str] = []
    questions: List[str] = []
    answers: List[str] = []

    with open(data_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    data_list = data if isinstance(data, list) else [data]
    if limit is not None and limit > 0:
        data_list = data_list[:limit]

    for item in data_list:
        if not isinstance(item, dict):
            continue
        question = str(item.get("question", "")).strip()
        choices = item.get("choices", [])
        question += " You should choose from the following options: " + "; ".join(str(c) for c in choices if str(c))
        answer = item.get("answer", "")
        context = item.get("context", "")
        if not question or not context:
            continue
        if isinstance(answer, list):
            answer = "; ".join(str(a) for a in answer if str(a))
        else:
            answer = str(answer)
        texts.append(str(context))
        questions.append(question)
        answers.append(answer)

    data_info = {
        "texts": texts,
        "questions": questions,
        "answers": answers,
    }

    return data_info


if __name__ == "__main__":
    cond_info = get_cond_info("cond")
    print(len(cond_info["texts"]), len(cond_info["questions"]), len(cond_info["answers"]))
