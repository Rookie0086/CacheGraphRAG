import json
import os
from typing import Dict, List, Optional
from src.utils.base import save_to_json

from data.paths import TWOWIKIMULTIHOPQA_DATAPATH
from src.utils.base import file_exist


def get_2wikimultihopqa_info(q_type: Optional[str] = None) -> Dict[str, List]:
    data_file = os.path.join(TWOWIKIMULTIHOPQA_DATAPATH, "2wikimultihopqa.json")
    assert file_exist(data_file), f"{data_file} not exist!"

    def _concat_context(context) -> str:
        if not isinstance(context, list):
            return ""
        parts = []
        for item in context:
            if not isinstance(item, list) or len(item) < 2:
                continue
            title = str(item[0])
            sentences = item[1] if isinstance(item[1], list) else [item[1]]
            text = " ".join(str(s) for s in sentences if str(s))
            if title and text:
                parts.append(f"{title}: {text}")
            elif title:
                parts.append(title)
            elif text:
                parts.append(text)
        return "\n\n".join(parts)

    questions: List[str] = []
    answers: List[str] = []
    texts: List[str] = []
    with open(data_file, "r", encoding="utf-8") as f:
        data_list = json.load(f)
        if not isinstance(data_list, list):
            data_list = [data_list]
        for data in data_list:
            if not isinstance(data, dict):
                continue
            if q_type and data.get("type") != q_type:
                continue
            questions.append(data.get("question", ""))
            answers.append(data.get("answer", ""))
            texts.append(_concat_context(data.get("context")))

    data_info = {
        "texts": texts,
        "questions": questions,
        "answers": answers,
    }

    return data_info


if __name__ == "__main__":
    data_info = get_2wikimultihopqa_info()
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
    save_to_json(f"data/data_format/qa_pairs_2wikimultihopqa.json", {
        "questions": questions[:10],
        "answers": answers[:10],
        "texts": texts[:10],
    }, indent=2, info=False)
