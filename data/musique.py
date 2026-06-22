import json
import os
from typing import Dict, List, Optional, Iterable

from data.paths import MUSIQUE_DATAPATH
from src.utils.base import file_exist, save_to_json


def _iter_records(data_file: str) -> Iterable[dict]:
    with open(data_file, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            f.seek(0)
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue
            return
    if isinstance(data, list):
        for item in data:
            yield item
    else:
        yield data


def _concat_context(context) -> str:
    if not isinstance(context, list):
        return ""
    parts = []
    for item in context:
        title = ""
        text = ""
        if isinstance(item, dict):
            title = str(item.get("title", "")).strip()
            text = str(item.get("paragraph_text", "") or item.get("text", "")).strip()
        elif isinstance(item, list) and len(item) >= 2:
            title = str(item[0]).strip()
            sentences = item[1] if isinstance(item[1], list) else [item[1]]
            text = " ".join(str(s) for s in sentences if str(s)).strip()
        if title and text:
            parts.append(f"{title}: {text}")
        elif title:
            parts.append(title)
        elif text:
            parts.append(text)
    return "\n\n".join(parts)


def get_musique_info(limit: Optional[int] = None) -> Dict[str, List]:
    data_file = os.path.join(MUSIQUE_DATAPATH, "musique.json")
    assert file_exist(data_file), f"{data_file} not exist!"

    questions: List[str] = []
    answers: List = []
    texts: List[str] = []

    count = 0
    for data in _iter_records(data_file):
        if not isinstance(data, dict):
            continue
        question = data.get("question", "")
        answer = [data.get("answer", "")]
        aliases = data.get("answer_aliases", []) or []
        if aliases:
            combined = [answer + [str(a) for a in aliases if str(a)]]
            answers.append(combined)
        else:
            answers.append(answer)
        questions.append(str(question))
        paragraphs = data.get("paragraphs")
        context = data.get("context") if paragraphs is None else paragraphs
        texts.append(_concat_context(context))
        count += 1
        if limit is not None and limit > 0 and count >= limit:
            break

    data_info = {
        "texts": texts,
        "questions": questions,
        "answers": answers,
    }

    return data_info


if __name__ == "__main__":
    data_info = get_musique_info()
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
        "data/data_format/qa_pairs_musique.json",
        {
            "questions": questions[:10],
            "answers": answers[:10],
            "texts": texts[:10],
        },
        indent=2,
        info=False,
    )
