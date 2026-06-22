import json
import os
import random
from typing import Dict, List

from data.paths import HOTPOTQA_DATAPATH
from src.utils import file_exist
from src.utils.base import save_to_json


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


def _merge_sentences(entry) -> str:
    """Extract the sentence list from an entry and merge into plain text (without title)."""
    if not isinstance(entry, list) or len(entry) != 2:
        return ""
    _, sentences = entry
    if isinstance(sentences, list):
        return " ".join(str(s) for s in sentences if str(s))
    return str(sentences)


def get_hotpotqa_corpus(file: str = "hotpot_dev_distractor_v1", num: int = 300) -> List[str]:
    """Extract all unique documents, merge into one large text, and return a single-element list for LangChain to split.

    Dedup logic: only the first occurrence of each title is kept.
    """
    data_file = os.path.join(HOTPOTQA_DATAPATH, f"{file}.json")
    assert file_exist(data_file), f"{data_file} not exist!"
    with open(data_file, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    data_list = data if isinstance(data, list) else [data]
    sample_size = min(num, len(data_list))
    sampled_items = data_list[:sample_size]

    seen_titles = set()
    all_texts = []
    for item in sampled_items:
        for entry in item.get("context", []):
            if not isinstance(entry, list) or len(entry) != 2:
                continue
            title = entry[0]
            if title in seen_titles:
                continue
            seen_titles.add(title)
            text = _merge_sentences(entry)
            if text:
                all_texts.append(text)

    # Merge into one large text for LangChain to split
    merged = "\n\n".join(all_texts)
    print(f"Corpus: {len(all_texts)} unique documents, {len(merged)} characters after merging")
    return [merged]


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
    sampled_items = data_list[:sample_size]
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
