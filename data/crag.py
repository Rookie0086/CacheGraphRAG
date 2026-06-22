import bz2
import json
import os
from typing import Iterable, List, Optional

from src.utils.base import create_dir, file_exist, save_to_json


def _default_data_path() -> str:
    repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    return os.path.join(
        repo_dir,
        "..",
        "datasets",
        "crag",
        "crag_task_1_and_2_dev_v5.jsonl.bz2",
    )


def _iter_jsonl_bz2(data_file: str) -> Iterable[dict]:
    with bz2.open(data_file, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _strip_page_result(record: dict) -> dict:
    cleaned = dict(record)
    search_results = cleaned.get("search_results")
    if isinstance(search_results, list):
        cleaned_results = []
        for item in search_results:
            if isinstance(item, dict):
                filtered = {k: v for k, v in item.items() if k != "page_result"}
                cleaned_results.append(filtered)
            else:
                cleaned_results.append(item)
        cleaned["search_results"] = cleaned_results
    return cleaned


def get_crag_samples(
    limit: Optional[int] = 10,
    data_file: Optional[str] = None,
) -> List[dict]:
    data_file = data_file or _default_data_path()
    assert file_exist(data_file), f"{data_file} not exist!"

    samples: List[dict] = []
    for record in _iter_jsonl_bz2(data_file):
        if not isinstance(record, dict):
            continue
        samples.append(_strip_page_result(record))
        if limit is not None and limit > 0 and len(samples) >= limit:
            break
    return samples


if __name__ == "__main__":
    output_path = "data/data_format/qa_pairs_crag.json"
    create_dir(os.path.dirname(output_path))
    samples = get_crag_samples(limit=10)
    save_to_json(output_path, samples, indent=2, info=False)
    print(f"saved {len(samples)} items to {output_path}")
