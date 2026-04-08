import argparse
import os
from typing import List

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

from utils.base import read_json, save_to_json


def _load_list(path: str) -> List[dict]:
    data = read_json(path)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {path}, got {type(data).__name__}.")
    return data


def merge_files(input_files: List[str], output_file: str) -> int:
    merged = []
    for path in input_files:
        merged.extend(_load_list(path))
    save_to_json(output_file, merged, indent=2, info=True)
    return len(merged)


def main():
    parser = argparse.ArgumentParser(
        description="Merge retrieval results JSON files into one list."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Input JSON files to merge (each must be a list).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSON file path.",
    )
    args = parser.parse_args()

    input_files = [os.path.abspath(p) for p in args.inputs]
    output_file = os.path.abspath(args.output)

    for path in input_files:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing input file: {path}")

    total = merge_files(input_files, output_file)
    print(f"Merged {total} items into {output_file}")


if __name__ == "__main__":
    main()
