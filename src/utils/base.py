import datetime
import json
import os
import random
import re
from pathlib import Path
from typing import Dict, List, Optional

# import evaluate
import numpy as np
import yaml


def read_yaml(file_path: str) -> Dict:
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


# This function is adapted from the `llama_index` project to provide color-coded console output without external dependencies.
def print_text(text: str, color: Optional[str] = None, end: str = "") -> None:
    """
    Print the text with the specified color.

    Args:
        text (str): Text to be printed.
        color (str, optional): Color to be applied to the text. Supported colors are:
            llama_pink, llama_blue, llama_turquoise, llama_lavender,
            red, green, yellow, blue, magenta, cyan, pink.
        end (str, optional): String appended after the last character of the text.

    Returns:
        None
    """
    _LLAMA_INDEX_COLORS = {
        "llama_pink": "38;2;237;90;200",
        "llama_blue": "38;2;90;149;237",
        "llama_turquoise": "38;2;11;159;203",
        "llama_lavender": "38;2;155;135;227",
    }

    _ANSI_COLORS = {
        "red": "31",
        "green": "32",
        "yellow": "33",
        "blue": "34",
        "magenta": "35",
        "cyan": "36",
        "pink": "38;5;200",  # 256-color mode
    }

    all_colors = {**_LLAMA_INDEX_COLORS, **_ANSI_COLORS}

    if color and color in all_colors:
        ansi_code = all_colors[color]
        text = f"\033[1;3;{ansi_code}m{text}\033[0m"
    elif color:
        # fallback: italic + bold if unsupported color
        text = f"\033[1;3m{text}\033[0m"

    print(text, end=end)


def get_project_dir():
    if os.environ.get("PROJECT_BASE_DIR"):
        depcache_dir = os.environ.get("PROJECT_BASE_DIR")
    else:
        home_dir = os.path.expanduser("~")
        depcache_dir = os.path.join(home_dir, "KGUPDATER")

    return depcache_dir


def get_config():
    # 配置加载优先级:
    #   1) CACHEGRAPH_CONFIG 环境变量(显式指定路径)
    #   2) 本仓库内 config/config.yaml(base.py 位于 <仓库根>/src/utils/base.py)
    #   3) PROJECT_BASE_DIR / ~/KGUPDATER/config/config.yaml(历史部署路径)
    if os.environ.get("CACHEGRAPH_CONFIG"):
        config_path = Path(os.environ["CACHEGRAPH_CONFIG"])
    else:
        repo_root = Path(__file__).resolve().parent.parent.parent
        repo_config = repo_root / "config" / "config.yaml"
        if repo_config.exists():
            config_path = repo_config
        else:
            project_dir = get_project_dir()
            config_path = Path(project_dir) / "config" / "config.yaml"
    config = read_yaml(config_path)
    # Runtime-only secret/model overrides. This keeps local API credentials out
    # of tracked YAML and reproducibility snapshots.
    model_cfg = config.setdefault("model", {})
    env_overrides = {
        "CACHEGRAPH_MODEL_BACKEND": "backend",
        "CACHEGRAPH_MODEL_NAME": "model_name",
        "CACHEGRAPH_MODEL_BASE_URL": "base_url",
        "CACHEGRAPH_MODEL_API_KEY": "api_key",
    }
    for env_name, config_name in env_overrides.items():
        if os.environ.get(env_name):
            model_cfg[config_name] = os.environ[env_name]
    return config


def set_global_seed(seed: int = 42) -> None:
    """设置全局随机种子,保证数据切分/采样可复现。

    注意:LLM API 输出本身非确定性,此种子不控制 API 采样;
    仅覆盖 random / numpy / torch 的本地随机源。
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def get_date_now():
    current_datetime = datetime.datetime.now()
    datetime_string = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
    return datetime_string


def is_file_processed(log_file, doc_id):
    if not os.path.exists(log_file):
        return False

    with open(log_file, "r") as log_file:
        processed_ids_line = log_file.readline().strip()

    if processed_ids_line:
        processed_ids = set(processed_ids_line.split(","))
    else:
        processed_ids = set()

    return doc_id in processed_ids


def append_log(log_file, stuff):
    with open(log_file, "a") as out:
        if os.path.getsize(log_file) > 0:
            out.write(f",{stuff}")
        else:
            out.write(stuff)


def file_exist(path):
    return os.path.exists(path)


def isfile(path):
    return os.path.isfile(path)


def create_dir(path=None):
    if path and not file_exist(path):
        os.makedirs(path, exist_ok=True)


def parse_num(filename, mode, type=float, num=1, start=False):
    assert file_exist(filename), f"{filename} not exist!"
    assert isfile(filename), f"{filename} not a file!"

    ret = []
    with open(filename) as f:
        for line in f.readlines():
            if line.find(mode) >= 0:
                if start:
                    numbers = re.findall(r"\d+\.?\d*", line[line.find(start) :])
                else:
                    numbers = re.findall(r"\d+\.?\d*", line[line.find(mode) :])
                numbers = [type(x) for x in numbers][:num]
                ret.append(numbers if len(numbers) > 1 else numbers[0])
    return ret


def parse_str(filename, start, end=None):
    assert file_exist(filename), f"{filename} not exist!"
    assert isfile(filename), f"{filename} not a file!"

    ret = []
    with open(filename) as f:
        for line in f.readlines():
            if line.find(start) >= 0:
                start_idx = len(start) + line.index(start)
                end_idx = -1 if not end else line.index(end)
                ret.append(line[start_idx:end_idx])
    return ret


def read_json(file_path: str):
    assert file_exist(file_path), f"{file_path} not exist!"
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data


def save_to_json(file_path: str, data, indent=2, info=True):
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=indent)
    if info:
        print(f"save {len(data)} items to {file_path}")


def read_jsonl(file_path: str) -> List[Dict]:
    assert file_exist(file_path), f"{file_path} not exist!"
    with open(file_path, "r") as file:
        instances = [json.loads(line.strip()) for line in file if line.strip()]
    return instances


def save_to_jsonl(file_path: str, data):
    with open(file_path, "w", encoding="utf-8") as file:
        for item in data:
            # json.dump(data, file, ensure_ascii=False, indent=2)
            file.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"save {len(data)} items to {file_path}")


def escape_str(value: str) -> str:
    if not value or len(value) == 0:
        return value

    patterns = {
        '"': "",
        "{": "",
        "}": "",
    }
    for pattern in patterns:
        if pattern in value:
            value = value.replace(pattern, patterns[pattern])
    if value[0] == " " or value[-1] == " ":
        value = value.strip()
    value = " ".join(value.split())
    return value


class RunInfo:

    def __init__(self):
        self.run_info = {}

    def insert(self, **kv_args):
        for k, v in kv_args.items():
            if k not in self.run_info:
                self.run_info[k] = []
            self.run_info[k].append(v)

    def update(self, run_info):
        self.run_info.update(run_info)

    def get(self):
        return self.run_info

    def get_item(self, key):
        return self.run_info.get(key, None)

    def clear(self):
        self.run_info.clear()


def extract_json_str(text: str) -> str:
    """Extract JSON string from text."""
    # NOTE: this regex parsing is taken from langchain.output_parsers.pydantic
    match = re.search(r"\{.*\}", text.strip(), re.MULTILINE | re.IGNORECASE | re.DOTALL)
    if not match:
        raise ValueError(f"Could not extract json string from output: {text}")
    return match.group()


def checkanswer(prediction, ground_truth, verbose=False):
    """
    Check whether the predicted answer matches the ground truth.

    :param str prediction:
        Predicted answer, will be lowercased for comparison.

    :param ground_truth:
        Default is a list. If a string is passed, it is manually converted to a list.
        Elements in the list represent candidate answers.
        If it is a nested list, the question has multiple answers, all of which must be correct.

    :return:
        List of binary labels, 1 for match, 0 for no match.
    :rtype: List[int]

    Example:

    >>> prediction = \"The cat sits on the mat\"
    >>> ground_truth = [[\"cat\", \"CAT\"]]
    >>> checkanswer(\"cat\", ground_truth)
    [1]

    >>> checkanswer(\"cat and mat\", [[\"cat\"], [\"MAT\", \"mat\"]])
    [1, 1]
    """
    def _normalize_answer(text: str) -> str:
        normalized = text.lower().strip()
        normalized = re.sub(r"[\"'`]", "", normalized)
        normalized = normalized.replace("-", " ")
        normalized = re.sub(r"[\.,，。;；:：]", " ", normalized)
        normalized = normalized.rstrip("!?！？")
        normalized = " ".join(normalized.split())
        return normalized

    prediction = _normalize_answer(prediction)
    if not isinstance(ground_truth, list):
        ground_truth = [ground_truth]
    labels = []
    for instance in ground_truth:
        flag = True
        if isinstance(instance, list):
            flag = False
            instance = [_normalize_answer(i) for i in instance]
            for i in instance:
                if i in prediction:
                    flag = True
                    break
        else:
            instance = _normalize_answer(instance)
            if instance not in prediction:
                flag = False
        labels.append(int(flag))

    if verbose:
        print_text(
            f"\nprediction: {prediction}, \nground_truth: {ground_truth}, \nlabels: {labels}\n",
            color="yellow",
        )

    return labels


def get_accuracy(labels, info=None):
    tt = 0
    for label in labels:
        if 0 not in label and 1 in label:
            tt += 1
    acc = tt / len(labels)

    if info:
        print_text(f"{info} accuracy {acc}\n", color="green")

    return acc


def checkanswer_rougel(prediction, ground_truth):

    from ignite.metrics import RougeL

    # print("prediction", f"#{prediction}#")
    # print("prediction", f"#{ground_truth}#")
    # print()

    m = RougeL(multiref="best")

    candidate = prediction.split()
    references = [ground_truth.split()]

    m.update(([candidate], [references]))

    # print([candidate])
    # print([references])

    score = m.compute()
    # print(f"{score=}")

    return score


def get_accuracy_rougel(labels):

    precision = np.array([x["Rouge-L-P"] for x in labels])
    recall = np.array([x["Rouge-L-R"] for x in labels])
    f1 = np.array([x["Rouge-L-F"] for x in labels])

    # precision = np.array([x["rouge1"] for x in labels])
    # recall = np.array([x["rouge2"] for x in labels])
    # f1 = np.array([x["rougeL"] for x in labels])

    return {
        "precision": np.average(precision),
        "recall": np.average(recall),
        "f1": np.average(f1),
    }


def generate_sample_idx(range_length, num):
    idx = list(range(range_length))
    if num >= range_length:
        return idx
    random.seed(2000)
    sampled_idx = random.sample(idx, num)
    return sampled_idx


if __name__ == "__main__":

    predict = "I"
    answer = "I. Hill committed multiple thefts in January 2022. On 15th January, she stole a gold necklace valued at approximately £2,000 from a high-end jewelry shop. On 20th January, she took several electronic gadgets, including a brand new Apple iPhone and a Samsung tablet, from a local electronics store, with a total value of approximately £1,500. On 24th January, she shoplifted various high-value cosmetics and skincare products summing up to £800 from a prominent beauty shop. On 29th January, she executed a theft at a supermarket, taking groceries and alcohol worth £450."

    predict = "hello goodbye"
    answer = "goodbye"

    predict = "the cat is not there"
    answer = "the cat is on the mat"

    # >>> rouge = evaluate.load('rouge')
    # >>> predictions = ["hello goodbye", "ankh morpork"]
    # >>> references = ["goodbye", "general kenobi"]
    # >>> results = rouge.compute(predictions=predictions,
    # ...                         references=references,
    # ...                         use_aggregator=False)
    # >>> print(list(results.keys()))
    # ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']
    # >>> print(results["rouge1"])
    # [0.5, 0.0]

    score = checkanswer_rougel(predict, answer)
    print(score)
