import json
import os
import random

from data.paths import TQA_DATAPATH
from utils import file_exist, read_json


def process_instance(tqa_instances):
    # data_log = {}
    # if file_exist(data_log_file):
    #     data_log = read_json(data_log_file)
    questions, answers, texts = [], [], []
    for item in tqa_instances:
        questions.append(item["question"])
        answers.append(item["answers"])
        # documents.append([i['text'] for i in item['documents']][:2])
        texts.append([i["text"] for i in item["documents"]][:])

    print(f"load {len(questions)} questions.")
    print(f"load {len(answers)} answers.")
    print(f"load {len(texts)} texts.")

    # all_docs = sum([len(x) for x in texts])
    # print([len(x) for x in texts])
    # print(len([len(x) for x in texts]))
    # print(all_docs)

    data_info = {
        "questions": questions,
        "answers": answers,
        "texts": texts,
    }

    return data_info


def get_tqa_info(num_samples: int = -1, file="train"):
    data_path = os.path.join(TQA_DATAPATH, f"{file}.json")
    assert file_exist(data_path)

    tqa_instances = read_json(data_path)

    # with open(data_path, "r", encoding="utf-8") as f:
    #     tqa_instances = [json.loads(i) for i in f]

    print(f"tqa {file} has {len(tqa_instances)} item.")

    random.seed(2000)
    if num_samples != -1:
        print(f"sample {num_samples} samples.")
        tqa_instances = random.sample(population=tqa_instances, k=num_samples)

    return process_instance(tqa_instances)


if __name__ == "__main__":

    tqa_info = get_tqa_info(num_samples=100, file="test")
    tqa_info = get_tqa_info(num_samples=100, file="train")
