import json
import os
from typing import List

from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
)

from data.paths import RGB_DATAPATH
from utils import file_exist

tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path="/home/shuyurui/model/Llama-3.1-8B",
    use_fast=False,
)


def compact_string(texts: List[str], chunk_size=2048):
    compact_texts = []
    cur_string = ""
    for text in texts:
        cur_string += text + "\n"

        # if len(cur_string) > length:
        #     compact_texts.append(cur_string)
        #     cur_string = ''

        input_ids = tokenizer.encode(cur_string, add_special_tokens=False)

        if len(input_ids) > chunk_size:
            compact_texts.append(cur_string)
            cur_string = ""

        # print(len(input_ids), chunk_size)

    if cur_string:
        compact_texts.append(cur_string)

    return compact_texts


def concat_strings_in_list(input_list):
    if not isinstance(input_list, list):
        raise ValueError("The input must be a list.")

    if isinstance(input_list[0], list):
        input_list = [" ".join(sublist) for sublist in input_list]

    assert all(isinstance(item, str) for item in input_list)
    return input_list


def get_rgb_info(file="en", chunk_size=512):
    data_file = os.path.join(RGB_DATAPATH, f"{file}.json")
    assert file_exist(data_file), f"{data_file} not exist!"

    texts = []
    questions = []
    answers = []
    with open(data_file, "r") as f:
        for line in f:
            instance = json.loads(line)
            if file == "en_fact":
                pos_text = " ".join(concat_strings_in_list(instance["positive"]))
                neg_text = " ".join(concat_strings_in_list(instance["negative"]))
                texts.append(pos_text + "\n" + neg_text)
            elif file == "en_int":
                pos_texts = concat_strings_in_list(instance["positive"])
                neg_texts = concat_strings_in_list(instance["negative"])
                # texts.append(compact_string(pos_texts, chunk_size=chunk_size))
                texts.append(
                    compact_string(pos_texts + neg_texts, chunk_size=chunk_size)
                )
                # print(len(texts[-1]))

            else:
                texts += concat_strings_in_list(instance["positive"])
                # texts += concat_strings_in_list(instance["negative"])
            questions.append(instance["query"])
            answers.append(instance["answer"])

    # all_len = [len(x) for x in texts]
    # print(all_len[:5])
    # print(sum(all_len))

    # alpha_count = sum([len(x) for x in texts])
    # word_count = sum([len(x.split(" ")) for x in texts])
    # print(f"texts: {len(texts)}, alpha: {alpha_count}, word_count: {word_count}")

    data_info = {
        "texts": texts,
        "questions": questions,
        "answers": answers,
    }

    return data_info


if __name__ == "__main__":

    # choices=["en", "zh", "en_int", "zh_int", "en_fact", "zh_fact"],

    rgb_info = get_rgb_info("en")
    print(len(rgb_info["texts"]), len(rgb_info["questions"]), len(rgb_info["answers"]))

    rgb_info = get_rgb_info("en_int")
    print(len(rgb_info["texts"]), len(rgb_info["questions"]), len(rgb_info["answers"]))

    rgb_info = get_rgb_info("en_fact")
    print(len(rgb_info["texts"]), len(rgb_info["questions"]), len(rgb_info["answers"]))

    rgb_info = get_rgb_info("en_refine")
    print(len(rgb_info["texts"]), len(rgb_info["questions"]), len(rgb_info["answers"]))
