import json
import os
import math
import random
from typing import List

from data.paths import RGB_DATAPATH
from src.utils import file_exist


_tokenizer = None


def _get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        from transformers import AutoTokenizer
        _tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")
    return _tokenizer


def compact_string(texts: List[str], chunk_size=2048):
    compact_texts = []
    cur_string = ""
    for text in texts:
        cur_string += text + "\n"

        # if len(cur_string) > length:
        #     compact_texts.append(cur_string)
        #     cur_string = ''

        input_ids = _get_tokenizer().encode(cur_string, add_special_tokens=False)

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

def processdata(instance, noise_rate, passage_num, filename, correct_rate = 0):
    query = instance['query']
    ans = instance['answer']

    neg_num = math.ceil(passage_num * noise_rate)
    pos_num = passage_num - neg_num

    if '_int' in filename:
        for i in instance['positive']:
            random.shuffle(i)
        print(len(instance['positive']))
        docs = [i[0] for i in instance['positive']]
        if len(docs) < pos_num:
            maxnum = max([len(i) for i in instance['positive']])
            for i in range(1,maxnum):
                for j in instance['positive']:
                    if len(j) > i:
                        docs.append(j[i])
                        if len(docs) == pos_num:
                            break
                if len(docs) == pos_num:
                    break
        neg_num = passage_num - len(docs)
        if neg_num > 0:
            negative = instance['negative'][:neg_num]
            docs += negative
    elif '_fact' in filename:
        correct_num = math.ceil(passage_num * correct_rate)
        pos_num = passage_num - neg_num - correct_num
        indexs = list(range(len(instance['positive'])))
        selected = random.sample(indexs,min(len(indexs),pos_num))
        docs = [instance['positive_wrong'][i] for i in selected]
        remain = [i for i in indexs if i not in selected]
        if correct_num > 0 and len(remain) > 0:
            docs += [instance['positive'][i] for i in random.sample(remain,min(len(remain),correct_num))]
        if neg_num > 0:
            docs += instance['negative'][:neg_num]
    else:
        if noise_rate == 1:
            neg_num = passage_num
            pos_num = 0
        else:
            if neg_num > len(instance['negative']):
                neg_num = len(instance['negative'])
                pos_num = passage_num - neg_num
            elif pos_num > len(instance['positive']):
                pos_num = len(instance['positive'])
                neg_num = passage_num - pos_num
        

        positive = instance['positive'][:pos_num]
        negative = instance['negative'][:neg_num]

        docs = positive + negative

    random.shuffle(docs)
    
    return query, ans, docs

def _iter_json_objects(file_path):
    """Read JSON objects one by one (compatible with both pretty-print multi-line and single-line formats)."""
    with open(file_path, "r") as f:
        buf = ""
        depth = 0
        for line in f:
            buf += line
            for ch in line:
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
            if depth == 0 and buf.strip():
                yield json.loads(buf)
                buf = ""


def get_rgb_info(file="en", chunk_size=512):
    data_file = os.path.join(RGB_DATAPATH, f"{file}.json")
    assert file_exist(data_file), f"{data_file} not exist!"

    texts = []
    questions = []
    answers = []
    for instance in _iter_json_objects(data_file):
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
        elif file == "en_refine":
            pos_texts = concat_strings_in_list(instance["positive"])
            neg_texts = concat_strings_in_list(instance["negative"])
            texts.extend(pos_texts + neg_texts)
        else:
            pos_text = " ".join(concat_strings_in_list(instance["positive"]))
            texts.append(pos_text)
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
