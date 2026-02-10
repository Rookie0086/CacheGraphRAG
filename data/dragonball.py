# import argparse
import json
import os
from collections import Counter

# from llama_index.core import Document
from data.paths import DRAGONBALL_DATAPATH
from utils import file_exist, read_jsonl, save_to_json


def get_dragonball_info(language=None, query_type=None):
    data_path = os.path.join(DRAGONBALL_DATAPATH, "dragonball_queries.jsonl")
    corpus_path = os.path.join(DRAGONBALL_DATAPATH, "dragonball_docs.jsonl")
    assert file_exist(data_path), f"{data_path} not exist!"
    assert file_exist(corpus_path), f"{corpus_path} not exist!"

    # query_type {'Factual Question', 'Multi-hop Reasoning Question', 'Summary Question','Summarization Question',
    #             'Irrelevant Unsolvable Question', 'Multi-document Information Integration Question',
    #             'Multi-document Comparison Question', 'Multi-document Time Sequence Question'}

    questions = []
    answers = []
    question_types = []
    languages = []
    domains = []
    texts = []

    data = read_jsonl(data_path)
    corpus = read_jsonl(corpus_path)
    #  with open(data_path, "r", encoding="utf-8") as f:
    #     for line in f:
    #         data = json.loads(line)

    for ins in data:
        # contexts.append(ins["context"])

        domain = ins["domain"]
        language_ins = ins["language"]
        query_type_ins = ins["query"]["query_type"]

        # query_type_st.add(query["query_type"])
        # if query["query_type"] not in ["Factual Question"]:
        #     continue
        # ground_truth = ins["ground_truth"]
        # print(language, language_ins)
        if language is not None and language != language_ins:
            continue

        # print(query_type, query_type_ins)
        if query_type is not None and query_type != query_type_ins:
            continue

        question = ins["query"]["content"]
        answer = ins["ground_truth"]["content"]

        assert isinstance(question, str)
        assert isinstance(answer, str)

        questions.append(question)
        answers.append(answer)
        languages.append(language_ins)
        domains.append(domain)
        question_types.append(query_type_ins)

    for chunk in corpus:
        if language is not None and language != chunk["language"]:
            continue
        texts.append(chunk["content"])

        domain = ins["domain"]
        language_ins = ins["language"]
        query_type_ins = ins["query"]["query_type"]

        # query_type_st.add(query["query_type"])
        # if query["query_type"] not in ["Factual Question"]:
        #     continue
        # ground_truth = ins["ground_truth"]
        if language is not None and language != language_ins:
            continue

        if query_type is not None and query_type != query_type_ins:
            continue

        question = ins["query"]["content"]
        answer = ins["ground_truth"]["content"]

        assert isinstance(question, str)
        assert isinstance(answer, str)

        questions.append(question)
        answers.append(answer)
        languages.append(language_ins)
        domains.append(domain)
        question_types.append(query_type_ins)

    for chunk in corpus:
        if language is not None and language != chunk["language"]:
            continue
        texts.append(chunk["content"])

    data_info = {
        # "contexts": contexts,
        "questions": questions,
        "answers": answers,
        "languages": languages,
        "domains": domains,
        "question_types": question_types,
        "texts": texts,
    }

    return data_info


if __name__ == "__main__":

    # Summary Question: 415 questions
    # Summarization Question: 118 questions

    data_info = get_dragonball_info()
    print(f"questions: {len(data_info['questions'])}")
    print(f"answers: {len(data_info['answers'])}")
    print(f"texts: {len(data_info['texts'])}")

    data_info_en = get_dragonball_info(language="en")
    print(f"Number of questions with language='en': {len(data_info_en['questions'])}")
    print(f"Number of texts with language='en': {len(data_info_en['texts'])}")

    question_type_counts = Counter(data_info["question_types"])

    print(f"Total number of question types: {len(question_type_counts)}")

    for qtype, count in question_type_counts.items():
        print(f"{qtype}: {count} questions")

    # print(f"contexts: {len(data_info['contexts'])}")
