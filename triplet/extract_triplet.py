import argparse
import json
import os
import re
import time
from typing import Literal

from tqdm import tqdm

from data.dragonball import get_dragonball_info
from data.rgb import get_rgb_info
from triplet.prompts import (
    alignment_prompt,
    prompt_extract_triplest_str,
)
from utils import file_exist, get_config, read_json, save_to_json
from utils.base import print_text, read_json

# from openai_extractor.gpt_extract_triplets import OpenAIExtractor
from utils.llm_env import LLMEnv


def extract_triplet(llm: LLMEnv, context):
    # prompt_template_strformat(context=context)
    # print(prompt_extract_triplest_str.format(context=context))
    prompt_text = prompt_extract_triplest_str.format(context=context)
    print("Prompt for extracting triplets:", prompt_text)
    return llm.complete(prompt=prompt_extract_triplest_str.format(context=context))


def align_entity_relation(self, context):

    return llm.complete(prompt=alignment_prompt.format(input_data=context))


def get_pasrse_output(self, output_str, field=Literal["aligned_triplets", "keywords"]):
    retry = 3
    while retry > 0:
        try:
            output_data = json.loads(extract_json_str(output_str))
            assert field in output_data
            if field == "aligned_triplets":
                aligned_triplets = output_data[field]
                capitalized_triplets = [
                    [
                        [
                            (phrase.capitalize() if isinstance(phrase, str) else phrase)
                            for phrase in triplet
                        ]
                        for triplet in each_item
                    ]
                    for each_item in aligned_triplets
                ]
                return capitalized_triplets
            elif field == "keywords":
                keywords = output_data[field]
                assert isinstance(keywords, list)
                return keywords
                # print("Converted output to list:")
                # print(output_data)
                # print(type(output_data))
        except json.JSONDecodeError as e:
            print(output_str)
            print("Failed to decode JSON:", e)
            retry -= 1


def get_align_output(self, output_str):
    retry = 3
    while retry > 0:
        try:
            output_data = json.loads(extract_json_str(output_str))
            assert "aligned_triplets" in output_data
            aligned_triplets = output_data["aligned_triplets"]
            capitalized_triplets = [
                [
                    [
                        phrase.capitalize() if isinstance(phrase, str) else phrase
                        for phrase in triplet
                    ]
                    for triplet in each_item
                ]
                for each_item in aligned_triplets
            ]
            return capitalized_triplets
            # print("Converted output to list:")
            # print(output_data)
            # print(type(output_data))
        except json.JSONDecodeError as e:
            print(output_str)
            print("Failed to decode JSON:", e)
            retry -= 1


def llm_extract(llm: LLMEnv, context):
    retry = 3
    while retry > 0:
        try:
            response = extract_triplet(llm, context=context)
            print("Raw response from LLM:", response)
            output = extract_json_str(response)
            parsed_output = json.loads(output)
            assert "entities" in parsed_output
            if len(parsed_output["entities"]) == 0:
                retry -= 1
                continue
            return parsed_output
        except Exception as e:
            print(f"extract triplet error: {e}")
            retry -= 1  # Decrement the retry counter
    return None


def extract_json_str(text: str) -> str:
    """Extract JSON string from text."""
    # NOTE: this regex parsing is taken from langchain.output_parsers.pydantic
    match = re.search(r"\{.*\}", text.strip(), re.MULTILINE | re.IGNORECASE | re.DOTALL)
    if not match:
        raise ValueError(f"Could not extract json string from output: {text}")
    return match.group()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Process some entities and triplets for knowledge extraction."
    )

    # parser.add_argument("--start", type=int, default=0)
    # parser.add_argument("--end", type=int, default=-1)
    # parser.add_argument("--dataset", type=str, default="rgb_en")
    parser.add_argument("--backend", type=str, default="openai")

    args = parser.parse_args()
    # print(args)

    config = get_config()
    # print(config)

    if args.backend == "openai":
        model_name = "gpt-4o-mini"
        api_key = config["model"]["OPENAI_API_KEY"]
        base_url = config["model"]["OPENAI_BASE_URL"]
    elif args.backend == "deepseek":
        model_name = "deepseek-chat"
        api_key = config["model"]["DEEPSEEK_API_KEY"]
        base_url = config["model"]["DEEPSEEK_BASE_URL"]
    elif args.backend == "ollama":
        # print("***********************************************")
        model_name = "llama3.1:8b"
        api_key = None
        base_url = config["model"]["LLAMA3_8B_URL"]
    else:
        raise ValueError(f"Unsupported backend: {args.backend}")

    llm = LLMEnv(
        backend=args.backend,
        model=model_name,
        api_key=api_key,
        base_url=base_url,
    )
    
    input_context = "Maibritt Kviesgaard (born 15 May 1986) is a former Danish handball player. She has also played on the Danish national team.\
        She competed at the 2010 European Women's Handball Championship, where the Danish team placed fourth, \
            and Kviesgaard was voted into the All-Star Team as Best Right Wing."
    output = llm_extract(llm, context=input_context)
    log_path = f"./triplet/raw_triplets/example.json"
    if output is not None:
        save_to_json(log_path, output)
    else:
        print("No triplets extracted for example input.")
    exit(0)

    if "rgb" in args.dataset:
        # data_info = get_rgb_info(f"{args.dataset[4:]}", chunk_size=2048)
        data_info = get_rgb_info()

    elif "dragonball" == args.dataset:
        data_info = get_dragonball_info("en", "Summary Question")
        texts = data_info["texts"]
        # Summary Question: 415 questions
        # "questions": questions,
        # "answers": answers,
        # "languages": languages,
        # "domains": domains,
        # "question_types": question_types,
        # "texts": texts,
        # print(len(data_info["questions"]))
        # print(len(texts), type(texts[0]))

    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    start_time = time.time()

    questions, answers, texts = (
        data_info["questions"],
        data_info["answers"],
        data_info["texts"],
    )

    print("number of questions:", len(questions))
    print("number of answers:", len(answers))
    print("number of texts:", len(texts))
    # exit(0)
    if args.end == -1:
        args.end = len(questions)

    questions = questions[args.start : args.end]
    answers = answers[args.start : args.end]
    texts = texts[args.start : args.end]

    # log_path = f'./tqa_2wiki_triplets{args.start}-{args.end}.json'
    log_path = f"./raw_triplets/{args.dataset}_triplets{args.start}-{args.end}_{args.backend}.json"

    triplet_data = {}
    if file_exist(log_path):
        triplet_data = read_json(log_path)

    for i, (q, a, text) in enumerate(
        tqdm(
            zip(questions, answers, texts),
            desc="extract triplets",
            total=len(questions),
        )
    ):
        if q in triplet_data:
            print(f"<Q: {q[:15]}>", "is alrealy processed!")
            continue

        if isinstance(text, str):
            text = [text]

        curr_triplets = []

        for t in text:

            output = llm_extract(llm, context=t)

            if output is None:
                continue

            triplets = [
                [
                    phrase.capitalize() if isinstance(phrase, str) else phrase
                    for phrase in triplet
                ]
                for triplet in output["triplets"]
            ]
            print(f"text_len {len(t)}, extract {len(triplets)} triplets.")

            curr_triplets.extend(triplets)

        print(f"doc_{i} extract {len(curr_triplets)} triplets.")
        triplet_data[q] = curr_triplets
        save_to_json(log_path, triplet_data)
    print(f"time taken: {time.time() - start_time} seconds")

# python extract_triplet.py --start 0 --end 5 --dataset rgb_en_int --backend deepseek
