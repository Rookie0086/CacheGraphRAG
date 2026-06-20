import copy
import json
import os
import re

from tqdm import tqdm

from data.multihop import get_multihop_info
from data.rgb import get_rgb_info
from src.utils import file_exist, read_json, save_to_json
from src.utils.base import read_yaml
from src.llm.env import LLMEnv

home_dir = os.path.expanduser("~")

generate_similar_question_prompt = """
### Instruction
You are an advanced AI language model specialized in generating semantically similar questions. Given an input question, your task is to generate up to {num} similar questions while maintaining the core meaning but varying the phrasing.

Your final response should be formatted as follows:

**Output:**
```json
{{
    "similar_questions": [
        "Generated similar question 1",
        "Generated similar question 2",
        ...
    ]
}}

### Example

Input: “What is the capital of France?”

Output:
{{
    "similar_questions": [
        "Which city serves as the capital of France?",
        "What city is recognized as the capital of France?"
    ]
}}


### Task
Your task is to generate {num} similar questions based on the input question.

Input: "{query}"

Output:
"""


def extract_json_str(text: str) -> str:
    match = re.search(r"\{.*\}", text.strip(), re.MULTILINE | re.IGNORECASE | re.DOTALL)
    if not match:
        raise ValueError(f"Could not extract json string from output: {text}")
    return match.group()


def similar_question(llm: LLMEnv, question, num=5):
    prompt = generate_similar_question_prompt.format(query=question, num=num)
    for attempt in range(3):
        try:
            response = llm.complete(prompt)

            output = extract_json_str(response)

            parsed_output = json.loads(output)
            assert "similar_questions" in parsed_output

            return parsed_output["similar_questions"]

        except (ValueError, json.JSONDecodeError) as e:
            print(f"Attempt {attempt + 1} failed: {e}")
    return []


def generate_similar_question(llm, questions, output_path, num=5):

    data = {}
    if file_exist(output_path):
        data = read_json(output_path)

    for question in tqdm(
        questions, desc="Generating similar questions", unit="question"
    ):
        if question not in data:
            data[question] = []
        last_num = num - len(data[question])
        if last_num <= 0:
            continue
        similar_questions = similar_question(llm, question, max(3, last_num))
        data[question] = list(set(data[question]) | set(similar_questions))
        save_to_json(output_path, data, info=False)


def generate_retrieve_with_similar_question(input_path, output_path, similary_path):
    similary_questions = read_json(similary_path)
    input_data = read_json(input_path)
    output_data = []

    for item in input_data:
        output_data.append(item)
        question = item["question"]
        for sim_q in similary_questions.get(question, []):
            if sim_q == question:
                continue
            item_q = copy.copy(item)
            item_q["question"] = sim_q
            output_data.append(item_q)

    all_question = set([item["question"] for item in output_data])
    assert len(all_question) == len(
        output_data
    ), f"{len(all_question)} {len(output_data)}"

    for i, item in enumerate(output_data):
        item["id"] = i
    save_to_json(output_path, output_data)


def generate_rgb_question(llm, num=5):
    # generate similary question
    questions = get_rgb_info("en")["questions"]
    output_path = "./rgb_sim.json"

    generate_similar_question(
        llm, questions=questions, output_path=output_path, num=num
    )

    # generate similary retrieve dataset
    retrieve_input_path = f"{home_dir}/DepCache/depattn/cache/rgb_ent10_pruning30.json"
    retrieve_output_path = (
        f"{home_dir}/DepCache/depattn/cache/rgb_ent10_pruning30_similary.json"
    )
    generate_retrieve_with_similar_question(
        retrieve_input_path, retrieve_output_path, output_path
    )


def generate_multihop_question(llm, num=5):
    # generate similary question
    questions = get_multihop_info("inference_query")["questions"]
    output_path = "./multihop_sim.json"
    generate_similar_question(
        llm, questions=questions, output_path=output_path, num=num
    )

    # generate similary retrieve dataset
    retrieve_input_path = (
        f"{home_dir}/DepCache/depattn/cache/multihop_ent10_pruning30.json"
    )
    retrieve_output_path = (
        f"{home_dir}/DepCache/depattn/cache/multihop_ent10_pruning30_similary.json"
    )
    generate_retrieve_with_similar_question(
        retrieve_input_path, retrieve_output_path, output_path
    )


if __name__ == "__main__":

    questions = [
        "What is a Graph Neural Network (GNN), and how is it applied in recommendation systems?",
        "What are the main differences between GraphRAG and VectorRAG in a Retrieval-Augmented Generation (RAG) system?",
        "What is the role of KV Cache in LLM inference?",
        "How can artificial intelligence improve the accuracy of hot metal quality assessment in manufacturing?",
        "What factors influence a website’s search engine ranking in SEO?",
    ]

    config = read_yaml("../config/config.yaml")
    deepseek = LLMEnv(
        backend="deepseek",
        model="deepseek-chat",
        api_key=config["model"]["DEEPSEEK_API_KEY"],
        base_url=config["model"]["DEEPSEEK_BASE_URL"],
    )

    generate_similar_question(
        deepseek, questions=questions, output_path="./test.log", num=5
    )

    generate_rgb_question(deepseek, num=5)
    generate_multihop_question(deepseek, num=5)
