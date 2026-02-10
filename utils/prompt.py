# from utils.llm_env import OllamaEnv
import json

from llama_index.core.prompts.base import PromptTemplate
from llama_index.core.prompts.prompt_type import PromptType

# from litellm import completion
from utils.base import extract_json_str, print_text

# import re
from utils.timer import Timer

llama_preamble = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are an intelligent AI assistant. Please answer questions based on the user's instructions. Below are some reference graph retrieval results that may help you in answering the user's question.\n\n"
llama_query = "<|eot_id|>\n<|start_header_id|>user<|end_header_id|>\n\nPlease write a high-quantify answer for the given question using only the provided context information (some of which might be irrelevant). Answer directly without explanation and keep the response short and direct.\nQuestion: {question}<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>"

mistral_preamble = "<s>[INST] You are an intelligent AI assistant. Please answer questions based on the user's instructions. Below are some reference graph retrieval results that may help you in answering the user's question.\n\n"
mistral_query = "\n\nPlease write a high-quantify answer for the given question using only the provided context information (some of which might be irrelevant). Answer directly without explanation and keep the response short and direct.\nQuestion: {question}\nAnswer: [/INST]"

qwen_preamble = "<|im_start|>system\nYou are an intelligent AI assistant. Please answer questions based on the user's instructions. Below are some reference graph retrieval results that may help you in answering the user's question.\n\n"
qwen_query = "<|im_end|>\n<|im_start|>user\nPlease write a high-quantify answer for the given question using only the provided context information (some of which might be irrelevant). Answer directly without explanation and keep the response short and direct.\nQuestion: {question}<|im_end|>\n<|im_start|>assistant\n"


gpt_preamble = "You are an intelligent AI assistant. Please answer questions based on the user's instructions. Below are some reference graph retrieval results that may help you in answering the user's question.\n\n"
gpt_query = "\nPlease answer the given question using only the provided context. Respond with the exact answer only, without any additional text or explanation. If the provided context information is insufficient to answer the question, respond with 'Insufficient Information.'\nQuestion: {question}\n"


llama_query_norag = "<|start_header_id|>user<|end_header_id|>\n\nPlease write a high-quantify answer for the given question. Answer directly without explanation. The answer to the question is a word or entity.\nQuestion: {question}\nAnswer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>"
qwen_query_norag = "<|im_start|>user\nPlease write a high-quantify answer for the given question. Answer directly without explanation. The answer to the question is a word or entity.\nQuestion: {question}\nAnswer: <|im_end|>\n<|im_start|>assistant\n"


## prompt for microsoft's GraphRAG
preamble_ms = "You are a helpful assistant responding to questions about data in the tables provided.\n---Goal---:\nAnswer directly without explanation and keep the response short and direct.\nIf you don't know the answer, just say so. Do not make anything up.\n---Data tables---:\n\n"
query_ms = "Please write a high-quantify answer for the given question using only the provided context information (some of which might be irrelevant). Answer directly without explanation and keep the response short and direct.\nQuestion: {question}"


llama_preamble_ms = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant responding to questions about data in the tables provided.\n---Goal---:\nAnswer directly without explanation and keep the response short and direct.\nIf you don't know the answer, just say so. Do not make anything up.\n---Data tables---:\n\n"
llama_query_ms = "<|eot_id|>\n<|start_header_id|>user<|end_header_id|>\n\nPlease write a high-quantify answer for the given question using only the provided context information (some of which might be irrelevant). Answer directly without explanation and keep the response short and direct.\nQuestion: {question}<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>"

mistral_preamble_ms = "<s>[INST] You are a helpful assistant responding to questions about data in the tables provided.\n---Goal---:\nAnswer directly without explanation and keep the response short and direct.\nIf you don't know the answer, just say so. Do not make anything up.\n---Data tables---:\n\n"
mistral_query_ms = "\n\nPlease write a high-quantify answer for the given question using only the provided context information (some of which might be irrelevant). Answer directly without explanation and keep the response short and direct.\nQuestion: {question}\nAnswer: [/INST]"


llama3_keyword_synonyms_prompt_template = """
Given the user query below, identify the main keywords, important entities, and relevant concepts that could be used for knowledge graph search. Additionally, for each extracted keyword or entity, generate a list of possible synonyms or related terms to increase the matching probability.

Your final response must be a single JSON object in the following format and should not include any additional text or explanations:
**Output:**
```json
{{
    "Keywords": ["keyword_1", "keyword_2", ...],
    "Synonyms": {{
        "keyword_1": ["synonym_1", "synonym_2", ...],
        "keyword_2": ["synonym_1", "synonym_2", ...],
        ...
    }}
}}
```

### Example:
**Input:**
Question: “What is the capital of France?”

**Output:**
```json
{{
    "Keywords": ["capital", "France"],
    "Synonyms": {{
        "capital": ["main city", "metropolis", "seat of government"],
        "France": ["French Republic", "Hexagon", "Country in Europe"],
    }}
}}
```

### Task:
Your task is to extract up to {max_keywords} keywords from the question and generate their synonyms. Return only the JSON object and nothing else.

**Input:**
Question: "{question}"

**Output:**
"""

llama3_keyword_prompt_template = """
Given the user query below, identify the main keywords, important entities, and relevant concepts that could be used for knowledge graph search.

Your final response must be a single JSON object in the following format and should not include any additional text or explanations:
**Output:**
```json
{{
    "Keywords": ["keyword_1", "keyword_2", ...]
}}
```

### Example:
**Input:**
Question: “What is the capital of France?”

**Output:**
```json
{{
    "Keywords": ["capital", "France"]
}}
```

### Task:
Your task is to extract up to {max_keywords} keywords from the question. Return only the JSON object and nothing else.

**Input:**
Question: "{question}"

**Output:**
"""


def test_extract_keyword(llm):
    questions = [
        "How much did Avatar: The Way of Water earn in its debut weekend at U.S.?",
        "When was the Final Fantasy Pixel Remaster Series be released on Nintendo Switch?",
        "What movie won the Oscar for Best Animated Film in 2023?",
        "Who discovered asteroid 2022 YG?",
        "Which team won the Peach Bowl 2022?",
        "Which team won the 2023 Desert Hockey Classic?",
        "What is the price of Microsoft 365 Basic per month?",
    ]

    # llama3_2_keyword_extract_prompt = (
    #     "A question is provided below. Given the question, extract up to {max_keywords} "
    #     "keywords from the text. Focus on extracting the keywords that we can use "
    #     "to best lookup answers to the question. Avoid stopwords.\n"
    #     "Note, result should be in the following comma-separated format, and start with KEYWORDS:'\n"
    #     "Only response the results, do not say any word or explain.\n"
    #     "---------------------\n"
    #     "question: {question}\n"
    #     "---------------------\n")

    # llama3_2_keyword_extract_prompt = (
    #     "Extract up to {max_keywords} relevant keywords from the following question to facilitate entity matching in a knowledge graph. The keywords should focus on the core entities, actions, and important descriptors mentioned. Separate each keyword with a comma. \n"
    #     "Question: {question}\n"
    #     "Keywords (up to 5, comma-separated):\n")

    timer = Timer(verbose=True)
    for question in questions:
        print(f"question_{question}")
        # prompt = llama3_2_keyword_extract_prompt.format(question=question,
        #                                                 max_keywords=5)

        # prompt = llama3_keyword_synonyms_prompt_template.format(
        #     question=question, max_keywords=5)

        prompt = llama3_keyword_prompt_template.format(
            question=question, max_keywords=5
        )
        # prompt = llama3_keyword_extract_prompt.format(question=question, max_keywords=5)

        with timer.timing("extract keywords"):
            # output = llm.complete(prompt)
            # output = extract_json_str(output)
            # parsed_output = json.loads(output)
            # keywords_list = parsed_output.get("Keywords", [])
            # synonyms_list = [synonym for synonyms in parsed_output.get("Synonyms", {}).values() for synonym in synonyms]
            # print("Keywords:", keywords_list)
            # print("Synonyms:", synonyms_list)

            retry = 3
            while retry > 0:
                try:
                    output = llm.complete(prompt)
                    output = extract_json_str(output)
                    parsed_output = json.loads(output)
                    assert "Keywords" in parsed_output or "Synonyms" in parsed_output
                    keywords_list = parsed_output.get("Keywords", [])
                    synonyms_list = [
                        synonym
                        for synonyms in parsed_output.get("Synonyms", {}).values()
                        for synonym in synonyms
                    ]

                    print("Keywords:", keywords_list)
                    print("Synonyms:", synonyms_list)
                    break
                except Exception as e:
                    print(f"JSON format error: {e}")
                    retry -= 1  # Decrement the retry counter
    print(timer.summary())


# Response only with JSON in a format where we can jsonify in python and feed directly into  cy.add(data); to display a graph on the front-end.
# Make sure the target and source of edges match an existing node.
# Do not include the markdown triple quotes above and below the JSON, jump straight into it with a curly bracket.

if __name__ == "__main__":
    # llm_env = OllamaEnv(llm_mode_name=args.llm, port=args.port)
    from utils.llm_env import OllamaEnv

    # llm_env = OllamaEnv(llm_mode_name='llama3.1:70b', port=11434)
    llm_env = OllamaEnv(llm_model_name="llama3.2:3b", port=11434)
    # llm_env = OllamaEnv(llm_model_name='llama3.2:3b', port=11434)
    # llm_env = OllamaEnv(llm_model_name='llama3.1:70b', port=11434)
    test_extract_keyword(llm_env)
