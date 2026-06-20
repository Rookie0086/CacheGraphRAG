import json
import re
import os
import random
from typing import Dict, List
from src.utils import get_config
from src.llm.env import LLMEnv
from data.paths import WHOQA_DATAPATH
from src.utils.base import file_exist

config = get_config()
model_name = "gpt-4o-mini"
api_key = config["model"]["OPENAI_API_KEY"]
base_url = config["model"]["OPENAI_BASE_URL"]
llm = LLMEnv(
    backend="openai",
    model="gpt-4o-mini",
    api_key=api_key,
    base_url=base_url,
) 

def _extract_json_block(text: str):
    if not text:
        return None
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    candidate = cleaned[start : end + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None

def generate_specific_question(entity_name, context_text, target_property="occupation", answer_list=None):
    """
    Generate a specific question using LLM based on context, strictly preventing leakage of the target attribute.
    """
    prompt = """
    You are an expert dataset annotator creating rigorous evaluation data for a Knowledge Graph RAG system.
    Your task is to ask about a specific person's {target_property}, but you MUST uniquely identify them using OTHER facts from their life, strictly avoiding any mention of their actual {target_property}.
    CRITICAL: The final question MUST NOT contain any word or phrase that is the same as, a synonym of, a hypernym of, a hyponym of, or otherwise semantically similar to ANY item in Answer List. If a term could be interpreted as meaning any answer_list item, it is forbidden.

    INPUT DATA:
    - Entity Name: {entity_name}
    - Target Property to Ask About: {target_property}
    - Context Text: {context_text}
    - Answer List : {answer_list}

    THINKING PROCESS (Chain of Thought):
    Step 1 (forbidden_words): Read the text and identify the exact value(s) of their {target_property} or similar to the {answer_list} (e.g., if asking for occupation, words like "politician", "dentist", "activist", "doctor", "work", "job", "career"). Expand to include synonyms, hypernyms, hyponyms, abbreviations, translations, and common paraphrases. List all as forbidden words.
    Step 2 (safe_clues): Identify 2-3 highly specific, unique facts from the text that DO NOT relate to their {target_property}. Good clues include: exact birth/death dates, specific birth/death locations, names of family members, or specific schools attended.
    Step 3 (draft): Write a draft question using ONLY the Entity Name and the safe_clues. Ask "What was the {target_property} of..."
    Step 4 (review): Check your draft against the forbidden_words and Answer List. If any word/phrase is identical to, a synonym of, a hypernym of, a hyponym of, or semantically similar to any Answer List item, rewrite it.
    Step 5 (final_question): Output the final, clean question.

    OUTPUT FORMAT:
    You MUST output a valid JSON object with exactly the following keys and no extra text:
    {{
        "forbidden_words": ["word1", "word2"],
        "safe_clues": ["clue1", "clue2"],
        "draft": "...",
        "final_question": "..."
    }}
    """
    
    # Call the LLM (set temperature low to ensure stable output)
    response = llm.complete(
        prompt=prompt.format(
            target_property=target_property,
            entity_name=entity_name,
            context_text=context_text,
            answer_list=answer_list,
        )
    )
    res_json = _extract_json_block(response)
    if not isinstance(res_json, dict):
        print("Warning: invalid JSON response from LLM, skipping.")
        return ""
    final_question = res_json.get("final_question", "")
    return final_question

def prepare_incremental_experiment_data(whoqa_file_path, output_file_path, num: int = 120):
    """
    Process WhoQA data and generate structured data for incremental update experiments.
    """
    with open(whoqa_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    experiment_cases = []

    data_list = data if isinstance(data, list) else [data]
    sample_size = min(num, len(data_list))
    sampled_items = random.sample(data_list, sample_size) if sample_size > 0 else []
    
    for data_item in sampled_items:
        # Parse metadata
        target_property = data_item.get("question_type_metadata", {}).get("label", "occupation")
        contexts = data_item.get("contexts", [])
        answers_dict = data_item.get("answer_by_context", {})
        
        # We use the first entity as the Target (C2) and the rest as Distractors (C1)
        # In actual batch processing, you can iterate to use each entity as the Target in turn
        if len(contexts) < 2:
            continue
        
        for target_idx, target_context in enumerate(contexts):  
            # Extract basic info
            page_id = target_context["page_id"]
            
            # Extract the pure name (remove disambiguation text in parentheses, e.g. "Charles Fox (Irish politician)" -> "Charles Fox")
            clean_name = re.sub(r'\s*\(.*?\)\s*', '', page_id)
            target_text = target_context["candidate_texts"]
            
            # Construct C1 (distractor set) and C2 (target)
            distractor_contexts = [c["candidate_texts"] for i, c in enumerate(contexts) if i != target_idx]
            
            # Get the ground truth answer for the Target
            # Note: the keys in WhoQA's answers dictionary correspond to context_ids indices and need conversion
            truth_answers = answers_dict.get(str(target_idx), [])
            if len(truth_answers) > 1: # Combine multiple lists into one
                truth_answers = [item for sublist in truth_answers for item in sublist]
            if not isinstance(truth_answers[0], list):
                truth_answers = [truth_answers]
            
            # Core: generate the specific question
            specific_question = generate_specific_question(clean_name, target_text, target_property, truth_answers)
            if not specific_question:
                print(f"Failed to generate question for {clean_name}, skipping.")
                continue
            
            experiment_cases.append({
                "target_entity": clean_name,
                "specific_question": specific_question,
                "phase_1_data": distractor_contexts, # Inject this first to build the graph
                "phase_2_data": [target_text],       # Then incrementally inject this
                "ground_truth": truth_answers        # Correct answer list for automated evaluation
            })
            
            print(f"Generated Question for {clean_name}: {specific_question}")
            # For demonstration, only process the first one as Target
            break 

    # Save experiment data
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(experiment_cases, f, indent=4, ensure_ascii=False)
    print(f"\\nExperiment dataset saved to {output_file_path}")

if __name__ == "__main__":
    # Assume your attached data is stored as whoqa_sample.json
    data_file = os.path.join(WHOQA_DATAPATH, "WhoQA.json")
    assert file_exist(data_file), f"{data_file} not exist!"
    output_file = "whoqa_experiment_dataset_600.json"
    prepare_incremental_experiment_data(data_file, output_file, num=600)

    # CUDA_VISIBLE_DEVICES="1" python -m data.whoqa_process_qa