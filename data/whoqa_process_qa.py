import json
import re
import os
import random
from typing import Dict, List
from utils import get_config
from utils.llm_env import LLMEnv
from data.paths import WHOQA_DATAPATH
from utils.base import file_exist

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
    使用 LLM 根据上下文生成特指问题，严格防止泄露目标属性。
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
    
    # 调用你的大模型 (温度设低一点，保证输出稳定)
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
    # 这里模拟 LLM 的返回结果
    # response = "What was the occupation of the Charles Fox who died at Mount Anville in 1862?" 
    return final_question

def prepare_incremental_experiment_data(whoqa_file_path, output_file_path, num: int = 120):
    """
    处理 WhoQA 数据，生成用于增量更新实验的结构化数据。
    """
    with open(whoqa_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    experiment_cases = []

    data_list = data if isinstance(data, list) else [data]
    sample_size = min(num, len(data_list))
    sampled_items = random.sample(data_list, sample_size) if sample_size > 0 else []
    
    for data_item in sampled_items:
        # 解析元数据
        target_property = data_item.get("question_type_metadata", {}).get("label", "occupation")
        contexts = data_item.get("contexts", [])
        answers_dict = data_item.get("answer_by_context", {})
        
        # 我们以第一个 entity 作为 Target (C2)，剩下的作为 Distractors (C1)
        # 在实际批量处理中，你可以循环将每一个 entity 都轮流作为 Target
        if len(contexts) < 2:
            continue
        
        for target_idx, target_context in enumerate(contexts):  
            # 提取基础信息
            page_id = target_context["page_id"]
            
            # 提取纯粹的人名 (去掉括号里的消歧词，例如 "Charles Fox (Irish politician)" -> "Charles Fox")
            clean_name = re.sub(r'\s*\(.*?\)\s*', '', page_id)
            target_text = target_context["candidate_texts"]
            
            # 构建 C1 (干扰项集合) 和 C2 (目标项)
            distractor_contexts = [c["candidate_texts"] for i, c in enumerate(contexts) if i != target_idx]
            
            # 获取 Target 的 Ground Truth 答案
            # 注意：WhoQA 的 answers 字典的 key 对应的是 context_ids 的索引，需要转换
            truth_answers = answers_dict.get(str(target_idx), [])
            if len(truth_answers) > 1: # 多个列表组合成一个
                truth_answers = [item for sublist in truth_answers for item in sublist]
            if not isinstance(truth_answers, list):
                truth_answers = [truth_answers]
            
            # 核心：生成特指问题
            specific_question = generate_specific_question(clean_name, target_text, target_property, truth_answers)
            
            experiment_cases.append({
                "target_entity": clean_name,
                "specific_question": specific_question,
                "phase_1_data": distractor_contexts, # 先注入这个建图
                "phase_2_data": [target_text],       # 再增量注入这个
                "ground_truth": truth_answers        # 用于自动化评测的正确答案列表
            })
            
            print(f"Generated Question for {clean_name}: {specific_question}")
            # 为了演示，只处理第一个作为 Target
            break 

    # 保存实验数据
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(experiment_cases, f, indent=4, ensure_ascii=False)
    print(f"\\nExperiment dataset saved to {output_file_path}")

if __name__ == "__main__":
    # 假设你的附件数据存为 whoqa_sample.json
    data_file = os.path.join(WHOQA_DATAPATH, "WhoQA.json")
    assert file_exist(data_file), f"{data_file} not exist!"
    output_file = "whoqa_experiment_dataset.json"
    prepare_incremental_experiment_data(data_file, output_file, num=120)

    # CUDA_VISIBLE_DEVICES="1" python -m data.whoqa_process_qa