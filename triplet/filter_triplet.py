import json
import argparse
from utils import file_exist, read_json, save_to_json

def clean_extracted_data(data_or_str):
    """
    清洗LLM输出：
    1. 解析JSON
    2. 移除 source 或 target 不在 entities 列表中的“悬空关系”
    """
    if isinstance(data_or_str, str):
        try:
            data = json.loads(data_or_str)
        except json.JSONDecodeError:
            return None, None  # 或者尝试修复JSON
    else:
        data = data_or_str

    entities = data.get("entities", [])
    relations = data.get("relations", [])

    # 1. 过滤无效实体（缺失 id/type 或为空字符串）
    valid_entities = []
    dropped_entities = 0
    for e in entities:
        if not isinstance(e, dict):
            dropped_entities += 1
            continue
        entity_id = e.get("id")
        entity_type = e.get("type")
        if not isinstance(entity_id, str) or not entity_id.strip():
            dropped_entities += 1
            continue
        if not isinstance(entity_type, str) or not entity_type.strip():
            dropped_entities += 1
            continue
        valid_entities.append(e)

    if dropped_entities > 0:
        print(f"Warning: Dropped {dropped_entities} invalid entities.")

    # 2. 构建实体 ID 集合 (Set for O(1) lookup)
    valid_entity_ids = {e["id"] for e in valid_entities}

    valid_relations = []
    dropped_relations = 0

    # 3. 过滤悬空关系
    for r in relations:
        src = r.get("src")
        tgt = r.get("tgt")
        
        # 核心检查：两端都必须是已定义的实体
        if src in valid_entity_ids and tgt in valid_entity_ids:
            valid_relations.append(r)
        else:
            dropped_relations += 1
            
    if dropped_relations > 0:
        print(f"Warning: Dropped {dropped_relations} dangling relations.")

    return valid_entities, valid_relations

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    print(args)

    input_file = args.input
    output_file = args.output

    assert file_exist(input_file)

    raw_data = read_json(input_file)
    entities, relations = clean_extracted_data(raw_data)
    if entities is None or relations is None:
        raise ValueError("Failed to parse input JSON.")

    filtered_data = {
        "entities": entities,
        "relations": relations,
    }
    save_to_json(output_file, filtered_data)

# python -m triplet.filter_triplet --input triplet/raw_triplets/example.json --output triplet/filtered_triplets/example.json