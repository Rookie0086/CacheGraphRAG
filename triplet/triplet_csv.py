# from utils import save_to_json
import csv

from tqdm import tqdm

from utils import read_json


def generate_csv_files(
    triplets, entities_file="entities.csv", relationships_file="relationships.csv"
):

    print(f"convert {len(triplets)} triplets to {entities_file}, {relationships_file}")

    entities = set()
    relationships = []

    for triplet in triplets:
        src_entity, relationship, dst_entity = triplet
        from utils.base import escape_str

        src_entity = escape_str(src_entity).capitalize()
        relationship = escape_str(relationship).capitalize()
        dst_entity = escape_str(dst_entity).capitalize()

        # if 'nce more, with feeling' in dst_entity:
        #     print(src_entity)
        #     print(relationship)
        #     print(dst_entity)
        #     exit(0)

        entities.add((src_entity, src_entity))  # (id, name)
        entities.add((dst_entity, dst_entity))

        relationships.append((src_entity, dst_entity, relationship))

    # 写入 entities.csv 文件
    with open(entities_file, mode="w", newline="", encoding="utf-8") as entities_csv:
        writer = csv.writer(entities_csv)
        # writer.writerow(['id', 'name'])  # 写入表头
        for entity in tqdm(entities, "write entity..."):
            writer.writerow(entity)

    # 写入 relationships.csv 文件
    with open(
        relationships_file, mode="w", newline="", encoding="utf-8"
    ) as relationships_csv:
        writer = csv.writer(relationships_csv)
        # writer.writerow(['src_id', 'dst_id', 'relationship'])  # 写入表头
        for relationship in tqdm(relationships, "write relationship..."):
            writer.writerow(relationship)


if __name__ == "__main__":

    import argparse

    from utils import create_dir, file_exist

    parser = argparse.ArgumentParser(description="")

    parser.add_argument("--data", type=str, default="rgb", help="")

    args = parser.parse_args()

    triplet_path = f"./triplets/{args.data}_triplets.json"
    triplets = read_json(triplet_path)

    loaded_triplets = [(str(x), str(y), str(z)) for x, y, z in triplets]
    loaded_triplets = list(set(loaded_triplets))

    assert file_exist(triplet_path)

    create_dir("./csv")

    entities_path = f"./csv/{args.data}_entities.csv"
    relationships_path = f"./csv/{args.data}_relationships.csv"

    # triplets = [
    #     ('Elon Musk', 'owned', 'X.com'),
    #     ('Elon Musk', 'founded', 'SpaceX'),
    #     ('X.com', 'merged', 'PayPal')
    # ]
    generate_csv_files(loaded_triplets, entities_path, relationships_path)
