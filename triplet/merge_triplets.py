from utils import read_json, save_to_json

input_json = [
    "tqa_2wiki_triplets0-100.json",
    "tqa_2wiki_triplets100-200.json",
    "tqa_2wiki_triplets200-300.json",
]

output_json = "tqa_2wiki_triplets0-300.json"
all_data = {}

for file in input_json:
    data = read_json(file)
    print(f"read {len(data)} from {file}")
    all_data.update(data)

print(f"total {len(all_data)} data.")

save_to_json(output_json, all_data)
