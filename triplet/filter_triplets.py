def escape_str(value: str) -> str:
    if not value or len(value) == 0:
        return value

    patterns = {
        '"': " ",
        "{": "",
        "}": "",
    }
    for pattern in patterns:
        if pattern in value:
            value = value.replace(pattern, patterns[pattern])
    if value[0] == " " or value[-1] == " ":
        value = value.strip()
    value = " ".join(value.split())
    return value


if __name__ == "__main__":
    # questions, answers, documents, ids = get_tqa_train_instances()
    # tqa train has 78785 item
    # load 78785 questions.
    # load 78785 answers.
    # load 7878500 documents.
    # load 7878500 ids.
    # documents min/max/avg length: 201/2510/609.6103345814558
    # query_triplet_file = './tq_triplets.json'
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    print(args)

    input_file = args.input
    output_file = args.output

    from utils import file_exist, read_json, save_to_json

    assert file_exist(input_file)

    triplet_list = read_json(input_file)
    all_triplets = []
    for k, v in triplet_list.items():
        for doc_triplet in v:
            if isinstance(doc_triplet[0], list):
                all_triplets.extend(doc_triplet)
            else:
                all_triplets.append(doc_triplet)

    print("all triplets", len(all_triplets))

    filter_triplets = []
    for i, triplet in enumerate(all_triplets):
        assert isinstance(triplet, list)
        if len(triplet) != 3:
            # print(triplet)
            continue
        if any(isinstance(elem, list) for elem in triplet):
            # print(triplet)
            continue

        # assert len(triplet) == 3, triplet
        triplet = tuple(str(x) for x in triplet)
        triplet = tuple(escape_str(x) for x in triplet)
        triplet = tuple(x.capitalize() for x in triplet)

        if any(elem.lower() in ["", "null", "none", "", "unknown"] for elem in triplet):
            continue
        if any(elem.lower().startswith("unknown") for elem in triplet):
            continue
        x, y, z = triplet
        if len(x) >= 256 or len(z) >= 256:
            continue
        filter_triplets.append(triplet)

    filter_triplets = [(str(x), str(y), str(z)) for x, y, z in filter_triplets]
    filter_triplets = list(set(filter_triplets))

    # triplet_file = './triplets/tqa_triplets.json'
    # triplet_file = './triplets/tqa_triplets20000-30000.json'
    # doc_triplet = list(set(doc_triplet))
    save_to_json(output_file, filter_triplets)
