
1. download datasets from the following links

- rgb dataset: https://github.com/chen700564/RGB/tree/master/data

- multihop dataset: https://github.com/yixuantt/MultiHop-RAG/tree/main/dataset

- dragonball dataset: https://github.com/OpenBMB/RAGEval/tree/main/dragonball_dataset

- tqa dataset: https://huggingface.co/datasets/Seongill/trivia (need convert to json format)


2. modify the dataset paths in `utils/paths.py`

    The expected dataset directory structure is as follows:

    ```bash
    rgb
    ├── en_fact.json
    ├── en_int.json
    ├── en.json
    └── en_refine.json

    multihop
    ├── corpus.json
    └── MultiHopRAG.json

    dragonball
    ├── dragonball_docs.jsonl
    ├── dragonball.json
    └── dragonball_queries.jsonl

    tqa
    ├── test.json
    └── train.json
    ```
