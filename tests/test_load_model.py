import os
from huggingface_hub import snapshot_download


def main():
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    repo_id = "BAAI/bge-reranker-v2-m3"
    target_dir = "/home/shuyurui/model"
    os.makedirs(target_dir, exist_ok=True)

    snapshot_download(
        repo_id=repo_id,
        local_dir=os.path.join(target_dir, "bge-reranker-v2-m3"),
        local_dir_use_symlinks=False,
    )

    print("Download complete.")


if __name__ == "__main__":
    main()
