from utils import get_config
from utils.llm_env import LLMEnv


if __name__ == "__main__":
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

    response = llm.complete(prompt="What is the capital of France?")

    print(response)