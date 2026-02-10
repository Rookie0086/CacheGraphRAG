import argparse
import os
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
import torch
import yaml
from llama_index.llms.ollama import Ollama
from openai import OpenAI
from pydantic import BaseModel, Field

# from llama_index.llms.openai import OpenAI
from transformers import (  # PreTrainedModel,; set_seed,; AutoConfig
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedTokenizer,
    set_seed,
)

from utils import get_config, get_project_dir
from utils.base import print_text, read_yaml

EMBEDD_DIMS = {
    "/home/shuyurui/model/bge-large-en-v1.5": 1024,
    "BAAI/bge-base-en-v1.5": 768,
    "BAAI/bge-small-en-v1.5": 384,
    "text-embedding-ada-002": 1536,
}


class EmbeddingEnv:
    def __init__(
        self,
        model_name: str = "/home/shuyurui/model/bge-large-en-v1.5",
        device: str = None,  # e.g., "cuda:0" / "cpu" / None(自动)
        normalize: bool = True,
        batch_size=-1,
    ):
        self.model_name = model_name
        self.normalize = normalize
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size

        print(f"Loading model {self.model_name} to {self.device} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        with torch.no_grad():
            dummy_input = self.tokenizer("test", return_tensors="pt").to(self.device)
            dummy_output = self.model(**dummy_input)[0][:, 0]
        self.dim = dummy_output.shape[-1]

        print(
            f"EmbeddingEnv init -> model={self.model_name}, dim={self.dim}, device={self.device}"
        )

    def __str__(self):
        return f"{self.model_name} ({self.dim}d)"

    def _encode(self, texts):
        single = isinstance(texts, str)
        texts = [texts] if single else texts

        all_embeddings = []

        batch_size = self.batch_size if self.batch_size > 0 else len(texts)

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            encoded_input = self.tokenizer(
                batch_texts, padding=True, truncation=True, return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                model_output = self.model(**encoded_input)
                batch_embeddings = model_output[0][:, 0]  # CLS pooling

            if self.normalize:
                batch_embeddings = torch.nn.functional.normalize(
                    batch_embeddings, p=2, dim=1
                )

            all_embeddings.append(batch_embeddings.cpu())

        all_embeddings = torch.cat(all_embeddings, dim=0).numpy()
        return all_embeddings[0] if single else all_embeddings

    def get_embedding(self, text: str) -> np.ndarray:
        return self._encode(text)

    def get_embeddings(self, texts: list) -> np.ndarray:
        return self._encode(texts)

    def calculate_similarity(self, text1: str, text2: str) -> float:
        e1 = self.get_embedding(text1)
        e2 = self.get_embedding(text2)
        sim = np.dot(e1, e2)
        return round(float(sim), 6)


class EmbedCache:

    def __init__(
        self, db_name="EmbedCache", embed_name="/home/shuyurui/model/bge-large-en-v1.5", verbose=False
    ):
        from database.milvus import MilvusDB

        self.embed_model = EmbeddingEnv(embed_name=embed_name)
        self.cache = MilvusDB(
            db_name, 1024, overwrite=True, metric="COSINE", verbose=False
        )
        self.cache.create(consistency_level="Session")
        self.verbose = verbose
        #   "Strong"
        #   Bounded
        #   Eventually
        #   "Session"

    def get_embedding(self, query):
        time_cost = -time.time()
        ret = self.embed_model.get_embedding(query)
        time_cost += time.time()
        if self.verbose:
            print(f"EmbedCache embedding cost {time_cost:.3f}s.")
        return ret

    def search(self, query_embedding, limit=3):
        time_cost = -time.time()
        ids, distances = self.cache.search([query_embedding], limit=limit)
        time_cost += time.time()
        if self.verbose:
            print(f"EmbedCache search embedding cost {time_cost:.3f}s.")
        return (ids, distances)

    def insert(self, id, query_embedding):
        time_cost = -time.time()
        self.cache.insert([[id], [query_embedding]])
        time_cost += time.time()
        if self.verbose:
            print(f"EmbedCache insert embedding cost {time_cost:.3f}s.")


class BaseLLMEnv(ABC):
    @abstractmethod
    def complete(self, prompt, verbose=False, return_info=False):
        pass


class OpenAIEnv(BaseLLMEnv):

    def __init__(
        self,
        model="gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature=0,
    ):
        self.model = model
        self.temperature = temperature
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        base_url = base_url or os.getenv("OPENAI_API_BASE")
        if not api_key or not base_url:
            raise ValueError("OpenAI api_key and base_url must be provided")
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def _flatten_rich_text(self, node) -> str:
        """提取接口自定义 message 结构里的纯文本内容。"""
        if node is None:
            return ""
        if isinstance(node, str):
            return node
        if isinstance(node, dict):
            if "text" in node:
                return self._flatten_rich_text(node.get("text"))
            if "children" in node:
                return self._flatten_rich_text(node.get("children"))
            if "richText" in node:
                return self._flatten_rich_text(node.get("richText"))
            if "content" in node:
                return self._flatten_rich_text(node.get("content"))
            return "".join(self._flatten_rich_text(v) for v in node.values())
        if isinstance(node, list):
            return "".join(self._flatten_rich_text(item) for item in node)
        return ""

    def complete(self, prompt, verbose=False, return_info=False):
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                temperature=self.temperature,
                stream=False,
            )
            # 解析标准 OpenAI 响应结构，优先使用 choices[0].message.content。
            if getattr(response, "choices", None):
                message = response.choices[0].message
                if message and message.content:
                    return message.content.strip()

            # 出现 choices 为空时尝试读取提供商自定义字段，避免 NoneType 异常。
            fallback_fields = [
                getattr(response, "theContent", None),
                getattr(response, "message", None),
            ]
            for field in fallback_fields:
                content = self._flatten_rich_text(field)
                if content:
                    return content

            # 无法解析内容时抛出具体错误，便于排查 API 返回。
            try:
                payload = response.model_dump()
            except Exception:  # pragma: no cover - 极端情况下 model_dump 不存在
                payload = repr(response)
            raise RuntimeError(f"API 响应缺少有效内容: {payload}")
        
        except Exception as e:
            print(f"Error in LLM API call: {e}")
            return None


class DeepSeekEnv(BaseLLMEnv):

    def __init__(
        self,
        model="deepseek-chat",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature=0,
    ):
        self.model = model
        self.temperature = temperature
        api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        base_url = base_url or os.getenv("DEEPSEEK_API_BASE")
        if not api_key or not base_url:
            raise ValueError("DeepSeek api_key and base_url must be provided")
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def complete(self, prompt, verbose=False, return_info=False):
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                temperature=self.temperature,
                stream=False,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error in LLM API call: {e}")
            return None


class OllamaEnv(BaseLLMEnv):

    def __init__(
        self,
        model="llama3.1:8b",
        timeout=300,
        temperature=0,
        base_url="http://localhost:11434",
        max_tokens=500,
    ):
        self.llm = Ollama(
            model=model,
            request_timeout=timeout,
            base_url=base_url,
            temperature=temperature,
            additional_kwargs={"num_predict": max_tokens},
        )

    def complete(self, prompt, verbose=False, return_info=False):
        output = self.llm.complete(prompt)
        response = output.text
        (
            generate_time,
            load_time,
            prefill_time,
            decode_time,
            prompt_len,
            generate_len,
        ) = self.parse_response_info(output)

        if verbose:
            print_text(f"Prompt: {prompt}\n", color="yellow")
            print_text(f"Response: {response}\n", color="green")
            print_text(
                f"generate_time {generate_time:.3f}s, load_time {load_time:.3f}s, prefill_time {prefill_time:.3f}s, decode_time {decode_time:.3f}s, prompt_len {prompt_len}, generate_len {generate_len}\n",
                color="red",
            )

        if return_info:
            return (
                response,
                prompt_len,
                generate_len,
                generate_time,
                load_time,
                prefill_time,
                decode_time,
            )
        else:
            return response

    def parse_response_info(self, response):
        # total_duration: time spent generating the response
        total_time = response.raw["total_duration"] / 1e9

        # load_duration: time spent in nanoseconds loading the model
        load_time = response.raw["load_duration"] / 1e9

        # (prefill): prompt_eval_duration: time spent in nanoseconds evaluating the prompt
        prefill_time = response.raw["prompt_eval_duration"] / 1e9

        # (generation): eval_duration: time in nanoseconds spent generating the response
        decode_time = response.raw["eval_duration"] / 1e9

        # prompt_eval_count: number of tokens in the prompt
        prompt_len = (
            response.raw["prompt_eval_count"]
            if "prompt_eval_count" in response.raw
            else 0
        )

        # eval_count: number of tokens in the response
        generate_len = response.raw["eval_count"]

        return (
            total_time,
            load_time,
            prefill_time,
            decode_time,
            prompt_len,
            generate_len,
        )


class VllmEnv(BaseLLMEnv):

    def __init__(
        self,
        model,
        tensor_parallel_size=1,
        memory_utilization=0.8,
        max_tokens=2048,
        enable_prefix_caching=True,
        # cuda_visible_devices="",
        # logging_level="WARN",
        temperature=0,
        # top_p=0.95,
    ):

        # if cuda_visible_devices:
        #     os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
        #     print(f"CUDA_VISIBLE_DEVICES={cuda_visible_devices}")
        # os.environ["VLLM_LOGGING_LEVEL"] = logging_level

        from vllm import LLM, SamplingParams

        self.sampling_params = SamplingParams(
            temperature=temperature,
            # top_p=top_p,
            max_tokens=max_tokens,
            stop=["<|im_end|>", "<|eot_id|>", "<|end_of_text|>", "<|endoftext|>"],
        )

        self.llm = LLM(
            model=model,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=memory_utilization,
            enable_prefix_caching=enable_prefix_caching,
            disable_custom_all_reduce=True,
            enforce_eager=True,
            # tokenizer_mode="auto",
        )
        # print(self.llm)
        self.tokenizer = AutoTokenizer.from_pretrained(model)

    def complete(self, prompt: str, verbose: bool = False, return_info: bool = False):
        outputs = self.llm.generate(prompt, self.sampling_params)
        response = outputs[0].outputs[0].text.strip()

        prompt_len = len(outputs[0].prompt_token_ids)
        generate_len = len(outputs[0].outputs[0].token_ids)

        generate_time, prefill_time, decode_time, wait_scheduled_time = (
            self.parse_vllm_metrics(outputs[0].metrics)
        )

        if verbose:
            print_text(f"Prompt: {prompt}\n", color="yellow")
            print_text(f"Response: {response}\n", color="green")

            print_text(
                f"prompt_len {prompt_len} generate_len {generate_len}\ngenerate_time {generate_time:.3f} prefill_time {prefill_time:.3f} decode_time {decode_time:.3f} wait_scheduled_time {wait_scheduled_time:.3f}\n",
                color="red",
            )

        if return_info:
            return (
                response,
                prompt_len,
                generate_len,
                generate_time,
                prefill_time,
                decode_time,
                wait_scheduled_time,
            )
        else:
            return response

    def parse_vllm_metrics(self, metrics):
        arrival_time = metrics.arrival_time
        first_scheduled_time = metrics.first_scheduled_time
        first_token_time = metrics.first_token_time
        finished_time = metrics.finished_time
        last_token_time = metrics.last_token_time
        time_in_queue = metrics.time_in_queue

        ttft_time = first_token_time - first_scheduled_time
        decode_time = finished_time - first_token_time
        generate_time = finished_time - first_scheduled_time
        wait_scheduled_time = first_scheduled_time - arrival_time

        return generate_time, ttft_time, decode_time, wait_scheduled_time


class HuggingfaceEnv(BaseLLMEnv):

    def __init__(
        self,
        model,
        device="cuda:0",
        max_tokens=2048,
        temperature=0,
    ):

        self.max_tokens = max_tokens
        self.temperature = temperature
        torch.cuda.set_device(device)
        self.llm = (
            AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=model,
                torch_dtype=torch.bfloat16,
                # device_map=f"cuda:{device}",
                attn_implementation="sdpa",
                # attn_implementation="flash_attention_2",
            )
            .cuda()
            .eval()
        )

        # self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
        #     pretrained_model_name_or_path=model, use_fast=False
        # )

        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=model
            # , use_fast=False
        )

    def chat(
        self,
        context: str,
        query,
        idx=0,
        verbose: bool = False,
        return_info: bool = False,
    ):

        chat = [
            # {"role": "assistant", "content": context},
            {"role": "system", "content": context},
            {"role": "user", "content": query},
        ]

        # print(chat)
        # prompt = self.tokenizer.apply_chat_template(chat, tokenize=False)
        # print_text(f"{prompt}", color='green')

        prompt = self.tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )

        if idx % 20 == 0:
            print_text(f"{prompt}", color="yellow")

        return self.complete(prompt, verbose=verbose, return_info=return_info)

    def complete(self, prompt: str, verbose: bool = False, return_info: bool = False):
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        input_ids = torch.tensor(data=[input_ids], dtype=torch.int64).cuda()
        input_length = input_ids.size(-1)

        self.tokenizer.pad_token_id = (
            self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        )

        generation_config = GenerationConfig(
            do_sample=False,
            temperature=self.temperature,
            repetition_penalty=1.0,
            num_beams=1,
            pad_token_id=self.tokenizer.pad_token_id,
            max_new_tokens=self.max_tokens,
            eos_token_id=self.tokenizer.eos_token_id,
            stop_strings=[
                "<|im_end|>",
                "<|eot_id|>",
                "<|end_of_text|>",
                "<|endoftext|>",
            ],
            # eos_token_id=999999,
            # stop_strings=None,
        )

        time_generate = -time.time()
        outputs = self.llm.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            use_cache=True,
            # eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            tokenizer=self.tokenizer,
            attention_mask=torch.ones_like(input_ids).cuda(),
        )

        output_token = outputs[0][input_length:]
        response = self.tokenizer.decode(token_ids=output_token.tolist())
        time_generate += time.time()

        if verbose:
            print_text(f"Prompt:\n{prompt}\n", color="yellow")
            print_text(f"Response:\n{response}\n", color="green")

        if return_info:
            generate_len = output_token.size(-1)
            return response, input_length, generate_len, time_generate
        else:
            return response


class DepCacheInfo(BaseModel):
    system: str
    context: Union[str, List[List[str]]]
    query: str
    id: int


class SGLangEnv(BaseLLMEnv):
    def __init__(
        self,
        model,
        max_tokens=20,
        temperature=0.0,
        port=31000,
        host="127.0.0.1",
        chunked_prefill_size=20000,
        disable_overlap_schedule=True,
        mem_fraction_static=0.9,
        log_level="info",
        gpu_id=2,
        tp_size=1,
    ):
        self.max_tokens = max_tokens
        self.temperature = temperature

        import third_party.sglang.python.sglang as sgl
        from third_party.sglang.python.sglang.srt.server_args import ServerArgs

        server_args = ServerArgs(
            model_path=model,
            port=port,
            host=host,
            device="cuda",
            tp_size=tp_size,
            base_gpu_id=gpu_id,
            chunked_prefill_size=chunked_prefill_size,
            mem_fraction_static=mem_fraction_static,
            # context_length=20000,  # 输入长度会超过默认值，需要设置
            log_level=log_level,
            disable_overlap_schedule=disable_overlap_schedule,
        )

        self.sampling_params = {
            "temperature": self.temperature,
            "max_new_tokens": self.max_tokens,
            "stop": ["<|im_end|>", "<|eot_id|>", "<|end_of_text|>", "<|endoftext|>"],
        }

        self.llm = sgl.Engine(server_args=server_args)

    def complete(self, prompt, verbose=False, return_info=False):
        outputs = self.llm.generate(prompt, self.sampling_params)
        response = outputs.get("text", "").strip()

        if return_info:
            prompt_tokens = outputs["meta_info"]["prompt_tokens"]
            cached_tokens = outputs["meta_info"]["cached_tokens"]
            cache_hit_rate = cached_tokens / prompt_tokens if prompt_tokens else 0.0
            return response, prompt_tokens, cached_tokens, cache_hit_rate
        else:
            return response

    def complete_depattn(
        self,
        prompt: str,
        system: str,
        context: list,
        query: str,
        verbose: bool = False,
        return_info: bool = False,
    ):

        depcache_info = DepCacheInfo(system=system, context=context, query=query, id=10)
        outputs = self.llm.generate(
            prompt=prompt,
            sampling_params=self.sampling_params,
            depcache_info=depcache_info,
        )
        response = outputs.get("text", "").strip()
        prompt_tokens = outputs["meta_info"]["prompt_tokens"]
        cached_tokens = outputs["meta_info"]["cached_tokens"]
        cache_hit_rate = cached_tokens / prompt_tokens

        if return_info:
            return response, prompt_tokens, cached_tokens, cache_hit_rate
        else:
            return response


class LLMEnv:

    def __init__(
        self,
        backend: Literal[
            "openai", "deepseek", "ollama", "huggingface", "vllm", "sglang"
        ] = "ollama",
        model="llama3.1:8b",
        embed_model_name="/home/shuyurui/model/bge-large-en-v1.5",
        # openai
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        # ollama
        timeout=300,
        max_tokens=20,
        # huggingface
        device="cuda:0",
        # vllm
        memory_utilization=0.9,
        enable_prefix_caching=True,
        # logging_level="WARN",
        temperature=0,
        # top_p=0.95,
        # sglang
        tp_size=1,
        port=31000,
        host="127.0.0.1",
        chunked_prefill_size=20000,
        disable_overlap_schedule=True,
        mem_fraction_static=0.9,
        log_level="info",
        gpu_id=0,
    ):
        self.backend = backend
        self.model_name = model
        self.max_tokens = max_tokens

        self.embed_model = EmbeddingEnv(model_name=embed_model_name)

        if backend == "ollama":
            self.llm = OllamaEnv(
                model=model,
                timeout=timeout,
                base_url=base_url,
                temperature=temperature,
                max_tokens=max_tokens,
            )

        elif backend == "huggingface":
            self.llm = HuggingfaceEnv(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                device=device,
            )

        elif backend == "vllm":
            self.llm = VllmEnv(
                model=model,
                max_tokens=max_tokens,
                tensor_parallel_size=tp_size,
                memory_utilization=memory_utilization,
                enable_prefix_caching=enable_prefix_caching,
                # logging_level=logging_level,
                temperature=temperature,
                # top_p=top_p,
            )

        elif backend == "sglang":
            self.llm = SGLangEnv(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                port=port,
                host=host,
                chunked_prefill_size=chunked_prefill_size,
                disable_overlap_schedule=disable_overlap_schedule,
                mem_fraction_static=mem_fraction_static,
                log_level=log_level,
                gpu_id=gpu_id,
                tp_size=tp_size,
            )

        elif backend == "openai":
            self.llm = OpenAIEnv(
                model=model,
                api_key=api_key,
                base_url=base_url,
                temperature=temperature,
                # max_tokens=max_tokens,
            )
        
        elif backend == "deepseek":
            self.llm = DeepSeekEnv(
                model=model,
                api_key=api_key,
                base_url=base_url,
                temperature=temperature,
                # max_tokens=max_tokens,
            )

        else:
            raise ValueError(f"Unsupported backend: {backend}")

    def chat(
        self,
        context: str,
        query: str,
        idx=0,
        verbose: bool = False,
        return_info: bool = False,
    ):
        return self.llm.chat(
            context, query, idx=idx, verbose=verbose, return_info=return_info
        )

    def complete(self, prompt: str, verbose: bool = False, return_info: bool = False):
        return self.llm.complete(prompt, verbose=verbose, return_info=return_info)

    def hello_world(self):
        response = self.complete("who are you?")
        print_text("Q: who are you?\n", color="red")
        print_text(f"A: {response}\n", color="green")

    # def complete_depattn(
    #     self,
    #     prompt: str,
    #     system: str,
    #     context: str,
    #     query: str,
    #     verbose: bool = False,
    #     return_info: bool = False,
    # ):
    #     return self.llm.complete_depattn(
    #         prompt, system, context, query, verbose=verbose, return_info=return_info
    #     )


if __name__ == "__main__":

    config = get_config()

    openai = LLMEnv(
        backend="openai",
        model="gpt-4o-mini",
        api_key=config["model"]["OPENAI_API_KEY"],
        base_url=config["model"]["OPENAI_BASE_URL"],
    )
    print_text(f"{openai.complete('hello, who are you?')}\n", color="red")

    deepseek = LLMEnv(
        backend="deepseek",
        model="deepseek-chat",
        api_key=config["model"]["DEEPSEEK_API_KEY"],
        base_url=config["model"]["DEEPSEEK_BASE_URL"],
    )
    print_text(f"{deepseek.complete('hello, who are you?')}\n", color="red")

    exit(0)

    ollama = LLMEnv(
        backend="ollama",
        model="llama3.1:8b",
        base_url="http://localhost:11434",
        timeout=300,
    )
    print_text(f"{ollama.complete('hello, who are you?')}\n", color="red")

    huggingface = LLMEnv(
        backend="huggingface",
        # model=config["model"]["llama-3.1-8b-instruct"],
        model=config["model"]["mistral-7b-instruct"],
        max_tokens=10,
        device="cuda:2",
    )
    # print_text(f"{huggingface.complete('hello, who are you?')}\n", color="red")
    print_text(f"{huggingface.chat('hello', 'who are you?')}\n", color="red")

    # CUDA_VISIBLE_DEVICES=1 python llm_env.py
    vllm = LLMEnv(
        backend="vllm",
        model=config["model"]["llama-3.1-8b-instruct"],
        max_tokens=20,
        tp_size=1,
        memory_utilization=0.85,
        enable_prefix_caching=False,
        temperature=0,
    )
    print_text(f"{vllm.complete('hello, who are you?')}\n", color="red")

    sglang = LLMEnv(
        backend="sglang",
        model=config["model"]["llama-3.1-8b-instruct"],
        max_tokens=20,
        temperature=0,
        tp_size=1,
        memory_utilization=0.85,
        gpu_id=1,
    )
    print_text(f"{sglang.complete('hello, who are you?')}\n", color="red")
