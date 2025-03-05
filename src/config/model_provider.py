import os
from dotenv import load_dotenv
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.dashscope import DashScope, DashScopeGenerationModels
from llama_index.embeddings.dashscope import (
    DashScopeEmbedding,
    DashScopeTextEmbeddingModels,
)
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.postprocessor.flag_embedding_reranker import (
    FlagEmbeddingReranker,
)


def _load_api_key(key_name: str):
    api_key = os.getenv(key_name)
    if not api_key:
        load_dotenv(".env")
        api_key = os.getenv(key_name)
    return api_key


def ollama_llama_llm():
    return Ollama("llama3.2")


def ollama_jina_embedding():
    return OllamaEmbedding("jina/jina-embeddings-v2-small-en")


def hf_llama_1b_llm():
    _load_api_key("HF_TOKEN")

    # todo: 添加 completion_to_prompt

    return HuggingFaceLLM(
        model_name="meta-llama/Llama-3.2-1B",
        tokenizer_name="meta-llama/Llama-3.2-1B",
        context_window=4000,
        max_new_tokens=256,
        model_kwargs={"torch_dtype": "float16"},
        generate_kwargs={"temperature": 0.7, "top_k": 50, "top_p": 0.95},
        device_map="auto",
    )


def hf_llama_8b_llm():
    _load_api_key("HF_TOKEN")

    # HuggingFaceLLM 默认不会改动原始 prompt，所以不符合 llama 所需格式，导致输出中有重复、胡言乱语
    def completion_to_prompt(completion):
        return (
            f"<|start_header_id|>user<|end_header_id|>\n{completion}<|eot_id|>\n"
            f"<|start_header_id|>assistant<|end_header_id|>\n"
        )

    return HuggingFaceLLM(
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        tokenizer_name="meta-llama/Llama-3.1-8B-Instruct",
        context_window=8192,
        max_new_tokens=1024,
        model_kwargs={"torch_dtype": "float16"},
        generate_kwargs={"temperature": 0.7, "top_k": 50, "top_p": 0.95},
        device_map="auto",
        completion_to_prompt=completion_to_prompt,
    )


def hf_bge_small_embedding():
    _load_api_key("HF_TOKEN")

    return HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")


def hf_bge_large_embedding():
    _load_api_key("HF_TOKEN")

    return HuggingFaceEmbedding(model_name="BAAI/bge-large-en-v1.5")


def qwen_llm():
    return DashScope(
        model_name=DashScopeGenerationModels.QWEN_PLUS,
        api_key=_load_api_key("DASHSCOPE_API_KEY"),
        max_tokens=8192,  # default 256 will truncate output
    )


def qwen_embedding():
    return DashScopeEmbedding(
        model_name=DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V3,
        api_key=_load_api_key("DASHSCOPE_API_KEY"),
    )


def cohere_rerank(top_n: int):
    return CohereRerank(
        api_key=_load_api_key("COHERE_API_KEY"),
        model="rerank-v3.5",
        top_n=top_n,
    )


def bge_rerank_base(top_n: int):
    # Model size: 1.11GB, will be downloaded automatically
    return FlagEmbeddingReranker(
        model="BAAI/bge-reranker-base",
        top_n=top_n,
    )


def bge_rerank_large(top_n: int):
    # Model size: 2.24GB, will be downloaded automatically
    return FlagEmbeddingReranker(
        model="BAAI/bge-reranker-large",
        top_n=top_n,
    )
