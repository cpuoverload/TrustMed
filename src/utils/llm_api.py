import os
from dotenv import load_dotenv
from llama_index.llms.dashscope import DashScope, DashScopeGenerationModels
from llama_index.embeddings.dashscope import (
    DashScopeEmbedding,
    DashScopeTextEmbeddingModels,
)
from llama_index.postprocessor.cohere_rerank import CohereRerank


def _load_api_key(key_name: str):
    api_key = os.getenv(key_name)
    if not api_key:
        load_dotenv(".env")
        api_key = os.getenv(key_name)
    return api_key


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


def cohere_rerank(top_n: int = 3):
    return CohereRerank(
        api_key=_load_api_key("COHERE_API_KEY"),
        model="rerank-v3.5",
        top_n=top_n,
    )
