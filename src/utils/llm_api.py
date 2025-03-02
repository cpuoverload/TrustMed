import os
from dotenv import load_dotenv
from llama_index.llms.dashscope import DashScope, DashScopeGenerationModels
from llama_index.embeddings.dashscope import (
    DashScopeEmbedding,
    DashScopeTextEmbeddingModels,
)


def _load_api_key():
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        load_dotenv(".env")
        api_key = os.getenv("DASHSCOPE_API_KEY")
    return api_key


def get_llm():
    api_key = _load_api_key()
    return DashScope(
        model_name=DashScopeGenerationModels.QWEN_PLUS,
        api_key=api_key,
        max_tokens=8192,  # default 256 will truncate output
    )


def get_embedding():
    api_key = _load_api_key()
    return DashScopeEmbedding(
        model_name=DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V3,
        api_key=api_key,
    )
