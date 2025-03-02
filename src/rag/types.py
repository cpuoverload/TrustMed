from typing import TypedDict
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.llms.utils import LLMType


class ProfileType(TypedDict):
    profile_name: str
    collection_name: str
    embedding_model: BaseEmbedding
    chunk_size: int
    top_k: int
    llm: LLMType
    hybrid_search: bool
