from typing import TypedDict, Optional, Callable
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.llms.utils import LLMType
from llama_index.core.postprocessor.types import BaseNodePostprocessor


class ProfileType(TypedDict):
    profile_name: str
    collection_name: str
    chunk_size: int
    chunk_overlap: int
    embedding_model: Callable[[], BaseEmbedding]
    top_k: int
    llm: Callable[[], LLMType]
    hybrid_search: Optional[bool]
    query_rewrite_num: Optional[int]
    reranker: Optional[Callable[[int], BaseNodePostprocessor]]
    reranker_top_n: Optional[int]
