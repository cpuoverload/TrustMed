from rag.types import ProfileType
from utils.model_provider import (
    ollama_llama_llm,
    ollama_jina_embedding,
    cohere_rerank,
    bge_rerank_base,
)

CHROMA_DB_DIR = "./chroma_db"
RAW_DATA_DIR = "./data/raw"
EVALUATION_DATA_DIR = "./data/evaluation"
BM25_RETRIEVER_DIR = "./bm25_retriever"

APP_PROFILE: ProfileType = {
    "profile_name": "app",
    "collection_name": "app",
    "embedding_model": ollama_jina_embedding(),
    "chunk_size": 512,
    "chunk_overlap": 100,
    "top_k": 6,
    "llm": ollama_llama_llm(),
}

EVALUATION_PROFILES: list[ProfileType] = [
    {
        "profile_name": "baseline",
        "collection_name": "baseline",
        "embedding_model": ollama_jina_embedding(),
        "chunk_size": 1024,
        "chunk_overlap": 200,
        "top_k": 3,
        "llm": ollama_llama_llm(),
    },
    {
        "profile_name": "vector_search",
        "collection_name": "vector_search",
        "embedding_model": ollama_jina_embedding(),
        "chunk_size": 256,
        "chunk_overlap": 20,
        "top_k": 12,
        "llm": ollama_llama_llm(),
        "hybrid_search": False,
    },
    {
        "profile_name": "hybrid_search",
        "collection_name": "hybrid_search",
        "embedding_model": ollama_jina_embedding(),
        "chunk_size": 256,
        "chunk_overlap": 20,
        "top_k": 12,
        "llm": ollama_llama_llm(),
        "hybrid_search": True,
        # "query_rewrite_num": 3,
        # "reranker": cohere_rerank(top_n=3),
        "reranker": bge_rerank_base(top_n=3),
    },
]
