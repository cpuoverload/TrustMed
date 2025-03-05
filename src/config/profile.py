from rag.types import ProfileType
from config.model_provider import (
    ollama_llama_llm,
    ollama_jina_embedding,
    hf_llama_1b_llm,
    hf_llama_8b_llm,
    hf_bge_small_embedding,
    hf_bge_large_embedding,
    bge_rerank_base,
    bge_rerank_large,
)


APP_PROFILE: ProfileType = {
    "profile_name": "app",
    "collection_name": "app",
    "embedding_model": ollama_jina_embedding,
    "chunk_size": 512,
    "chunk_overlap": 100,
    "top_k": 6,
    "llm": ollama_llama_llm,
}

# server evaluation
SERVER_EVALUATION_PROFILES: list[ProfileType] = [
    {
        "profile_name": "baseline",
        "collection_name": "baseline",
        "embedding_model": hf_bge_large_embedding,
        "chunk_size": 1024,
        "chunk_overlap": 100,
        "top_k": 3,
        "llm": hf_llama_8b_llm,
    },
    {
        "profile_name": "chunk_size_512",
        "collection_name": "chunk_size_512",
        "embedding_model": hf_bge_large_embedding,
        "chunk_size": 512,
        "chunk_overlap": 50,
        "top_k": 6,
        "llm": hf_llama_8b_llm,
    },
    {
        "profile_name": "chunk_size_256",
        "collection_name": "chunk_size_256",
        "embedding_model": hf_bge_large_embedding,
        "chunk_size": 256,
        "chunk_overlap": 25,
        "top_k": 12,
        "llm": hf_llama_8b_llm,
    },
    # the profiles below are based on chunk size 512
    {
        "profile_name": "hybrid_search",
        "collection_name": "hybrid_search",
        "embedding_model": hf_bge_large_embedding,
        "chunk_size": 512,
        "chunk_overlap": 50,
        "top_k": 6,
        "llm": hf_llama_8b_llm,
        "hybrid_search": True,
    },
    {
        "profile_name": "query_rewrite",
        "collection_name": "query_rewrite",
        "embedding_model": hf_bge_large_embedding,
        "chunk_size": 512,
        "chunk_overlap": 50,
        "top_k": 6,
        "llm": hf_llama_8b_llm,
        "hybrid_search": True,
        "query_rewrite_num": 3,
    },
    {
        "profile_name": "query_rewrite_rerank",
        "collection_name": "query_rewrite_rerank",
        "embedding_model": hf_bge_large_embedding,
        "chunk_size": 512,
        "chunk_overlap": 50,
        "top_k": 6,
        "llm": hf_llama_8b_llm,
        "hybrid_search": True,
        "query_rewrite_num": 3,
        "reranker": bge_rerank_large,
        "reranker_top_n": 6,
    },
]

# local test
LOCAL_EVALUATION_PROFILES: list[ProfileType] = [
    {
        "profile_name": "baseline",
        "collection_name": "baseline",
        "embedding_model": ollama_jina_embedding,
        "chunk_size": 1024,
        "chunk_overlap": 100,
        "top_k": 3,
        "llm": ollama_llama_llm,
    },
    {
        "profile_name": "huggingface_llm_test",
        "collection_name": "huggingface_llm_test",
        "embedding_model": hf_bge_small_embedding,
        "chunk_size": 256,
        "chunk_overlap": 25,
        "top_k": 12,
        "llm": hf_llama_1b_llm,
    },
    {
        "profile_name": "vector_search",
        "collection_name": "vector_search",
        "embedding_model": ollama_jina_embedding,
        "chunk_size": 256,
        "chunk_overlap": 25,
        "top_k": 12,
        "llm": ollama_llama_llm,
        "hybrid_search": False,
    },
    {
        "profile_name": "hybrid_search",
        "collection_name": "hybrid_search",
        "embedding_model": ollama_jina_embedding,
        "chunk_size": 256,
        "chunk_overlap": 25,
        "top_k": 12,
        "llm": ollama_llama_llm,
        "hybrid_search": True,
        # "query_rewrite_num": 3,
        "reranker": bge_rerank_base,
        "reranker_top_n": 3,
    },
]
