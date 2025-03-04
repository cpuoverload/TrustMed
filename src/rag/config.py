from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from rag.types import ProfileType
from utils.model_provider import cohere_rerank, bge_rerank_base

CHROMA_DB_DIR = "./chroma_db"
RAW_DATA_DIR = "./data/raw"
EVALUATION_DATA_DIR = "./data/evaluation"
BM25_RETRIEVER_DIR = "./bm25_retriever"

APP_PROFILE: ProfileType = {
    "profile_name": "app",
    "collection_name": "app",
    "embedding_model": OllamaEmbedding("quentinz/bge-small-zh-v1.5:latest"),
    "chunk_size": 512,
    "chunk_overlap": 100,
    "top_k": 6,
    "llm": Ollama("llama3.2"),
}

EVALUATION_PROFILES: list[ProfileType] = [
    {
        "profile_name": "baseline",
        "collection_name": "baseline",
        "embedding_model": OllamaEmbedding("quentinz/bge-small-zh-v1.5:latest"),
        "chunk_size": 1024,
        "chunk_overlap": 200,
        "top_k": 3,
        "llm": Ollama("llama3.2"),
    },
    {
        "profile_name": "vector_search",
        "collection_name": "vector_search",
        "embedding_model": OllamaEmbedding("quentinz/bge-small-zh-v1.5:latest"),
        "chunk_size": 256,
        "chunk_overlap": 20,
        "top_k": 12,
        "llm": Ollama("llama3.2"),
        "hybrid_search": False,
    },
    {
        "profile_name": "hybrid_search",
        "collection_name": "hybrid_search",
        "embedding_model": OllamaEmbedding("quentinz/bge-small-zh-v1.5:latest"),
        "chunk_size": 256,
        "chunk_overlap": 20,
        "top_k": 12,
        "llm": Ollama("llama3.2"),
        "hybrid_search": True,
        # "query_rewrite_num": 3,
        # "reranker": cohere_rerank(top_n=3),
        "reranker": bge_rerank_base(top_n=3),
    },
]
