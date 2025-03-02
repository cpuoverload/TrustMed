from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from rag.types import ProfileType

CHROMA_DB_DIR = "./chroma_db"
RAW_DATA_DIR = "./data/raw"
EVALUATION_DATA_DIR = "./data/evaluation"
BM25_RETRIEVER_DIR = "./bm25_retriever"

PROFILES: list[ProfileType] = [
    {
        "profile_name": "app",
        "collection_name": "app",
        "embedding_model": OllamaEmbedding("quentinz/bge-small-zh-v1.5:latest"),
        "chunk_size": 512,
        "top_k": 6,
        "llm": Ollama("llama3.2"),
        "hybrid_search": False,
    },
    {
        "profile_name": "vector_search",
        "collection_name": "vector_search",
        "embedding_model": OllamaEmbedding("quentinz/bge-small-zh-v1.5:latest"),
        "chunk_size": 256,
        "top_k": 12,
        "llm": Ollama("llama3.2"),
        "hybrid_search": False,
    },
    {
        "profile_name": "hybrid_search",
        "collection_name": "hybrid_search",
        "embedding_model": OllamaEmbedding("quentinz/bge-small-zh-v1.5:latest"),
        "chunk_size": 256,
        "top_k": 12,
        "llm": Ollama("llama3.2"),
        "hybrid_search": True,
    },
    {
        "profile_name": "evaluation_1",
        "collection_name": "evaluation_1",
        "embedding_model": OllamaEmbedding("quentinz/bge-small-zh-v1.5:latest"),
        "chunk_size": 1024,
        "top_k": 3,
        "llm": Ollama("llama3.2"),
        "hybrid_search": False,
    },
    {
        "profile_name": "evaluation_2",
        "collection_name": "evaluation_2",
        "embedding_model": OllamaEmbedding("quentinz/bge-small-zh-v1.5:latest"),
        "chunk_size": 512,
        "top_k": 6,
        "llm": Ollama("llama3.2"),
        "hybrid_search": False,
    },
    {
        "profile_name": "evaluation_3",
        "collection_name": "evaluation_3",
        "embedding_model": OllamaEmbedding("quentinz/bge-small-zh-v1.5:latest"),
        "chunk_size": 256,
        "top_k": 12,
        "llm": Ollama("llama3.2"),
        "hybrid_search": False,
    },
]
