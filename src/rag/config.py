from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

CHROMA_DB_DIR = "./chroma_db"
RAW_DATA_DIR = "./data/raw"
EVALUATION_DATA_DIR = "./data/evaluation"

PROFILES = [
    {
        "collection_name": "app",
        "embedding_model": OllamaEmbedding("quentinz/bge-small-zh-v1.5:latest"),
        "llm": Ollama("llama3.1"),
    },
    {
        "collection_name": "evaluation_1",
        "embedding_model": OllamaEmbedding("quentinz/bge-small-zh-v1.5:latest"),
        "llm": Ollama("llama3.1"),
    },
]
