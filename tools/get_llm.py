from llama_index.llms.ollama import Ollama
from config.rag_config import LLM_MODEL


def get_llm():
    if LLM_MODEL == "llama3.1":
        return Ollama(model=LLM_MODEL, request_timeout=360.0)
    else:
        raise ValueError(f"Unsupported LLM: {LLM_MODEL}")
