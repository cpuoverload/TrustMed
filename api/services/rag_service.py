# RAG related services
from llama_index.core import VectorStoreIndex
from tools.get_llm import get_llm


class RAGService:
    def __init__(self, index: VectorStoreIndex, streaming: bool = False):
        self.index: VectorStoreIndex = index
        self.query_engine = index.as_query_engine(llm=get_llm(), streaming=streaming)

    def query(self, question: str):
        return self.query_engine.query(question)
