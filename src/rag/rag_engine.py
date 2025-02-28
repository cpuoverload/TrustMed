from typing import TypedDict
from llama_index.core import VectorStoreIndex
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.llms.utils import LLMType
from rag.chroma_index import ChromaIndexManager


class ProfileType(TypedDict):
    collection_name: str
    embedding_model: BaseEmbedding
    llm: LLMType


class RAGEngine:
    def __init__(
        self,
        index: VectorStoreIndex,
        llm: LLMType,
        streaming: bool = False,
    ):
        self.index = index
        self.query_engine = index.as_query_engine(llm=llm, streaming=streaming)

    def query(self, question: str):
        return self.query_engine.query(question)


def create_rag_engine(profile: ProfileType, data_dir: str, streaming: bool = False):
    chroma_manager = ChromaIndexManager(
        collection_name=profile["collection_name"],
        embedding_model=profile["embedding_model"],
    )
    if chroma_manager.get_count() == 0:
        index = chroma_manager.create_index(data_dir)
    else:
        index = chroma_manager.load_index()
    rag_engine = RAGEngine(index, llm=profile["llm"], streaming=streaming)
    return rag_engine
