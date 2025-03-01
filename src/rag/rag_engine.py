from llama_index.core import VectorStoreIndex
from llama_index.core.llms.utils import LLMType
from rag.chroma_index import ChromaIndexManager
from rag.types import ProfileType
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.response_synthesizers import get_response_synthesizer


class RAGEngine:
    def __init__(
        self,
        index: VectorStoreIndex,
        top_k: int,
        llm: LLMType,
        streaming: bool = False,
    ):
        self.index = index

        retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=top_k,
        )
        response_synthesizer = get_response_synthesizer(
            llm=llm,
            streaming=streaming,
        )
        self.query_engine = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
        )

    def query(self, question: str):
        return self.query_engine.query(question)


def create_rag_engine(profile: ProfileType, data_dir: str, streaming: bool = False):
    chroma_manager = ChromaIndexManager(
        collection_name=profile["collection_name"],
        embedding_model=profile["embedding_model"],
        chunk_size=profile["chunk_size"],
    )
    if chroma_manager.get_count() == 0:
        index = chroma_manager.create_index(data_dir)
    else:
        index = chroma_manager.load_index()
    rag_engine = RAGEngine(
        index, top_k=profile["top_k"], llm=profile["llm"], streaming=streaming
    )
    return rag_engine
