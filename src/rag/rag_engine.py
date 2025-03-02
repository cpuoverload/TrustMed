import os
from llama_index.core import VectorStoreIndex
from llama_index.core.llms.utils import LLMType
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever, QueryFusionRetriever
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.retrievers.bm25 import BM25Retriever
from rag.chroma_index import ChromaIndexManager
from rag.types import ProfileType
from utils.llm_api import get_llm


class RAGEngine:
    def __init__(
        self,
        retriever: BaseRetriever,
        llm: LLMType,
        streaming: bool = False,
    ):
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
    collection_name = profile["collection_name"]
    embedding_model = profile["embedding_model"]
    chunk_size = profile["chunk_size"]
    top_k = profile["top_k"]
    llm = profile["llm"]
    hybrid_search = profile["hybrid_search"]
    docstore_path = profile["profile_name"] + "_docstore.json"

    # get chroma index
    chroma_manager = ChromaIndexManager(
        collection_name=collection_name,
        embedding_model=embedding_model,
        chunk_size=chunk_size,
    )
    docstore = None
    if chroma_manager.get_count() == 0:
        if hybrid_search:
            index, docstore = chroma_manager.create_index_and_docstore(data_dir)
            docstore.persist(docstore_path)
        else:
            index = chroma_manager.create_index(data_dir)
    else:
        index = chroma_manager.load_index()

    # get retriever
    if hybrid_search:
        if docstore is None:
            docstore = SimpleDocumentStore.from_persist_path(docstore_path)
        bm25_retriever = BM25Retriever.from_defaults(
            docstore=docstore,
            similarity_top_k=top_k,
        )
        vector_retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=top_k,
        )
        retriever = QueryFusionRetriever(
            [vector_retriever, bm25_retriever],
            similarity_top_k=top_k,
            num_queries=1,  # set this to 1 to disable query generation
            llm=get_llm(),
            mode="reciprocal_rerank",
            verbose=True,
            # query_gen_prompt="...",  # we could override the query generation prompt here
        )
    else:
        retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=top_k,
        )

    # get rag engine
    rag_engine = RAGEngine(
        retriever=retriever,
        llm=llm,
        streaming=streaming,
    )
    return rag_engine
