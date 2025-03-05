import os
from llama_index.core import VectorStoreIndex
from llama_index.core.llms.utils import LLMType
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever, QueryFusionRetriever
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.docstore.types import BaseDocumentStore
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from rag.chroma_index import ChromaIndexManager
from rag.types import ProfileType
from config.model_provider import qwen_llm


class RAGEngine:
    def __init__(
        self,
        retriever: BaseRetriever,
        reranker: BaseNodePostprocessor,
        llm: LLMType,
        streaming: bool = False,
    ):
        node_postprocessors = [reranker] if reranker else []
        response_synthesizer = get_response_synthesizer(
            llm=llm,
            streaming=streaming,
        )
        self.query_engine = RetrieverQueryEngine(
            retriever=retriever,
            node_postprocessors=node_postprocessors,
            response_synthesizer=response_synthesizer,
        )

    def query(self, question: str):
        return self.query_engine.query(question)


def _get_index_and_docstore(profile: ProfileType, data_dir: str):
    collection_name = profile["collection_name"]
    chunk_size = profile["chunk_size"]
    chunk_overlap = profile["chunk_overlap"]
    docstore_path = profile["profile_name"] + "_docstore.json"
    embedding_model = profile["embedding_model"]
    hybrid_search = profile.get("hybrid_search", False)

    chroma_manager = ChromaIndexManager(
        collection_name=collection_name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embedding_model=embedding_model(),
    )

    index = None
    docstore = None

    if chroma_manager.get_count() == 0:
        if hybrid_search:
            index, docstore = chroma_manager.create_index_and_docstore(data_dir)
            docstore.persist(docstore_path)
        else:
            index = chroma_manager.create_index(data_dir)
    else:
        if hybrid_search:
            docstore = SimpleDocumentStore.from_persist_path(docstore_path)
        index = chroma_manager.load_index()

    return index, docstore


def _create_retriever(
    profile: ProfileType, index: VectorStoreIndex, docstore: BaseDocumentStore
):
    hybrid_search = profile.get("hybrid_search", False)
    top_k = profile["top_k"]
    query_rewrite_num = profile.get("query_rewrite_num", 0)

    vector_retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=top_k,
    )

    retrievers = [vector_retriever]

    if hybrid_search:
        bm25_retriever = BM25Retriever.from_defaults(
            docstore=docstore,
            similarity_top_k=top_k,
        )
        retrievers.append(bm25_retriever)

    retriever = QueryFusionRetriever(
        retrievers,
        similarity_top_k=top_k,
        num_queries=1 + query_rewrite_num,  # set this to 1 to disable query generation
        llm=qwen_llm(),
        mode="reciprocal_rerank",
        # verbose=True,  # print generated queries
        # query_gen_prompt="...",  # we could override the query generation prompt here
    )

    # test
    # retrieved_result = retriever.retrieve(
    #     "How does anxiety in Parkinson's disease patients impact their performance on the mini-mental state exam and other cognitive tests?"
    # )
    # for i, result in enumerate(retrieved_result):
    #     print(f"Retrieved Result {i+1}:")
    #     print(f"  Text: {result.text}")
    #     print("-" * 50)

    return retriever


def create_rag_engine(profile: ProfileType, data_dir: str, streaming: bool = False):
    reranker = profile.get("reranker", None)
    reranker_top_n = profile.get("reranker_top_n", 3)
    reranker = reranker(reranker_top_n) if reranker else None
    
    llm = profile["llm"]

    index, docstore = _get_index_and_docstore(profile, data_dir)

    retriever = _create_retriever(profile, index, docstore)

    return RAGEngine(
        retriever=retriever,
        reranker=reranker,
        llm=llm(),
        streaming=streaming,
    )
