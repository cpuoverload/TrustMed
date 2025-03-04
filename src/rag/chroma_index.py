from typing import Tuple, List
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
    StorageContext,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.schema import BaseNode
from llama_index.core.base.embeddings.base import BaseEmbedding
import chromadb
from config import CHROMA_DB_DIR


class ChromaIndexManager:

    def __init__(
        self,
        collection_name: str,
        chunk_size: int,
        chunk_overlap: int,
        embedding_model: BaseEmbedding,
    ):
        Settings.embed_model = embedding_model

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.db = chromadb.PersistentClient(path=CHROMA_DB_DIR)
        self.collection = self.db.get_or_create_collection(collection_name)
        self.vector_store = ChromaVectorStore(chroma_collection=self.collection)
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store
        )

    def _create_nodes(self, data_dir: str) -> List[BaseNode]:
        # if there is data, raise an error
        if self.collection.count() > 0:
            raise ValueError("ChromaDB is not empty, cannot create index.")

        try:
            documents = SimpleDirectoryReader(data_dir).load_data()
            print(
                f"No data in ChromaDB, re-constructing index, read {len(documents)} documents"
            )
        except Exception as e:
            print(
                f"error: failed to read {data_dir} directory, please check {data_dir} directory."
            )
            raise e

        parser = SentenceSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
        nodes = parser.get_nodes_from_documents(documents, show_progress=True)
        return nodes

    def create_index(self, data_dir: str) -> VectorStoreIndex:
        nodes = self._create_nodes(data_dir)
        return VectorStoreIndex(nodes, storage_context=self.storage_context)

    def create_index_and_docstore(
        self, data_dir: str
    ) -> Tuple[VectorStoreIndex, SimpleDocumentStore]:
        nodes = self._create_nodes(data_dir)
        docstore = SimpleDocumentStore()
        docstore.add_documents(nodes)
        return VectorStoreIndex(nodes, storage_context=self.storage_context), docstore

    def load_index(self) -> VectorStoreIndex:
        # if there is no data, raise an error
        if self.collection.count() == 0:
            raise ValueError("ChromaDB is empty, cannot load index.")

        print("Loading index from ChromaDB")

        return VectorStoreIndex.from_vector_store(
            self.vector_store, storage_context=self.storage_context
        )

    def get_count(self) -> int:
        return self.collection.count()
