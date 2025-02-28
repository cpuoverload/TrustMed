from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
    StorageContext,
)
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.base.embeddings.base import BaseEmbedding
import chromadb
from rag.config import CHROMA_DB_DIR


class ChromaIndexManager:

    def __init__(
        self,
        collection_name: str,
        embedding_model: BaseEmbedding,
        # todo: 加 chunk size
    ):
        Settings.embed_model = embedding_model

        self.db = chromadb.PersistentClient(path=CHROMA_DB_DIR)
        self.collection = self.db.get_or_create_collection(collection_name)

        self.vector_store = ChromaVectorStore(chroma_collection=self.collection)
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store
        )

    def create_index(self, data_dir: str) -> VectorStoreIndex:
        # if there is data, raise an error
        if self.collection.count() > 0:
            raise ValueError("ChromaDB is not empty, cannot create index.")

        try:
            documents = SimpleDirectoryReader(data_dir).load_data()
            print(f"read {len(documents)} documents")
        except Exception as e:
            print(
                f"error: failed to read {data_dir} directory, please check {data_dir} directory."
            )
            raise e

        return VectorStoreIndex.from_documents(
            documents, storage_context=self.storage_context
        )

    def load_index(self) -> VectorStoreIndex:
        # if there is no data, raise an error
        if self.collection.count() == 0:
            raise ValueError("ChromaDB is empty, cannot load index.")

        return VectorStoreIndex.from_vector_store(
            self.vector_store, storage_context=self.storage_context
        )

    def get_count(self) -> int:
        return self.collection.count()
