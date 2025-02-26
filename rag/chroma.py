import chromadb
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
    StorageContext,
)
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from config.rag_config import (
    CHROMA_DB_DIR,
    CHROMA_COLLECTION_NAME,
    RAW_DATA_DIR,
    EMBEDDING_MODEL,
)

Settings.embed_model = OllamaEmbedding(model_name=EMBEDDING_MODEL)


# this step will create the persistent directory, after that, using os.path.exists(CHROMA_DB_DIR) to judge whether it exists is wrong!
db = chromadb.PersistentClient(path=CHROMA_DB_DIR)
chroma_collection = db.get_or_create_collection(CHROMA_COLLECTION_NAME)
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)


def load_or_create_chroma_index():
    vector_count = chroma_collection.count()

    if vector_count > 0:
        print("found existing ChromaDB data, load index")
        index = VectorStoreIndex.from_vector_store(
            vector_store, storage_context=storage_context
        )
    else:
        print("ChromaDB is empty, reindex...")
        try:
            documents = SimpleDirectoryReader(RAW_DATA_DIR).load_data()
            print(f"read {len(documents)} documents")
        except Exception as e:
            print(
                f"error: failed to read {RAW_DATA_DIR} directory, please check {RAW_DATA_DIR} directory."
            )
            raise e
        index = VectorStoreIndex.from_documents(
            documents, storage_context=storage_context
        )

    return index


def insert_documents(data_dir: str):
    vector_count = chroma_collection.count()

    documents = SimpleDirectoryReader(data_dir).load_data()
    print(f"read {len(documents)} new documents")

    if vector_count > 0:
        print("found existing ChromaDB data, insert documents")
        index = VectorStoreIndex.from_vector_store(
            vector_store, storage_context=storage_context
        )
        for document in tqdm(documents, desc="insert documents"):
            index.insert(document)
    else:
        print("ChromaDB is empty, insert documents")
        index = VectorStoreIndex.from_documents(
            documents, storage_context=storage_context
        )
