# 通过 python -m evaluate.generate_question_answer 运行此文件

from llama_index.core import SimpleDirectoryReader
from llama_index.llms.dashscope import DashScope, DashScopeGenerationModels
from llama_index.embeddings.dashscope import (
    DashScopeEmbedding,
    DashScopeTextEmbeddingModels,
)
from ragas.testset import TestsetGenerator
from dotenv import load_dotenv
import os
from config.rag_config import RAW_DATA_DIR, EVALUATE_DATA_DIR


load_dotenv(".env")
dashscope_api_key = os.getenv("DASHSCOPE_API_KEY")

llm = DashScope(
    model_name=DashScopeGenerationModels.QWEN_PLUS, api_key=dashscope_api_key
)
embeddings = DashScopeEmbedding(
    model_name=DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V3, api_key=dashscope_api_key
)

generator = TestsetGenerator.from_llama_index(
    llm=llm,
    embedding_model=embeddings,
)

documents = SimpleDirectoryReader(RAW_DATA_DIR).load_data()

testset = generator.generate_with_llamaindex_docs(
    documents,
    testset_size=3,
)

path = os.path.join(EVALUATE_DATA_DIR, "testset.csv")
testset.to_pandas().to_csv(path, index=False)
print(f"generated testset saved to {path}")
