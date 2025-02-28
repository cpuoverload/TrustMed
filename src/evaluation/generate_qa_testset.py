from llama_index.core import SimpleDirectoryReader
from llama_index.llms.dashscope import DashScope, DashScopeGenerationModels
from llama_index.embeddings.dashscope import (
    DashScopeEmbedding,
    DashScopeTextEmbeddingModels,
)
from ragas.testset import TestsetGenerator
from dotenv import load_dotenv
import os
from rag.config import EVALUATION_DATA_DIR


def generate_qa_testset(
    articles_dir: str,
    testset_size: int,
    output_path: str,
) -> None:
    """
    Generate Q&A samples from articles

    Args:
        articles_dir: Directory containing article files
        testset_size: Number of samples to generate
        output_path: Path to save the generated test set CSV
    """
    load_dotenv(".env")
    dashscope_api_key = os.getenv("DASHSCOPE_API_KEY")

    llm = DashScope(
        model_name=DashScopeGenerationModels.QWEN_PLUS,
        api_key=dashscope_api_key,
        max_tokens=8192,  # 默认 256，会导致输出被截断
    )
    embeddings = DashScopeEmbedding(
        model_name=DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V3,
        api_key=dashscope_api_key,
    )

    generator = TestsetGenerator.from_llama_index(
        llm=llm,
        embedding_model=embeddings,
    )

    # Load documents and generate samples
    documents = SimpleDirectoryReader(articles_dir).load_data()
    testset = generator.generate_with_llamaindex_docs(
        documents,
        testset_size=testset_size,
    )

    # Save to CSV
    testset.to_pandas().to_csv(output_path, index=False)
    print(f"Generated testset saved to {output_path}")


if __name__ == "__main__":
    articles_dir = os.path.join(EVALUATION_DATA_DIR, "articles")
    output_path = os.path.join(EVALUATION_DATA_DIR, "testset.csv")
    generate_qa_testset(
        articles_dir=articles_dir,
        testset_size=2,
        output_path=output_path,
    )
