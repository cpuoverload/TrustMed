from llama_index.core import SimpleDirectoryReader
from ragas.testset import TestsetGenerator
import os
from rag.config import EVALUATION_DATA_DIR
from utils.model_provider import qwen_llm, qwen_embedding


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

    generator = TestsetGenerator.from_llama_index(
        llm=qwen_llm(),
        embedding_model=qwen_embedding(),
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
        testset_size=100,
        output_path=output_path,
    )
