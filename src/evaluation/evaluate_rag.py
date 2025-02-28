import os
import ast
import pandas as pd
from typing import Optional, Sequence
from dotenv import load_dotenv
from llama_index.llms.dashscope import DashScope, DashScopeGenerationModels
from llama_index.embeddings.dashscope import (
    DashScopeEmbedding,
    DashScopeTextEmbeddingModels,
)
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)
from ragas.metrics.base import Metric
from ragas import EvaluationDataset, evaluate
from ragas.llms import LlamaIndexLLMWrapper
from ragas.embeddings import LlamaIndexEmbeddingsWrapper


def evaluate_rag(
    dataset_path: str,
    result_path: str,
    metrics: Optional[Sequence[Metric]] = None,
) -> None:
    """
    Evaluate RAG system using Ragas metrics

    Args:
        dataset_path: Path to the evaluation dataset CSV file
        result_path: Path to save evaluation results
        metrics: List of Ragas metrics to use for evaluation. If None, uses default metrics
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

    # Use default metrics if none provided
    if metrics is None:
        metrics = [
            ContextPrecision(),
            ContextRecall(),
            Faithfulness(),
            AnswerRelevancy(),
        ]

    # Load and process dataset
    df = pd.read_csv(dataset_path)
    df["retrieved_contexts"] = df["retrieved_contexts"].apply(ast.literal_eval)
    df["reference_contexts"] = df["reference_contexts"].apply(ast.literal_eval)
    dataset = df.to_dict(orient="records")
    evaluation_dataset = EvaluationDataset.from_list(dataset)

    # Run evaluation
    result = evaluate(
        dataset=evaluation_dataset,
        metrics=metrics,
        llm=LlamaIndexLLMWrapper(llm),
        embeddings=LlamaIndexEmbeddingsWrapper(
            embeddings
        ),  # 部分 metrics 需要 embeddings
    )

    # Save results
    result.to_pandas().to_csv(result_path, index=False)
    print(f"Evaluation result saved to {result_path}")


if __name__ == "__main__":
    from rag.config import EVALUATION_DATA_DIR

    dataset_path = os.path.join(EVALUATION_DATA_DIR, "evaluation_dataset.csv")
    result_path = os.path.join(EVALUATION_DATA_DIR, "evaluation_result.csv")

    evaluate_rag(dataset_path, result_path)
