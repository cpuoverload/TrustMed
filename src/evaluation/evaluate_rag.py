import os
import ast
import pandas as pd
from typing import Optional, Sequence
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
from rag.config import EVALUATION_DATA_DIR, PROFILES
from utils.llm_api import get_llm, get_embedding


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
        llm=LlamaIndexLLMWrapper(get_llm()),
        embeddings=LlamaIndexEmbeddingsWrapper(
            get_embedding()
        ),  # 部分 metrics 需要 embeddings
    )

    # Save results
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    result.to_pandas().to_csv(result_path, index=False)
    print(f"Evaluation result saved to {result_path}")


if __name__ == "__main__":
    profile = PROFILES[1]
    dataset_path = os.path.join(
        EVALUATION_DATA_DIR, profile["profile_name"], "evaluation_dataset.csv"
    )
    result_path = os.path.join(
        EVALUATION_DATA_DIR, profile["profile_name"], "evaluation_result.csv"
    )
    evaluate_rag(dataset_path, result_path)
