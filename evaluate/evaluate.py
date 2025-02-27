# 通过 python -m 运行，否则 ModuleNotFoundError

import os
import ast
import pandas as pd
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
from ragas import EvaluationDataset, evaluate
from ragas.llms import LlamaIndexLLMWrapper
from ragas.embeddings import LlamaIndexEmbeddingsWrapper
from config.rag_config import EVALUATE_DATA_DIR

metrics = [
    ContextPrecision(),
    ContextRecall(),
    Faithfulness(),
    AnswerRelevancy(),
]

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

df = pd.read_csv(os.path.join(EVALUATE_DATA_DIR, "dataset.csv"))
df["retrieved_contexts"] = df["retrieved_contexts"].apply(ast.literal_eval)
df["reference_contexts"] = df["reference_contexts"].apply(ast.literal_eval)
dataset = df.to_dict(orient="records")

evaluation_dataset = EvaluationDataset.from_list(dataset)

result = evaluate(
    dataset=evaluation_dataset,
    metrics=metrics,
    llm=LlamaIndexLLMWrapper(llm),
    embeddings=LlamaIndexEmbeddingsWrapper(embeddings),  # 部分 metrics 需要 embeddings
)

path = os.path.join(EVALUATE_DATA_DIR, "evaluate_result.csv")
result.to_pandas().to_csv(path, index=False)
print(f"evaluate result saved to {path}")
