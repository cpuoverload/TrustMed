import pandas as pd
import ast
import os
from config.rag_config import EVALUATE_DATA_DIR
from rag.chroma import load_or_create_chroma_index
from api.services.rag_service import RAGService


df = pd.read_csv(os.path.join(EVALUATE_DATA_DIR, "testset.csv"))
df["reference_contexts"] = df["reference_contexts"].apply(
    ast.literal_eval
)  # str 转换为 list
samples = df.to_dict(orient="records")

dataset = []
index = load_or_create_chroma_index()
rag_service = RAGService(index)

for sample in samples:
    response = rag_service.query(sample["user_input"])
    dataset.append(
        {
            "user_input": sample["user_input"],
            "retrieved_contexts": [n.node.text for n in response.source_nodes],
            "reference_contexts": sample["reference_contexts"],
            "response": response.response,
            "reference": sample["reference"],
        }
    )
df = pd.DataFrame(dataset)
saved_path = os.path.join(EVALUATE_DATA_DIR, "dataset.csv")
df.to_csv(saved_path, index=False, encoding="utf-8")
print(f"constructed dataset saved to {saved_path}")
