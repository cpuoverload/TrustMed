import pandas as pd
import ast
import os
from tqdm import tqdm
from rag.config import EVALUATION_DATA_DIR, PROFILES
from rag.rag_engine import create_rag_engine
from rag.types import ProfileType


def create_evaluation_dataset(profile: ProfileType):
    # load testset
    testset_path = os.path.join(EVALUATION_DATA_DIR, "testset.csv")
    df = pd.read_csv(testset_path)
    df["reference_contexts"] = df["reference_contexts"].apply(ast.literal_eval)
    samples = df.to_dict(orient="records")

    # initialize rag engine
    articles_dir = os.path.join(EVALUATION_DATA_DIR, "articles")
    rag_engine = create_rag_engine(profile, articles_dir)

    # generate dataset
    dataset = []
    for sample in tqdm(samples, desc="Generating dataset"):
        response = rag_engine.query(sample["user_input"])
        dataset.append(
            {
                "user_input": sample["user_input"],
                "retrieved_contexts": [n.node.text for n in response.source_nodes],
                "reference_contexts": sample[
                    "reference_contexts"
                ],  # 经测试，加不加这个字段不会影响 evaluate 消耗 token 数，因为 metric 没用到
                "response": response.response,
                "reference": sample["reference"],
            }
        )

    # Save dataset
    df = pd.DataFrame(dataset)
    saved_path = os.path.join(EVALUATION_DATA_DIR, "evaluation_dataset.csv")
    df.to_csv(saved_path, index=False, encoding="utf-8")
    print(f"constructed dataset saved to {saved_path}")


if __name__ == "__main__":
    create_evaluation_dataset(PROFILES[1])
