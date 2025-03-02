import pandas as pd
import ast
import os
from tqdm import tqdm
from rag.config import EVALUATION_DATA_DIR, PROFILES
from rag.rag_engine import create_rag_engine
from rag.types import ProfileType


def create_evaluation_dataset(
    profile: ProfileType, articles_dir: str, testset_path: str, saved_path: str
):
    # load testset
    df = pd.read_csv(testset_path)
    df["reference_contexts"] = df["reference_contexts"].apply(ast.literal_eval)
    samples = df.to_dict(orient="records")

    # initialize rag engine
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
    os.makedirs(os.path.dirname(saved_path), exist_ok=True)
    df.to_csv(saved_path, index=False, encoding="utf-8")
    print(f"constructed dataset saved to {saved_path}")


if __name__ == "__main__":
    profile = PROFILES[2]
    articles_dir = os.path.join(EVALUATION_DATA_DIR, "articles")
    testset_path = os.path.join(EVALUATION_DATA_DIR, "testset.csv")
    saved_path = os.path.join(
        EVALUATION_DATA_DIR, profile["profile_name"], "evaluation_dataset.csv"
    )
    create_evaluation_dataset(profile, articles_dir, testset_path, saved_path)
