import pandas as pd
import argparse
import os
from config.data_dir import EVALUATION_DATA_DIR


def get_dir_name():
    """Get directory name from command line arguments"""
    parser = argparse.ArgumentParser(description="input directory name")
    parser.add_argument(
        "--dir_name", type=str, required=True, help="Directory name to use"
    )
    args = parser.parse_args()
    return args.dir_name


def calc_and_save_stats(df: pd.DataFrame, columns: list[str], output_path: str) -> None:
    """Calculate statistics for multiple dataframe columns and save to txt file

    Args:
        df: Input dataframe
        columns: List of column names to analyze
        output_path: Path to save the statistics file
    """
    # Create statistics text
    stats_text = ""

    for column in columns:
        mean_value = df[column].mean()
        quantiles = df[column].quantile([0.25, 0.5, 0.75])
        std_dev = df[column].std()

        stats_text += f"""{column} Statistics:
{'-'*30}
Average: {mean_value:.4f}
Q1 (25th percentile): {quantiles[0.25]:.4f}
Q2 (Median): {quantiles[0.50]:.4f} 
Q3 (75th percentile): {quantiles[0.75]:.4f}
Standard Deviation: {std_dev:.4f}

"""

    # Save to txt file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(stats_text)
    print(f"Statistics saved to {output_path}")


if __name__ == "__main__":
    dir_name = get_dir_name()
    evaluation_result_path = os.path.join(
        EVALUATION_DATA_DIR, dir_name, "evaluation_result.csv"
    )
    df = pd.read_csv(evaluation_result_path)

    # Calculate stats for all metric columns
    metric_columns = [
        "context_precision",
        "context_recall",
        "faithfulness",
        "answer_relevancy",
    ]
    output_path = os.path.join(EVALUATION_DATA_DIR, dir_name, "stats.txt")
    calc_and_save_stats(df, metric_columns, output_path)
