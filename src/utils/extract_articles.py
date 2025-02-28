import pandas as pd
from tqdm import tqdm


def extract_articles(raw_data, chunksize=10000):
    """Extract articles from Hugging Face downloaded csv file"""

    # chunk_size is the size of each block to prevent loading too much data into memory at once (not used since we only have 10k records)

    df = pd.read_csv(raw_data, header=0, quoting=1, chunksize=chunksize)

    for chunk_id, chunk in enumerate(df):
        # Get content from first column
        chunk_articles = chunk.iloc[:, 0]

        for i, article in tqdm(
            enumerate(chunk_articles, start=chunk_id * chunksize + 1),
            total=len(chunk_articles),
            desc=f"Saving Articles (Chunk {chunk_id+1})",
        ):
            # Skip empty or whitespace-only articles
            if pd.isna(article) or str(article).strip() == "":
                continue

            file_name = f"articles/article_{i}.txt"

            with open(file_name, "w", encoding="utf-8") as file:
                file.write(article)

    print("files saved!")
