import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
INPUT_FILE = BASE_DIR / "data2" / "processed" / "clean_articles.csv"
OUTPUT_FILE = BASE_DIR / "data2" / "processed" / "article_chunks.csv"

CHUNK_SIZE = 3000
OVERLAP = 400


def split_into_chunks(text):
    text = str(text).strip()
    chunks = []
    start = 0

    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))

        if end < len(text):
            boundary = text.rfind(" ", start, end)
            if boundary > start:
                end = boundary

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = max(end - OVERLAP, start + 1)

    return chunks


def main():
    articles = pd.read_csv(INPUT_FILE)

    rows = []

    for _, article in articles.iterrows():
        chunks = split_into_chunks(article["news"])

        for i, chunk in enumerate(chunks):
            rows.append({
                "article_id": article["article_id"],
                "chunk_id": i + 1,
                "total_chunks": len(chunks),
                "chunk_text": chunk
            })

    chunks_df = pd.DataFrame(rows)
    chunks_df.to_csv(OUTPUT_FILE, index=False)

    print("CHUNKING COMPLETE")
    print("Articles:", len(articles))
    print("Total chunks:", len(chunks_df))
    print("Average chunks per article:", round(
        len(chunks_df) / len(articles), 2
    ))
    print("Output:", OUTPUT_FILE)


if __name__ == "__main__":
    main()