
import hashlib
import re
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data2"
OUTPUT_DIR = DATA_DIR / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def normalize_url(url):
    """Remove tracking parameters and normalize article URLs."""
    if pd.isna(url) or not str(url).strip():
        return ""

    url = str(url).strip()

    try:
        parts = urlsplit(url)
        return urlunsplit((
            parts.scheme.lower(),
            parts.netloc.lower(),
            parts.path.rstrip("/"),
            "",
            ""
        ))
    except ValueError:
        return url


def make_article_id(row):
    url = row["canonical_url"]

    if url:
        identity = url
    else:
        identity = (
            str(row["title"]).lower().strip()
            + str(row["news"]).lower().strip()[:1000]
        )

    return hashlib.sha256(
        identity.encode("utf-8")
    ).hexdigest()[:16]


def main():
    files = sorted(DATA_DIR.glob("*_raw.csv"))

    if not files:
        print("No raw article CSV files found.")
        return

    frames = []

    for file in files:
        print(f"Reading {file.name}...")

        df = pd.read_csv(file, low_memory=False)
        df.columns = (
            df.columns.str.strip()
            .str.lower()
            .str.replace(" ", "_")
        )

        required = {"date", "news", "url"}

        if not required.issubset(df.columns):
            print("Skipping: required columns missing.")
            continue

        if "title" not in df.columns:
            df["title"] = ""

        df["source"] = file.stem.replace("_raw", "")
        frames.append(df[["date", "title", "news", "url", "source"]])

        print("Rows:", len(df))

    if not frames:
        print("No usable datasets.")
        return

    data = pd.concat(frames, ignore_index=True)

    print("\nTotal raw records:", len(data))

    data["date"] = pd.to_datetime(
        data["date"], errors="coerce", utc=True
    ).dt.tz_localize(None)

    data["news"] = data["news"].fillna("").astype(str).str.strip()
    data["title"] = data["title"].fillna("").astype(str).str.strip()
    data["url"] = data["url"].fillna("").astype(str).str.strip()

    # Remove empty or invalid records.
    data = data[
        data["date"].notna() &
        (data["news"].str.len() > 100)
    ].copy()

    data["canonical_url"] = data["url"].apply(normalize_url)
    data["article_id"] = data.apply(make_article_id, axis=1)

    # Keep one article record for LLM processing.
    data = data.sort_values(["date", "source"])
    articles = data.drop_duplicates(
        subset="article_id", keep="first"
    ).copy()

    # Use the separate return dataset only to recover stock mappings.
    mapping_file = DATA_DIR / "news_with_returns.csv"

    if mapping_file.exists():
        print("\nReading stock mappings...")

        returns = pd.read_csv(
            mapping_file,
            usecols=lambda col: col.lower() in {"url", "symbol"}
        )

        returns.columns = returns.columns.str.lower()

        if {"url", "symbol"}.issubset(returns.columns):
            returns["canonical_url"] = returns["url"].apply(normalize_url)
            returns["symbol"] = returns["symbol"].astype(str).str.strip()

            returns = returns[
                (returns["canonical_url"] != "") &
                (returns["symbol"] != "") &
                (returns["symbol"].str.lower() != "nan")
            ]

            url_to_id = articles[
                ["canonical_url", "article_id"]
            ].drop_duplicates()

            symbol_map = returns.merge(
                url_to_id,
                on="canonical_url",
                how="inner"
            )[["article_id", "symbol"]].drop_duplicates()
        else:
            print("URL or symbol column missing; mapping not created.")
            symbol_map = pd.DataFrame(columns=["article_id", "symbol"])
    else:
        print("news_with_returns.csv not found.")
        symbol_map = pd.DataFrame(columns=["article_id", "symbol"])

    # Keep only article content and metadata.
    articles = articles[
        [
            "article_id",
            "date",
            "title",
            "news",
            "url",
            "source"
        ]
    ].copy()

    articles.to_csv(
        OUTPUT_DIR / "clean_articles.csv",
        index=False
    )

    symbol_map.to_csv(
        OUTPUT_DIR / "article_symbol_map.csv",
        index=False
    )

    print("\nPREPROCESSING COMPLETE")
    print("Raw records:", len(data))
    print("Unique articles:", len(articles))
    print("Article-symbol mappings:", len(symbol_map))
    print("Output folder:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
