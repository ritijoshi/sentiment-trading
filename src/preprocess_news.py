import pandas as pd
from pathlib import Path

DATA_DIR = Path("data2")

news_files = {
    "Business Standard": "businessstandard_raw.csv",
    "Economic Times": "economictimes_raw.csv",
    "Financial Express": "financialexpress_raw.csv",
    "Moneycontrol": "moneycontrol_raw.csv",
}

all_news = []

for source, filename in news_files.items():
    file_path = DATA_DIR / filename
    df = pd.read_csv(file_path)

    # Standardize column names
    df.columns = df.columns.str.strip().str.lower()

    # Parse dates
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Record the source
    df["source"] = source

    all_news.append(df)

# Combine all sources
news = pd.concat(all_news, ignore_index=True)

print("Total rows before cleaning:", len(news))

# Remove records without essential information
news = news.dropna(subset=["date", "title", "news"])

# Clean whitespace
for col in ["title", "news", "url"]:
    news[col] = news[col].fillna("").astype(str).str.strip()

# Remove empty titles and articles
news = news[
    (news["title"] != "") &
    (news["news"] != "")
].copy()

# Remove duplicate URLs where available
has_url = news["url"] != ""

with_url = news[has_url].drop_duplicates(subset=["url"])
without_url = news[~has_url]

news = pd.concat([with_url, without_url], ignore_index=True)

# Remove exact duplicate titles and article text
news = news.drop_duplicates(subset=["title", "news"])

# Sort chronologically
news = news.sort_values("date").reset_index(drop=True)

# Save the cleaned dataset
output_path = DATA_DIR / "combined_news.csv"
news.to_csv(output_path, index=False)

print("\nTotal rows after cleaning:", len(news))
print("Date range:", news["date"].min(), "to", news["date"].max())
print("\nArticles by source:")
print(news["source"].value_counts())
print("\nSaved to:", output_path)