import pandas as pd
import re
from pathlib import Path

DATA_DIR = Path("data2")
REPORT_DIR = DATA_DIR / "validation_report"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = DATA_DIR / "matched_news.csv"

df = pd.read_csv(INPUT_FILE, low_memory=False)

required = ["title", "news", "symbol", "matched_names"]

missing = [col for col in required if col not in df.columns]

if missing:
    raise ValueError(f"Missing required columns: {missing}")

# Normalize fields
for col in required:
    df[col] = df[col].fillna("").astype(str)

df["title_lower"] = df["title"].str.lower()
df["news_lower"] = df["news"].str.lower()
df["matched_names_lower"] = df["matched_names"].str.lower()

flags = []

# 1. Flag empty titles or article text
flags.append(
    df["title"].str.strip().eq("") |
    df["news"].str.strip().eq("")
)

# 2. Flag matches where the matched company name is not in the title.
# This is a heuristic: aliases and abbreviations may not be recorded
# exactly in matched_names, so these rows need closer automated checks.
title_match = pd.Series(False, index=df.index)

for _, row in df.iterrows():
    names = [
        name.strip()
        for name in re.split(r"[,;|]", row["matched_names"])
        if name.strip()
    ]

    for name in names:
        if len(name) >= 3 and name.lower() in row["title_lower"]:
            title_match.loc[row.name] = True
            break

flags.append(~title_match)

# 3. Flag titles that match multiple stocks.
# This is not necessarily an error; roundup articles can mention
# multiple companies legitimately.
if "url" in df.columns:
    stock_counts = df.groupby("url")["symbol"].transform("nunique")
    flags.append(stock_counts > 1)

# 4. Flag generic-looking titles that may be market roundups.
roundup_pattern = (
    r"\b(?:top stocks|stocks to watch|stock picks|"
    r"market update|share market|top gainers|top losers|"
    r"analyst calls|shares in focus|stocks in focus)\b"
)

flags.append(
    df["title_lower"].str.contains(roundup_pattern, regex=True, na=False)
)

# Combine flags
df["empty_content_flag"] = flags[0]
df["company_not_found_in_title_flag"] = flags[1]

if "url" in df.columns:
    df["multi_stock_article_flag"] = flags[2]
else:
    df["multi_stock_article_flag"] = False

df["possible_roundup_flag"] = flags[-1]

flag_columns = [
    "empty_content_flag",
    "company_not_found_in_title_flag",
    "multi_stock_article_flag",
    "possible_roundup_flag"
]

df["flag_count"] = df[flag_columns].sum(axis=1)

# Keep all records, including flagged ones
output_columns = [
    col for col in [
        "date", "title", "news", "url", "source",
        "symbol", "matched_names"
    ]
    if col in df.columns
]

output_columns += flag_columns + ["flag_count"]

flagged = df.loc[df["flag_count"] > 0, output_columns].copy()

# Save all flags for automatic follow-up
flagged.to_csv(
    REPORT_DIR / "ticker_match_flags.csv",
    index=False
)

summary = {
    "total_matched_rows": int(len(df)),
    "rows_with_any_flag": int((df["flag_count"] > 0).sum()),
    "flag_counts": {
        col: int(df[col].sum())
        for col in flag_columns
    },
    "unique_symbols": int(df["symbol"].nunique()),
    "output_file": "ticker_match_flags.csv"
}

pd.Series(summary).to_json(
    REPORT_DIR / "ticker_match_summary.json",
    indent=4
)

print("\nTICKER MATCH QUALITY CHECK")
print("=" * 45)
print("Total matched rows:", len(df))

for col in flag_columns:
    print(f"{col}: {int(df[col].sum())}")

print("Rows with at least one flag:", len(flagged))
print("Unique symbols:", df["symbol"].nunique())

print("\nReports saved in:", REPORT_DIR)
print("- ticker_match_flags.csv")
print("- ticker_match_summary.json")