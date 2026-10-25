import pandas as pd

df = pd.read_csv("data2/matched_news.csv")

# Inspect matches for stocks with very few articles
stocks_to_check = [
    "COALINDIA.NS",
    "HINDUNILVR.NS",
    "BAJFINANCE.NS",
    "MAXHEALTH.NS",
    "TRENT.NS",
]

for symbol in stocks_to_check:
    subset = df[df["symbol"] == symbol]

    print(f"\n{'=' * 60}")
    print(f"{symbol}: {len(subset)} matches")

    for title in subset["title"].head(10):
        print("-", title)

# Inspect articles matched to multiple stocks
counts = df.groupby("url")["symbol"].nunique()
multiple = counts[counts > 1].index

print("\nArticles matched to multiple stocks:", len(multiple))

print("\nSample multi-stock matches:")
print(
    df[df["url"].isin(multiple)][
        ["title", "symbol", "matched_names"]
    ].head(20).to_string(index=False)
)