import pandas as pd

df = pd.read_csv("data2/news_with_returns.csv")

print("Total rows:", len(df))
print("Duplicate URLs:", df.duplicated(subset=["url"]).sum())

print(
    "Duplicate article-stock pairs:",
    df.duplicated(subset=["url", "symbol"]).sum()
)

print(
    "Unique article URLs:",
    df["url"].nunique()
)

print(
    "\nArticles per stock:",
    df["symbol"].nunique()
)

print(
    "\nArticles per date:",
    df.groupby("date").size().describe()
)

print("\nSample duplicate URLs:")
duplicates = df[df.duplicated(subset=["url", "symbol"], keep=False)]
print(
    duplicates[["date", "title", "symbol", "url"]]
    .head(10)
    .to_string(index=False)
)