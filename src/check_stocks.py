import pandas as pd

df = pd.read_csv("data2/nifty50_ticker.csv")

print("Unique symbols:", df["Symbol"].nunique())

print("\nRows per symbol:")
print(df.groupby("Symbol").size().sort_values())

print("\nIndex values:")
print(df["Index"].value_counts(dropna=False))

print("\nSample companies:")
print(
    df[["Symbol", "Company"]]
    .drop_duplicates()
    .head(20)
    .to_string(index=False)
)