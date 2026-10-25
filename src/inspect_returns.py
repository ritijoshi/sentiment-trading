import pandas as pd

df = pd.read_csv("data2/news_with_returns.csv")

columns = [
    "date",
    "title",
    "symbol",
    "entry_date",
    "exit_date",
    "entry_price",
    "exit_price",
    "stock_return_3d",
    "market_return_3d",
    "excess_return_3d",
    "label",
]

print(df[columns].head(10).to_string(index=False))

print("\nDate validation:")
df["date"] = pd.to_datetime(df["date"])
df["entry_date"] = pd.to_datetime(df["entry_date"])
df["exit_date"] = pd.to_datetime(df["exit_date"])

print("Entry on/after news date:",
      (df["entry_date"] >= df["date"]).all())

print("Exit after entry:",
      (df["exit_date"] > df["entry_date"]).all())

print("\nLabels by class:")
print(df["label"].value_counts(normalize=True).mul(100).round(2))