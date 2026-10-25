import pandas as pd

df = pd.read_csv("data2/news_with_returns.csv", nrows=5)

print("Columns in news_with_returns.csv:")
for col in df.columns:
    print(repr(col))

print("\nSample data:")
print(df.head(3).to_string())