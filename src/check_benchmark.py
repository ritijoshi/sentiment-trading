import pandas as pd

df = pd.read_csv("data2/nifty50_index.csv")

print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nFirst 5 rows:")
print(df.head().to_string())