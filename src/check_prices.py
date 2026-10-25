import pandas as pd

df = pd.read_csv("data2/nifty50_ticker.csv")

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

# Check for missing prices
print("Missing Close prices:", df["Close"].isna().sum())
print("Missing Adj Close prices:", df["Adj Close"].isna().sum())

# Check for duplicate stock-date combinations
duplicates = df.duplicated(subset=["Date", "Symbol"]).sum()
print("Duplicate date-symbol rows:", duplicates)

# Check the trading dates available
print("Unique trading dates:", df["Date"].nunique())

# Show the first and last few dates
print("Date range:", df["Date"].min(), "to", df["Date"].max())

# Check whether the dataset contains an index series
print("\nSymbols containing INDEX or NIFTY:")
print(df[df["Symbol"].str.contains("NIFTY|INDEX", case=False, na=False)]
      [["Symbol", "Company"]]
      .drop_duplicates())