import pandas as pd
import yfinance as yf
from pathlib import Path

DATA_DIR = Path("data2")

price_file = DATA_DIR / "nifty50_ticker.csv"

prices = pd.read_csv(price_file)

symbol = "HEROMOTOCO.NS"

print("Downloading historical prices for", symbol)

stock = yf.download(
    symbol,
    start="2020-01-01",
    end="2026-04-01",
    auto_adjust=False,
    progress=False
)

if stock.empty:
    raise RuntimeError(f"No price data returned for {symbol}")

# Flatten yfinance's MultiIndex columns if necessary
if isinstance(stock.columns, pd.MultiIndex):
    stock.columns = stock.columns.get_level_values(0)

stock = stock.reset_index()

# Keep the same columns as the existing dataset
stock["Symbol"] = symbol
stock["Company"] = "Hero MotoCorp"
stock["Index"] = "NIFTY 50"

required_columns = [
    "Date", "Symbol", "Company", "Index",
    "Open", "High", "Low", "Close",
    "Adj Close", "Volume", "Dividends", "Stock Splits"
]

for col in required_columns:
    if col not in stock.columns:
        stock[col] = pd.NA

stock = stock[required_columns]

# Normalize dates before checking duplicates
stock["Date"] = pd.to_datetime(stock["Date"], utc=True).dt.strftime(
    "%Y-%m-%d"
)

prices["Date"] = pd.to_datetime(
    prices["Date"], utc=True
).dt.strftime("%Y-%m-%d")

# Append only missing date-symbol pairs
existing = set(zip(prices["Date"], prices["Symbol"]))

stock = stock[
    ~stock.apply(
        lambda row: (row["Date"], row["Symbol"]) in existing,
        axis=1
    )
]

updated = pd.concat([prices, stock], ignore_index=True)

updated = updated.drop_duplicates(
    subset=["Date", "Symbol"],
    keep="first"
)

updated.to_csv(price_file, index=False)

print("\nHero MotoCorp price rows downloaded:", len(stock))
print("Updated price dataset rows:", len(updated))
print("Saved to:", price_file)