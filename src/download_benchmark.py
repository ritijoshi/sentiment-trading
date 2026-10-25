import yfinance as yf
from pathlib import Path

output_path = Path("data2/nifty50_index.csv")

print("Downloading NIFTY 50 data...")

data = yf.download(
    "^NSEI",
    start="2020-01-01",
    end="2026-04-01",
    auto_adjust=False,
    progress=False
)

if data.empty:
    raise ValueError("No benchmark data downloaded.")

# Flatten multi-level column headers
if isinstance(data.columns, __import__("pandas").MultiIndex):
    data.columns = data.columns.get_level_values(0)

# Convert index into a regular Date column
data = data.reset_index()

# Keep only the columns we need
data = data[["Date", "Close"]]

# Save a clean CSV
data.to_csv(output_path, index=False)

print("\nSaved clean benchmark data!")
print("Rows:", len(data))
print("\nFirst 5 rows:")
print(data.head().to_string(index=False))
print("\nLast 5 rows:")
print(data.tail().to_string(index=False))