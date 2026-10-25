import pandas as pd
from pathlib import Path

DATA_DIR = Path("data2")
REPORT_DIR = DATA_DIR / "validation_report"

prices = pd.read_csv(DATA_DIR / "nifty50_ticker.csv")
matches = pd.read_csv(DATA_DIR / "matched_news.csv")
returns = pd.read_csv(DATA_DIR / "news_with_returns.csv")

symbols_to_check = [
    "ETERNAL.NS",
    "ITC.NS",
    "TMPV.NS",
    "HEROMOTOCO.NS"
]

results = []

for symbol in symbols_to_check:
    price_rows = prices[prices["Symbol"] == symbol]
    match_rows = matches[matches["symbol"] == symbol]
    return_rows = returns[returns["symbol"] == symbol]

    results.append({
        "symbol": symbol,
        "price_rows": len(price_rows),
        "price_start": (
            price_rows["Date"].min()
            if len(price_rows) else None
        ),
        "price_end": (
            price_rows["Date"].max()
            if len(price_rows) else None
        ),
        "matched_news_rows": len(match_rows),
        "return_labeled_rows": len(return_rows)
    })

result_df = pd.DataFrame(results)

print("\nSYMBOL INVESTIGATION")
print("=" * 75)
print(result_df.to_string(index=False))

output_path = REPORT_DIR / "symbol_investigation.csv"
result_df.to_csv(output_path, index=False)

print(f"\nReport saved to: {output_path}")