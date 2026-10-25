import pandas as pd
from pathlib import Path

DATA_DIR = Path("data2")
REPORT_DIR = DATA_DIR / "validation_report"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

prices = pd.read_csv(DATA_DIR / "nifty50_ticker.csv")
matches = pd.read_csv(DATA_DIR / "matched_news.csv")

price_symbols = set(prices["Symbol"].dropna().unique())
matched_symbols = set(matches["symbol"].dropna().unique())

missing_symbols = sorted(price_symbols - matched_symbols)
unexpected_symbols = sorted(matched_symbols - price_symbols)

summary = {
    "symbols_in_price_data": len(price_symbols),
    "symbols_in_matched_news": len(matched_symbols),
    "symbols_with_no_news_matches": missing_symbols,
    "symbols_not_in_price_data": unexpected_symbols,
}

print("\nSYMBOL COVERAGE REPORT")
print("=" * 45)
print("Symbols in stock-price data:", len(price_symbols))
print("Symbols in matched news:", len(matched_symbols))
print("Symbols with no news matches:", len(missing_symbols))

if missing_symbols:
    print("\nSymbols without news matches:")
    for symbol in missing_symbols:
        print("-", symbol)

if unexpected_symbols:
    print("\nSymbols absent from price data:")
    for symbol in unexpected_symbols:
        print("-", symbol)

pd.DataFrame({
    "symbol_without_news_matches": pd.Series(missing_symbols),
    "symbol_not_in_price_data": pd.Series(unexpected_symbols),
}).to_csv(
    REPORT_DIR / "symbol_coverage.csv",
    index=False
)

print("\nReport saved to:")
print(REPORT_DIR / "symbol_coverage.csv")