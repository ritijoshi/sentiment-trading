import os
import sys
import io
import pandas as pd

# Force UTF-8 output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

news_path = os.path.join(SCRIPT_DIR, "raw_single_stock_news.parquet")
prices_path = os.path.join(SCRIPT_DIR, "daily_prices.csv")
returns_path = os.path.join(SCRIPT_DIR, "daily_returns.csv")

news_df = pd.read_parquet(news_path)
prices_df = pd.read_csv(prices_path, index_col=0)

# 1. Drop low-volume tickers (< 100 items)
valid_tickers = news_df['ticker'].value_counts()[lambda x: x >= 100].index.tolist()
news_df = news_df[news_df['ticker'].isin(valid_tickers)].copy()

# 2. Remove exact duplicate headlines per ticker
news_df = news_df.drop_duplicates(subset=['ticker', 'headline']).reset_index(drop=True)

# 3. Align price matrix to 2019-2023 range and fill missing values
prices_df.index = pd.to_datetime(prices_df.index)
prices_df = prices_df.loc['2019-01-01':'2023-12-31'].ffill().bfill()

# 4. RECOMPUTE RETURNS MATRIX FROM CLEANED PRICES (Fixes missing TSLA returns)
returns_df = prices_df.pct_change().dropna(how='all')

# Overwrite cleaned step 1 artifacts
news_df.to_parquet(news_path, index=False)
prices_df.to_csv(prices_path)
returns_df.to_csv(returns_path)

print("=" * 50)
print(" CLEANED DATASET REPORT ")
print("=" * 50)
print(f"Final Cleaned News Count : {len(news_df):,} items")
print(f"Retained Tickers ({len(valid_tickers)})    : {', '.join(valid_tickers)}")
print(f"Trading Days             : {len(prices_df):,} days")
print(f"Missing Price Values     : {prices_df.isnull().sum().sum()}")
print(f"Missing Return Values    : {returns_df.isnull().sum().sum()}")
print("\n[OK] Step 1 artifacts successfully fixed!")