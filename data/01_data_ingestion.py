import os
import sys
import io
import pandas as pd
import yfinance as yf

# Force UTF-8 encoding on Windows console output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Resolve directory paths relative to this script's location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = BASE_DIR if os.path.basename(BASE_DIR) == "data" else os.path.join(BASE_DIR, "data")
CHECKPOINT_PARQUET = os.path.join(DATA_DIR, "checkpoints", "extracted_news_checkpoint.parquet")

os.makedirs(DATA_DIR, exist_ok=True)

# ---------------------------------------------------------
# 1. DEFINE STOCK UNIVERSE (S&P 100 Sample Sub-List)
# ---------------------------------------------------------
TARGET_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B", 
    "JPM", "JNJ", "V", "PG", "UNH", "HD", "MA", "BAC", "XOM", "DIS"
]

BENCHMARK_TICKER = "^GSPC"
CHECKPOINT_PARQUET = "data/checkpoints/extracted_news_checkpoint.parquet"

print(f"[+] Defined target universe of {len(TARGET_TICKERS)} liquid tickers.")

# ---------------------------------------------------------
# 2. LOAD & CLEAN CHECKPOINTED NEWS DATA
# ---------------------------------------------------------
if not os.path.exists(CHECKPOINT_PARQUET):
    raise FileNotFoundError(f"Checkpoint file not found at {CHECKPOINT_PARQUET}")

print(f"[+] Loading saved news dataset from {CHECKPOINT_PARQUET}...")
news_df = pd.read_parquet(CHECKPOINT_PARQUET)

print(f"[+] Total raw saved items loaded: {len(news_df):,}")

# Filter out nulls and clean timestamps
news_df = news_df.dropna(subset=["published_at", "headline", "ticker"])
news_df["published_at"] = pd.to_datetime(news_df["published_at"], errors="coerce", utc=True)
news_df = news_df.dropna(subset=["published_at"])

print(f"[+] Total cleaned single-stock news items: {len(news_df):,}")

# Determine date boundaries for stock price retrieval
min_date = (news_df["published_at"].min() - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
max_date = (news_df["published_at"].max() + pd.Timedelta(days=7)).strftime("%Y-%m-%d")

# ---------------------------------------------------------
# 3. FETCH MARKET PRICE DATA (YFINANCE)
# ---------------------------------------------------------
print(f"[+] Pulling stock price data from {min_date} to {max_date} via yfinance...")

all_tickers_to_pull = TARGET_TICKERS + [BENCHMARK_TICKER]

market_raw = yf.download(
    tickers=all_tickers_to_pull,
    start=min_date,
    end=max_date,
    interval="1d",
    auto_adjust=True
)

prices_df = pd.DataFrame()

# Extract Close prices
for ticker in TARGET_TICKERS:
    if ("Close", ticker) in market_raw.columns:
        prices_df[ticker] = market_raw[("Close", ticker)]
    elif ticker in market_raw.columns:
        prices_df[ticker] = market_raw[ticker]["Close"]

# Extract Benchmark price
if ("Close", BENCHMARK_TICKER) in market_raw.columns:
    prices_df["MARKET_BENCHMARK"] = market_raw[("Close", BENCHMARK_TICKER)]
elif BENCHMARK_TICKER in market_raw.columns:
    prices_df["MARKET_BENCHMARK"] = market_raw[BENCHMARK_TICKER]["Close"]

# Calculate Daily Returns
returns_df = prices_df.pct_change().dropna(how="all")

# ---------------------------------------------------------
# 4. SAVE FINAL STEP 1 ARTIFACTS
# ---------------------------------------------------------
news_df.to_parquet("data/raw_single_stock_news.parquet", index=False)
prices_df.to_csv("data/daily_prices.csv")
returns_df.to_csv("data/daily_returns.csv")

print("\n[✓] Step 1 Complete:")
print(f"    - Cleaned News saved to 'data/raw_single_stock_news.parquet' ({len(news_df):,} rows)")
print(f"    - Daily Prices saved to 'data/daily_prices.csv' ({len(prices_df)} trading days)")
print(f"    - Daily Returns saved to 'data/daily_returns.csv' ({len(returns_df)} trading days)")