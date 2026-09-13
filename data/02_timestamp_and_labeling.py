import os
import sys
import io
import pandas as pd
import numpy as np

# Force UTF-8 console output for Windows terminal stability
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------
# 1. LOAD CLEANED STEP 1 ARTIFACTS
# ---------------------------------------------------------
news_path = os.path.join(SCRIPT_DIR, "raw_single_stock_news.parquet")
returns_path = os.path.join(SCRIPT_DIR, "daily_returns.csv")

if not os.path.exists(news_path) or not os.path.exists(returns_path):
    raise FileNotFoundError("Step 1 clean artifacts not found. Please verify files in data/ directory.")

news_df = pd.read_parquet(news_path)
returns_df = pd.read_csv(returns_path, index_col=0)
returns_df.index = pd.to_datetime(returns_df.index)

# Standardize ticker casing and strip spaces
news_df['ticker'] = news_df['ticker'].astype(str).str.strip().str.upper()
returns_df.columns = returns_df.columns.astype(str).str.strip().str.upper()

print(f"[+] Total input raw cleaned news items: {len(news_df):,}")

# ---------------------------------------------------------
# 2. TRIM DATASET TO TARGET ERA (2019 – 2023)
# ---------------------------------------------------------
print("[+] Filtering dataset to 2019-2023 range...")

news_df['published_at'] = pd.to_datetime(news_df['published_at'], utc=True)
news_df['published_et'] = news_df['published_at'].dt.tz_convert('US/Eastern')

news_df = news_df[
    (news_df['published_et'].dt.year >= 2019) & 
    (news_df['published_et'].dt.year <= 2023)
].copy()

print(f"[+] Retained items after 2019-2023 filter: {len(news_df):,}")

# ---------------------------------------------------------
# 3. TRADING DAY ALIGNMENT PROTOCOL (FIXED)
# ---------------------------------------------------------
valid_trading_days = pd.Series(returns_df.index.sort_values())

def align_to_trading_day_fixed(ts, trading_days_series):
    """
    Paper Alignment Protocol:
    - Published before 16:00 ET -> Market session t
    - Published at/after 16:00 ET -> Next market session t+1
    """
    # Force conversion to normalized Timestamp
    if ts.time() >= pd.to_datetime("16:00:00").time():
        target_dt = pd.Timestamp(ts.date()) + pd.Timedelta(days=1)
    else:
        target_dt = pd.Timestamp(ts.date())
        
    matches = trading_days_series[trading_days_series >= target_dt]
    if not matches.empty:
        return matches.iloc[0]
    return pd.NaT

print("[+] Aligning timestamps to trading days...")
news_df['aligned_date'] = news_df['published_et'].apply(lambda x: align_to_trading_day_fixed(x, valid_trading_days))
news_df = news_df.dropna(subset=['aligned_date']).copy()

# ---------------------------------------------------------
# 4. 3-DAY EXCESS RETURN COMPUTATION & BINARY LABELING (OPTIMIZED)
# ---------------------------------------------------------
print("[+] Computing 3-day cumulative excess returns (r_excess_3d)...")

# Excess Daily Return Matrix: r_{i,t} - r_{m,t}
excess_daily_returns = returns_df.drop(columns=['MARKET_BENCHMARK']).sub(returns_df['MARKET_BENCHMARK'], axis=0)

# 3-day forward cumulative excess returns: (t+1) + (t+2) + (t+3)
fwd_3d_excess = (
    excess_daily_returns.shift(-1) + 
    excess_daily_returns.shift(-2) + 
    excess_daily_returns.shift(-3)
)

# Melt to long format for direct high-speed vector join
returns_long = fwd_3d_excess.melt(
    ignore_index=False, 
    var_name='ticker', 
    value_name='r_excess_3d'
).reset_index()

returns_long.rename(columns={'index': 'aligned_date', 'Date': 'aligned_date'}, inplace=True)
returns_long['aligned_date'] = pd.to_datetime(returns_long['aligned_date'])
returns_long['ticker'] = returns_long['ticker'].astype(str).str.strip().str.upper()

# Merge news with computed forward returns
labeled_df = news_df.merge(
    returns_long, 
    on=['aligned_date', 'ticker'], 
    how='inner'
).dropna(subset=['r_excess_3d']).copy()

# Binary classification label
labeled_df['label'] = (labeled_df['r_excess_3d'] > 0).astype(int)

# ---------------------------------------------------------
# 5. SAVE LABELED ARTIFACT & SUMMARY
# ---------------------------------------------------------
output_path = os.path.join(SCRIPT_DIR, "labeled_news_dataset.parquet")
labeled_df.to_parquet(output_path, index=False)

pos_count = (labeled_df['label'] == 1).sum()
neg_count = (labeled_df['label'] == 0).sum()
total = len(labeled_df)

print("\n" + "=" * 65)
print(" STEP 2 FULL-SCALE LABELING COMPLETE ")
print("=" * 65)
print(f"Target Era               : 2019 – 2023")
print(f"Final Labeled Dataset    : {total:,} rows")
print(f"Positive Labels (1)      : {pos_count:,} ({pos_count/total:.2%})")
print(f"Negative Labels (0)      : {neg_count:,} ({neg_count/total:.2%})")
print(f"Saved Labeled Dataset to : '{output_path}'")
print("=" * 65)