import os
import sys
import io
import pandas as pd

# Force UTF-8 output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(SCRIPT_DIR, "labeled_news_dataset.parquet")

if not os.path.exists(file_path):
    print(f"[!] Error: Could not find {file_path}")
    sys.exit(1)

df = pd.read_parquet(file_path)

print("=" * 65)
print(" LABELED DATASET INSPECTION REPORT (2019–2023) ")
print("=" * 65)

# 1. Dataset Shape & Null Checks
print("\n[1] DATASET SUMMARY")
print(f"    - Total Rows       : {len(df):,}")
print(f"    - Total Columns    : {len(df.columns)}")
print(f"    - Column List      : {list(df.columns)}")
print(f"    - Missing Values   : {df.isnull().sum().sum()}")

# 2. Label Balance Check
pos_count = (df['label'] == 1).sum()
neg_count = (df['label'] == 0).sum()
total = len(df)
print("\n[2] TARGET LABEL BALANCE (3-Day Excess Return)")
print(f"    - Positive (1)     : {pos_count:,} ({pos_count/total:.2%})")
print(f"    - Negative (0)     : {neg_count:,} ({neg_count/total:.2%})")

# 3. Date Range & Annual Breakdown
df['aligned_date'] = pd.to_datetime(df['aligned_date'])
print("\n[3] TEMPORAL COVERAGE")
print(f"    - Start Date       : {df['aligned_date'].min().strftime('%Y-%m-%d')}")
print(f"    - End Date         : {df['aligned_date'].max().strftime('%Y-%m-%d')}")
print("\n    Headlines per Year:")
df['year'] = df['aligned_date'].dt.year
for yr, cnt in df['year'].value_counts().sort_index().items():
    print(f"      • {yr} : {cnt:,} headlines ({cnt/total:.2%})")

# 4. Ticker Distribution (2019-2023 Era)
print("\n[4] TICKER DISTRIBUTION")
ticker_counts = df['ticker'].value_counts()
for ticker, cnt in ticker_counts.items():
    print(f"    - {ticker:<6} : {cnt:,} ({cnt/total:.2%})")

# 5. Quick Sample Inspection
print("\n[5] FIRST 3 SAMPLE ROWS")
print("-" * 65)
for idx, row in df.head(3).iterrows():
    print(f"Ticker    : {row['ticker']}")
    print(f"Headline  : {row['headline']}")
    print(f"Date      : {row['aligned_date'].strftime('%Y-%m-%d')}")
    print(f"r_excess  : {row['r_excess_3d']:.4f}")
    print(f"Label     : {row['label']}")
    print("-" * 65)

print("\n[OK] Inspection complete! Ready for Git staging.")