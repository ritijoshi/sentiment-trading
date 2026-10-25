import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("data2")
OUTPUT_FILE = DATA_DIR / "news_with_returns.csv"

HOLDING_SESSIONS = 3

print("Loading datasets...")

# --------------------------------------------------
# 1. Load matched news
# --------------------------------------------------

news = pd.read_csv(DATA_DIR / "matched_news.csv")

news["date"] = pd.to_datetime(
    news["date"], errors="coerce"
).dt.normalize()

news = news.dropna(
    subset=["date", "title", "symbol"]
).copy()

# Research period
news = news[
    (news["date"] >= "2020-01-01") &
    (news["date"] <= "2026-03-30")
].copy()

# --------------------------------------------------
# 2. Load stock prices
# --------------------------------------------------

stocks = pd.read_csv(DATA_DIR / "nifty50_ticker.csv")

# Convert timezone-aware dates to IST, then remove timezone
stocks["Date"] = (
    pd.to_datetime(stocks["Date"], errors="coerce", utc=True)
    .dt.tz_convert("Asia/Kolkata")
    .dt.tz_localize(None)
    .dt.normalize()
)

# Prefer adjusted prices
price_column = "Adj Close"

if price_column not in stocks.columns:
    raise ValueError("Adjusted closing prices are unavailable.")

stocks[price_column] = pd.to_numeric(
    stocks[price_column], errors="coerce"
)

stocks = stocks.dropna(
    subset=["Date", "Symbol", price_column]
).copy()

stocks = stocks.rename(
    columns={price_column: "stock_price"}
)

stocks = stocks[
    ["Date", "Symbol", "stock_price"]
].sort_values(["Symbol", "Date"])

# --------------------------------------------------
# 3. Load NIFTY 50 benchmark
# --------------------------------------------------

market = pd.read_csv(DATA_DIR / "nifty50_index.csv")

market["Date"] = pd.to_datetime(
    market["Date"], errors="coerce"
).dt.normalize()

market["Close"] = pd.to_numeric(
    market["Close"], errors="coerce"
)

market = market.dropna(
    subset=["Date", "Close"]
).sort_values("Date")

market = market.drop_duplicates(
    subset=["Date"], keep="last"
)

market = market.set_index("Date")["Close"]

# --------------------------------------------------
# 4. Calculate returns using common trading dates
# --------------------------------------------------

print("Calculating three-session returns...")

results = []

# Group news by stock to avoid repeatedly filtering all prices
news_groups = news.groupby("symbol", sort=False)

for symbol, article_group in news_groups:

    stock_group = stocks[
        stocks["Symbol"] == symbol
    ].copy()

    if stock_group.empty:
        continue

    stock_group = stock_group.set_index("Date")
    stock_group = stock_group[
        ~stock_group.index.duplicated(keep="last")
    ]

    # Use only dates available in both datasets
    common_dates = stock_group.index.intersection(
        market.index
    ).sort_values()

    if len(common_dates) <= HOLDING_SESSIONS:
        continue

    stock_prices = stock_group.loc[
        common_dates, "stock_price"
    ].to_numpy(dtype=float)

    market_prices = market.loc[
        common_dates
    ].to_numpy(dtype=float)

    # Entry is the first common trading session strictly
    # after the article's date.
    article_dates = pd.DatetimeIndex(
        article_group["date"]
    )

    entry_positions = common_dates.searchsorted(
        article_dates, side="right"
    )

    # Exit three trading sessions after entry
    exit_positions = entry_positions + HOLDING_SESSIONS

    valid = exit_positions < len(common_dates)

    if not valid.any():
        continue

    valid_articles = article_group.iloc[
        np.flatnonzero(valid)
    ].copy()

    entry_pos = entry_positions[valid]
    exit_pos = exit_positions[valid]

    stock_entry = stock_prices[entry_pos]
    stock_exit = stock_prices[exit_pos]

    market_entry = market_prices[entry_pos]
    market_exit = market_prices[exit_pos]

    stock_return = stock_exit / stock_entry - 1
    market_return = market_exit / market_entry - 1

    excess_return = stock_return - market_return

    valid_articles["entry_date"] = common_dates.take(
        entry_pos
    ).to_numpy()

    valid_articles["exit_date"] = common_dates.take(
        exit_pos
    ).to_numpy()

    valid_articles["entry_price"] = stock_entry
    valid_articles["exit_price"] = stock_exit

    valid_articles["stock_return_3d"] = stock_return
    valid_articles["market_return_3d"] = market_return
    valid_articles["excess_return_3d"] = excess_return

    # 1 = positive excess return, 0 = zero or negative
    valid_articles["label"] = (
        excess_return > 0
    ).astype("int8")

    results.append(valid_articles)

# --------------------------------------------------
# 5. Combine and save
# --------------------------------------------------

if not results:
    raise ValueError(
        "No valid returns calculated. Check stock symbols "
        "and price coverage."
    )

final_df = pd.concat(results, ignore_index=True)

final_df = final_df.sort_values(
    ["date", "symbol"]
).reset_index(drop=True)

final_df.to_csv(OUTPUT_FILE, index=False)

# --------------------------------------------------
# 6. Quality summary
# --------------------------------------------------

print("\nReturn calculation complete.")
print("Input article-stock pairs:", len(news))
print("Output article-stock pairs:", len(final_df))

print(
    "Unique articles:",
    final_df["url"].nunique()
)

print("\nLabel distribution:")
print(final_df["label"].value_counts())

print("\nLabel proportions:")
print(
    (final_df["label"].value_counts(normalize=True) * 100)
    .round(2)
    .astype(str) + "%"
)

print("\nReturn statistics:")
print(
    final_df[
        [
            "stock_return_3d",
            "market_return_3d",
            "excess_return_3d",
        ]
    ].describe().round(6)
)

print("\nDate validation:")
print(
    "Entry after publication:",
    (
        final_df["entry_date"] > final_df["date"]
    ).all()
)

print(
    "Exit after entry:",
    (
        pd.to_datetime(final_df["exit_date"])
        > pd.to_datetime(final_df["entry_date"])
    ).all()
)

print("\nSaved to:", OUTPUT_FILE)