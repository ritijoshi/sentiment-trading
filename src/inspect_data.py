import pandas as pd
from pathlib import Path

data_folder = Path("data2")

files = [
    "businessstandard_raw.csv",
    "economictimes_raw.csv",
    "financialexpress_raw.csv",
    "moneycontrol_raw.csv",
    "nifty50_ticker.csv"
]

for filename in files:
    path = data_folder / filename
    df = pd.read_csv(path)

    print(f"\n{'=' * 50}")
    print(f"File: {filename}")
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())

    date_col = next(
        (col for col in ["date", "Date"] if col in df.columns),
        None
    )

    if date_col:
        print("Sample raw dates:")
        print(df[date_col].dropna().sample(min(15, df[date_col].notna().sum())).tolist())
        dates = pd.to_datetime(df[date_col], errors="coerce")

        print("Earliest date:", dates.min())
        print("Latest date:", dates.max())
        print("Invalid dates:", dates.isna().sum())
        print("Sample invalid dates:")
        print(df.loc[dates.isna(), date_col].head(5).tolist())

    print(df.head(3))