import pandas as pd
import json
from pathlib import Path

DATA_DIR = Path("data2")
REPORT_DIR = DATA_DIR / "validation_report"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

report = {
    "checks": {},
    "warnings": [],
    "errors": []
}


def check_file(filename):
    path = DATA_DIR / filename

    if not path.exists():
        report["errors"].append(f"Missing file: {filename}")
        return None

    try:
        df = pd.read_csv(path, low_memory=False)
        print(f"\n{'=' * 60}")
        print(f"FILE: {filename}")
        print(f"Rows: {len(df):,}")
        print(f"Columns: {len(df.columns)}")

        report["checks"][filename] = {
            "rows": int(len(df)),
            "columns": list(df.columns),
            "missing_values": {
                col: int(count)
                for col, count in df.isna().sum().items()
                if count > 0
            }
        }

        return df

    except Exception as e:
        report["errors"].append(
            f"Could not read {filename}: {str(e)}"
        )
        return None


# --------------------------------------------------
# 1. Validate final article-return dataset
# --------------------------------------------------

df = check_file("news_with_returns.csv")

if df is not None:

    required = [
    "date", "title", "news", "symbol",
    "stock_return_3d", "market_return_3d",
    "excess_return_3d", "label"
]

    missing_columns = [
        col for col in required if col not in df.columns
    ]

    if missing_columns:
        report["errors"].append(
            f"Missing required columns in news_with_returns.csv: "
            f"{missing_columns}"
        )
    else:
        # Check missing values in important columns
        important = [
    "date", "title", "news", "symbol",
    "stock_return_3d", "market_return_3d",
    "excess_return_3d", "label"
]

        missing = df[important].isna().sum().to_dict()

        report["checks"]["final_dataset_missing_values"] = {
            k: int(v) for k, v in missing.items()
        }

        # Check invalid dates
        dates = pd.to_datetime(df["date"], errors="coerce")
        invalid_dates = int(dates.isna().sum())

        # Check non-numeric and infinite returns
        return_columns = [
            "stock_return_3d",
            "market_return_3d",
            "excess_return_3d"
        ]

        for col in return_columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        import numpy as np

        nonfinite_returns = int(
            (~np.isfinite(df[return_columns])).any(axis=1).sum()
        )

        # Check labels
        labels = pd.to_numeric(df["label"], errors="coerce")
        invalid_labels = int(
            (~labels.isin([0, 1])).sum()
        )

        # Check duplicate article-stock pairs
        if {"url", "symbol"}.issubset(df.columns):
            duplicate_pairs = int(
                df.duplicated(
                    subset=["url", "symbol"],
                    keep=False
                ).sum()
            )
        else:
            duplicate_pairs = None

        # Check duplicate URLs
        if "url" in df.columns:
            duplicate_urls = int(
                df.loc[
                    df["url"].notna() & df["url"].ne(""),
                    "url"
                ].duplicated().sum()
            )
        else:
            duplicate_urls = None

        report["checks"]["final_dataset_quality"] = {
            "invalid_dates": invalid_dates,
            "rows_with_nonfinite_returns": nonfinite_returns,
            "invalid_labels": invalid_labels,
            "duplicate_article_stock_rows": duplicate_pairs,
            "duplicate_url_rows": duplicate_urls,
            "label_distribution": {
                str(k): int(v)
                for k, v in labels.value_counts(dropna=False).items()
            },
            "date_min": (
                str(dates.min().date())
                if dates.notna().any() else None
            ),
            "date_max": (
                str(dates.max().date())
                if dates.notna().any() else None
            )
        }

        if invalid_dates:
            report["warnings"].append(
                f"{invalid_dates} rows have invalid dates."
            )

        if missing["title"] or missing["news"] or missing["symbol"]:
            report["warnings"].append(
                "Some records have missing titles, article text, or symbols."
            )

        if nonfinite_returns:
            report["warnings"].append(
                f"{nonfinite_returns} rows have invalid or infinite returns."
            )

        if invalid_labels:
            report["warnings"].append(
                f"{invalid_labels} rows have labels outside 0 and 1."
            )

        if duplicate_pairs:
            report["warnings"].append(
                f"{duplicate_pairs} rows belong to duplicated URL-symbol pairs."
            )


# --------------------------------------------------
# 2. Validate train, validation, and test splits
# --------------------------------------------------

split_names = ["train.csv", "validation.csv", "test.csv"]
splits = {}

for filename in split_names:
    split_df = check_file(filename)

    if split_df is not None:
        splits[filename] = split_df

        if "label" in split_df.columns:
            labels = pd.to_numeric(
                split_df["label"], errors="coerce"
            )

            report["checks"][filename]["label_distribution"] = {
                str(k): round(float(v), 2)
                for k, v in (
                    labels.value_counts(normalize=True) * 100
                ).items()
            }

# Check split sizes and overlap
if len(splits) == 3:

    total_split_rows = sum(len(x) for x in splits.values())

    report["checks"]["split_validation"] = {
        "total_split_rows": int(total_split_rows),
        "original_rows": (
            int(len(df)) if df is not None else None
        ),
        "split_sizes": {
            name: int(len(split))
            for name, split in splits.items()
        }
    }

    if df is not None and total_split_rows != len(df):
        report["errors"].append(
            "Total split rows do not equal the final dataset row count."
        )

    # Check whether identical URL-symbol pairs occur across splits.
    if all(
        {"url", "symbol"}.issubset(split.columns)
        for split in splits.values()
    ):
        pair_sets = {}

        for name, split in splits.items():
            pairs = set(
                zip(
                    split["url"].fillna("").astype(str),
                    split["symbol"].fillna("").astype(str)
                )
            )
            pair_sets[name] = pairs

        overlap = {}

        names = list(pair_sets)

        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                overlap[f"{a} vs {b}"] = len(
                    pair_sets[a] & pair_sets[b]
                )

        report["checks"]["split_pair_overlap"] = overlap

        if any(overlap.values()):
            report["warnings"].append(
                "Identical URL-symbol pairs appear across splits."
            )


# --------------------------------------------------
# 3. Save machine-readable report
# --------------------------------------------------

json_path = REPORT_DIR / "validation_report.json"

with open(json_path, "w", encoding="utf-8") as file:
    json.dump(report, file, indent=4, default=str)


# --------------------------------------------------
# 4. Save a readable text report
# --------------------------------------------------

txt_path = REPORT_DIR / "validation_report.txt"

with open(txt_path, "w", encoding="utf-8") as file:
    file.write("FINAL DATA VALIDATION REPORT\n")
    file.write("=" * 60 + "\n\n")

    file.write("CHECK RESULTS\n")
    file.write("-" * 60 + "\n")

    for name, result in report["checks"].items():
        file.write(f"\n{name}\n")
        file.write(json.dumps(result, indent=2, default=str))
        file.write("\n")

    file.write("\nWARNINGS\n")
    file.write("-" * 60 + "\n")

    if report["warnings"]:
        for warning in report["warnings"]:
            file.write(f"- {warning}\n")
    else:
        file.write("No warnings detected.\n")

    file.write("\nERRORS\n")
    file.write("-" * 60 + "\n")

    if report["errors"]:
        for error in report["errors"]:
            file.write(f"- {error}\n")
    else:
        file.write("No errors detected.\n")


# --------------------------------------------------
# 5. Print summary
# --------------------------------------------------

print("\n" + "=" * 60)
print("VALIDATION COMPLETE")
print("=" * 60)

print(f"\nWarnings: {len(report['warnings'])}")
for warning in report["warnings"]:
    print("WARNING:", warning)

print(f"\nErrors: {len(report['errors'])}")
for error in report["errors"]:
    print("ERROR:", error)

print(f"\nReports saved to: {REPORT_DIR}")
print(" - validation_report.txt")
print(" - validation_report.json")