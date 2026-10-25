import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

DATA_DIR = Path("data2")
RANDOM_STATE = 42

# Load cleaned dataset
df = pd.read_csv(DATA_DIR / "news_with_returns.csv")
df["date"] = pd.to_datetime(df["date"], errors="coerce")

df = df.dropna(
    subset=["date", "label", "excess_return_3d"]
).copy()

df["label"] = df["label"].astype(int)

print("Total records:", len(df))

# First, reserve 20% for testing
train_val, test = train_test_split(
    df,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=df["label"]
)

# From the remaining 80%, reserve 20% for validation
train, validation = train_test_split(
    train_val,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=train_val["label"]
)

# Save datasets
train.to_csv(DATA_DIR / "train.csv", index=False)
validation.to_csv(DATA_DIR / "validation.csv", index=False)
test.to_csv(DATA_DIR / "test.csv", index=False)

# Report split details
for name, part in [
    ("Training", train),
    ("Validation", validation),
    ("Test", test),
]:
    print(f"\n{name} set")
    print("Rows:", len(part))
    print(f"Percentage: {len(part) / len(df) * 100:.2f}%")
    print("Label distribution (%):")
    print(
        (part["label"].value_counts(normalize=True) * 100)
        .round(2)
    )

print("\nTotal split records:", len(train) + len(validation) + len(test))
print("Original records:", len(df))
print("\nFiles saved in:", DATA_DIR)