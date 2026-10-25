import pandas as pd

df = pd.read_csv("data2/matched_news.csv")

# Sample up to 100 distinct article-stock pairs
sample = df.sample(
    n=min(100, len(df)),
    random_state=42
).copy()

sample["match_correct"] = ""
sample["notes"] = ""

sample.to_csv(
    "data2/manual_match_review.csv",
    index=False
)

print("Saved sample of", len(sample), "matches.")
print("File: data2/manual_match_review.csv")
print("Review match_correct as yes/no and add notes if needed.")