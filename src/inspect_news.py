import pandas as pd

df = pd.read_csv("data2/combined_news.csv")

print("Columns:", df.columns.tolist())

print("\nSample titles:")
for i, title in enumerate(df["title"].dropna().sample(20, random_state=42), 1):
    print(f"{i}. {title}")

print("\nSample article text:")
for i, article in enumerate(df["news"].dropna().sample(3, random_state=42), 1):
    print(f"\nArticle {i}:")
    print(article[:800])