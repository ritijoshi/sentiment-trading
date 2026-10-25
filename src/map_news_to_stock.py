import pandas as pd
import re
from pathlib import Path

# Paths
DATA_DIR = Path("data2")

# Load datasets
print("Loading datasets...")

news = pd.read_csv(DATA_DIR / "combined_news.csv")
stocks = pd.read_csv(DATA_DIR / "nifty50_ticker.csv")

print("News loaded:", len(news), "articles")

# Parse dates and restrict research period
news["date"] = pd.to_datetime(news["date"], errors="coerce")

news = news[
    (news["date"] >= "2020-01-01") &
    (news["date"] <= "2026-03-30")
].copy()

news = news.dropna(subset=["date", "title", "news"])

print("Articles in research period:", len(news))

# Get unique companies and symbols
companies = stocks[["Symbol", "Company"]].drop_duplicates()

# Company aliases
aliases = {
    "ADANIENT.NS": ["Adani Enterprises", "Adani Enterprises Limited"],
    "ADANIPORTS.NS": ["Adani Ports", "Adani Ports and Special Economic Zone"],
    "APOLLOHOSP.NS": ["Apollo Hospitals", "Apollo Hospitals Enterprise"],
    "ASIANPAINT.NS": ["Asian Paints"],
    "AXISBANK.NS": ["Axis Bank"],
    "BAJAJ-AUTO.NS": ["Bajaj Auto", "Bajaj Auto Limited"],
    "BAJAJFINSV.NS": ["Bajaj Finserv"],
    "BAJFINANCE.NS": ["Bajaj Finance"],
    "BEL.NS": ["Bharat Electronics", "BEL"],
    "BHARTIARTL.NS": ["Bharti Airtel"],
    "CIPLA.NS": ["Cipla"],
    "COALINDIA.NS": ["Coal India", "Coal India Limited"],
    "DRREDDY.NS": ["Dr. Reddy's Laboratories", "Dr Reddy's Laboratories"],
    "EICHERMOT.NS": ["Eicher Motors"],
    "ETERNAL.NS": [
    "Eternal Limited",
    "Eternal",
    "Zomato Limited",
    "Zomato"
],
    "GRASIM.NS": ["Grasim Industries"],
    "HCLTECH.NS": ["HCL Technologies", "HCL Tech"],
    "HDFCBANK.NS": ["HDFC Bank"],
    "HDFCLIFE.NS": ["HDFC Life"],
    "HEROMOTOCO.NS": [
    "Hero MotoCorp",
    "Hero MotoCorp Limited",
    "Hero Honda Motors"
],
    "HINDALCO.NS": ["Hindalco", "Hindalco Industries"],
    "HINDUNILVR.NS": ["Hindustan Unilever", "HUL"],
    "ICICIBANK.NS": ["ICICI Bank"],
    "INDIGO.NS": ["IndiGo", "InterGlobe Aviation"],
    "INFY.NS": ["Infosys"],
    "ITC.NS": [
    "ITC Limited",
    "ITC Ltd",
    "ITC"
],
    "JSWSTEEL.NS": ["JSW Steel"],
    "JIOFIN.NS": ["Jio Financial Services"],
    "KOTAKBANK.NS": ["Kotak Mahindra Bank", "Kotak Bank"],
    "LT.NS": ["Larsen & Toubro", "L&T"],
    "M&M.NS": ["Mahindra & Mahindra"],
    "MARUTI.NS": ["Maruti Suzuki"],
    "MAXHEALTH.NS": ["Max Healthcare"],
    "NESTLEIND.NS": ["Nestle India", "Nestlé India"],
    "NTPC.NS": ["NTPC"],
    "ONGC.NS": ["Oil and Natural Gas Corporation", "ONGC"],
    "POWERGRID.NS": ["Power Grid Corporation", "Power Grid"],
    "RELIANCE.NS": ["Reliance Industries"],
    "SBILIFE.NS": ["SBI Life"],
    "SBIN.NS": ["State Bank of India", "SBI"],
    "SHRIRAMFIN.NS": ["Shriram Finance"],
    "SUNPHARMA.NS": ["Sun Pharmaceutical", "Sun Pharma"],
    "TATACONSUM.NS": ["Tata Consumer Products"],
    "TATASTEEL.NS": ["Tata Steel"],
    "TCS.NS": ["Tata Consultancy Services", "TCS"],
    "TECHM.NS": ["Tech Mahindra"],
    "TITAN.NS": ["Titan Company"],
    "TRENT.NS": ["Trent"],
    "ULTRACEMCO.NS": ["UltraTech Cement"],
    "WIPRO.NS": ["Wipro"],
}

# Add official company names
for _, row in companies.iterrows():
    symbol = row["Symbol"]
    company = str(row["Company"]).strip()

    if company and company.lower() != "nan":
        aliases.setdefault(symbol, []).append(company)

# Compile patterns once
patterns = {}

for symbol, names in aliases.items():
    unique_names = sorted(set(names), key=len, reverse=True)

    patterns[symbol] = [
        (
            name,
            re.compile(
                r"(?<!\w)" + re.escape(name) + r"(?!\w)",
                flags=re.IGNORECASE
            )
        )
        for name in unique_names
        if name.strip()
    ]

# Match companies in article titles
print("Matching company names in titles...")

matched_rows = []

for row in news.itertuples(index=False):
    title = str(row.title)
    matches_for_article = []

    for symbol, name_patterns in patterns.items():
        matched_names = [
            name
            for name, pattern in name_patterns
            if pattern.search(title)
        ]

        if matched_names:
            matches_for_article.append({
                "date": row.date,
                "title": title,
                "news": row.news,
                "url": row.url,
                "source": row.source,
                "symbol": symbol,
                "matched_names": "; ".join(matched_names),
            })

    matched_rows.extend(matches_for_article)

# Convert results to DataFrame
matched = pd.DataFrame(matched_rows)

# Save results
output_path = DATA_DIR / "matched_news.csv"
matched.to_csv(output_path, index=False)

# Report results
print("\nMatching complete!")
print("Articles in research period:", len(news))
print("Article-company matches:", len(matched))

if not matched.empty:
    print("Unique matched articles:", matched["url"].nunique())

    print("\nMatches by stock:")
    print(matched["symbol"].value_counts().to_string())

    print("\nMatches by source:")
    print(matched["source"].value_counts().to_string())
else:
    print("No matches found. Check the company names and aliases.")

print("\nSaved to:", output_path)