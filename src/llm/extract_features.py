
import json
import time
import urllib.request
import argparse
from pathlib import Path
from collections import Counter

import pandas as pd


# ==================================================
# PATHS AND CONFIGURATION
# ==================================================

BASE_DIR = Path(__file__).resolve().parents[2]

CHUNKS_FILE = BASE_DIR / "data2" / "processed" / "article_chunks.csv"
ARTICLES_FILE = BASE_DIR / "data2" / "processed" / "clean_articles.csv"

OUTPUT_DIR = BASE_DIR / "data2" / "processed"

# Separate files for the improved experiment
CHECKPOINT_FILE = OUTPUT_DIR / "chunk_features_qwen3b_relevance.csv"
OUTPUT_FILE = OUTPUT_DIR / "llm_features_qwen3b_relevance.csv"

MODEL = "qwen2.5:3b"
API_URL = "http://localhost:11434/api/chat"


# ==================================================
# LLM PROMPT
# ==================================================

SYSTEM_PROMPT = """
You are an expert financial news analyst.

Analyse the supplied news text.

STEP 1: FINANCIAL RELEVANCE

Set is_financially_relevant to true when the text contains meaningful
information about:

- Listed companies and business activities.
- Corporate earnings, revenue, profits and losses.
- Investments, fundraising, acquisitions, mergers and stake sales.
- Stock markets, indices, equities and portfolio performance.
- Economic policies, interest rates, inflation and macroeconomic trends.
- Corporate management changes and regulatory developments.
- Industry developments that may materially affect businesses or markets.

Set it to false for unrelated entertainment, sports, travel advertisements,
lifestyle articles and other content without meaningful financial relevance.

Do not classify an article as irrelevant merely because it lacks a
specific earnings announcement or stock-price movement.

STEP 2: EVENT CLASSIFICATION

Choose one event_type:

earnings, merger, acquisition, product_launch, regulation,
macroeconomics, litigation, management_change, expansion,
partnership, operational_issue, market_update, investment,
industry_development, other.

Use other only when no meaningful financial or business development
is identifiable.

STEP 3: FEATURE EXTRACTION

sentiment_score:
-1 = strongly negative
 0 = neutral or balanced
+1 = strongly positive

impact_score:
0 = negligible business or market significance
1 = potentially very high significance

risk_score:
0 = negligible risk
1 = very high risk

uncertainty_score:
0 = clear and specific information
1 = highly uncertain information

confidence:
0 = very low confidence
1 = very high confidence

IMPORTANT RULES:

- Evaluate all meaningful events in the supplied text.
- If events have conflicting implications, use balanced sentiment.
- Positive company news does not guarantee a share-price increase.
- Do not invent facts, figures or company relationships.
- Do not predict future stock returns.
- If information is incomplete, express uncertainty.
- Financial relevance can be true even when sentiment is neutral.
- Keep explanations short and based only on the supplied text.

Return valid JSON with exactly these fields:

{
  "is_financially_relevant": true,
  "relevance_reason": "Short explanation",
  "sentiment_score": 0.5,
  "event_type": "earnings",
  "impact_score": 0.7,
  "risk_score": 0.2,
  "uncertainty_score": 0.3,
  "confidence": 0.8,
  "reasoning": "Short explanation"
}

Return JSON only. Do not include Markdown.
"""


# ==================================================
# HELPER FUNCTIONS
# ==================================================

def clamp(value, low, high):
    return max(low, min(high, float(value)))


def parse_bool(value):
    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes")

    return bool(value)


# ==================================================
# LLM ANALYSIS
# ==================================================


def analyse_chunk(text):
    payload = {
        "model": MODEL,
        "stream": False,
        "format": "json",
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": str(text)
            }
        ],
        "options": {
            "temperature": 0
        }
    }

    request = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    with urllib.request.urlopen(request, timeout=180) as response:
        result = json.loads(response.read().decode("utf-8"))

    content = json.loads(result["message"]["content"])

    # Check which fields are missing
    required_fields = [
        "is_financially_relevant",
        "relevance_reason",
        "sentiment_score",
        "event_type",
        "impact_score",
        "risk_score",
        "uncertainty_score",
        "confidence",
        "reasoning"
    ]

    missing = [
        field for field in required_fields
        if field not in content or content[field] is None
    ]

    if missing:
        raise ValueError(
            f"Qwen returned incomplete JSON. Missing fields: {missing}"
        )

    return {
        "is_financially_relevant": parse_bool(
            content["is_financially_relevant"]
        ),
        "relevance_reason": str(
            content["relevance_reason"]
        ).strip(),
        "sentiment_score": clamp(
            content["sentiment_score"], -1, 1
        ),
        "event_type": str(
            content["event_type"]
        ).strip().lower(),
        "impact_score": clamp(
            content["impact_score"], 0, 1
        ),
        "risk_score": clamp(
            content["risk_score"], 0, 1
        ),
        "uncertainty_score": clamp(
            content["uncertainty_score"], 0, 1
        ),
        "confidence": clamp(
            content["confidence"], 0, 1
        ),
        "reasoning": str(
            content["reasoning"]
        ).strip()
    }


# ==================================================
# AGGREGATE CHUNK RESULTS INTO ARTICLE FEATURES
# ==================================================

def aggregate_results(df):

    results = []

    for article_id, group in df.groupby("article_id"):

        event_counts = Counter(group["event_type"])

        results.append({

            "article_id": article_id,

            "is_financially_relevant":
                group["is_financially_relevant"].mean() >= 0.5,

            "relevance_reason": " | ".join(
                group["relevance_reason"]
                .drop_duplicates()
                .head(3)
            ),

            "sentiment_score":
                group["sentiment_score"].mean(),

            "event_type":
                event_counts.most_common(1)[0][0],

            "impact_score":
                group["impact_score"].mean(),

            "risk_score":
                group["risk_score"].mean(),

            "uncertainty_score":
                group["uncertainty_score"].mean(),

            "confidence":
                group["confidence"].mean(),

            "reasoning": " | ".join(
                group["reasoning"]
                .drop_duplicates()
                .head(3)
            ),

            "chunks_analysed": len(group)
        })

    return pd.DataFrame(results)


# ==================================================
# TEST ON ONE SAMPLE ARTICLE
# ==================================================

def test_real_article():

    sample = """
    Bharti Telecom raised Rs 8,433 crore by selling
    2.75 percent stake in Bharti Airtel.

    The promoter group will continue to own 56.23
    percent in the company.

    ITC signed an agreement to acquire 100 percent
    of Sunrise Foods to strengthen its spices portfolio.

    Coromandel International share price rallied
    8 percent following its fourth-quarter earnings.

    Deepak Nitrite reported an 88 percent increase
    in consolidated net profit at Rs 172.3 crore.

    Titan reported business challenges due to
    COVID-19 lockdown restrictions.
    """

    print("TEST ARTICLE:")
    print(sample)

    print("\nLLM FEATURES:\n")

    result = analyse_chunk(sample)

    print(json.dumps(result, indent=2))


# ==================================================
# MAIN PIPELINE
# ==================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum new chunks to process; 0 means all remaining chunks."
    )

    parser.add_argument(
        "--test-article",
        action="store_true"
    )

    parser.add_argument(
        "--smoke-test",
        action="store_true"
    )

    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Test Ollama connection and JSON extraction
    if args.smoke_test:

        sample = (
            "A company reported a 30 percent increase "
            "in quarterly profits and raised its "
            "annual revenue forecast."
        )

        print(json.dumps(analyse_chunk(sample), indent=2))
        return

    # Test on sample financial article
    if args.test_article:
        test_real_article()
        return

    # Check input dataset
    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(
            f"Missing chunks file: {CHUNKS_FILE}"
        )

    chunks = pd.read_csv(CHUNKS_FILE)

    print("Total available chunks:", len(chunks))

    # Load only the new experiment's checkpoint
    if CHECKPOINT_FILE.exists():

        completed = pd.read_csv(CHECKPOINT_FILE)

        done_ids = set(
            zip(
                completed["article_id"],
                completed["chunk_id"]
            )
        )

        print("Previously completed chunks:", len(done_ids))

    else:

        completed = pd.DataFrame()
        done_ids = set()

    # Identify chunks not yet processed
    remaining_mask = [
        (row.article_id, row.chunk_id) not in done_ids
        for row in chunks.itertuples(index=False)
    ]

    remaining = chunks.loc[remaining_mask].copy()

    if args.limit > 0:
        remaining = remaining.head(args.limit)

    total = len(remaining)

    print("Chunks remaining in this run:", total)

    if total == 0:
        print("No new chunks to process.")
    else:

        for count, (_, row) in enumerate(
            remaining.iterrows(), start=1
        ):

            print(
                f"\nProcessing {count}/{total}: "
                f"{row['article_id']} "
                f"(chunk {row['chunk_id']})"
            )

            for attempt in range(3):

                try:

                    result = analyse_chunk(row["chunk_text"])

                    result["article_id"] = row["article_id"]
                    result["chunk_id"] = row["chunk_id"]

                    pd.DataFrame([result]).to_csv(
                        CHECKPOINT_FILE,
                        mode="a",
                        header=not CHECKPOINT_FILE.exists(),
                        index=False
                    )

                    print(
                        "Financially relevant:",
                        result["is_financially_relevant"]
                    )

                    break

                except Exception as error:

                    print(
                        f"Attempt {attempt + 1} failed: {error}"
                    )

                    if attempt < 2:
                        time.sleep(2 ** (attempt + 1))
                    else:
                        print("Skipping this chunk.")

    # Aggregate all results from this experiment
    if CHECKPOINT_FILE.exists():

        all_results = pd.read_csv(CHECKPOINT_FILE)

        article_results = aggregate_results(all_results)

        article_results.to_csv(
            OUTPUT_FILE,
            index=False
        )

        print("\nEXTRACTION RUN FINISHED")

        print("Processed chunks:", len(all_results))

        print("Articles with results:", len(article_results))

        print(
            "Financially relevant articles:",
            int(article_results["is_financially_relevant"].sum())
        )

        print("Output:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
