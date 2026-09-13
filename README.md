# Financial News Sentiment & Market Impact Dataset Pipeline

An end-to-end data pipeline designed to ingest, clean, align, and label historical financial news headlines against market return metrics for S&P 100 constituents (2019–2023). 

This project maps intra-day and after-hours financial news to daily market trading sessions and labels headlines using a **3-day forward excess return methodology** relative to a benchmark index ($SPY$).

---

## 📌 Dataset Overview

* **Temporal Scope:** 2019–2023 (5 years)
* **Total Labeled Rows:** 32,201
* **Coverage:** 14 major S&P 100 constituents (AAPL, AMZN, DIS, TSLA, JPM, NVDA, HD, GOOGL, JNJ, BAC, XOM, UNH, MA, PG)
* **Label Distribution:** 
  * `1` (Positive Excess Return): **51.76%** (16,668 rows)
  * `0` (Negative/Zero Excess Return): **48.24%** (15,533 rows)
* **Data Integrity:** Strict chronological partitioning; zero date overlap between Train (2019–2021), Validation (2022), and Test (2023) sets.

---

## 📐 Labeling & Trading Alignment Methodology

1. **Market Session Alignment:**
   * Headlines published **before 16:00 ET (4:00 PM)** align to the current trading day session ($t$).
   * Headlines published **at or after 16:00 ET**, or on weekends/holidays, roll forward to the next valid market trading session ($t+1$).

2. **Target Calculation (3-Day Excess Return):**
   * Daily excess return: $r_{i,t}^{excess} = r_{i,t} - r_{m,t}$ (where $r_{m}$ is the market benchmark return).
   * 3-day cumulative forward excess return: $R_{i,t}^{3d} = r_{i,t+1}^{excess} + r_{i,t+2}^{excess} + r_{i,t+3}^{excess}$
   * Binary Label Assignment:
     $$\text{Label} = \begin{cases} 1 & \text{if } R_{i,t}^{3d} > 0 \\ 0 & \text{otherwise} \end{cases}$$

---

## 📁 Repository Structure

```text
├── data/
│   ├── 01_data_ingestion.py         # Pulls market data & raw headline sources
│   ├── 01b_clean_dataset.py         # Standardizes timestamps & text fields
│   ├── 02_timestamp_and_labeling.py # Vectorized trading day alignment & return calculation
│   └── 03_inspect_labeled_dataset.py# Data integrity audit & distribution sanity checks
├── .gitignore                       # Ignored build & large binary files
└── README.md                        # Project documentation
🚀 Quick Start & Usage
1. Prerequisites & Installation
Clone the repository and install the project dependencies:

Bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/sentiment-trading-prod.git
cd sentiment-trading-prod

# Install dependencies from requirements.txt
pip install -r requirements.txt
2. Running the Pipeline
Execute the data pipeline sequentially:

Bash
# Ingest and clean raw inputs
python data/01_data_ingestion.py
python data/01b_clean_dataset.py

# Perform trading session alignment & 3-day return labeling
python data/02_timestamp_and_labeling.py

# Verify dataset health metrics
python data/03_inspect_labeled_dataset.py

⚙️ Model Training Next Steps
The processed dataset is structured for downstream NLP benchmarking:

Baseline Benchmarks: CountVectorizer / TF-IDF + Logistic Regression / XGBoost.

Deep Learning & Embeddings: FinBERT fine-tuning and daily headline embedding mean-pooling.