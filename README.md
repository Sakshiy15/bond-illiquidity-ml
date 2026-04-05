# Predicting Corporate Bond Illiquidity Using Machine Learning

> **Key Finding:** Time to maturity and issue size — structural features fixed at bond issuance — outperformed bid-ask spread history by **3.5×** as predictors of future illiquidity.

## Overview

This project builds a machine learning framework to predict whether a U.S. corporate bond will become illiquid in the following month, using an integrated panel of bond market microstructure data, structural bond characteristics, and firm-level financial fundamentals.

**Data source:** WRDS (Wharton Research Data Services) — the institutional-grade platform used by top banks, hedge funds, and research universities globally.

---

## Results at a Glance

| Model | AUC | Recall |
|-------|-----|--------|
| Logistic Regression | 0.864 | 78.4% |
| Random Forest | **0.881** | **84.4%** |
| XGBoost (Balanced) | 0.878 | 80.5% |

### Stress Period Robustness

| Period | AUC | Recall |
|--------|-----|--------|
| COVID-19 Shock (2020) | **0.943** | **96.2%** |
| Rate Hike Cycle (2022) | 0.935 | 95.3% |
| Post-Hike Adjustment (2023 H1) | 0.938 | 94.0% |

### Top SHAP Features (What Actually Predicts Illiquidity)

| Rank | Feature | SHAP Score | Category |
|------|---------|-----------|----------|
| 1 | Time to Maturity (TMT) | 1.336 | Bond Structural |
| 2 | Issue Size (AMOUNT_OUTSTANDING) | 0.835 | Bond Structural |
| 3 | Lagged Bid-Ask Spread | 0.381 | Market Microstructure |
| 4 | Coupon Rate | 0.240 | Bond Structural |
| 5 | Amihud Illiquidity (lagged) | 0.211 | Market Microstructure |

---

## Data Sources (WRDS)

| Source | Content |
|--------|---------|
| **TRACE** (Trade Reporting & Compliance Engine) | Bond-level transaction data — spreads, volumes, returns |
| **FISD** (Fixed Income Securities Database) | Bond structural characteristics — maturity, coupon, issue size, rating |
| **Compustat North America** | Firm-level financials — ROA, leverage, coverage, liquidity ratios |
| **CRSP** | Market-wide indicators |
| **FRED** | Macroeconomic data |

**Sample:** 88,025 bond-month observations · January 2018 – December 2023  
**Firm data match rate:** 53.8% via CUSIP linkage. Unmatched observations 
use cross-sectional median imputation — this attenuates firm-level feature 
importance estimates (ROA, price-to-book). A more complete GVKEY-CUSIP 
linkage would strengthen the firm-level dimension.

---

## Repository Structure

```
bond-illiquidity-ml/
│
├── bond_illiquidity_ml.ipynb   ← Main notebook (run this)
├── README.md                   ← This file
│
├── data/                       ← Place your WRDS data files here
│   ├── bond_data.csv           (from FISD/TRACE)
│   └── firm_level_financial.csv (from Compustat)
│
└── outputs/                    ← Generated automatically
    ├── model_comparison.csv
    ├── robustness_results.csv
    ├── shap_importance.csv
    ├── roc_comparison.png
    └── shap_summary.png
```

---

## Setup

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/bond-illiquidity-ml.git
cd bond-illiquidity-ml

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn xgboost shap jupyter

# Create output folder
mkdir outputs

# Add your WRDS data to the data/ folder, then run
jupyter notebook bond_illiquidity_ml.ipynb
```

---

## Key Methodology

- **Time-based train/test split** — pre-July 2023 train, July–December 2023 test. Prevents temporal leakage.
- **25 features** across three categories: market microstructure, bond structural, firm fundamentals
- **Class imbalance** (~9:1 liquid:illiquid) handled via class-weight balancing
- **SHAP** for explainability — every prediction has an economic reason, not just a number
- **No look-ahead bias** — all predictors lagged; firm data uses public_date (announcement date)
- **Firm-level merge:** 53.8% CUSIP match rate to Compustat. 
  Remaining observations imputed with monthly cross-sectional medians. 
  Firm fundamental SHAP values (ROA, PTB) should be interpreted as 
  conservative lower bounds on their true predictive importance.
---

## References

- Cabrol, Drobetz, Otto & Puhan (2024) — *Financial Analysts Journal* 80(3): 103–127
- Gu, Kelly & Xiu (2020) — *Review of Financial Studies* 33(5): 2223–2273
- Lundberg & Lee (2017) — *NeurIPS* 30: 4765–4774
- Amihud (2002) — *Journal of Financial Markets* 5(1): 31–56
- Merton (1974) — *The Journal of Finance* 29(2): 449–470

---

*Data accessed via WRDS (Wharton Research Data Services). Raw data not included in this repository due to WRDS licensing terms. Access requires institutional subscription.*
