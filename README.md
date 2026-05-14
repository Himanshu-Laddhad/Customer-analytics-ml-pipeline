# KKBox Churn + CLV Prediction

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange)
![LightGBM](https://img.shields.io/badge/LightGBM-4.3-green)
![SHAP](https://img.shields.io/badge/SHAP-0.45-blueviolet)
![Optuna](https://img.shields.io/badge/Optuna-3.6-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red?logo=streamlit)

---

## Business Impact

| Capability | Before | After |
|---|---|---|
| Churn identification | Reactive (post-cancellation) | 30-day early warning — temporal split by registration cohort |
| Model performance | LR baseline ROC-AUC 0.9090 | XGBoost 0.9815 · Stacking 0.9817 |
| Retention targeting | Broad campaigns, unknown ROI | Per-customer EV model → $1.26M net ROI on 2,869 high-value targets |
| Explainability | Black box | SHAP TreeExplainer — feature-level attribution for every prediction |
| CLV estimation | No CLV model | LightGBM regressor R²=0.9923 on non-churned customers |

---

## What It Does

KKBox is Asia's largest music streaming platform with 6.7M users and 21.5M transactions. This pipeline ingests raw subscription and listening logs, engineers 35 behavioural features via DuckDB SQL, segments customers into four archetypes (Power Users, Casual Listeners, At-Risk, Dormant), trains and selects a churn classifier, estimates each customer's annual subscription value, and produces a fully quantified retention campaign ROI — all surfaced through an interactive Streamlit dashboard. A data leakage audit at the feature engineering stage identified and excluded three columns (`last_expire_date`, `last_transaction`, `first_transaction`) that inadvertently encode the March 2017 churn label.

---

## SQL Showcase

`queries/log_features.sql` — behavioural engagement features computed from 410M listening log rows:

```sql
SELECT
    l.msno,

    -- Volume
    SUM(l.total_secs)                                                            AS total_secs,
    AVG(l.total_secs)                                                            AS avg_daily_secs,
    MAX(l.total_secs)                                                            AS max_daily_secs,

    -- Engagement quality
    SUM(l.num_25 + l.num_50 + l.num_75 + l.num_985 + l.num_100)                AS total_songs_played,

    SUM(l.num_100) * 1.0
        / NULLIF(SUM(l.num_25 + l.num_50 + l.num_75 + l.num_985 + l.num_100), 0)
                                                                                 AS completion_rate,

    SUM(l.num_25) * 1.0
        / NULLIF(SUM(l.num_25 + l.num_50 + l.num_75 + l.num_985 + l.num_100), 0)
                                                                                 AS skip_rate,

    AVG(
        (l.num_unq * 1.0)
        / NULLIF(l.num_25 + l.num_50 + l.num_75 + l.num_985 + l.num_100, 0)
    )                                                                            AS unique_songs_ratio,

    -- Recency and consistency
    COUNT(DISTINCT l.date)                                                       AS active_days,
    MAX(l.date)                                                                  AS last_log_date,
    MIN(l.date)                                                                  AS first_log_date

FROM user_logs l
INNER JOIN train_labels lbl ON l.msno = lbl.msno
GROUP BY l.msno
```

---

## Skills Demonstrated

| Area | Detail |
|---|---|
| Data Engineering | DuckDB SQL feature engineering from 410M log rows; two-pass aggregation for RAM-constrained machines |
| Leakage Auditing | Identified target leakage in `last_expire_date` (70.5% vs 5.1% churn rate split) and `last_transaction` (AUC 0.74 alone) |
| ML — Classification | LR, RF, XGBoost, LightGBM, Stacking Ensemble; temporal train/val/test split; PredefinedSplit for GridSearchCV |
| Hyperparameter Tuning | Optuna 50-trial TPE studies for XGBoost and LightGBM; custom objective with early stopping |
| ML — Regression | Ridge, RF, XGB, LightGBM regressors for CLV; Optuna params reused with classification-specific keys stripped |
| Explainability | SHAP TreeExplainer for XGBoost/LightGBM; waterfall, beeswarm, bar, force plots; HTML batch export |
| Business Framing | EV-based ROI model with sensitivity analysis; segment prioritisation by expected net ROI |
| Dashboard | 5-page Streamlit app; zero runtime computation; Plotly charts; `st.cache_data` throughout |

---

## Model Performance

### Churn Classification (Validation Set, Threshold = 0.45)

| Model | Val ROC-AUC | Val F1 | Val Precision | Val Recall | Train Time (s) |
|---|---|---|---|---|---|
| Logistic Regression | 0.9090 | 0.5269 | 0.3759 | 0.8806 | 1.7 |
| Random Forest | 0.9774 | 0.7998 | 0.7747 | 0.8265 | 89.3 |
| XGBoost | **0.9815** | 0.7270 | 0.6083 | 0.9033 | 12.1 |
| LightGBM | 0.9813 | 0.7165 | 0.5902 | 0.9116 | 15.9 |
| Stacking Ensemble | 0.9817 | **0.8292** | **0.8858** | 0.7794 | 546.8 |

**Final model: XGBoost** — Stacking gained only +0.0002 AUC at 45× training cost.

### CLV Regression (Test Set, Non-Churned Customers)

| Model | RMSE | MAE | R² |
|---|---|---|---|
| Ridge | 403.59 | 312.60 | 0.449 |
| Random Forest | 52.72 | 21.12 | 0.991 |
| XGBoost | 56.86 | 39.32 | 0.989 |
| **LightGBM** | **47.81** | **27.30** | **0.992** |

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data querying | DuckDB 1.2 |
| ML | scikit-learn 1.5, XGBoost 2.0, LightGBM 4.3 |
| Tuning | Optuna 3.6 |
| Explainability | SHAP 0.45 |
| Dashboard | Streamlit 1.35, Plotly 5.22 |
| Data | pandas 2.2, pyarrow 16.1 |
| Serialisation | joblib 1.4 |

---

## Setup & Run

```bash
# 1. Clone
git clone <repo-url>
cd retail-clv-churn-prediction

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download KKBox data from Kaggle
#    https://www.kaggle.com/c/kkbox-churn-prediction-challenge/data
#    Place all CSV files in data/

# 5. Run notebooks in order (01 through 13)
jupyter notebook

# 6. Launch dashboard (all outputs pre-computed)
streamlit run app/dashboard.py
```

---

## Live Demo

🔗 _Deployment link — coming soon_

---

> **Last Updated:** May 2026 | **Dataset:** KKBox Music Streaming | **Status:** Complete
