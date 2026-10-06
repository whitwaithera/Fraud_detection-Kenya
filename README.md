# FraudShield Kenya — Financial Fraud Detection with XGBoost + SMOTE

An end-to-end machine learning pipeline that detects financial fraud across **M-Pesa mobile money**, **bank account transactions**, and **KRA tax filings** using synthetic Kenyan financial data.

Handles extreme class imbalance (fraud < 1.5% of transactions) using **SMOTE** and achieves **ROC-AUC of 1.00** with **XGBoost** and **LightGBM**.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    DATA LAYER                           │
│  M-Pesa Transactions │ Bank Transfers │ KRA Tax Filings │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│              FEATURE ENGINEERING                        │
│  ratio_to_avg_amount │ velocity_score │ risk_score      │
│  account_maturity    │ tax_risk       │ is_night        │
│  amount_vs_income    │ location flags │ device flags    │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│           CLASS IMBALANCE HANDLING                      │
│  SMOTE — Synthetic Minority Over-sampling Technique     │
│  1.5% fraud → 50% balanced training set                 │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────┐
│               ML MODELS                                  │
│   XGBoost (AUC: 1.00)  │  LightGBM (AUC: 1.00)         │
│   SHAP explainability for every prediction               │
└──────────────────────┬───────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│            STREAMLIT DEMO                               │
│  Live fraud scan │ Risk gauge │ Flagged transactions    │
│  Analytics dashboard │ Export flagged CSV               │
└─────────────────────────────────────────────────────────┘
```

---

## Fraud Types Detected

| Type | Description | Signal |
|---|---|---|
| **Velocity fraud** | Many rapid small transactions | `transactions_last_hour > 8` |
| **Large unusual** | Single amount far above norm | `ratio_to_avg_amount > 12` |
| **Account takeover** | New device + location + large transfer | `new_device + location_mismatch` |
| **KRA evasion** | Declared income far below expected | `declared_vs_expected_ratio < 0.15` |
| **Structuring** | Just-below-threshold transactions (AML) | `amount ≈ KES 99,000` |

---

## Key Features Engineered

```python
ratio_to_avg_amount    # Is this transaction 10x larger than their normal?
velocity_score         # (txns_last_hour × 3) + txns_last_day
risk_score             # Composite: ratio + device + location + time signals
tax_risk               # (1 - declared_ratio) + (days_late / 365)
account_maturity       # log(account_age_days) — newer accounts riskier
is_night               # Transactions between midnight and 5 AM
```

---

## Results

| Model | ROC-AUC | Avg Precision | F1 (Fraud) |
|---|---|---|---|
| XGBoost | **1.0000** | **1.0000** | **0.9934** |
| LightGBM | **1.0000** | **1.0000** | **0.9934** |

*Note: High AUC reflects strong synthetic signal injection. Real-world performance will vary.*

---

## Project Structure

```
fraud_detection/
├── 1_generate_data.py     # Synthetic data generator (Faker + Kenyan context)
├── 2_train_model.py       # SMOTE + XGBoost + LightGBM + SHAP
├── 3_app.py               # Streamlit live demo
├── data/
│   ├── kenya_fraud_dataset.csv   # 50,000 synthetic transactions
│   └── test_set.csv              # Held-out test set with predictions
├── models/
│   ├── best_model.pkl            # Saved XGBoost model
│   └── feature_cols.pkl          # Feature column list
├── outputs/
│   ├── confusion_matrix_xgboost.png
│   ├── roc_pr_xgboost.png
│   ├── shap_xgboost.png
│   └── ...
└── requirements.txt
```

---

## Quick Start

```bash

pip install -r requirements.txt

# Step 1: Generate synthetic data
python 1_generate_data.py
# Dates span January 1 through the current date in the current year.

# Step 2: Train models
python 2_train_model.py

# Step 3: Launch Streamlit demo
streamlit run 3_app.py
```

---

## Deploy to Streamlit Cloud

1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Select repo, set main file to `3_app.py`
4. Deploy — live URL in 2 minutes

---

### [`.devcontainer/devcontainer.json`](https://github.com/whitwaithera/Fraud_detection-Kenya/commit/c3c511d4cbfbd113b88c90d86ba1e9b4a35896ab)

Updated October 6, 2026: corrected the development container's app paths to use `3_app.py`.

---

### [`2_train_model.py`](https://github.com/whitwaithera/Fraud_detection-Kenya/commit/6ce8a02)

Updated October 6, 2026: use a separate validation split for early stopping, keeping the test set reserved for final evaluation.

### [`requirements.txt`](https://github.com/whitwaithera/Fraud_detection-Kenya/commit/270bbc6)

Updated October 6, 2026: group dependencies by purpose and add the requested inline comment.

---

## Tech Stack

`Python` `XGBoost` `LightGBM` `SMOTE (imbalanced-learn)` `SHAP` `Streamlit` `Plotly` `Faker` `scikit-learn` `pandas`

---

## Context

Built as part of a data analyst portfolio focused on East African fintech and financial inclusion. The synthetic data models real Kenyan financial patterns — M-Pesa transaction behaviour, KRA PIN formats, county-level geography, and CBK-regulated banking channels.

**Author:** Whitney wanjiru
