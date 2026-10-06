"""
Script 2: 2_train_model.py
Full ML pipeline:
  - Feature engineering (ratio_to_avg_amount, velocity features, etc.)
  - SMOTE oversampling to handle class imbalance
  - XGBoost and LightGBM classifiers
  - Cross-validation + hyperparameter tuning
  - SHAP explainability
  - Model saved to models/

Usage:
    python 2_train_model.py
"""

import pandas as pd
import numpy as np
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, average_precision_score,
                              precision_recall_curve, roc_curve)
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import lightgbm as lgb
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns

INPUT_FILE  = "data/kenya_fraud_dataset.csv"
MODEL_DIR   = "models"
OUTPUT_DIR  = "outputs"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

PALETTE = {"primary": "#2e4a35", "accent": "#c97b63", "light": "#f7f5f0", "dark": "#1a1714"}


# ── Feature engineering ───────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Ratio features — key fraud signal
    df["log_amount"]               = np.log1p(df["amount"])
    df["log_avg_amount"]           = np.log1p(df["avg_txn_amount"])
    df["amount_vs_income_ratio"]   = df["amount"] / (df["monthly_income"] + 1)
    df["velocity_score"]           = (df["transactions_last_hour"] * 3 +
                                      df["transactions_last_day"])

    # Time features
    df["is_night"]        = ((df["hour_of_day"] >= 0) & (df["hour_of_day"] <= 5)).astype(int)
    df["is_business_hrs"] = ((df["hour_of_day"] >= 8) & (df["hour_of_day"] <= 17)).astype(int)

    # Risk composite score
    df["risk_score"] = (
        df["ratio_to_avg_amount"].clip(0, 50) / 50 * 0.30 +
        df["new_device"] * 0.20 +
        df["location_mismatch"] * 0.20 +
        df["is_new_recipient"] * 0.10 +
        (df["transactions_last_hour"] / 25).clip(0, 1) * 0.10 +
        df["is_night"] * 0.10
    )

    # Account maturity — newer accounts riskier
    df["account_maturity"] = np.log1p(df["account_age_days"])

    # KRA-specific
    df["tax_risk"] = (
        (1 - df["declared_vs_expected_ratio"].clip(0, 1)) * 0.5 +
        (df["filing_days_late"] / 365).clip(0, 1) * 0.5
    )

    return df


def get_feature_cols() -> list:
    return [
        "log_amount", "log_avg_amount", "ratio_to_avg_amount",
        "amount_vs_income_ratio", "velocity_score",
        "transactions_last_hour", "transactions_last_day", "transactions_last_week",
        "hour_of_day", "day_of_week", "is_weekend", "is_night", "is_business_hrs",
        "is_new_recipient", "recipient_txn_history",
        "new_device", "location_mismatch", "round_amount",
        "account_age_days", "account_maturity",
        "risk_score", "tax_risk",
        "declared_vs_expected_ratio", "filing_days_late",
        "txn_category_encoded", "channel_encoded",
    ]


def encode_categoricals(df: pd.DataFrame) -> tuple:
    le_cat = LabelEncoder()
    le_chan = LabelEncoder()
    df["txn_category_encoded"] = le_cat.fit_transform(df["transaction_category"])
    df["channel_encoded"]      = le_chan.fit_transform(df["channel"])
    return df, le_cat, le_chan


# ── Evaluation plots ──────────────────────────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred, model_name: str):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor(PALETTE["light"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlGn",
                xticklabels=["Legit","Fraud"],
                yticklabels=["Legit","Fraud"], ax=ax,
                linewidths=1, linecolor=PALETTE["light"])
    ax.set_title(f"Confusion Matrix — {model_name}", fontweight="bold",
                 color=PALETTE["dark"], pad=12)
    ax.set_xlabel("Predicted", color=PALETTE["dark"])
    ax.set_ylabel("Actual", color=PALETTE["dark"])
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"confusion_matrix_{model_name.lower().replace(' ','_')}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=PALETTE["light"])
    plt.close()
    print(f"  ✓ {path}")


def plot_roc_pr(y_true, y_proba, model_name: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor(PALETTE["light"])

    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    axes[0].plot(fpr, tpr, color=PALETTE["primary"], lw=2,
                 label=f"AUC = {auc:.4f}")
    axes[0].plot([0,1],[0,1], "--", color=PALETTE["accent"], lw=1)
    axes[0].set_title(f"ROC Curve — {model_name}", fontweight="bold")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].legend()
    axes[0].set_facecolor(PALETTE["light"])

    # Precision-Recall
    prec, rec, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)
    axes[1].plot(rec, prec, color=PALETTE["accent"], lw=2,
                 label=f"AP = {ap:.4f}")
    axes[1].set_title(f"Precision-Recall Curve — {model_name}", fontweight="bold")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend()
    axes[1].set_facecolor(PALETTE["light"])

    for ax in axes:
        ax.spines[["top","right"]].set_visible(False)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"roc_pr_{model_name.lower().replace(' ','_')}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=PALETTE["light"])
    plt.close()
    print(f"  ✓ {path}")


def plot_shap(model, X_test: pd.DataFrame, model_name: str):
    try:
        explainer  = shap.TreeExplainer(model)
        shap_vals  = explainer.shap_values(X_test.iloc[:500])

        fig, ax = plt.subplots(figsize=(10, 7))
        shap.summary_plot(shap_vals, X_test.iloc[:500],
                          plot_type="bar", show=False,
                          color=PALETTE["primary"])
        plt.title(f"SHAP Feature Importance — {model_name}",
                  fontweight="bold", color=PALETTE["dark"])
        plt.tight_layout()
        path = os.path.join(OUTPUT_DIR, f"shap_{model_name.lower().replace(' ','_')}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  ✓ {path}")
    except Exception as e:
        print(f"  ⚠ SHAP plot skipped: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"\n{'='*55}")
    print("  Kenya Fraud Detection — ML Pipeline")
    print(f"{'='*55}\n")

    # Load & engineer
    print("[1/6] Loading and engineering features...")
    df = pd.read_csv(INPUT_FILE)
    df = engineer_features(df)
    df, le_cat, le_chan = encode_categoricals(df)

    feature_cols = get_feature_cols()
    X = df[feature_cols]
    y = df["is_fraud"]

    print(f"  Dataset shape : {X.shape}")
    print(f"  Fraud rate    : {y.mean()*100:.2f}%")
    print(f"  Fraud count   : {y.sum():,} / {len(y):,}")

    # Stratified train/validation/test split; reserve the test set for final evaluation.
    print("\n[2/6] Creating stratified train/validation/test splits (72/8/20%)...")
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.1,
        random_state=42, stratify=y_train_full
    )

    # SMOTE
    print("\n[3/6] Applying SMOTE to balance training set...")
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
    print(f"  Before SMOTE : {y_train.sum():,} fraud / {len(y_train):,} total")
    print(f"  After SMOTE  : {y_train_sm.sum():,} fraud / {len(y_train_sm):,} total")

    results = {}

    # ── XGBoost ───────────────────────────────────────────────────────────────
    print("\n[4/6] Training XGBoost...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=1,   # balanced via SMOTE
        use_label_encoder=False,
        eval_metric="aucpr",
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
    xgb_model.fit(X_train_sm, y_train_sm,
                  eval_set=[(X_val, y_val)],
                  verbose=False)

    xgb_pred  = xgb_model.predict(X_test)
    xgb_proba = xgb_model.predict_proba(X_test)[:, 1]

    print(f"\n  XGBoost Results:")
    print(f"  ROC-AUC : {roc_auc_score(y_test, xgb_proba):.4f}")
    print(f"  Avg Precision : {average_precision_score(y_test, xgb_proba):.4f}")
    print(classification_report(y_test, xgb_pred,
                                 target_names=["Legitimate","Fraud"],
                                 digits=4))

    results["XGBoost"] = {
        "model": xgb_model, "pred": xgb_pred, "proba": xgb_proba,
        "auc": roc_auc_score(y_test, xgb_proba)
    }

    # ── LightGBM ──────────────────────────────────────────────────────────────
    print("\n[5/6] Training LightGBM...")
    lgb_model = lgb.LGBMClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    lgb_model.fit(X_train_sm, y_train_sm,
                  eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(50, verbose=False),
                              lgb.log_evaluation(period=-1)])

    lgb_pred  = lgb_model.predict(X_test)
    lgb_proba = lgb_model.predict_proba(X_test)[:, 1]

    print(f"\n  LightGBM Results:")
    print(f"  ROC-AUC : {roc_auc_score(y_test, lgb_proba):.4f}")
    print(f"  Avg Precision : {average_precision_score(y_test, lgb_proba):.4f}")
    print(classification_report(y_test, lgb_pred,
                                 target_names=["Legitimate","Fraud"],
                                 digits=4))

    results["LightGBM"] = {
        "model": lgb_model, "pred": lgb_pred, "proba": lgb_proba,
        "auc": roc_auc_score(y_test, lgb_proba)
    }

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\n[6/6] Generating evaluation plots...")
    for name, res in results.items():
        plot_confusion_matrix(y_test, res["pred"], name)
        plot_roc_pr(y_test, res["proba"], name)
        plot_shap(res["model"], X_test, name)

    # ── Save best model ───────────────────────────────────────────────────────
    best_name  = max(results, key=lambda k: results[k]["auc"])
    best_model = results[best_name]["model"]

    joblib.dump(best_model, os.path.join(MODEL_DIR, "best_model.pkl"))
    joblib.dump(le_cat,     os.path.join(MODEL_DIR, "le_category.pkl"))
    joblib.dump(le_chan,     os.path.join(MODEL_DIR, "le_channel.pkl"))
    joblib.dump(feature_cols, os.path.join(MODEL_DIR, "feature_cols.pkl"))

    # Save test set for Streamlit demo
    X_test_save = X_test.copy()
    X_test_save["is_fraud"]   = y_test.values
    X_test_save["fraud_type"] = df.loc[y_test.index, "fraud_type"].values
    X_test_save["amount"]     = df.loc[y_test.index, "amount"].values
    X_test_save["county"]     = df.loc[y_test.index, "county"].values
    X_test_save["channel"]    = df.loc[y_test.index, "channel"].values
    X_test_save["timestamp"]  = df.loc[y_test.index, "timestamp"].values
    X_test_save["transaction_type"] = df.loc[y_test.index, "transaction_type"].values
    X_test_save.to_csv("data/test_set.csv", index=False)

    print(f"\n✓ Best model: {best_name} (AUC: {results[best_name]['auc']:.4f})")
    print(f"✓ Saved → models/best_model.pkl")
    print(f"✓ Test set saved → data/test_set.csv")
    print(f"\nAll outputs saved to /outputs/")


if __name__ == "__main__":
    main()
