"""
Script 3: 3_app.py
Streamlit fraud detection demo — recruiters can:
  - Browse flagged transactions
  - Run a live fraud scan on new transactions
  - See SHAP explainability per prediction
  - View model performance metrics

Usage:
    streamlit run 3_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Kenya Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
h1,h2,h3 { font-family: 'DM Serif Display', serif !important; }

.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"],
[data-testid="stHeader"] { background-color: #f59e0b !important; }

.dash-header {
    background: linear-gradient(135deg, #1a1714 0%, #2e4a35 100%);
    color: #e8e3da; padding: 1.8rem 2rem; border-radius: 14px;
    margin-bottom: 1.6rem; display: flex; justify-content: space-between; align-items: center;
}
.dash-header h1 { color: #e8e3da; margin: 0; font-size: 1.7rem; }
.dash-header .sub { color: #a0c8a8; font-size: 0.85rem; margin-top: 4px; }

[data-testid="metric-container"] {
    background: #f7f5f0; border: 0.5px solid #e8e3da;
    border-radius: 12px; padding: 1rem 1.2rem;
}
[data-testid="stMetricValue"] { font-size: 1.8rem; font-weight: 500; }

.fraud-badge {
    background: #fde8e8; color: #b94a48; padding: 3px 10px;
    border-radius: 999px; font-size: 0.75rem; font-weight: 500;
}
.legit-badge {
    background: #d4edda; color: #1a5c2a; padding: 3px 10px;
    border-radius: 999px; font-size: 0.75rem; font-weight: 500;
}
.risk-high   { color: #b94a48; font-weight: 600; }
.risk-medium { color: #c97b63; font-weight: 600; }
.risk-low    { color: #2e7d52; font-weight: 600; }

[data-testid="stSidebar"] { background: #1a1714; }
[data-testid="stSidebar"] * { color: #e8e3da !important; }

.section-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.05rem; color: #1a1714;
    border-bottom: 2px solid #e8e3da;
    padding-bottom: 6px; margin: 1.2rem 0 1rem;
}
</style>
""", unsafe_allow_html=True)


# ── Load model & data ─────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model        = joblib.load("models/best_model.pkl")
    feature_cols = joblib.load("models/feature_cols.pkl")
    category_encoder = joblib.load("models/le_category.pkl")
    channel_encoder  = joblib.load("models/le_channel.pkl")
    return model, feature_cols, category_encoder, channel_encoder

@st.cache_data
def load_test_data():
    return pd.read_csv("data/test_set.csv")

model, feature_cols, category_encoder, channel_encoder = load_model()
test_df = load_test_data()

# Add predictions to test set
test_df["fraud_probability"] = model.predict_proba(test_df[feature_cols])[:, 1]
test_df["risk_level"] = pd.cut(
    test_df["fraud_probability"],
    bins=[0, 0.3, 0.6, 1.0],
    labels=["Low", "Medium", "High"]
)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ FraudShield Kenya")
    st.markdown("---")
    st.markdown("**Filter transactions**")
    risk_filter = st.multiselect(
        "Risk level", ["High","Medium","Low"],
        default=["High","Medium"]
    )
    channel_filter = st.multiselect(
        "Channel", test_df["channel"].unique().tolist(),
        default=test_df["channel"].unique().tolist()
    )
    threshold = st.slider("Fraud probability threshold", 0.1, 0.9, 0.5, 0.05)
    st.markdown("---")
    st.markdown('<p style="font-size:0.72rem;color:#a09890;">FraudShield v1.0 · XGBoost + SMOTE · Kenya Financial Data</p>',
                unsafe_allow_html=True)


# ── Apply filters ─────────────────────────────────────────────────────────────
filtered = test_df[
    (test_df["risk_level"].isin(risk_filter)) &
    (test_df["channel"].isin(channel_filter))
]
flagged = filtered[filtered["fraud_probability"] >= threshold]
flagged_rate = len(flagged) / len(filtered) * 100 if not filtered.empty else 0


# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="dash-header">
  <div>
    <h1>🛡️ Kenya Financial Fraud Detection</h1>
    <div class="sub">Real-time fraud scoring · M-Pesa · Bank · KRA · Powered by XGBoost + SMOTE</div>
  </div>
  <div style="text-align:right;color:#a0c8a8;font-size:0.82rem;">
    {len(filtered):,} transactions match filters<br>
    {len(flagged):,} flagged at {threshold:.0%} threshold
  </div>
</div>
""", unsafe_allow_html=True)


# ── KPI Cards ─────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Transactions",  f"{len(filtered):,}")
c2.metric("Flagged as Fraud",    f"{len(flagged):,}",
          f"{flagged_rate:.1f}% of filtered")
c3.metric(
    "Avg Fraud Probability",
    f"{filtered['fraud_probability'].mean():.3f}" if not filtered.empty else "—"
)
c4.metric("High Risk Transactions",
          f"{len(filtered[filtered['risk_level']=='High']):,}",
          "above 60% probability")


# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔍 Flagged Transactions", "📊 Analytics", "⚡ Live Scan"])


# ── Tab 1: Flagged transactions ───────────────────────────────────────────────
with tab1:
    st.markdown('<div class="section-title">Transactions flagged for review</div>',
                unsafe_allow_html=True)

    display_cols = ["timestamp","transaction_type","channel","county",
                    "amount","fraud_probability","risk_level","is_fraud"]
    flag_display = flagged[display_cols].sort_values(
        "fraud_probability", ascending=False
    ).head(200).copy()
    flag_display["amount"] = flag_display["amount"].apply(lambda x: f"KES {x:,.0f}")
    flag_display["fraud_probability"] = flag_display["fraud_probability"].apply(
        lambda x: f"{x:.1%}"
    )
    flag_display["actual"] = flag_display["is_fraud"].map({1:"✓ Fraud", 0:"Legitimate"})
    flag_display = flag_display.drop("is_fraud", axis=1)

    st.dataframe(flag_display, use_container_width=True, height=400)
    if flagged.empty:
        st.info("No transactions match the selected filters and fraud probability threshold.")

    csv = flagged.to_csv(index=False).encode("utf-8")
    st.download_button("⬇ Export flagged transactions", csv,
                       "flagged_transactions.csv", "text/csv")


# ── Tab 2: Analytics ───────────────────────────────────────────────────────────
with tab2:
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="section-title">Fraud probability distribution</div>',
                    unsafe_allow_html=True)
        fig = px.histogram(
            filtered, x="fraud_probability", nbins=50,
            color_discrete_sequence=["#2e4a35"],
            labels={"fraud_probability": "Fraud Probability"}
        )
        fig.add_vline(x=threshold, line_dash="dash", line_color="#c97b63",
                      annotation_text=f"Threshold: {threshold:.0%}")
        fig.update_layout(
            plot_bgcolor="#f7f5f0", paper_bgcolor="#f7f5f0",
            height=300, margin=dict(t=20, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-title">Fraud by transaction channel</div>',
                    unsafe_allow_html=True)
        channel_fraud = (filtered.assign(
                            flagged=filtered["fraud_probability"] >= threshold
                         ).groupby("channel")["flagged"]
                         .agg(["sum","count"])
                         .reset_index()
                         .rename(columns={"sum":"fraud","count":"total"}))
        channel_fraud["rate"] = channel_fraud["fraud"] / channel_fraud["total"] * 100
        fig2 = px.bar(channel_fraud, x="channel", y="rate",
                      color="rate", color_continuous_scale="YlGn",
                      labels={"rate": "Fraud Rate %", "channel": "Channel"})
        fig2.update_layout(plot_bgcolor="#f7f5f0", paper_bgcolor="#f7f5f0",
                           height=300, margin=dict(t=20, b=20), showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    col_c, col_d = st.columns(2)

    with col_c:
        st.markdown('<div class="section-title">Flagged transactions by county</div>',
                    unsafe_allow_html=True)
        county_fraud = (flagged.groupby("county").size()
                        .reset_index(name="count")
                        .sort_values("count", ascending=False))
        fig3 = px.bar(county_fraud, x="count", y="county", orientation="h",
                      color_discrete_sequence=["#c97b63"])
        fig3.update_layout(plot_bgcolor="#f7f5f0", paper_bgcolor="#f7f5f0",
                           height=350, margin=dict(t=20))
        st.plotly_chart(fig3, use_container_width=True)

    with col_d:
        st.markdown('<div class="section-title">Fraud amount distribution</div>',
                    unsafe_allow_html=True)
        fraud_amounts = filtered[filtered["is_fraud"]==1]["amount"]
        legit_candidates = filtered[filtered["is_fraud"]==0]["amount"]
        legit_amounts = legit_candidates.sample(
            min(len(fraud_amounts) * 3, len(legit_candidates), 3000)
        )
        fig4 = go.Figure()
        fig4.add_trace(go.Box(y=np.log1p(fraud_amounts), name="Fraud",
                              marker_color="#b94a48"))
        fig4.add_trace(go.Box(y=np.log1p(legit_amounts), name="Legitimate",
                              marker_color="#2e4a35"))
        fig4.update_layout(
            plot_bgcolor="#f7f5f0", paper_bgcolor="#f7f5f0",
            height=350, margin=dict(t=20),
            yaxis_title="log(Amount KES)"
        )
        st.plotly_chart(fig4, use_container_width=True)


# ── Tab 3: Live Scan ───────────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-title">⚡ Run a live fraud scan on a new transaction</div>',
                unsafe_allow_html=True)
    st.markdown("Enter transaction details below and click **Run Fraud Scan** to get a real-time prediction.")

    col1, col2, col3 = st.columns(3)
    with col1:
        amount          = st.number_input("Transaction Amount (KES)", 100, 5000000, 15000, step=500)
        avg_txn_amount  = st.number_input("Customer Avg Transaction (KES)", 100, 500000, 8000, step=500)
        monthly_income  = st.number_input("Customer Monthly Income (KES)", 5000, 2000000, 50000, step=1000)
        account_age     = st.number_input("Account Age (days)", 0, 5000, 365)

    with col2:
        hour_of_day     = st.slider("Hour of Day", 0, 23, 14)
        day_of_week     = st.selectbox("Day of Week", ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"])
        txn_per_hour    = st.number_input("Transactions in Last Hour", 0, 30, 1)
        txn_per_day     = st.number_input("Transactions Today", 0, 100, 5)

    with col3:
        new_device      = st.toggle("New Device Detected", False)
        location_mis    = st.toggle("Location Mismatch", False)
        new_recipient   = st.toggle("New Recipient", False)
        channel         = st.selectbox("Channel", ["M-Pesa","Mobile Banking","ATM","Online","Branch","iTax Portal"])
        txn_category    = st.selectbox("Type", ["mpesa","bank","kra"])
        filing_late     = st.number_input("Filing Days Late (KRA)", 0, 365, 0)
        declared_ratio  = st.number_input("Declared vs Expected Ratio", 0.0, 2.0, 1.0, step=0.05)

    if st.button("⚡ Run Fraud Scan", type="primary", use_container_width=True):
        # Build feature vector
        dow_map = {"Mon":0,"Tue":1,"Wed":2,"Thu":3,"Fri":4,"Sat":5,"Sun":6}
        ratio   = amount / max(avg_txn_amount, 1)

        input_dict = {
            "log_amount":                  np.log1p(amount),
            "log_avg_amount":              np.log1p(avg_txn_amount),
            "ratio_to_avg_amount":         ratio,
            "amount_vs_income_ratio":      amount / max(monthly_income, 1),
            "velocity_score":              txn_per_hour * 3 + txn_per_day,
            "transactions_last_hour":      txn_per_hour,
            "transactions_last_day":       txn_per_day,
            "transactions_last_week":      txn_per_day * 7,
            "hour_of_day":                 hour_of_day,
            "day_of_week":                 dow_map[day_of_week],
            "is_weekend":                  int(dow_map[day_of_week] >= 5),
            "is_night":                    int(hour_of_day <= 5),
            "is_business_hrs":             int(8 <= hour_of_day <= 17),
            "is_new_recipient":            int(new_recipient),
            "recipient_txn_history":       0,
            "new_device":                  int(new_device),
            "location_mismatch":           int(location_mis),
            "round_amount":                int(amount % 100 == 0),
            "account_age_days":            account_age,
            "account_maturity":            np.log1p(account_age),
            "risk_score": (
                min(ratio/50, 1) * 0.30 +
                int(new_device) * 0.20 +
                int(location_mis) * 0.20 +
                int(new_recipient) * 0.10 +
                min(txn_per_hour/25, 1) * 0.10 +
                int(hour_of_day <= 5) * 0.10
            ),
            "tax_risk":                    (1 - min(declared_ratio, 1)) * 0.5 + min(filing_late/365, 1) * 0.5,
            "declared_vs_expected_ratio":  declared_ratio,
            "filing_days_late":            filing_late,
            "txn_category_encoded":        int(category_encoder.transform([txn_category])[0]),
            "channel_encoded":              int(channel_encoder.transform([channel])[0]),
        }

        input_df = pd.DataFrame([input_dict])[feature_cols]
        probabilities = model.predict_proba(input_df)[0]
        fraud_class_index = list(model.classes_).index(1)
        proba = probabilities[fraud_class_index]

        st.markdown("---")
        r1, r2, r3 = st.columns(3)

        if proba >= 0.7:
            risk_label = "🔴 HIGH RISK"
            risk_color = "#b94a48"
            recommendation = "Block transaction and alert compliance team immediately."
        elif proba >= 0.4:
            risk_label = "🟡 MEDIUM RISK"
            risk_color = "#c97b63"
            recommendation = "Flag for manual review before processing."
        else:
            risk_label = "🟢 LOW RISK"
            risk_color = "#2e7d52"
            recommendation = "Transaction appears legitimate. Proceed normally."

        r1.metric("Fraud Probability", f"{proba:.1%}")
        r2.metric("Risk Level", risk_label)
        r3.metric("Recommendation", recommendation)

        # Gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=proba * 100,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "Fraud Risk Score", "font": {"size": 18}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar":  {"color": risk_color},
                "steps": [
                    {"range": [0, 40],  "color": "#d4edda"},
                    {"range": [40, 70], "color": "#fff3cd"},
                    {"range": [70, 100],"color": "#fde8e8"},
                ],
                "threshold": {
                    "line": {"color": "#1a1714", "width": 3},
                    "thickness": 0.75,
                    "value": threshold * 100
                }
            }
        ))
        fig_gauge.update_layout(
            height=280, paper_bgcolor="#f7f5f0",
            margin=dict(t=40, b=10)
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Key signals
        st.markdown("**Key fraud signals detected:**")
        signals = []
        if ratio > 10:    signals.append(f"⚠ Amount is **{ratio:.0f}x** above customer average")
        if new_device:    signals.append("⚠ **New device** detected for this account")
        if location_mis:  signals.append("⚠ **Location mismatch** from registered address")
        if txn_per_hour > 5: signals.append(f"⚠ **{txn_per_hour} transactions** in the last hour")
        if hour_of_day <= 5: signals.append(f"⚠ Transaction at **{hour_of_day:02d}:00** (unusual hour)")
        if filing_late > 60: signals.append(f"⚠ KRA filing **{filing_late} days late**")
        if declared_ratio < 0.3: signals.append(f"⚠ Declared amount is only **{declared_ratio:.0%}** of expected")

        if signals:
            for s in signals:
                st.markdown(f"- {s}")
        else:
            st.markdown("- ✓ No significant fraud signals detected")


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<p style="font-size:0.75rem;color:#a09890;text-align:center;">'
    'FraudShield Kenya · XGBoost + SMOTE · Synthetic data for demonstration · '
    'Built by <span style="color:#4a7c6f;">Whitney wanjiru</span></p>',
    unsafe_allow_html=True
)
