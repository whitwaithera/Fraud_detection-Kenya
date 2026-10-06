"""
Script 1: 1_generate_data.py
Generates a realistic synthetic Kenyan financial fraud dataset combining:
  - M-Pesa mobile money transactions
  - KRA tax filing records
  - Bank account transactions

Fraud is injected at ~1.5% rate to simulate real-world class imbalance.

Usage:
    python 1_generate_data.py
"""

import pandas as pd
import numpy as np
import random
import os
from faker import Faker
from datetime import datetime, timedelta

fake = Faker("en_GB")   # closest to Kenyan English names
Faker.seed(42)
np.random.seed(42)
random.seed(42)

OUTPUT_DIR  = "data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "kenya_fraud_dataset.csv")
N_RECORDS   = 50000   # total transactions
FRAUD_RATE  = 0.015   # 1.5% fraud

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Kenyan context helpers ────────────────────────────────────────────────────
KENYAN_FIRST_NAMES = [
    "Ammon","Brian","Faith","Grace","Kevin","Diana","Peter","Mary","James","Rose",
    "John","Alice","David","Sarah","Samuel","Nancy","Joseph","Caroline","George",
    "Esther","Michael","Lilian","Paul","Beatrice","Patrick","Mercy","Francis",
    "Winnie","Daniel","Agnes","Charles","Purity","Philip","Josephine","Simon",
    "Wanjiku","Moses","Akinyi","Ochieng","Njeri","Kimani","Wambui","Kariuki",
    "Atieno","Kamau","Adhiambo","Mutua","Nyambura","Odhiambo","Muthoni"
]

KENYAN_LAST_NAMES = [
    "Mwangi","Otieno","Kamau","Odhiambo","Kariuki","Achieng","Mutua","Wanjiku",
    "Kimani","Adhiambo","Ngugi","Omondi","Njoroge","Awino","Gitau","Nyambura",
    "Kenyatta","Odinga","Waweru","Auma","Maina","Ogola","Githinji","Anyango",
    "Muturi","Onyango","Gacheru","Akinyi","Njeru","Owino","Karanja","Makena",
    "Kiprotich","Chebet","Rono","Korir","Ngetich","Chepkemoi","Bett","Rotich"
]

COUNTIES = [
    "Nairobi","Mombasa","Kisumu","Nakuru","Eldoret","Thika","Malindi",
    "Kitale","Garissa","Nyeri","Machakos","Meru","Kericho","Kakamega","Bungoma"
]

MPESA_TRANSACTION_TYPES = [
    "Send Money","Buy Goods","Pay Bill","Withdraw Cash","Deposit Cash",
    "Buy Airtime","Lipa Na M-Pesa","Till Number Payment","Paybill"
]

BANK_TRANSACTION_TYPES = [
    "Wire Transfer","ATM Withdrawal","POS Purchase","Online Transfer",
    "Cheque Deposit","RTGS Transfer","EFT Transfer","Standing Order"
]

KRA_FILING_TYPES = [
    "VAT Return","Income Tax","Corporate Tax","PAYE","Withholding Tax",
    "Capital Gains","Excise Duty","Customs Duty"
]

BUSINESSES = [
    "Safaricom","Kenya Power","KPLC","Nairobi Water","KCB Bank",
    "Equity Bank","Co-operative Bank","Family Bank","NCBA","Stanbic",
    "Airtel Money","Faulu Microfinance","KWFT","Tala","Branch"
]


def generate_kra_pin() -> str:
    """Generate a realistic KRA PIN format: A000000000X"""
    letters = "ABCDEFGHJKLMNPRSTUVWXYZ"
    return (f"{random.choice(letters)}"
            f"{random.randint(100000000, 999999999)}"
            f"{random.choice(letters)}")


def generate_phone() -> str:
    """Generate Kenyan phone number."""
    prefixes = ["0700","0701","0710","0711","0712","0720","0721","0722",
                "0729","0730","0740","0745","0750","0757","0768","0790"]
    return f"{random.choice(prefixes)}{random.randint(100000, 999999)}"


def generate_account_number() -> str:
    return f"{random.randint(1000000000, 9999999999)}"


def build_customer_profiles(n_customers: int) -> pd.DataFrame:
    """Build customer baseline profiles for realistic behaviour modelling."""
    profiles = []
    for _ in range(n_customers):
        fname = random.choice(KENYAN_FIRST_NAMES)
        lname = random.choice(KENYAN_LAST_NAMES)
        avg_txn = np.random.lognormal(mean=9.5, sigma=1.2)   # KES, log-normal
        profiles.append({
            "customer_id":      f"CUST{random.randint(100000, 999999)}",
            "full_name":        f"{fname} {lname}",
            "kra_pin":          generate_kra_pin(),
            "phone":            generate_phone(),
            "county":           random.choice(COUNTIES),
            "account_number":   generate_account_number(),
            "avg_txn_amount":   avg_txn,
            "account_age_days": random.randint(30, 3650),
            "monthly_income":   np.random.lognormal(mean=10.5, sigma=0.8),
        })
    return pd.DataFrame(profiles)


def inject_fraud_patterns(row: dict, fraud_type: str) -> dict:
    """
    Inject realistic fraud patterns based on type.
    Returns modified row with fraud signals.
    """
    if fraud_type == "velocity":
        # Multiple rapid transactions — amount normal but high frequency
        row["amount"]              = row["avg_txn_amount"] * random.uniform(0.8, 1.5)
        row["transactions_last_hour"] = random.randint(8, 25)
        row["transactions_last_day"]  = random.randint(30, 80)
        row["ratio_to_avg_amount"] = random.uniform(0.7, 1.8)

    elif fraud_type == "large_unusual":
        # Single large transaction far above customer norm
        row["amount"]              = row["avg_txn_amount"] * random.uniform(12, 50)
        row["transactions_last_hour"] = random.randint(1, 3)
        row["ratio_to_avg_amount"] = random.uniform(12, 50)
        row["is_new_recipient"]    = 1
        row["recipient_txn_history"] = random.randint(0, 2)

    elif fraud_type == "account_takeover":
        # Login from new device + location + large transfer
        row["amount"]              = row["avg_txn_amount"] * random.uniform(5, 20)
        row["new_device"]          = 1
        row["location_mismatch"]   = 1
        row["hour_of_day"]         = random.randint(0, 4)   # unusual hours
        row["ratio_to_avg_amount"] = random.uniform(5, 20)

    elif fraud_type == "kra_evasion":
        # Filed amount far below expected based on business activity
        row["amount"]              = row["monthly_income"] * random.uniform(0.02, 0.08)
        row["declared_vs_expected_ratio"] = random.uniform(0.05, 0.15)
        row["filing_days_late"]    = random.randint(30, 365)

    elif fraud_type == "structuring":
        # Many just-below-threshold transactions (anti-money laundering)
        row["amount"]              = random.uniform(95000, 99999)   # just under KES 100k
        row["transactions_last_day"]  = random.randint(5, 15)
        row["round_amount"]        = 0
        row["transactions_last_hour"] = random.randint(3, 8)

    return row


def generate_transaction(customer: dict, txn_date: datetime,
                          is_fraud: bool, fraud_type: str = None) -> dict:
    """Generate a single transaction record."""

    # Determine transaction type based on weighted mix
    txn_category = random.choices(
        ["mpesa", "bank", "kra"],
        weights=[0.55, 0.30, 0.15]
    )[0]

    if txn_category == "mpesa":
        txn_type = random.choice(MPESA_TRANSACTION_TYPES)
        channel  = "M-Pesa"
    elif txn_category == "bank":
        txn_type = random.choice(BANK_TRANSACTION_TYPES)
        channel  = random.choice(["Mobile Banking","ATM","Branch","Online"])
    else:
        txn_type = random.choice(KRA_FILING_TYPES)
        channel  = "iTax Portal"

    # Normal transaction amount — log-normal around customer average
    amount = abs(np.random.normal(
        loc=customer["avg_txn_amount"],
        scale=customer["avg_txn_amount"] * 0.4
    ))
    amount = max(amount, 10)   # minimum KES 10

    hour   = random.choices(
        range(24),
        weights=[1,1,1,1,1,2,4,8,10,10,9,9,9,9,8,8,7,6,5,4,3,2,2,1]
    )[0]

    row = {
        # Identity
        "transaction_id":         f"TXN{random.randint(10000000, 99999999)}",
        "customer_id":            customer["customer_id"],
        "kra_pin":                customer["kra_pin"],
        "phone":                  customer["phone"],
        "county":                 customer["county"],
        "account_age_days":       customer["account_age_days"],

        # Transaction details
        "transaction_type":       txn_type,
        "transaction_category":   txn_category,
        "channel":                channel,
        "amount":                 round(amount, 2),
        "currency":               "KES",
        "recipient_account":      generate_account_number(),
        "recipient_business":     random.choice(BUSINESSES) if random.random() > 0.5 else "",

        # Temporal features
        "timestamp":              txn_date.strftime("%Y-%m-%d %H:%M:%S"),
        "hour_of_day":            hour,
        "day_of_week":            txn_date.weekday(),
        "is_weekend":             int(txn_date.weekday() >= 5),
        "month":                  txn_date.month,

        # Behavioural features
        "avg_txn_amount":         round(customer["avg_txn_amount"], 2),
        "monthly_income":         round(customer["monthly_income"], 2),
        "ratio_to_avg_amount":    round(amount / max(customer["avg_txn_amount"], 1), 4),
        "transactions_last_hour": random.randint(0, 3),
        "transactions_last_day":  random.randint(1, 12),
        "transactions_last_week": random.randint(3, 40),
        "is_new_recipient":       int(random.random() < 0.15),
        "recipient_txn_history":  random.randint(0, 50),
        "new_device":             int(random.random() < 0.05),
        "location_mismatch":      int(random.random() < 0.03),
        "round_amount":           int(amount % 100 == 0),
        "filing_days_late":       0,
        "declared_vs_expected_ratio": 1.0,

        # Label
        "is_fraud":               int(is_fraud),
        "fraud_type":             fraud_type if is_fraud else "none",
    }

    # Inject fraud patterns
    if is_fraud and fraud_type:
        row = inject_fraud_patterns(row, fraud_type)

    return row


def main():
    print(f"\n{'='*55}")
    print("  Kenya Financial Fraud — Synthetic Data Generator")
    print(f"  {N_RECORDS:,} transactions | {FRAUD_RATE*100:.1f}% fraud rate")
    print(f"{'='*55}\n")

    # Build customer pool
    n_customers = N_RECORDS // 20   # avg 20 txns per customer
    print(f"[1/3] Generating {n_customers:,} customer profiles...")
    customers_df = build_customer_profiles(n_customers)
    customers    = customers_df.to_dict("records")

    # Determine fraud counts per type
    n_fraud  = int(N_RECORDS * FRAUD_RATE)
    n_legit  = N_RECORDS - n_fraud

    fraud_types = ["velocity","large_unusual","account_takeover",
                   "kra_evasion","structuring"]
    fraud_type_dist = np.random.choice(fraud_types, size=n_fraud,
                                        p=[0.25, 0.25, 0.20, 0.15, 0.15])

    print(f"[2/3] Generating transactions...")
    print(f"      Legitimate : {n_legit:,}")
    print(f"      Fraudulent : {n_fraud:,}")

    records = []
    current_date = datetime.now()
    start_date = datetime(current_date.year, 1, 1)
    available_days = (current_date.date() - start_date.date()).days

    # Legitimate transactions
    for _ in range(n_legit):
        customer = random.choice(customers)
        txn_date = start_date + timedelta(
            days=random.randint(0, available_days),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59)
        )
        records.append(generate_transaction(customer, txn_date, is_fraud=False))

    # Fraudulent transactions
    for fraud_type in fraud_type_dist:
        customer = random.choice(customers)
        txn_date = start_date + timedelta(
            days=random.randint(0, available_days),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59)
        )
        records.append(generate_transaction(customer, txn_date,
                                            is_fraud=True, fraud_type=fraud_type))

    print(f"[3/3] Saving dataset...")
    df = pd.DataFrame(records).sample(frac=1, random_state=42).reset_index(drop=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"\nSaved {len(df):,} records to {OUTPUT_FILE}")
    print(f"\nFraud breakdown:")
    print(df[df["is_fraud"]==1]["fraud_type"].value_counts().to_string())
    print(f"\nTransaction category mix:")
    print(df["transaction_category"].value_counts().to_string())
    print(f"\nAmount stats (KES):")
    print(df["amount"].describe().round(2).to_string())


if __name__ == "__main__":
    main()
