"""Feature engineering shared by TRAINING and INFERENCE (prevents train/serve skew).

LEAKAGE POLICY
--------------
A feature may only use information that is known BEFORE the transaction executes:
the transaction type, its amount, the two account balances BEFORE it, and the time of day.
Anything that is an OUTCOME of the transaction (balances afterwards, the simulator's own
fraud flag) is excluded, because it would not exist when screening a transaction in advance
and in PaySim it makes fraud trivially detectable.

Data flow (identical for training and the website):

    PaySim CSV --from_paysim()--\
                                 +--> INPUT_COLUMNS --engineer_features()--> FEATURES --> model
    Website / API form ---------/
"""
import numpy as np
import pandas as pd

TARGET = "isFraud"
ALLOWED_TYPES = ["TRANSFER", "CASH_OUT"]   # the only PaySim types that contain fraud

# What a user / the API supplies for one transaction. All known BEFORE execution.
INPUT_COLUMNS = ["type", "amount", "oldbalanceOrg", "oldbalanceDest", "hour_of_day"]
# PaySim columns needed for training (step is converted to hour_of_day)
PAYSIM_REQUIRED_COLUMNS = ["step", "type", "amount", "oldbalanceOrg", "oldbalanceDest", TARGET]

# PaySim columns that must never become model features, and why (saved into model_config.json)
EXCLUDED_COLUMNS = {
    "newbalanceOrig": "post-transaction sender balance (outcome of the transaction)",
    "newbalanceDest": "post-transaction receiver balance (outcome of the transaction)",
    "isFlaggedFraud": "output of the simulator's own rule-based fraud detector",
    "nameOrig": "account identifier (memorisation, not a general pattern)",
    "nameDest": "account identifier (memorisation, not a general pattern)",
    "step": "absolute simulation clock; only its hour of day (step % 24) is kept",
}

CATEGORICAL_FEATURES = ["type"]
NUMERIC_FEATURES = ["amount", "oldbalanceOrg", "oldbalanceDest", "hour_of_day",
                    "amount_to_orig_balance", "drains_origin", "orig_balance_zero", "dest_balance_zero"]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

# Yes/no features (used to format explanations)
BINARY_FEATURES = {"drains_origin", "orig_balance_zero", "dest_balance_zero",
                   "type_TRANSFER", "type_CASH_OUT"}

# Human-readable names used in SHAP explanations on the website
FEATURE_LABELS = {
    "amount": "Transaction amount",
    "oldbalanceOrg": "Sender balance before transaction",
    "oldbalanceDest": "Receiver balance before transaction",
    "hour_of_day": "Hour of day",
    "amount_to_orig_balance": "Amount as a share of sender balance",
    "drains_origin": "Amount would empty the sender account",
    "orig_balance_zero": "Sender balance is zero",
    "dest_balance_zero": "Receiver balance is zero",
    "type_CASH_OUT": "Type is CASH_OUT",
    "type_TRANSFER": "Type is TRANSFER",
}

# Safety net: fails at import time if someone adds an excluded column as a feature.
assert not (set(FEATURES) | set(INPUT_COLUMNS)) & set(EXCLUDED_COLUMNS), \
    "A post-transaction / identifier column was added to the feature set"
assert set(FEATURE_LABELS) >= set(NUMERIC_FEATURES) | {f"type_{t}" for t in ALLOWED_TYPES}


def from_paysim(df: pd.DataFrame) -> pd.DataFrame:
    """Convert raw PaySim rows into the INPUT_COLUMNS schema (training only)."""
    out = df[["type", "amount", "oldbalanceOrg", "oldbalanceDest"]].copy()
    out["hour_of_day"] = (df["step"] % 24).astype(int)
    return out


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """INPUT_COLUMNS -> model features. Reads ONLY INPUT_COLUMNS; extra columns are ignored."""
    d = df[INPUT_COLUMNS].copy()
    old = d["oldbalanceOrg"].astype(float)
    amt = d["amount"].astype(float)
    d["amount_to_orig_balance"] = np.where(old > 0, np.minimum(amt / np.where(old > 0, old, 1.0), 10.0), 0.0)
    d["drains_origin"] = ((old > 0) & (amt >= old - 0.005)).astype(int)   # 0.5-cent tolerance
    d["orig_balance_zero"] = (old == 0).astype(int)
    d["dest_balance_zero"] = (d["oldbalanceDest"].astype(float) == 0).astype(int)
    return d[FEATURES]
