"""Exploratory Data Analysis for PaySim. Usage: python -m ml.eda --data datasets/<file>.csv"""
import argparse, json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default=str(ROOT / "static" / "img" / "ml"))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(a.data)
    summary = {
        "rows": int(len(df)), "columns": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "missing_values": {c: int(v) for c, v in df.isna().sum().items()},
        "duplicate_rows": int(df.duplicated().sum()),
        "fraud_counts": {str(k): int(v) for k, v in df["isFraud"].value_counts().items()},
        "fraud_rate_percent": float(df["isFraud"].mean() * 100),
        "fraud_by_type": {k: int(v) for k, v in df.groupby("type")["isFraud"].sum().items()},
        "describe": json.loads(df.describe().to_json()),
    }
    (out / "eda_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "describe"}, indent=2))

    sns.countplot(x="isFraud", data=df); plt.title("Class distribution (log scale)")
    plt.yscale("log"); plt.savefig(out / "eda_class_distribution.png", dpi=120); plt.close()
    sns.countplot(x="type", hue="isFraud", data=df); plt.yscale("log")
    plt.title("Transaction type vs fraud"); plt.savefig(out / "eda_type_vs_fraud.png", dpi=120); plt.close()
    num = df.select_dtypes("number")
    plt.figure(figsize=(8, 6)); sns.heatmap(num.corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation matrix"); plt.tight_layout()
    plt.savefig(out / "eda_correlation.png", dpi=120); plt.close()
    sns.histplot(df.loc[df.amount > 0, "amount"], log_scale=True, bins=60, hue=df["isFraud"])
    plt.title("Amount distribution by class"); plt.savefig(out / "eda_amount.png", dpi=120); plt.close()
    print(f"EDA files written to {out}")


if __name__ == "__main__":
    main()
