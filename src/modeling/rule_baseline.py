import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data/processed"


def rule_score(df: pd.DataFrame) -> pd.Series:
    score = pd.Series(0, index=df.index)

    # Large amount
    score += (df["amount"] > df["amount"].quantile(0.99)).astype(int)

    # High velocity
    score += (df["tx_count_1h"] >= 5).astype(int)

    # Structurally important account
    score += (df["pagerank"] > df["pagerank"].quantile(0.99)).astype(int)

    return score


def evaluate():
    df = pd.read_parquet(DATA_DIR / "test_model_input.parquet")

    df["rule_score"] = rule_score(df)
    df["alert"] = (df["rule_score"] >= 2).astype(int)

    y_true = df["label"].values
    y_pred = df["alert"].values

    precision = (y_pred & y_true).sum() / max(y_pred.sum(), 1)
    recall = (y_pred & y_true).sum() / max(y_true.sum(), 1)
    alert_rate = y_pred.mean()

    print("Rule-based baseline")
    print(f"Alert rate : {alert_rate:.4f}")
    print(f"Precision  : {precision:.4f}")
    print(f"Recall     : {recall:.4f}")


if __name__ == "__main__":
    evaluate()
