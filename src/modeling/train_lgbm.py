import pandas as pd
import lightgbm as lgb
from pathlib import Path
from metrics import evaluate_at_threshold, evaluate_pr_auc

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data/processed"


def train():
    train = pd.read_parquet(DATA_DIR / "train_model_input.parquet")
    test = pd.read_parquet(DATA_DIR / "test_model_input.parquet")

    X_train = train.drop(columns=["transaction_id", "label"])
    y_train = train["label"]

    X_test = test.drop(columns=["transaction_id", "label"])
    y_test = test["label"]

    model = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=64,
        class_weight="balanced",
        device="gpu",
        random_state=42,
    )

    model.fit(X_train, y_train)

    y_score = model.predict_proba(X_test)[:, 1]

    print("=== LightGBM ===")
    print("PR-AUC:", evaluate_pr_auc(y_test, y_score))

    for t in [0.5, 0.9]:
        print(f"Threshold {t}:", evaluate_at_threshold(y_test, y_score, t))


if __name__ == "__main__":
    train()
