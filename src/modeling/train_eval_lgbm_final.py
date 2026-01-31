import json
import numpy as np
import pandas as pd
import lightgbm as lgb
import mlflow
import mlflow.lightgbm

from pathlib import Path
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
)

# --------------------------------------------------
# Paths
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models" / "lgbm_final"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# Config
# --------------------------------------------------
EXCLUDE_COLS = ["transaction_id", "label"]

BEST_PARAMS = {
    "objective": "binary",
    "metric": "aucpr",
    "boosting_type": "gbdt",
    "verbosity": -1,
    "seed": 42,
    "learning_rate": 0.02930335126924217,
    "num_leaves": 47,
    "max_depth": 9,
    "min_child_samples": 1464,
    "subsample": 0.659116171715938,
    "colsample_bytree": 0.6427407482673019,
    "reg_alpha": 1.5073159705876662,
    "reg_lambda": 0.001140686026549309,
}

ALERT_THRESHOLDS = [0.5, 0.9]

# --------------------------------------------------
# Helper metrics
# --------------------------------------------------
def alert_metrics(y_true, y_score, threshold):
    preds = (y_score >= threshold).astype(int)
    return {
        "alert_rate": preds.mean(),
        "precision": precision_score(y_true, preds, zero_division=0),
        "recall": recall_score(y_true, preds, zero_division=0),
    }


def precision_at_k(y_true, y_score, k_frac):
    k = int(len(y_score) * k_frac)
    idx = np.argsort(y_score)[::-1][:k]
    return y_true[idx].mean() if k > 0 else 0.0


# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    print("Loading data...")
    train_df = pd.read_parquet(DATA_DIR / "train_model_input.parquet")
    test_df = pd.read_parquet(DATA_DIR / "test_model_input.parquet")

    X_train = train_df.drop(columns=EXCLUDE_COLS)
    y_train = train_df["label"].values

    X_test = test_df.drop(columns=EXCLUDE_COLS)
    y_test = test_df["label"].values

    print(f"Train shape: {X_train.shape}")
    print(f"Test shape : {X_test.shape}")

    lgb_train = lgb.Dataset(X_train, y_train)

    # --------------------------------------------------
    # MLflow run
    # --------------------------------------------------
    mlflow.set_experiment("LightGBM_Final_Model")

    with mlflow.start_run(run_name="lgbm_final_train_eval"):
        mlflow.log_params(BEST_PARAMS)

        print("Training final LightGBM model...")
        model = lgb.train(
            BEST_PARAMS,
            lgb_train,
            num_boost_round=300,
        )

        # --------------------------------------------------
        # Evaluation
        # --------------------------------------------------
        print("Evaluating on test set...")
        test_scores = model.predict(X_test)

        pr_auc = average_precision_score(y_test, test_scores)
        mlflow.log_metric("pr_auc", pr_auc)

        print("\n=== Final LightGBM Results ===")
        print(f"PR-AUC: {pr_auc:.4f}")

        for th in ALERT_THRESHOLDS:
            m = alert_metrics(y_test, test_scores, th)
            print(f"\nThreshold = {th}")
            for k, v in m.items():
                print(f"  {k}: {v:.4f}")
                mlflow.log_metric(f"{k}_th_{th}", v)

        p_at_1 = precision_at_k(y_test, test_scores, 0.01)
        p_at_05 = precision_at_k(y_test, test_scores, 0.005)

        print("\nPrecision@K")
        print(f"  Precision@1%  : {p_at_1:.4f}")
        print(f"  Precision@0.5%: {p_at_05:.4f}")

        mlflow.log_metric("precision_at_1pct", p_at_1)
        mlflow.log_metric("precision_at_05pct", p_at_05)

        # --------------------------------------------------
        # Save model artifacts
        # --------------------------------------------------
        print("Saving model artifacts...")

        model_path = MODEL_DIR / "model.txt"
        model.save_model(model_path)

        with open(MODEL_DIR / "features.json", "w") as f:
            json.dump(X_train.columns.tolist(), f, indent=2)

        with open(MODEL_DIR / "threshold.json", "w") as f:
            json.dump({"threshold": 0.9}, f, indent=2)

        mlflow.lightgbm.log_model(
            model,
            artifact_path="model",
            registered_model_name="aml_lgbm",
        )

        print("Model saved and logged to MLflow")


if __name__ == "__main__":
    main()
