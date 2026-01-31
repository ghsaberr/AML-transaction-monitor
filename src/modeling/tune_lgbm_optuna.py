import optuna
import lightgbm as lgb
import pandas as pd
import numpy as np
import mlflow
import mlflow.lightgbm

from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import average_precision_score

# -----------------------------
# Config
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data/processed"

N_SPLITS = 3
N_TRIALS = 20
SEED = 42

EXCLUDE_COLS = ["transaction_id", "label"]

# -----------------------------
# Load data
# -----------------------------
print("Loading training data...")
df = pd.read_parquet(DATA_DIR / "train_model_input.parquet")

X = df.drop(columns=EXCLUDE_COLS)
y = df["label"].values

tscv = TimeSeriesSplit(n_splits=N_SPLITS)

# -----------------------------
# Optuna objective
# -----------------------------
def objective(trial):

    params = {
        "objective": "binary",
        "metric": "aucpr",
        "boosting_type": "gbdt",
        "verbosity": -1,
        "seed": SEED,

        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 32, 128),
        "max_depth": trial.suggest_int("max_depth", 5, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 50, 2000),
        "subsample": trial.suggest_float("subsample", 0.6, 0.9),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.9),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }

    pr_aucs = []

    with mlflow.start_run(nested=True):
        mlflow.log_params(params)

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            lgb_train = lgb.Dataset(X_train, y_train)
            lgb_val = lgb.Dataset(X_val, y_val)

            try:
                model = lgb.train(
                    params,
                    lgb_train,
                    num_boost_round=300,
                )
            except lgb.basic.LightGBMError:
                return None

            val_pred = model.predict(X_val)
            pr_auc = average_precision_score(y_val, val_pred)
            pr_aucs.append(pr_auc)

        mean_pr_auc = float(np.mean(pr_aucs))
        mlflow.log_metric("cv_pr_auc", mean_pr_auc)

        return mean_pr_auc


# -----------------------------
# Run study
# -----------------------------
print("Starting Optuna study...")
mlflow.set_experiment("LightGBM_Model_Selection")

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=N_TRIALS)

print("\nBest PR-AUC:", study.best_value)
print("Best params:")
for k, v in study.best_params.items():
    print(f"  {k}: {v}")
