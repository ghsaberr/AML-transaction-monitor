import pandas as pd
from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config():
    with open(PROJECT_ROOT / "configs/ingestion.yaml", "r") as f:
        return yaml.safe_load(f)


def temporal_split():
    cfg = load_config()
    input_path = PROJECT_ROOT / cfg["paths"]["output_file"]

    df = pd.read_parquet(input_path)

    df["date"] = df["timestamp"].dt.date

    train_start = pd.to_datetime(cfg["dataset"]["train_start"]).date()
    train_end = pd.to_datetime(cfg["dataset"]["train_end"]).date()
    test_date = pd.to_datetime(cfg["dataset"]["test_date"]).date()

    train_df = df[(df["date"] >= train_start) & (df["date"] <= train_end)]
    test_df = df[df["date"] == test_date]

    # ---- sanity checks ----
    assert train_df["date"].max() < test_date, "Leakage: train sees test date"
    assert test_df["date"].min() == test_date, "Test date mismatch"

    out_dir = PROJECT_ROOT / "data/processed"
    train_df.drop(columns=["date"]).to_parquet(out_dir / "train_base.parquet", index=False)
    test_df.drop(columns=["date"]).to_parquet(out_dir / "test_base.parquet", index=False)

    print("Temporal split completed")
    print(f"Train size: {len(train_df):,}")
    print(f"Test size : {len(test_df):,}")


if __name__ == "__main__":
    temporal_split()
