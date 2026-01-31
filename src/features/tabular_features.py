import pandas as pd
import numpy as np
from pathlib import Path
import hashlib

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data/processed"


def add_stateless_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["log_amount"] = np.log1p(df["amount"])
    df["hour_of_day"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    return df


def add_velocity_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["from_account", "timestamp"])
    df["time_since_prev_tx"] = (
        df.groupby("from_account")["timestamp"]
        .diff()
        .dt.total_seconds()
    )
    df["amount_delta_prev_tx"] = (
        df.groupby("from_account")["amount"]
        .diff()
    )
    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["from_account", "timestamp"])
    df = df.set_index("timestamp")

    windows = {
        "1h": "1H",
        "24h": "24H",
        "7d": "7D",
    }

    for name, window in windows.items():
        grp = df.groupby("from_account")["amount"]

        df[f"tx_count_{name}"] = (
            grp.rolling(window, closed="left").count().reset_index(level=0, drop=True)
        )

        df[f"tx_amount_mean_{name}"] = (
            grp.rolling(window, closed="left").mean().reset_index(level=0, drop=True)
        )

    df = df.reset_index()
    return df


def feature_manifest_hash(df: pd.DataFrame) -> str:
    cols = sorted(df.columns)
    raw = "|".join(cols)
    return hashlib.sha1(raw.encode()).hexdigest()


def build_features(split: str):
    assert split in {"train", "test"}

    input_path = DATA_DIR / f"{split}_base.parquet"
    output_path = DATA_DIR / f"{split}_tabular_features.parquet"

    df = pd.read_parquet(input_path)

    df = add_stateless_features(df)
    df = add_velocity_features(df)
    df = add_rolling_features(df)

    manifest_hash = feature_manifest_hash(df)
    df.attrs["feature_manifest_hash"] = manifest_hash

    df.to_parquet(output_path, index=False)

    print(f"{split} tabular features created")
    print(f"Output: {output_path}")
    print(f"Feature hash: {manifest_hash}")


if __name__ == "__main__":
    build_features("train")
    build_features("test")
