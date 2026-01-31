import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data/processed"


def assemble(split: str):
    assert split in {"train", "test"}

    print(f"Assembling graph model input for {split}...")

    graph_df = pd.read_parquet(DATA_DIR / f"{split}_graph_features.parquet")
    tab_df = pd.read_parquet(DATA_DIR / f"{split}_tabular_features.parquet")

    tab_cols = [
        "transaction_id",
        "log_amount",
        "tx_count_1h",
        "tx_count_24h",
        "tx_count_7d",
    ]

    merged = graph_df.merge(
        tab_df[tab_cols],
        on="transaction_id",
        how="left",
        validate="many_to_many" 
    )

    merged.reset_index(drop=True, inplace=True)

    merged.to_parquet(
        DATA_DIR / f"{split}_graph_model_input.parquet",
        index=False
    )

    print(f"Saved {split}_graph_model_input.parquet")


if __name__ == "__main__":
    assemble("train")
    assemble("test")
