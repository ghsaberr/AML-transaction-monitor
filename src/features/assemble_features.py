import pandas as pd
from pathlib import Path
import hashlib

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data/processed"


META_COLS = {
    "timestamp",
    "from_account",
    "to_account",
    "channel",
    "tags",
    "provenance_source",
    "ingestion_time",
    "country",  # always null
}


def assemble(split: str):
    assert split in {"train", "test"}

    tabular = pd.read_parquet(DATA_DIR / f"{split}_tabular_features.parquet")
    graph = pd.read_parquet(DATA_DIR / f"{split}_graph_features.parquet")

    # ---- ensure same row order ----
    sort_cols = ["transaction_id", "timestamp"]
    tabular = tabular.sort_values(sort_cols).reset_index(drop=True)
    graph = graph.sort_values(sort_cols).reset_index(drop=True)

    # ---- sanity checks ----
    assert len(tabular) == len(graph), "Row count mismatch"
    assert (tabular["transaction_id"].values == graph["transaction_id"].values).all(), \
        "Transaction ID alignment mismatch"

    # ---- select graph-only columns ----
    graph_feature_cols = [
        c for c in graph.columns
        if c not in META_COLS
        and c not in tabular.columns
        and c not in {"label"}
    ]

    # ---- column-wise assembly ----
    df = pd.concat(
        [
            tabular.drop(columns=META_COLS, errors="ignore"),
            graph[graph_feature_cols],
        ],
        axis=1
    )

    # ---- feature manifest ----
    feature_cols = sorted(
        c for c in df.columns
        if c not in {"transaction_id", "label"}
    )

    manifest_raw = "|".join(feature_cols)
    feature_hash = hashlib.sha1(manifest_raw.encode()).hexdigest()
    df.attrs["feature_manifest_hash"] = feature_hash

    out_path = DATA_DIR / f"{split}_model_input.parquet"
    df.to_parquet(out_path, index=False)

    print(f"{split} model dataset assembled")
    print(f"Output: {out_path}")
    print(f"Rows: {len(df):,} | Features: {len(feature_cols)}")
    print(f"Feature hash: {feature_hash}")


if __name__ == "__main__":
    assemble("train")
    assemble("test")
