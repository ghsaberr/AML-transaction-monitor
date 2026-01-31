import pandas as pd
from pathlib import Path
from datetime import datetime
import hashlib
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config():
    with open(PROJECT_ROOT / "configs/ingestion.yaml", "r") as f:
        return yaml.safe_load(f)


def make_transaction_id(row) -> str:
    raw = (
        f"{row['Timestamp']}|"
        f"{row['From Bank']}|{row['Account']}|"
        f"{row['To Bank']}|{row['Account.1']}|"
        f"{row['Amount Paid']}"
    )
    return hashlib.sha1(raw.encode()).hexdigest()


def ingest():
    cfg = load_config()
    raw_dir = PROJECT_ROOT / cfg["paths"]["raw_dir"]
    output_path = PROJECT_ROOT / cfg["paths"]["output_file"]

    tx = pd.read_csv(raw_dir / "HI-Small_Trans.csv")

    tx["transaction_id"] = tx.apply(make_transaction_id, axis=1)

    canonical = pd.DataFrame({
        "transaction_id": tx["transaction_id"],
        "timestamp": pd.to_datetime(tx["Timestamp"], errors="coerce"),
        "from_account": tx["From Bank"].astype(str) + "_" + tx["Account"],
        "to_account": tx["To Bank"].astype(str) + "_" + tx["Account.1"],
        "amount": tx["Amount Paid"],
        "channel": tx["Payment Format"],
        "country": None,
        "tags": None,
        "label": tx["Is Laundering"].astype("Int64"),
        "provenance_source": "HI-Small_Trans.csv",
        "ingestion_time": datetime.utcnow(),
    })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canonical.to_parquet(output_path, index=False)

    print(f"Ingestion complete: {len(canonical):,} transactions")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    ingest()
