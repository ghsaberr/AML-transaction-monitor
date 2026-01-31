from pathlib import Path
from model_runner import ModelRunner
from diagnostics import DiagnosticTools
from aml_agent import AMLAgent

MODEL_DIR = Path("models/lgbm_final")

# RAW transaction (for diagnostics)
raw_tx = {
    "amount": 250000,
    "hour_of_day": 3,
    "day_of_week": 2,
    "tx_count_1h": 14,
    "tx_count_24h": 42,
    "out_degree": 15
}

# Engineered features (for model)
tx_features = {
    "amount": 250000,
    "log_amount": 7.8,
    "hour_of_day": 3,
    "day_of_week": 2,
    "time_since_prev_tx": 120,
    "amount_delta_prev_tx": 180000,
    "tx_count_1h": 14,
    "tx_amount_mean_1h": 12000,
    "tx_count_24h": 42,
    "tx_amount_mean_24h": 15000,
    "tx_count_7d": 210,
    "tx_amount_mean_7d": 17000,
    "out_degree": 15,
    "in_degree": 4,
    "total_degree": 19,
    "pagerank": 0.002,
    "ego_tx_count": 55,
    "ego_amount_sum": 1.2e6,
    "ego_amount_mean": 22000,
    **{f"n2v_{i}": 0.0 for i in range(16)}
}

tx_text = "Rapid transfer between multiple accounts within one hour"

agent = AMLAgent(
    model_runner=ModelRunner(MODEL_DIR),
    diagnostics=DiagnosticTools()
)

result = agent.run(
    tx_features=tx_features,
    raw_tx=raw_tx,
    tx_text=tx_text
)

print(result)
