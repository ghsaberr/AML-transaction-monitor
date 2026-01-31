import lightgbm as lgb
import json
import numpy as np

class ModelRunner:
    def __init__(self, model_dir):
        self.model = lgb.Booster(
            model_file=str(model_dir / "model.txt")
        )
        with open(model_dir / "features.json") as f:
            self.features = json.load(f)
        with open(model_dir / "threshold.json") as f:
            self.threshold = json.load(f)["threshold"]

    def run(self, tx_features: dict):
        x = np.array([[tx_features[f] for f in self.features]])
        score = float(self.model.predict(x)[0])
        decision = "ALERT" if score >= self.threshold else "PASS"

        return {
            "score": score,
            "decision": decision
        }
