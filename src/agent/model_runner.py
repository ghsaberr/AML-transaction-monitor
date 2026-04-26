import lightgbm as lgb
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List

from src.features.feature_contract import FeatureContract


class ModelRunner:
    """Load, validate, and run LightGBM model with strict feature contracts."""
    
    def __init__(self, model_dir):
        """
        Initialize model with feature contract and metadata.
        
        Args:
            model_dir: Path to model directory (lgbm_final)
        """
        model_dir = Path(model_dir)
        
        # Load model
        self.model = lgb.Booster(
            model_file=str(model_dir / "model.txt")
        )
        
        # Load legacy feature list for backward compatibility
        with open(model_dir / "features.json") as f:
            self.features = json.load(f)
        
        # Load threshold
        with open(model_dir / "threshold.json") as f:
            threshold_data = json.load(f)
            self.threshold = threshold_data["threshold"]
        
        # Load feature contract (strict enforcement)
        with open(model_dir / "feature_contract.json") as f:
            self.feature_contract = FeatureContract.from_json(f.read())
        
        # Load artifact metadata
        with open(model_dir / "artifact_metadata.json") as f:
            self.artifact_metadata = json.load(f)
        
        # Expose key metadata
        self.model_version = self.artifact_metadata["model"]["version"]
        self.feature_contract_version = self.artifact_metadata["feature_contract"]["version"]
        self.threshold_version = self.artifact_metadata["threshold"]["version"]
        self.training_window = self.artifact_metadata["training"]
    
    def validate_features(self, tx_features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate features against contract.
        
        Args:
            tx_features: Input feature dictionary
        
        Returns:
            Validated features in contract order
        
        Raises:
            ValueError: If validation fails
        """
        return self.feature_contract.validate_features(tx_features)
    
    def run(self, tx_features: dict) -> Dict[str, Any]:
        """
        Score a transaction with feature validation.
        
        Args:
            tx_features: Transaction features (must pass contract validation)
        
        Returns:
            Dictionary with score, decision, versions
        
        Raises:
            ValueError: If features fail contract validation
        """
        # Validate and normalize features
        validated_features = self.validate_features(tx_features)
        
        # Get ordered vector
        x = np.array([self.feature_contract.get_ordered_vector(validated_features)])
        
        # Score
        score = float(self.model.predict(x)[0])
        decision = "ALERT" if score >= self.threshold else "PASS"
        
        return {
            "score": score,
            "decision": decision,
            "model_version": self.model_version,
            "feature_contract_version": self.feature_contract_version,
            "threshold_version": self.threshold_version,
            "schema_hash": self.feature_contract.schema_hash,
        }
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get complete model and artifact metadata."""
        return {
            "model": self.artifact_metadata["model"],
            "training": self.artifact_metadata["training"],
            "feature_contract": {
                "version": self.feature_contract_version,
                "schema_hash": self.feature_contract.schema_hash,
                "feature_count": len(self.feature_contract.features),
            },
            "threshold": self.artifact_metadata["threshold"],
            "performance": self.artifact_metadata["performance"],
            "monitoring": self.artifact_metadata["monitoring"],
        }
    
    def get_feature_contract_spec(self) -> Dict[str, Any]:
        """Get feature contract specification."""
        return self.feature_contract.to_dict()
