"""Tests for feature contract enforcement and artifact metadata."""

import pytest
import json
from pathlib import Path
from uuid import uuid4
from src.features.feature_contract import (
    FeatureContract, FeatureDefinition, FeatureType, NullBehavior
)


class TestFeatureDefinition:
    """Test individual feature definitions."""
    
    def test_create_feature_definition(self):
        """Create a basic feature definition."""
        feat = FeatureDefinition(
            name="amount",
            dtype=FeatureType.FLOAT,
            position=0,
            null_behavior=NullBehavior.DISALLOW,
        )
        
        assert feat.name == "amount"
        assert feat.dtype == FeatureType.FLOAT
        assert feat.position == 0
    
    def test_feature_with_default(self):
        """Feature with default value."""
        feat = FeatureDefinition(
            name="tx_count",
            dtype=FeatureType.INT,
            position=5,
            null_behavior=NullBehavior.DEFAULT,
            default_value=0,
        )
        
        assert feat.default_value == 0
    
    def test_feature_default_requires_value(self):
        """DEFAULT behavior requires default_value."""
        with pytest.raises(ValueError):
            FeatureDefinition(
                name="bad",
                dtype=FeatureType.INT,
                position=0,
                null_behavior=NullBehavior.DEFAULT,
            )
    
    def test_feature_to_dict(self):
        """Serialize feature to dict."""
        feat = FeatureDefinition(
            name="amount",
            dtype=FeatureType.FLOAT,
            position=0,
            null_behavior=NullBehavior.DISALLOW,
            description="Transaction amount",
        )
        
        d = feat.to_dict()
        assert d["name"] == "amount"
        assert d["dtype"] == "float"
        assert d["position"] == 0
    
    def test_feature_from_dict(self):
        """Deserialize feature from dict."""
        d = {
            "name": "amount",
            "dtype": "float",
            "position": 0,
            "null_behavior": "disallow",
            "default_value": None,
            "description": "Test",
        }
        
        feat = FeatureDefinition.from_dict(d)
        assert feat.name == "amount"
        assert feat.dtype == FeatureType.FLOAT


class TestFeatureContract:
    """Test feature contract enforcement."""
    
    @pytest.fixture
    def simple_contract(self):
        """Create a simple test contract."""
        features = [
            FeatureDefinition(
                name="amount",
                dtype=FeatureType.FLOAT,
                position=0,
                null_behavior=NullBehavior.DISALLOW,
            ),
            FeatureDefinition(
                name="tx_count",
                dtype=FeatureType.INT,
                position=1,
                null_behavior=NullBehavior.ZERO,
            ),
        ]
        return FeatureContract(features, version="1.0.0")
    
    def test_create_contract(self, simple_contract):
        """Create a feature contract."""
        assert simple_contract.version == "1.0.0"
        assert len(simple_contract.features) == 2
        assert simple_contract.schema_hash is not None
    
    def test_contract_sequential_positions(self):
        """Positions must be sequential."""
        features = [
            FeatureDefinition("f1", FeatureType.FLOAT, 0),
            FeatureDefinition("f2", FeatureType.FLOAT, 2),  # Gap!
        ]
        
        with pytest.raises(ValueError):
            FeatureContract(features)
    
    def test_validate_features_valid(self, simple_contract):
        """Validate valid features."""
        input_features = {
            "amount": 1000.0,
            "tx_count": 5,
        }
        
        result = simple_contract.validate_features(input_features)
        assert result["amount"] == 1000.0
        assert result["tx_count"] == 5
    
    def test_validate_features_missing_required(self):
        """Missing required feature raises error."""
        features = [
            FeatureDefinition(
                name="amount",
                dtype=FeatureType.FLOAT,
                position=0,
                null_behavior=NullBehavior.DISALLOW,
            ),
        ]
        contract = FeatureContract(features)
        
        input_features = {}  # amount missing
        
        with pytest.raises(ValueError, match="required but missing"):
            contract.validate_features(input_features)
    
    def test_validate_features_null_disallow(self, simple_contract):
        """Null with DISALLOW behavior raises error."""
        input_features = {
            "amount": None,
            "tx_count": 5,
        }
        
        with pytest.raises(ValueError, match="required but missing"):
            simple_contract.validate_features(input_features)
    
    def test_validate_features_null_zero(self, simple_contract):
        """Null with ZERO behavior fills with 0."""
        input_features = {
            "amount": 1000.0,
            "tx_count": None,
        }
        
        result = simple_contract.validate_features(input_features)
        assert result["tx_count"] == 0
    
    def test_validate_features_null_default(self):
        """Null with DEFAULT behavior uses default_value."""
        features = [
            FeatureDefinition(
                name="field",
                dtype=FeatureType.INT,
                position=0,
                null_behavior=NullBehavior.DEFAULT,
                default_value=42,
            ),
        ]
        contract = FeatureContract(features)
        
        result = contract.validate_features({"field": None})
        assert result["field"] == 42
    
    def test_validate_features_type_coercion(self, simple_contract):
        """Features are coerced to correct dtype."""
        input_features = {
            "amount": "1000.5",  # String -> float
            "tx_count": 5,       # Already int
        }
        
        result = simple_contract.validate_features(input_features)
        assert isinstance(result["amount"], float)
        assert isinstance(result["tx_count"], int)
        assert result["amount"] == 1000.5
        assert result["tx_count"] == 5
    
    def test_validate_features_float_to_int_conversion(self):
        """Float values can be converted to int."""
        features = [
            FeatureDefinition("count", FeatureType.INT, 0),
        ]
        contract = FeatureContract(features)
        
        # Float that converts cleanly
        result = contract.validate_features({"count": 5.0})
        assert result["count"] == 5
    
    def test_validate_features_type_error(self, simple_contract):
        """Invalid type coercion raises error."""
        input_features = {
            "amount": "not_a_number",
            "tx_count": 5,
        }
        
        with pytest.raises(ValueError, match="cannot coerce"):
            simple_contract.validate_features(input_features)
    
    def test_get_ordered_vector(self, simple_contract):
        """Get features as ordered vector."""
        features = {
            "amount": 1000.0,
            "tx_count": 5,
        }
        
        vector = simple_contract.get_ordered_vector(features)
        assert vector == [1000.0, 5.0]  # Converted to float
    
    def test_contract_to_dict(self, simple_contract):
        """Serialize contract to dict."""
        d = simple_contract.to_dict()
        
        assert d["version"] == "1.0.0"
        assert d["feature_count"] == 2
        assert len(d["features"]) == 2
        assert d["schema_hash"] is not None
    
    def test_contract_to_json(self, simple_contract):
        """Serialize contract to JSON."""
        json_str = simple_contract.to_json()
        
        # Should be valid JSON
        data = json.loads(json_str)
        assert data["version"] == "1.0.0"
    
    def test_contract_from_dict(self, simple_contract):
        """Deserialize contract from dict."""
        d = simple_contract.to_dict()
        
        contract2 = FeatureContract.from_dict(d)
        assert contract2.version == simple_contract.version
        assert contract2.schema_hash == simple_contract.schema_hash
    
    def test_contract_from_json(self, simple_contract):
        """Deserialize contract from JSON."""
        json_str = simple_contract.to_json()
        
        contract2 = FeatureContract.from_json(json_str)
        assert contract2.version == simple_contract.version
    
    def test_contract_hash_deterministic(self):
        """Contract hash is deterministic."""
        features = [
            FeatureDefinition("f1", FeatureType.FLOAT, 0),
            FeatureDefinition("f2", FeatureType.INT, 1),
        ]
        
        contract1 = FeatureContract(features, version="1.0.0")
        contract2 = FeatureContract(features, version="1.0.0")
        
        assert contract1.schema_hash == contract2.schema_hash


class TestTrainingMetadata:
    """Test training metadata for null handling."""
    
    def test_validate_with_means(self):
        """Use training means for null values."""
        features = [
            FeatureDefinition(
                name="velocity",
                dtype=FeatureType.FLOAT,
                position=0,
                null_behavior=NullBehavior.MEAN,
            ),
        ]
        
        training_metadata = {
            "means": {"velocity": 100.5}
        }
        
        contract = FeatureContract(
            features,
            training_metadata=training_metadata
        )
        
        result = contract.validate_features({"velocity": None})
        assert result["velocity"] == 100.5
    
    def test_mean_missing_raises_error(self):
        """Missing mean in metadata raises error."""
        features = [
            FeatureDefinition(
                name="velocity",
                dtype=FeatureType.FLOAT,
                position=0,
                null_behavior=NullBehavior.MEAN,
            ),
        ]
        
        contract = FeatureContract(features, training_metadata={})
        
        with pytest.raises(ValueError, match="training_metadata missing"):
            contract.validate_features({"velocity": None})


class TestLoadFromFile:
    """Test loading contracts from files."""
    
    def test_load_feature_contract_from_file(self):
        """Load feature contract from actual file."""
        path = Path("models/lgbm_final/feature_contract.json")
        
        if path.exists():
            contract = FeatureContract.from_file(str(path))
            
            assert contract.version == "1.0.0"
            assert len(contract.features) == 35
            # Schema hash is deterministically computed
            assert contract.schema_hash is not None
            assert len(contract.schema_hash) == 16
            
            # Check feature names
            names = [f.name for f in contract.features]
            assert "amount" in names
            assert "pagerank" in names
    
    def test_validate_realistic_features(self):
        """Validate realistic feature set."""
        path = Path("models/lgbm_final/feature_contract.json")
        
        if path.exists():
            contract = FeatureContract.from_file(str(path))
            
            # Create realistic features
            features = {
                "amount": 5000.0,
                "log_amount": 8.5,
                "hour_of_day": 14,
                "day_of_week": 2,
                "time_since_prev_tx": 3600.0,
                "amount_delta_prev_tx": 100.0,
                "tx_count_1h": 2,
                "tx_amount_mean_1h": 4500.0,
                "tx_count_24h": 5,
                "tx_amount_mean_24h": 3200.0,
                "tx_count_7d": 15,
                "tx_amount_mean_7d": 2800.0,
                "out_degree": 10.0,
                "in_degree": 20.0,
                "total_degree": 30.0,
                "pagerank": 0.002,
                "ego_tx_count": 50,
                "ego_amount_sum": 100000.0,
                "ego_amount_mean": 2000.0,
            }
            
            # Add n2v embeddings
            for i in range(16):
                features[f"n2v_{i}"] = 0.01 * (i - 8)
            
            # Should validate
            result = contract.validate_features(features)
            assert len(result) == 35
            assert result["amount"] == 5000.0
