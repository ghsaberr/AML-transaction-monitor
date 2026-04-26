"""
Feature contract enforcement for model serving.

Ensures strict schema discipline:
- Feature names, order, and data types
- Default/null behavior
- Version/schema hash for reproducibility
"""

import hashlib
import json
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from enum import Enum


class FeatureType(str, Enum):
    """Supported feature data types."""
    FLOAT = "float"
    INT = "int"
    BOOL = "bool"
    STRING = "string"


class NullBehavior(str, Enum):
    """How to handle missing/null values."""
    DISALLOW = "disallow"  # Raise error
    DEFAULT = "default"    # Use provided default
    ZERO = "zero"          # Fill with 0
    MEAN = "mean"          # Fill with training mean (from metadata)


class FeatureDefinition:
    """Single feature specification."""
    
    def __init__(
        self,
        name: str,
        dtype: FeatureType,
        position: int,
        null_behavior: NullBehavior = NullBehavior.DISALLOW,
        default_value: Optional[Union[float, int, bool, str]] = None,
        description: str = "",
    ):
        """
        Define a single feature.
        
        Args:
            name: Feature name
            dtype: Data type
            position: Position in feature vector (order matters)
            null_behavior: How to handle missing values
            default_value: Default value if null_behavior=DEFAULT
            description: Human-readable description
        """
        self.name = name
        self.dtype = dtype
        self.position = position
        self.null_behavior = null_behavior
        self.default_value = default_value
        self.description = description
        
        # Validate
        if null_behavior == NullBehavior.DEFAULT and default_value is None:
            raise ValueError(f"Feature {name}: DEFAULT behavior requires default_value")
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "name": self.name,
            "dtype": self.dtype.value,
            "position": self.position,
            "null_behavior": self.null_behavior.value,
            "default_value": self.default_value,
            "description": self.description,
        }
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "FeatureDefinition":
        """Deserialize from dictionary."""
        return FeatureDefinition(
            name=data["name"],
            dtype=FeatureType(data["dtype"]),
            position=data["position"],
            null_behavior=NullBehavior(data.get("null_behavior", "disallow")),
            default_value=data.get("default_value"),
            description=data.get("description", ""),
        )


class FeatureContract:
    """
    Strict contract enforcement for feature schemas.
    
    Ensures:
    - All required features present and correct dtype
    - Features in expected order
    - Null/default handling is deterministic
    - Version/hash for reproducibility
    """
    
    def __init__(
        self,
        features: List[FeatureDefinition],
        version: str = "1.0.0",
        description: str = "",
        training_metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Create feature contract.
        
        Args:
            features: List of FeatureDefinition objects
            version: Contract version (semver)
            description: Contract description
            training_metadata: Stats from training set (means, mins, maxes, etc)
        """
        # Validate positions are sequential
        positions = sorted([f.position for f in features])
        if positions != list(range(len(features))):
            raise ValueError("Feature positions must be sequential 0..N")
        
        self.features = sorted(features, key=lambda f: f.position)
        self.version = version
        self.description = description
        self.training_metadata = training_metadata or {}
        self.created_at = datetime.utcnow().isoformat()
        
        # Compute schema hash for reproducibility
        self.schema_hash = self._compute_hash()
    
    def _compute_hash(self) -> str:
        """Compute deterministic hash of contract."""
        contract_data = json.dumps(
            {
                "version": self.version,
                "features": [f.to_dict() for f in self.features],
            },
            sort_keys=True,
        )
        return hashlib.sha256(contract_data.encode()).hexdigest()[:16]
    
    def validate_features(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and normalize features against contract.
        
        Args:
            features: Input feature dictionary
        
        Returns:
            Normalized features in contract order
        
        Raises:
            ValueError: If validation fails
        """
        validated = {}
        
        for feature_def in self.features:
            name = feature_def.name
            
            # Check if feature exists and handle null
            if name not in features or features[name] is None:
                if feature_def.null_behavior == NullBehavior.DISALLOW:
                    raise ValueError(
                        f"Feature '{name}' is required but missing or null"
                    )
                elif feature_def.null_behavior == NullBehavior.DEFAULT:
                    value = feature_def.default_value
                elif feature_def.null_behavior == NullBehavior.ZERO:
                    value = 0
                elif feature_def.null_behavior == NullBehavior.MEAN:
                    if "means" not in self.training_metadata:
                        raise ValueError(
                            f"Feature '{name}' requires MEAN fill but training_metadata missing"
                        )
                    value = self.training_metadata["means"].get(name)
                    if value is None:
                        raise ValueError(
                            f"Feature '{name}': no mean in training_metadata"
                        )
            else:
                value = features[name]
            
            # Type validation and coercion
            try:
                if feature_def.dtype == FeatureType.FLOAT:
                    value = float(value)
                elif feature_def.dtype == FeatureType.INT:
                    # Try as float first (handles "5.7" -> 5 conversions)
                    value = int(float(value))
                elif feature_def.dtype == FeatureType.BOOL:
                    if isinstance(value, bool):
                        value = float(value)
                    else:
                        value = float(bool(value))
                elif feature_def.dtype == FeatureType.STRING:
                    value = str(value)
            except (ValueError, TypeError) as e:
                raise ValueError(
                    f"Feature '{name}': cannot coerce {value} to {feature_def.dtype.value}: {e}"
                )
            
            validated[name] = value
        
        # Check for unexpected features
        unexpected = set(features.keys()) - set(f.name for f in self.features)
        if unexpected:
            # Log but don't fail - extra features are ignored
            pass
        
        return validated
    
    def get_ordered_vector(self, features: Dict[str, Any]) -> List[float]:
        """
        Get features as ordered vector for model.
        
        Args:
            features: Validated features (call validate_features first)
        
        Returns:
            List of values in contract order
        """
        return [features[f.name] for f in self.features]
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize contract to dictionary."""
        return {
            "version": self.version,
            "description": self.description,
            "schema_hash": self.schema_hash,
            "created_at": self.created_at,
            "feature_count": len(self.features),
            "features": [f.to_dict() for f in self.features],
            "training_metadata": self.training_metadata,
        }
    
    def to_json(self) -> str:
        """Serialize contract to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "FeatureContract":
        """Deserialize from dictionary."""
        features = [FeatureDefinition.from_dict(f) for f in data["features"]]
        contract = FeatureContract(
            features=features,
            version=data.get("version", "1.0.0"),
            description=data.get("description", ""),
            training_metadata=data.get("training_metadata", {}),
        )
        return contract
    
    @staticmethod
    def from_json(json_str: str) -> "FeatureContract":
        """Deserialize from JSON string."""
        data = json.loads(json_str)
        return FeatureContract.from_dict(data)
    
    @staticmethod
    def from_file(path: str) -> "FeatureContract":
        """Load contract from JSON file."""
        with open(path, "r") as f:
            return FeatureContract.from_json(f.read())
