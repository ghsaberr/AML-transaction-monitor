# src/api/schemas.py
"""
Pydantic schemas for API contracts.
These enforce strict request/response validation across all endpoints.
"""

from typing import Optional, Dict, Any, List
from datetime import datetime
from pydantic import BaseModel, Field, validator
import json


# ============================================================================
# SCORE ENDPOINT CONTRACTS
# ============================================================================

class ScoreRequest(BaseModel):
    """Score request with feature contract validation."""
    
    tx_features: Dict[str, Any] = Field(
        ...,
        description="Transaction features matching the feature contract"
    )
    
    @validator('tx_features')
    def validate_features_not_empty(cls, v):
        if not v or len(v) == 0:
            raise ValueError("tx_features cannot be empty")
        return v


class ScoreResponse(BaseModel):
    """
    Score response with complete metadata.
    Enforces return of: case_id, score, review_flag, threshold_used, 
    model_version, and timestamp.
    """
    
    case_id: str = Field(
        ...,
        description="Unique case identifier (UUID)"
    )
    request_id: str = Field(
        ...,
        description="Unique request identifier (UUID)"
    )
    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Risk score in [0, 1]"
    )
    review_flag: bool = Field(
        ...,
        description="Whether case should be reviewed (score >= threshold)"
    )
    decision: str = Field(
        ...,
        description="Decision: ALERT or PASS"
    )
    threshold_used: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Threshold applied for decision"
    )
    model_version: str = Field(
        ...,
        description="Version of model used"
    )
    threshold_version: str = Field(
        ...,
        description="Version of threshold used"
    )
    feature_contract_version: str = Field(
        ...,
        description="Version of feature contract"
    )
    review_status: str = Field(
        default="queued_for_review",
        description="Initial review status"
    )
    timestamp: datetime = Field(
        ...,
        description="When the score was computed (ISO format)"
    )
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# ============================================================================
# EXPLAIN ENDPOINT CONTRACTS
# ============================================================================

class ExplainRequest(BaseModel):
    """Explanation request."""
    
    case_id: str = Field(
        ...,
        description="Case ID to explain"
    )
    tx_features: Dict[str, Any] = Field(
        ...,
        description="Transaction features"
    )
    raw_tx: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Raw transaction data for context"
    )
    tx_text: Optional[str] = Field(
        default=None,
        description="Textual description of transaction"
    )


class FeatureImportance(BaseModel):
    """Single feature importance entry."""
    
    feature_name: str
    importance_value: float
    contribution: str  # "positive" or "negative"


class ExplainResponse(BaseModel):
    """Explanation response with fallback feature importance."""
    
    case_id: str = Field(
        ...,
        description="Case being explained"
    )
    score: float = Field(
        ...,
        description="Risk score"
    )
    model_version: str = Field(
        ...,
        description="Model version used"
    )
    explanation_type: str = Field(
        ...,
        description="Type of explanation: 'agent' or 'feature_importance'"
    )
    # For agent-based explanations
    agent_response: Optional[str] = Field(
        default=None,
        description="Free-text explanation from LLM agent"
    )
    # For fallback feature importance
    top_features: Optional[List[FeatureImportance]] = Field(
        default=None,
        description="Top contributing features if agent unavailable"
    )
    timestamp: datetime = Field(
        ...,
        description="When explanation was generated"
    )
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# ============================================================================
# REVIEW ENDPOINT CONTRACTS
# ============================================================================

class ReviewRequest(BaseModel):
    """Manual review submission."""
    
    case_id: str = Field(
        ...,
        description="Case being reviewed"
    )
    reviewer_id: str = Field(
        ...,
        description="ID of reviewer"
    )
    decision: str = Field(
        ...,
        description="Decision: APPROVED, REJECTED, or ESCALATED"
    )
    note: Optional[str] = Field(
        default=None,
        max_length=1000,
        description="Reviewer notes"
    )
    
    @validator('decision')
    def validate_decision(cls, v):
        valid = {'APPROVED', 'REJECTED', 'ESCALATED'}
        if v not in valid:
            raise ValueError(f"decision must be one of {valid}")
        return v


class ReviewResponse(BaseModel):
    """Review submission response."""
    
    review_id: str = Field(
        ...,
        description="Unique review identifier"
    )
    case_id: str
    reviewer_id: str
    decision: str
    note: Optional[str]
    previous_status: str
    new_status: str
    timestamp: datetime
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# ============================================================================
# AUDIT ENDPOINT CONTRACTS
# ============================================================================

class AuditEvent(BaseModel):
    """Single audit trail event."""
    
    event_id: str
    case_id: str
    event_type: str
    actor: Optional[str]
    details: Optional[Dict[str, Any]]
    timestamp: datetime
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class AuditTrailResponse(BaseModel):
    """Audit trail for a case."""
    
    case_id: str
    events: List[AuditEvent]
    total_events: int
    timestamp: datetime
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# ============================================================================
# HEALTH ENDPOINT CONTRACTS
# ============================================================================

class HealthResponse(BaseModel):
    """Health check response with version metadata."""
    
    status: str = Field(
        ...,
        description="Service status: 'ok' or 'degraded'"
    )
    version: str = Field(
        ...,
        description="API version (semantic)"
    )
    model_version: str = Field(
        ...,
        description="Current model version"
    )
    model_status: str = Field(
        ...,
        description="Model load status: 'ready' or 'error'"
    )
    threshold_version: str = Field(
        ...,
        description="Current threshold version"
    )
    feature_contract_version: str = Field(
        ...,
        description="Current feature contract version"
    )
    database_status: str = Field(
        ...,
        description="Database status: 'ok', 'error', or 'initializing'"
    )
    timestamp: datetime = Field(
        ...,
        description="Health check timestamp"
    )
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# ============================================================================
# METRICS ENDPOINT CONTRACTS
# ============================================================================

class MetricsSnapshot(BaseModel):
    """Snapshot of system metrics."""
    
    # Request/latency metrics
    request_count: int
    success_count: int
    error_count: int
    average_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    
    # Score statistics
    score_mean: Optional[float]
    score_median: Optional[float]
    score_std: Optional[float]
    
    # Case counts
    cases_total: int = Field(default=0, description="Total cases processed")
    cases_queued: int = Field(default=0, description="Cases awaiting review")
    cases_approved: int = Field(default=0, description="Approved cases")
    cases_rejected: int = Field(default=0, description="Rejected cases")
    cases_escalated: int = Field(default=0, description="Escalated cases")
    
    # Drift metrics
    drift_features_analyzed: int = Field(default=0, description="Features analyzed for drift")
    drift_features_alert: int = Field(default=0, description="Features with drift alert")
    drift_alert: bool = Field(default=False, description="Overall drift alert status")
    
    # Review rate
    review_rate: float = Field(default=0.0, description="reviewed / total")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class MetricsResponse(BaseModel):
    """Metrics endpoint response."""
    
    snapshot: MetricsSnapshot
    timestamp: datetime
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# ============================================================================
# ERROR RESPONSE CONTRACTS
# ============================================================================

class ErrorDetail(BaseModel):
    """Structured error response."""
    
    error_code: str = Field(
        ...,
        description="Machine-readable error code"
    )
    message: str = Field(
        ...,
        description="Human-readable error message"
    )
    details: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional error context"
    )
    timestamp: datetime = Field(
        ...,
        description="When error occurred"
    )
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# ============================================================================
# CASE MANAGEMENT CONTRACTS
# ============================================================================

class CaseRecord(BaseModel):
    """Single case record."""
    
    case_id: str
    request_id: str
    model_version: str
    threshold_version: str
    score: float
    review_status: str
    created_at: datetime
    updated_at: datetime
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class CaseListResponse(BaseModel):
    """List of cases with filtering."""
    
    total_cases: int
    cases: List[Dict[str, Any]]
    timestamp: str  # ISO format string


class CaseDetailsResponse(BaseModel):
    """Complete case details including audit trail."""
    
    case: Dict[str, Any]
    audit_events: List[Dict[str, Any]]
    timestamp: str


# ============================================================================
# REVIEW STATISTICS CONTRACTS
# ============================================================================

class ReviewStatistics(BaseModel):
    """Review workflow statistics."""
    
    total_cases: int = Field(..., description="Total number of cases")
    queued_for_review: int = Field(..., description="Cases awaiting review")
    approved: int = Field(..., description="Approved cases")
    rejected: int = Field(..., description="Rejected cases")
    escalated: int = Field(..., description="Escalated cases")
    review_rate: float = Field(..., description="Percentage of cases reviewed")
    approval_rate: float = Field(..., description="Percentage of reviewed cases approved")
    escalation_rate: float = Field(..., description="Percentage of cases escalated")
    queue_health: str = Field(
        ...,
        description="Queue health status: ok, warning, or critical"
    )
    timestamp: str = Field(..., description="When stats were calculated")


# ============================================================================
# FEATURE CONTRACT & ARTIFACT METADATA
# ============================================================================

class FeatureDefinitionSchema(BaseModel):
    """Feature definition within contract."""
    
    name: str = Field(..., description="Feature name")
    dtype: str = Field(..., description="Data type: float, int, bool, string")
    position: int = Field(..., description="Position in feature vector")
    null_behavior: str = Field(..., description="null|default|zero|mean")
    default_value: Optional[Any] = Field(None, description="Default if null_behavior=default")
    description: str = Field(..., description="Feature description")


class FeatureContractSchema(BaseModel):
    """Complete feature contract specification."""
    
    version: str = Field(..., description="Contract version (semver)")
    description: str = Field(..., description="Contract description")
    schema_hash: str = Field(..., description="SHA256 hash of contract (first 16 chars)")
    created_at: str = Field(..., description="When contract was created (ISO format)")
    feature_count: int = Field(..., description="Number of features")
    features: List[FeatureDefinitionSchema] = Field(..., description="Feature specifications")
    training_metadata: Dict[str, Any] = Field(
        ...,
        description="Training set statistics (means, mins, maxes)"
    )


class ThresholdMetadataSchema(BaseModel):
    """Threshold and decision boundary metadata."""
    
    version: str = Field(..., description="Threshold version")
    value: float = Field(..., description="Decision threshold (0-1)")
    rationale: str = Field(..., description="Why this threshold was chosen")
    validation_auc: float = Field(..., description="AUC on validation set")
    validation_precision: float = Field(..., description="Precision at this threshold")
    validation_recall: float = Field(..., description="Recall at this threshold")
    validation_f1: float = Field(..., description="F1 score at this threshold")


class TrainingMetadataSchema(BaseModel):
    """Model training metadata."""
    
    window_start: str = Field(..., description="Training data start date")
    window_end: str = Field(..., description="Training data end date")
    dataset: str = Field(..., description="Dataset identifier")
    samples: int = Field(..., description="Number of training samples")
    positive_samples: int = Field(..., description="Number of positive samples")
    class_balance: float = Field(..., description="Ratio of positive samples")
    preprocessing: str = Field(..., description="Preprocessing steps applied")
    feature_engineering_version: str = Field(..., description="Version of feature engineering code")


class ModelPerformanceSchema(BaseModel):
    """Model performance metrics."""
    
    training_auc: float = Field(..., description="AUC on training set")
    validation_auc: float = Field(..., description="AUC on validation set")
    test_auc: float = Field(..., description="AUC on test set")
    feature_importance_count: int = Field(..., description="Number of features with importance")
    top_features: List[str] = Field(..., description="Top 5 important features")


class MLFlowMetadataSchema(BaseModel):
    """MLflow experiment and run tracking."""
    
    experiment_id: str = Field(..., description="MLflow experiment ID")
    experiment_name: str = Field(..., description="MLflow experiment name")
    run_id: str = Field(..., description="MLflow run ID")
    artifact_uri: str = Field(..., description="MLflow artifact storage URI")


class ArtifactMetadataResponse(BaseModel):
    """Complete artifact metadata for model lineage and governance."""
    
    model: Dict[str, Any] = Field(..., description="Model identity and version")
    training: TrainingMetadataSchema = Field(..., description="Training metadata")
    feature_contract: Dict[str, Any] = Field(..., description="Feature contract reference")
    threshold: ThresholdMetadataSchema = Field(..., description="Decision threshold")
    performance: ModelPerformanceSchema = Field(..., description="Model performance")
    mlflow: MLFlowMetadataSchema = Field(..., description="MLflow tracking")


class HealthResponseWithMetadata(BaseModel):
    """Enhanced health response with metadata."""
    
    version: str = Field(..., description="API version")
    status: str = Field(..., description="API status: healthy, degraded, error")
    model_version: str = Field(..., description="Model version")
    feature_contract_version: str = Field(..., description="Feature contract version")
    threshold_version: str = Field(..., description="Threshold version")
    schema_hash: str = Field(..., description="Feature contract schema hash")
    database: str = Field(..., description="Database status")
    timestamp: datetime = Field(..., description="Timestamp")


# ============================================================================
# DRIFT DETECTION & MONITORING CONTRACTS
# ============================================================================

class FeatureDriftSchema(BaseModel):
    """Drift analysis for a single feature."""
    
    name: str = Field(..., description="Feature name")
    ks_statistic: float = Field(..., description="Kolmogorov-Smirnov test statistic")
    p_value: float = Field(..., description="P-value from KS test")
    alert: bool = Field(..., description="Whether drift was detected")
    training_mean: float = Field(..., description="Mean from training set")
    current_mean: float = Field(..., description="Current production mean")
    std_devs_apart: float = Field(..., description="Number of std devs between means")


class DriftAnalysisResponse(BaseModel):
    """Complete drift analysis for features."""
    
    window_hours: int = Field(..., description="Analysis time window in hours")
    features_analyzed: int = Field(..., description="Number of features analyzed")
    features_drifting: int = Field(..., description="Number of features with drift")
    overall_alert: bool = Field(..., description="Overall drift alert status")
    drifting_features: List[str] = Field(..., description="Names of features with drift")
    details: Dict[str, FeatureDriftSchema] = Field(..., description="Per-feature drift details")


class PerformanceMetricsSchema(BaseModel):
    """Model performance metrics."""
    
    precision: float = Field(..., description="Precision (TP/(TP+FP))")
    recall: float = Field(..., description="Recall (TP/(TP+FN))")
    accuracy: float = Field(..., description="Accuracy ((TP+TN)/total)")
    f1: float = Field(..., description="F1 score")
    true_positives: int = Field(..., description="True positives")
    false_positives: int = Field(..., description="False positives")
    true_negatives: int = Field(..., description="True negatives")
    false_negatives: int = Field(..., description="False negatives")
    samples: int = Field(..., description="Number of samples evaluated")


class AlertSchema(BaseModel):
    """Alert status and details."""
    
    alert_type: str = Field(..., description="Type: drift, performance, latency, queue")
    triggered: bool = Field(..., description="Whether alert is active")
    reason: str = Field(default="", description="Reason for alert if triggered")
    threshold: float = Field(default=0.0, description="Threshold that triggered alert")
    current_value: float = Field(default=0.0, description="Current value")


class MonitoringStatusResponse(BaseModel):
    """Complete monitoring status."""
    
    timestamp: datetime = Field(..., description="When status was captured")
    metrics: MetricsSnapshot = Field(..., description="Current metrics snapshot")
    drift_analysis: Optional[DriftAnalysisResponse] = Field(None, description="Drift analysis")
    performance: Optional[PerformanceMetricsSchema] = Field(None, description="Performance metrics")
    alerts: List[AlertSchema] = Field(default=[], description="Active alerts")