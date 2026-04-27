# src/api/routers/metrics.py
"""Metrics endpoint with drift detection and performance monitoring."""

from fastapi import APIRouter
from src.api.schemas import MetricsResponse, MetricsSnapshot
from src.monitoring import MetricsCollector, AlertingPolicy
from datetime import datetime

router = APIRouter(tags=["metrics"])

# Singleton instances
_collector = MetricsCollector()
_policy = AlertingPolicy()


@router.get(
    "/metrics",
    response_model=MetricsResponse,
    summary="Get Metrics",
    description="Get system metrics with drift detection and performance monitoring"
)
def get_metrics():
    """
    Get comprehensive system metrics.
    
    Returns:
        MetricsResponse with latency, throughput, performance, and drift metrics
    """
    # Get current metrics
    metrics = _collector.get_current_metrics()
    
    # Get drift analysis
    drift = _collector.get_drift_analysis(feature_contract=None, window_hours=24)
    
    # Build snapshot
    snapshot = MetricsSnapshot(
        # Latency metrics
        request_count=metrics["cases"]["total"],
        success_count=metrics["cases"]["approved"] + metrics["cases"]["escalated"],
        error_count=metrics["cases"]["rejected"],
        average_latency_ms=metrics["latency_ms"]["avg"],
        p95_latency_ms=metrics["latency_ms"]["p95"],
        p99_latency_ms=metrics["latency_ms"]["p99"],
        
        # Score statistics
        score_mean=metrics["scores"]["mean"],
        score_median=metrics["scores"]["median"],
        score_std=metrics["scores"]["std"],
        
        # Case counts
        cases_total=metrics["cases"]["total"],
        cases_queued=metrics["cases"]["queued"],
        cases_approved=metrics["cases"]["approved"],
        cases_rejected=metrics["cases"]["rejected"],
        cases_escalated=metrics["cases"]["escalated"],
        
        # Drift metrics
        drift_features_analyzed=drift["features_analyzed"],
        drift_features_alert=drift["features_drifting"],
        drift_alert=drift["alert"],
        
        # Review rate
        review_rate=metrics["cases"]["total"] - metrics["cases"]["queued"] / max(metrics["cases"]["total"], 1),
    )
    
    return MetricsResponse(
        snapshot=snapshot,
        timestamp=datetime.utcnow(),
    )