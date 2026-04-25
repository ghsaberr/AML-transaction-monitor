# src/api/routers/metrics.py
"""Metrics endpoint (stub)."""

from fastapi import APIRouter
from src.api.schemas import MetricsResponse, MetricsSnapshot
from datetime import datetime

router = APIRouter(tags=["metrics"])


@router.get(
    "/metrics",
    response_model=MetricsResponse,
    summary="Get Metrics",
    description="Get system metrics snapshot"
)
def get_metrics():
    """
    Get system metrics.
    
    NOTE: This is a stub for Phase 4 (Monitoring & Observability).
    Currently returns placeholder values.
    
    Returns:
        MetricsResponse with system metrics
    """
    snapshot = MetricsSnapshot(
        request_count=0,
        success_count=0,
        error_count=0,
        average_latency_ms=0.0,
        p95_latency_ms=0.0,
        p99_latency_ms=0.0,
        score_mean=None,
        score_median=None,
        score_std=None,
        review_rate=0.0,
    )
    
    return MetricsResponse(
        snapshot=snapshot,
        timestamp=datetime.utcnow(),
    )