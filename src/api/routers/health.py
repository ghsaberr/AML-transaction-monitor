# src/api/routers/health.py
"""Health check endpoint."""

from fastapi import APIRouter, Depends
from src.api.schemas import HealthResponse
from src.api.service import HealthService
from src.agent.model_runner import ModelRunner

router = APIRouter(tags=["health"])


def get_health_service(model_runner: ModelRunner) -> HealthService:
    """Dependency injection for health service."""
    return HealthService(model_runner)


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    description="Get service health status and version information"
)
def health(service: HealthService = Depends(get_health_service)):
    """
    Health check endpoint with comprehensive status.
    
    Returns:
        HealthResponse with status, versions, and component health
    """
    return service.get_health_status()