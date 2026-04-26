# src/api/routers/health.py
"""Health check endpoint."""

from fastapi import APIRouter, HTTPException
from src.api.schemas import HealthResponse
from src.api.service import HealthService

router = APIRouter(tags=["health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    description="Get service health status and version information"
)
def health():
    """
    Health check endpoint with comprehensive status.
    
    Returns:
        HealthResponse with status, versions, and component health
    """
    # Import here to avoid circular imports
    from src.api.main import get_model_runner
    
    try:
        model_runner = get_model_runner()
        service = HealthService(model_runner)
        return service.get_health_status()
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))