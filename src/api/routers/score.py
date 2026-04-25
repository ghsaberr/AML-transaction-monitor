# src/api/routers/score.py
"""Scoring endpoint."""

from fastapi import APIRouter, Depends, HTTPException
from typing import Any, Dict
from src.api.schemas import ScoreRequest, ScoreResponse
from src.api.service import ScoringService
from src.agent.model_runner import ModelRunner
import logging

logger = logging.getLogger(__name__)

router = APIRouter(tags=["score"])


def get_scoring_service(model_runner: ModelRunner) -> ScoringService:
    """Dependency injection for scoring service."""
    return ScoringService(model_runner)


@router.post(
    "/score",
    response_model=ScoreResponse,
    summary="Score Transaction",
    description="Score a transaction and create a case for review if needed"
)
def score(
    req: ScoreRequest,
    service: ScoringService = Depends(get_scoring_service)
):
    """
    Score a transaction and create a persistent case.
    
    Args:
        req: ScoreRequest with tx_features
        service: ScoringService instance (injected)
    
    Returns:
        ScoreResponse with complete metadata including case_id, score,
        decision, threshold, model version, and timestamp
    
    Raises:
        HTTPException 400: If input validation fails
        HTTPException 500: If scoring or database operation fails
    """
    try:
        # Score transaction and create case
        result = service.score_and_create_case(tx_features=req.tx_features)
        return result
    
    except ValueError as e:
        # Input validation error
        logger.warning(f"Invalid input: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"Invalid features: {str(e)}"
        )
    
    except Exception as e:
        # Unexpected error
        logger.error(f"Scoring failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Internal server error during scoring"
        )