# src/api/routers/review.py
"""Review workflow endpoint."""

from fastapi import APIRouter, HTTPException
from src.api.schemas import ReviewRequest, ReviewResponse
from src.api.service import ReviewService
import logging

logger = logging.getLogger(__name__)

router = APIRouter(tags=["review"])


@router.post(
    "/review",
    response_model=ReviewResponse,
    summary="Submit Review",
    description="Submit a manual review decision for a case"
)
def submit_review(req: ReviewRequest):
    """
    Submit a manual review decision.
    
    Args:
        req: ReviewRequest with case_id, reviewer_id, decision, note
    
    Returns:
        ReviewResponse with review details and status transitions
    
    Raises:
        HTTPException 400: If validation fails or case not found
        HTTPException 500: If database operation fails
    """
    try:
        service = ReviewService()
        result = service.submit_review(
            case_id=req.case_id,
            reviewer_id=req.reviewer_id,
            decision=req.decision,
            note=req.note,
        )
        return result
    
    except ValueError as e:
        logger.warning(f"Review submission failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        logger.error(f"Unexpected error during review: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")