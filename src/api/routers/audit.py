# src/api/routers/audit.py
"""Audit trail endpoint."""

from fastapi import APIRouter, Depends, HTTPException
from src.api.schemas import AuditTrailResponse, AuditEvent
from src.api.service import ReviewService
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

router = APIRouter(tags=["audit"])


def get_review_service() -> ReviewService:
    """Dependency injection for review service."""
    return ReviewService()


@router.get(
    "/audit/{case_id}",
    response_model=AuditTrailResponse,
    summary="Get Audit Trail",
    description="Retrieve complete audit trail for a case"
)
def get_audit_trail(
    case_id: str,
    service: ReviewService = Depends(get_review_service),
):
    """
    Get audit trail for a case.
    
    Args:
        case_id: Case identifier
        service: ReviewService instance (injected)
    
    Returns:
        AuditTrailResponse with all events in chronological order
    
    Raises:
        HTTPException 404: If case not found
        HTTPException 500: If retrieval fails
    """
    try:
        events = service.get_case_audit_trail(case_id)
        
        # Convert to schema
        audit_events = [
            AuditEvent(
                event_id=event['event_id'],
                case_id=event['case_id'],
                event_type=event['event_type'],
                actor=event['actor'],
                details=event['details'],
                timestamp=event['timestamp'],
            )
            for event in events
        ]
        
        return AuditTrailResponse(
            case_id=case_id,
            events=audit_events,
            total_events=len(audit_events),
            timestamp=datetime.utcnow(),
        )
    
    except ValueError as e:
        logger.warning(f"Audit trail not found: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    
    except Exception as e:
        logger.error(f"Failed to retrieve audit trail: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")