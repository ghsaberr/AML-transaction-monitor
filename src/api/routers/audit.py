# src/api/routers/audit.py
"""Audit trail endpoint."""

from fastapi import APIRouter, HTTPException
from src.api.schemas import AuditTrailResponse, AuditEvent
from src.api.service import ReviewService
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get(
    "/audit/{case_id}",
    response_model=AuditTrailResponse,
    summary="Get Audit Trail",
    description="Retrieve complete audit trail for a case",
    tags=["audit"]
)
def get_audit_trail(case_id: str):
    """
    Get audit trail for a case.
    
    Args:
        case_id: Case identifier
    
    Returns:
        AuditTrailResponse with all events in chronological order
    
    Raises:
        HTTPException 404: If case not found
        HTTPException 500: If retrieval fails
    """
    try:
        service = ReviewService()
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


@router.get(
    "/cases",
    summary="List Cases",
    description="Get list of cases, optionally filtered by status",
    tags=["cases"]
)
def list_cases(status: str = None, limit: int = 100):
    """
    List cases with optional filtering by review status.
    
    Args:
        status: Filter by review status (queued_for_review, approved, rejected, escalated)
        limit: Maximum number of cases to return (default: 100)
    
    Returns:
        List of case records
    
    Raises:
        HTTPException 400: If invalid status provided
        HTTPException 500: If retrieval fails
    """
    try:
        from src.storage import get_db
        
        db = get_db()
        
        valid_statuses = {'queued_for_review', 'approved', 'rejected', 'escalated'}
        
        if status is not None:
            if status not in valid_statuses:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid status. Must be one of {valid_statuses}"
                )
            cases = db.get_cases_by_status(status, limit=limit)
        else:
            # Get all cases
            counts = db.get_case_count_by_status()
            cases = []
            for s in valid_statuses:
                cases.extend(db.get_cases_by_status(s, limit=limit))
        
        return {
            "total_cases": len(cases),
            "cases": cases,
            "timestamp": datetime.utcnow().isoformat(),
        }
    
    except HTTPException:
        raise
    
    except Exception as e:
        logger.error(f"Failed to list cases: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get(
    "/cases/{case_id}",
    summary="Get Case Details",
    description="Get complete details for a single case",
    tags=["cases"]
)
def get_case_details(case_id: str):
    """
    Get full case details including score, status, and recent audit events.
    
    Args:
        case_id: Case identifier
    
    Returns:
        Case record with associated metadata
    
    Raises:
        HTTPException 404: If case not found
        HTTPException 500: If retrieval fails
    """
    try:
        from src.storage import get_db
        
        db = get_db()
        case = db.get_case(case_id)
        
        if case is None:
            raise HTTPException(status_code=404, detail=f"Case {case_id} not found")
        
        # Get recent audit trail
        audit_trail = db.get_audit_trail(case_id)
        
        return {
            "case": case,
            "audit_events": audit_trail[-10:],  # Last 10 events
            "timestamp": datetime.utcnow().isoformat(),
        }
    
    except HTTPException:
        raise
    
    except Exception as e:
        logger.error(f"Failed to get case details: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get(
    "/review-stats",
    summary="Review Workflow Statistics",
    description="Get statistics on review workflow (queue sizes, decision counts)",
    tags=["cases"]
)
def get_review_stats():
    """
    Get review workflow statistics and queue health.
    
    Returns:
        Dictionary with case counts by status, review rates, etc.
    """
    try:
        from src.storage import get_db
        
        db = get_db()
        counts = db.get_case_count_by_status()
        
        total_cases = sum(counts.values())
        queued = counts.get('queued_for_review', 0)
        approved = counts.get('approved', 0)
        rejected = counts.get('rejected', 0)
        escalated = counts.get('escalated', 0)
        
        return {
            "total_cases": total_cases,
            "queued_for_review": queued,
            "approved": approved,
            "rejected": rejected,
            "escalated": escalated,
            "review_rate": approved / total_cases if total_cases > 0 else 0.0,
            "approval_rate": approved / (approved + rejected) if (approved + rejected) > 0 else 0.0,
            "escalation_rate": escalated / total_cases if total_cases > 0 else 0.0,
            "queue_health": "ok" if queued < (total_cases * 0.5) else "warning" if queued < (total_cases * 0.8) else "critical",
            "timestamp": datetime.utcnow().isoformat(),
        }
    
    except Exception as e:
        logger.error(f"Failed to get review stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")