# tests/test_api_contracts.py
"""Tests for API schema contracts and validation."""

import pytest
from datetime import datetime
from src.api.schemas import (
    ScoreRequest, ScoreResponse,
    ExplainRequest, ExplainResponse,
    ReviewRequest, ReviewResponse,
    AuditTrailResponse, AuditEvent,
    HealthResponse,
)


class TestScoreContracts:
    """Test ScoreRequest and ScoreResponse contracts."""
    
    def test_score_request_valid(self):
        """Valid score request."""
        req = ScoreRequest(
            tx_features={"feature1": 1.0, "feature2": 2.0}
        )
        assert req.tx_features == {"feature1": 1.0, "feature2": 2.0}
    
    def test_score_request_empty_features_fails(self):
        """Empty features should fail."""
        with pytest.raises(ValueError):
            ScoreRequest(tx_features={})
    
    def test_score_response_valid(self):
        """Valid score response with all required fields."""
        resp = ScoreResponse(
            case_id="case-123",
            request_id="req-123",
            score=0.75,
            review_flag=True,
            decision="ALERT",
            threshold_used=0.7,
            model_version="1.0.0",
            threshold_version="1.0.0",
            feature_contract_version="1.0.0",
            review_status="queued_for_review",
            timestamp=datetime.utcnow(),
        )
        assert resp.case_id == "case-123"
        assert resp.score == 0.75
        assert resp.review_flag is True
        assert resp.decision == "ALERT"
        assert resp.model_version == "1.0.0"
    
    def test_score_response_score_range(self):
        """Score must be in [0, 1]."""
        # Valid
        resp = ScoreResponse(
            case_id="case-123",
            request_id="req-123",
            score=0.5,
            review_flag=False,
            decision="PASS",
            threshold_used=0.7,
            model_version="1.0.0",
            threshold_version="1.0.0",
            feature_contract_version="1.0.0",
            timestamp=datetime.utcnow(),
        )
        assert resp.score == 0.5
        
        # Invalid - exceeds range
        with pytest.raises(ValueError):
            ScoreResponse(
                case_id="case-123",
                request_id="req-123",
                score=1.5,  # Invalid
                review_flag=False,
                decision="PASS",
                threshold_used=0.7,
                model_version="1.0.0",
                threshold_version="1.0.0",
                feature_contract_version="1.0.0",
                timestamp=datetime.utcnow(),
            )


class TestReviewContracts:
    """Test ReviewRequest and ReviewResponse contracts."""
    
    def test_review_request_valid(self):
        """Valid review request."""
        req = ReviewRequest(
            case_id="case-123",
            reviewer_id="reviewer-1",
            decision="APPROVED",
            note="Looks legitimate after manual check",
        )
        assert req.decision == "APPROVED"
        assert req.note == "Looks legitimate after manual check"
    
    def test_review_request_invalid_decision(self):
        """Invalid decision should fail."""
        with pytest.raises(ValueError):
            ReviewRequest(
                case_id="case-123",
                reviewer_id="reviewer-1",
                decision="INVALID_DECISION",  # Invalid
            )
    
    def test_review_request_valid_decisions(self):
        """All valid decisions should pass."""
        for decision in ["APPROVED", "REJECTED", "ESCALATED"]:
            req = ReviewRequest(
                case_id="case-123",
                reviewer_id="reviewer-1",
                decision=decision,
            )
            assert req.decision == decision
    
    def test_review_response_has_status_transition(self):
        """Review response should include status transition."""
        resp = ReviewResponse(
            review_id="review-123",
            case_id="case-123",
            reviewer_id="reviewer-1",
            decision="APPROVED",
            note="Legitimate",
            previous_status="queued_for_review",
            new_status="approved",
            timestamp=datetime.utcnow(),
        )
        assert resp.previous_status == "queued_for_review"
        assert resp.new_status == "approved"


class TestAuditContracts:
    """Test audit trail contracts."""
    
    def test_audit_event_valid(self):
        """Valid audit event."""
        event = AuditEvent(
            event_id="event-123",
            case_id="case-123",
            event_type="SCORE_CREATED",
            actor="system",
            details={"score": 0.75},
            timestamp=datetime.utcnow(),
        )
        assert event.event_type == "SCORE_CREATED"
        assert event.details == {"score": 0.75}
    
    def test_audit_trail_response_valid(self):
        """Valid audit trail response."""
        events = [
            AuditEvent(
                event_id="event-1",
                case_id="case-123",
                event_type="SCORE_CREATED",
                actor="system",
                details=None,
                timestamp=datetime.utcnow(),
            ),
            AuditEvent(
                event_id="event-2",
                case_id="case-123",
                event_type="REVIEW_SUBMITTED",
                actor="reviewer-1",
                details={"decision": "APPROVED"},
                timestamp=datetime.utcnow(),
            ),
        ]
        
        resp = AuditTrailResponse(
            case_id="case-123",
            events=events,
            total_events=2,
            timestamp=datetime.utcnow(),
        )
        assert len(resp.events) == 2
        assert resp.total_events == 2


class TestHealthContracts:
    """Test health endpoint contracts."""
    
    def test_health_response_valid(self):
        """Valid health response."""
        resp = HealthResponse(
            status="ok",
            version="0.1.0",
            model_version="1.0.0",
            model_status="ready",
            threshold_version="1.0.0",
            feature_contract_version="1.0.0",
            database_status="ok",
            timestamp=datetime.utcnow(),
        )
        assert resp.status == "ok"
        assert resp.model_status == "ready"
        assert resp.database_status == "ok"
    
    def test_health_response_degraded(self):
        """Health response when degraded."""
        resp = HealthResponse(
            status="degraded",
            version="0.1.0",
            model_version="1.0.0",
            model_status="error",
            threshold_version="1.0.0",
            feature_contract_version="1.0.0",
            database_status="ok",
            timestamp=datetime.utcnow(),
        )
        assert resp.status == "degraded"
        assert resp.model_status == "error"