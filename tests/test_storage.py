# tests/test_storage.py
"""Tests for database storage layer."""

import pytest
import tempfile
import os
from pathlib import Path
from src.storage import WorkflowDB, DatabaseError


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        db = WorkflowDB(db_path)
        yield db
        db.close()


class TestCaseOperations:
    """Test case CRUD operations."""
    
    def test_create_case(self, temp_db):
        """Create a new case."""
        case = temp_db.create_case(
            case_id="case-123",
            request_id="req-123",
            model_version="1.0.0",
            threshold_version="1.0.0",
            feature_contract_version="1.0.0",
            score=0.75,
            raw_features={"amount": 1000, "time": "2024-01-01"},
        )
        
        assert case["case_id"] == "case-123"
        assert case["score"] == 0.75
        assert case["review_status"] == "queued_for_review"
    
    def test_get_case(self, temp_db):
        """Retrieve a case."""
        temp_db.create_case(
            case_id="case-123",
            request_id="req-123",
            model_version="1.0.0",
            threshold_version="1.0.0",
            feature_contract_version="1.0.0",
            score=0.75,
            raw_features={},
        )
        
        case = temp_db.get_case("case-123")
        assert case is not None
        assert case["case_id"] == "case-123"
        assert case["score"] == 0.75
    
    def test_get_nonexistent_case(self, temp_db):
        """Get non-existent case returns None."""
        case = temp_db.get_case("nonexistent")
        assert case is None
    
    def test_update_case_status(self, temp_db):
        """Update case status."""
        temp_db.create_case(
            case_id="case-123",
            request_id="req-123",
            model_version="1.0.0",
            threshold_version="1.0.0",
            feature_contract_version="1.0.0",
            score=0.75,
            raw_features={},
        )
        
        updated = temp_db.update_case_status("case-123", "approved")
        assert updated["review_status"] == "approved"
        
        # Verify persistence
        case = temp_db.get_case("case-123")
        assert case["review_status"] == "approved"
    
    def test_update_case_invalid_status(self, temp_db):
        """Invalid status should raise error."""
        temp_db.create_case(
            case_id="case-123",
            request_id="req-123",
            model_version="1.0.0",
            threshold_version="1.0.0",
            feature_contract_version="1.0.0",
            score=0.75,
            raw_features={},
        )
        
        with pytest.raises(DatabaseError):
            temp_db.update_case_status("case-123", "invalid_status")


class TestAuditTrail:
    """Test audit trail operations."""
    
    def test_audit_event_creation(self, temp_db):
        """Create audit event."""
        temp_db.create_case(
            case_id="case-123",
            request_id="req-123",
            model_version="1.0.0",
            threshold_version="1.0.0",
            feature_contract_version="1.0.0",
            score=0.75,
            raw_features={},
        )
        
        event = temp_db.log_audit_event(
            case_id="case-123",
            event_type="SCORE_CREATED",
            actor="system",
            details='{"score": 0.75}',
        )
        
        assert event["event_type"] == "SCORE_CREATED"
    
    def test_audit_trail_chronological(self, temp_db):
        """Audit trail returns events in chronological order."""
        temp_db.create_case(
            case_id="case-123",
            request_id="req-123",
            model_version="1.0.0",
            threshold_version="1.0.0",
            feature_contract_version="1.0.0",
            score=0.75,
            raw_features={},
        )
        
        # Events are auto-created on case creation
        trail = temp_db.get_audit_trail("case-123")
        assert len(trail) >= 1
        
        # Add another event
        temp_db.log_audit_event(
            case_id="case-123",
            event_type="STATUS_CHANGED",
            actor="system",
        )
        
        trail = temp_db.get_audit_trail("case-123")
        assert len(trail) >= 2
        # Verify chronological order
        for i in range(len(trail) - 1):
            assert trail[i]["timestamp"] <= trail[i + 1]["timestamp"]


class TestReviewWorkflow:
    """Test review recording."""
    
    def test_record_review(self, temp_db):
        """Record a review decision."""
        temp_db.create_case(
            case_id="case-123",
            request_id="req-123",
            model_version="1.0.0",
            threshold_version="1.0.0",
            feature_contract_version="1.0.0",
            score=0.75,
            raw_features={},
        )
        
        review = temp_db.record_review(
            review_id="review-1",
            case_id="case-123",
            reviewer_id="reviewer-1",
            decision="APPROVED",
            note="Looks good",
        )
        
        assert review["decision"] == "APPROVED"
        assert review["reviewer_id"] == "reviewer-1"
        
        # Verify case status updated
        case = temp_db.get_case("case-123")
        assert case["review_status"] == "approved"
    
    def test_review_creates_audit_event(self, temp_db):
        """Recording review creates audit event."""
        temp_db.create_case(
            case_id="case-123",
            request_id="req-123",
            model_version="1.0.0",
            threshold_version="1.0.0",
            feature_contract_version="1.0.0",
            score=0.75,
            raw_features={},
        )
        
        temp_db.record_review(
            review_id="review-1",
            case_id="case-123",
            reviewer_id="reviewer-1",
            decision="APPROVED",
        )
        
        trail = temp_db.get_audit_trail("case-123")
        review_events = [e for e in trail if e["event_type"] == "REVIEW_SUBMITTED"]
        assert len(review_events) >= 1


class TestCaseStatusFiltering:
    """Test case status queries."""
    
    def test_get_cases_by_status(self, temp_db):
        """Get cases filtered by status."""
        # Create cases with different statuses
        temp_db.create_case(
            case_id="case-1",
            request_id="req-1",
            model_version="1.0.0",
            threshold_version="1.0.0",
            feature_contract_version="1.0.0",
            score=0.75,
            raw_features={},
        )
        
        temp_db.create_case(
            case_id="case-2",
            request_id="req-2",
            model_version="1.0.0",
            threshold_version="1.0.0",
            feature_contract_version="1.0.0",
            score=0.85,
            raw_features={},
        )
        
        temp_db.update_case_status("case-2", "approved")
        
        # Query by status
        queued = temp_db.get_cases_by_status("queued_for_review")
        approved = temp_db.get_cases_by_status("approved")
        
        assert len(queued) >= 1
        assert len(approved) >= 1
    
    def test_case_count_by_status(self, temp_db):
        """Get count of cases by status."""
        temp_db.create_case(
            case_id="case-1",
            request_id="req-1",
            model_version="1.0.0",
            threshold_version="1.0.0",
            feature_contract_version="1.0.0",
            score=0.75,
            raw_features={},
        )
        
        counts = temp_db.get_case_count_by_status()
        assert "queued_for_review" in counts
        assert counts["queued_for_review"] >= 1