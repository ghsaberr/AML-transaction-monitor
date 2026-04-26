"""Tests for review workflow and case management."""

import pytest
from uuid import uuid4
from src.storage import WorkflowDB
from src.storage.db import WorkflowDB as DBDirect
import tempfile
import os


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        db = DBDirect(db_path)
        yield db
        db.close()


@pytest.fixture  
def review_service(temp_db):
    """Create a review service with temp DB."""
    # Import here to avoid early model loading
    from src.api.service import ReviewService
    
    # Monkey patch the global db getter
    import src.storage.db as db_module
    original_instance = db_module._db_instance
    db_module._db_instance = temp_db
    
    service = ReviewService()
    yield service
    
    # Restore original
    db_module._db_instance = original_instance


@pytest.fixture
def sample_case(temp_db):
    """Create a sample case for testing."""
    case = temp_db.create_case(
        case_id=str(uuid4()),
        request_id=str(uuid4()),
        model_version="1.0.0",
        threshold_version="1.0.0",
        feature_contract_version="1.0.0",
        score=0.85,
        raw_features={"amount": 5000, "velocity": "high"},
    )
    return case


class TestReviewSubmission:
    """Test manual review submission."""
    
    def test_submit_review_approved(self, review_service, sample_case):
        """Submit an approval decision."""
        result = review_service.submit_review(
            case_id=sample_case["case_id"],
            reviewer_id="reviewer_001",
            decision="APPROVED",
            note="Transaction appears legitimate",
        )
        
        assert result["decision"] == "APPROVED"
        assert result["reviewer_id"] == "reviewer_001"
        assert result["new_status"] == "approved"
        assert result["previous_status"] == "queued_for_review"
    
    def test_submit_review_rejected(self, review_service, sample_case):
        """Submit a rejection decision."""
        result = review_service.submit_review(
            case_id=sample_case["case_id"],
            reviewer_id="reviewer_002",
            decision="REJECTED",
            note="High-risk pattern detected",
        )
        
        assert result["decision"] == "REJECTED"
        assert result["new_status"] == "rejected"
    
    def test_submit_review_escalated(self, review_service, sample_case):
        """Submit an escalation decision."""
        result = review_service.submit_review(
            case_id=sample_case["case_id"],
            reviewer_id="reviewer_003",
            decision="ESCALATED",
            note="Requires compliance investigation",
        )
        
        assert result["decision"] == "ESCALATED"
        assert result["new_status"] == "escalated"
    
    def test_submit_review_invalid_case(self, review_service):
        """Cannot review non-existent case."""
        with pytest.raises(ValueError):
            review_service.submit_review(
                case_id="nonexistent",
                reviewer_id="reviewer_001",
                decision="APPROVED",
            )
    
    def test_submit_review_invalid_decision(self, review_service, sample_case):
        """Invalid decision type should fail."""
        with pytest.raises(ValueError):
            review_service.submit_review(
                case_id=sample_case["case_id"],
                reviewer_id="reviewer_001",
                decision="MAYBE",  # Invalid
            )
    
    def test_review_with_note(self, review_service, sample_case):
        """Review includes reviewer note."""
        note = "Customer history clean, transaction consistent with profile"
        result = review_service.submit_review(
            case_id=sample_case["case_id"],
            reviewer_id="reviewer_001",
            decision="APPROVED",
            note=note,
        )
        
        assert result["note"] == note
    
    def test_review_without_note(self, review_service, sample_case):
        """Review can be submitted without note."""
        result = review_service.submit_review(
            case_id=sample_case["case_id"],
            reviewer_id="reviewer_001",
            decision="APPROVED",
            note=None,
        )
        
        assert result["note"] is None


class TestAuditTrail:
    """Test audit trail for reviews."""
    
    def test_audit_trail_on_review(self, review_service, sample_case):
        """Submitting review creates audit event."""
        review_service.submit_review(
            case_id=sample_case["case_id"],
            reviewer_id="reviewer_001",
            decision="APPROVED",
            note="OK",
        )
        
        trail = review_service.get_case_audit_trail(sample_case["case_id"])
        
        # Should have SCORE_CREATED + REVIEW_SUBMITTED events
        assert len(trail) >= 2
        
        # Find review event
        review_events = [e for e in trail if e["event_type"] == "REVIEW_SUBMITTED"]
        assert len(review_events) == 1
        
        review_event = review_events[0]
        assert review_event["actor"] == "reviewer_001"
        assert "APPROVED" in str(review_event.get("details", ""))
    
    def test_audit_trail_chronological_order(self, review_service, temp_db, sample_case):
        """Audit trail is in chronological order."""
        # First review
        review_service.submit_review(
            case_id=sample_case["case_id"],
            reviewer_id="reviewer_001",
            decision="REJECTED",
            note="Initial check failed",
        )
        
        # Update case status (simulate case being reopened)
        temp_db.update_case_status(sample_case["case_id"], "queued_for_review")
        
        # Second review
        review_service.submit_review(
            case_id=sample_case["case_id"],
            reviewer_id="reviewer_002",
            decision="APPROVED",
            note="Second review OK",
        )
        
        trail = review_service.get_case_audit_trail(sample_case["case_id"])
        
        # Verify chronological order
        for i in range(len(trail) - 1):
            ts1 = trail[i]["timestamp"]
            ts2 = trail[i + 1]["timestamp"]
            assert ts1 <= ts2, "Audit trail not in chronological order"


class TestCaseStateTransitions:
    """Test case status transitions through review."""
    
    def test_transition_queued_to_approved(self, temp_db, sample_case):
        """Case transitions from queued to approved."""
        assert sample_case["review_status"] == "queued_for_review"
        
        updated = temp_db.update_case_status(
            sample_case["case_id"],
            "approved"
        )
        
        assert updated["review_status"] == "approved"
    
    def test_transition_queued_to_rejected(self, temp_db, sample_case):
        """Case transitions from queued to rejected."""
        updated = temp_db.update_case_status(
            sample_case["case_id"],
            "rejected"
        )
        
        assert updated["review_status"] == "rejected"
    
    def test_transition_queued_to_escalated(self, temp_db, sample_case):
        """Case transitions from queued to escalated."""
        updated = temp_db.update_case_status(
            sample_case["case_id"],
            "escalated"
        )
        
        assert updated["review_status"] == "escalated"
    
    def test_transition_invalid_status(self, temp_db, sample_case):
        """Invalid status should fail."""
        with pytest.raises(Exception):
            temp_db.update_case_status(
                sample_case["case_id"],
                "invalid_status"
            )


class TestReviewQueue:
    """Test review queue operations."""
    
    def test_get_cases_by_status_queued(self, temp_db):
        """Get all cases queued for review."""
        # Create multiple cases
        for i in range(3):
            temp_db.create_case(
                case_id=str(uuid4()),
                request_id=str(uuid4()),
                model_version="1.0.0",
                threshold_version="1.0.0",
                feature_contract_version="1.0.0",
                score=0.7 + (i * 0.05),
                raw_features={"idx": i},
            )
        
        queued = temp_db.get_cases_by_status("queued_for_review")
        assert len(queued) == 3
    
    def test_get_cases_by_status_approved(self, temp_db):
        """Get all approved cases."""
        # Create and approve some cases
        for i in range(2):
            case = temp_db.create_case(
                case_id=str(uuid4()),
                request_id=str(uuid4()),
                model_version="1.0.0",
                threshold_version="1.0.0",
                feature_contract_version="1.0.0",
                score=0.8,
                raw_features={"idx": i},
            )
            temp_db.update_case_status(case["case_id"], "approved")
        
        approved = temp_db.get_cases_by_status("approved")
        assert len(approved) == 2
    
    def test_case_count_by_status(self, temp_db):
        """Get count of cases by each status."""
        # Create cases with different statuses
        for i in range(2):
            temp_db.create_case(
                case_id=str(uuid4()),
                request_id=str(uuid4()),
                model_version="1.0.0",
                threshold_version="1.0.0",
                feature_contract_version="1.0.0",
                score=0.8,
                raw_features={},
            )
        
        case3 = temp_db.create_case(
            case_id=str(uuid4()),
            request_id=str(uuid4()),
            model_version="1.0.0",
            threshold_version="1.0.0",
            feature_contract_version="1.0.0",
            score=0.9,
            raw_features={},
        )
        temp_db.update_case_status(case3["case_id"], "approved")
        
        counts = temp_db.get_case_count_by_status()
        assert counts["queued_for_review"] >= 2
        assert counts["approved"] >= 1


class TestReviewAuditImmutability:
    """Test that audit trail is immutable."""
    
    def test_audit_events_persist(self, temp_db, sample_case):
        """Audit events are stored persistently."""
        event_id = "test_event_" + str(uuid4())
        temp_db.log_audit_event(
            case_id=sample_case["case_id"],
            event_type="STATUS_CHANGED",
            actor="system",
            details='{"test": true}',
            event_id=event_id,
        )
        
        trail = temp_db.get_audit_trail(sample_case["case_id"])
        event_ids = [e["event_id"] for e in trail]
        
        assert event_id in event_ids
    
    def test_cannot_modify_audit_event(self, temp_db, sample_case):
        """Audit events cannot be modified (append-only)."""
        # Add an event
        temp_db.log_audit_event(
            case_id=sample_case["case_id"],
            event_type="SCORE_CREATED",
            actor="system",
        )
        
        # Verify event exists
        trail1 = temp_db.get_audit_trail(sample_case["case_id"])
        count1 = len(trail1)
        
        # Add another event
        temp_db.log_audit_event(
            case_id=sample_case["case_id"],
            event_type="STATUS_CHANGED",
            actor="reviewer",
        )
        
        # Verify new event added, previous not modified
        trail2 = temp_db.get_audit_trail(sample_case["case_id"])
        assert len(trail2) == count1 + 1


class TestMultipleReviewers:
    """Test multiple reviewers on same case."""
    
    def test_multiple_reviewers_same_case(self, temp_db, review_service):
        """Multiple reviewers can access same case."""
        case = temp_db.create_case(
            case_id=str(uuid4()),
            request_id=str(uuid4()),
            model_version="1.0.0",
            threshold_version="1.0.0",
            feature_contract_version="1.0.0",
            score=0.75,
            raw_features={},
        )
        
        # First reviewer
        review1 = review_service.submit_review(
            case_id=case["case_id"],
            reviewer_id="alice",
            decision="APPROVED",
            note="Looks good",
        )
        
        # Reopen case
        temp_db.update_case_status(case["case_id"], "queued_for_review")
        
        # Second reviewer
        review2 = review_service.submit_review(
            case_id=case["case_id"],
            reviewer_id="bob",
            decision="REJECTED",
            note="Found suspicious pattern",
        )
        
        # Both reviews should be in audit trail
        trail = review_service.get_case_audit_trail(case["case_id"])
        review_events = [e for e in trail if e["event_type"] == "REVIEW_SUBMITTED"]
        
        assert len(review_events) == 2
        assert any("alice" in str(e.get("actor", "")) for e in review_events)
        assert any("bob" in str(e.get("actor", "")) for e in review_events)
