"""
End-to-End Integration Tests for AML Transaction Monitoring.

Tests complete workflows WITHOUT model dependency:
- Case creation -> Review -> Audit -> Metrics
- Review workflow with status transitions
- Performance metrics from reviews
- Queue management and statistics
"""

import pytest
import tempfile
import os
from uuid import uuid4
from src.storage import WorkflowDB
from src.api.schemas import ReviewRequest
from src.api.service import ReviewService
from src.monitoring.metrics import (
    MetricsCollector, FeatureDriftDetector, 
    ModelPerformanceCalculator, AlertingPolicy
)
from src.features.feature_contract import FeatureContract
import json


@pytest.fixture
def temp_db():
    """Create a temporary database for each test."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        db = WorkflowDB(db_path)
        yield db
        db.close()


@pytest.fixture
def review_service(temp_db):
    """Create a review service with temp DB."""
    # Monkey patch the global db getter
    import src.storage.db as db_module
    original_instance = db_module._db_instance
    db_module._db_instance = temp_db
    
    service = ReviewService()
    yield service
    
    # Restore original
    db_module._db_instance = original_instance


class TestCaseToReviewWorkflow:
    """Full workflow: Create case -> Review decision -> Audit trail verification."""

    def test_complete_case_to_review_workflow(self, temp_db, review_service):
        """End-to-end: Case Created -> Review Submitted -> Audit Trail."""
        # 1. Create a case (simulating scoring service)
        case_id = str(uuid4())
        request_id = str(uuid4())
        
        case = temp_db.create_case(
            case_id=case_id,
            request_id=request_id,
            model_version="1.0.0",
            threshold_version="1.0.0",
            feature_contract_version="1.0.0",
            score=0.72,
            raw_features={
                "transaction_amount": 5000.0,
                "transaction_count_24h": 15,
                "unique_destinations_24h": 3,
            },
        )
        
        # Verify case created in QUEUED status
        assert case is not None
        assert case["review_status"] == "queued_for_review"
        assert case["score"] == 0.72
        
        # 2. Verify case exists in database
        retrieved_case = temp_db.get_case(case_id)
        assert retrieved_case is not None
        assert retrieved_case["case_id"] == case_id
        
        # 3. Submit a review decision
        review_request = ReviewRequest(
            case_id=case_id,
            reviewer_id="reviewer_001",
            decision="APPROVED",
            note="Transaction verified as legitimate"
        )
        
        review_response = review_service.submit_review(
            case_id=case_id,
            reviewer_id="reviewer_001",
            decision="APPROVED",
            note="Transaction verified as legitimate"
        )
        
        # Verify review response
        assert review_response is not None
        assert review_response.get("case_id") == case_id
        assert review_response.get("decision") == "APPROVED"
        
        # 4. Verify case status updated
        updated_case = temp_db.get_case(case_id)
        assert updated_case["review_status"] == "approved"
        
        # 5. Verify audit trail has both events
        audit_trail = temp_db.get_audit_trail(case_id)
        assert len(audit_trail) == 2  # Case created + Review submitted
        
        # Check event types
        event_types = [e["event_type"] for e in audit_trail]
        assert "SCORE_CREATED" in event_types or "CASE_CREATED" in event_types
        assert "REVIEW_SUBMITTED" in event_types
        
        # Verify first event is case/score creation
        first_event = audit_trail[0]["event_type"]
        assert first_event in ["CASE_CREATED", "SCORE_CREATED"]

    def test_multiple_review_decisions_workflow(self, temp_db, review_service):
        """Multiple reviews on same case track independently."""
        # Create case
        case_id = str(uuid4())
        case = temp_db.create_case(
            case_id=case_id,
            request_id=str(uuid4()),
            model_version="1.0.0",
            threshold_version="1.0.0",
            feature_contract_version="1.0.0",
            score=0.85,
            raw_features={"high_risk": True},
        )
        
        # First review: ESCALATED
        review_1 = review_service.submit_review(
            case_id=case_id,
            reviewer_id="reviewer_001",
            decision="ESCALATED",
            note="Requires investigation"
        )
        
        case_after_1 = temp_db.get_case(case_id)
        assert case_after_1["review_status"] == "escalated"
        
        # Second review: REJECTED
        review_2 = review_service.submit_review(
            case_id=case_id,
            reviewer_id="reviewer_002",
            decision="REJECTED",
            note="False positive - confirmed suspicious"
        )
        
        case_after_2 = temp_db.get_case(case_id)
        assert case_after_2["review_status"] == "rejected"
        
        # Verify audit trail has 3 events (case created + 2 reviews)
        audit_trail = temp_db.get_audit_trail(case_id)
        assert len(audit_trail) == 3
        
        # Verify reviews recorded with correct actors
        reviews = [e for e in audit_trail if e["event_type"] == "REVIEW_SUBMITTED"]
        assert len(reviews) == 2
        actors = [r["actor"] for r in reviews]
        assert "reviewer_001" in actors
        assert "reviewer_002" in actors


class TestMetricsWorkflow:
    """Workflow: Multiple scoring/review events -> Metrics reflect updates."""

    def test_metrics_update_on_scoring_events(self):
        """Metrics collector updates on scoring events."""
        metrics_collector = MetricsCollector()
        
        # Record multiple scoring events with varying latencies
        metrics_collector.record_scoring_event(
            case_id="case_001",
            score=0.85,
            latency_ms=150,
            decision="ALERT"
        )
        metrics_collector.record_scoring_event(
            case_id="case_002",
            score=0.20,
            latency_ms=120,
            decision="PASS"
        )
        metrics_collector.record_scoring_event(
            case_id="case_003",
            score=0.65,
            latency_ms=200,
            decision="ALERT"
        )
        
        # Get current metrics
        metrics = metrics_collector.get_current_metrics()
        
        # Verify metrics structure
        assert "latency_ms" in metrics
        assert "cases" in metrics
        assert metrics["latency_ms"]["avg"] > 0
        assert metrics["latency_ms"]["p95"] > 0
        assert metrics["latency_ms"]["p99"] > 0

    def test_performance_metrics_from_reviews(self, temp_db, review_service):
        """Performance metrics calculated from review decisions."""
        # Create cases with different model decisions and human reviews
        
        # Case 1: Model predicted high risk (score > 0.5), Reviewer APPROVED
        case_1 = temp_db.create_case(
            case_id=str(uuid4()),
            request_id=str(uuid4()),
            model_version="1.0.0",
            threshold_version="1.0.0",
            feature_contract_version="1.0.0",
            score=0.85,  # High risk score -> ALERT
            raw_features={"high_risk": True},
        )
        
        review_service.submit_review(
            case_id=case_1["case_id"],
            reviewer_id="reviewer_001",
            decision="APPROVED",
            note="Legitimate transaction"
        )
        
        # Case 2: Model predicted low risk (score < 0.5), Reviewer REJECTED
        case_2 = temp_db.create_case(
            case_id=str(uuid4()),
            request_id=str(uuid4()),
            model_version="1.0.0",
            threshold_version="1.0.0",
            feature_contract_version="1.0.0",
            score=0.15,  # Low risk score -> PASS
            raw_features={"low_risk": True},
        )
        
        review_service.submit_review(
            case_id=case_2["case_id"],
            reviewer_id="reviewer_001",
            decision="REJECTED",
            note="Fraudulent activity"
        )
        
        # Get review audit trail
        reviews_1 = temp_db.get_audit_trail(case_1["case_id"])
        reviews_2 = temp_db.get_audit_trail(case_2["case_id"])
        all_reviews = [r for r in reviews_1 + reviews_2 if r["event_type"] == "REVIEW_SUBMITTED"]
        
        # Calculate performance metrics
        perf_calc = ModelPerformanceCalculator()
        performance = perf_calc.calculate_from_reviews(reviews=all_reviews)
        
        # Verify performance metrics calculated
        assert "precision" in performance
        assert "recall" in performance
        assert "f1" in performance
        assert "samples" in performance


class TestFullSystemIntegration:
    """End-to-end system integration covering all components."""

    def test_complete_case_lifecycle(self, temp_db, review_service):
        """Full lifecycle: Case Created -> Review -> Audit -> Metrics."""
        # 1. Create case
        case_id = str(uuid4())
        case = temp_db.create_case(
            case_id=case_id,
            request_id=str(uuid4()),
            model_version="1.0.0",
            threshold_version="1.0.0",
            feature_contract_version="1.0.0",
            score=0.72,
            raw_features={"transaction_amount": 15000.0},
        )
        
        # 2. Verify case in QUEUED status
        assert case["review_status"] == "queued_for_review"
        
        # 3. Record metrics
        metrics_collector = MetricsCollector()
        metrics_collector.record_scoring_event(
            case_id=case_id,
            score=case["score"],
            latency_ms=145,
            decision="ALERT" if case["score"] >= 0.5 else "PASS"
        )
        
        # 4. Submit review
        review_response = review_service.submit_review(
            case_id=case_id,
            reviewer_id="reviewer_001",
            decision="APPROVED",
            note="Verified legitimate"
        )
        
        # 5. Verify complete audit trail
        audit_trail = temp_db.get_audit_trail(case_id)
        assert len(audit_trail) >= 2
        event_types = [e["event_type"] for e in audit_trail]
        assert any(t in event_types for t in ["CASE_CREATED", "SCORE_CREATED"])
        assert "REVIEW_SUBMITTED" in event_types
        
        # 6. Verify metrics collected
        metrics = metrics_collector.get_current_metrics()
        assert metrics["cases"]["total"] >= 1
        
        # 7. Verify case status updated
        final_case = temp_db.get_case(case_id)
        assert final_case["review_status"] == "approved"

    def test_queue_management_workflow(self, temp_db, review_service):
        """Test queue filtering and status statistics."""
        # Create multiple cases with different statuses
        cases_data = [
            (0.25, "APPROVED"),
            (0.35, "APPROVED"),
            (0.75, "REJECTED"),
            (0.85, "ESCALATED"),
        ]
        
        for score, decision in cases_data:
            case = temp_db.create_case(
                case_id=str(uuid4()),
                request_id=str(uuid4()),
                model_version="1.0.0",
                threshold_version="1.0.0",
                feature_contract_version="1.0.0",
                score=score,
                raw_features={},
            )
            
            review_service.submit_review(
                case_id=case["case_id"],
                reviewer_id="reviewer_001",
                decision=decision,
                note=f"Decision: {decision}"
            )
        
        # Plus one pending case
        pending = temp_db.create_case(
            case_id=str(uuid4()),
            request_id=str(uuid4()),
            model_version="1.0.0",
            threshold_version="1.0.0",
            feature_contract_version="1.0.0",
            score=0.50,
            raw_features={},
        )
        
        # Get queue statistics
        counts = temp_db.get_case_count_by_status()
        assert counts["approved"] == 2
        assert counts["rejected"] == 1
        assert counts["escalated"] == 1
        assert counts["queued_for_review"] == 1
        
        # Filter by status
        queued_cases = temp_db.get_cases_by_status("queued_for_review")
        assert len(queued_cases) == 1
        
        approved_cases = temp_db.get_cases_by_status("approved")
        assert len(approved_cases) == 2


class TestDataConsistency:
    """Verify data consistency across storage and services."""

    def test_case_immutability_on_review(self, temp_db, review_service):
        """Original case data preserved; only status changes on review."""
        # Create case
        case_id = str(uuid4())
        original_case = temp_db.create_case(
            case_id=case_id,
            request_id=str(uuid4()),
            model_version="1.0.0",
            threshold_version="1.0.0",
            feature_contract_version="1.0.0",
            score=0.72,
            raw_features={"test": "data"},
        )
        
        # Record original values
        original_score = original_case["score"]
        original_version = original_case["model_version"]
        
        # Submit review
        review_service.submit_review(
            case_id=case_id,
            reviewer_id="reviewer_001",
            decision="APPROVED",
            note="Test"
        )
        
        # Verify score and version unchanged
        updated_case = temp_db.get_case(case_id)
        assert updated_case["score"] == original_score
        assert updated_case["model_version"] == original_version
        assert updated_case["review_status"] == "approved"  # Only status changed
        
        # Verify audit shows transition
        audit = temp_db.get_audit_trail(case_id)
        review_events = [e for e in audit if e["event_type"] == "REVIEW_SUBMITTED"]
        assert len(review_events) > 0

    def test_audit_trail_immutability(self, temp_db, review_service):
        """Audit trail events are immutable and chronological."""
        case_id = str(uuid4())
        
        # Create case
        temp_db.create_case(
            case_id=case_id,
            request_id=str(uuid4()),
            model_version="1.0.0",
            threshold_version="1.0.0",
            feature_contract_version="1.0.0",
            score=0.50,
            raw_features={},
        )
        
        # Get initial audit trail
        audit_1 = temp_db.get_audit_trail(case_id)
        initial_count = len(audit_1)
        
        # Submit first review
        review_service.submit_review(
            case_id=case_id,
            reviewer_id="reviewer_001",
            decision="ESCALATED",
            note="Needs investigation"
        )
        
        # Submit second review
        review_service.submit_review(
            case_id=case_id,
            reviewer_id="reviewer_002",
            decision="REJECTED",
            note="Confirmed suspicious"
        )
        
        # Verify audit trail grew correctly
        audit_2 = temp_db.get_audit_trail(case_id)
        assert len(audit_2) == initial_count + 2
        
        # Verify chronological order preserved
        timestamps = [e["timestamp"] for e in audit_2]
        assert timestamps == sorted(timestamps)


class TestDriftMonitoring:
    """Drift detection functionality."""

    def test_drift_detector_initialization(self):
        """Drift detector can be initialized with metadata."""
        training_metadata = {
            "means": {"feature1": 0.5, "feature2": 1.0},
            "mins": {"feature1": 0.0, "feature2": 0.0},
            "maxes": {"feature1": 1.0, "feature2": 2.0},
        }
        detector = FeatureDriftDetector(training_metadata)
        assert detector is not None
        
        # Test KS statistic computation
        result = detector.compute_ks_statistic(
            feature_name="feature1",
            current_values=[0.3, 0.4, 0.5, 0.6, 0.7],
            percentiles=100
        )
        
        # Verify result structure
        assert "ks_statistic" in result
        assert "p_value" in result
        assert "alert" in result
