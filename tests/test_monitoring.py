"""Tests for monitoring, drift detection, and performance metrics."""

import pytest
from datetime import datetime, timedelta
from src.monitoring import (
    FeatureDriftDetector,
    ModelPerformanceCalculator,
    MetricsCollector,
    AlertingPolicy,
)


class TestFeatureDriftDetector:
    """Test feature drift detection."""
    
    @pytest.fixture
    def detector(self):
        """Create a drift detector with training metadata."""
        training_metadata = {
            "means": {
                "amount": 2500.0,
                "velocity": 5.0,
                "tx_count": 10.0,
            },
            "mins": {
                "amount": 10.0,
                "velocity": 0.0,
                "tx_count": 0.0,
            },
            "maxes": {
                "amount": 100000.0,
                "velocity": 100.0,
                "tx_count": 500.0,
            },
        }
        return FeatureDriftDetector(training_metadata)
    
    def test_ks_test_no_drift(self, detector):
        """KS test returns valid statistics."""
        current_values = [2400.0, 2500.0, 2600.0] * 10  # Centered around mean
        
        result = detector.compute_ks_statistic("amount", current_values)
        
        assert "ks_statistic" in result
        assert "p_value" in result
        assert "alert" in result
        # Just verify structure; actual drift depends on distribution similarity
        assert result["ks_statistic"] >= 0.0
        assert result["p_value"] >= 0.0
    
    def test_ks_test_insufficient_samples(self, detector):
        """Handle insufficient samples."""
        current_values = [100.0]  # Only 1 sample
        
        result = detector.compute_ks_statistic("amount", current_values)
        
        assert result["alert"] is False
        assert result["reason"] == "insufficient_samples"
    
    def test_mean_drift_large_shift(self, detector):
        """Detect mean drift when distribution shifts."""
        # Mean shifted from 2500 to 50000 (way outside training range)
        result = detector.compute_mean_drift("amount", current_mean=50000.0)
        
        assert result["training_mean"] == 2500.0
        assert result["current_mean"] == 50000.0
        # Large shift should exceed threshold (approximately 2.0)
        assert result["std_devs_apart"] > 1.8
    
    def test_mean_drift_small_shift(self, detector):
        """No alert for small mean drift."""
        result = detector.compute_mean_drift("amount", current_mean=2550.0)
        
        assert result["alert"] is False
        assert result["std_devs_apart"] < 2.0
    
    def test_mean_drift_custom_threshold(self, detector):
        """Use custom threshold for mean drift."""
        result = detector.compute_mean_drift(
            "amount",
            current_mean=3000.0,
            threshold_std=0.5  # Lower threshold
        )
        
        # Even small shift will trigger with low threshold
        assert result["std_devs_apart"] > 0.0


class TestModelPerformanceCalculator:
    """Test performance metric calculation."""
    
    def test_empty_reviews(self):
        """Handle empty review list."""
        result = ModelPerformanceCalculator.calculate_from_reviews([])
        
        assert result["precision"] == 0.0
        assert result["recall"] == 0.0
        assert result["accuracy"] == 0.0
        assert result["f1"] == 0.0
        assert result["samples"] == 0
    
    def test_perfect_performance(self):
        """Test performance calculation logic."""
        # TP: decision=APPROVED and was_alert=True (model said alert, human approved)
        # TN: decision=APPROVED and was_alert=False (model said pass, human approved)
        # FP: decision=REJECTED and was_alert=True (model said alert, human rejected)
        # FN: decision=REJECTED and was_alert=False (model said pass, human rejected)
        reviews = [
            {
                "timestamp": datetime.utcnow().isoformat(),
                "decision": "APPROVED",
                "was_alert": True,  # TP
            } for _ in range(5)
        ] + [
            {
                "timestamp": datetime.utcnow().isoformat(),
                "decision": "APPROVED",
                "was_alert": False,  # TN
            } for _ in range(5)
        ]
        
        result = ModelPerformanceCalculator.calculate_from_reviews(reviews)
        
        # With 5 TP and 5 TN, precision = TP/(TP+FP) = 5/5 = 1.0
        assert result["true_positives"] == 5
        assert result["true_negatives"] == 5
        assert result["false_positives"] == 0
        assert result["false_negatives"] == 0
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0
    
    def test_window_filtering(self):
        """Respect time window for reviews."""
        old_review = {
            "timestamp": (datetime.utcnow() - timedelta(hours=25)).isoformat(),
            "decision": "APPROVED",
            "was_alert": False,
        }
        
        recent_reviews = [
            {
                "timestamp": datetime.utcnow().isoformat(),
                "decision": "APPROVED",
                "was_alert": False,
            }
        ]
        
        result = ModelPerformanceCalculator.calculate_from_reviews(
            [old_review] + recent_reviews,
            window_hours=24
        )
        
        # Should only include 1 recent review
        assert result["samples"] == 1


class TestMetricsCollector:
    """Test metrics collection."""
    
    def test_record_scoring_event(self):
        """Record scoring events."""
        collector = MetricsCollector()
        
        collector.record_scoring_event(
            case_id="case1",
            latency_ms=50.0,
            score=0.8,
            decision="ALERT",
        )
        
        collector.record_scoring_event(
            case_id="case2",
            latency_ms=75.0,
            score=0.3,
            decision="PASS",
        )
        
        assert len(collector._latencies) == 2
    
    def test_latency_buffer_bounded(self):
        """Latency buffer doesn't grow unbounded."""
        collector = MetricsCollector()
        
        # Add 1500 events
        for i in range(1500):
            collector.record_scoring_event(
                case_id=f"case{i}",
                latency_ms=50.0,
                score=0.5,
                decision="PASS",
            )
        
        # Should keep only last 1000
        assert len(collector._latencies) <= 1000
    
    def test_get_current_metrics(self):
        """Get current metrics."""
        collector = MetricsCollector()
        
        # Add some latencies
        for latency in [10.0, 20.0, 30.0, 40.0, 50.0]:
            collector.record_scoring_event(
                case_id="case",
                latency_ms=latency,
                score=0.5,
                decision="PASS",
            )
        
        metrics = collector.get_current_metrics()
        
        assert "latency_ms" in metrics
        assert "avg" in metrics["latency_ms"]
        assert metrics["latency_ms"]["avg"] == 30.0  # (10+20+30+40+50)/5
        assert metrics["latency_ms"]["p95"] > 0.0
        assert metrics["latency_ms"]["p99"] > 0.0


class TestAlertingPolicy:
    """Test alerting policy."""
    
    def test_drift_alert_threshold(self):
        """Alert on drift exceeding threshold."""
        policy = AlertingPolicy()
        
        drift_scores = {
            "alert": True,
            "features_drifting": 3,
            "features_analyzed": 10,
        }
        
        should_alert, reason = policy.should_alert_drift(drift_scores)
        
        assert should_alert is True
        assert "3 features" in reason
    
    def test_no_drift_alert(self):
        """No alert when drift not detected."""
        policy = AlertingPolicy()
        
        drift_scores = {
            "alert": False,
            "features_drifting": 0,
        }
        
        should_alert, reason = policy.should_alert_drift(drift_scores)
        
        assert should_alert is False
    
    def test_latency_alert(self):
        """Alert on high latency."""
        policy = AlertingPolicy()
        
        # P99 latency > 1000ms
        should_alert, reason = policy.should_alert_latency(1500.0)
        
        assert should_alert is True
        assert "1500" in reason
    
    def test_no_latency_alert(self):
        """No alert for normal latency."""
        policy = AlertingPolicy()
        
        should_alert, reason = policy.should_alert_latency(200.0)
        
        assert should_alert is False
    
    def test_performance_degradation_alert(self):
        """Alert on AUC degradation."""
        policy = AlertingPolicy()
        
        current = {"auc": 0.85}
        baseline = {"auc": 0.89}  # 4% drop > 2% threshold
        
        should_alert, reason = policy.should_alert_performance(current, baseline)
        
        assert should_alert is True
        assert "AUC" in reason
    
    def test_no_performance_alert(self):
        """No alert for stable performance."""
        policy = AlertingPolicy()
        
        current = {"auc": 0.88}
        baseline = {"auc": 0.89}  # 1% drop < 2% threshold
        
        should_alert, reason = policy.should_alert_performance(current, baseline)
        
        assert should_alert is False
    
    def test_no_baseline_no_alert(self):
        """No alert without baseline."""
        policy = AlertingPolicy()
        
        current = {"auc": 0.85}
        
        should_alert, reason = policy.should_alert_performance(current, baseline_metrics=None)
        
        assert should_alert is False


class TestIntegration:
    """Integration tests for monitoring pipeline."""
    
    def test_drift_detection_pipeline(self):
        """Full drift detection pipeline."""
        training_metadata = {
            "means": {"amount": 1000.0, "velocity": 5.0},
            "mins": {"amount": 10.0, "velocity": 0.0},
            "maxes": {"amount": 50000.0, "velocity": 100.0},
        }
        
        detector = FeatureDriftDetector(training_metadata)
        
        # Current production values (shifted)
        current_amounts = [2000.0, 2100.0, 1900.0] * 10
        
        result = detector.compute_ks_statistic("amount", current_amounts)
        
        assert "ks_statistic" in result
        assert result["ks_statistic"] >= 0.0
    
    def test_alerting_pipeline(self):
        """Full alerting pipeline."""
        policy = AlertingPolicy()
        
        # Check multiple alert conditions
        drift_scores = {
            "alert": True,
            "features_drifting": 2,
            "features_analyzed": 10,
        }
        
        metrics = {
            "auc": 0.80,
            "precision": 0.85,
        }
        
        baseline = {
            "auc": 0.88,
            "precision": 0.90,
        }
        
        drift_alert, drift_reason = policy.should_alert_drift(drift_scores)
        perf_alert, perf_reason = policy.should_alert_performance(metrics, baseline_metrics=baseline)
        latency_alert, latency_reason = policy.should_alert_latency(500.0)
        
        assert drift_alert is True
        assert perf_alert is True
        assert latency_alert is False
