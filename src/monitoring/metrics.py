"""
Monitoring and drift detection for AML model.

Tracks:
- Feature distributions (training vs current)
- Model performance (precision, recall, AUC)
- System health (latency, throughput)
- Alerts on drift/performance degradation
"""

import json
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from statistics import mean, stdev, median
from collections import defaultdict
import numpy as np
from scipy import stats as scipy_stats

from src.storage import get_db


class FeatureDriftDetector:
    """Detect data drift in features between training and production."""
    
    def __init__(self, training_metadata: Dict[str, Any]):
        """
        Initialize drift detector with training statistics.
        
        Args:
            training_metadata: Dict with means, mins, maxes from training
        """
        self.training_means = training_metadata.get("means", {})
        self.training_mins = training_metadata.get("mins", {})
        self.training_maxes = training_metadata.get("maxes", {})
    
    def compute_ks_statistic(
        self,
        feature_name: str,
        current_values: List[float],
        percentiles: int = 100
    ) -> Dict[str, float]:
        """
        Kolmogorov-Smirnov test for drift.
        
        Compares current distribution to training distribution.
        
        Args:
            feature_name: Feature to test
            current_values: Current production values
            percentiles: Number of percentile bins
        
        Returns:
            Dict with ks_statistic, p_value, alert (bool)
        """
        if not current_values or len(current_values) < 10:
            return {
                "ks_statistic": 0.0,
                "p_value": 1.0,
                "alert": False,
                "reason": "insufficient_samples"
            }
        
        # Get training distribution as percentiles
        training_min = self.training_mins.get(feature_name, 0.0)
        training_max = self.training_maxes.get(feature_name, 1.0)
        
        # Generate training uniform reference (simple model)
        training_samples = np.linspace(training_min, training_max, 1000)
        current_samples = np.array(current_values)
        
        # KS test
        ks_stat, p_value = scipy_stats.ks_2samp(training_samples, current_samples)
        
        # Alert if p-value < 0.05 (5% significance level)
        alert = p_value < 0.05
        
        return {
            "ks_statistic": float(ks_stat),
            "p_value": float(p_value),
            "alert": alert,
            "interpretation": "distribution shifted" if alert else "no significant shift"
        }
    
    def compute_mean_drift(
        self,
        feature_name: str,
        current_mean: float,
        threshold_std: float = 2.0
    ) -> Dict[str, float]:
        """
        Detect mean drift in feature.
        
        Args:
            feature_name: Feature to check
            current_mean: Current production mean
            threshold_std: Alert if drift > N standard deviations
        
        Returns:
            Dict with training_mean, current_mean, std_devs, alert
        """
        training_mean = self.training_means.get(feature_name, 0.0)
        training_min = self.training_mins.get(feature_name, training_mean)
        training_max = self.training_maxes.get(feature_name, training_mean)
        
        # Estimate std from min/max (rough approximation)
        training_std = (training_max - training_min) / 4.0 if training_max > training_min else 1.0
        
        # How many std devs apart?
        std_devs = abs(current_mean - training_mean) / training_std if training_std > 0 else 0.0
        alert = std_devs > threshold_std
        
        return {
            "training_mean": float(training_mean),
            "current_mean": float(current_mean),
            "std_devs_apart": float(std_devs),
            "alert": alert,
            "threshold_std": threshold_std,
        }


class ModelPerformanceCalculator:
    """Calculate model performance metrics from audit trail."""
    
    @staticmethod
    def calculate_from_reviews(
        reviews: List[Dict[str, Any]],
        window_hours: int = 24
    ) -> Dict[str, float]:
        """
        Calculate performance metrics from recent reviews.
        
        Args:
            reviews: List of review records with decision + previous model decision
            window_hours: Only include reviews from last N hours
        
        Returns:
            Dict with precision, recall, accuracy, f1
        """
        if not reviews:
            return {
                "precision": 0.0,
                "recall": 0.0,
                "accuracy": 0.0,
                "f1": 0.0,
                "true_positives": 0,
                "false_positives": 0,
                "true_negatives": 0,
                "false_negatives": 0,
                "samples": 0,
            }
        
        cutoff_time = datetime.utcnow() - timedelta(hours=window_hours)
        recent_reviews = [
            r for r in reviews
            if datetime.fromisoformat(r.get("timestamp", datetime.utcnow().isoformat())) > cutoff_time
        ]
        
        if not recent_reviews:
            return {
                "precision": 0.0,
                "recall": 0.0,
                "accuracy": 0.0,
                "f1": 0.0,
                "true_positives": 0,
                "false_positives": 0,
                "true_negatives": 0,
                "false_negatives": 0,
                "samples": 0,
            }
        
        # Extract TP/FP/TN/FN
        # Assumption: review decision is ground truth
        # Model decision comes from audit trail
        tp = sum(1 for r in recent_reviews if r.get("decision") == "APPROVED" and r.get("was_alert"))
        fp = sum(1 for r in recent_reviews if r.get("decision") == "REJECTED" and r.get("was_alert"))
        tn = sum(1 for r in recent_reviews if r.get("decision") == "APPROVED" and not r.get("was_alert"))
        fn = sum(1 for r in recent_reviews if r.get("decision") == "REJECTED" and not r.get("was_alert"))
        
        total = tp + fp + tn + fn
        
        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        accuracy = (tp + tn) / total if total > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            "precision": precision,
            "recall": recall,
            "accuracy": accuracy,
            "f1": f1,
            "true_positives": tp,
            "false_positives": fp,
            "true_negatives": tn,
            "false_negatives": fn,
            "samples": total,
        }


class MetricsCollector:
    """Collect and persist metrics for monitoring."""
    
    def __init__(self):
        """Initialize metrics collector."""
        self.db = get_db()
        self._latencies = []  # In-memory buffer for latency
    
    def record_scoring_event(
        self,
        case_id: str,
        latency_ms: float,
        score: float,
        decision: str,
    ) -> None:
        """
        Record scoring event for metrics.
        
        Args:
            case_id: Case ID
            latency_ms: Scoring latency in milliseconds
            score: Risk score
            decision: ALERT or PASS
        """
        self._latencies.append(latency_ms)
        
        # Keep only last 1000 latencies to avoid unbounded growth
        if len(self._latencies) > 1000:
            self._latencies = self._latencies[-1000:]
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """
        Get current system metrics.
        
        Returns:
            Dict with latency, throughput, scoring metrics
        """
        db = get_db()
        
        # Latency metrics (from buffer)
        if self._latencies:
            p95_latency = np.percentile(self._latencies, 95)
            p99_latency = np.percentile(self._latencies, 99)
            avg_latency = mean(self._latencies)
        else:
            p95_latency = 0.0
            p99_latency = 0.0
            avg_latency = 0.0
        
        # Case counts
        case_counts = db.get_case_count_by_status()
        total_cases = sum(case_counts.values())
        
        # Score statistics (from recent cases)
        # TODO: Add score statistics query to DB
        score_mean = None
        score_median = None
        score_std = None
        
        return {
            "latency_ms": {
                "avg": avg_latency,
                "p95": p95_latency,
                "p99": p99_latency,
            },
            "cases": {
                "total": total_cases,
                "queued": case_counts.get("queued_for_review", 0),
                "approved": case_counts.get("approved", 0),
                "rejected": case_counts.get("rejected", 0),
                "escalated": case_counts.get("escalated", 0),
            },
            "scores": {
                "mean": score_mean,
                "median": score_median,
                "std": score_std,
            },
        }
    
    def get_drift_analysis(
        self,
        feature_contract,
        window_hours: int = 24
    ) -> Dict[str, Any]:
        """
        Analyze feature drift for recent transactions.
        
        Args:
            feature_contract: FeatureContract object
            window_hours: Time window for analysis
        
        Returns:
            Dict with drift scores per feature
        """
        # TODO: Query recent case features from DB
        # For now, return empty
        return {
            "window_hours": window_hours,
            "features_analyzed": 0,
            "features_drifting": 0,
            "alert": False,
            "details": {}
        }


class AlertingPolicy:
    """Define alert thresholds for monitoring."""
    
    def __init__(self):
        """Initialize default alert thresholds."""
        self.thresholds = {
            # Drift detection
            "drift_ks_p_value": 0.05,  # 5% significance
            "drift_mean_std_threshold": 2.0,  # 2 std devs
            
            # Performance degradation
            "performance_auc_drop": 0.02,  # >2% drop in AUC
            "performance_precision_drop": 0.05,  # >5% drop in precision
            "performance_recall_drop": 0.10,  # >10% drop in recall
            
            # System health
            "latency_p99_ms": 1000.0,  # 1 second
            "error_rate_pct": 5.0,  # 5% errors
            
            # Review workflow
            "queue_backlog_percent": 0.50,  # Queue > 50% of total
        }
    
    def should_alert_drift(self, drift_scores: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Check if drift alert should be triggered.
        
        Args:
            drift_scores: Drift analysis output
        
        Returns:
            Tuple of (should_alert, reason)
        """
        if drift_scores.get("alert"):
            return True, f"Drift detected in {drift_scores.get('features_drifting', 0)} features"
        return False, ""
    
    def should_alert_performance(
        self,
        current_metrics: Dict[str, float],
        baseline_metrics: Optional[Dict[str, float]] = None
    ) -> Tuple[bool, str]:
        """
        Check if performance degradation alert should trigger.
        
        Args:
            current_metrics: Current performance metrics
            baseline_metrics: Baseline for comparison
        
        Returns:
            Tuple of (should_alert, reason)
        """
        if baseline_metrics is None:
            return False, ""
        
        current_auc = current_metrics.get("auc", 0.0)
        baseline_auc = baseline_metrics.get("auc", 0.0)
        auc_drop = baseline_auc - current_auc
        
        if auc_drop > self.thresholds["performance_auc_drop"]:
            return True, f"AUC dropped {auc_drop:.4f} (threshold: {self.thresholds['performance_auc_drop']})"
        
        return False, ""
    
    def should_alert_latency(self, p99_latency_ms: float) -> Tuple[bool, str]:
        """Check if latency alert should trigger."""
        if p99_latency_ms > self.thresholds["latency_p99_ms"]:
            return True, f"P99 latency {p99_latency_ms}ms exceeds {self.thresholds['latency_p99_ms']}ms"
        return False, ""
