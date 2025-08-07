"""
Sentiment Analysis Monitoring and Validation Module

Provides comprehensive monitoring, validation, and quality assurance for sentiment analysis.
"""

import time
import json
import statistics
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import threading
import warnings
from pathlib import Path

from .ml_sentiment_analyzer import EnhancedSentimentResult, AnalyzerType, ConfidenceLevel
from .utils.logging import get_logger

logger = get_logger(__name__)


class AlertLevel(Enum):
    """Alert levels for monitoring."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MetricType(Enum):
    """Types of metrics to monitor."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class Alert:
    """Monitoring alert."""
    level: AlertLevel
    message: str
    timestamp: datetime
    metric_name: str
    current_value: Any
    threshold: Any
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QualityMetrics:
    """Quality metrics for sentiment analysis."""
    total_analyses: int = 0
    confidence_distribution: Dict[str, int] = field(default_factory=lambda: {
        "very_low": 0, "low": 0, "medium": 0, "high": 0, "very_high": 0
    })
    sentiment_distribution: Dict[str, int] = field(default_factory=lambda: {
        "positive": 0, "negative": 0, "neutral": 0
    })
    average_confidence: float = 0.0
    average_processing_time: float = 0.0
    error_count: int = 0
    cache_hit_rate: float = 0.0
    analyzer_performance: Dict[str, Dict] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_analyses": self.total_analyses,
            "confidence_distribution": self.confidence_distribution,
            "sentiment_distribution": self.sentiment_distribution,
            "average_confidence": self.average_confidence,
            "average_processing_time": self.average_processing_time,
            "error_count": self.error_count,
            "cache_hit_rate": self.cache_hit_rate,
            "analyzer_performance": self.analyzer_performance
        }


class SentimentValidator:
    """Validates sentiment analysis results for quality assurance."""
    
    def __init__(self):
        self.logger = get_logger(f"{__name__}.Validator")
        self.validation_rules = self._initialize_validation_rules()
        self.validation_history = deque(maxlen=1000)
    
    def _initialize_validation_rules(self) -> List[Callable]:
        """Initialize validation rules."""
        return [
            self._validate_confidence_score_consistency,
            self._validate_score_normalization,
            self._validate_processing_time,
            self._validate_text_label_consistency,
            self._validate_feature_reasonableness
        ]
    
    def validate_result(self, result: EnhancedSentimentResult) -> Tuple[bool, List[str]]:
        """Validate a sentiment analysis result."""
        issues = []
        
        for rule in self.validation_rules:
            try:
                is_valid, issue = rule(result)
                if not is_valid:
                    issues.append(issue)
            except Exception as e:
                issues.append(f"Validation rule failed: {e}")
        
        # Record validation result
        validation_record = {
            "timestamp": datetime.now(),
            "text_length": len(result.text),
            "confidence": result.confidence,
            "label": result.label.value,
            "analyzer_type": result.analyzer_type.value,
            "issues_count": len(issues),
            "issues": issues
        }
        self.validation_history.append(validation_record)
        
        return len(issues) == 0, issues
    
    def _validate_confidence_score_consistency(self, result: EnhancedSentimentResult) -> Tuple[bool, str]:
        """Validate that confidence aligns with score differences."""
        scores = result.scores
        max_score = max(scores.values())
        second_max = sorted(scores.values(), reverse=True)[1]
        score_diff = max_score - second_max
        
        # High confidence should correspond to large score differences
        if result.confidence > 0.8 and score_diff < 0.3:
            return False, f"High confidence ({result.confidence:.3f}) with low score difference ({score_diff:.3f})"
        
        # Low score difference should not have very high confidence
        if score_diff < 0.1 and result.confidence > 0.6:
            return False, f"Low score difference ({score_diff:.3f}) with high confidence ({result.confidence:.3f})"
        
        return True, ""
    
    def _validate_score_normalization(self, result: EnhancedSentimentResult) -> Tuple[bool, str]:
        """Validate that sentiment scores are properly normalized."""
        scores_sum = sum(result.scores.values())
        
        if abs(scores_sum - 1.0) > 0.05:  # Allow 5% tolerance
            return False, f"Scores don't sum to 1.0: {scores_sum:.3f}"
        
        # Check for negative scores
        for sentiment, score in result.scores.items():
            if score < 0:
                return False, f"Negative score for {sentiment}: {score:.3f}"
        
        return True, ""
    
    def _validate_processing_time(self, result: EnhancedSentimentResult) -> Tuple[bool, str]:
        """Validate processing time is reasonable."""
        # Very long processing times might indicate issues
        if result.processing_time_ms > 1000:  # 1 second
            return False, f"Excessive processing time: {result.processing_time_ms:.2f}ms"
        
        # Negative or zero processing time indicates timing issues
        if result.processing_time_ms <= 0:
            return False, f"Invalid processing time: {result.processing_time_ms:.2f}ms"
        
        return True, ""
    
    def _validate_text_label_consistency(self, result: EnhancedSentimentResult) -> Tuple[bool, str]:
        """Basic sanity check for obvious text-label mismatches."""
        text_lower = result.text.lower()
        
        # Very obvious positive indicators
        strong_positive = ['amazing', 'excellent', 'fantastic', 'love it', 'best ever']
        if any(phrase in text_lower for phrase in strong_positive):
            if result.label.value == 'negative' and result.confidence > 0.5:
                return False, f"Strong positive text labeled as negative with high confidence"
        
        # Very obvious negative indicators
        strong_negative = ['terrible', 'awful', 'hate it', 'worst ever', 'disgusting']
        if any(phrase in text_lower for phrase in strong_negative):
            if result.label.value == 'positive' and result.confidence > 0.5:
                return False, f"Strong negative text labeled as positive with high confidence"
        
        return True, ""
    
    def _validate_feature_reasonableness(self, result: EnhancedSentimentResult) -> Tuple[bool, str]:
        """Validate that extracted features are reasonable."""
        features = result.features
        
        if not features:
            return True, ""  # No features to validate
        
        # Check for unreasonable feature values
        if 'word_count' in features:
            if features['word_count'] < 0:
                return False, f"Negative word count: {features['word_count']}"
            if features['word_count'] > 10000:
                return False, f"Excessive word count: {features['word_count']}"
        
        if 'caps_ratio' in features:
            if not (0 <= features['caps_ratio'] <= 1):
                return False, f"Invalid caps ratio: {features['caps_ratio']}"
        
        if 'unique_word_ratio' in features:
            if not (0 <= features['unique_word_ratio'] <= 1):
                return False, f"Invalid unique word ratio: {features['unique_word_ratio']}"
        
        return True, ""
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """Get validation statistics."""
        if not self.validation_history:
            return {"total_validations": 0}
        
        total = len(self.validation_history)
        failed = sum(1 for record in self.validation_history if record['issues_count'] > 0)
        
        common_issues = defaultdict(int)
        for record in self.validation_history:
            for issue in record['issues']:
                # Extract issue type (first part before colon)
                issue_type = issue.split(':')[0] if ':' in issue else issue
                common_issues[issue_type] += 1
        
        return {
            "total_validations": total,
            "failed_validations": failed,
            "success_rate": (total - failed) / total,
            "common_issues": dict(common_issues),
            "avg_issues_per_validation": sum(r['issues_count'] for r in self.validation_history) / total
        }


class SentimentMonitor:
    """Comprehensive monitoring system for sentiment analysis."""
    
    def __init__(self, alert_thresholds: Optional[Dict[str, Any]] = None):
        self.logger = get_logger(f"{__name__}.Monitor")
        self.metrics = QualityMetrics()
        self.validator = SentimentValidator()
        self.alerts = deque(maxlen=100)
        self.analysis_history = deque(maxlen=5000)
        self.performance_history = defaultdict(lambda: deque(maxlen=1000))
        
        # Default alert thresholds
        self.alert_thresholds = alert_thresholds or {
            "min_confidence": 0.1,  # Alert if avg confidence too low
            "max_error_rate": 0.05,  # Alert if error rate > 5%
            "max_processing_time": 500,  # Alert if avg processing time > 500ms
            "min_cache_hit_rate": 0.3,  # Alert if cache hit rate < 30%
            "max_validation_failures": 0.1  # Alert if validation failure rate > 10%
        }
        
        self._lock = threading.Lock()
        self.logger.info("Sentiment monitor initialized")
    
    def record_analysis(self, result: EnhancedSentimentResult, cache_hit: bool = False) -> None:
        """Record a sentiment analysis result."""
        with self._lock:
            # Update basic metrics
            self.metrics.total_analyses += 1
            self.metrics.confidence_distribution[result.confidence_level.value] += 1
            self.metrics.sentiment_distribution[result.label.value] += 1
            
            # Update averages (running average)
            n = self.metrics.total_analyses
            self.metrics.average_confidence = (
                (self.metrics.average_confidence * (n - 1) + result.confidence) / n
            )
            self.metrics.average_processing_time = (
                (self.metrics.average_processing_time * (n - 1) + result.processing_time_ms) / n
            )
            
            # Record in history
            record = {
                "timestamp": result.timestamp,
                "label": result.label.value,
                "confidence": result.confidence,
                "processing_time": result.processing_time_ms,
                "analyzer_type": result.analyzer_type.value,
                "cache_hit": cache_hit,
                "text_length": len(result.text)
            }
            self.analysis_history.append(record)
            
            # Update analyzer-specific performance
            analyzer_name = result.analyzer_type.value
            if analyzer_name not in self.metrics.analyzer_performance:
                self.metrics.analyzer_performance[analyzer_name] = {
                    "count": 0,
                    "avg_confidence": 0.0,
                    "avg_processing_time": 0.0,
                    "error_count": 0
                }
            
            perf = self.metrics.analyzer_performance[analyzer_name]
            perf["count"] += 1
            perf["avg_confidence"] = (
                (perf["avg_confidence"] * (perf["count"] - 1) + result.confidence) / perf["count"]
            )
            perf["avg_processing_time"] = (
                (perf["avg_processing_time"] * (perf["count"] - 1) + result.processing_time_ms) / perf["count"]
            )
            
            # Validate result
            is_valid, issues = self.validator.validate_result(result)
            if not is_valid:
                self.logger.warning(f"Validation failed: {issues}")
                self._create_alert(
                    AlertLevel.WARNING,
                    f"Validation failed: {len(issues)} issues",
                    "validation_failures",
                    len(issues),
                    0
                )
    
    def record_error(self, error_msg: str, analyzer_type: Optional[AnalyzerType] = None) -> None:
        """Record an analysis error."""
        with self._lock:
            self.metrics.error_count += 1
            
            if analyzer_type:
                analyzer_name = analyzer_type.value
                if analyzer_name in self.metrics.analyzer_performance:
                    self.metrics.analyzer_performance[analyzer_name]["error_count"] += 1
            
            # Check error rate threshold
            error_rate = self.metrics.error_count / max(self.metrics.total_analyses + self.metrics.error_count, 1)
            if error_rate > self.alert_thresholds["max_error_rate"]:
                self._create_alert(
                    AlertLevel.ERROR,
                    f"High error rate: {error_rate:.3f}",
                    "error_rate",
                    error_rate,
                    self.alert_thresholds["max_error_rate"]
                )
    
    def update_cache_stats(self, hit_rate: float) -> None:
        """Update cache statistics."""
        with self._lock:
            self.metrics.cache_hit_rate = hit_rate
            
            # Check cache hit rate threshold
            if hit_rate < self.alert_thresholds["min_cache_hit_rate"]:
                self._create_alert(
                    AlertLevel.WARNING,
                    f"Low cache hit rate: {hit_rate:.3f}",
                    "cache_hit_rate",
                    hit_rate,
                    self.alert_thresholds["min_cache_hit_rate"]
                )
    
    def _create_alert(self, level: AlertLevel, message: str, metric_name: str, 
                     current_value: Any, threshold: Any) -> None:
        """Create and record an alert."""
        alert = Alert(
            level=level,
            message=message,
            timestamp=datetime.now(),
            metric_name=metric_name,
            current_value=current_value,
            threshold=threshold
        )
        
        self.alerts.append(alert)
        self.logger.log(
            getattr(logging, level.value.upper()),
            f"ALERT [{level.value.upper()}]: {message} (metric: {metric_name}, value: {current_value}, threshold: {threshold})"
        )
    
    def get_performance_report(self, time_window_minutes: int = 60) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        with self._lock:
            cutoff_time = datetime.now() - timedelta(minutes=time_window_minutes)
            
            # Filter recent analyses
            recent_analyses = [
                record for record in self.analysis_history
                if record["timestamp"] >= cutoff_time
            ]
            
            if not recent_analyses:
                return {"message": "No analyses in specified time window"}
            
            # Calculate time-windowed metrics
            confidences = [r["confidence"] for r in recent_analyses]
            processing_times = [r["processing_time"] for r in recent_analyses]
            
            sentiment_counts = defaultdict(int)
            analyzer_counts = defaultdict(int)
            
            for record in recent_analyses:
                sentiment_counts[record["label"]] += 1
                analyzer_counts[record["analyzer_type"]] += 1
            
            return {
                "time_window_minutes": time_window_minutes,
                "total_analyses": len(recent_analyses),
                "confidence_stats": {
                    "mean": statistics.mean(confidences),
                    "median": statistics.median(confidences),
                    "stdev": statistics.stdev(confidences) if len(confidences) > 1 else 0.0,
                    "min": min(confidences),
                    "max": max(confidences)
                },
                "processing_time_stats": {
                    "mean": statistics.mean(processing_times),
                    "median": statistics.median(processing_times),
                    "stdev": statistics.stdev(processing_times) if len(processing_times) > 1 else 0.0,
                    "min": min(processing_times),
                    "max": max(processing_times)
                },
                "sentiment_distribution": dict(sentiment_counts),
                "analyzer_distribution": dict(analyzer_counts),
                "cache_hit_rate": sum(1 for r in recent_analyses if r.get("cache_hit", False)) / len(recent_analyses),
                "validation_stats": self.validator.get_validation_stats()
            }
    
    def get_alerts(self, level: Optional[AlertLevel] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent alerts."""
        with self._lock:
            alerts = list(self.alerts)
            
            if level:
                alerts = [alert for alert in alerts if alert.level == level]
            
            # Sort by timestamp (most recent first) and limit
            alerts.sort(key=lambda x: x.timestamp, reverse=True)
            alerts = alerts[:limit]
            
            return [
                {
                    "level": alert.level.value,
                    "message": alert.message,
                    "timestamp": alert.timestamp.isoformat(),
                    "metric_name": alert.metric_name,
                    "current_value": alert.current_value,
                    "threshold": alert.threshold,
                    "metadata": alert.metadata
                } for alert in alerts
            ]
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get overall health status."""
        with self._lock:
            # Calculate health score based on various factors
            health_score = 100.0
            issues = []
            
            # Check error rate
            total_attempts = self.metrics.total_analyses + self.metrics.error_count
            if total_attempts > 0:
                error_rate = self.metrics.error_count / total_attempts
                if error_rate > self.alert_thresholds["max_error_rate"]:
                    health_score -= 30
                    issues.append(f"High error rate: {error_rate:.3f}")
            
            # Check average confidence
            if self.metrics.average_confidence < self.alert_thresholds["min_confidence"]:
                health_score -= 20
                issues.append(f"Low average confidence: {self.metrics.average_confidence:.3f}")
            
            # Check processing time
            if self.metrics.average_processing_time > self.alert_thresholds["max_processing_time"]:
                health_score -= 15
                issues.append(f"High processing time: {self.metrics.average_processing_time:.2f}ms")
            
            # Check cache performance
            if self.metrics.cache_hit_rate < self.alert_thresholds["min_cache_hit_rate"]:
                health_score -= 10
                issues.append(f"Low cache hit rate: {self.metrics.cache_hit_rate:.3f}")
            
            # Check validation failures
            validation_stats = self.validator.get_validation_stats()
            if validation_stats.get("total_validations", 0) > 0:
                validation_failure_rate = 1 - validation_stats["success_rate"]
                if validation_failure_rate > self.alert_thresholds["max_validation_failures"]:
                    health_score -= 15
                    issues.append(f"High validation failure rate: {validation_failure_rate:.3f}")
            
            # Determine health status
            if health_score >= 90:
                status = "excellent"
            elif health_score >= 70:
                status = "good"
            elif health_score >= 50:
                status = "fair"
            elif health_score >= 30:
                status = "poor"
            else:
                status = "critical"
            
            return {
                "status": status,
                "health_score": health_score,
                "issues": issues,
                "metrics_summary": self.metrics.to_dict(),
                "recent_alerts_count": len([a for a in self.alerts if a.timestamp >= datetime.now() - timedelta(hours=1)])
            }
    
    def export_metrics(self, file_path: Path) -> None:
        """Export metrics to file."""
        with self._lock:
            data = {
                "export_timestamp": datetime.now().isoformat(),
                "metrics": self.metrics.to_dict(),
                "performance_report": self.get_performance_report(60),
                "health_status": self.get_health_status(),
                "recent_alerts": self.get_alerts(limit=20),
                "validation_stats": self.validator.get_validation_stats()
            }
            
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            self.logger.info(f"Metrics exported to {file_path}")
    
    def reset_metrics(self) -> None:
        """Reset all metrics (use with caution)."""
        with self._lock:
            self.metrics = QualityMetrics()
            self.alerts.clear()
            self.analysis_history.clear()
            self.performance_history.clear()
            self.validator.validation_history.clear()
            
            self.logger.warning("All metrics have been reset")


# Global monitor instance
_global_monitor: Optional[SentimentMonitor] = None


def get_global_monitor() -> SentimentMonitor:
    """Get or create global sentiment monitor instance."""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = SentimentMonitor()
    return _global_monitor


def record_analysis(result: EnhancedSentimentResult, cache_hit: bool = False) -> None:
    """Convenience function to record analysis in global monitor."""
    monitor = get_global_monitor()
    monitor.record_analysis(result, cache_hit)


def record_error(error_msg: str, analyzer_type: Optional[AnalyzerType] = None) -> None:
    """Convenience function to record error in global monitor."""
    monitor = get_global_monitor()
    monitor.record_error(error_msg, analyzer_type)
