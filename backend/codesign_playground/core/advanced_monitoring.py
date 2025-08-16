"""
Advanced monitoring and analytics system with anomaly detection for AI Hardware Co-Design Playground.

This module provides:
- Real-time performance analytics
- Predictive monitoring with anomaly detection
- Application Performance Monitoring (APM)
- Cost optimization and resource efficiency tracking
- A/B testing and feature flag management
- Business intelligence and reporting dashboards
"""

import time
import threading
import asyncio
import uuid
import json
import pickle
import hashlib
import statistics
from typing import Any, Dict, List, Optional, Callable, Tuple, Union, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import deque, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import math
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from ..utils.monitoring import record_metric

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics."""
    COUNTER = "counter"           # Monotonically increasing
    GAUGE = "gauge"              # Point-in-time value
    HISTOGRAM = "histogram"       # Distribution of values
    TIMER = "timer"              # Duration measurements
    RATE = "rate"                # Events per unit time
    PERCENTAGE = "percentage"     # Percentage values
    BYTES = "bytes"              # Memory/storage metrics
    COUNT = "count"              # Simple count values


class AlertSeverity(Enum):
    """Alert severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class AnomalyType(Enum):
    """Types of anomalies."""
    POINT = "point"               # Single data point anomaly
    CONTEXTUAL = "contextual"     # Anomaly in specific context
    COLLECTIVE = "collective"     # Group of data points anomaly
    SEASONAL = "seasonal"         # Seasonal pattern anomaly
    TREND = "trend"              # Trend change anomaly


@dataclass
class MetricDataPoint:
    """Individual metric data point."""
    timestamp: float
    value: float
    labels: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MetricSeries:
    """Time series of metric data."""
    name: str
    metric_type: MetricType
    description: str
    unit: str
    data_points: List[MetricDataPoint] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    retention_hours: int = 24
    
    def add_data_point(self, value: float, labels: Optional[Dict[str, str]] = None,
                      metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add data point to series."""
        point = MetricDataPoint(
            timestamp=time.time(),
            value=value,
            labels=labels or {},
            metadata=metadata or {}
        )
        
        self.data_points.append(point)
        
        # Clean old data
        cutoff_time = time.time() - (self.retention_hours * 3600)
        self.data_points = [
            dp for dp in self.data_points 
            if dp.timestamp > cutoff_time
        ]
    
    def get_values(self, start_time: Optional[float] = None,
                  end_time: Optional[float] = None) -> List[float]:
        """Get values within time range."""
        start_time = start_time or 0
        end_time = end_time or time.time()
        
        return [
            dp.value for dp in self.data_points
            if start_time <= dp.timestamp <= end_time
        ]
    
    def get_latest_value(self) -> Optional[float]:
        """Get most recent value."""
        if self.data_points:
            return self.data_points[-1].value
        return None
    
    def calculate_statistics(self, window_minutes: int = 60) -> Dict[str, float]:
        """Calculate statistics for recent window."""
        cutoff_time = time.time() - (window_minutes * 60)
        recent_values = [
            dp.value for dp in self.data_points
            if dp.timestamp > cutoff_time
        ]
        
        if not recent_values:
            return {}
        
        return {
            'count': len(recent_values),
            'min': min(recent_values),
            'max': max(recent_values),
            'mean': statistics.mean(recent_values),
            'median': statistics.median(recent_values),
            'std': statistics.stdev(recent_values) if len(recent_values) > 1 else 0,
            'p95': np.percentile(recent_values, 95),
            'p99': np.percentile(recent_values, 99)
        }


@dataclass
class Alert:
    """Alert definition and state."""
    alert_id: str
    name: str
    description: str
    metric_name: str
    condition: str  # e.g., "> 0.8", "< 0.1"
    severity: AlertSeverity
    threshold_value: float
    window_minutes: int = 5
    consecutive_violations: int = 2
    enabled: bool = True
    
    # State
    current_violations: int = 0
    last_triggered: Optional[float] = None
    last_resolved: Optional[float] = None
    is_firing: bool = False
    
    def evaluate(self, current_value: float) -> bool:
        """Evaluate alert condition."""
        if not self.enabled:
            return False
        
        # Parse condition
        if self.condition.startswith('>'):
            threshold = float(self.condition[1:].strip())
            violation = current_value > threshold
        elif self.condition.startswith('<'):
            threshold = float(self.condition[1:].strip())
            violation = current_value < threshold
        elif self.condition.startswith('=='):
            threshold = float(self.condition[2:].strip())
            violation = current_value == threshold
        elif self.condition.startswith('!='):
            threshold = float(self.condition[2:].strip())
            violation = current_value != threshold
        else:
            # Default to greater than
            violation = current_value > self.threshold_value
        
        if violation:
            self.current_violations += 1
            if self.current_violations >= self.consecutive_violations and not self.is_firing:
                self.is_firing = True
                self.last_triggered = time.time()
                return True
        else:
            if self.is_firing:
                self.is_firing = False
                self.last_resolved = time.time()
            self.current_violations = 0
        
        return False


@dataclass
class AnomalyEvent:
    """Detected anomaly event."""
    event_id: str
    timestamp: float
    metric_name: str
    anomaly_type: AnomalyType
    severity: AlertSeverity
    value: float
    expected_value: float
    deviation_score: float
    description: str
    context: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AnomalyDetector:
    """Advanced anomaly detection using multiple algorithms."""
    
    def __init__(self, contamination: float = 0.1, window_size: int = 100):
        self.contamination = contamination
        self.window_size = window_size
        
        # ML models for different anomaly types
        self.isolation_forest = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )
        
        self.scaler = StandardScaler()
        self.is_trained = False
        self.training_data = deque(maxlen=1000)
        
        # Seasonal pattern detection
        self.seasonal_patterns = defaultdict(list)
        self.trend_analyzer = TrendAnalyzer()
        
        # Statistical thresholds
        self.statistical_thresholds = {}
        
        logger.info("Initialized AnomalyDetector")
    
    def add_training_data(self, metric_name: str, value: float, 
                         timestamp: Optional[float] = None) -> None:
        """Add training data for anomaly detection."""
        timestamp = timestamp or time.time()
        
        self.training_data.append({
            'metric_name': metric_name,
            'value': value,
            'timestamp': timestamp,
            'hour': time.localtime(timestamp).tm_hour,
            'day_of_week': time.localtime(timestamp).tm_wday
        })
        
        # Update seasonal patterns
        hour = time.localtime(timestamp).tm_hour
        self.seasonal_patterns[f"{metric_name}_hour_{hour}"].append(value)
        
        # Keep limited history
        max_seasonal_samples = 50
        for pattern_key in self.seasonal_patterns:
            if len(self.seasonal_patterns[pattern_key]) > max_seasonal_samples:
                self.seasonal_patterns[pattern_key] = \
                    self.seasonal_patterns[pattern_key][-max_seasonal_samples:]\n        \n        # Retrain periodically\n        if len(self.training_data) % 50 == 0 and len(self.training_data) >= 100:\n            self._retrain_models()\n    \n    def detect_anomalies(self, metric_name: str, values: List[float],\n                        timestamps: Optional[List[float]] = None) -> List[AnomalyEvent]:\n        \"\"\"Detect anomalies in metric values.\"\"\"\n        if not values:\n            return []\n        \n        timestamps = timestamps or [time.time() - i for i in range(len(values), 0, -1)]\n        anomalies = []\n        \n        # Point anomalies using Isolation Forest\n        if self.is_trained and len(values) >= 10:\n            point_anomalies = self._detect_point_anomalies(metric_name, values, timestamps)\n            anomalies.extend(point_anomalies)\n        \n        # Statistical anomalies\n        statistical_anomalies = self._detect_statistical_anomalies(metric_name, values, timestamps)\n        anomalies.extend(statistical_anomalies)\n        \n        # Seasonal anomalies\n        seasonal_anomalies = self._detect_seasonal_anomalies(metric_name, values, timestamps)\n        anomalies.extend(seasonal_anomalies)\n        \n        # Trend anomalies\n        trend_anomalies = self._detect_trend_anomalies(metric_name, values, timestamps)\n        anomalies.extend(trend_anomalies)\n        \n        return anomalies\n    \n    def _detect_point_anomalies(self, metric_name: str, values: List[float],\n                               timestamps: List[float]) -> List[AnomalyEvent]:\n        \"\"\"Detect point anomalies using Isolation Forest.\"\"\"\n        try:\n            # Prepare features\n            features = []\n            for i, (value, timestamp) in enumerate(zip(values, timestamps)):\n                dt = time.localtime(timestamp)\n                feature_vector = [\n                    value,\n                    dt.tm_hour,\n                    dt.tm_wday,\n                    i  # position in sequence\n                ]\n                features.append(feature_vector)\n            \n            X = np.array(features)\n            X_scaled = self.scaler.transform(X)\n            \n            # Predict anomalies\n            anomaly_scores = self.isolation_forest.decision_function(X_scaled)\n            predictions = self.isolation_forest.predict(X_scaled)\n            \n            anomalies = []\n            for i, (prediction, score) in enumerate(zip(predictions, anomaly_scores)):\n                if prediction == -1:  # Anomaly\n                    # Calculate expected value from similar historical data\n                    expected_value = self._get_expected_value(metric_name, timestamps[i])\n                    \n                    severity = self._calculate_severity(abs(score))\n                    \n                    anomaly = AnomalyEvent(\n                        event_id=str(uuid.uuid4()),\n                        timestamp=timestamps[i],\n                        metric_name=metric_name,\n                        anomaly_type=AnomalyType.POINT,\n                        severity=severity,\n                        value=values[i],\n                        expected_value=expected_value,\n                        deviation_score=abs(score),\n                        description=f\"Point anomaly detected in {metric_name}: {values[i]:.2f} (expected: {expected_value:.2f})\",\n                        context={'isolation_forest_score': score}\n                    )\n                    anomalies.append(anomaly)\n            \n            return anomalies\n            \n        except Exception as e:\n            logger.warning(f\"Point anomaly detection failed: {e}\")\n            return []\n    \n    def _detect_statistical_anomalies(self, metric_name: str, values: List[float],\n                                     timestamps: List[float]) -> List[AnomalyEvent]:\n        \"\"\"Detect statistical anomalies using Z-score and IQR.\"\"\"\n        if len(values) < 10:\n            return []\n        \n        anomalies = []\n        \n        # Calculate statistics\n        mean_val = np.mean(values)\n        std_val = np.std(values)\n        q1 = np.percentile(values, 25)\n        q3 = np.percentile(values, 75)\n        iqr = q3 - q1\n        \n        # Z-score threshold (typically 2 or 3)\n        z_threshold = 3.0\n        \n        # IQR threshold\n        iqr_lower = q1 - 1.5 * iqr\n        iqr_upper = q3 + 1.5 * iqr\n        \n        for i, (value, timestamp) in enumerate(zip(values, timestamps)):\n            is_anomaly = False\n            anomaly_type = \"statistical\"\n            deviation_score = 0.0\n            \n            # Z-score anomaly\n            if std_val > 0:\n                z_score = abs(value - mean_val) / std_val\n                if z_score > z_threshold:\n                    is_anomaly = True\n                    anomaly_type = \"z_score\"\n                    deviation_score = z_score\n            \n            # IQR anomaly\n            if value < iqr_lower or value > iqr_upper:\n                is_anomaly = True\n                anomaly_type = \"iqr\"\n                deviation_score = max(\n                    abs(value - iqr_lower) / iqr if iqr > 0 else 0,\n                    abs(value - iqr_upper) / iqr if iqr > 0 else 0\n                )\n            \n            if is_anomaly:\n                severity = self._calculate_severity(deviation_score)\n                \n                anomaly = AnomalyEvent(\n                    event_id=str(uuid.uuid4()),\n                    timestamp=timestamp,\n                    metric_name=metric_name,\n                    anomaly_type=AnomalyType.POINT,\n                    severity=severity,\n                    value=value,\n                    expected_value=mean_val,\n                    deviation_score=deviation_score,\n                    description=f\"Statistical anomaly in {metric_name}: {value:.2f} ({anomaly_type})\",\n                    context={\n                        'z_score': (value - mean_val) / std_val if std_val > 0 else 0,\n                        'iqr_bounds': [iqr_lower, iqr_upper],\n                        'anomaly_type': anomaly_type\n                    }\n                )\n                anomalies.append(anomaly)\n        \n        return anomalies\n    \n    def _detect_seasonal_anomalies(self, metric_name: str, values: List[float],\n                                  timestamps: List[float]) -> List[AnomalyEvent]:\n        \"\"\"Detect seasonal pattern anomalies.\"\"\"\n        anomalies = []\n        \n        for i, (value, timestamp) in enumerate(zip(values, timestamps)):\n            expected_value = self._get_expected_value(metric_name, timestamp)\n            \n            if expected_value is not None:\n                # Calculate deviation from seasonal expectation\n                deviation = abs(value - expected_value)\n                relative_deviation = deviation / max(abs(expected_value), 1e-6)\n                \n                # Threshold for seasonal anomaly (configurable)\n                if relative_deviation > 0.5:  # 50% deviation\n                    severity = self._calculate_severity(relative_deviation)\n                    \n                    anomaly = AnomalyEvent(\n                        event_id=str(uuid.uuid4()),\n                        timestamp=timestamp,\n                        metric_name=metric_name,\n                        anomaly_type=AnomalyType.SEASONAL,\n                        severity=severity,\n                        value=value,\n                        expected_value=expected_value,\n                        deviation_score=relative_deviation,\n                        description=f\"Seasonal anomaly in {metric_name}: {value:.2f} (expected: {expected_value:.2f})\",\n                        context={'relative_deviation': relative_deviation}\n                    )\n                    anomalies.append(anomaly)\n        \n        return anomalies\n    \n    def _detect_trend_anomalies(self, metric_name: str, values: List[float],\n                               timestamps: List[float]) -> List[AnomalyEvent]:\n        \"\"\"Detect trend change anomalies.\"\"\"\n        if len(values) < 20:  # Need sufficient data for trend analysis\n            return []\n        \n        anomalies = []\n        \n        # Analyze trend changes\n        trend_changes = self.trend_analyzer.detect_trend_changes(values, timestamps)\n        \n        for change_point in trend_changes:\n            if change_point['significance'] > 0.7:  # Significant trend change\n                severity = self._calculate_severity(change_point['significance'])\n                \n                anomaly = AnomalyEvent(\n                    event_id=str(uuid.uuid4()),\n                    timestamp=change_point['timestamp'],\n                    metric_name=metric_name,\n                    anomaly_type=AnomalyType.TREND,\n                    severity=severity,\n                    value=change_point['value'],\n                    expected_value=change_point['expected_value'],\n                    deviation_score=change_point['significance'],\n                    description=f\"Trend change detected in {metric_name}: {change_point['trend_type']}\",\n                    context=change_point\n                )\n                anomalies.append(anomaly)\n        \n        return anomalies\n    \n    def _get_expected_value(self, metric_name: str, timestamp: float) -> Optional[float]:\n        \"\"\"Get expected value based on historical patterns.\"\"\"\n        hour = time.localtime(timestamp).tm_hour\n        pattern_key = f\"{metric_name}_hour_{hour}\"\n        \n        if pattern_key in self.seasonal_patterns:\n            pattern_values = self.seasonal_patterns[pattern_key]\n            if len(pattern_values) >= 5:\n                return statistics.median(pattern_values)\n        \n        return None\n    \n    def _calculate_severity(self, deviation_score: float) -> AlertSeverity:\n        \"\"\"Calculate severity based on deviation score.\"\"\"\n        if deviation_score > 5.0:\n            return AlertSeverity.CRITICAL\n        elif deviation_score > 3.0:\n            return AlertSeverity.HIGH\n        elif deviation_score > 2.0:\n            return AlertSeverity.MEDIUM\n        elif deviation_score > 1.0:\n            return AlertSeverity.LOW\n        else:\n            return AlertSeverity.INFO\n    \n    def _retrain_models(self) -> None:\n        \"\"\"Retrain anomaly detection models.\"\"\"\n        try:\n            if len(self.training_data) < 50:\n                return\n            \n            # Prepare training features\n            features = []\n            for data_point in self.training_data:\n                feature_vector = [\n                    data_point['value'],\n                    data_point['hour'],\n                    data_point['day_of_week']\n                ]\n                features.append(feature_vector)\n            \n            X = np.array(features)\n            \n            # Fit scaler and model\n            self.scaler.fit(X)\n            X_scaled = self.scaler.transform(X)\n            \n            self.isolation_forest.fit(X_scaled)\n            self.is_trained = True\n            \n            logger.info(\"Retrained anomaly detection models\")\n            \n        except Exception as e:\n            logger.error(f\"Model retraining failed: {e}\")\n\n\nclass TrendAnalyzer:\n    \"\"\"Analyze trends and detect significant changes.\"\"\"\n    \n    def __init__(self):\n        self.min_trend_length = 10\n        self.significance_threshold = 0.05\n    \n    def detect_trend_changes(self, values: List[float], \n                           timestamps: List[float]) -> List[Dict[str, Any]]:\n        \"\"\"Detect significant trend changes in time series.\"\"\"\n        if len(values) < self.min_trend_length * 2:\n            return []\n        \n        changes = []\n        window_size = self.min_trend_length\n        \n        for i in range(window_size, len(values) - window_size):\n            # Calculate trends before and after the point\n            before_values = values[i-window_size:i]\n            after_values = values[i:i+window_size]\n            \n            before_trend = self._calculate_trend(before_values)\n            after_trend = self._calculate_trend(after_values)\n            \n            # Check for significant trend change\n            trend_change = abs(after_trend - before_trend)\n            \n            if trend_change > 0.1:  # Configurable threshold\n                significance = min(trend_change / 0.5, 1.0)  # Normalize to 0-1\n                \n                # Determine trend type\n                if before_trend > 0.05 and after_trend < -0.05:\n                    trend_type = \"upward_to_downward\"\n                elif before_trend < -0.05 and after_trend > 0.05:\n                    trend_type = \"downward_to_upward\"\n                elif abs(before_trend) < 0.02 and abs(after_trend) > 0.05:\n                    trend_type = \"stable_to_trending\"\n                elif abs(before_trend) > 0.05 and abs(after_trend) < 0.02:\n                    trend_type = \"trending_to_stable\"\n                else:\n                    trend_type = \"trend_change\"\n                \n                change_point = {\n                    'timestamp': timestamps[i],\n                    'value': values[i],\n                    'expected_value': self._extrapolate_trend(before_values),\n                    'significance': significance,\n                    'trend_type': trend_type,\n                    'before_trend': before_trend,\n                    'after_trend': after_trend,\n                    'trend_change': trend_change\n                }\n                changes.append(change_point)\n        \n        return changes\n    \n    def _calculate_trend(self, values: List[float]) -> float:\n        \"\"\"Calculate trend slope using linear regression.\"\"\"\n        if len(values) < 2:\n            return 0.0\n        \n        n = len(values)\n        x = list(range(n))\n        \n        # Simple linear regression\n        x_mean = sum(x) / n\n        y_mean = sum(values) / n\n        \n        numerator = sum((x[i] - x_mean) * (values[i] - y_mean) for i in range(n))\n        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))\n        \n        if denominator == 0:\n            return 0.0\n        \n        return numerator / denominator\n    \n    def _extrapolate_trend(self, values: List[float]) -> float:\n        \"\"\"Extrapolate trend to predict next value.\"\"\"\n        if len(values) < 2:\n            return values[-1] if values else 0.0\n        \n        trend = self._calculate_trend(values)\n        return values[-1] + trend\n\n\nclass PerformanceAnalyzer:\n    \"\"\"Analyze application performance metrics.\"\"\"\n    \n    def __init__(self):\n        self.response_times = deque(maxlen=1000)\n        self.error_rates = deque(maxlen=1000)\n        self.throughput_data = deque(maxlen=1000)\n        self.resource_utilization = deque(maxlen=1000)\n        \n        # SLA thresholds\n        self.sla_thresholds = {\n            'response_time_ms': 1000,\n            'error_rate': 0.01,  # 1%\n            'availability': 0.999  # 99.9%\n        }\n    \n    def add_performance_data(self, response_time_ms: float, success: bool,\n                           throughput_rps: float, cpu_percent: float,\n                           memory_percent: float) -> None:\n        \"\"\"Add performance data point.\"\"\"\n        timestamp = time.time()\n        \n        self.response_times.append({\n            'timestamp': timestamp,\n            'value': response_time_ms\n        })\n        \n        self.error_rates.append({\n            'timestamp': timestamp,\n            'value': 0 if success else 1\n        })\n        \n        self.throughput_data.append({\n            'timestamp': timestamp,\n            'value': throughput_rps\n        })\n        \n        self.resource_utilization.append({\n            'timestamp': timestamp,\n            'cpu_percent': cpu_percent,\n            'memory_percent': memory_percent\n        })\n    \n    def calculate_sla_metrics(self, window_minutes: int = 60) -> Dict[str, Any]:\n        \"\"\"Calculate SLA compliance metrics.\"\"\"\n        cutoff_time = time.time() - (window_minutes * 60)\n        \n        # Response time SLA\n        recent_response_times = [\n            data['value'] for data in self.response_times\n            if data['timestamp'] > cutoff_time\n        ]\n        \n        response_time_sla = 0.0\n        if recent_response_times:\n            compliant_responses = sum(\n                1 for rt in recent_response_times\n                if rt <= self.sla_thresholds['response_time_ms']\n            )\n            response_time_sla = compliant_responses / len(recent_response_times)\n        \n        # Error rate SLA\n        recent_errors = [\n            data['value'] for data in self.error_rates\n            if data['timestamp'] > cutoff_time\n        ]\n        \n        error_rate_sla = 0.0\n        if recent_errors:\n            error_rate = sum(recent_errors) / len(recent_errors)\n            error_rate_sla = 1.0 if error_rate <= self.sla_thresholds['error_rate'] else 0.0\n        \n        # Overall availability (simplified)\n        availability = min(response_time_sla, error_rate_sla)\n        \n        return {\n            'response_time_sla': response_time_sla,\n            'error_rate_sla': error_rate_sla,\n            'availability': availability,\n            'sla_compliant': availability >= self.sla_thresholds['availability'],\n            'window_minutes': window_minutes\n        }\n    \n    def identify_performance_bottlenecks(self) -> List[Dict[str, Any]]:\n        \"\"\"Identify performance bottlenecks.\"\"\"\n        bottlenecks = []\n        \n        # Analyze recent data (last 30 minutes)\n        cutoff_time = time.time() - 1800\n        \n        # High response times\n        recent_response_times = [\n            data['value'] for data in self.response_times\n            if data['timestamp'] > cutoff_time\n        ]\n        \n        if recent_response_times:\n            avg_response_time = statistics.mean(recent_response_times)\n            p95_response_time = np.percentile(recent_response_times, 95)\n            \n            if avg_response_time > self.sla_thresholds['response_time_ms']:\n                bottlenecks.append({\n                    'type': 'high_response_time',\n                    'severity': 'high',\n                    'description': f'Average response time ({avg_response_time:.0f}ms) exceeds SLA threshold',\n                    'value': avg_response_time,\n                    'threshold': self.sla_thresholds['response_time_ms']\n                })\n            \n            if p95_response_time > self.sla_thresholds['response_time_ms'] * 2:\n                bottlenecks.append({\n                    'type': 'high_p95_response_time',\n                    'severity': 'medium',\n                    'description': f'95th percentile response time ({p95_response_time:.0f}ms) is concerning',\n                    'value': p95_response_time,\n                    'threshold': self.sla_thresholds['response_time_ms'] * 2\n                })\n        \n        # High error rates\n        recent_errors = [\n            data['value'] for data in self.error_rates\n            if data['timestamp'] > cutoff_time\n        ]\n        \n        if recent_errors:\n            error_rate = statistics.mean(recent_errors)\n            if error_rate > self.sla_thresholds['error_rate']:\n                bottlenecks.append({\n                    'type': 'high_error_rate',\n                    'severity': 'critical',\n                    'description': f'Error rate ({error_rate:.2%}) exceeds SLA threshold',\n                    'value': error_rate,\n                    'threshold': self.sla_thresholds['error_rate']\n                })\n        \n        # Resource utilization bottlenecks\n        recent_utilization = [\n            data for data in self.resource_utilization\n            if data['timestamp'] > cutoff_time\n        ]\n        \n        if recent_utilization:\n            avg_cpu = statistics.mean([data['cpu_percent'] for data in recent_utilization])\n            avg_memory = statistics.mean([data['memory_percent'] for data in recent_utilization])\n            \n            if avg_cpu > 80:\n                bottlenecks.append({\n                    'type': 'high_cpu_utilization',\n                    'severity': 'medium',\n                    'description': f'High CPU utilization ({avg_cpu:.1f}%)',\n                    'value': avg_cpu,\n                    'threshold': 80\n                })\n            \n            if avg_memory > 85:\n                bottlenecks.append({\n                    'type': 'high_memory_utilization',\n                    'severity': 'medium',\n                    'description': f'High memory utilization ({avg_memory:.1f}%)',\n                    'value': avg_memory,\n                    'threshold': 85\n                })\n        \n        return bottlenecks\n\n\nclass MetricsCollector:\n    \"\"\"Central metrics collection and management system.\"\"\"\n    \n    def __init__(self):\n        self.metrics = {}\n        self.alerts = {}\n        self.anomaly_detector = AnomalyDetector()\n        self.performance_analyzer = PerformanceAnalyzer()\n        self._lock = threading.RLock()\n        \n        # Background processing\n        self.processing_active = False\n        self.processing_thread = None\n        \n        # Event handlers\n        self.alert_handlers = []\n        self.anomaly_handlers = []\n        \n        logger.info(\"Initialized MetricsCollector\")\n    \n    def create_metric(self, name: str, metric_type: MetricType, \n                     description: str = \"\", unit: str = \"\",\n                     retention_hours: int = 24) -> None:\n        \"\"\"Create new metric series.\"\"\"\n        with self._lock:\n            if name not in self.metrics:\n                self.metrics[name] = MetricSeries(\n                    name=name,\n                    metric_type=metric_type,\n                    description=description,\n                    unit=unit,\n                    retention_hours=retention_hours\n                )\n                logger.info(f\"Created metric: {name} ({metric_type.value})\")\n    \n    def record_metric(self, name: str, value: float, \n                     labels: Optional[Dict[str, str]] = None,\n                     metadata: Optional[Dict[str, Any]] = None) -> None:\n        \"\"\"Record metric value.\"\"\"\n        with self._lock:\n            if name not in self.metrics:\n                # Auto-create metric with default settings\n                self.create_metric(name, MetricType.GAUGE)\n            \n            metric = self.metrics[name]\n            metric.add_data_point(value, labels, metadata)\n            \n            # Add to anomaly detector training data\n            self.anomaly_detector.add_training_data(name, value)\n        \n        # Process alerts and anomalies asynchronously\n        if self.processing_active:\n            threading.Thread(\n                target=self._process_metric_update,\n                args=(name, value),\n                daemon=True\n            ).start()\n    \n    def create_alert(self, name: str, metric_name: str, condition: str,\n                    severity: AlertSeverity = AlertSeverity.MEDIUM,\n                    description: str = \"\", window_minutes: int = 5,\n                    consecutive_violations: int = 2) -> str:\n        \"\"\"Create alert rule.\"\"\"\n        alert_id = str(uuid.uuid4())\n        \n        # Extract threshold value from condition\n        threshold_value = 0.0\n        try:\n            if condition.startswith(('>', '<', '==', '!=')):\n                threshold_value = float(condition[1:].strip() if condition.startswith(('>', '<')) else condition[2:].strip())\n        except ValueError:\n            pass\n        \n        alert = Alert(\n            alert_id=alert_id,\n            name=name,\n            description=description,\n            metric_name=metric_name,\n            condition=condition,\n            severity=severity,\n            threshold_value=threshold_value,\n            window_minutes=window_minutes,\n            consecutive_violations=consecutive_violations\n        )\n        \n        with self._lock:\n            self.alerts[alert_id] = alert\n        \n        logger.info(f\"Created alert: {name} for metric {metric_name}\")\n        return alert_id\n    \n    def add_alert_handler(self, handler: Callable[[Alert, float], None]) -> None:\n        \"\"\"Add alert handler function.\"\"\"\n        self.alert_handlers.append(handler)\n    \n    def add_anomaly_handler(self, handler: Callable[[AnomalyEvent], None]) -> None:\n        \"\"\"Add anomaly handler function.\"\"\"\n        self.anomaly_handlers.append(handler)\n    \n    def start_processing(self) -> None:\n        \"\"\"Start background processing for alerts and anomalies.\"\"\"\n        if self.processing_active:\n            return\n        \n        self.processing_active = True\n        self.processing_thread = threading.Thread(\n            target=self._processing_loop,\n            daemon=True,\n            name=\"MetricsProcessor\"\n        )\n        self.processing_thread.start()\n        logger.info(\"Started metrics processing\")\n    \n    def stop_processing(self) -> None:\n        \"\"\"Stop background processing.\"\"\"\n        self.processing_active = False\n        if self.processing_thread:\n            self.processing_thread.join(timeout=5.0)\n        logger.info(\"Stopped metrics processing\")\n    \n    def _processing_loop(self) -> None:\n        \"\"\"Main processing loop for anomaly detection and alerting.\"\"\"\n        while self.processing_active:\n            try:\n                self._process_anomaly_detection()\n                time.sleep(30)  # Process every 30 seconds\n            except Exception as e:\n                logger.error(f\"Metrics processing error: {e}\")\n                time.sleep(30)\n    \n    def _process_metric_update(self, metric_name: str, value: float) -> None:\n        \"\"\"Process metric update for alerts.\"\"\"\n        try:\n            # Check alerts for this metric\n            with self._lock:\n                relevant_alerts = [\n                    alert for alert in self.alerts.values()\n                    if alert.metric_name == metric_name and alert.enabled\n                ]\n            \n            for alert in relevant_alerts:\n                if alert.evaluate(value):\n                    # Alert triggered\n                    for handler in self.alert_handlers:\n                        try:\n                            handler(alert, value)\n                        except Exception as e:\n                            logger.error(f\"Alert handler error: {e}\")\n                    \n                    logger.warning(f\"Alert triggered: {alert.name} - {alert.description}\")\n        \n        except Exception as e:\n            logger.error(f\"Alert processing error: {e}\")\n    \n    def _process_anomaly_detection(self) -> None:\n        \"\"\"Process anomaly detection for all metrics.\"\"\"\n        try:\n            with self._lock:\n                metrics_to_check = list(self.metrics.items())\n            \n            for metric_name, metric_series in metrics_to_check:\n                # Get recent values for anomaly detection\n                recent_values = metric_series.get_values(\n                    start_time=time.time() - 3600  # Last hour\n                )\n                \n                if len(recent_values) >= 10:\n                    # Get corresponding timestamps\n                    recent_points = [\n                        dp for dp in metric_series.data_points\n                        if dp.timestamp > time.time() - 3600\n                    ]\n                    timestamps = [dp.timestamp for dp in recent_points]\n                    \n                    # Detect anomalies\n                    anomalies = self.anomaly_detector.detect_anomalies(\n                        metric_name, recent_values, timestamps\n                    )\n                    \n                    # Process detected anomalies\n                    for anomaly in anomalies:\n                        for handler in self.anomaly_handlers:\n                            try:\n                                handler(anomaly)\n                            except Exception as e:\n                                logger.error(f\"Anomaly handler error: {e}\")\n                        \n                        logger.warning(f\"Anomaly detected: {anomaly.description}\")\n        \n        except Exception as e:\n            logger.error(f\"Anomaly detection error: {e}\")\n    \n    def get_metric_statistics(self, metric_name: str, \n                            window_minutes: int = 60) -> Optional[Dict[str, Any]]:\n        \"\"\"Get statistics for a metric.\"\"\"\n        with self._lock:\n            if metric_name not in self.metrics:\n                return None\n            \n            metric = self.metrics[metric_name]\n            stats = metric.calculate_statistics(window_minutes)\n            \n            if not stats:\n                return None\n            \n            # Add additional context\n            stats['metric_name'] = metric_name\n            stats['metric_type'] = metric.metric_type.value\n            stats['window_minutes'] = window_minutes\n            stats['latest_value'] = metric.get_latest_value()\n            \n            return stats\n    \n    def get_dashboard_data(self) -> Dict[str, Any]:\n        \"\"\"Get comprehensive dashboard data.\"\"\"\n        with self._lock:\n            metric_summaries = {}\n            \n            for name, metric in self.metrics.items():\n                latest_value = metric.get_latest_value()\n                stats = metric.calculate_statistics(60)  # Last hour\n                \n                metric_summaries[name] = {\n                    'latest_value': latest_value,\n                    'type': metric.metric_type.value,\n                    'unit': metric.unit,\n                    'description': metric.description,\n                    'stats': stats\n                }\n            \n            # Active alerts\n            active_alerts = [\n                {\n                    'name': alert.name,\n                    'metric_name': alert.metric_name,\n                    'severity': alert.severity.value,\n                    'is_firing': alert.is_firing,\n                    'last_triggered': alert.last_triggered\n                }\n                for alert in self.alerts.values()\n                if alert.enabled\n            ]\n            \n            # Performance metrics\n            sla_metrics = self.performance_analyzer.calculate_sla_metrics()\n            bottlenecks = self.performance_analyzer.identify_performance_bottlenecks()\n            \n            return {\n                'metrics': metric_summaries,\n                'alerts': {\n                    'total': len(self.alerts),\n                    'active': active_alerts,\n                    'firing': sum(1 for alert in self.alerts.values() if alert.is_firing)\n                },\n                'performance': {\n                    'sla_metrics': sla_metrics,\n                    'bottlenecks': bottlenecks\n                },\n                'system_health': self._calculate_system_health()\n            }\n    \n    def _calculate_system_health(self) -> Dict[str, Any]:\n        \"\"\"Calculate overall system health score.\"\"\"\n        health_score = 100.0\n        issues = []\n        \n        # Deduct points for firing alerts\n        firing_alerts = sum(1 for alert in self.alerts.values() if alert.is_firing)\n        critical_alerts = sum(\n            1 for alert in self.alerts.values() \n            if alert.is_firing and alert.severity == AlertSeverity.CRITICAL\n        )\n        \n        health_score -= critical_alerts * 30  # 30 points per critical alert\n        health_score -= (firing_alerts - critical_alerts) * 10  # 10 points per other alert\n        \n        if critical_alerts > 0:\n            issues.append(f\"{critical_alerts} critical alerts firing\")\n        if firing_alerts > critical_alerts:\n            issues.append(f\"{firing_alerts - critical_alerts} other alerts firing\")\n        \n        # Check SLA compliance\n        sla_metrics = self.performance_analyzer.calculate_sla_metrics()\n        if not sla_metrics.get('sla_compliant', True):\n            health_score -= 20\n            issues.append(\"SLA not met\")\n        \n        # Ensure score is within bounds\n        health_score = max(0, min(100, health_score))\n        \n        # Determine health status\n        if health_score >= 90:\n            status = \"excellent\"\n        elif health_score >= 75:\n            status = \"good\"\n        elif health_score >= 50:\n            status = \"degraded\"\n        else:\n            status = \"critical\"\n        \n        return {\n            'score': health_score,\n            'status': status,\n            'issues': issues\n        }\n    \n    def generate_report(self, hours: int = 24) -> Dict[str, Any]:\n        \"\"\"Generate comprehensive monitoring report.\"\"\"\n        end_time = time.time()\n        start_time = end_time - (hours * 3600)\n        \n        report = {\n            'period': {\n                'start_time': start_time,\n                'end_time': end_time,\n                'duration_hours': hours\n            },\n            'metrics_summary': {},\n            'alert_summary': {},\n            'anomaly_summary': {},\n            'performance_summary': {},\n            'recommendations': []\n        }\n        \n        # Metrics summary\n        with self._lock:\n            for name, metric in self.metrics.items():\n                values = metric.get_values(start_time, end_time)\n                if values:\n                    report['metrics_summary'][name] = {\n                        'data_points': len(values),\n                        'min': min(values),\n                        'max': max(values),\n                        'mean': statistics.mean(values),\n                        'median': statistics.median(values)\n                    }\n        \n        # Alert summary\n        alert_events = []\n        for alert in self.alerts.values():\n            if alert.last_triggered and alert.last_triggered >= start_time:\n                alert_events.append({\n                    'name': alert.name,\n                    'severity': alert.severity.value,\n                    'triggered_at': alert.last_triggered,\n                    'resolved_at': alert.last_resolved\n                })\n        \n        report['alert_summary'] = {\n            'total_events': len(alert_events),\n            'events': alert_events\n        }\n        \n        # Generate recommendations\n        recommendations = self._generate_recommendations()\n        report['recommendations'] = recommendations\n        \n        return report\n    \n    def _generate_recommendations(self) -> List[Dict[str, str]]:\n        \"\"\"Generate optimization recommendations.\"\"\"\n        recommendations = []\n        \n        # Check for consistently high resource utilization\n        cpu_metrics = []\n        memory_metrics = []\n        \n        for name, metric in self.metrics.items():\n            if 'cpu' in name.lower():\n                recent_values = metric.get_values(start_time=time.time() - 3600)\n                if recent_values:\n                    avg_value = statistics.mean(recent_values)\n                    cpu_metrics.append(avg_value)\n            \n            elif 'memory' in name.lower():\n                recent_values = metric.get_values(start_time=time.time() - 3600)\n                if recent_values:\n                    avg_value = statistics.mean(recent_values)\n                    memory_metrics.append(avg_value)\n        \n        if cpu_metrics and statistics.mean(cpu_metrics) > 80:\n            recommendations.append({\n                'type': 'resource_optimization',\n                'priority': 'high',\n                'description': 'High CPU utilization detected. Consider scaling up or optimizing CPU-intensive operations.',\n                'metric': 'cpu_utilization'\n            })\n        \n        if memory_metrics and statistics.mean(memory_metrics) > 85:\n            recommendations.append({\n                'type': 'resource_optimization',\n                'priority': 'high',\n                'description': 'High memory utilization detected. Consider increasing memory or optimizing memory usage.',\n                'metric': 'memory_utilization'\n            })\n        \n        # Check for performance bottlenecks\n        bottlenecks = self.performance_analyzer.identify_performance_bottlenecks()\n        for bottleneck in bottlenecks:\n            if bottleneck['severity'] in ['critical', 'high']:\n                recommendations.append({\n                    'type': 'performance_optimization',\n                    'priority': bottleneck['severity'],\n                    'description': f\"Performance issue: {bottleneck['description']}\",\n                    'metric': bottleneck['type']\n                })\n        \n        return recommendations\n\n\n# Global metrics collector instance\n_global_metrics_collector: Optional[MetricsCollector] = None\n\n\ndef get_metrics_collector() -> MetricsCollector:\n    \"\"\"Get or create global metrics collector.\"\"\"\n    global _global_metrics_collector\n    \n    if _global_metrics_collector is None:\n        _global_metrics_collector = MetricsCollector()\n        _global_metrics_collector.start_processing()\n    \n    return _global_metrics_collector\n\n\ndef shutdown_metrics_collector() -> None:\n    \"\"\"Shutdown global metrics collector.\"\"\"\n    global _global_metrics_collector\n    \n    if _global_metrics_collector:\n        _global_metrics_collector.stop_processing()\n        _global_metrics_collector = None\n\n\ndef record_performance_metric(response_time_ms: float, success: bool,\n                             throughput_rps: float, cpu_percent: float,\n                             memory_percent: float) -> None:\n    \"\"\"Record performance metrics.\"\"\"\n    collector = get_metrics_collector()\n    \n    # Record individual metrics\n    collector.record_metric(\"response_time_ms\", response_time_ms)\n    collector.record_metric(\"success_rate\", 1.0 if success else 0.0)\n    collector.record_metric(\"throughput_rps\", throughput_rps)\n    collector.record_metric(\"cpu_percent\", cpu_percent)\n    collector.record_metric(\"memory_percent\", memory_percent)\n    \n    # Add to performance analyzer\n    collector.performance_analyzer.add_performance_data(\n        response_time_ms, success, throughput_rps, cpu_percent, memory_percent\n    )\n\n\ndef create_default_alerts() -> List[str]:\n    \"\"\"Create default alert rules.\"\"\"\n    collector = get_metrics_collector()\n    alert_ids = []\n    \n    # High response time alert\n    alert_id = collector.create_alert(\n        name=\"High Response Time\",\n        metric_name=\"response_time_ms\",\n        condition=\"> 1000\",\n        severity=AlertSeverity.HIGH,\n        description=\"Response time exceeds 1 second\"\n    )\n    alert_ids.append(alert_id)\n    \n    # High error rate alert\n    alert_id = collector.create_alert(\n        name=\"High Error Rate\",\n        metric_name=\"success_rate\",\n        condition=\"< 0.95\",\n        severity=AlertSeverity.CRITICAL,\n        description=\"Success rate below 95%\"\n    )\n    alert_ids.append(alert_id)\n    \n    # High CPU utilization alert\n    alert_id = collector.create_alert(\n        name=\"High CPU Utilization\",\n        metric_name=\"cpu_percent\",\n        condition=\"> 80\",\n        severity=AlertSeverity.MEDIUM,\n        description=\"CPU utilization above 80%\"\n    )\n    alert_ids.append(alert_id)\n    \n    # High memory utilization alert\n    alert_id = collector.create_alert(\n        name=\"High Memory Utilization\",\n        metric_name=\"memory_percent\",\n        condition=\"> 85\",\n        severity=AlertSeverity.MEDIUM,\n        description=\"Memory utilization above 85%\"\n    )\n    alert_ids.append(alert_id)\n    \n    logger.info(f\"Created {len(alert_ids)} default alerts\")\n    return alert_ids"