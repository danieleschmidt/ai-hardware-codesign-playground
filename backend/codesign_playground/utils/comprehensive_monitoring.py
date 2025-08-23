"""
Comprehensive Monitoring and Observability System.

This module provides advanced monitoring, tracing, and observability features
for the hardware co-design platform with real-time analytics.
"""

import asyncio
import time
import psutil
import threading
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import logging
from collections import deque, defaultdict
from concurrent.futures import ThreadPoolExecutor
from .logging import get_logger

logger = get_logger(__name__)


class MetricType(Enum):
    """Types of metrics for monitoring."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"
    RATE = "rate"


class AlertSeverity(Enum):
    """Alert severity levels."""
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"


@dataclass
class MetricPoint:
    """Single metric data point."""
    
    timestamp: float
    name: str
    value: Union[int, float]
    metric_type: MetricType
    tags: Dict[str, str] = field(default_factory=dict)
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class Alert:
    """System alert definition."""
    
    alert_id: str
    name: str
    description: str
    severity: AlertSeverity
    metric_name: str
    threshold_value: float
    comparison_operator: str  # >, <, >=, <=, ==, !=
    duration_seconds: int
    triggered_at: Optional[float] = None
    resolved_at: Optional[float] = None
    is_active: bool = False


@dataclass
class SystemHealth:
    """System health snapshot."""
    
    timestamp: float
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: Dict[str, float]
    process_count: int
    thread_count: int
    open_files: int
    uptime_seconds: float
    load_average: List[float]


class AdvancedMetricsCollector:
    """
    Advanced metrics collection system with real-time processing.
    
    Collects, processes, and analyzes metrics in real-time with
    intelligent alerting and anomaly detection.
    """
    
    def __init__(self, retention_hours: int = 24, collection_interval: int = 30):
        self.retention_hours = retention_hours
        self.collection_interval = collection_interval
        
        # Metric storage
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=int(retention_hours * 3600 / collection_interval)))
        self.metric_metadata: Dict[str, Dict[str, Any]] = {}
        
        # Real-time processing
        self.metric_processors: List[Callable] = []
        self.anomaly_detectors: Dict[str, Callable] = {}
        
        # Alerting
        self.alerts: Dict[str, Alert] = {}
        self.alert_history: List[Dict[str, Any]] = []
        self.alert_callbacks: List[Callable] = []
        
        # System monitoring
        self.system_health_history: deque = deque(maxlen=1440)  # 24 hours at 1-minute intervals
        
        # Performance tracking
        self.performance_baselines: Dict[str, float] = {}
        self.sla_targets: Dict[str, float] = {}
        
        # Threading
        self.collection_thread: Optional[threading.Thread] = None
        self.processing_thread: Optional[threading.Thread] = None
        self.is_running = False
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Initialize default SLA targets
        self._initialize_default_slas()
    
    def start_monitoring(self) -> None:
        """Start continuous monitoring and collection."""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Start collection thread
        self.collection_thread = threading.Thread(
            target=self._continuous_collection, daemon=True
        )
        self.collection_thread.start()
        
        # Start processing thread
        self.processing_thread = threading.Thread(
            target=self._continuous_processing, daemon=True
        )
        self.processing_thread.start()
        
        logger.info("Advanced monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop continuous monitoring."""
        self.is_running = False
        
        if self.collection_thread:
            self.collection_thread.join(timeout=5)
        if self.processing_thread:
            self.processing_thread.join(timeout=5)
        
        logger.info("Advanced monitoring stopped")
    
    def record_metric(
        self,
        name: str,
        value: Union[int, float],
        metric_type: MetricType = MetricType.GAUGE,
        tags: Optional[Dict[str, str]] = None,
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Record a metric with metadata.
        
        Args:
            name: Metric name
            value: Metric value
            metric_type: Type of metric
            tags: Additional tags for filtering
            labels: Additional labels for grouping
        """
        timestamp = time.time()
        
        metric_point = MetricPoint(
            timestamp=timestamp,
            name=name,
            value=value,
            metric_type=metric_type,
            tags=tags or {},
            labels=labels or {}
        )
        
        # Store metric
        self.metrics[name].append(metric_point)
        
        # Update metadata
        if name not in self.metric_metadata:
            self.metric_metadata[name] = {
                "created_at": timestamp,
                "metric_type": metric_type,
                "sample_count": 0,
                "last_value": value,
                "min_value": value,
                "max_value": value,
                "sum_value": 0.0
            }
        
        metadata = self.metric_metadata[name]
        metadata["sample_count"] += 1
        metadata["last_value"] = value
        metadata["min_value"] = min(metadata["min_value"], value)
        metadata["max_value"] = max(metadata["max_value"], value)
        metadata["sum_value"] += value
        metadata["updated_at"] = timestamp
        
        # Process metric in real-time
        self._process_metric_realtime(metric_point)
    
    def add_alert(
        self,
        name: str,
        description: str,
        metric_name: str,
        threshold_value: float,
        comparison_operator: str,
        severity: AlertSeverity = AlertSeverity.WARNING,
        duration_seconds: int = 300
    ) -> str:
        """
        Add alert rule for metric monitoring.
        
        Args:
            name: Alert name
            description: Alert description
            metric_name: Metric to monitor
            threshold_value: Threshold for triggering alert
            comparison_operator: Comparison operator
            severity: Alert severity
            duration_seconds: Duration threshold
            
        Returns:
            Alert ID
        """
        alert_id = f"alert_{name}_{int(time.time())}"
        
        alert = Alert(
            alert_id=alert_id,
            name=name,
            description=description,
            severity=severity,
            metric_name=metric_name,
            threshold_value=threshold_value,
            comparison_operator=comparison_operator,
            duration_seconds=duration_seconds
        )
        
        self.alerts[alert_id] = alert
        logger.info(f"Added alert: {name} for metric {metric_name}")
        
        return alert_id
    
    def add_alert_callback(self, callback: Callable[[Alert], None]) -> None:
        """Add callback function for alert notifications."""
        self.alert_callbacks.append(callback)
    
    def get_metric_statistics(self, name: str, time_range_minutes: int = 60) -> Dict[str, Any]:
        """
        Get statistical analysis of a metric.
        
        Args:
            name: Metric name
            time_range_minutes: Time range for analysis
            
        Returns:
            Statistical summary
        """
        if name not in self.metrics:
            return {}
        
        cutoff_time = time.time() - (time_range_minutes * 60)
        recent_points = [
            point for point in self.metrics[name]
            if point.timestamp >= cutoff_time
        ]
        
        if not recent_points:
            return {}
        
        values = [point.value for point in recent_points]
        
        # Calculate statistics
        import statistics
        
        stats = {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "std_dev": statistics.stdev(values) if len(values) > 1 else 0.0,
            "percentile_95": self._percentile(values, 95),
            "percentile_99": self._percentile(values, 99),
            "first_timestamp": recent_points[0].timestamp,
            "last_timestamp": recent_points[-1].timestamp
        }
        
        # Add rate calculation for counters
        if recent_points[0].metric_type == MetricType.COUNTER:
            time_diff = recent_points[-1].timestamp - recent_points[0].timestamp
            value_diff = recent_points[-1].value - recent_points[0].value
            stats["rate_per_second"] = value_diff / time_diff if time_diff > 0 else 0.0
        
        return stats
    
    def collect_system_health(self) -> SystemHealth:
        """Collect comprehensive system health metrics."""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            load_avg = psutil.getloadavg() if hasattr(psutil, 'getloadavg') else [0, 0, 0]
            
            # Memory metrics
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            
            # Network metrics
            network = psutil.net_io_counters()
            network_io = {
                "bytes_sent": network.bytes_sent,
                "bytes_recv": network.bytes_recv,
                "packets_sent": network.packets_sent,
                "packets_recv": network.packets_recv
            }
            
            # Process metrics
            process_count = len(psutil.pids())
            
            # Thread and file descriptor counts
            current_process = psutil.Process()
            thread_count = current_process.num_threads()
            
            try:
                open_files = len(current_process.open_files())
            except (psutil.AccessDenied, OSError):
                open_files = 0
            
            # Uptime
            boot_time = psutil.boot_time()
            uptime = time.time() - boot_time
            
            health = SystemHealth(
                timestamp=time.time(),
                cpu_usage=cpu_percent,
                memory_usage=memory_percent,
                disk_usage=disk_percent,
                network_io=network_io,
                process_count=process_count,
                thread_count=thread_count,
                open_files=open_files,
                uptime_seconds=uptime,
                load_average=load_avg
            )
            
            # Record as metrics
            self.record_metric("system.cpu_usage", cpu_percent, MetricType.GAUGE)
            self.record_metric("system.memory_usage", memory_percent, MetricType.GAUGE)
            self.record_metric("system.disk_usage", disk_percent, MetricType.GAUGE)
            self.record_metric("system.process_count", process_count, MetricType.GAUGE)
            self.record_metric("system.thread_count", thread_count, MetricType.GAUGE)
            self.record_metric("system.open_files", open_files, MetricType.GAUGE)
            
            return health
            
        except Exception as e:
            logger.error(f"Failed to collect system health: {e}")
            return SystemHealth(
                timestamp=time.time(),
                cpu_usage=0.0,
                memory_usage=0.0,
                disk_usage=0.0,
                network_io={},
                process_count=0,
                thread_count=0,
                open_files=0,
                uptime_seconds=0.0,
                load_average=[0, 0, 0]
            )
    
    def detect_anomalies(self, metric_name: str, sensitivity: float = 2.0) -> List[Dict[str, Any]]:
        """
        Detect anomalies in metric data using statistical analysis.
        
        Args:
            metric_name: Name of metric to analyze
            sensitivity: Sensitivity multiplier for anomaly detection
            
        Returns:
            List of detected anomalies
        """
        if metric_name not in self.metrics or len(self.metrics[metric_name]) < 10:
            return []
        
        points = list(self.metrics[metric_name])
        values = [point.value for point in points]
        
        # Calculate statistical thresholds
        import statistics
        mean = statistics.mean(values)
        std_dev = statistics.stdev(values) if len(values) > 1 else 0
        
        upper_threshold = mean + (sensitivity * std_dev)
        lower_threshold = mean - (sensitivity * std_dev)
        
        # Find anomalies
        anomalies = []
        for point in points:
            if point.value > upper_threshold or point.value < lower_threshold:
                anomalies.append({
                    "timestamp": point.timestamp,
                    "value": point.value,
                    "expected_range": [lower_threshold, upper_threshold],
                    "deviation": abs(point.value - mean) / std_dev if std_dev > 0 else 0,
                    "type": "high" if point.value > upper_threshold else "low"
                })
        
        return anomalies
    
    def get_sla_compliance(self, time_range_hours: int = 24) -> Dict[str, Dict[str, Any]]:
        """
        Calculate SLA compliance for monitored metrics.
        
        Args:
            time_range_hours: Time range for SLA calculation
            
        Returns:
            SLA compliance report
        """
        compliance_report = {}
        cutoff_time = time.time() - (time_range_hours * 3600)
        
        for metric_name, target_value in self.sla_targets.items():
            if metric_name not in self.metrics:
                continue
            
            recent_points = [
                point for point in self.metrics[metric_name]
                if point.timestamp >= cutoff_time
            ]
            
            if not recent_points:
                continue
            
            # Calculate compliance based on metric type
            if "latency" in metric_name.lower() or "response_time" in metric_name.lower():
                # For latency metrics, compliance is when values are below target
                compliant_points = sum(1 for p in recent_points if p.value <= target_value)
            elif "availability" in metric_name.lower() or "uptime" in metric_name.lower():
                # For availability metrics, compliance is when values are above target
                compliant_points = sum(1 for p in recent_points if p.value >= target_value)
            else:
                # Default: compliance when within 10% of target
                tolerance = target_value * 0.1
                compliant_points = sum(
                    1 for p in recent_points 
                    if abs(p.value - target_value) <= tolerance
                )
            
            total_points = len(recent_points)
            compliance_percentage = (compliant_points / total_points) * 100 if total_points > 0 else 0
            
            compliance_report[metric_name] = {
                "target_value": target_value,
                "compliance_percentage": compliance_percentage,
                "total_samples": total_points,
                "compliant_samples": compliant_points,
                "time_range_hours": time_range_hours,
                "is_meeting_sla": compliance_percentage >= 95.0  # 95% compliance threshold
            }
        
        return compliance_report
    
    def _continuous_collection(self) -> None:
        """Continuous metric collection in background thread."""
        while self.is_running:
            try:
                # Collect system health
                health = self.collect_system_health()
                self.system_health_history.append(health)
                
                # Sleep until next collection
                time.sleep(self.collection_interval)
                
            except Exception as e:
                logger.error(f"Error in continuous collection: {e}")
                time.sleep(self.collection_interval)
    
    def _continuous_processing(self) -> None:
        """Continuous metric processing and alerting."""
        while self.is_running:
            try:
                # Check alerts
                self._check_alerts()
                
                # Process anomaly detection
                self._process_anomaly_detection()
                
                # Clean up old data
                self._cleanup_old_data()
                
                # Sleep for processing interval
                time.sleep(60)  # Process every minute
                
            except Exception as e:
                logger.error(f"Error in continuous processing: {e}")
                time.sleep(60)
    
    def _process_metric_realtime(self, metric_point: MetricPoint) -> None:
        """Process metric in real-time."""
        # Apply metric processors
        for processor in self.metric_processors:
            try:
                processor(metric_point)
            except Exception as e:
                logger.error(f"Error in metric processor: {e}")
    
    def _check_alerts(self) -> None:
        """Check all alert conditions."""
        current_time = time.time()
        
        for alert in self.alerts.values():
            try:
                # Get recent metric values
                if alert.metric_name not in self.metrics:
                    continue
                
                recent_points = [
                    p for p in self.metrics[alert.metric_name]
                    if p.timestamp >= current_time - alert.duration_seconds
                ]
                
                if not recent_points:
                    continue
                
                # Check if alert condition is met
                latest_value = recent_points[-1].value
                condition_met = self._evaluate_alert_condition(
                    latest_value, alert.threshold_value, alert.comparison_operator
                )
                
                # Handle alert state changes
                if condition_met and not alert.is_active:
                    self._trigger_alert(alert)
                elif not condition_met and alert.is_active:
                    self._resolve_alert(alert)
                
            except Exception as e:
                logger.error(f"Error checking alert {alert.name}: {e}")
    
    def _evaluate_alert_condition(
        self, value: float, threshold: float, operator: str
    ) -> bool:
        """Evaluate alert condition."""
        operators = {
            ">": lambda v, t: v > t,
            "<": lambda v, t: v < t,
            ">=": lambda v, t: v >= t,
            "<=": lambda v, t: v <= t,
            "==": lambda v, t: v == t,
            "!=": lambda v, t: v != t
        }
        
        return operators.get(operator, lambda v, t: False)(value, threshold)
    
    def _trigger_alert(self, alert: Alert) -> None:
        """Trigger an alert."""
        alert.is_active = True
        alert.triggered_at = time.time()
        
        logger.warning(f"ALERT TRIGGERED: {alert.name} - {alert.description}")
        
        # Add to history
        self.alert_history.append({
            "alert_id": alert.alert_id,
            "name": alert.name,
            "action": "triggered",
            "timestamp": alert.triggered_at,
            "severity": alert.severity.value
        })
        
        # Notify callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")
    
    def _resolve_alert(self, alert: Alert) -> None:
        """Resolve an alert."""
        alert.is_active = False
        alert.resolved_at = time.time()
        
        logger.info(f"ALERT RESOLVED: {alert.name}")
        
        # Add to history
        self.alert_history.append({
            "alert_id": alert.alert_id,
            "name": alert.name,
            "action": "resolved", 
            "timestamp": alert.resolved_at,
            "severity": alert.severity.value
        })
    
    def _process_anomaly_detection(self) -> None:
        """Run anomaly detection on all metrics."""
        for metric_name in self.metrics.keys():
            if metric_name in self.anomaly_detectors:
                try:
                    detector = self.anomaly_detectors[metric_name]
                    anomalies = detector(metric_name)
                    
                    if anomalies:
                        logger.info(f"Anomalies detected in {metric_name}: {len(anomalies)}")
                        
                except Exception as e:
                    logger.error(f"Error in anomaly detection for {metric_name}: {e}")
    
    def _cleanup_old_data(self) -> None:
        """Clean up old metric data and alerts."""
        # Metric data is automatically cleaned up by deque maxlen
        
        # Clean up old alert history (keep last 1000 alerts)
        if len(self.alert_history) > 1000:
            self.alert_history = self.alert_history[-1000:]
    
    def _initialize_default_slas(self) -> None:
        """Initialize default SLA targets."""
        self.sla_targets = {
            "response_time_ms": 500.0,        # 500ms response time
            "availability_percent": 99.9,      # 99.9% availability
            "error_rate_percent": 1.0,         # <1% error rate
            "cpu_usage_percent": 80.0,         # <80% CPU usage
            "memory_usage_percent": 85.0,      # <85% memory usage
            "disk_usage_percent": 90.0         # <90% disk usage
        }
    
    def _percentile(self, values: List[float], percentile: float) -> float:
        """Calculate percentile of values."""
        if not values:
            return 0.0
        
        sorted_values = sorted(values)
        index = int((percentile / 100) * len(sorted_values))
        index = min(index, len(sorted_values) - 1)
        
        return sorted_values[index]
    
    def get_monitoring_dashboard(self) -> Dict[str, Any]:
        """Generate comprehensive monitoring dashboard data."""
        current_time = time.time()
        
        # System overview
        recent_health = self.system_health_history[-1] if self.system_health_history else None
        
        # Alert summary
        active_alerts = [alert for alert in self.alerts.values() if alert.is_active]
        alert_summary = {
            severity.value: len([a for a in active_alerts if a.severity == severity])
            for severity in AlertSeverity
        }
        
        # Top metrics by activity
        metric_activity = {
            name: len(points) for name, points in self.metrics.items()
        }
        top_metrics = sorted(
            metric_activity.items(), key=lambda x: x[1], reverse=True
        )[:10]
        
        # SLA compliance
        sla_compliance = self.get_sla_compliance(24)
        overall_sla = (
            sum(report["compliance_percentage"] for report in sla_compliance.values()) /
            len(sla_compliance) if sla_compliance else 100.0
        )
        
        return {
            "timestamp": current_time,
            "system_health": {
                "cpu_usage": recent_health.cpu_usage if recent_health else 0,
                "memory_usage": recent_health.memory_usage if recent_health else 0,
                "disk_usage": recent_health.disk_usage if recent_health else 0,
                "uptime_hours": (recent_health.uptime_seconds / 3600) if recent_health else 0
            },
            "alerts": {
                "active_count": len(active_alerts),
                "by_severity": alert_summary,
                "recent_alerts": self.alert_history[-10:]
            },
            "metrics": {
                "total_metrics": len(self.metrics),
                "total_data_points": sum(len(points) for points in self.metrics.values()),
                "top_active_metrics": top_metrics
            },
            "sla": {
                "overall_compliance": overall_sla,
                "individual_compliance": sla_compliance
            },
            "monitoring_status": {
                "is_running": self.is_running,
                "collection_interval": self.collection_interval,
                "retention_hours": self.retention_hours
            }
        }


# Global monitoring instance
global_monitor = AdvancedMetricsCollector()