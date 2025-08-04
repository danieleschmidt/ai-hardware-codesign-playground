"""
Comprehensive monitoring and health check system for AI Hardware Co-Design Playground.

This module provides system monitoring, health checks, metrics collection,
alerting, and observability features.
"""

import time
import psutil
import threading
import asyncio
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from collections import deque, defaultdict
from enum import Enum
import json
from pathlib import Path
import statistics
import logging
from datetime import datetime, timedelta

from .logging import get_logger
from .exceptions import MonitoringError, SystemHealthError

logger = get_logger(__name__)


class HealthStatus(Enum):
    """System health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class MetricType(Enum):
    """Types of metrics collected."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class MetricValue:
    """Individual metric value with metadata."""
    name: str
    value: Union[int, float]
    type: MetricType
    timestamp: float
    labels: Dict[str, str] = field(default_factory=dict)
    unit: str = ""
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metric to dictionary format."""
        return {
            "name": self.name,
            "value": self.value,
            "type": self.type.value,
            "timestamp": self.timestamp,
            "labels": self.labels,
            "unit": self.unit,
            "description": self.description
        }


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    name: str
    status: HealthStatus
    message: str
    timestamp: float
    duration_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert health check result to dictionary."""
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "timestamp": self.timestamp,
            "duration_ms": self.duration_ms,
            "metadata": self.metadata
        }


@dataclass
class SystemMetrics:
    """System-wide metrics snapshot."""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_available_gb: float
    disk_usage_percent: float
    disk_free_gb: float
    network_bytes_sent: int
    network_bytes_recv: int
    active_connections: int
    load_average: List[float]
    uptime_seconds: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert system metrics to dictionary."""
        return {
            "timestamp": self.timestamp,
            "cpu_percent": self.cpu_percent,
            "memory_percent": self.memory_percent,
            "memory_used_gb": self.memory_used_gb,
            "memory_available_gb": self.memory_available_gb,
            "disk_usage_percent": self.disk_usage_percent,
            "disk_free_gb": self.disk_free_gb,
            "network_bytes_sent": self.network_bytes_sent,
            "network_bytes_recv": self.network_bytes_recv,
            "active_connections": self.active_connections,
            "load_average": self.load_average,
            "uptime_seconds": self.uptime_seconds
        }


class MetricCollector:
    """Collects and stores application metrics."""
    
    def __init__(self, max_history: int = 1000):
        """
        Initialize metric collector.
        
        Args:
            max_history: Maximum number of metric values to store per metric
        """
        self.max_history = max_history
        self._metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history))
        self._lock = threading.Lock()
        self._start_time = time.time()
        
        logger.info("Initialized MetricCollector", max_history=max_history)
    
    def record_counter(self, name: str, value: Union[int, float] = 1, 
                      labels: Optional[Dict[str, str]] = None, **kwargs) -> None:
        """Record a counter metric (monotonically increasing)."""
        self._record_metric(name, value, MetricType.COUNTER, labels, **kwargs)
    
    def record_gauge(self, name: str, value: Union[int, float], 
                    labels: Optional[Dict[str, str]] = None, **kwargs) -> None:
        """Record a gauge metric (can increase or decrease)."""
        self._record_metric(name, value, MetricType.GAUGE, labels, **kwargs)
    
    def record_histogram(self, name: str, value: Union[int, float], 
                        labels: Optional[Dict[str, str]] = None, **kwargs) -> None:
        """Record a histogram metric (for distributions)."""
        self._record_metric(name, value, MetricType.HISTOGRAM, labels, **kwargs)
    
    def record_timer(self, name: str, duration_seconds: float, 
                    labels: Optional[Dict[str, str]] = None, **kwargs) -> None:
        """Record a timer metric (duration measurement)."""
        self._record_metric(name, duration_seconds, MetricType.TIMER, labels, **kwargs)
    
    def _record_metric(self, name: str, value: Union[int, float], metric_type: MetricType,
                      labels: Optional[Dict[str, str]] = None, **kwargs) -> None:
        """Internal method to record a metric."""
        metric = MetricValue(
            name=name,
            value=value,
            type=metric_type,
            timestamp=time.time(),
            labels=labels or {},
            unit=kwargs.get("unit", ""),
            description=kwargs.get("description", "")
        )
        
        with self._lock:
            self._metrics[name].append(metric)
        
        logger.debug(f"Recorded metric: {name} = {value} ({metric_type.value})")
    
    def get_metric_history(self, name: str, limit: Optional[int] = None) -> List[MetricValue]:
        """Get historical values for a metric."""
        with self._lock:
            if name not in self._metrics:
                return []
            
            history = list(self._metrics[name])
            if limit:
                history = history[-limit:]
            
            return history
    
    def get_metric_stats(self, name: str, window_seconds: Optional[float] = None) -> Dict[str, Any]:
        """Get statistical summary of a metric."""
        history = self.get_metric_history(name)
        
        if not history:
            return {"error": "No data available"}
        
        # Filter by time window if specified
        if window_seconds:
            cutoff_time = time.time() - window_seconds
            history = [m for m in history if m.timestamp >= cutoff_time]
        
        if not history:
            return {"error": "No data in specified time window"}
        
        values = [m.value for m in history]
        
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "latest": values[-1],
            "latest_timestamp": history[-1].timestamp
        }
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get current values for all metrics."""
        with self._lock:
            return {
                name: {
                    "latest_value": list(history)[-1].value if history else None,
                    "latest_timestamp": list(history)[-1].timestamp if history else None,
                    "count": len(history)
                }
                for name, history in self._metrics.items()
            }
    
    def export_metrics(self, format: str = "json") -> str:
        """Export metrics in specified format."""
        all_metrics = []
        
        with self._lock:
            for name, history in self._metrics.items():
                for metric in history:
                    all_metrics.append(metric.to_dict())
        
        if format.lower() == "json":
            return json.dumps(all_metrics, indent=2)
        elif format.lower() == "prometheus":
            return self._export_prometheus_format(all_metrics)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _export_prometheus_format(self, metrics: List[Dict[str, Any]]) -> str:
        """Export metrics in Prometheus format."""
        lines = []
        
        # Group metrics by name
        by_name = defaultdict(list)
        for metric in metrics:
            by_name[metric["name"]].append(metric)
        
        for name, metric_list in by_name.items():
            latest = metric_list[-1]  # Get latest value
            
            # Add help and type comments
            lines.append(f"# HELP {name} {latest.get('description', '')}")
            lines.append(f"# TYPE {name} {latest['type']}")
            
            # Add metric value with labels
            labels_str = ""
            if latest["labels"]:
                label_pairs = [f'{k}="{v}"' for k, v in latest["labels"].items()]
                labels_str = "{" + ",".join(label_pairs) + "}"
            
            lines.append(f"{name}{labels_str} {latest['value']} {int(latest['timestamp'] * 1000)}")
            lines.append("")
        
        return "\n".join(lines)


class HealthChecker:
    """Performs comprehensive system health checks."""
    
    def __init__(self):
        """Initialize health checker."""
        self._checks: Dict[str, Callable] = {}
        self._results: Dict[str, HealthCheckResult] = {}
        self._lock = threading.Lock()
        
        # Register default health checks
        self._register_default_checks()
        
        logger.info("Initialized HealthChecker")
    
    def register_check(self, name: str, check_func: Callable[[], HealthCheckResult]) -> None:
        """Register a custom health check."""
        with self._lock:
            self._checks[name] = check_func
        
        logger.info(f"Registered health check: {name}")
    
    def run_check(self, name: str) -> HealthCheckResult:
        """Run a specific health check."""
        with self._lock:
            if name not in self._checks:
                return HealthCheckResult(
                    name=name,
                    status=HealthStatus.UNKNOWN,
                    message=f"Health check '{name}' not found",
                    timestamp=time.time(),
                    duration_ms=0.0
                )
            
            check_func = self._checks[name]
        
        start_time = time.time()
        
        try:
            result = check_func()
            result.duration_ms = (time.time() - start_time) * 1000
            
            with self._lock:
                self._results[name] = result
            
            logger.debug(f"Health check '{name}' completed: {result.status.value}")
            return result
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            result = HealthCheckResult(
                name=name,
                status=HealthStatus.CRITICAL,
                message=f"Health check failed: {e}",
                timestamp=time.time(),
                duration_ms=duration_ms
            )
            
            with self._lock:
                self._results[name] = result
            
            logger.error(f"Health check '{name}' failed", exception=e)
            return result
    
    def run_all_checks(self) -> Dict[str, HealthCheckResult]:
        """Run all registered health checks."""
        results = {}
        
        with self._lock:
            check_names = list(self._checks.keys())
        
        for name in check_names:
            results[name] = self.run_check(name)
        
        return results
    
    def get_overall_status(self) -> HealthStatus:
        """Get overall system health status."""
        results = self.run_all_checks()
        
        if not results:
            return HealthStatus.UNKNOWN
        
        statuses = [result.status for result in results.values()]
        
        if HealthStatus.CRITICAL in statuses:
            return HealthStatus.CRITICAL
        elif HealthStatus.WARNING in statuses:
            return HealthStatus.WARNING
        elif all(status == HealthStatus.HEALTHY for status in statuses):
            return HealthStatus.HEALTHY
        else:
            return HealthStatus.UNKNOWN
    
    def _register_default_checks(self) -> None:
        """Register default system health checks."""
        self.register_check("cpu_usage", self._check_cpu_usage)
        self.register_check("memory_usage", self._check_memory_usage)
        self.register_check("disk_space", self._check_disk_space)
        self.register_check("system_load", self._check_system_load)
        self.register_check("process_health", self._check_process_health)
    
    def _check_cpu_usage(self) -> HealthCheckResult:
        """Check CPU usage levels."""
        cpu_percent = psutil.cpu_percent(interval=1)
        
        if cpu_percent > 90:
            status = HealthStatus.CRITICAL
            message = f"Critical CPU usage: {cpu_percent:.1f}%"
        elif cpu_percent > 75:
            status = HealthStatus.WARNING
            message = f"High CPU usage: {cpu_percent:.1f}%"
        else:
            status = HealthStatus.HEALTHY
            message = f"CPU usage normal: {cpu_percent:.1f}%"
        
        return HealthCheckResult(
            name="cpu_usage",
            status=status,
            message=message,
            timestamp=time.time(),
            duration_ms=0.0,
            metadata={"cpu_percent": cpu_percent}
        )
    
    def _check_memory_usage(self) -> HealthCheckResult:
        """Check memory usage levels."""
        memory = psutil.virtual_memory()
        
        if memory.percent > 90:
            status = HealthStatus.CRITICAL
            message = f"Critical memory usage: {memory.percent:.1f}%"
        elif memory.percent > 80:
            status = HealthStatus.WARNING
            message = f"High memory usage: {memory.percent:.1f}%"
        else:
            status = HealthStatus.HEALTHY
            message = f"Memory usage normal: {memory.percent:.1f}%"
        
        return HealthCheckResult(
            name="memory_usage",
            status=status,
            message=message,
            timestamp=time.time(),
            duration_ms=0.0,
            metadata={
                "memory_percent": memory.percent,
                "memory_used_gb": memory.used / (1024**3),
                "memory_available_gb": memory.available / (1024**3)
            }
        )
    
    def _check_disk_space(self) -> HealthCheckResult:
        """Check available disk space."""
        disk = psutil.disk_usage('/')
        usage_percent = (disk.used / disk.total) * 100
        
        if usage_percent > 95:
            status = HealthStatus.CRITICAL
            message = f"Critical disk usage: {usage_percent:.1f}%"
        elif usage_percent > 85:
            status = HealthStatus.WARNING
            message = f"High disk usage: {usage_percent:.1f}%"
        else:
            status = HealthStatus.HEALTHY
            message = f"Disk usage normal: {usage_percent:.1f}%"
        
        return HealthCheckResult(
            name="disk_space",
            status=status,
            message=message,
            timestamp=time.time(),
            duration_ms=0.0,
            metadata={
                "disk_usage_percent": usage_percent,
                "disk_free_gb": disk.free / (1024**3),
                "disk_total_gb": disk.total / (1024**3)
            }
        )
    
    def _check_system_load(self) -> HealthCheckResult:
        """Check system load average."""
        try:
            load_avg = psutil.getloadavg()
            cpu_count = psutil.cpu_count()
            
            # Normalize load average by CPU count
            normalized_load = load_avg[0] / cpu_count if cpu_count > 0 else load_avg[0]
            
            if normalized_load > 2.0:
                status = HealthStatus.CRITICAL
                message = f"Critical system load: {normalized_load:.2f}"
            elif normalized_load > 1.5:
                status = HealthStatus.WARNING
                message = f"High system load: {normalized_load:.2f}"
            else:
                status = HealthStatus.HEALTHY
                message = f"System load normal: {normalized_load:.2f}"
            
            return HealthCheckResult(
                name="system_load",
                status=status,
                message=message,
                timestamp=time.time(),
                duration_ms=0.0,
                metadata={
                    "load_average": list(load_avg),
                    "normalized_load": normalized_load,
                    "cpu_count": cpu_count
                }
            )
            
        except (AttributeError, OSError):
            # getloadavg not available on all systems
            return HealthCheckResult(
                name="system_load",
                status=HealthStatus.UNKNOWN,
                message="System load monitoring not available",
                timestamp=time.time(),
                duration_ms=0.0
            )
    
    def _check_process_health(self) -> HealthCheckResult:
        """Check current process health."""
        try:
            process = psutil.Process()
            
            # Check if process is responsive
            memory_info = process.memory_info()
            cpu_percent = process.cpu_percent()
            
            # Check for memory leaks (simple heuristic)
            memory_mb = memory_info.rss / (1024 * 1024)
            
            if memory_mb > 2048:  # More than 2GB
                status = HealthStatus.WARNING
                message = f"High process memory usage: {memory_mb:.1f}MB"
            elif cpu_percent > 50:  # High CPU usage
                status = HealthStatus.WARNING
                message = f"High process CPU usage: {cpu_percent:.1f}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"Process health normal (Memory: {memory_mb:.1f}MB, CPU: {cpu_percent:.1f}%)"
            
            return HealthCheckResult(
                name="process_health",
                status=status,
                message=message,
                timestamp=time.time(),
                duration_ms=0.0,
                metadata={
                    "memory_mb": memory_mb,
                    "cpu_percent": cpu_percent,
                    "pid": process.pid,
                    "threads": process.num_threads()
                }
            )
            
        except psutil.Error as e:
            return HealthCheckResult(
                name="process_health",
                status=HealthStatus.CRITICAL,
                message=f"Process health check failed: {e}",
                timestamp=time.time(),
                duration_ms=0.0
            )


class SystemMonitor:
    """Comprehensive system monitoring with metrics collection and health checks."""
    
    def __init__(self, collection_interval: float = 60.0):
        """
        Initialize system monitor.
        
        Args:
            collection_interval: Interval in seconds between metric collections
        """
        self.collection_interval = collection_interval
        self.metric_collector = MetricCollector()
        self.health_checker = HealthChecker()
        
        # Monitoring state
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._start_time = time.time()
        
        logger.info("Initialized SystemMonitor", collection_interval=collection_interval)
    
    def start_monitoring(self) -> None:
        """Start background monitoring."""
        if self._monitoring:
            logger.warning("System monitoring already running")
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self._monitor_thread.start()
        
        logger.info("Started system monitoring")
    
    def stop_monitoring(self) -> None:
        """Stop background monitoring."""
        if not self._monitoring:
            logger.warning("System monitoring not running")
            return
        
        self._monitoring = False
        
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
        
        logger.info("Stopped system monitoring")
    
    def collect_system_metrics(self) -> SystemMetrics:
        """Collect comprehensive system metrics."""
        timestamp = time.time()
        
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # Memory metrics
        memory = psutil.virtual_memory()
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        disk_usage_percent = (disk.used / disk.total) * 100
        
        # Network metrics
        network = psutil.net_io_counters()
        
        # System load
        try:
            load_avg = list(psutil.getloadavg())
        except (AttributeError, OSError):
            load_avg = [0.0, 0.0, 0.0]
        
        # Connection count
        try:
            connections = len(psutil.net_connections())
        except (psutil.AccessDenied, OSError):
            connections = 0
        
        # Uptime
        uptime = timestamp - self._start_time
        
        metrics = SystemMetrics(
            timestamp=timestamp,
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used_gb=memory.used / (1024**3),
            memory_available_gb=memory.available / (1024**3),
            disk_usage_percent=disk_usage_percent,
            disk_free_gb=disk.free / (1024**3),
            network_bytes_sent=network.bytes_sent,
            network_bytes_recv=network.bytes_recv,
            active_connections=connections,
            load_average=load_avg,
            uptime_seconds=uptime
        )
        
        # Record metrics
        self.metric_collector.record_gauge("system_cpu_percent", cpu_percent)
        self.metric_collector.record_gauge("system_memory_percent", memory.percent)
        self.metric_collector.record_gauge("system_memory_used_gb", memory.used / (1024**3))
        self.metric_collector.record_gauge("system_disk_usage_percent", disk_usage_percent)
        self.metric_collector.record_gauge("system_network_bytes_sent", network.bytes_sent)
        self.metric_collector.record_gauge("system_network_bytes_recv", network.bytes_recv)
        self.metric_collector.record_gauge("system_uptime_seconds", uptime)
        
        return metrics
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status."""
        overall_status = self.health_checker.get_overall_status()
        individual_results = self.health_checker.run_all_checks()
        
        return {
            "overall_status": overall_status.value,
            "timestamp": time.time(),
            "checks": {name: result.to_dict() for name, result in individual_results.items()}
        }
    
    def get_monitoring_summary(self) -> Dict[str, Any]:
        """Get monitoring summary with key metrics and health status."""
        system_metrics = self.collect_system_metrics()
        health_status = self.get_health_status()
        
        return {
            "system_metrics": system_metrics.to_dict(),
            "health_status": health_status,
            "monitoring": {
                "uptime_seconds": time.time() - self._start_time,
                "monitoring_active": self._monitoring,
                "collection_interval": self.collection_interval
            },
            "metrics_summary": self.metric_collector.get_all_metrics()
        }
    
    def export_monitoring_data(self, format: str = "json", 
                              include_history: bool = False) -> str:
        """Export monitoring data in specified format."""
        data = {
            "timestamp": time.time(),
            "system_metrics": self.collect_system_metrics().to_dict(),
            "health_status": self.get_health_status(),
            "uptime_seconds": time.time() - self._start_time
        }
        
        if include_history:
            data["metrics_history"] = json.loads(self.metric_collector.export_metrics("json"))
        
        if format.lower() == "json":
            return json.dumps(data, indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _monitoring_loop(self) -> None:
        """Background monitoring loop."""
        logger.info("System monitoring loop started")
        
        while self._monitoring:
            try:
                # Collect system metrics
                self.collect_system_metrics()
                
                # Run health checks periodically (every 5 collection cycles)
                if int(time.time()) % (self.collection_interval * 5) == 0:
                    self.health_checker.run_all_checks()
                
                # Wait for next collection
                time.sleep(self.collection_interval)
                
            except Exception as e:
                logger.error("Error in monitoring loop", exception=e)
                time.sleep(self.collection_interval)
        
        logger.info("System monitoring loop stopped")


# Global monitoring instance
_monitor: Optional[SystemMonitor] = None


def get_system_monitor() -> SystemMonitor:
    """Get global system monitor instance."""
    global _monitor
    
    if _monitor is None:
        _monitor = SystemMonitor()
        _monitor.start_monitoring()
    
    return _monitor


def record_metric(name: str, value: Union[int, float], metric_type: str = "gauge", 
                 labels: Optional[Dict[str, str]] = None, **kwargs) -> None:
    """Convenience function to record a metric."""
    monitor = get_system_monitor()
    
    if metric_type.lower() == "counter":
        monitor.metric_collector.record_counter(name, value, labels, **kwargs)
    elif metric_type.lower() == "gauge":
        monitor.metric_collector.record_gauge(name, value, labels, **kwargs)
    elif metric_type.lower() == "histogram":
        monitor.metric_collector.record_histogram(name, value, labels, **kwargs)
    elif metric_type.lower() == "timer":
        monitor.metric_collector.record_timer(name, value, labels, **kwargs)
    else:
        raise ValueError(f"Unknown metric type: {metric_type}")


def get_health_status() -> Dict[str, Any]:
    """Convenience function to get system health status."""
    monitor = get_system_monitor()
    return monitor.get_health_status()


class MonitoringDecorator:
    """Decorator for monitoring function execution."""
    
    def __init__(self, metric_name: Optional[str] = None, 
                 record_errors: bool = True, 
                 record_duration: bool = True):
        """
        Initialize monitoring decorator.
        
        Args:
            metric_name: Custom metric name (defaults to function name)
            record_errors: Whether to record error metrics
            record_duration: Whether to record execution duration
        """
        self.metric_name = metric_name
        self.record_errors = record_errors
        self.record_duration = record_duration
    
    def __call__(self, func: Callable) -> Callable:
        """Apply monitoring to function."""
        metric_name = self.metric_name or f"function_{func.__name__}"
        
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                
                # Record success
                record_metric(f"{metric_name}_calls", 1, "counter", {"status": "success"})
                
                # Record duration
                if self.record_duration:
                    duration = time.time() - start_time
                    record_metric(f"{metric_name}_duration_seconds", duration, "timer")
                
                return result
                
            except Exception as e:
                # Record error
                if self.record_errors:
                    record_metric(f"{metric_name}_calls", 1, "counter", {"status": "error"})
                    record_metric(f"{metric_name}_errors", 1, "counter", {"error_type": type(e).__name__})
                
                raise
        
        return wrapper


def monitor_function(metric_name: Optional[str] = None, 
                    record_errors: bool = True, 
                    record_duration: bool = True):
    """Decorator factory for monitoring functions."""
    return MonitoringDecorator(metric_name, record_errors, record_duration)