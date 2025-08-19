"""
Comprehensive health monitoring system for AI Hardware Co-Design Playground.

This module provides deep health checks, system diagnostics, and predictive
failure detection for the entire platform.
"""

import time
# Optional dependency with fallback
try:
    import psutil
except ImportError:
    psutil = None
import threading
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
from collections import deque, defaultdict
import json
from pathlib import Path

from .logging import get_logger
from .monitoring import record_metric, get_system_monitor
from .exceptions import CodesignError

logger = get_logger(__name__)


class ComponentHealth(Enum):
    """Health status levels for system components."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    FAILED = "failed"
    UNKNOWN = "unknown"


class CheckResult(Enum):
    """Health check result types."""
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"
    ERROR = "error"


@dataclass
class HealthCheckResult:
    """Result of a health check execution."""
    check_name: str
    result: CheckResult
    value: Any
    message: str
    timestamp: float
    execution_time_ms: float
    metadata: Dict[str, Any]


@dataclass
class ComponentStatus:
    """Overall status of a system component."""
    name: str
    health: ComponentHealth
    last_check: float
    check_results: List[HealthCheckResult]
    dependencies: List[str]
    metrics: Dict[str, Any]


class HealthChecker:
    """Individual health check implementation."""
    
    def __init__(
        self,
        name: str,
        check_function: Callable[[], Any],
        warning_threshold: Optional[float] = None,
        critical_threshold: Optional[float] = None,
        interval_seconds: float = 30.0,
        timeout_seconds: float = 10.0,
        enabled: bool = True
    ):
        """Initialize health checker."""
        self.name = name
        self.check_function = check_function
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.interval_seconds = interval_seconds
        self.timeout_seconds = timeout_seconds
        self.enabled = enabled
        
        self.last_result: Optional[HealthCheckResult] = None
        self.result_history = deque(maxlen=100)
        self.consecutive_failures = 0
        self.last_execution = 0.0
    
    def should_run(self) -> bool:
        """Check if health check should run."""
        if not self.enabled:
            return False
        
        return time.time() - self.last_execution >= self.interval_seconds
    
    def execute(self) -> HealthCheckResult:
        """Execute the health check."""
        start_time = time.time()
        execution_start = start_time
        
        try:
            # Execute with timeout
            value = self._execute_with_timeout()
            
            # Determine result based on thresholds
            result = self._evaluate_result(value)
            
            execution_time = (time.time() - execution_start) * 1000
            
            # Create result
            check_result = HealthCheckResult(
                check_name=self.name,
                result=result,
                value=value,
                message=self._get_result_message(result, value),
                timestamp=start_time,
                execution_time_ms=execution_time,
                metadata={}
            )
            
            # Update state
            self.last_result = check_result
            self.result_history.append(check_result)
            self.last_execution = start_time
            
            if result == CheckResult.PASS:
                self.consecutive_failures = 0
            else:
                self.consecutive_failures += 1
            
            # Record metrics
            record_metric(f"health_check_{self.name}_result", 
                         {"pass": 1, "warn": 0.5, "fail": 0, "error": 0}[result.value],
                         "gauge")
            record_metric(f"health_check_{self.name}_execution_time", execution_time, "histogram")
            
            return check_result
            
        except Exception as e:
            execution_time = (time.time() - execution_start) * 1000
            
            check_result = HealthCheckResult(
                check_name=self.name,
                result=CheckResult.ERROR,
                value=None,
                message=f"Health check error: {e}",
                timestamp=start_time,
                execution_time_ms=execution_time,
                metadata={"error": str(e)}
            )
            
            self.last_result = check_result
            self.result_history.append(check_result)
            self.last_execution = start_time
            self.consecutive_failures += 1
            
            logger.error(f"Health check {self.name} failed with error: {e}")
            record_metric(f"health_check_{self.name}_error", 1, "counter")
            
            return check_result
    
    def _execute_with_timeout(self) -> Any:
        """Execute check function with timeout."""
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Health check {self.name} timed out after {self.timeout_seconds}s")
        
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(int(self.timeout_seconds))
        
        try:
            result = self.check_function()
            return result
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
    
    def _evaluate_result(self, value: Any) -> CheckResult:
        """Evaluate check result based on thresholds."""
        if value is None:
            return CheckResult.ERROR
        
        # Boolean checks
        if isinstance(value, bool):
            return CheckResult.PASS if value else CheckResult.FAIL
        
        # Numeric checks with thresholds
        if isinstance(value, (int, float)) and (self.warning_threshold or self.critical_threshold):
            if self.critical_threshold and value >= self.critical_threshold:
                return CheckResult.FAIL
            elif self.warning_threshold and value >= self.warning_threshold:
                return CheckResult.WARN
            else:
                return CheckResult.PASS
        
        # Default to pass for other value types
        return CheckResult.PASS
    
    def _get_result_message(self, result: CheckResult, value: Any) -> str:
        """Get human-readable message for result."""
        if result == CheckResult.PASS:
            return f"{self.name} is healthy"
        elif result == CheckResult.WARN:
            return f"{self.name} warning: value {value} exceeds warning threshold {self.warning_threshold}"
        elif result == CheckResult.FAIL:
            return f"{self.name} critical: value {value} exceeds critical threshold {self.critical_threshold}"
        else:
            return f"{self.name} error during check"


class SystemHealthMonitor:
    """Comprehensive system health monitoring."""
    
    def __init__(self):
        """Initialize system health monitor."""
        self.components: Dict[str, ComponentStatus] = {}
        self.health_checkers: Dict[str, List[HealthChecker]] = defaultdict(list)
        self.global_health = ComponentHealth.UNKNOWN
        
        self._monitoring_thread = None
        self._stop_monitoring = threading.Event()
        self._lock = threading.Lock()
        
        # Initialize built-in health checks
        self._setup_builtin_checks()
        
        logger.info("System health monitor initialized")
    
    def _setup_builtin_checks(self) -> None:
        """Set up built-in system health checks."""
        # System resource checks
        self.add_health_check(
            "system",
            HealthChecker(
                "cpu_usage",
                lambda: psutil.cpu_percent(interval=0.1),
                warning_threshold=80.0,
                critical_threshold=95.0,
                interval_seconds=30.0
            )
        )
        
        self.add_health_check(
            "system",
            HealthChecker(
                "memory_usage", 
                lambda: psutil.virtual_memory().percent,
                warning_threshold=80.0,
                critical_threshold=95.0,
                interval_seconds=30.0
            )
        )
        
        self.add_health_check(
            "system",
            HealthChecker(
                "disk_usage",
                lambda: psutil.disk_usage('/').percent,
                warning_threshold=85.0,
                critical_threshold=95.0,
                interval_seconds=60.0
            )
        )
        
        # Application-specific checks
        self.add_health_check(
            "application",
            HealthChecker(
                "api_server",
                self._check_api_server_health,
                interval_seconds=30.0
            )
        )
        
        self.add_health_check(
            "application", 
            HealthChecker(
                "worker_processes",
                self._check_worker_processes,
                interval_seconds=45.0
            )
        )
        
        # Database/storage checks (if applicable)
        self.add_health_check(
            "storage",
            HealthChecker(
                "temp_directory",
                lambda: Path("/tmp").exists() and Path("/tmp").is_dir(),
                interval_seconds=60.0
            )
        )
    
    def add_health_check(self, component: str, checker: HealthChecker) -> None:
        """Add health check for component."""
        with self._lock:
            self.health_checkers[component].append(checker)
            
            if component not in self.components:
                self.components[component] = ComponentStatus(
                    name=component,
                    health=ComponentHealth.UNKNOWN,
                    last_check=0.0,
                    check_results=[],
                    dependencies=[],
                    metrics={}
                )
        
        logger.info(f"Added health check {checker.name} for component {component}")
    
    def start_monitoring(self) -> None:
        """Start health monitoring thread."""
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            return
        
        self._stop_monitoring.clear()
        self._monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._monitoring_thread.start()
        
        logger.info("Health monitoring started")
        record_metric("health_monitor_started", 1, "counter")
    
    def stop_monitoring(self) -> None:
        """Stop health monitoring thread."""
        self._stop_monitoring.set()
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5.0)
        
        logger.info("Health monitoring stopped")
        record_metric("health_monitor_stopped", 1, "counter")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while not self._stop_monitoring.is_set():
            try:
                self._run_health_checks()
                self._update_component_health()
                self._update_global_health()
                
                # Sleep for a short interval
                self._stop_monitoring.wait(10.0)  # 10 second intervals
                
            except Exception as e:
                logger.error(f"Error in health monitoring loop: {e}")
                self._stop_monitoring.wait(30.0)  # Longer interval on error
    
    def _run_health_checks(self) -> None:
        """Run all due health checks."""
        for component, checkers in self.health_checkers.items():
            for checker in checkers:
                if checker.should_run():
                    try:
                        result = checker.execute()
                        
                        with self._lock:
                            # Update component status
                            if component in self.components:
                                self.components[component].check_results.append(result)
                                self.components[component].last_check = result.timestamp
                                
                                # Keep only recent results (last 50)
                                self.components[component].check_results = (
                                    self.components[component].check_results[-50:]
                                )
                        
                    except Exception as e:
                        logger.error(f"Failed to run health check {checker.name}: {e}")
    
    def _update_component_health(self) -> None:
        """Update health status for all components."""
        with self._lock:
            for component_name, component in self.components.items():
                if not component.check_results:
                    component.health = ComponentHealth.UNKNOWN
                    continue
                
                # Get recent check results (last 10)
                recent_results = component.check_results[-10:]
                
                # Count result types
                fail_count = sum(1 for r in recent_results if r.result == CheckResult.FAIL)
                error_count = sum(1 for r in recent_results if r.result == CheckResult.ERROR) 
                warn_count = sum(1 for r in recent_results if r.result == CheckResult.WARN)
                pass_count = sum(1 for r in recent_results if r.result == CheckResult.PASS)
                
                # Determine health based on recent results
                total_recent = len(recent_results)
                if fail_count > 0 or error_count >= total_recent * 0.5:
                    component.health = ComponentHealth.FAILED
                elif error_count > 0 or fail_count + warn_count >= total_recent * 0.3:
                    component.health = ComponentHealth.CRITICAL
                elif warn_count >= total_recent * 0.5:
                    component.health = ComponentHealth.WARNING
                elif pass_count > 0:
                    component.health = ComponentHealth.HEALTHY
                else:
                    component.health = ComponentHealth.UNKNOWN
                
                # Update component metrics
                component.metrics = {
                    "total_checks": len(component.check_results),
                    "recent_pass_rate": pass_count / max(1, total_recent),
                    "recent_fail_count": fail_count,
                    "recent_error_count": error_count,
                    "recent_warn_count": warn_count,
                    "last_check_age": time.time() - component.last_check
                }
                
                # Record component health metric
                health_score = {
                    ComponentHealth.HEALTHY: 1.0,
                    ComponentHealth.WARNING: 0.7,
                    ComponentHealth.CRITICAL: 0.3,
                    ComponentHealth.FAILED: 0.0,
                    ComponentHealth.UNKNOWN: 0.5
                }[component.health]
                
                record_metric(f"component_health_{component_name}", health_score, "gauge")
    
    def _update_global_health(self) -> None:
        """Update global system health."""
        if not self.components:
            self.global_health = ComponentHealth.UNKNOWN
            return
        
        component_healths = [comp.health for comp in self.components.values()]
        
        # Global health is the worst component health
        if ComponentHealth.FAILED in component_healths:
            self.global_health = ComponentHealth.FAILED
        elif ComponentHealth.CRITICAL in component_healths:
            self.global_health = ComponentHealth.CRITICAL
        elif ComponentHealth.WARNING in component_healths:
            self.global_health = ComponentHealth.WARNING
        elif all(h == ComponentHealth.HEALTHY for h in component_healths):
            self.global_health = ComponentHealth.HEALTHY
        else:
            self.global_health = ComponentHealth.WARNING
        
        # Record global health
        global_health_score = {
            ComponentHealth.HEALTHY: 1.0,
            ComponentHealth.WARNING: 0.7,
            ComponentHealth.CRITICAL: 0.3,
            ComponentHealth.FAILED: 0.0,
            ComponentHealth.UNKNOWN: 0.5
        }[self.global_health]
        
        record_metric("global_health_score", global_health_score, "gauge")
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status."""
        with self._lock:
            return {
                "global_health": self.global_health.value,
                "timestamp": time.time(),
                "components": {
                    name: {
                        "health": comp.health.value,
                        "last_check": comp.last_check,
                        "metrics": comp.metrics,
                        "recent_results": [
                            {
                                "check": r.check_name,
                                "result": r.result.value,
                                "message": r.message,
                                "timestamp": r.timestamp
                            }
                            for r in comp.check_results[-5:]  # Last 5 results
                        ]
                    }
                    for name, comp in self.components.items()
                }
            }
    
    def _check_api_server_health(self) -> bool:
        """Check if API server is responding."""
        try:
            import requests
            response = requests.get("http://localhost:8000/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def _check_worker_processes(self) -> int:
        """Count active worker processes."""
        try:
            # Count processes with 'worker' or 'celery' in name
            count = 0
            for proc in psutil.process_iter(['name', 'cmdline']):
                try:
                    name = proc.info['name'] or ''
                    cmdline = ' '.join(proc.info['cmdline'] or [])
                    
                    if 'worker' in name.lower() or 'worker' in cmdline.lower():
                        count += 1
                except:
                    continue
            return count
        except:
            return 0


# Global health monitor instance
_health_monitor: Optional[SystemHealthMonitor] = None
_monitor_lock = threading.Lock()


def get_health_monitor() -> SystemHealthMonitor:
    """Get global health monitor instance."""
    global _health_monitor
    
    with _monitor_lock:
        if _health_monitor is None:
            _health_monitor = SystemHealthMonitor()
            _health_monitor.start_monitoring()
        
        return _health_monitor


def get_system_health() -> Dict[str, Any]:
    """Get current system health status."""
    monitor = get_health_monitor()
    return monitor.get_health_status()


def add_custom_health_check(component: str, checker: HealthChecker) -> None:
    """Add custom health check to global monitor."""
    monitor = get_health_monitor()
    monitor.add_health_check(component, checker)