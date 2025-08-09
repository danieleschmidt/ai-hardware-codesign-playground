"""
Enhanced circuit breaker implementation for the AI Hardware Co-Design Playground.

This module provides advanced circuit breaker patterns with health checks,
adaptive thresholds, and integration with monitoring systems.
"""

import time
import asyncio
from typing import Any, Callable, Dict, Optional, List
from dataclasses import dataclass
from enum import Enum
import threading
import functools
from collections import deque

from .logging import get_logger
from .monitoring import record_metric
from .exceptions import CodesignError

logger = get_logger(__name__)


class CircuitState(Enum):
    """Circuit breaker states with health indicators."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, blocking requests
    HALF_OPEN = "half_open"  # Testing recovery
    DEGRADED = "degraded"  # Partial functionality


class HealthStatus(Enum):
    """Health status for system components."""
    HEALTHY = "healthy"
    DEGRADED = "degraded" 
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheck:
    """Health check configuration."""
    name: str
    check_function: Callable[[], bool]
    interval_seconds: float = 30.0
    timeout_seconds: float = 5.0
    required: bool = True


@dataclass 
class AdaptiveThresholds:
    """Adaptive thresholds based on system load and history."""
    base_failure_threshold: int = 5
    min_threshold: int = 2
    max_threshold: int = 20
    adaptation_window: int = 100
    load_factor: float = 1.0
    

class AdvancedCircuitBreaker:
    """Advanced circuit breaker with health checks and adaptive behavior."""
    
    def __init__(
        self, 
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        success_threshold: int = 3,
        request_timeout: float = 30.0,
        health_checks: Optional[List[HealthCheck]] = None,
        adaptive_thresholds: bool = False
    ):
        """Initialize advanced circuit breaker."""
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        self.request_timeout = request_timeout
        
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0.0
        self.last_success_time = time.time()
        
        self.total_requests = 0
        self.total_failures = 0
        self.request_history = deque(maxlen=1000)
        
        self.health_checks = health_checks or []
        self.health_status = HealthStatus.UNKNOWN
        self.last_health_check = 0.0
        
        self.adaptive_thresholds = AdaptiveThresholds() if adaptive_thresholds else None
        self._lock = threading.Lock()
        
        # Start health check thread
        if self.health_checks:
            self._start_health_monitoring()
        
        logger.info(f"Advanced circuit breaker {name} initialized")
        record_metric(f"circuit_breaker_{name}_init", 1, "counter")
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator interface."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return self.execute(func, *args, **kwargs)
        return wrapper
    
    def execute(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        with self._lock:
            self.total_requests += 1
            
            # Update adaptive thresholds
            if self.adaptive_thresholds:
                self._update_adaptive_thresholds()
            
            # Check if we should attempt reset
            if self._should_attempt_reset():
                self._attempt_reset()
            
            # Block if circuit is open
            if self.state == CircuitState.OPEN:
                record_metric(f"circuit_breaker_{self.name}_blocked", 1, "counter")
                raise CircuitBreakerOpenError(
                    f"Circuit breaker {self.name} is open. "
                    f"Health: {self.health_status.value}, "
                    f"Failures: {self.failure_count}/{self._get_current_threshold()}"
                )
        
        # Execute with timeout and monitoring
        start_time = time.time()
        try:
            if asyncio.iscoroutinefunction(func):
                result = asyncio.wait_for(func(*args, **kwargs), timeout=self.request_timeout)
            else:
                result = func(*args, **kwargs)
            
            execution_time = time.time() - start_time
            self._record_success(execution_time)
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self._record_failure(e, execution_time)
            raise
    
    def _get_current_threshold(self) -> int:
        """Get current failure threshold (adaptive or static)."""
        if self.adaptive_thresholds:
            return max(
                self.adaptive_thresholds.min_threshold,
                min(
                    self.adaptive_thresholds.max_threshold,
                    int(self.adaptive_thresholds.base_failure_threshold * 
                        self.adaptive_thresholds.load_factor)
                )
            )
        return self.failure_threshold
    
    def _update_adaptive_thresholds(self) -> None:
        """Update adaptive thresholds based on system state."""
        if not self.adaptive_thresholds or len(self.request_history) < 10:
            return
        
        # Calculate recent failure rate
        recent_requests = list(self.request_history)[-self.adaptive_thresholds.adaptation_window:]
        failure_rate = sum(1 for req in recent_requests if not req.get('success', False)) / len(recent_requests)
        
        # Calculate system load factor
        current_time = time.time()
        recent_load = len([req for req in recent_requests 
                          if current_time - req.get('timestamp', 0) < 60])  # Last minute
        
        # Adjust load factor (higher load = more tolerant of failures)
        self.adaptive_thresholds.load_factor = min(2.0, max(0.5, recent_load / 10.0))
        
        # Health-based adjustments
        if self.health_status == HealthStatus.DEGRADED:
            self.adaptive_thresholds.load_factor *= 0.8
        elif self.health_status == HealthStatus.UNHEALTHY:
            self.adaptive_thresholds.load_factor *= 0.6
    
    def _should_attempt_reset(self) -> bool:
        """Determine if circuit should attempt reset."""
        if self.state != CircuitState.OPEN:
            return False
        
        time_since_failure = time.time() - self.last_failure_time
        
        # Standard timeout check
        if time_since_failure < self.recovery_timeout:
            return False
        
        # Health-based reset delays
        if self.health_status == HealthStatus.UNHEALTHY:
            return time_since_failure >= self.recovery_timeout * 2
        elif self.health_status == HealthStatus.DEGRADED:
            return time_since_failure >= self.recovery_timeout * 1.5
        
        return True
    
    def _attempt_reset(self) -> None:
        """Attempt to reset circuit to half-open."""
        self.state = CircuitState.HALF_OPEN
        self.success_count = 0
        
        logger.info(f"Circuit breaker {self.name} attempting reset (half-open)")
        record_metric(f"circuit_breaker_{self.name}_half_open", 1, "counter")
    
    def _record_success(self, execution_time: float) -> None:
        """Record successful execution."""
        with self._lock:
            self.success_count += 1
            self.last_success_time = time.time()
            
            # Record in history
            self.request_history.append({
                'timestamp': self.last_success_time,
                'success': True,
                'execution_time': execution_time
            })
            
            # State transitions
            if self.state == CircuitState.HALF_OPEN:
                if self.success_count >= self.success_threshold:
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0
                    self.success_count = 0
                    
                    logger.info(f"Circuit breaker {self.name} closed (recovered)")
                    record_metric(f"circuit_breaker_{self.name}_closed", 1, "counter")
            
            # Reset failure count on success in closed state
            elif self.state == CircuitState.CLOSED:
                self.failure_count = max(0, self.failure_count - 1)
            
            record_metric(f"circuit_breaker_{self.name}_success", 1, "counter")
            record_metric(f"circuit_breaker_{self.name}_execution_time", execution_time, "histogram")
    
    def _record_failure(self, exception: Exception, execution_time: float) -> None:
        """Record failed execution."""
        with self._lock:
            self.failure_count += 1
            self.total_failures += 1
            self.last_failure_time = time.time()
            
            # Record in history
            self.request_history.append({
                'timestamp': self.last_failure_time,
                'success': False,
                'execution_time': execution_time,
                'exception': str(exception)
            })
            
            current_threshold = self._get_current_threshold()
            
            # Open circuit if threshold exceeded
            if (self.state in [CircuitState.CLOSED, CircuitState.HALF_OPEN] and
                self.failure_count >= current_threshold):
                
                self.state = CircuitState.OPEN
                
                logger.warning(
                    f"Circuit breaker {self.name} opened: "
                    f"{self.failure_count} failures >= {current_threshold} threshold"
                )
                record_metric(f"circuit_breaker_{self.name}_opened", 1, "counter")
            
            record_metric(f"circuit_breaker_{self.name}_failure", 1, "counter")
    
    def _start_health_monitoring(self) -> None:
        """Start health check monitoring thread."""
        def health_monitor():
            while True:
                try:
                    self._perform_health_checks()
                    time.sleep(min(hc.interval_seconds for hc in self.health_checks))
                except Exception as e:
                    logger.error(f"Health monitoring error for {self.name}: {e}")
                    time.sleep(30)  # Fallback interval
        
        thread = threading.Thread(target=health_monitor, daemon=True)
        thread.start()
    
    def _perform_health_checks(self) -> None:
        """Perform all configured health checks."""
        if not self.health_checks:
            return
        
        current_time = time.time()
        if current_time - self.last_health_check < 10:  # Min 10s between checks
            return
        
        self.last_health_check = current_time
        health_results = []
        
        for health_check in self.health_checks:
            try:
                result = health_check.check_function()
                health_results.append((health_check.name, result, health_check.required))
                
                record_metric(
                    f"health_check_{self.name}_{health_check.name}",
                    1 if result else 0,
                    "gauge"
                )
                
            except Exception as e:
                logger.error(f"Health check {health_check.name} failed: {e}")
                health_results.append((health_check.name, False, health_check.required))
        
        # Determine overall health status
        required_checks = [r for r in health_results if r[2]]  # required checks
        optional_checks = [r for r in health_results if not r[2]]
        
        required_passing = all(r[1] for r in required_checks) if required_checks else True
        optional_passing_rate = (sum(1 for r in optional_checks if r[1]) / 
                               len(optional_checks)) if optional_checks else 1.0
        
        if required_passing and optional_passing_rate >= 0.8:
            self.health_status = HealthStatus.HEALTHY
        elif required_passing and optional_passing_rate >= 0.5:
            self.health_status = HealthStatus.DEGRADED
        elif required_passing:
            self.health_status = HealthStatus.DEGRADED
        else:
            self.health_status = HealthStatus.UNHEALTHY
        
        record_metric(f"circuit_breaker_{self.name}_health_status", 
                     {"healthy": 1, "degraded": 0.5, "unhealthy": 0}[self.health_status.value],
                     "gauge")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive circuit breaker metrics."""
        current_time = time.time()
        
        # Calculate recent metrics
        recent_requests = [req for req in self.request_history 
                          if current_time - req.get('timestamp', 0) < 300]  # Last 5 minutes
        
        recent_failure_rate = 0.0
        avg_execution_time = 0.0
        
        if recent_requests:
            recent_failures = sum(1 for req in recent_requests if not req.get('success', False))
            recent_failure_rate = recent_failures / len(recent_requests)
            
            execution_times = [req.get('execution_time', 0) for req in recent_requests]
            avg_execution_time = sum(execution_times) / len(execution_times)
        
        return {
            "name": self.name,
            "state": self.state.value,
            "health_status": self.health_status.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "total_requests": self.total_requests,
            "total_failures": self.total_failures,
            "current_threshold": self._get_current_threshold(),
            "failure_rate": self.total_failures / max(1, self.total_requests),
            "recent_failure_rate": recent_failure_rate,
            "avg_execution_time": avg_execution_time,
            "last_failure_time": self.last_failure_time,
            "last_success_time": self.last_success_time,
            "time_since_last_failure": current_time - self.last_failure_time if self.last_failure_time > 0 else -1,
            "adaptive_enabled": self.adaptive_thresholds is not None,
            "load_factor": self.adaptive_thresholds.load_factor if self.adaptive_thresholds else 1.0
        }
    
    def reset(self) -> None:
        """Reset circuit breaker state."""
        with self._lock:
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.success_count = 0
            
        logger.info(f"Circuit breaker {self.name} manually reset")
        record_metric(f"circuit_breaker_{self.name}_reset", 1, "counter")


class CircuitBreakerOpenError(CodesignError):
    """Exception raised when circuit breaker is open."""
    pass


# Global circuit breaker registry
_circuit_breakers: Dict[str, AdvancedCircuitBreaker] = {}
_lock = threading.Lock()


def get_circuit_breaker(
    name: str, 
    **kwargs
) -> AdvancedCircuitBreaker:
    """Get or create circuit breaker."""
    with _lock:
        if name not in _circuit_breakers:
            _circuit_breakers[name] = AdvancedCircuitBreaker(name, **kwargs)
        return _circuit_breakers[name]


def circuit_breaker(name: str, **kwargs) -> Callable:
    """Decorator for circuit breaker protection."""
    def decorator(func: Callable) -> Callable:
        cb = get_circuit_breaker(name, **kwargs)
        return cb(func)
    return decorator


def get_all_circuit_breaker_metrics() -> Dict[str, Any]:
    """Get metrics for all circuit breakers."""
    return {name: cb.get_metrics() for name, cb in _circuit_breakers.items()}


def reset_all_circuit_breakers() -> None:
    """Reset all circuit breakers."""
    with _lock:
        for cb in _circuit_breakers.values():
            cb.reset()
    logger.info("Reset all circuit breakers")