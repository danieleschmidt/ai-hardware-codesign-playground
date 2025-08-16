"""
Enhanced resilience patterns and circuit breaker integration for AI Hardware Co-Design Playground.

This module provides advanced resilience patterns including bulkhead isolation,
adaptive timeout management, and sophisticated failure detection.
"""

import time
import asyncio
import threading
from typing import Dict, List, Any, Optional, Callable, Union, Type
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
from functools import wraps
import random
import statistics

from .circuit_breaker import AdvancedCircuitBreaker, CircuitState, HealthStatus
from .resilience import RetryMechanism, RetryConfig, GracefulDegradation
from .logging import get_logger, get_performance_logger
from .monitoring import record_metric
from .exceptions import CodesignError
from .distributed_tracing import trace_span, SpanType

logger = get_logger(__name__)
performance_logger = get_performance_logger(__name__)


class FailureMode(Enum):
    """Types of failure modes for resilience patterns."""
    TIMEOUT = "timeout"
    CIRCUIT_OPEN = "circuit_open"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    DEPENDENCY_FAILURE = "dependency_failure"
    RATE_LIMITED = "rate_limited"
    VALIDATION_ERROR = "validation_error"
    UNKNOWN = "unknown"


class ResilienceLevel(Enum):
    """Resilience configuration levels."""
    MINIMAL = "minimal"
    STANDARD = "standard"
    AGGRESSIVE = "aggressive"
    MAXIMUM = "maximum"


@dataclass
class BulkheadConfig:
    """Configuration for bulkhead pattern."""
    name: str
    max_concurrent: int = 10
    max_queue_size: int = 100
    timeout_seconds: float = 30.0
    reject_on_full: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "max_concurrent": self.max_concurrent,
            "max_queue_size": self.max_queue_size,
            "timeout_seconds": self.timeout_seconds,
            "reject_on_full": self.reject_on_full
        }


@dataclass
class ResilienceMetrics:
    """Metrics for resilience patterns."""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    timeout_calls: int = 0
    circuit_breaker_calls: int = 0
    retry_calls: int = 0
    fallback_calls: int = 0
    bulkhead_rejections: int = 0
    avg_response_time: float = 0.0
    success_rate: float = 0.0
    
    def update_success(self, response_time: float) -> None:
        """Update metrics for successful call."""
        self.total_calls += 1
        self.successful_calls += 1
        self._update_response_time(response_time)
    
    def update_failure(self, failure_mode: FailureMode, response_time: float = 0.0) -> None:
        """Update metrics for failed call."""
        self.total_calls += 1
        self.failed_calls += 1
        
        if failure_mode == FailureMode.TIMEOUT:
            self.timeout_calls += 1
        elif failure_mode == FailureMode.CIRCUIT_OPEN:
            self.circuit_breaker_calls += 1
        
        if response_time > 0:
            self._update_response_time(response_time)
    
    def _update_response_time(self, response_time: float) -> None:
        """Update average response time."""
        if self.total_calls == 1:
            self.avg_response_time = response_time
        else:
            # Exponential moving average
            alpha = 0.1
            self.avg_response_time = alpha * response_time + (1 - alpha) * self.avg_response_time
        
        # Update success rate
        self.success_rate = self.successful_calls / self.total_calls if self.total_calls > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "failed_calls": self.failed_calls,
            "timeout_calls": self.timeout_calls,
            "circuit_breaker_calls": self.circuit_breaker_calls,
            "retry_calls": self.retry_calls,
            "fallback_calls": self.fallback_calls,
            "bulkhead_rejections": self.bulkhead_rejections,
            "avg_response_time": self.avg_response_time,
            "success_rate": self.success_rate
        }


class BulkheadIsolation:
    """Bulkhead pattern for resource isolation."""
    
    def __init__(self, config: BulkheadConfig):
        """Initialize bulkhead isolation."""
        self.config = config
        self.semaphore = threading.Semaphore(config.max_concurrent)
        self.queue = deque(maxlen=config.max_queue_size)
        self.active_requests = 0
        self.total_requests = 0
        self.rejected_requests = 0
        self._lock = threading.Lock()
        
        logger.info("Initialized BulkheadIsolation", 
                   name=config.name, 
                   max_concurrent=config.max_concurrent)
    
    def execute(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with bulkhead isolation."""
        with self._lock:
            self.total_requests += 1
        
        # Try to acquire semaphore
        acquired = self.semaphore.acquire(blocking=False)
        
        if not acquired:
            if self.config.reject_on_full:
                with self._lock:
                    self.rejected_requests += 1
                
                record_metric(f"bulkhead_{self.config.name}_rejected", 1, "counter")
                
                raise CodesignError(
                    f"Bulkhead {self.config.name} at capacity",
                    "BULKHEAD_CAPACITY_EXCEEDED",
                    {"max_concurrent": self.config.max_concurrent}
                )
            else:
                # Wait with timeout
                acquired = self.semaphore.acquire(timeout=self.config.timeout_seconds)
                if not acquired:
                    with self._lock:
                        self.rejected_requests += 1
                    
                    raise CodesignError(
                        f"Bulkhead {self.config.name} timeout",
                        "BULKHEAD_TIMEOUT"
                    )
        
        try:
            with self._lock:
                self.active_requests += 1
            
            # Execute function
            start_time = time.time()
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            record_metric(f"bulkhead_{self.config.name}_success", 1, "counter")
            record_metric(f"bulkhead_{self.config.name}_duration", execution_time, "histogram")
            
            return result
            
        finally:
            with self._lock:
                self.active_requests -= 1
            self.semaphore.release()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get bulkhead statistics."""
        with self._lock:
            return {
                "name": self.config.name,
                "max_concurrent": self.config.max_concurrent,
                "active_requests": self.active_requests,
                "total_requests": self.total_requests,
                "rejected_requests": self.rejected_requests,
                "rejection_rate": self.rejected_requests / max(1, self.total_requests),
                "available_capacity": self.config.max_concurrent - self.active_requests
            }


class AdaptiveTimeoutManager:
    """Adaptive timeout management based on response time patterns."""
    
    def __init__(self, initial_timeout: float = 30.0, 
                 min_timeout: float = 1.0, 
                 max_timeout: float = 300.0,
                 percentile: float = 95.0,
                 adaptation_factor: float = 1.5):
        """Initialize adaptive timeout manager."""
        self.initial_timeout = initial_timeout
        self.min_timeout = min_timeout
        self.max_timeout = max_timeout
        self.percentile = percentile
        self.adaptation_factor = adaptation_factor
        
        self.current_timeout = initial_timeout
        self.response_times = deque(maxlen=1000)  # Keep last 1000 response times
        self.last_adaptation = time.time()
        self.adaptation_interval = 60.0  # Adapt every minute
        
        self._lock = threading.Lock()
        
        logger.info("Initialized AdaptiveTimeoutManager",
                   initial_timeout=initial_timeout,
                   adaptation_factor=adaptation_factor)
    
    def get_timeout(self) -> float:
        """Get current adaptive timeout."""
        current_time = time.time()
        
        # Adapt timeout if interval has passed
        if current_time - self.last_adaptation >= self.adaptation_interval:
            self._adapt_timeout()
            self.last_adaptation = current_time
        
        return self.current_timeout
    
    def record_response_time(self, response_time: float) -> None:
        """Record response time for timeout adaptation."""
        with self._lock:
            self.response_times.append(response_time)
    
    def record_timeout(self) -> None:
        """Record timeout occurrence."""
        with self._lock:
            # Record current timeout as a data point
            self.response_times.append(self.current_timeout)
        
        record_metric("adaptive_timeout_occurred", 1, "counter")
    
    def _adapt_timeout(self) -> None:
        """Adapt timeout based on recent response times."""
        with self._lock:
            if len(self.response_times) < 10:
                return  # Not enough data
            
            # Calculate percentile-based timeout
            response_times_list = list(self.response_times)
            percentile_time = self._calculate_percentile(response_times_list, self.percentile)
            
            # Apply adaptation factor
            new_timeout = percentile_time * self.adaptation_factor
            
            # Apply bounds
            new_timeout = max(self.min_timeout, min(self.max_timeout, new_timeout))
            
            # Smooth adaptation to avoid drastic changes
            alpha = 0.3  # Smoothing factor
            self.current_timeout = alpha * new_timeout + (1 - alpha) * self.current_timeout
            
            logger.debug("Adapted timeout",
                        old_timeout=self.current_timeout,
                        new_timeout=new_timeout,
                        percentile_time=percentile_time,
                        sample_size=len(response_times_list))
            
            record_metric("adaptive_timeout_value", self.current_timeout, "gauge")
    
    def _calculate_percentile(self, data: List[float], percentile: float) -> float:
        """Calculate percentile of data."""
        if not data:
            return self.initial_timeout
        
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100)
        index = min(index, len(sorted_data) - 1)
        
        return sorted_data[index]


class EnhancedResilienceManager:
    """Comprehensive resilience management with multiple patterns."""
    
    def __init__(self, level: ResilienceLevel = ResilienceLevel.STANDARD):
        """Initialize enhanced resilience manager."""
        self.level = level
        self.circuit_breakers: Dict[str, AdvancedCircuitBreaker] = {}
        self.bulkheads: Dict[str, BulkheadIsolation] = {}
        self.timeout_managers: Dict[str, AdaptiveTimeoutManager] = {}
        self.retry_mechanisms: Dict[str, RetryMechanism] = {}
        self.fallback_handlers: Dict[str, GracefulDegradation] = {}
        self.metrics: Dict[str, ResilienceMetrics] = defaultdict(ResilienceMetrics)
        
        self._lock = threading.Lock()
        
        # Setup default configurations based on level
        self._setup_default_configs()
        
        logger.info("Initialized EnhancedResilienceManager", level=level.value)
    
    def _setup_default_configs(self) -> None:
        """Setup default resilience configurations."""
        configs = {
            ResilienceLevel.MINIMAL: {
                "circuit_breaker": {"failure_threshold": 10, "recovery_timeout": 60.0},
                "retry": {"max_attempts": 2, "base_delay": 1.0},
                "timeout": {"initial": 30.0, "max": 60.0},
                "bulkhead": {"max_concurrent": 20}
            },
            ResilienceLevel.STANDARD: {
                "circuit_breaker": {"failure_threshold": 5, "recovery_timeout": 60.0},
                "retry": {"max_attempts": 3, "base_delay": 1.0},
                "timeout": {"initial": 30.0, "max": 120.0},
                "bulkhead": {"max_concurrent": 10}
            },
            ResilienceLevel.AGGRESSIVE: {
                "circuit_breaker": {"failure_threshold": 3, "recovery_timeout": 30.0},
                "retry": {"max_attempts": 5, "base_delay": 0.5},
                "timeout": {"initial": 15.0, "max": 60.0},
                "bulkhead": {"max_concurrent": 5}
            },
            ResilienceLevel.MAXIMUM: {
                "circuit_breaker": {"failure_threshold": 2, "recovery_timeout": 15.0},
                "retry": {"max_attempts": 7, "base_delay": 0.1},
                "timeout": {"initial": 10.0, "max": 30.0},
                "bulkhead": {"max_concurrent": 3}
            }
        }
        
        self.default_config = configs[self.level]
    
    def add_circuit_breaker(self, name: str, **kwargs) -> AdvancedCircuitBreaker:
        """Add circuit breaker with default configuration."""
        config = {**self.default_config["circuit_breaker"], **kwargs}
        
        with self._lock:
            if name not in self.circuit_breakers:
                self.circuit_breakers[name] = AdvancedCircuitBreaker(name, **config)
        
        return self.circuit_breakers[name]
    
    def add_bulkhead(self, name: str, **kwargs) -> BulkheadIsolation:
        """Add bulkhead isolation."""
        config_params = {**self.default_config["bulkhead"], **kwargs}
        config = BulkheadConfig(name=name, **config_params)
        
        with self._lock:
            if name not in self.bulkheads:
                self.bulkheads[name] = BulkheadIsolation(config)
        
        return self.bulkheads[name]
    
    def add_timeout_manager(self, name: str, **kwargs) -> AdaptiveTimeoutManager:
        """Add adaptive timeout manager."""
        config = {**self.default_config["timeout"], **kwargs}
        
        with self._lock:
            if name not in self.timeout_managers:
                self.timeout_managers[name] = AdaptiveTimeoutManager(
                    initial_timeout=config["initial"],
                    max_timeout=config["max"],
                    **{k: v for k, v in kwargs.items() if k not in ["initial", "max"]}
                )
        
        return self.timeout_managers[name]
    
    def add_retry_mechanism(self, name: str, **kwargs) -> RetryMechanism:
        """Add retry mechanism."""
        config = {**self.default_config["retry"], **kwargs}
        retry_config = RetryConfig(**config)
        
        with self._lock:
            if name not in self.retry_mechanisms:
                self.retry_mechanisms[name] = RetryMechanism(name, retry_config)
        
        return self.retry_mechanisms[name]
    
    def add_fallback_handler(self, name: str) -> GracefulDegradation:
        """Add fallback handler."""
        with self._lock:
            if name not in self.fallback_handlers:
                self.fallback_handlers[name] = GracefulDegradation(name)
        
        return self.fallback_handlers[name]
    
    def execute_with_resilience(self, name: str, func: Callable, 
                               *args, **kwargs) -> Any:
        """Execute function with all configured resilience patterns."""
        start_time = time.time()
        
        try:
            with trace_span(f"resilience_{name}", SpanType.FUNCTION_CALL) as span:
                span.set_tag("resilience.name", name)
                span.set_tag("resilience.level", self.level.value)
                
                # Apply patterns in order: bulkhead -> circuit breaker -> timeout -> retry
                result = self._execute_with_patterns(name, func, *args, **kwargs)
                
                execution_time = time.time() - start_time
                self.metrics[name].update_success(execution_time)
                
                span.set_tag("resilience.success", True)
                span.set_tag("resilience.execution_time", execution_time)
                
                return result
                
        except Exception as e:
            execution_time = time.time() - start_time
            failure_mode = self._classify_failure(e)
            self.metrics[name].update_failure(failure_mode, execution_time)
            
            logger.error(f"Resilience execution failed for {name}",
                        error=str(e),
                        failure_mode=failure_mode.value,
                        execution_time=execution_time)
            
            record_metric(f"resilience_{name}_failure", 1, "counter",
                         {"failure_mode": failure_mode.value})
            
            raise
    
    def _execute_with_patterns(self, name: str, func: Callable, *args, **kwargs) -> Any:
        """Execute function applying resilience patterns."""
        # 1. Bulkhead isolation
        if name in self.bulkheads:
            bulkhead = self.bulkheads[name]
            func = lambda: bulkhead.execute(func, *args, **kwargs)
            args, kwargs = (), {}
        
        # 2. Circuit breaker
        if name in self.circuit_breakers:
            circuit_breaker = self.circuit_breakers[name]
            func = circuit_breaker(func)
        
        # 3. Adaptive timeout
        if name in self.timeout_managers:
            timeout_manager = self.timeout_managers[name]
            timeout = timeout_manager.get_timeout()
            
            def timeout_wrapper():
                try:
                    start = time.time()
                    if asyncio.iscoroutinefunction(func):
                        result = asyncio.wait_for(func(*args, **kwargs), timeout=timeout)
                    else:
                        # Simple timeout for sync functions
                        result = func(*args, **kwargs)
                    
                    response_time = time.time() - start
                    timeout_manager.record_response_time(response_time)
                    return result
                    
                except asyncio.TimeoutError:
                    timeout_manager.record_timeout()
                    raise CodesignError("Operation timed out", "TIMEOUT_ERROR",
                                      {"timeout": timeout})
            
            func = timeout_wrapper
            args, kwargs = (), {}
        
        # 4. Retry mechanism
        if name in self.retry_mechanisms:
            retry_mechanism = self.retry_mechanisms[name]
            func = retry_mechanism(func)
        
        # 5. Fallback handling
        if name in self.fallback_handlers:
            fallback_handler = self.fallback_handlers[name]
            original_func = func
            
            def fallback_wrapper():
                return fallback_handler.execute_with_fallback(
                    name, original_func, *args, **kwargs
                )
            
            func = fallback_wrapper
            args, kwargs = (), {}
        
        # Execute final function
        return func(*args, **kwargs)
    
    def _classify_failure(self, exception: Exception) -> FailureMode:
        """Classify failure mode based on exception."""
        exception_str = str(exception).lower()
        
        if "timeout" in exception_str or "timed out" in exception_str:
            return FailureMode.TIMEOUT
        elif "circuit" in exception_str and "open" in exception_str:
            return FailureMode.CIRCUIT_OPEN
        elif "rate limit" in exception_str:
            return FailureMode.RATE_LIMITED
        elif "bulkhead" in exception_str or "capacity" in exception_str:
            return FailureMode.RESOURCE_EXHAUSTION
        elif "validation" in exception_str:
            return FailureMode.VALIDATION_ERROR
        elif "dependency" in exception_str or "service" in exception_str:
            return FailureMode.DEPENDENCY_FAILURE
        else:
            return FailureMode.UNKNOWN
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive resilience statistics."""
        stats = {
            "level": self.level.value,
            "circuit_breakers": {},
            "bulkheads": {},
            "timeout_managers": {},
            "metrics": {}
        }
        
        with self._lock:
            # Circuit breaker stats
            for name, cb in self.circuit_breakers.items():
                stats["circuit_breakers"][name] = cb.get_metrics()
            
            # Bulkhead stats
            for name, bulkhead in self.bulkheads.items():
                stats["bulkheads"][name] = bulkhead.get_stats()
            
            # Timeout manager stats
            for name, tm in self.timeout_managers.items():
                stats["timeout_managers"][name] = {
                    "current_timeout": tm.current_timeout,
                    "response_time_samples": len(tm.response_times),
                    "min_timeout": tm.min_timeout,
                    "max_timeout": tm.max_timeout
                }
            
            # Resilience metrics
            for name, metrics in self.metrics.items():
                stats["metrics"][name] = metrics.to_dict()
        
        return stats


# Global resilience manager
_resilience_manager: Optional[EnhancedResilienceManager] = None
_resilience_lock = threading.Lock()


def get_resilience_manager(level: ResilienceLevel = ResilienceLevel.STANDARD) -> EnhancedResilienceManager:
    """Get global resilience manager."""
    global _resilience_manager
    
    with _resilience_lock:
        if _resilience_manager is None:
            _resilience_manager = EnhancedResilienceManager(level)
        
        return _resilience_manager


def resilient(name: str, 
             level: Optional[ResilienceLevel] = None,
             circuit_breaker: bool = True,
             bulkhead: bool = False,
             adaptive_timeout: bool = True,
             retry: bool = True,
             fallback: bool = False,
             **kwargs):
    """Decorator for comprehensive resilience patterns."""
    def decorator(func: Callable) -> Callable:
        manager = get_resilience_manager(level or ResilienceLevel.STANDARD)
        
        # Setup patterns
        if circuit_breaker:
            manager.add_circuit_breaker(name, **kwargs.get("circuit_breaker", {}))
        
        if bulkhead:
            manager.add_bulkhead(name, **kwargs.get("bulkhead", {}))
        
        if adaptive_timeout:
            manager.add_timeout_manager(name, **kwargs.get("timeout", {}))
        
        if retry:
            manager.add_retry_mechanism(name, **kwargs.get("retry", {}))
        
        if fallback:
            manager.add_fallback_handler(name)
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            return manager.execute_with_resilience(name, func, *args, **kwargs)
        
        return wrapper
    
    return decorator