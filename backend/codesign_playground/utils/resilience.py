"""
Resilience patterns for AI Hardware Co-Design Playground.

This module provides circuit breakers, retry mechanisms, and graceful degradation
patterns to ensure system reliability and fault tolerance.
"""

import time
import asyncio
import random
from typing import Any, Callable, Dict, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import threading
import functools
import logging

from .logging import get_logger
from .monitoring import record_metric

logger = get_logger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, blocking requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5       # Failures before opening
    recovery_timeout: float = 60.0   # Seconds before trying half-open
    success_threshold: int = 3       # Successes needed to close from half-open
    timeout: float = 30.0            # Request timeout
    expected_exceptions: tuple = (Exception,)


@dataclass
class CircuitBreakerStats:
    """Circuit breaker statistics."""
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: float = 0.0
    total_requests: int = 0
    total_failures: int = 0
    total_successes: int = 0
    state_changes: int = 0


class CircuitBreaker:
    """Circuit breaker implementation for fault tolerance."""
    
    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        """
        Initialize circuit breaker.
        
        Args:
            name: Circuit breaker identifier
            config: Configuration parameters
        """
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.stats = CircuitBreakerStats()
        self._lock = threading.Lock()
        
        logger.info(f"Initialized circuit breaker: {name}")
        record_metric(f"circuit_breaker_{name}_initialized", 1, "counter")
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator to wrap function with circuit breaker."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return self.call(func, *args, **kwargs)
        return wrapper
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection.
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            CircuitBreakerOpenError: If circuit is open
            Various: Underlying function exceptions
        """
        with self._lock:
            self.stats.total_requests += 1
            
            # Check if circuit should be opened
            if self._should_attempt_reset():
                self._attempt_reset()
            
            # Block requests if circuit is open
            if self.stats.state == CircuitState.OPEN:
                record_metric(f"circuit_breaker_{self.name}_blocked", 1, "counter")
                raise CircuitBreakerOpenError(f"Circuit breaker {self.name} is open")
        
        # Execute the function
        start_time = time.time()
        try:
            # Apply timeout
            if asyncio.iscoroutinefunction(func):
                result = asyncio.wait_for(func(*args, **kwargs), timeout=self.config.timeout)
            else:
                result = func(*args, **kwargs)
            
            # Record success
            self._record_success()
            return result
            
        except self.config.expected_exceptions as e:
            # Record failure
            self._record_failure()
            raise
        except Exception as e:
            # Unexpected exception - still record as failure
            self._record_failure()
            logger.error(f"Unexpected exception in circuit breaker {self.name}", exception=e)
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt to reset."""
        return (
            self.stats.state == CircuitState.OPEN and
            time.time() - self.stats.last_failure_time >= self.config.recovery_timeout
        )
    
    def _attempt_reset(self) -> None:
        """Attempt to reset circuit breaker to half-open state."""
        self.stats.state = CircuitState.HALF_OPEN
        self.stats.success_count = 0
        self.stats.state_changes += 1
        
        logger.info(f"Circuit breaker {self.name} attempting reset (half-open)")
        record_metric(f"circuit_breaker_{self.name}_half_open", 1, "counter")
    
    def _record_success(self) -> None:
        """Record successful operation."""
        with self._lock:
            self.stats.success_count += 1
            self.stats.total_successes += 1
            
            if self.stats.state == CircuitState.HALF_OPEN:
                if self.stats.success_count >= self.config.success_threshold:
                    # Close the circuit
                    self.stats.state = CircuitState.CLOSED
                    self.stats.failure_count = 0
                    self.stats.success_count = 0
                    self.stats.state_changes += 1
                    
                    logger.info(f"Circuit breaker {self.name} closed (recovered)")
                    record_metric(f"circuit_breaker_{self.name}_closed", 1, "counter")
            
            record_metric(f"circuit_breaker_{self.name}_success", 1, "counter")
    
    def _record_failure(self) -> None:
        """Record failed operation."""
        with self._lock:
            self.stats.failure_count += 1
            self.stats.total_failures += 1
            self.stats.last_failure_time = time.time()
            
            # Open circuit if failure threshold exceeded
            if (self.stats.state in [CircuitState.CLOSED, CircuitState.HALF_OPEN] and
                self.stats.failure_count >= self.config.failure_threshold):
                
                self.stats.state = CircuitState.OPEN
                self.stats.state_changes += 1
                
                logger.warning(f"Circuit breaker {self.name} opened due to failures")
                record_metric(f"circuit_breaker_{self.name}_opened", 1, "counter")
            
            record_metric(f"circuit_breaker_{self.name}_failure", 1, "counter")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        return {
            "name": self.name,
            "state": self.stats.state.value,
            "failure_count": self.stats.failure_count,
            "success_count": self.stats.success_count,
            "total_requests": self.stats.total_requests,
            "total_failures": self.stats.total_failures,
            "total_successes": self.stats.total_successes,
            "failure_rate": self.stats.total_failures / max(1, self.stats.total_requests),
            "state_changes": self.stats.state_changes,
            "last_failure_time": self.stats.last_failure_time
        }
    
    def reset(self) -> None:
        """Manually reset circuit breaker."""
        with self._lock:
            self.stats.state = CircuitState.CLOSED
            self.stats.failure_count = 0
            self.stats.success_count = 0
            
        logger.info(f"Circuit breaker {self.name} manually reset")
        record_metric(f"circuit_breaker_{self.name}_manual_reset", 1, "counter")


class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open."""
    pass


@dataclass
class RetryConfig:
    """Configuration for retry mechanism."""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_backoff: bool = True
    jitter: bool = True
    retriable_exceptions: tuple = (Exception,)


class RetryMechanism:
    """Retry mechanism with exponential backoff and jitter."""
    
    def __init__(self, name: str, config: Optional[RetryConfig] = None):
        """
        Initialize retry mechanism.
        
        Args:
            name: Retry mechanism identifier
            config: Configuration parameters
        """
        self.name = name
        self.config = config or RetryConfig()
        
        logger.info(f"Initialized retry mechanism: {name}")
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator to wrap function with retry mechanism."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return self.execute(func, *args, **kwargs)
        return wrapper
    
    def execute(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with retry mechanism.
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            Various: Last exception if all retries fail
        """
        last_exception = None
        
        for attempt in range(1, self.config.max_attempts + 1):
            try:
                record_metric(f"retry_{self.name}_attempt", 1, "counter", {"attempt": str(attempt)})
                result = func(*args, **kwargs)
                
                if attempt > 1:
                    logger.info(f"Retry {self.name} succeeded on attempt {attempt}")
                    record_metric(f"retry_{self.name}_success", 1, "counter", {"attempt": str(attempt)})
                
                return result
                
            except self.config.retriable_exceptions as e:
                last_exception = e
                
                if attempt == self.config.max_attempts:
                    logger.error(f"Retry {self.name} exhausted all {self.config.max_attempts} attempts")
                    record_metric(f"retry_{self.name}_exhausted", 1, "counter")
                    break
                
                # Calculate delay
                delay = self._calculate_delay(attempt)
                
                logger.warning(
                    f"Retry {self.name} attempt {attempt} failed, retrying in {delay:.2f}s: {e}"
                )
                record_metric(f"retry_{self.name}_failed", 1, "counter", {"attempt": str(attempt)})
                
                time.sleep(delay)
            
            except Exception as e:
                # Non-retriable exception
                logger.error(f"Retry {self.name} non-retriable exception: {e}")
                record_metric(f"retry_{self.name}_non_retriable", 1, "counter")
                raise
        
        # Re-raise the last exception
        if last_exception:
            raise last_exception
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate retry delay with exponential backoff and jitter."""
        if self.config.exponential_backoff:
            delay = self.config.base_delay * (2 ** (attempt - 1))
        else:
            delay = self.config.base_delay
        
        # Apply maximum delay
        delay = min(delay, self.config.max_delay)
        
        # Add jitter to prevent thundering herd
        if self.config.jitter:
            jitter_amount = delay * 0.1  # 10% jitter
            delay += random.uniform(-jitter_amount, jitter_amount)
        
        return max(0.1, delay)  # Minimum delay


class GracefulDegradation:
    """Graceful degradation mechanism for system resilience."""
    
    def __init__(self, name: str):
        """
        Initialize graceful degradation.
        
        Args:
            name: Degradation mechanism identifier
        """
        self.name = name
        self.fallback_functions: Dict[str, Callable] = {}
        
        logger.info(f"Initialized graceful degradation: {name}")
    
    def register_fallback(self, operation: str, fallback_func: Callable) -> None:
        """
        Register fallback function for operation.
        
        Args:
            operation: Operation name
            fallback_func: Fallback function
        """
        self.fallback_functions[operation] = fallback_func
        logger.info(f"Registered fallback for {operation} in {self.name}")
    
    def execute_with_fallback(
        self, 
        operation: str, 
        primary_func: Callable,
        *args, 
        **kwargs
    ) -> Any:
        """
        Execute operation with fallback.
        
        Args:
            operation: Operation name
            primary_func: Primary function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Primary function result or fallback result
        """
        try:
            result = primary_func(*args, **kwargs)
            record_metric(f"degradation_{self.name}_{operation}_primary", 1, "counter")
            return result
            
        except Exception as e:
            logger.warning(f"Primary function failed for {operation}, using fallback: {e}")
            record_metric(f"degradation_{self.name}_{operation}_fallback", 1, "counter")
            
            if operation in self.fallback_functions:
                try:
                    fallback_result = self.fallback_functions[operation](*args, **kwargs)
                    record_metric(f"degradation_{self.name}_{operation}_fallback_success", 1, "counter")
                    return fallback_result
                except Exception as fallback_error:
                    logger.error(f"Fallback also failed for {operation}: {fallback_error}")
                    record_metric(f"degradation_{self.name}_{operation}_fallback_failed", 1, "counter")
                    raise
            else:
                logger.error(f"No fallback registered for {operation}")
                record_metric(f"degradation_{self.name}_{operation}_no_fallback", 1, "counter")
                raise


# Global circuit breakers for common operations
_circuit_breakers: Dict[str, CircuitBreaker] = {}


def get_circuit_breaker(name: str, config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
    """Get or create circuit breaker."""
    if name not in _circuit_breakers:
        _circuit_breakers[name] = CircuitBreaker(name, config)
    return _circuit_breakers[name]


def circuit_breaker(
    name: str, 
    config: Optional[CircuitBreakerConfig] = None
) -> Callable:
    """Decorator for circuit breaker protection."""
    def decorator(func: Callable) -> Callable:
        cb = get_circuit_breaker(name, config)
        return cb(func)
    return decorator


def retry(
    name: str, 
    config: Optional[RetryConfig] = None
) -> Callable:
    """Decorator for retry mechanism."""
    def decorator(func: Callable) -> Callable:
        retry_mech = RetryMechanism(name, config)
        return retry_mech(func)
    return decorator


def resilient(
    circuit_breaker_name: str,
    retry_name: str,
    cb_config: Optional[CircuitBreakerConfig] = None,
    retry_config: Optional[RetryConfig] = None
) -> Callable:
    """Combined circuit breaker and retry decorator."""
    def decorator(func: Callable) -> Callable:
        # Apply retry first, then circuit breaker
        retry_decorated = retry(retry_name, retry_config)(func)
        circuit_decorated = circuit_breaker(circuit_breaker_name, cb_config)(retry_decorated)
        return circuit_decorated
    return decorator


def get_all_circuit_breaker_stats() -> Dict[str, Any]:
    """Get statistics for all circuit breakers."""
    return {
        name: cb.get_stats() 
        for name, cb in _circuit_breakers.items()
    }


def reset_all_circuit_breakers() -> None:
    """Reset all circuit breakers."""
    for cb in _circuit_breakers.values():
        cb.reset()
    logger.info("Reset all circuit breakers")