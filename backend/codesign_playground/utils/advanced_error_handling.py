"""
Advanced Error Handling and Recovery System.

This module provides comprehensive error handling, recovery mechanisms,
and fault tolerance for the hardware co-design platform.
"""

import asyncio
import traceback
import time
import functools
from typing import Dict, List, Any, Optional, Callable, Type, Union
from dataclasses import dataclass
from enum import Enum
import logging
from concurrent.futures import TimeoutError
from contextlib import asynccontextmanager
from .monitoring import record_metric
from .logging import get_logger

logger = get_logger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels for classification and handling."""
    CRITICAL = "critical"
    HIGH = "high"  
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class RecoveryStrategy(Enum):
    """Recovery strategies for different error types."""
    RETRY = "retry"
    FALLBACK = "fallback"
    CIRCUIT_BREAKER = "circuit_breaker"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    FAIL_FAST = "fail_fast"
    RESTART = "restart"


@dataclass
class ErrorContext:
    """Context information for error analysis and recovery."""
    
    error_id: str
    timestamp: float
    function_name: str
    module_name: str
    error_type: str
    error_message: str
    stack_trace: str
    severity: ErrorSeverity
    recovery_strategy: RecoveryStrategy
    context_data: Dict[str, Any]
    retry_count: int = 0
    max_retries: int = 3
    recovery_attempted: bool = False
    recovery_successful: bool = False


class ErrorRecoveryManager:
    """
    Advanced error recovery manager with intelligent recovery strategies.
    
    Provides automatic error detection, classification, and recovery
    with learning capabilities for improved reliability.
    """
    
    def __init__(self):
        self.error_history: List[ErrorContext] = []
        self.error_patterns: Dict[str, Any] = {}
        self.recovery_strategies: Dict[str, Callable] = {}
        self.circuit_breakers: Dict[str, Dict[str, Any]] = {}
        self.error_statistics: Dict[str, int] = {}
        
        # Initialize recovery strategies
        self._initialize_recovery_strategies()
        
        # Error analysis and learning
        self.learning_enabled = True
        self.pattern_threshold = 3  # Minimum occurrences to identify pattern
        
    def _initialize_recovery_strategies(self) -> None:
        """Initialize built-in recovery strategies."""
        self.recovery_strategies = {
            RecoveryStrategy.RETRY.value: self._retry_recovery,
            RecoveryStrategy.FALLBACK.value: self._fallback_recovery,
            RecoveryStrategy.CIRCUIT_BREAKER.value: self._circuit_breaker_recovery,
            RecoveryStrategy.GRACEFUL_DEGRADATION.value: self._graceful_degradation_recovery,
            RecoveryStrategy.FAIL_FAST.value: self._fail_fast_recovery,
            RecoveryStrategy.RESTART.value: self._restart_recovery
        }
    
    async def handle_error_with_recovery(
        self,
        error: Exception,
        context: Dict[str, Any],
        recovery_strategy: Optional[RecoveryStrategy] = None,
        max_retries: int = 3
    ) -> Any:
        """
        Handle error with intelligent recovery.
        
        Args:
            error: The exception that occurred
            context: Context information about the error
            recovery_strategy: Preferred recovery strategy
            max_retries: Maximum retry attempts
            
        Returns:
            Recovered result or raises if recovery fails
        """
        error_ctx = self._create_error_context(error, context, max_retries)
        
        # Record error for analysis
        self._record_error(error_ctx)
        
        # Determine recovery strategy if not specified
        if not recovery_strategy:
            recovery_strategy = self._determine_recovery_strategy(error_ctx)
        
        error_ctx.recovery_strategy = recovery_strategy
        
        # Attempt recovery
        try:
            recovery_func = self.recovery_strategies.get(recovery_strategy.value)
            if recovery_func:
                error_ctx.recovery_attempted = True
                result = await recovery_func(error_ctx)
                error_ctx.recovery_successful = True
                
                logger.info(f"Error recovery successful: {error_ctx.error_id}")
                record_metric("error_recovery_success", 1, "counter")
                
                return result
            else:
                logger.error(f"No recovery strategy for {recovery_strategy}")
                raise error
                
        except Exception as recovery_error:
            error_ctx.recovery_successful = False
            logger.error(f"Error recovery failed: {error_ctx.error_id} - {recovery_error}")
            record_metric("error_recovery_failure", 1, "counter")
            
            # Try alternative recovery if available
            alternative_strategy = self._get_alternative_recovery(recovery_strategy)
            if alternative_strategy and error_ctx.retry_count < max_retries:
                error_ctx.retry_count += 1
                return await self.handle_error_with_recovery(
                    error, context, alternative_strategy, max_retries
                )
            
            # All recovery attempts failed
            self._handle_unrecoverable_error(error_ctx)
            raise error
    
    def _create_error_context(
        self,
        error: Exception,
        context: Dict[str, Any],
        max_retries: int
    ) -> ErrorContext:
        """Create detailed error context for analysis."""
        error_id = f"error_{int(time.time() * 1000)}_{hash(str(error))}"
        
        return ErrorContext(
            error_id=error_id,
            timestamp=time.time(),
            function_name=context.get("function_name", "unknown"),
            module_name=context.get("module_name", "unknown"),
            error_type=type(error).__name__,
            error_message=str(error),
            stack_trace=traceback.format_exc(),
            severity=self._classify_error_severity(error),
            recovery_strategy=RecoveryStrategy.RETRY,  # Default
            context_data=context,
            max_retries=max_retries
        )
    
    def _classify_error_severity(self, error: Exception) -> ErrorSeverity:
        """Classify error severity based on error type and context."""
        # Critical errors
        if isinstance(error, (SystemError, MemoryError, KeyboardInterrupt)):
            return ErrorSeverity.CRITICAL
        
        # High severity errors
        if isinstance(error, (RuntimeError, OSError, IOError)):
            return ErrorSeverity.HIGH
        
        # Medium severity errors
        if isinstance(error, (ValueError, TypeError, AttributeError)):
            return ErrorSeverity.MEDIUM
        
        # Low severity errors
        if isinstance(error, (Warning, UserWarning)):
            return ErrorSeverity.LOW
        
        return ErrorSeverity.MEDIUM  # Default
    
    def _determine_recovery_strategy(self, error_ctx: ErrorContext) -> RecoveryStrategy:
        """Determine optimal recovery strategy based on error context and patterns."""
        # Check circuit breaker state
        circuit_key = f"{error_ctx.module_name}.{error_ctx.function_name}"
        if self._is_circuit_open(circuit_key):
            return RecoveryStrategy.CIRCUIT_BREAKER
        
        # Check error patterns
        pattern = self._identify_error_pattern(error_ctx)
        if pattern:
            return pattern.get("recommended_strategy", RecoveryStrategy.RETRY)
        
        # Default strategy based on error type and severity
        if error_ctx.severity == ErrorSeverity.CRITICAL:
            return RecoveryStrategy.FAIL_FAST
        elif error_ctx.severity == ErrorSeverity.HIGH:
            return RecoveryStrategy.CIRCUIT_BREAKER
        elif error_ctx.error_type in ["TimeoutError", "ConnectionError", "HTTPError"]:
            return RecoveryStrategy.RETRY
        elif error_ctx.error_type in ["ValueError", "TypeError"]:
            return RecoveryStrategy.FALLBACK
        else:
            return RecoveryStrategy.GRACEFUL_DEGRADATION
    
    async def _retry_recovery(self, error_ctx: ErrorContext) -> Any:
        """Implement retry recovery with exponential backoff."""
        if error_ctx.retry_count >= error_ctx.max_retries:
            raise RuntimeError(f"Max retries exceeded: {error_ctx.error_id}")
        
        # Exponential backoff
        wait_time = min(2 ** error_ctx.retry_count, 60)  # Cap at 60 seconds
        await asyncio.sleep(wait_time)
        
        error_ctx.retry_count += 1
        logger.info(f"Retrying operation: {error_ctx.error_id} (attempt {error_ctx.retry_count})")
        
        # Re-execute original function (would need to be passed in context)
        original_func = error_ctx.context_data.get("original_function")
        if original_func:
            return await original_func()
        else:
            raise RuntimeError("Original function not available for retry")
    
    async def _fallback_recovery(self, error_ctx: ErrorContext) -> Any:
        """Implement fallback recovery with alternative implementation."""
        fallback_func = error_ctx.context_data.get("fallback_function")
        if fallback_func:
            logger.info(f"Using fallback implementation: {error_ctx.error_id}")
            return await fallback_func()
        else:
            # Return safe default value
            default_value = error_ctx.context_data.get("default_value")
            if default_value is not None:
                logger.info(f"Returning default value: {error_ctx.error_id}")
                return default_value
            else:
                raise RuntimeError("No fallback implementation available")
    
    async def _circuit_breaker_recovery(self, error_ctx: ErrorContext) -> Any:
        """Implement circuit breaker recovery pattern."""
        circuit_key = f"{error_ctx.module_name}.{error_ctx.function_name}"
        
        # Check if circuit is open
        if self._is_circuit_open(circuit_key):
            logger.warning(f"Circuit breaker open: {circuit_key}")
            raise RuntimeError(f"Circuit breaker open for {circuit_key}")
        
        # Try operation and update circuit state
        try:
            original_func = error_ctx.context_data.get("original_function")
            if original_func:
                result = await original_func()
                self._close_circuit(circuit_key)
                return result
            else:
                raise RuntimeError("Original function not available")
        except Exception as e:
            self._record_circuit_failure(circuit_key)
            raise e
    
    async def _graceful_degradation_recovery(self, error_ctx: ErrorContext) -> Any:
        """Implement graceful degradation recovery."""
        logger.info(f"Graceful degradation for: {error_ctx.error_id}")
        
        # Return reduced functionality result
        degraded_func = error_ctx.context_data.get("degraded_function")
        if degraded_func:
            return await degraded_func()
        else:
            # Return minimal viable result
            minimal_result = error_ctx.context_data.get("minimal_result", {})
            logger.info(f"Returning minimal result: {minimal_result}")
            return minimal_result
    
    async def _fail_fast_recovery(self, error_ctx: ErrorContext) -> Any:
        """Implement fail-fast recovery (immediate failure)."""
        logger.error(f"Failing fast for critical error: {error_ctx.error_id}")
        raise RuntimeError(f"Fail-fast triggered for {error_ctx.error_id}")
    
    async def _restart_recovery(self, error_ctx: ErrorContext) -> Any:
        """Implement restart recovery for system-level errors."""
        logger.warning(f"Restart recovery for: {error_ctx.error_id}")
        
        # Restart specific component or service
        restart_func = error_ctx.context_data.get("restart_function")
        if restart_func:
            await restart_func()
            
            # Retry original operation after restart
            original_func = error_ctx.context_data.get("original_function")
            if original_func:
                return await original_func()
        
        raise RuntimeError("Restart recovery failed")
    
    def _record_error(self, error_ctx: ErrorContext) -> None:
        """Record error for analysis and pattern detection."""
        self.error_history.append(error_ctx)
        
        # Update statistics
        error_key = f"{error_ctx.error_type}_{error_ctx.severity.value}"
        self.error_statistics[error_key] = self.error_statistics.get(error_key, 0) + 1
        
        # Record metrics
        record_metric("error_occurred", 1, "counter", {"severity": error_ctx.severity.value})
        record_metric("error_by_type", 1, "counter", {"error_type": error_ctx.error_type})
        
        # Learn from error patterns if enabled
        if self.learning_enabled:
            self._analyze_error_patterns(error_ctx)
    
    def _analyze_error_patterns(self, error_ctx: ErrorContext) -> None:
        """Analyze error patterns for improved recovery strategies."""
        # Look for similar errors in recent history
        recent_errors = [
            e for e in self.error_history[-100:]  # Last 100 errors
            if e.error_type == error_ctx.error_type 
            and e.function_name == error_ctx.function_name
        ]
        
        if len(recent_errors) >= self.pattern_threshold:
            pattern_key = f"{error_ctx.error_type}_{error_ctx.function_name}"
            
            if pattern_key not in self.error_patterns:
                # Analyze successful recovery strategies
                successful_recoveries = [
                    e.recovery_strategy for e in recent_errors 
                    if e.recovery_successful
                ]
                
                if successful_recoveries:
                    # Most successful strategy becomes recommended
                    from collections import Counter
                    strategy_counts = Counter(successful_recoveries)
                    recommended_strategy = strategy_counts.most_common(1)[0][0]
                    
                    self.error_patterns[pattern_key] = {
                        "recommended_strategy": recommended_strategy,
                        "occurrence_count": len(recent_errors),
                        "success_rate": len(successful_recoveries) / len(recent_errors),
                        "first_seen": recent_errors[0].timestamp,
                        "last_seen": error_ctx.timestamp
                    }
                    
                    logger.info(f"Learned error pattern: {pattern_key} -> {recommended_strategy}")
    
    def _identify_error_pattern(self, error_ctx: ErrorContext) -> Optional[Dict[str, Any]]:
        """Identify if error matches a known pattern."""
        pattern_key = f"{error_ctx.error_type}_{error_ctx.function_name}"
        return self.error_patterns.get(pattern_key)
    
    def _get_alternative_recovery(self, failed_strategy: RecoveryStrategy) -> Optional[RecoveryStrategy]:
        """Get alternative recovery strategy when primary fails."""
        alternatives = {
            RecoveryStrategy.RETRY: RecoveryStrategy.FALLBACK,
            RecoveryStrategy.FALLBACK: RecoveryStrategy.GRACEFUL_DEGRADATION,
            RecoveryStrategy.CIRCUIT_BREAKER: RecoveryStrategy.FALLBACK,
            RecoveryStrategy.GRACEFUL_DEGRADATION: RecoveryStrategy.FAIL_FAST,
            RecoveryStrategy.RESTART: RecoveryStrategy.FAIL_FAST
        }
        return alternatives.get(failed_strategy)
    
    def _is_circuit_open(self, circuit_key: str) -> bool:
        """Check if circuit breaker is open."""
        circuit = self.circuit_breakers.get(circuit_key)
        if not circuit:
            return False
        
        failure_threshold = circuit.get("failure_threshold", 5)
        failure_count = circuit.get("failure_count", 0)
        last_failure = circuit.get("last_failure", 0)
        timeout_duration = circuit.get("timeout_duration", 300)  # 5 minutes
        
        # Circuit is open if failure threshold exceeded and timeout not expired
        if failure_count >= failure_threshold:
            if time.time() - last_failure < timeout_duration:
                return True
            else:
                # Timeout expired, reset circuit to half-open
                circuit["failure_count"] = 0
                return False
        
        return False
    
    def _record_circuit_failure(self, circuit_key: str) -> None:
        """Record circuit breaker failure."""
        if circuit_key not in self.circuit_breakers:
            self.circuit_breakers[circuit_key] = {
                "failure_count": 0,
                "failure_threshold": 5,
                "timeout_duration": 300,
                "last_failure": 0
            }
        
        circuit = self.circuit_breakers[circuit_key]
        circuit["failure_count"] += 1
        circuit["last_failure"] = time.time()
        
        logger.warning(f"Circuit breaker failure recorded: {circuit_key} "
                      f"({circuit['failure_count']}/{circuit['failure_threshold']})")
    
    def _close_circuit(self, circuit_key: str) -> None:
        """Close circuit breaker after successful operation."""
        if circuit_key in self.circuit_breakers:
            self.circuit_breakers[circuit_key]["failure_count"] = 0
            logger.info(f"Circuit breaker closed: {circuit_key}")
    
    def _handle_unrecoverable_error(self, error_ctx: ErrorContext) -> None:
        """Handle errors that cannot be recovered."""
        logger.critical(f"Unrecoverable error: {error_ctx.error_id}")
        
        # Send alert for critical errors
        if error_ctx.severity == ErrorSeverity.CRITICAL:
            self._send_critical_error_alert(error_ctx)
        
        # Record for post-mortem analysis
        record_metric("unrecoverable_error", 1, "counter", {
            "error_type": error_ctx.error_type,
            "function": error_ctx.function_name
        })
    
    def _send_critical_error_alert(self, error_ctx: ErrorContext) -> None:
        """Send alert for critical errors."""
        # Implementation would send alert via configured channels
        logger.critical(f"CRITICAL ERROR ALERT: {error_ctx.error_message}")
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics."""
        total_errors = len(self.error_history)
        successful_recoveries = sum(1 for e in self.error_history if e.recovery_successful)
        
        return {
            "total_errors": total_errors,
            "successful_recoveries": successful_recoveries,
            "recovery_rate": successful_recoveries / total_errors if total_errors > 0 else 0,
            "error_patterns_learned": len(self.error_patterns),
            "circuit_breakers_active": len(self.circuit_breakers),
            "error_types": dict(self.error_statistics),
            "recent_error_rate": len([e for e in self.error_history if e.timestamp > time.time() - 3600]) / 3600  # Errors per hour
        }


def robust_error_handler(
    recovery_strategy: RecoveryStrategy = RecoveryStrategy.RETRY,
    max_retries: int = 3,
    fallback_function: Optional[Callable] = None,
    default_value: Any = None
):
    """
    Decorator for robust error handling with recovery.
    
    Args:
        recovery_strategy: Primary recovery strategy
        max_retries: Maximum retry attempts
        fallback_function: Alternative function to call on error
        default_value: Default return value if all recovery fails
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            manager = ErrorRecoveryManager()
            
            context = {
                "function_name": func.__name__,
                "module_name": func.__module__,
                "original_function": lambda: func(*args, **kwargs),
                "fallback_function": fallback_function,
                "default_value": default_value
            }
            
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                return await manager.handle_error_with_recovery(
                    e, context, recovery_strategy, max_retries
                )
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # For sync functions, provide basic error handling
                if fallback_function:
                    logger.warning(f"Error in {func.__name__}, using fallback: {e}")
                    return fallback_function(*args, **kwargs)
                elif default_value is not None:
                    logger.warning(f"Error in {func.__name__}, using default value: {e}")
                    return default_value
                else:
                    raise
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


@asynccontextmanager
async def error_recovery_context(
    recovery_manager: Optional[ErrorRecoveryManager] = None,
    recovery_strategy: RecoveryStrategy = RecoveryStrategy.RETRY
):
    """
    Context manager for error recovery in code blocks.
    
    Usage:
        async with error_recovery_context():
            # Code that might fail
            result = await some_operation()
    """
    manager = recovery_manager or ErrorRecoveryManager()
    
    try:
        yield manager
    except Exception as e:
        context = {
            "function_name": "context_block",
            "module_name": __name__
        }
        
        await manager.handle_error_with_recovery(e, context, recovery_strategy)


# Global error recovery manager instance
global_error_manager = ErrorRecoveryManager()