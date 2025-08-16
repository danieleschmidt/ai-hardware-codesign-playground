"""
Logging utilities for AI Hardware Co-Design Playground.

This module provides structured logging, monitoring, and observability
features for the platform.
"""

import logging
import logging.handlers
import json
import time
import sys
from typing import Any, Dict, Optional, Union
from pathlib import Path
from datetime import datetime
import traceback
from contextlib import contextmanager


class StructuredFormatter(logging.Formatter):
    """Formatter for structured JSON logging."""
    
    def __init__(self, include_extra: bool = True, include_tracing: bool = True):
        super().__init__()
        self.include_extra = include_extra
        self.include_tracing = include_tracing
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON."""
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "host": self._get_hostname(),
            "service": "ai_hardware_codesign"
        }
        
        # Add process/thread info
        if record.process:
            log_data["process_id"] = record.process
        if record.thread:
            log_data["thread_id"] = record.thread
        
        # Add distributed tracing information
        if self.include_tracing:
            trace_info = self._get_trace_info()
            if trace_info:
                log_data["trace"] = trace_info
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info)
            }
        
        # Add extra fields
        if self.include_extra:
            extra_fields = {
                k: v for k, v in record.__dict__.items()
                if k not in {
                    'name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                    'filename', 'module', 'exc_info', 'exc_text', 'stack_info',
                    'lineno', 'funcName', 'created', 'msecs', 'relativeCreated',
                    'thread', 'threadName', 'processName', 'process', 'message'
                }
            }
            if extra_fields:
                log_data["extra"] = extra_fields
        
        return json.dumps(log_data, default=str, ensure_ascii=False)
    
    def _get_hostname(self) -> str:
        """Get hostname for log entries."""
        import socket
        try:
            return socket.gethostname()
        except:
            return "unknown"
    
    def _get_trace_info(self) -> Optional[Dict[str, str]]:
        """Get current tracing information."""
        try:
            # Import here to avoid circular imports
            from .distributed_tracing import get_tracer
            
            tracer = get_tracer()
            active_span = tracer.get_active_span()
            
            if active_span:
                return {
                    "trace_id": active_span.context.trace_id,
                    "span_id": active_span.context.span_id,
                    "parent_span_id": active_span.context.parent_span_id,
                    "operation": active_span.operation_name
                }
        except:
            pass  # Gracefully handle import or other errors
        
        return None


class CodesignLogger:
    """Enhanced logger for the codesign platform with tracing support."""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.context_stack = []
        self._correlation_id = None
    
    def debug(self, message: str, **kwargs) -> None:
        """Log debug message with context."""
        self._log(logging.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs) -> None:
        """Log info message with context."""
        self._log(logging.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs) -> None:
        """Log warning message with context."""
        self._log(logging.WARNING, message, **kwargs)
    
    def error(self, message: str, exception: Optional[Exception] = None, **kwargs) -> None:
        """Log error message with context."""
        if exception:
            kwargs["exception_type"] = type(exception).__name__
            kwargs["exception_message"] = str(exception)
        self._log(logging.ERROR, message, exc_info=exception is not None, **kwargs)
    
    def critical(self, message: str, exception: Optional[Exception] = None, **kwargs) -> None:
        """Log critical message with context."""
        if exception:
            kwargs["exception_type"] = type(exception).__name__
            kwargs["exception_message"] = str(exception)
        self._log(logging.CRITICAL, message, exc_info=exception is not None, **kwargs)
    
    def _log(self, level: int, message: str, **kwargs) -> None:
        """Internal logging method with context and tracing injection."""
        # Add context from stack
        for context in self.context_stack:
            kwargs.update(context)
        
        # Add timestamp
        kwargs["log_timestamp"] = time.time()
        
        # Add correlation ID from tracing if available
        correlation_id = self._get_correlation_id()
        if correlation_id:
            kwargs["correlation_id"] = correlation_id
        
        # Add custom correlation ID if set
        if self._correlation_id:
            kwargs["custom_correlation_id"] = self._correlation_id
        
        # Add span information
        span_info = self._get_span_info()
        if span_info:
            kwargs.update(span_info)
        
        self.logger.log(level, message, extra=kwargs)
    
    def _get_correlation_id(self) -> Optional[str]:
        """Get correlation ID from active tracing context."""
        try:
            from .distributed_tracing import get_correlation_id
            return get_correlation_id()
        except:
            return None
    
    def _get_span_info(self) -> Dict[str, Any]:
        """Get current span information."""
        try:
            from .distributed_tracing import get_tracer
            
            tracer = get_tracer()
            active_span = tracer.get_active_span()
            
            if active_span:
                return {
                    "span_operation": active_span.operation_name,
                    "span_type": active_span.span_type.value,
                    "span_tags": active_span.tags
                }
        except:
            pass
        
        return {}
    
    def set_correlation_id(self, correlation_id: str) -> None:
        """Set custom correlation ID for this logger."""
        self._correlation_id = correlation_id
    
    def clear_correlation_id(self) -> None:
        """Clear custom correlation ID."""
        self._correlation_id = None
    
    @contextmanager
    def context(self, **context_data):
        """Add context to all log messages within this block."""
        self.context_stack.append(context_data)
        try:
            yield
        finally:
            self.context_stack.pop()


class PerformanceLogger:
    """Logger for performance metrics and monitoring."""
    
    def __init__(self, logger: CodesignLogger):
        self.logger = logger
        self.metrics = {}
    
    @contextmanager
    def timer(self, operation: str, **context):
        """Time an operation and log the duration."""
        start_time = time.time()
        self.logger.info(f"Starting {operation}", operation=operation, **context)
        
        try:
            yield
            duration = time.time() - start_time
            self.logger.info(
                f"Completed {operation}",
                operation=operation,
                duration_seconds=duration,
                status="success",
                **context
            )
            self._record_metric(f"{operation}_duration", duration, context)
            
        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(
                f"Failed {operation}",
                operation=operation,
                duration_seconds=duration,
                status="error",
                exception=e,
                **context
            )
            self._record_metric(f"{operation}_error", 1, context)
            raise
    
    def record_metric(self, name: str, value: Union[int, float], **context) -> None:
        """Record a custom metric."""
        self._record_metric(name, value, context)
        self.logger.info(
            f"Metric recorded: {name}",
            metric_name=name,
            metric_value=value,
            **context
        )
    
    def _record_metric(self, name: str, value: Union[int, float], context: Dict[str, Any]) -> None:
        """Internal method to record metrics."""
        timestamp = time.time()
        
        if name not in self.metrics:
            self.metrics[name] = []
        
        self.metrics[name].append({
            "value": value,
            "timestamp": timestamp,
            "context": context
        })
        
        # Keep only recent metrics (last 1000 per metric)
        if len(self.metrics[name]) > 1000:
            self.metrics[name] = self.metrics[name][-1000:]


class AuditLogger:
    """Logger for security and compliance auditing."""
    
    def __init__(self, logger: CodesignLogger):
        self.logger = logger
    
    def log_access(self, user_id: str, resource: str, action: str, success: bool, **context) -> None:
        """Log resource access attempt."""
        self.logger.info(
            f"Access {action} on {resource}",
            audit_type="access",
            user_id=user_id,
            resource=resource,
            action=action,
            success=success,
            **context
        )
    
    def log_security_event(self, event_type: str, details: str, severity: str = "medium", **context) -> None:
        """Log security-related events."""
        log_method = {
            "low": self.logger.info,
            "medium": self.logger.warning,
            "high": self.logger.error,
            "critical": self.logger.critical
        }.get(severity, self.logger.warning)
        
        log_method(
            f"Security event: {details}",
            audit_type="security",
            event_type=event_type,
            severity=severity,
            **context
        )
    
    def log_configuration_change(self, component: str, old_value: Any, new_value: Any, user_id: str, **context) -> None:
        """Log configuration changes."""
        self.logger.info(
            f"Configuration changed: {component}",
            audit_type="configuration",
            component=component,
            old_value=str(old_value),
            new_value=str(new_value),
            user_id=user_id,
            **context
        )


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,  
    structured: bool = True,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> None:
    """
    Set up logging configuration for the platform.
    
    Args:
        log_level: Logging level
        log_file: Optional log file path
        structured: Whether to use structured JSON logging
        max_bytes: Maximum log file size before rotation
        backup_count: Number of backup files to keep
    """
    # Convert string level to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    
    if structured:
        console_formatter = StructuredFormatter()
    else:
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(numeric_level)
        
        if structured:
            file_formatter = StructuredFormatter()
        else:
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
    
    # Configure specific loggers to prevent noise
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("celery").setLevel(logging.WARNING)


def get_logger(name: str) -> CodesignLogger:
    """
    Get a logger instance for the specified name.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        CodesignLogger instance
    """
    return CodesignLogger(name)


def get_performance_logger(name: str) -> PerformanceLogger:
    """
    Get a performance logger instance.
    
    Args:
        name: Logger name
        
    Returns:
        PerformanceLogger instance
    """
    logger = get_logger(name)
    return PerformanceLogger(logger)


def get_audit_logger(name: str) -> AuditLogger:
    """
    Get an audit logger instance.
    
    Args:
        name: Logger name
        
    Returns:
        AuditLogger instance
    """
    logger = get_logger(name)
    return AuditLogger(logger)


# Context managers for common logging patterns
@contextmanager
def log_operation(operation: str, logger: Optional[CodesignLogger] = None, **context):
    """Log the start and completion of an operation."""
    if logger is None:
        logger = get_logger(__name__)
    
    start_time = time.time()
    logger.info(f"Starting {operation}", operation=operation, **context)
    
    try:
        yield
        duration = time.time() - start_time
        logger.info(
            f"Completed {operation}",
            operation=operation,
            duration_seconds=duration,
            status="success",
            **context
        )
    except Exception as e:
        duration = time.time() - start_time
        logger.error(
            f"Failed {operation}",
            operation=operation,
            duration_seconds=duration,
            status="error",
            exception=e,
            **context
        )
        raise


@contextmanager
def log_context(**context_data):
    """Add context to all log messages within this block."""
    logger = get_logger(__name__)
    with logger.context(**context_data):
        yield