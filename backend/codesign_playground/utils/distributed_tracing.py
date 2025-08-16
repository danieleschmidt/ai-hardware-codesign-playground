"""
Distributed tracing and correlation ID management for AI Hardware Co-Design Playground.

This module provides comprehensive distributed tracing capabilities with correlation ID
propagation, span management, and integration with monitoring systems.
"""

import time
import uuid
import threading
import json
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from contextlib import contextmanager
from functools import wraps
from enum import Enum

from .logging import get_logger, get_performance_logger, get_audit_logger
from .monitoring import record_metric

logger = get_logger(__name__)
performance_logger = get_performance_logger(__name__)


class SpanType(Enum):
    """Types of spans for different operations."""
    HTTP_REQUEST = "http_request"
    DATABASE = "database"
    FUNCTION_CALL = "function_call"
    AI_MODEL = "ai_model"
    HARDWARE_DESIGN = "hardware_design"
    OPTIMIZATION = "optimization"
    VALIDATION = "validation"
    EXTERNAL_SERVICE = "external_service"


@dataclass
class SpanContext:
    """Context information for a span."""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    baggage: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert span context to dictionary."""
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "baggage": self.baggage
        }


@dataclass
class Span:
    """Distributed tracing span with detailed metadata."""
    context: SpanContext
    operation_name: str
    span_type: SpanType
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    duration: Optional[float] = None
    tags: Dict[str, Any] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "ok"
    error: Optional[str] = None
    
    def set_tag(self, key: str, value: Any) -> None:
        """Set a tag on the span."""
        self.tags[key] = value
        
    def log(self, event: str, payload: Optional[Dict[str, Any]] = None) -> None:
        """Add a log entry to the span."""
        log_entry = {
            "timestamp": time.time(),
            "event": event,
            "payload": payload or {}
        }
        self.logs.append(log_entry)
        
    def set_error(self, error: Union[str, Exception]) -> None:
        """Mark span as having an error."""
        self.status = "error"
        if isinstance(error, Exception):
            self.error = f"{type(error).__name__}: {str(error)}"
        else:
            self.error = str(error)
        self.set_tag("error", True)
        self.set_tag("error.object", self.error)
        
    def finish(self) -> None:
        """Finish the span and calculate duration."""
        if self.end_time is None:
            self.end_time = time.time()
            self.duration = self.end_time - self.start_time
            
            # Record span metrics
            record_metric(f"span_duration_{self.span_type.value}", self.duration, "histogram",
                         {"operation": self.operation_name, "status": self.status})
            record_metric(f"span_count_{self.span_type.value}", 1, "counter",
                         {"operation": self.operation_name, "status": self.status})
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert span to dictionary for serialization."""
        return {
            "context": self.context.to_dict(),
            "operation_name": self.operation_name,
            "span_type": self.span_type.value,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "tags": self.tags,
            "logs": self.logs,
            "status": self.status,
            "error": self.error
        }


class DistributedTracer:
    """Distributed tracing implementation with correlation ID management."""
    
    def __init__(self):
        """Initialize distributed tracer."""
        self._active_spans: Dict[str, Span] = {}
        self._span_storage: List[Span] = []
        self._lock = threading.Lock()
        self.max_spans = 10000  # Limit memory usage
        
        logger.info("Initialized DistributedTracer")
    
    def start_span(self, 
                   operation_name: str, 
                   span_type: SpanType = SpanType.FUNCTION_CALL,
                   parent_context: Optional[SpanContext] = None,
                   tags: Optional[Dict[str, Any]] = None) -> Span:
        """Start a new span."""
        # Generate IDs
        trace_id = parent_context.trace_id if parent_context else self._generate_trace_id()
        span_id = self._generate_span_id()
        parent_span_id = parent_context.span_id if parent_context else None
        
        # Create span context
        context = SpanContext(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            baggage=parent_context.baggage.copy() if parent_context else {}
        )
        
        # Create span
        span = Span(
            context=context,
            operation_name=operation_name,
            span_type=span_type,
            tags=tags or {}
        )
        
        # Set default tags
        span.set_tag("thread.id", threading.get_ident())
        span.set_tag("thread.name", threading.current_thread().name)
        
        # Store span
        with self._lock:
            self._active_spans[span_id] = span
            # Limit memory usage
            if len(self._span_storage) >= self.max_spans:
                self._span_storage = self._span_storage[-self.max_spans//2:]
        
        logger.debug("Started span",
                    operation_name=operation_name,
                    trace_id=trace_id,
                    span_id=span_id,
                    parent_span_id=parent_span_id)
        
        return span
    
    def finish_span(self, span: Span) -> None:
        """Finish a span and move it to storage."""
        span.finish()
        
        with self._lock:
            # Remove from active spans
            if span.context.span_id in self._active_spans:
                del self._active_spans[span.context.span_id]
            
            # Add to storage
            self._span_storage.append(span)
        
        # Log span completion
        logger.debug("Finished span",
                    operation_name=span.operation_name,
                    trace_id=span.context.trace_id,
                    span_id=span.context.span_id,
                    duration=span.duration,
                    status=span.status)
        
        # Log performance metrics
        if span.duration:
            performance_logger.record_metric(
                f"span_{span.operation_name.replace('.', '_')}_duration",
                span.duration,
                operation=span.operation_name,
                trace_id=span.context.trace_id,
                status=span.status
            )
    
    def get_active_span(self) -> Optional[Span]:
        """Get the currently active span for this thread."""
        thread_local_data = getattr(threading.current_thread(), '_tracer_data', None)
        if thread_local_data and 'active_span' in thread_local_data:
            return thread_local_data['active_span']
        return None
    
    def set_active_span(self, span: Optional[Span]) -> None:
        """Set the active span for this thread."""
        if not hasattr(threading.current_thread(), '_tracer_data'):
            threading.current_thread()._tracer_data = {}
        threading.current_thread()._tracer_data['active_span'] = span
    
    def inject_context(self, span_context: SpanContext) -> Dict[str, str]:
        """Inject span context into headers for propagation."""
        return {
            "X-Trace-Id": span_context.trace_id,
            "X-Span-Id": span_context.span_id,
            "X-Parent-Span-Id": span_context.parent_span_id or "",
            "X-Baggage": json.dumps(span_context.baggage) if span_context.baggage else ""
        }
    
    def extract_context(self, headers: Dict[str, str]) -> Optional[SpanContext]:
        """Extract span context from headers."""
        trace_id = headers.get("X-Trace-Id")
        span_id = headers.get("X-Span-Id")
        
        if not trace_id or not span_id:
            return None
        
        parent_span_id = headers.get("X-Parent-Span-Id") or None
        baggage_str = headers.get("X-Baggage", "")
        
        try:
            baggage = json.loads(baggage_str) if baggage_str else {}
        except json.JSONDecodeError:
            baggage = {}
        
        return SpanContext(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            baggage=baggage
        )
    
    def get_trace(self, trace_id: str) -> List[Span]:
        """Get all spans for a trace."""
        with self._lock:
            trace_spans = []
            
            # Check active spans
            for span in self._active_spans.values():
                if span.context.trace_id == trace_id:
                    trace_spans.append(span)
            
            # Check stored spans
            for span in self._span_storage:
                if span.context.trace_id == trace_id:
                    trace_spans.append(span)
            
            # Sort by start time
            trace_spans.sort(key=lambda s: s.start_time)
            return trace_spans
    
    def get_trace_summary(self, trace_id: str) -> Dict[str, Any]:
        """Get summary information for a trace."""
        spans = self.get_trace(trace_id)
        
        if not spans:
            return {"trace_id": trace_id, "spans": [], "summary": {}}
        
        root_span = min(spans, key=lambda s: s.start_time)
        total_duration = max(s.end_time or time.time() for s in spans) - root_span.start_time
        error_count = sum(1 for s in spans if s.status == "error")
        
        summary = {
            "trace_id": trace_id,
            "root_operation": root_span.operation_name,
            "total_duration": total_duration,
            "span_count": len(spans),
            "error_count": error_count,
            "status": "error" if error_count > 0 else "ok",
            "start_time": root_span.start_time,
            "end_time": max(s.end_time or time.time() for s in spans)
        }
        
        return {
            "trace_id": trace_id,
            "summary": summary,
            "spans": [span.to_dict() for span in spans]
        }
    
    def _generate_trace_id(self) -> str:
        """Generate a new trace ID."""
        return str(uuid.uuid4()).replace('-', '')
    
    def _generate_span_id(self) -> str:
        """Generate a new span ID."""
        return str(uuid.uuid4()).replace('-', '')[:16]


# Global tracer instance
_tracer = DistributedTracer()


def get_tracer() -> DistributedTracer:
    """Get the global tracer instance."""
    return _tracer


@contextmanager
def trace_span(operation_name: str, 
               span_type: SpanType = SpanType.FUNCTION_CALL,
               tags: Optional[Dict[str, Any]] = None):
    """Context manager for creating and managing spans."""
    tracer = get_tracer()
    parent_span = tracer.get_active_span()
    parent_context = parent_span.context if parent_span else None
    
    span = tracer.start_span(operation_name, span_type, parent_context, tags)
    tracer.set_active_span(span)
    
    try:
        yield span
    except Exception as e:
        span.set_error(e)
        raise
    finally:
        tracer.finish_span(span)
        tracer.set_active_span(parent_span)


def traced_function(operation_name: Optional[str] = None,
                   span_type: SpanType = SpanType.FUNCTION_CALL,
                   tags: Optional[Dict[str, Any]] = None):
    """Decorator for automatic function tracing."""
    def decorator(func: Callable) -> Callable:
        op_name = operation_name or f"{func.__module__}.{func.__name__}"
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            with trace_span(op_name, span_type, tags) as span:
                # Add function metadata
                span.set_tag("function.name", func.__name__)
                span.set_tag("function.module", func.__module__)
                span.set_tag("function.args_count", len(args))
                span.set_tag("function.kwargs_count", len(kwargs))
                
                try:
                    result = func(*args, **kwargs)
                    span.set_tag("function.result_type", type(result).__name__)
                    return result
                except Exception as e:
                    span.set_tag("function.exception", str(e))
                    raise
        
        return wrapper
    return decorator


def get_correlation_id() -> Optional[str]:
    """Get the current correlation ID (trace ID) from active span."""
    tracer = get_tracer()
    active_span = tracer.get_active_span()
    return active_span.context.trace_id if active_span else None


def set_baggage(key: str, value: str) -> None:
    """Set baggage item in current span context."""
    tracer = get_tracer()
    active_span = tracer.get_active_span()
    if active_span:
        active_span.context.baggage[key] = value


def get_baggage(key: str) -> Optional[str]:
    """Get baggage item from current span context."""
    tracer = get_tracer()
    active_span = tracer.get_active_span()
    if active_span:
        return active_span.context.baggage.get(key)
    return None


class CorrelationContext:
    """Context manager for correlation ID propagation."""
    
    def __init__(self, correlation_id: Optional[str] = None, **baggage):
        """Initialize correlation context."""
        self.correlation_id = correlation_id or str(uuid.uuid4()).replace('-', '')
        self.baggage = baggage
        self._previous_span = None
    
    def __enter__(self):
        """Enter correlation context."""
        tracer = get_tracer()
        self._previous_span = tracer.get_active_span()
        
        # Create a new root span if no active span
        if self._previous_span is None:
            context = SpanContext(
                trace_id=self.correlation_id,
                span_id=tracer._generate_span_id(),
                baggage=self.baggage
            )
            span = Span(
                context=context,
                operation_name="correlation_context",
                span_type=SpanType.FUNCTION_CALL
            )
            tracer.set_active_span(span)
        else:
            # Update baggage in existing span
            self._previous_span.context.baggage.update(self.baggage)
        
        return self.correlation_id
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit correlation context."""
        tracer = get_tracer()
        current_span = tracer.get_active_span()
        
        if current_span and current_span != self._previous_span:
            tracer.finish_span(current_span)
        
        tracer.set_active_span(self._previous_span)


@contextmanager
def correlation_context(correlation_id: Optional[str] = None, **baggage):
    """Context manager for correlation ID management."""
    with CorrelationContext(correlation_id, **baggage) as cid:
        yield cid