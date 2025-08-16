"""
Advanced validation framework for comprehensive error handling and security.

This module provides enterprise-grade validation patterns with detailed error
reporting, audit logging, and integration with monitoring systems.
"""

from typing import Any, Dict, List, Optional, Union, Callable, Type
import re
import time
import hashlib
import inspect
from dataclasses import dataclass, field
from functools import wraps
from contextlib import contextmanager
import threading

from .validation import ValidationResult, SecurityValidator
from .logging import get_logger, get_audit_logger
from .monitoring import record_metric
from .exceptions import ValidationError, SecurityError

logger = get_logger(__name__)
audit_logger = get_audit_logger(__name__)


@dataclass
class ValidationContext:
    """Context for validation operations with correlation tracking."""
    correlation_id: str = field(default_factory=lambda: hashlib.md5(str(time.time()).encode()).hexdigest()[:12])
    user_id: Optional[str] = None
    client_id: Optional[str] = None
    operation: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ValidationSchema:
    """Schema-based validation with type checking and constraints."""
    
    def __init__(self, schema: Dict[str, Any]):
        """Initialize validation schema."""
        self.schema = schema
        self.security_validator = SecurityValidator()
        
    def validate(self, data: Dict[str, Any], context: Optional[ValidationContext] = None) -> ValidationResult:
        """Validate data against schema."""
        context = context or ValidationContext()
        result = ValidationResult(is_valid=True)
        result.context.update({
            "schema_validation": True,
            "correlation_id": context.correlation_id,
            "operation": context.operation
        })
        
        # Check required fields
        required_fields = self.schema.get("required", [])
        for field_name in required_fields:
            if field_name not in data:
                result.add_error(f"Required field missing: {field_name}", 
                               field_name, "REQUIRED_FIELD_MISSING")
        
        # Validate each field in schema
        properties = self.schema.get("properties", {})
        for field_name, field_schema in properties.items():
            if field_name in data:
                field_result = self._validate_field(data[field_name], field_schema, field_name, context)
                if not field_result:
                    result.errors.extend(field_result.errors)
                    result.warnings.extend(field_result.warnings)
                    result.is_valid = False
        
        # Check for unexpected fields
        if self.schema.get("additionalProperties", True) is False:
            for field_name in data:
                if field_name not in properties:
                    result.add_warning(f"Unexpected field: {field_name}", field_name, "UNEXPECTED_FIELD")
        
        return result
    
    def _validate_field(self, value: Any, field_schema: Dict[str, Any], 
                       field_name: str, context: ValidationContext) -> ValidationResult:
        """Validate individual field against its schema."""
        result = ValidationResult(is_valid=True)
        
        # Type validation
        expected_type = field_schema.get("type")
        if expected_type:
            if not self._check_type(value, expected_type):
                result.add_error(f"Invalid type for {field_name}. Expected {expected_type}, got {type(value).__name__}",
                               field_name, "TYPE_MISMATCH",
                               {"expected": expected_type, "actual": type(value).__name__})
                return result
        
        # String validation
        if isinstance(value, str):
            string_result = self.security_validator.validate_string_input(value, field_name, context.client_id)
            if not string_result:
                result.errors.extend(string_result.errors)
                result.warnings.extend(string_result.warnings)
                result.is_valid = False
            
            # Length constraints
            min_length = field_schema.get("minLength")
            max_length = field_schema.get("maxLength")
            if min_length is not None and len(value) < min_length:
                result.add_error(f"String too short: {len(value)} < {min_length}",
                               field_name, "STRING_TOO_SHORT")
            if max_length is not None and len(value) > max_length:
                result.add_error(f"String too long: {len(value)} > {max_length}",
                               field_name, "STRING_TOO_LONG")
            
            # Pattern validation
            pattern = field_schema.get("pattern")
            if pattern and not re.match(pattern, value):
                result.add_error(f"String does not match pattern: {pattern}",
                               field_name, "PATTERN_MISMATCH")
        
        # Numeric validation
        elif isinstance(value, (int, float)):
            numeric_result = self.security_validator.validate_numeric_input(
                value, field_name, 
                field_schema.get("minimum"), 
                field_schema.get("maximum")
            )
            if not numeric_result:
                result.errors.extend(numeric_result.errors)
                result.warnings.extend(numeric_result.warnings)
                result.is_valid = False
        
        # Array validation
        elif isinstance(value, list):
            min_items = field_schema.get("minItems")
            max_items = field_schema.get("maxItems")
            if min_items is not None and len(value) < min_items:
                result.add_error(f"Array too short: {len(value)} < {min_items}",
                               field_name, "ARRAY_TOO_SHORT")
            if max_items is not None and len(value) > max_items:
                result.add_error(f"Array too long: {len(value)} > {max_items}",
                               field_name, "ARRAY_TOO_LONG")
            
            # Validate array items
            item_schema = field_schema.get("items")
            if item_schema:
                for i, item in enumerate(value):
                    item_result = self._validate_field(item, item_schema, f"{field_name}[{i}]", context)
                    if not item_result:
                        result.errors.extend(item_result.errors)
                        result.warnings.extend(item_result.warnings)
                        result.is_valid = False
        
        return result
    
    def _check_type(self, value: Any, expected_type: str) -> bool:
        """Check if value matches expected type."""
        type_map = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "array": list,
            "object": dict,
            "null": type(None)
        }
        
        expected_python_type = type_map.get(expected_type)
        if expected_python_type is None:
            return True  # Unknown type, skip validation
        
        return isinstance(value, expected_python_type)


class ValidatedFunction:
    """Decorator for comprehensive function validation."""
    
    def __init__(self, 
                 input_schema: Optional[Dict[str, Any]] = None,
                 output_schema: Optional[Dict[str, Any]] = None,
                 security_level: str = "medium",
                 rate_limit: Optional[int] = None,
                 timeout: Optional[float] = None):
        """Initialize validated function decorator."""
        self.input_schema = ValidationSchema(input_schema) if input_schema else None
        self.output_schema = ValidationSchema(output_schema) if output_schema else None
        self.security_level = security_level
        self.rate_limit = rate_limit
        self.timeout = timeout
        self._call_counts = {}
        self._lock = threading.Lock()
    
    def __call__(self, func: Callable) -> Callable:
        """Apply validation to function."""
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            context = ValidationContext(
                operation=f"{func.__module__}.{func.__name__}",
                client_id=kwargs.pop('_client_id', None),
                user_id=kwargs.pop('_user_id', None)
            )
            
            try:
                # Rate limiting
                if self.rate_limit and not self._check_rate_limit(context.client_id or "anonymous"):
                    raise ValidationError("Rate limit exceeded", "RATE_LIMIT_EXCEEDED")
                
                # Input validation
                if self.input_schema:
                    input_data = self._extract_input_data(func, args, kwargs)
                    input_result = self.input_schema.validate(input_data, context)
                    if not input_result:
                        audit_logger.log_security_event("validation_failed",
                                                       f"Input validation failed for {func.__name__}",
                                                       self.security_level,
                                                       correlation_id=context.correlation_id,
                                                       errors=input_result.errors)
                        raise ValidationError(f"Input validation failed: {input_result.errors}", 
                                            "INPUT_VALIDATION_FAILED",
                                            {"validation_result": input_result.to_dict()})
                
                # Execute function
                result = func(*args, **kwargs)
                
                # Output validation
                if self.output_schema and result is not None:
                    if isinstance(result, dict):
                        output_result = self.output_schema.validate(result, context)
                        if not output_result:
                            logger.error("Output validation failed", 
                                       function=func.__name__,
                                       errors=output_result.errors,
                                       correlation_id=context.correlation_id)
                
                # Record success metrics
                execution_time = time.time() - start_time
                record_metric(f"function_{func.__name__}_success", 1, "counter")
                record_metric(f"function_{func.__name__}_duration", execution_time, "histogram")
                
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                
                # Record error metrics
                record_metric(f"function_{func.__name__}_error", 1, "counter", 
                             {"error_type": type(e).__name__})
                record_metric(f"function_{func.__name__}_duration", execution_time, "histogram")
                
                # Enhanced error logging
                logger.error(f"Function {func.__name__} failed",
                           error=str(e),
                           error_type=type(e).__name__,
                           correlation_id=context.correlation_id,
                           execution_time=execution_time)
                
                raise
        
        return wrapper
    
    def _extract_input_data(self, func: Callable, args: tuple, kwargs: dict) -> Dict[str, Any]:
        """Extract input data for validation."""
        sig = inspect.signature(func)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        return dict(bound_args.arguments)
    
    def _check_rate_limit(self, client_id: str) -> bool:
        """Check rate limiting for client."""
        if not self.rate_limit:
            return True
        
        current_time = time.time()
        window_start = current_time - 60  # 1-minute window
        
        with self._lock:
            if client_id not in self._call_counts:
                self._call_counts[client_id] = []
            
            # Clean old calls
            self._call_counts[client_id] = [
                call_time for call_time in self._call_counts[client_id]
                if call_time > window_start
            ]
            
            # Check limit
            if len(self._call_counts[client_id]) >= self.rate_limit:
                return False
            
            # Record call
            self._call_counts[client_id].append(current_time)
            return True


def validate_function(input_schema: Optional[Dict[str, Any]] = None,
                     output_schema: Optional[Dict[str, Any]] = None,
                     security_level: str = "medium",
                     rate_limit: Optional[int] = None) -> Callable:
    """Convenience decorator for function validation."""
    return ValidatedFunction(input_schema, output_schema, security_level, rate_limit)


@contextmanager
def validation_context(correlation_id: Optional[str] = None,
                      user_id: Optional[str] = None,
                      operation: Optional[str] = None):
    """Context manager for validation operations."""
    context = ValidationContext(
        correlation_id=correlation_id or hashlib.md5(str(time.time()).encode()).hexdigest()[:12],
        user_id=user_id,
        operation=operation
    )
    
    # Store context in thread-local storage
    if not hasattr(threading.current_thread(), 'validation_context'):
        threading.current_thread().validation_context = context
    
    try:
        yield context
    finally:
        if hasattr(threading.current_thread(), 'validation_context'):
            delattr(threading.current_thread(), 'validation_context')


def get_validation_context() -> Optional[ValidationContext]:
    """Get current validation context from thread-local storage."""
    return getattr(threading.current_thread(), 'validation_context', None)


class ModelConfigValidator(ValidationSchema):
    """Specialized validator for model configurations."""
    
    def __init__(self):
        schema = {
            "type": "object",
            "required": ["path"],
            "properties": {
                "path": {
                    "type": "string",
                    "pattern": r"^[^<>&|;`$]*\.(onnx|pt|pb|h5|tflite)$"
                },
                "input_shape": {
                    "type": "array",
                    "items": {"type": "integer", "minimum": 1, "maximum": 10000},
                    "minItems": 1,
                    "maxItems": 10
                },
                "batch_size": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 1000
                },
                "precision": {
                    "type": "string",
                    "enum": ["fp32", "fp16", "int8", "int4"]
                },
                "optimization_level": {
                    "type": "string",
                    "enum": ["O0", "O1", "O2", "O3"]
                }
            },
            "additionalProperties": False
        }
        super().__init__(schema)


class HardwareConfigValidator(ValidationSchema):
    """Specialized validator for hardware configurations."""
    
    def __init__(self):
        schema = {
            "type": "object",
            "required": ["compute_units", "dataflow"],
            "properties": {
                "compute_units": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 2048
                },
                "dataflow": {
                    "type": "string",
                    "enum": ["weight_stationary", "output_stationary", "row_stationary"]
                },
                "frequency_mhz": {
                    "type": "number",
                    "minimum": 1.0,
                    "maximum": 5000.0
                },
                "data_width": {
                    "type": "integer",
                    "enum": [8, 16, 32, 64]
                },
                "precision": {
                    "type": "string",
                    "enum": ["int8", "int16", "fp16", "fp32"]
                },
                "power_budget_w": {
                    "type": "number",
                    "minimum": 0.1,
                    "maximum": 1000.0
                },
                "area_budget_mm2": {
                    "type": "number",
                    "minimum": 0.1,
                    "maximum": 10000.0
                }
            },
            "additionalProperties": False
        }
        super().__init__(schema)


# Global validators
_model_validator = ModelConfigValidator()
_hardware_validator = HardwareConfigValidator()


def validate_model_config(config: Dict[str, Any], 
                         context: Optional[ValidationContext] = None) -> ValidationResult:
    """Validate model configuration with enhanced security."""
    return _model_validator.validate(config, context)


def validate_hardware_config(config: Dict[str, Any], 
                            context: Optional[ValidationContext] = None) -> ValidationResult:
    """Validate hardware configuration with enhanced security."""
    return _hardware_validator.validate(config, context)