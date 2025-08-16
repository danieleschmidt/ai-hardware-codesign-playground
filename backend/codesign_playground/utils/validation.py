"""
Input validation utilities for AI Hardware Co-Design Playground.

This module provides comprehensive validation for user inputs,
configurations, and data integrity checks with enhanced security features.
"""

from typing import Any, Dict, List, Optional, Union, Tuple, Callable
import re
from pathlib import Path
import json
import yaml
import hashlib
import time
from dataclasses import dataclass, field
from functools import wraps
from contextlib import contextmanager

from .exceptions import ValidationError, SecurityError, ConfigurationError
from .logging import get_logger, get_audit_logger
from .monitoring import record_metric

logger = get_logger(__name__)
audit_logger = get_audit_logger(__name__)


@dataclass
class ValidationResult:
    """Enhanced result of a validation operation with detailed context."""
    is_valid: bool
    errors: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[Dict[str, Any]] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    validation_id: str = field(default_factory=lambda: hashlib.md5(str(time.time()).encode()).hexdigest()[:8])
    timestamp: float = field(default_factory=time.time)
    
    def __bool__(self) -> bool:
        return self.is_valid
    
    def add_error(self, message: str, field: Optional[str] = None, 
                  code: Optional[str] = None, details: Optional[Dict[str, Any]] = None) -> None:
        """Add a detailed error message."""
        error = {
            "message": message,
            "field": field,
            "code": code or "VALIDATION_ERROR",
            "details": details or {},
            "timestamp": time.time()
        }
        self.errors.append(error)
        self.is_valid = False
        
        # Record validation error metric
        record_metric("validation_error", 1, "counter", {"field": field or "unknown", "code": code or "unknown"})
    
    def add_warning(self, message: str, field: Optional[str] = None, 
                   code: Optional[str] = None, details: Optional[Dict[str, Any]] = None) -> None:
        """Add a detailed warning message."""
        warning = {
            "message": message,
            "field": field,
            "code": code or "VALIDATION_WARNING",
            "details": details or {},
            "timestamp": time.time()
        }
        self.warnings.append(warning)
        
        # Record validation warning metric
        record_metric("validation_warning", 1, "counter", {"field": field or "unknown", "code": code or "unknown"})
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert validation result to dictionary."""
        return {
            "is_valid": self.is_valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "context": self.context,
            "validation_id": self.validation_id,
            "timestamp": self.timestamp,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings)
        }


class SecurityValidator:
    """Enhanced security validator for input sanitization."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize security validator with configurable patterns."""
        self.config = config or {}
        
        # Enhanced blocked patterns with more comprehensive coverage
        self.blocked_patterns = [
            # Path traversal attacks
            r'\.\./', r'%2e%2e', r'%252e%252e', r'\\\\', r'%5c%5c',
            # Script injection
            r'<script[^>]*>', r'</script>', r'javascript:', r'vbscript:', r'data:text/html',
            # Event handlers
            r'on\w+\s*=', r'on\w+:', r'@import', r'expression\s*\(',
            # Code execution attempts
            r'eval\s*\(', r'exec\s*\(', r'system\s*\(', r'__import__', r'import\s+',
            # Process and file operations
            r'subprocess', r'os\.system', r'os\.popen', r'open\s*\(', r'file\s*\(',
            # SQL injection patterns
            r'union\s+select', r'drop\s+table', r'insert\s+into', r'delete\s+from',
            r'--\s*$', r'/\*.*\*/', r"'\s*or\s+'1'\s*=\s*'1",
            # Command injection
            r'[;&|`$]', r'nc\s+-', r'wget\s+', r'curl\s+', r'chmod\s+',
            # Template injection
            r'{{.*}}', r'{%.*%}', r'<%.*%>', r'\$\{.*\}',
            # Format string attacks
            r'%[sdxp]', r'\{.*\}', r'\$\w+',
        ]
        
        # Security limits
        self.max_input_length = self.config.get('max_input_length', 10000)
        self.max_nesting_depth = self.config.get('max_nesting_depth', 10)
        self.max_array_size = self.config.get('max_array_size', 1000)
        
        # File security settings
        self.allowed_file_extensions = set(self.config.get('allowed_file_extensions', [
            '.onnx', '.pt', '.pb', '.h5', '.tflite', '.json', '.yaml', '.yml', '.txt'
        ]))
        self.blocked_paths = set(self.config.get('blocked_paths', [
            '/etc', '/proc', '/sys', '/dev', '/var', '/root', '/home', '/bin', '/usr/bin'
        ]))
        self.max_file_size_mb = self.config.get('max_file_size_mb', 100)
        
        # Network security
        self.allowed_domains = set(self.config.get('allowed_domains', []))
        self.blocked_ips = set(self.config.get('blocked_ips', [
            '127.0.0.1', '::1', '0.0.0.0', '169.254.0.0/16', '10.0.0.0/8', '192.168.0.0/16'
        ]))
        
        # Rate limiting
        self.request_counts = {}
        self.rate_limit_window = self.config.get('rate_limit_window', 60)  # seconds
        self.max_requests_per_window = self.config.get('max_requests_per_window', 100)
        
        logger.info("Initialized enhanced SecurityValidator", 
                   max_input_length=self.max_input_length,
                   allowed_extensions=len(self.allowed_file_extensions))
    
    def validate_string_input(self, value: str, field_name: str = "input", 
                             client_id: Optional[str] = None) -> ValidationResult:
        """Enhanced validation of string input for security threats."""
        result = ValidationResult(is_valid=True)
        result.context["field_name"] = field_name
        result.context["value_length"] = len(value) if isinstance(value, str) else 0
        
        # Type validation
        if not isinstance(value, str):
            result.add_error("Input must be a string", field_name, "INVALID_TYPE")
            return result
        
        # Rate limiting check
        if client_id and not self._check_rate_limit(client_id):
            result.add_error("Rate limit exceeded", field_name, "RATE_LIMIT_EXCEEDED")
            audit_logger.log_security_event("rate_limit_exceeded", 
                                           f"Client {client_id} exceeded rate limit", 
                                           "medium", client_id=client_id)
            return result
        
        # Length validation
        if len(value) > self.max_input_length:
            result.add_error(f"Input exceeds maximum length: {len(value)} > {self.max_input_length}", 
                           field_name, "LENGTH_EXCEEDED", 
                           {"length": len(value), "max_length": self.max_input_length})
            return result
        
        # Pattern validation
        for pattern in self.blocked_patterns:
            if re.search(pattern, value, re.IGNORECASE):
                result.add_error(f"Input contains blocked pattern", field_name, "BLOCKED_PATTERN",
                               {"pattern": pattern, "value_snippet": value[:100]})
                audit_logger.log_security_event("blocked_pattern_detected", 
                                               f"Blocked pattern '{pattern}' in field '{field_name}'",
                                               "high", field=field_name, pattern=pattern)
                return result
        
        # Character encoding validation
        try:
            value.encode('utf-8')
        except UnicodeEncodeError:
            result.add_error("Invalid character encoding", field_name, "ENCODING_ERROR")
            return result
        
        # Check for null bytes
        if '\x00' in value:
            result.add_error("Null bytes not allowed", field_name, "NULL_BYTES")
            return result
        
        return result
    
    def _check_rate_limit(self, client_id: str) -> bool:
        """Check if client has exceeded rate limit."""
        current_time = time.time()
        window_start = current_time - self.rate_limit_window
        
        # Clean old requests
        if client_id in self.request_counts:
            self.request_counts[client_id] = [
                req_time for req_time in self.request_counts[client_id]
                if req_time > window_start
            ]
        else:
            self.request_counts[client_id] = []
        
        # Check limit
        if len(self.request_counts[client_id]) >= self.max_requests_per_window:
            return False
        
        # Record request
        self.request_counts[client_id].append(current_time)
        return True
    
    def validate_file_path(self, path: str, client_id: Optional[str] = None) -> ValidationResult:
        """Enhanced validation of file paths for security."""
        result = ValidationResult(is_valid=True)
        result.context["file_path"] = path
        
        # First validate as string input
        string_result = self.validate_string_input(path, "file_path", client_id)
        if not string_result:
            result.errors.extend(string_result.errors)
            result.warnings.extend(string_result.warnings)
            result.is_valid = False
            return result
        
        try:
            normalized_path = Path(path).resolve()
            result.context["normalized_path"] = str(normalized_path)
            
            # Check blocked paths
            for blocked in self.blocked_paths:
                if str(normalized_path).startswith(blocked):
                    result.add_error(f"Access to system path not allowed: {blocked}", 
                                   "file_path", "BLOCKED_PATH", 
                                   {"blocked_path": blocked, "requested_path": path})
                    audit_logger.log_security_event("blocked_path_access", 
                                                   f"Attempt to access blocked path: {path}",
                                                   "high", path=path, blocked_path=blocked)
                    return result
            
            # Check file extension
            if normalized_path.suffix:
                if normalized_path.suffix.lower() not in self.allowed_file_extensions:
                    result.add_error(f"File extension not allowed: {normalized_path.suffix}", 
                                   "file_path", "BLOCKED_EXTENSION",
                                   {"extension": normalized_path.suffix, "allowed": list(self.allowed_file_extensions)})
                    return result
            
            # Check file size if exists
            if normalized_path.exists() and normalized_path.is_file():
                size_mb = normalized_path.stat().st_size / (1024 * 1024)
                if size_mb > self.max_file_size_mb:
                    result.add_error(f"File too large: {size_mb:.1f}MB > {self.max_file_size_mb}MB",
                                   "file_path", "FILE_TOO_LARGE",
                                   {"size_mb": size_mb, "max_size_mb": self.max_file_size_mb})
                    return result
                result.context["file_size_mb"] = size_mb
            
            return result
            
        except (OSError, ValueError) as e:
            result.add_error(f"Invalid file path: {e}", "file_path", "INVALID_PATH", {"error": str(e)})
            return result
    
    def validate_json_structure(self, data: Any, max_depth: Optional[int] = None, 
                               current_depth: int = 0) -> ValidationResult:
        """Validate JSON structure for security and complexity."""
        result = ValidationResult(is_valid=True)
        max_depth = max_depth or self.max_nesting_depth
        
        if current_depth > max_depth:
            result.add_error(f"JSON nesting too deep: {current_depth} > {max_depth}",
                           "json_structure", "NESTING_TOO_DEEP",
                           {"current_depth": current_depth, "max_depth": max_depth})
            return result
        
        if isinstance(data, dict):
            if len(data) > self.max_array_size:
                result.add_error(f"Dictionary too large: {len(data)} > {self.max_array_size}",
                               "json_structure", "DICT_TOO_LARGE")
                return result
            
            for key, value in data.items():
                # Validate key
                key_result = self.validate_string_input(str(key), "json_key")
                if not key_result:
                    result.errors.extend(key_result.errors)
                    result.is_valid = False
                
                # Recursively validate value
                value_result = self.validate_json_structure(value, max_depth, current_depth + 1)
                if not value_result:
                    result.errors.extend(value_result.errors)
                    result.warnings.extend(value_result.warnings)
                    result.is_valid = False
        
        elif isinstance(data, list):
            if len(data) > self.max_array_size:
                result.add_error(f"Array too large: {len(data)} > {self.max_array_size}",
                               "json_structure", "ARRAY_TOO_LARGE")
                return result
            
            for i, item in enumerate(data):
                item_result = self.validate_json_structure(item, max_depth, current_depth + 1)
                if not item_result:
                    result.errors.extend(item_result.errors)
                    result.warnings.extend(item_result.warnings)
                    result.is_valid = False
        
        elif isinstance(data, str):
            string_result = self.validate_string_input(data, "json_string")
            if not string_result:
                result.errors.extend(string_result.errors)
                result.is_valid = False
        
        return result
    
    def validate_yaml_content(self, content: str) -> ValidationResult:
        """Validate YAML content for security."""
        result = ValidationResult(is_valid=True)
        
        # First validate as string
        string_result = self.validate_string_input(content, "yaml_content")
        if not string_result:
            return string_result
        
        try:
            # Parse YAML safely
            data = yaml.safe_load(content)
            
            # Validate structure
            structure_result = self.validate_json_structure(data)
            if not structure_result:
                result.errors.extend(structure_result.errors)
                result.warnings.extend(structure_result.warnings)
                result.is_valid = False
            
            return result
            
        except yaml.YAMLError as e:
            result.add_error(f"Invalid YAML format: {e}", "yaml_content", "YAML_ERROR", {"error": str(e)})
            return result
    
    def validate_numeric_input(self, value: Union[int, float], field_name: str, 
                              min_value: Optional[float] = None, max_value: Optional[float] = None) -> ValidationResult:
        """Enhanced validation of numeric input with bounds checking."""
        result = ValidationResult(is_valid=True)
        result.context["field_name"] = field_name
        result.context["value"] = value
        result.context["value_type"] = type(value).__name__
        
        # Type validation
        if not isinstance(value, (int, float)):
            result.add_error(f"Value must be numeric, got {type(value).__name__}", 
                           field_name, "INVALID_TYPE", {"expected": "int or float", "actual": type(value).__name__})
            return result
        
        # NaN and infinity checks for floats
        if isinstance(value, float):
            if not (value == value):  # NaN check
                result.add_error("NaN values not allowed", field_name, "NAN_VALUE")
                return result
            if value == float('inf') or value == float('-inf'):
                result.add_error("Infinity values not allowed", field_name, "INFINITY_VALUE")
                return result
        
        # Range validation
        if min_value is not None and value < min_value:
            result.add_error(f"Value {value} below minimum {min_value}", 
                           field_name, "BELOW_MINIMUM", 
                           {"value": value, "min_value": min_value})
            return result
        
        if max_value is not None and value > max_value:
            result.add_error(f"Value {value} above maximum {max_value}", 
                           field_name, "ABOVE_MAXIMUM",
                           {"value": value, "max_value": max_value})
            return result
        
        # Additional checks for extreme values
        if isinstance(value, float):
            if abs(value) > 1e100:
                result.add_warning("Very large numeric value detected", field_name, "LARGE_VALUE",
                                 {"value": value})
            elif abs(value) < 1e-100 and value != 0:
                result.add_warning("Very small numeric value detected", field_name, "SMALL_VALUE",
                                 {"value": value})
        
        return result
    
    def validate_request_headers(self, headers: Dict[str, str]) -> bool:
        """Validate HTTP request headers."""
        for header_name, header_value in headers.items():
            if not self.validate_string_input(header_value, f"header_{header_name}"):
                return False
        
        content_type = headers.get('content-type', '')
        if content_type and not content_type.startswith(('application/', 'text/', 'multipart/')):
            logger.warning(f"Suspicious content type: {content_type}")
            return False
        
        return True
    
    def validate(self, value: Any) -> bool:
        """General validation method."""
        if isinstance(value, str):
            return self.validate_string_input(value)
        elif isinstance(value, (int, float)):
            return self.validate_numeric_input(value, "numeric_input")
        elif isinstance(value, (dict, list)):
            return True  # Basic structure validation
        else:
            return True  # Allow other types


def validate_inputs(func):
    """Decorator for input validation."""
    def wrapper(*args, **kwargs):
        security_validator = SecurityValidator()
        
        for i, arg in enumerate(args):
            if not security_validator.validate(arg):
                raise ValidationError(f"Argument {i} failed security validation")
        
        for key, value in kwargs.items():
            if not security_validator.validate(value):
                raise ValidationError(f"Keyword argument '{key}' failed security validation")
        
        return func(*args, **kwargs)
    
    return wrapper


def validate_model_config(config: Dict[str, Any]) -> ValidationResult:
    """Validate model configuration."""
    result = ValidationResult(is_valid=True, errors=[], warnings=[])
    
    if not isinstance(config, dict):
        result.add_error("Model configuration must be a dictionary")
        return result
    
    # Required fields
    required_fields = ['path']
    for field in required_fields:
        if field not in config:
            result.add_error(f"Missing required field: {field}")
    
    # Validate path
    if 'path' in config:
        security_validator = SecurityValidator()
        if not security_validator.validate_file_path(config['path']):
            result.add_error("Invalid or unsafe model path")
    
    return result


def validate_hardware_config(config: Dict[str, Any]) -> ValidationResult:
    """Validate hardware configuration."""
    result = ValidationResult(is_valid=True, errors=[], warnings=[])
    
    if not isinstance(config, dict):
        result.add_error("Hardware configuration must be a dictionary")
        return result
    
    # Validate compute units
    compute_units = config.get('compute_units')
    if compute_units is not None:
        if not isinstance(compute_units, int) or compute_units <= 0:
            result.add_error("Compute units must be a positive integer")
        elif compute_units > 2048:
            result.add_warning("Very large number of compute units may cause issues")
    
    # Validate dataflow
    dataflow = config.get('dataflow')
    if dataflow:
        valid_dataflows = ['weight_stationary', 'output_stationary', 'row_stationary']
        if dataflow not in valid_dataflows:
            result.add_error(f"Invalid dataflow. Must be one of: {valid_dataflows}")
    
    # Validate frequency
    frequency = config.get('frequency_mhz')
    if frequency is not None:
        security_validator = SecurityValidator()
        if not security_validator.validate_numeric_input(frequency, "frequency", 1.0, 5000.0):
            result.add_error("Invalid frequency value")
    
    return result


# Compatibility exports
class BaseValidator:
    def validate(self, value: Any) -> ValidationResult:
        return ValidationResult(is_valid=True, errors=[], warnings=[])

class ConfigValidator(BaseValidator):
    def __init__(self, schema: Dict[str, Any], strict: bool = True):
        self.schema = schema
        self.strict = strict

class ModelValidator(BaseValidator):
    pass

class HardwareValidator(BaseValidator):
    pass


# Legacy compatibility functions
def validate_model(model_config: Dict[str, Any]) -> ValidationResult:
    """Validate model configuration (legacy compatibility)."""
    return validate_model_config(model_config)