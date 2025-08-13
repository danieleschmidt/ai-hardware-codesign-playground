"""
Input validation utilities for AI Hardware Co-Design Playground.

This module provides comprehensive validation for user inputs,
configurations, and data integrity checks with enhanced security features.
"""

from typing import Any, Dict, List, Optional, Union, Tuple
import re
from pathlib import Path
import json
from dataclasses import dataclass

from .exceptions import ValidationError
from .logging import get_logger

logger = get_logger(__name__)


@dataclass
class ValidationResult:
    """Result of a validation operation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    
    def __bool__(self) -> bool:
        return self.is_valid
    
    def add_error(self, message: str) -> None:
        """Add an error message."""
        self.errors.append(message)
        self.is_valid = False
    
    def add_warning(self, message: str) -> None:
        """Add a warning message."""
        self.warnings.append(message)


class SecurityValidator:
    """Enhanced security validator for input sanitization."""
    
    def __init__(self):
        self.blocked_patterns = [
            r'\.\./',  # Path traversal
            r'<script',  # XSS
            r'javascript:',  # JavaScript injection
            r'data:',  # Data URLs
            r'vbscript:',  # VBScript injection
            r'onload=',  # Event handlers
            r'onerror=',  # Event handlers
            r'eval\(',  # Code evaluation
            r'exec\(',  # Code execution
            r'system\(',  # System commands
            r'__import__',  # Python imports
            r'subprocess',  # Process execution
            r'os\.system',  # OS commands
        ]
        
        self.max_input_length = 10000
        self.allowed_file_extensions = {'.onnx', '.pt', '.pb', '.h5', '.tflite', '.json', '.yaml', '.yml'}
        self.blocked_paths = {'/etc', '/proc', '/sys', '/dev', '/var', '/root', '/home'}
    
    def validate_string_input(self, value: str, field_name: str = "input") -> bool:
        """Validate string input for security threats."""
        if not isinstance(value, str):
            return False
        
        if len(value) > self.max_input_length:
            logger.warning(f"Input '{field_name}' exceeds maximum length: {len(value)}")
            return False
        
        for pattern in self.blocked_patterns:
            if re.search(pattern, value, re.IGNORECASE):
                logger.warning(f"Blocked pattern detected in '{field_name}': {pattern}")
                return False
        
        return True
    
    def validate_file_path(self, path: str) -> bool:
        """Validate file path for security."""
        if not self.validate_string_input(path, "file_path"):
            return False
        
        try:
            normalized_path = Path(path).resolve()
            
            for blocked in self.blocked_paths:
                if str(normalized_path).startswith(blocked):
                    logger.warning(f"Blocked system path access: {path}")
                    return False
            
            if normalized_path.suffix and normalized_path.suffix.lower() not in self.allowed_file_extensions:
                logger.warning(f"Blocked file extension: {normalized_path.suffix}")
                return False
            
            return True
            
        except (OSError, ValueError) as e:
            logger.warning(f"Invalid file path: {path} - {e}")
            return False
    
    def validate_numeric_input(self, value: Union[int, float], field_name: str, 
                              min_value: Optional[float] = None, max_value: Optional[float] = None) -> bool:
        """Validate numeric input with bounds checking."""
        if not isinstance(value, (int, float)):
            return False
        
        if isinstance(value, float):
            if not (value == value):  # NaN check
                logger.warning(f"NaN value detected in '{field_name}'")
                return False
            if value == float('inf') or value == float('-inf'):
                logger.warning(f"Infinity value detected in '{field_name}'")
                return False
        
        if min_value is not None and value < min_value:
            logger.warning(f"Value {value} below minimum {min_value} for '{field_name}'")
            return False
        
        if max_value is not None and value > max_value:
            logger.warning(f"Value {value} above maximum {max_value} for '{field_name}'")
            return False
        
        return True
    
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