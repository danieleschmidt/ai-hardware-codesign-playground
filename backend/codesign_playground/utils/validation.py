"""
Input validation utilities for AI Hardware Co-Design Playground.

This module provides comprehensive validation for user inputs,
configurations, and data integrity checks.
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


class BaseValidator:
    """Base class for all validators."""
    
    def __init__(self, strict: bool = True):
        """
        Initialize validator.
        
        Args:
            strict: If True, warnings are treated as errors
        """
        self.strict = strict
    
    def validate(self, value: Any) -> ValidationResult:
        """
        Validate a value.
        
        Args:
            value: Value to validate
            
        Returns:
            ValidationResult with validation status and messages
        """
        result = ValidationResult(is_valid=True, errors=[], warnings=[])
        self._validate_impl(value, result)
        
        # In strict mode, warnings become errors
        if self.strict and result.warnings:
            result.errors.extend(result.warnings)
            result.warnings.clear()
            result.is_valid = False
        
        return result
    
    def _validate_impl(self, value: Any, result: ValidationResult) -> None:
        """Implementation-specific validation logic."""
        raise NotImplementedError


class ConfigValidator(BaseValidator):
    """Validator for configuration dictionaries."""
    
    def __init__(self, schema: Dict[str, Dict[str, Any]], strict: bool = True):
        """
        Initialize configuration validator.
        
        Args:
            schema: Configuration schema with field definitions
            strict: If True, unknown fields cause errors
        """
        super().__init__(strict)
        self.schema = schema
    
    def _validate_impl(self, config: Dict[str, Any], result: ValidationResult) -> None:
        """Validate configuration against schema."""
        if not isinstance(config, dict):
            result.add_error("Configuration must be a dictionary")
            return
        
        # Check required fields
        for field_name, field_schema in self.schema.items():
            if field_schema.get("required", False) and field_name not in config:
                result.add_error(f"Required field '{field_name}' is missing")
                continue
            
            if field_name not in config:
                continue
            
            value = config[field_name]
            self._validate_field(field_name, value, field_schema, result)
        
        # Check for unknown fields in strict mode
        if self.strict:
            for field_name in config:
                if field_name not in self.schema:
                    result.add_error(f"Unknown field '{field_name}'")
    
    def _validate_field(
        self, 
        field_name: str, 
        value: Any, 
        field_schema: Dict[str, Any], 
        result: ValidationResult
    ) -> None:
        """Validate a single field."""
        # Type validation
        expected_type = field_schema.get("type")
        if expected_type and not isinstance(value, expected_type):
            result.add_error(
                f"Field '{field_name}' must be of type {expected_type.__name__}, "
                f"got {type(value).__name__}"
            )
            return
        
        # Range validation for numeric types
        if isinstance(value, (int, float)):
            min_val = field_schema.get("min")
            max_val = field_schema.get("max")
            
            if min_val is not None and value < min_val:
                result.add_error(f"Field '{field_name}' must be >= {min_val}, got {value}")
            
            if max_val is not None and value > max_val:
                result.add_error(f"Field '{field_name}' must be <= {max_val}, got {value}")
        
        # Length validation for strings and lists
        if isinstance(value, (str, list)):
            min_len = field_schema.get("min_length")
            max_len = field_schema.get("max_length")
            
            if min_len is not None and len(value) < min_len:
                result.add_error(
                    f"Field '{field_name}' must have length >= {min_len}, got {len(value)}"
                )
            
            if max_len is not None and len(value) > max_len:
                result.add_error(
                    f"Field '{field_name}' must have length <= {max_len}, got {len(value)}"
                )
        
        # Choice validation
        choices = field_schema.get("choices")
        if choices and value not in choices:
            result.add_error(
                f"Field '{field_name}' must be one of {choices}, got '{value}'"
            )
        
        # Pattern validation for strings
        if isinstance(value, str):
            pattern = field_schema.get("pattern")
            if pattern and not re.match(pattern, value):
                result.add_error(
                    f"Field '{field_name}' does not match required pattern '{pattern}'"
                )
        
        # Custom validation function
        custom_validator = field_schema.get("validator")
        if custom_validator and callable(custom_validator):
            try:
                if not custom_validator(value):
                    result.add_error(f"Field '{field_name}' failed custom validation")
            except Exception as e:
                result.add_error(f"Custom validation failed for '{field_name}': {e}")


class ModelValidator(BaseValidator):
    """Validator for model configurations and paths."""
    
    SUPPORTED_FORMATS = [".onnx", ".pb", ".pt", ".pth", ".h5", ".tflite"]
    SUPPORTED_FRAMEWORKS = ["pytorch", "tensorflow", "onnx", "auto"]
    
    def _validate_impl(self, model_config: Dict[str, Any], result: ValidationResult) -> None:
        """Validate model configuration."""
        if not isinstance(model_config, dict):
            result.add_error("Model configuration must be a dictionary")
            return
        
        # Validate model path
        model_path = model_config.get("path")
        if not model_path:
            result.add_error("Model path is required")
        else:
            self._validate_model_path(model_path, result)
        
        # Validate framework
        framework = model_config.get("framework", "auto")
        if framework not in self.SUPPORTED_FRAMEWORKS:
            result.add_error(
                f"Unsupported framework '{framework}'. "
                f"Supported: {self.SUPPORTED_FRAMEWORKS}"
            )
        
        # Validate input shapes
        input_shapes = model_config.get("input_shapes")
        if input_shapes:
            self._validate_input_shapes(input_shapes, result)
    
    def _validate_model_path(self, path: str, result: ValidationResult) -> None:
        """Validate model file path."""
        try:
            model_path = Path(path)
            
            # Check if path exists (warning if not, since it might be created later)
            if not model_path.exists():
                result.add_warning(f"Model file does not exist: {path}")
            
            # Check file extension
            if model_path.suffix.lower() not in self.SUPPORTED_FORMATS:
                result.add_warning(
                    f"Unsupported model format '{model_path.suffix}'. "
                    f"Supported: {self.SUPPORTED_FORMATS}"
                )
            
            # Check file size if exists
            if model_path.exists() and model_path.stat().st_size == 0:
                result.add_error("Model file is empty")
            
        except Exception as e:
            result.add_error(f"Invalid model path '{path}': {e}")
    
    def _validate_input_shapes(self, input_shapes: Dict[str, List[int]], result: ValidationResult) -> None:
        """Validate input tensor shapes."""
        if not isinstance(input_shapes, dict):
            result.add_error("Input shapes must be a dictionary")
            return
        
        for input_name, shape in input_shapes.items():
            if not isinstance(shape, list):
                result.add_error(f"Shape for input '{input_name}' must be a list")
                continue
            
            if not all(isinstance(dim, int) and dim > 0 for dim in shape):
                result.add_error(
                    f"All dimensions in shape for '{input_name}' must be positive integers"
                )
            
            if len(shape) < 1 or len(shape) > 6:
                result.add_warning(
                    f"Unusual number of dimensions ({len(shape)}) for input '{input_name}'"
                )


class HardwareValidator(BaseValidator):
    """Validator for hardware accelerator configurations."""
    
    SUPPORTED_DATAFLOWS = ["weight_stationary", "output_stationary", "row_stationary"]
    SUPPORTED_PRECISIONS = ["int8", "int16", "fp16", "fp32", "mixed"]
    MEMORY_TYPES = ["sram_32kb", "sram_64kb", "sram_128kb", "sram_256kb", "dram", "hbm"]
    
    def _validate_impl(self, hw_config: Dict[str, Any], result: ValidationResult) -> None:
        """Validate hardware configuration."""
        if not isinstance(hw_config, dict):
            result.add_error("Hardware configuration must be a dictionary")
            return
        
        self._validate_compute_units(hw_config.get("compute_units"), result)
        self._validate_dataflow(hw_config.get("dataflow"), result)
        self._validate_memory_hierarchy(hw_config.get("memory_hierarchy"), result)
        self._validate_frequency(hw_config.get("frequency_mhz"), result)
        self._validate_precision(hw_config.get("precision"), result)
        self._validate_power_budget(hw_config.get("power_budget_w"), result)
        self._validate_area_budget(hw_config.get("area_budget_mm2"), result)
    
    def _validate_compute_units(self, compute_units: Any, result: ValidationResult) -> None:
        """Validate compute units configuration."""
        if compute_units is None:
            return
        
        if not isinstance(compute_units, int):
            result.add_error("Compute units must be an integer")
            return
        
        if compute_units <= 0:
            result.add_error("Compute units must be positive")
        elif compute_units > 1024:
            result.add_warning(f"Very large number of compute units: {compute_units}")
        
        # Check if power of 2 (common for efficiency)
        if compute_units & (compute_units - 1) != 0:
            result.add_warning(f"Compute units ({compute_units}) is not a power of 2")
    
    def _validate_dataflow(self, dataflow: Any, result: ValidationResult) -> None:
        """Validate dataflow configuration."""
        if dataflow is None:
            return
        
        if not isinstance(dataflow, str):
            result.add_error("Dataflow must be a string")
            return
        
        if dataflow not in self.SUPPORTED_DATAFLOWS:
            result.add_error(
                f"Unsupported dataflow '{dataflow}'. "
                f"Supported: {self.SUPPORTED_DATAFLOWS}"
            )
    
    def _validate_memory_hierarchy(self, memory_hierarchy: Any, result: ValidationResult) -> None:
        """Validate memory hierarchy configuration."""
        if memory_hierarchy is None:
            return
        
        if not isinstance(memory_hierarchy, list):
            result.add_error("Memory hierarchy must be a list")
            return
        
        if len(memory_hierarchy) == 0:
            result.add_error("Memory hierarchy cannot be empty")
            return
        
        for i, memory_type in enumerate(memory_hierarchy):
            if not isinstance(memory_type, str):
                result.add_error(f"Memory type at level {i} must be a string")
                continue
            
            if memory_type not in self.MEMORY_TYPES:
                result.add_error(
                    f"Unsupported memory type '{memory_type}' at level {i}. "
                    f"Supported: {self.MEMORY_TYPES}"
                )
        
        # Validate hierarchy ordering (smaller/faster memory should come first)
        memory_sizes = {
            "sram_32kb": 32, "sram_64kb": 64, "sram_128kb": 128, 
            "sram_256kb": 256, "dram": 1000000, "hbm": 10000000
        }
        
        for i in range(len(memory_hierarchy) - 1):
            current = memory_hierarchy[i]
            next_mem = memory_hierarchy[i + 1]
            
            if current in memory_sizes and next_mem in memory_sizes:
                if memory_sizes[current] >= memory_sizes[next_mem]:
                    result.add_warning(
                        f"Memory hierarchy may be incorrect: {current} -> {next_mem}"
                    )
    
    def _validate_frequency(self, frequency: Any, result: ValidationResult) -> None:
        """Validate operating frequency."""
        if frequency is None:
            return
        
        if not isinstance(frequency, (int, float)):
            result.add_error("Frequency must be a number")
            return
        
        if frequency <= 0:
            result.add_error("Frequency must be positive")
        elif frequency < 10:
            result.add_warning(f"Very low frequency: {frequency} MHz")
        elif frequency > 2000:
            result.add_warning(f"Very high frequency: {frequency} MHz")
    
    def _validate_precision(self, precision: Any, result: ValidationResult) -> None:
        """Validate numerical precision."""
        if precision is None:
            return
        
        if not isinstance(precision, str):
            result.add_error("Precision must be a string")
            return
        
        if precision not in self.SUPPORTED_PRECISIONS:
            result.add_error(
                f"Unsupported precision '{precision}'. "
                f"Supported: {self.SUPPORTED_PRECISIONS}"
            )
    
    def _validate_power_budget(self, power_budget: Any, result: ValidationResult) -> None:
        """Validate power budget."""
        if power_budget is None:
            return
        
        if not isinstance(power_budget, (int, float)):
            result.add_error("Power budget must be a number")
            return
        
        if power_budget <= 0:
            result.add_error("Power budget must be positive")
        elif power_budget > 100:
            result.add_warning(f"Very high power budget: {power_budget} W")
    
    def _validate_area_budget(self, area_budget: Any, result: ValidationResult) -> None:
        """Validate area budget."""
        if area_budget is None:
            return
        
        if not isinstance(area_budget, (int, float)):
            result.add_error("Area budget must be a number")
            return
        
        if area_budget <= 0:
            result.add_error("Area budget must be positive")
        elif area_budget > 1000:
            result.add_warning(f"Very large area budget: {area_budget} mm²")


class OptimizationValidator(BaseValidator):
    """Validator for optimization configurations."""
    
    SUPPORTED_STRATEGIES = ["performance", "power", "balanced", "area"]
    SUPPORTED_OBJECTIVES = ["latency", "power", "area", "throughput", "efficiency", "accuracy"]
    
    def _validate_impl(self, opt_config: Dict[str, Any], result: ValidationResult) -> None:
        """Validate optimization configuration."""
        if not isinstance(opt_config, dict):
            result.add_error("Optimization configuration must be a dictionary")
            return
        
        # Validate strategy
        strategy = opt_config.get("strategy")
        if strategy and strategy not in self.SUPPORTED_STRATEGIES:
            result.add_error(
                f"Unsupported optimization strategy '{strategy}'. "
                f"Supported: {self.SUPPORTED_STRATEGIES}"
            )
        
        # Validate objectives
        objectives = opt_config.get("objectives", [])
        if not isinstance(objectives, list):
            result.add_error("Objectives must be a list")
        else:
            for obj in objectives:
                if obj not in self.SUPPORTED_OBJECTIVES:
                    result.add_error(
                        f"Unsupported objective '{obj}'. "
                        f"Supported: {self.SUPPORTED_OBJECTIVES}"
                    )
        
        # Validate target metrics
        target_fps = opt_config.get("target_fps")
        if target_fps is not None:
            if not isinstance(target_fps, (int, float)) or target_fps <= 0:
                result.add_error("Target FPS must be a positive number")
        
        power_budget = opt_config.get("power_budget")
        if power_budget is not None:
            if not isinstance(power_budget, (int, float)) or power_budget <= 0:
                result.add_error("Power budget must be a positive number")
        
        # Validate iteration count
        iterations = opt_config.get("iterations", 10)
        if not isinstance(iterations, int) or iterations <= 0:
            result.add_error("Iterations must be a positive integer")
        elif iterations > 1000:
            result.add_warning(f"Very high iteration count: {iterations}")


def validate_file_path(path: str, must_exist: bool = False) -> ValidationResult:
    """
    Validate a file path.
    
    Args:
        path: File path to validate
        must_exist: Whether the file must exist
        
    Returns:
        ValidationResult
    """
    result = ValidationResult(is_valid=True, errors=[], warnings=[])
    
    try:
        file_path = Path(path)
        
        # Security check - no path traversal
        if ".." in str(file_path):
            result.add_error("Path traversal not allowed")
        
        # Check if path exists
        if must_exist and not file_path.exists():
            result.add_error(f"File does not exist: {path}")
        
        # Check permissions
        if file_path.exists():
            if not file_path.is_file():
                result.add_error(f"Path is not a file: {path}")
            elif not file_path.stat().st_size > 0:
                result.add_warning("File is empty")
        
    except Exception as e:
        result.add_error(f"Invalid file path: {e}")
    
    return result


def validate_json_config(config_str: str, schema: Optional[Dict[str, Any]] = None) -> ValidationResult:
    """
    Validate JSON configuration string.
    
    Args:
        config_str: JSON configuration string
        schema: Optional schema for validation
        
    Returns:
        ValidationResult
    """
    result = ValidationResult(is_valid=True, errors=[], warnings=[])
    
    try:
        config = json.loads(config_str)
        
        if schema:
            validator = ConfigValidator(schema)
            schema_result = validator.validate(config)
            result.errors.extend(schema_result.errors)
            result.warnings.extend(schema_result.warnings)
            result.is_valid = result.is_valid and schema_result.is_valid
        
    except json.JSONDecodeError as e:
        result.add_error(f"Invalid JSON: {e}")
    except Exception as e:
        result.add_error(f"Configuration validation failed: {e}")
    
    return result


class SecurityValidator(BaseValidator):
    """Validator for security-related configurations and inputs."""
    
    DANGEROUS_PATTERNS = [
        r'eval\s*\(',
        r'exec\s*\(',
        r'__import__',
        r'open\s*\(',
        r'file\s*\(',
        r'input\s*\(',
        r'raw_input\s*\(',
    ]
    
    def validate_user_input(self, user_input: str, field_name: str = "input") -> ValidationResult:
        """
        Validate user input for security threats.
        
        Args:
            user_input: User provided input string
            field_name: Name of the input field
            
        Returns:
            ValidationResult with security validation results
        """
        result = ValidationResult(is_valid=True, errors=[], warnings=[])
        
        # Check for dangerous patterns
        for pattern in self.DANGEROUS_PATTERNS:
            if re.search(pattern, user_input, re.IGNORECASE):
                result.add_error(f"Potentially dangerous pattern detected in {field_name}")
        
        # Check for SQL injection patterns
        sql_patterns = [
            r'union\s+select',
            r'drop\s+table',
            r'delete\s+from',
            r'insert\s+into',
            r'update\s+.*set',
            r"'\s*or\s*'.*'\s*=\s*'",
            r'"\s*or\s*".*"\s*=\s*"',
        ]
        
        for pattern in sql_patterns:
            if re.search(pattern, user_input, re.IGNORECASE):
                result.add_error(f"Potential SQL injection pattern in {field_name}")
        
        # Check for path traversal
        if ".." in user_input or user_input.startswith("/"):
            result.add_error(f"Potential path traversal in {field_name}")
        
        # Check input length
        if len(user_input) > 10000:
            result.add_error(f"Input too long: {len(user_input)} characters (max: 10000)")
        
        return result
    
    def validate_file_path_security(self, file_path: str, allowed_extensions: Optional[List[str]] = None) -> ValidationResult:
        """
        Validate file path for security issues.
        
        Args:
            file_path: File path to validate
            allowed_extensions: List of allowed file extensions
            
        Returns:
            ValidationResult with security validation results
        """
        result = ValidationResult(is_valid=True, errors=[], warnings=[])
        path = Path(file_path)
        
        # Check for path traversal
        try:
            resolved_path = path.resolve()
            if ".." in str(resolved_path):
                result.add_error("Path traversal detected in file path")
        except Exception as e:
            result.add_error(f"Invalid file path: {e}")
        
        # Check file extension
        if allowed_extensions:
            if path.suffix.lower() not in [ext.lower() for ext in allowed_extensions]:
                result.add_error(f"File extension {path.suffix} not allowed")
        
        # Check for suspicious filenames
        suspicious_names = ["config", "passwd", "shadow", ".env", ".ssh"]
        if any(suspicious in path.name.lower() for suspicious in suspicious_names):
            result.add_warning("Potentially sensitive filename detected")
        
        return result


class EnhancedModelValidator(ModelValidator):
    """Enhanced model validator with additional security and robustness checks."""
    
    MAX_MODEL_SIZE_MB = 2048  # 2GB limit
    
    def _validate_impl(self, model_config: Dict[str, Any], result: ValidationResult) -> None:
        """Enhanced model validation with security checks."""
        super()._validate_impl(model_config, result)
        
        # Additional security validation
        security_validator = SecurityValidator()
        
        model_path = model_config.get("path", "")
        if model_path:
            sec_result = security_validator.validate_file_path_security(
                model_path, self.SUPPORTED_FORMATS
            )
            result.errors.extend(sec_result.errors)
            result.warnings.extend(sec_result.warnings)
            if not sec_result.is_valid:
                result.is_valid = False
        
        # Validate model size
        if model_path and Path(model_path).exists():
            size_mb = Path(model_path).stat().st_size / (1024 * 1024)
            if size_mb > self.MAX_MODEL_SIZE_MB:
                result.add_error(f"Model file too large: {size_mb:.1f}MB (max: {self.MAX_MODEL_SIZE_MB}MB)")
            elif size_mb > 500:
                result.add_warning(f"Large model file: {size_mb:.1f}MB may cause performance issues")


class EnhancedHardwareValidator(HardwareValidator):
    """Enhanced hardware validator with comprehensive checks."""
    
    def _validate_impl(self, hw_config: Dict[str, Any], result: ValidationResult) -> None:
        """Enhanced hardware validation with additional checks."""
        super()._validate_impl(hw_config, result)
        
        # Additional validation for resource constraints
        self._validate_resource_constraints(hw_config, result)
        self._validate_compatibility(hw_config, result)
    
    def _validate_resource_constraints(self, hw_config: Dict[str, Any], result: ValidationResult) -> None:
        """Validate resource constraints and feasibility."""
        compute_units = hw_config.get("compute_units", 64)
        frequency = hw_config.get("frequency_mhz", 200)
        power_budget = hw_config.get("power_budget_w", 5.0)
        
        # Rough power estimation check
        if isinstance(compute_units, int) and isinstance(frequency, (int, float)) and isinstance(power_budget, (int, float)):
            estimated_power = compute_units * frequency * 0.001  # Rough estimate
            if estimated_power > power_budget:
                result.add_warning(
                    f"Estimated power ({estimated_power:.2f}W) may exceed budget ({power_budget}W)"
                )
    
    def _validate_compatibility(self, hw_config: Dict[str, Any], result: ValidationResult) -> None:
        """Validate compatibility between configuration parameters."""
        dataflow = hw_config.get("dataflow")
        precision = hw_config.get("precision")
        
        # Check dataflow-precision compatibility
        if dataflow == "weight_stationary" and precision in ["fp32", "fp16"]:
            result.add_warning(
                "Weight stationary dataflow with floating point may require more memory"
            )


# Validation decorators for robust error handling
def validate_model_input(func):
    """Decorator to validate model inputs with comprehensive checks."""
    def wrapper(*args, **kwargs):
        # Extract model path and input shape from arguments
        model_validator = EnhancedModelValidator()
        
        if 'model_path' in kwargs:
            model_config = {'path': kwargs['model_path']}
            model_result = model_validator.validate(model_config)
            if not model_result.is_valid:
                raise ValidationError(f"Model validation failed: {'; '.join(model_result.errors)}")
        
        if 'input_shape' in kwargs:
            # Validate input shape format and values
            input_shape = kwargs['input_shape']
            if isinstance(input_shape, str):
                try:
                    shape_dims = tuple(map(int, input_shape.split(',')))
                    if any(dim <= 0 for dim in shape_dims):
                        raise ValidationError("All input dimensions must be positive")
                    if len(shape_dims) > 5:
                        raise ValidationError("Too many dimensions in input shape")
                except ValueError as e:
                    raise ValidationError(f"Invalid input shape format: {e}")
        
        return func(*args, **kwargs)
    
    return wrapper


def validate_hardware_config(func):
    """Decorator to validate hardware configuration with enhanced checks."""
    def wrapper(*args, **kwargs):
        hardware_validator = EnhancedHardwareValidator()
        
        # Look for hardware configuration in kwargs
        config_keys = ['config', 'hardware_config', 'accelerator_config']
        for key in config_keys:
            if key in kwargs and isinstance(kwargs[key], dict):
                result = hardware_validator.validate(kwargs[key])
                if not result.is_valid:
                    raise ValidationError(f"Hardware config validation failed: {'; '.join(result.errors)}")
        
        return func(*args, **kwargs)
    
    return wrapper


def validate_security(func):
    """Decorator to validate security of inputs."""
    def wrapper(*args, **kwargs):
        security_validator = SecurityValidator()
        
        # Validate string inputs
        for key, value in kwargs.items():
            if isinstance(value, str) and len(value) > 0:
                result = security_validator.validate_user_input(value, key)
                if not result.is_valid:
                    raise ValidationError(f"Security validation failed for {key}: {'; '.join(result.errors)}")
        
        return func(*args, **kwargs)
    
    return wrapper


def validate_json_config_enhanced(config_str: str, schema: Optional[Dict[str, Any]] = None) -> ValidationResult:
    """
    Enhanced JSON configuration validation with security checks.
    
    Args:
        config_str: JSON configuration string
        schema: Optional schema for validation
        
    Returns:
        ValidationResult with comprehensive validation
    """
    result = ValidationResult(is_valid=True, errors=[], warnings=[])
    
    try:
        # Security check first
        security_validator = SecurityValidator()
        sec_result = security_validator.validate_user_input(config_str, "config")
        result.errors.extend(sec_result.errors)
        result.warnings.extend(sec_result.warnings)
        if not sec_result.is_valid:
            result.is_valid = False
            return result
        
        config = json.loads(config_str)
        
        # Check for reasonable config size
        if len(config_str) > 1_000_000:  # 1MB limit
            result.add_error(f"Configuration too large: {len(config_str)} bytes")
        
        # Check nesting depth
        def get_depth(obj, current_depth=0):
            if isinstance(obj, dict):
                return max([get_depth(v, current_depth + 1) for v in obj.values()] or [current_depth])
            elif isinstance(obj, list):
                return max([get_depth(item, current_depth + 1) for item in obj] or [current_depth])
            return current_depth
        
        depth = get_depth(config)
        if depth > 10:
            result.add_warning(f"Deep nesting detected: {depth} levels")
        
        if schema:
            validator = ConfigValidator(schema)
            schema_result = validator.validate(config)
            result.errors.extend(schema_result.errors)
            result.warnings.extend(schema_result.warnings)
            result.is_valid = result.is_valid and schema_result.is_valid
        
    except json.JSONDecodeError as e:
        result.add_error(f"Invalid JSON: {e}")
    except Exception as e:
        result.add_error(f"Configuration validation error: {e}")
    
    return result