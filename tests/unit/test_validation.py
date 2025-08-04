"""
Unit tests for validation utilities.

This module tests the validation classes and functions for input
validation, configuration checking, and security validation.
"""

import pytest
from pathlib import Path
import tempfile
import json

from codesign_playground.utils.validation import (
    ValidationResult,
    BaseValidator,
    ConfigValidator,
    ModelValidator,
    HardwareValidator,
    OptimizationValidator,
    validate_file_path,
    validate_json_config
)
from codesign_playground.utils.exceptions import ValidationError


class TestValidationResult:
    """Test ValidationResult dataclass."""
    
    def test_validation_result_creation(self):
        """Test ValidationResult creation."""
        result = ValidationResult(is_valid=True, errors=[], warnings=[])
        
        assert result.is_valid is True
        assert result.errors == []
        assert result.warnings == []
        assert bool(result) is True
    
    def test_validation_result_with_errors(self):
        """Test ValidationResult with errors."""
        result = ValidationResult(is_valid=False, errors=["Error 1", "Error 2"], warnings=[])
        
        assert result.is_valid is False
        assert len(result.errors) == 2
        assert bool(result) is False
    
    def test_add_error(self):
        """Test adding errors to ValidationResult."""
        result = ValidationResult(is_valid=True, errors=[], warnings=[])
        
        result.add_error("Test error")
        
        assert result.is_valid is False
        assert "Test error" in result.errors
        assert bool(result) is False
    
    def test_add_warning(self):
        """Test adding warnings to ValidationResult."""
        result = ValidationResult(is_valid=True, errors=[], warnings=[])
        
        result.add_warning("Test warning")
        
        assert result.is_valid is True  # Warnings don't invalidate
        assert "Test warning" in result.warnings
        assert bool(result) is True


class MockValidator(BaseValidator):
    """Mock validator for testing base functionality."""
    
    def _validate_impl(self, value, result):
        if value == "invalid":
            result.add_error("Value is invalid")
        elif value == "warning":
            result.add_warning("Value has warning")


class TestBaseValidator:
    """Test BaseValidator base class."""
    
    def test_validator_creation(self):
        """Test validator creation."""
        validator = MockValidator(strict=True)
        assert validator.strict is True
        
        validator = MockValidator(strict=False)
        assert validator.strict is False
    
    def test_validate_valid_input(self):
        """Test validation with valid input."""
        validator = MockValidator()
        result = validator.validate("valid")
        
        assert result.is_valid is True
        assert len(result.errors) == 0
        assert len(result.warnings) == 0
    
    def test_validate_invalid_input(self):
        """Test validation with invalid input."""
        validator = MockValidator()
        result = validator.validate("invalid")
        
        assert result.is_valid is False
        assert len(result.errors) == 1
        assert "Value is invalid" in result.errors
    
    def test_validate_warning_strict_mode(self):
        """Test validation with warnings in strict mode."""
        validator = MockValidator(strict=True)
        result = validator.validate("warning")
        
        assert result.is_valid is False  # Warnings become errors in strict mode
        assert len(result.errors) == 1
        assert len(result.warnings) == 0
        assert "Value has warning" in result.errors
    
    def test_validate_warning_non_strict_mode(self):
        """Test validation with warnings in non-strict mode."""
        validator = MockValidator(strict=False)
        result = validator.validate("warning")
        
        assert result.is_valid is True  # Warnings don't invalidate in non-strict mode
        assert len(result.errors) == 0
        assert len(result.warnings) == 1
        assert "Value has warning" in result.warnings


class TestConfigValidator:
    """Test ConfigValidator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.schema = {
            "name": {
                "type": str,
                "required": True,
                "min_length": 1,
                "max_length": 50
            },
            "compute_units": {
                "type": int,
                "required": True,
                "min": 1,
                "max": 1024
            },
            "frequency_mhz": {
                "type": float,
                "required": False,
                "min": 10.0,
                "max": 2000.0
            },
            "dataflow": {
                "type": str,
                "required": True,
                "choices": ["weight_stationary", "output_stationary", "row_stationary"]
            },
            "precision": {
                "type": str,
                "required": False,
                "pattern": r"^(int8|int16|fp16|fp32)$"
            }
        }
        self.validator = ConfigValidator(self.schema)
    
    def test_valid_config(self):
        """Test validation with valid configuration."""
        config = {
            "name": "test_accelerator",
            "compute_units": 64,
            "frequency_mhz": 200.0,
            "dataflow": "weight_stationary",
            "precision": "int8"
        }
        
        result = self.validator.validate(config)
        
        assert result.is_valid is True
        assert len(result.errors) == 0
    
    def test_missing_required_field(self):
        """Test validation with missing required field."""
        config = {
            "compute_units": 64,
            "dataflow": "weight_stationary"
            # Missing required "name" field
        }
        
        result = self.validator.validate(config)
        
        assert result.is_valid is False
        assert any("Required field 'name' is missing" in error for error in result.errors)
    
    def test_wrong_type(self):
        """Test validation with wrong field type."""
        config = {
            "name": "test",
            "compute_units": "64",  # Should be int, not string
            "dataflow": "weight_stationary"
        }
        
        result = self.validator.validate(config)
        
        assert result.is_valid is False
        assert any("must be of type int" in error for error in result.errors)
    
    def test_value_out_of_range(self):
        """Test validation with value out of range."""
        config = {
            "name": "test",
            "compute_units": 2000,  # Exceeds max of 1024
            "dataflow": "weight_stationary"
        }
        
        result = self.validator.validate(config)
        
        assert result.is_valid is False
        assert any("must be <= 1024" in error for error in result.errors)
    
    def test_invalid_choice(self):
        """Test validation with invalid choice."""
        config = {
            "name": "test",
            "compute_units": 64,
            "dataflow": "invalid_dataflow"  # Not in choices
        }
        
        result = self.validator.validate(config)
        
        assert result.is_valid is False
        assert any("must be one of" in error for error in result.errors)
    
    def test_pattern_validation(self):
        """Test validation with pattern matching."""
        config = {
            "name": "test",
            "compute_units": 64,
            "dataflow": "weight_stationary",
            "precision": "invalid_precision"  # Doesn't match pattern
        }
        
        result = self.validator.validate(config)
        
        assert result.is_valid is False
        assert any("does not match required pattern" in error for error in result.errors)
    
    def test_string_length_validation(self):
        """Test validation of string length."""
        config = {
            "name": "",  # Too short
            "compute_units": 64,
            "dataflow": "weight_stationary"
        }
        
        result = self.validator.validate(config)
        
        assert result.is_valid is False
        assert any("must have length >= 1" in error for error in result.errors)
        
        # Test too long
        config["name"] = "x" * 100  # Too long
        result = self.validator.validate(config)
        
        assert result.is_valid is False
        assert any("must have length <= 50" in error for error in result.errors)
    
    def test_unknown_fields_strict(self):
        """Test validation with unknown fields in strict mode."""
        validator = ConfigValidator(self.schema, strict=True)
        config = {
            "name": "test",
            "compute_units": 64,
            "dataflow": "weight_stationary",
            "unknown_field": "value"  # Unknown field
        }
        
        result = validator.validate(config)
        
        assert result.is_valid is False
        assert any("Unknown field 'unknown_field'" in error for error in result.errors)
    
    def test_non_dict_input(self):
        """Test validation with non-dictionary input."""
        result = self.validator.validate("not a dict")
        
        assert result.is_valid is False
        assert any("must be a dictionary" in error for error in result.errors)


class TestModelValidator:
    """Test ModelValidator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.validator = ModelValidator()
    
    def test_valid_model_config(self):
        """Test validation with valid model configuration."""
        config = {
            "path": "model.onnx",
            "framework": "onnx",
            "input_shapes": {
                "input": [1, 3, 224, 224]
            }
        }
        
        result = self.validator.validate(config)
        
        # May have warnings about non-existent file, but should be valid
        assert result.is_valid or len(result.warnings) > 0
    
    def test_missing_model_path(self):
        """Test validation with missing model path."""
        config = {
            "framework": "onnx"
        }
        
        result = self.validator.validate(config)
        
        assert result.is_valid is False
        assert any("Model path is required" in error for error in result.errors)
    
    def test_unsupported_framework(self):
        """Test validation with unsupported framework."""
        config = {
            "path": "model.onnx",
            "framework": "unsupported_framework"
        }
        
        result = self.validator.validate(config)
        
        assert result.is_valid is False
        assert any("Unsupported framework" in error for error in result.errors)
    
    def test_invalid_input_shapes(self):
        """Test validation with invalid input shapes."""
        config = {
            "path": "model.onnx",
            "framework": "onnx",
            "input_shapes": {
                "input": [1, -1, 224, 224]  # Negative dimension
            }
        }
        
        result = self.validator.validate(config)
        
        assert result.is_valid is False
        assert any("must be positive integers" in error for error in result.errors)
    
    def test_non_dict_input(self):
        """Test validation with non-dictionary input."""
        result = self.validator.validate("not a dict")
        
        assert result.is_valid is False
        assert any("must be a dictionary" in error for error in result.errors)


class TestHardwareValidator:
    """Test HardwareValidator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.validator = HardwareValidator()
    
    def test_valid_hardware_config(self):
        """Test validation with valid hardware configuration."""
        config = {
            "compute_units": 64,
            "dataflow": "weight_stationary",
            "memory_hierarchy": ["sram_64kb", "dram"],
            "frequency_mhz": 200.0,
            "precision": "int8",
            "power_budget_w": 5.0,
            "area_budget_mm2": 10.0
        }
        
        result = self.validator.validate(config)
        
        assert result.is_valid is True or len(result.warnings) > 0  # May have warnings
    
    def test_invalid_compute_units(self):
        """Test validation with invalid compute units."""
        config = {"compute_units": -1}
        
        result = self.validator.validate(config)
        
        assert result.is_valid is False
        assert any("must be positive" in error for error in result.errors)
    
    def test_invalid_dataflow(self):
        """Test validation with invalid dataflow."""
        config = {"dataflow": "invalid_dataflow"}
        
        result = self.validator.validate(config)
        
        assert result.is_valid is False
        assert any("Unsupported dataflow" in error for error in result.errors)
    
    def test_invalid_memory_hierarchy(self):
        """Test validation with invalid memory hierarchy."""
        config = {
            "memory_hierarchy": ["invalid_memory_type", "dram"]
        }
        
        result = self.validator.validate(config)
        
        assert result.is_valid is False
        assert any("Unsupported memory type" in error for error in result.errors)
    
    def test_empty_memory_hierarchy(self):
        """Test validation with empty memory hierarchy."""
        config = {"memory_hierarchy": []}
        
        result = self.validator.validate(config)
        
        assert result.is_valid is False
        assert any("cannot be empty" in error for error in result.errors)
    
    def test_invalid_frequency(self):
        """Test validation with invalid frequency."""
        config = {"frequency_mhz": -100.0}
        
        result = self.validator.validate(config)
        
        assert result.is_valid is False
        assert any("must be positive" in error for error in result.errors)
    
    def test_invalid_precision(self):
        """Test validation with invalid precision."""
        config = {"precision": "invalid_precision"}
        
        result = self.validator.validate(config)
        
        assert result.is_valid is False
        assert any("Unsupported precision" in error for error in result.errors)


class TestOptimizationValidator:
    """Test OptimizationValidator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.validator = OptimizationValidator()
    
    def test_valid_optimization_config(self):
        """Test validation with valid optimization configuration."""
        config = {
            "strategy": "balanced",
            "objectives": ["latency", "power", "area"],
            "target_fps": 30.0,
            "power_budget": 5.0,
            "iterations": 10
        }
        
        result = self.validator.validate(config)
        
        assert result.is_valid is True
        assert len(result.errors) == 0
    
    def test_invalid_strategy(self):
        """Test validation with invalid strategy."""
        config = {"strategy": "invalid_strategy"}
        
        result = self.validator.validate(config)
        
        assert result.is_valid is False
        assert any("Unsupported optimization strategy" in error for error in result.errors)
    
    def test_invalid_objectives(self):
        """Test validation with invalid objectives."""
        config = {
            "objectives": ["latency", "invalid_objective", "power"]
        }
        
        result = self.validator.validate(config)
        
        assert result.is_valid is False
        assert any("Unsupported objective" in error for error in result.errors)
    
    def test_invalid_target_fps(self):
        """Test validation with invalid target FPS."""
        config = {"target_fps": -1.0}
        
        result = self.validator.validate(config)
        
        assert result.is_valid is False
        assert any("must be a positive number" in error for error in result.errors)
    
    def test_invalid_iterations(self):
        """Test validation with invalid iterations."""
        config = {"iterations": 0}
        
        result = self.validator.validate(config)
        
        assert result.is_valid is False
        assert any("must be a positive integer" in error for error in result.errors)


class TestValidationFunctions:
    """Test standalone validation functions."""
    
    def test_validate_file_path_valid(self):
        """Test file path validation with valid path."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_path = f.name
        
        try:
            result = validate_file_path(temp_path, must_exist=True)
            assert result.is_valid is True
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def test_validate_file_path_nonexistent(self):
        """Test file path validation with non-existent file."""
        result = validate_file_path("/nonexistent/path.txt", must_exist=False)
        # Should be valid if we don't require existence
        assert result.is_valid is True
        
        result = validate_file_path("/nonexistent/path.txt", must_exist=True)
        # Should be invalid if we require existence
        assert result.is_valid is False
    
    def test_validate_file_path_traversal(self):
        """Test file path validation with path traversal."""
        result = validate_file_path("../../../etc/passwd")
        
        assert result.is_valid is False
        assert any("Path traversal not allowed" in error for error in result.errors)
    
    def test_validate_json_config_valid(self):
        """Test JSON configuration validation with valid JSON."""
        config_str = '{"name": "test", "value": 42}'
        
        result = validate_json_config(config_str)
        
        assert result.is_valid is True
        assert len(result.errors) == 0
    
    def test_validate_json_config_invalid_json(self):
        """Test JSON configuration validation with invalid JSON."""
        config_str = '{"name": "test", "value":}'  # Invalid JSON
        
        result = validate_json_config(config_str)
        
        assert result.is_valid is False
        assert any("Invalid JSON" in error for error in result.errors)
    
    def test_validate_json_config_with_schema(self):
        """Test JSON configuration validation with schema."""
        schema = {
            "name": {"type": str, "required": True},
            "value": {"type": int, "required": True, "min": 0}
        }
        
        # Valid config
        config_str = '{"name": "test", "value": 42}'
        result = validate_json_config(config_str, schema)
        assert result.is_valid is True
        
        # Invalid config
        config_str = '{"name": "test", "value": -1}'  # Violates min constraint
        result = validate_json_config(config_str, schema)
        assert result.is_valid is False


@pytest.fixture
def sample_config():
    """Fixture for sample configuration."""
    return {
        "name": "test_accelerator",
        "compute_units": 64,
        "dataflow": "weight_stationary",
        "frequency_mhz": 200.0,
        "precision": "int8"
    }


@pytest.fixture
def config_schema():
    """Fixture for configuration schema."""
    return {
        "name": {"type": str, "required": True},
        "compute_units": {"type": int, "required": True, "min": 1, "max": 1024},
        "dataflow": {"type": str, "required": True, "choices": ["weight_stationary", "output_stationary"]},
        "frequency_mhz": {"type": float, "required": False, "min": 10.0},
        "precision": {"type": str, "required": False}
    }


class TestValidationFixtures:
    """Test validation functionality with fixtures."""
    
    def test_with_sample_config(self, sample_config, config_schema):
        """Test validation with sample configuration fixture."""
        validator = ConfigValidator(config_schema)
        result = validator.validate(sample_config)
        
        assert result.is_valid is True
        assert len(result.errors) == 0
    
    def test_modify_sample_config(self, sample_config, config_schema):
        """Test validation with modified sample configuration."""
        # Modify to make invalid
        sample_config["compute_units"] = 2000  # Exceeds max
        
        validator = ConfigValidator(config_schema)
        result = validator.validate(sample_config)
        
        assert result.is_valid is False
        assert any("must be <= 1024" in error for error in result.errors)