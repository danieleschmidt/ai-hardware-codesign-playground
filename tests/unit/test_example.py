"""
Example unit test module demonstrating testing patterns.

This module shows how to structure unit tests for the AI Hardware Co-Design Playground,
including mocking, parameterization, and various testing scenarios.
"""

import pytest
from unittest.mock import Mock, patch


class TestExampleComponent:
    """Example test class for a component."""
    
    def test_basic_functionality(self):
        """Test basic functionality."""
        # Arrange
        expected_result = 42
        
        # Act
        result = 40 + 2
        
        # Assert
        assert result == expected_result
    
    @pytest.mark.parametrize("input_value,expected", [
        (1, 2),
        (5, 6),
        (10, 11),
        (-1, 0),
    ])
    def test_parameterized_function(self, input_value, expected):
        """Test function with multiple parameter sets."""
        result = input_value + 1
        assert result == expected
    
    def test_with_mock(self):
        """Test using mock objects."""
        # Arrange
        mock_service = Mock()
        mock_service.process.return_value = "mocked_result"
        
        # Act
        result = mock_service.process("test_input")
        
        # Assert
        assert result == "mocked_result"
        mock_service.process.assert_called_once_with("test_input")
    
    @patch('builtins.open', new_callable=Mock)
    def test_with_patch(self, mock_open):
        """Test using patches."""
        # Arrange
        mock_file = Mock()
        mock_file.read.return_value = "file_content"
        mock_open.return_value.__enter__.return_value = mock_file
        
        # Act
        with open("test_file.txt", "r") as f:
            content = f.read()
        
        # Assert
        assert content == "file_content"
        mock_open.assert_called_once_with("test_file.txt", "r")
    
    def test_exception_handling(self):
        """Test exception handling."""
        with pytest.raises(ValueError, match="invalid value"):
            raise ValueError("invalid value")
    
    @pytest.mark.slow
    def test_slow_operation(self):
        """Test marked as slow (can be skipped with -m "not slow")."""
        import time
        time.sleep(0.1)  # Simulate slow operation
        assert True
    
    def test_with_fixtures(self, temp_directory, sample_files):
        """Test using fixtures from conftest.py."""
        assert temp_directory.exists()
        assert "model" in sample_files
        assert sample_files["model"].exists()


class TestAsyncComponent:
    """Example async test class."""
    
    @pytest.mark.asyncio
    async def test_async_function(self):
        """Test async functionality."""
        import asyncio
        
        async def async_add(a, b):
            await asyncio.sleep(0.01)  # Simulate async work
            return a + b
        
        result = await async_add(2, 3)
        assert result == 5


class TestDataValidation:
    """Example tests for data validation."""
    
    def test_valid_hardware_spec(self, sample_hardware_spec):
        """Test valid hardware specification."""
        spec = sample_hardware_spec
        
        assert spec["type"] in ["systolic_array", "vector_processor"]
        assert isinstance(spec["dimensions"], dict)
        assert spec["data_width"] > 0
        assert spec["frequency_mhz"] > 0
        assert spec["power_budget_w"] > 0
    
    def test_invalid_hardware_spec(self):
        """Test invalid hardware specification."""
        invalid_spec = {
            "type": "invalid_type",
            "dimensions": {"rows": -1, "cols": 0},
            "data_width": 0,
        }
        
        # This would normally call a validation function
        # For demo purposes, we'll just check the values
        assert invalid_spec["dimensions"]["rows"] < 0
        assert invalid_spec["dimensions"]["cols"] == 0
        assert invalid_spec["data_width"] == 0


class TestModelAnalysis:
    """Example tests for model analysis."""
    
    def test_model_profiling(self, mock_model_analyzer, sample_neural_network):
        """Test model profiling functionality."""
        # Arrange
        model = sample_neural_network
        analyzer = mock_model_analyzer
        
        # Act
        profile = analyzer.analyze(model)
        layers = analyzer.get_layers()
        
        # Assert
        assert "operations" in profile
        assert "parameters" in profile
        assert "memory_mb" in profile
        assert len(layers) > 0
        analyzer.analyze.assert_called_once_with(model)
    
    @pytest.mark.benchmark
    def test_analysis_performance(self, benchmark, mock_model_analyzer, sample_neural_network):
        """Benchmark test for model analysis performance."""
        model = sample_neural_network
        analyzer = mock_model_analyzer
        
        # Benchmark the analysis function
        result = benchmark(analyzer.analyze, model)
        assert result is not None


class TestHardwareGeneration:
    """Example tests for hardware generation."""
    
    def test_rtl_generation(self, mock_hardware_simulator, temp_directory):
        """Test RTL generation functionality."""
        # Arrange
        simulator = mock_hardware_simulator
        output_file = temp_directory / "generated.v"
        
        # Act
        simulator.compile()
        output_file.write_text("// Generated RTL code")
        
        # Assert
        assert output_file.exists()
        assert "Generated RTL" in output_file.read_text()
        simulator.compile.assert_called_once()
    
    @pytest.mark.hardware
    def test_hardware_simulation(self, mock_hardware_simulator):
        """Test hardware simulation (requires hardware tools)."""
        simulator = mock_hardware_simulator
        
        # Act
        metrics = simulator.run()
        
        # Assert
        assert "cycles" in metrics
        assert "power" in metrics
        assert "area" in metrics
        assert metrics["cycles"] > 0
        assert metrics["power"] > 0


class TestErrorHandling:
    """Example tests for error handling."""
    
    def test_file_not_found_error(self, temp_directory):
        """Test handling of file not found errors."""
        non_existent_file = temp_directory / "does_not_exist.txt"
        
        with pytest.raises(FileNotFoundError):
            with open(non_existent_file, 'r') as f:
                f.read()
    
    def test_validation_error(self):
        """Test handling of validation errors."""
        def validate_positive_number(value):
            if value <= 0:
                raise ValueError("Value must be positive")
            return value
        
        with pytest.raises(ValueError, match="Value must be positive"):
            validate_positive_number(-1)
    
    def test_recovery_from_error(self, mock_error_handler):
        """Test error recovery mechanisms."""
        handler = mock_error_handler
        
        # Simulate error and recovery
        error_response = handler.handle_error(Exception("test error"))
        
        assert "error" in error_response
        assert "code" in error_response
        handler.handle_error.assert_called_once()


# Utility functions for testing
def assert_valid_verilog(content: str) -> bool:
    """Assert that content looks like valid Verilog."""
    required_keywords = ["module", "endmodule"]
    return all(keyword in content for keyword in required_keywords)


def assert_performance_within_bounds(metrics: dict, max_latency: float = 1.0) -> bool:
    """Assert that performance metrics are within acceptable bounds."""
    return metrics.get("latency", float('inf')) <= max_latency