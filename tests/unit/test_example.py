"""Example unit tests demonstrating testing patterns."""

import pytest
from unittest.mock import Mock, patch


class TestExampleClass:
    """Example test class demonstrating unit testing patterns."""
    
    def test_basic_assertion(self):
        """Test basic assertion patterns."""
        # Arrange
        expected_value = 42
        
        # Act
        actual_value = 40 + 2
        
        # Assert
        assert actual_value == expected_value
    
    def test_with_fixture(self, random_seed):
        """Test using pytest fixture."""
        import random
        
        # Random should be deterministic due to fixture
        first_random = random.random()
        
        # Reset seed
        random.seed(42)
        second_random = random.random()
        
        assert first_random == second_random
    
    def test_exception_handling(self):
        """Test exception handling patterns."""
        with pytest.raises(ValueError, match="Invalid input"):
            raise ValueError("Invalid input provided")
    
    def test_parametrized(self, sample_data):
        """Test parametrized testing."""
        @pytest.mark.parametrize("input_val,expected", [
            (1, 2),
            (2, 4),
            (3, 6),
            (0, 0),
        ])
        def test_multiply_by_two(input_val, expected):
            assert input_val * 2 == expected
    
    @pytest.mark.slow
    def test_slow_operation(self):
        """Test marked as slow - can be skipped with -m 'not slow'."""
        import time
        time.sleep(0.1)  # Simulate slow operation
        assert True
    
    def test_with_mock(self):
        """Test using mocks."""
        # Create mock object
        mock_service = Mock()
        mock_service.get_data.return_value = {"key": "value"}
        
        # Test the mock
        result = mock_service.get_data()
        
        assert result == {"key": "value"}
        mock_service.get_data.assert_called_once()
    
    @patch('builtins.open')
    def test_with_patch(self, mock_open):
        """Test using patches."""
        # Configure mock
        mock_open.return_value.__enter__.return_value.read.return_value = "test content"
        
        # Test code that uses open
        with open("test.txt", "r") as f:
            content = f.read()
        
        assert content == "test content"
        mock_open.assert_called_once_with("test.txt", "r")
    
    def test_async_function(self, event_loop):
        """Test async function."""
        async def async_operation():
            return "async result"
        
        # Run async function in event loop
        result = event_loop.run_until_complete(async_operation())
        assert result == "async result"
    
    def test_temp_file(self, temp_file):
        """Test using temporary file fixture."""
        # Write to temp file
        with open(temp_file, 'w') as f:
            f.write("test content")
        
        # Read from temp file
        with open(temp_file, 'r') as f:
            content = f.read()
        
        assert content == "test content"
    
    @pytest.mark.skipif(True, reason="Example of conditional skip")
    def test_conditional_skip(self):
        """Test that is conditionally skipped."""
        assert False  # This won't run due to skip
    
    def test_approximate_equality(self):
        """Test approximate equality for floating point numbers."""
        result = 0.1 + 0.2
        expected = 0.3
        
        # Use pytest.approx for floating point comparison
        assert result == pytest.approx(expected, rel=1e-9)
    
    def test_list_comparison(self):
        """Test list comparison patterns."""
        actual_list = [1, 2, 3, 4, 5]
        
        # Test list contains specific items
        assert 3 in actual_list
        assert 6 not in actual_list
        
        # Test list length
        assert len(actual_list) == 5
        
        # Test list subset
        assert set([1, 2, 3]).issubset(set(actual_list))
    
    def test_dictionary_comparison(self):
        """Test dictionary comparison patterns."""
        actual_dict = {"a": 1, "b": 2, "c": 3}
        
        # Test key existence
        assert "a" in actual_dict
        assert "d" not in actual_dict
        
        # Test value access
        assert actual_dict["a"] == 1
        
        # Test partial dictionary matching
        expected_subset = {"a": 1, "b": 2}
        assert all(actual_dict[k] == v for k, v in expected_subset.items())


@pytest.mark.unit
class TestModelAnalysisExample:
    """Example tests for model analysis functionality."""
    
    def test_model_loading(self, sample_onnx_model):
        """Test model loading functionality."""
        # Test that model is properly loaded
        assert sample_onnx_model is not None
        assert hasattr(sample_onnx_model, 'graph')
    
    def test_model_profiling(self, sample_pytorch_model):
        """Test model profiling functionality."""
        # Mock the model analyzer
        with patch('codesign_playground.analysis.ModelAnalyzer') as mock_analyzer:
            mock_analyzer.return_value.profile.return_value = {
                'total_params': 1000,
                'total_flops': 50000,
                'memory_usage_mb': 10.5
            }
            
            # Test profiling
            from unittest.mock import MagicMock
            analyzer = MagicMock()
            profile = analyzer.profile(sample_pytorch_model)
            
            assert 'total_params' in profile
            assert 'total_flops' in profile
            assert 'memory_usage_mb' in profile


@pytest.mark.unit
class TestHardwareGenerationExample:
    """Example tests for hardware generation functionality."""
    
    def test_systolic_array_generation(self, sample_systolic_config):
        """Test systolic array generation."""
        # Mock the hardware generator
        with patch('codesign_playground.hardware.SystolicArray') as mock_array:
            mock_instance = Mock()
            mock_instance.generate_rtl.return_value = "module systolic_array();\nendmodule"
            mock_instance.estimate_resources.return_value = {
                'luts': 1000,
                'dsps': 64,
                'bram_kb': 128
            }
            mock_array.return_value = mock_instance
            
            # Test generation
            array = mock_array(**sample_systolic_config)
            rtl = array.generate_rtl()
            resources = array.estimate_resources()
            
            assert "module systolic_array" in rtl
            assert 'luts' in resources
            assert 'dsps' in resources
    
    def test_vector_processor_generation(self, sample_vector_processor_config):
        """Test vector processor generation."""
        # Mock the vector processor
        with patch('codesign_playground.hardware.VectorProcessor') as mock_processor:
            mock_instance = Mock()
            mock_instance.generate_rtl.return_value = "module vector_processor();\nendmodule"
            mock_instance.add_custom_instruction.return_value = True
            mock_processor.return_value = mock_instance
            
            # Test generation
            processor = mock_processor(**sample_vector_processor_config)
            rtl = processor.generate_rtl()
            custom_added = processor.add_custom_instruction("custom_op")
            
            assert "module vector_processor" in rtl
            assert custom_added is True


@pytest.mark.unit
class TestOptimizationExample:
    """Example tests for optimization functionality."""
    
    def test_genetic_algorithm(self, sample_design_space, sample_optimization_objectives):
        """Test genetic algorithm optimization."""
        with patch('codesign_playground.optimization.GeneticOptimizer') as mock_optimizer:
            mock_instance = Mock()
            mock_instance.optimize.return_value = {
                'best_solution': {'compute_units': 64, 'memory_size_kb': 128},
                'best_fitness': 0.85,
                'generations': 25,
                'convergence': True
            }
            mock_optimizer.return_value = mock_instance
            
            # Test optimization
            optimizer = mock_optimizer()
            result = optimizer.optimize(
                design_space=sample_design_space,
                objectives=sample_optimization_objectives
            )
            
            assert 'best_solution' in result
            assert 'best_fitness' in result
            assert result['convergence'] is True
    
    def test_pareto_frontier(self):
        """Test Pareto frontier calculation."""
        with patch('codesign_playground.optimization.ParetoFrontier') as mock_pareto:
            mock_instance = Mock()
            mock_instance.compute.return_value = [
                {'latency': 10, 'power': 5, 'area': 8},
                {'latency': 12, 'power': 4, 'area': 6},
                {'latency': 8, 'power': 6, 'area': 10}
            ]
            mock_pareto.return_value = mock_instance
            
            # Test Pareto frontier
            pareto = mock_pareto()
            frontier = pareto.compute([])
            
            assert len(frontier) == 3
            assert all('latency' in point for point in frontier)
            assert all('power' in point for point in frontier)
            assert all('area' in point for point in frontier)


# Example of async test
@pytest.mark.asyncio
async def test_async_api_call(async_api_client):
    """Example async API test."""
    response = await async_api_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data


# Example of performance test
@pytest.mark.performance
def test_performance_example(benchmark, sample_pytorch_model):
    """Example performance test using pytest-benchmark."""
    def operation_to_benchmark():
        # Simulate some operation
        import torch
        with torch.no_grad():
            input_tensor = torch.randn(1, 784)
            return sample_pytorch_model(input_tensor)
    
    # Benchmark the operation
    result = benchmark(operation_to_benchmark)
    
    # Assert result is valid
    assert result is not None
    assert result.shape == (1, 10)


# Example of property-based testing
@pytest.mark.unit
class TestPropertyBasedExample:
    """Example property-based tests using hypothesis."""
    
    @pytest.mark.parametrize("value", [1, 2, 3, 4, 5])
    def test_parametrized_property(self, value):
        """Test property holds for multiple values."""
        result = value * 2
        assert result > value  # Property: doubling always increases
        assert result % 2 == 0  # Property: result is always even
    
    def test_hypothesis_example(self):
        """Example using hypothesis for property-based testing."""
        try:
            from hypothesis import given, strategies as st
            
            @given(st.integers(min_value=1, max_value=1000))
            def test_square_property(n):
                square = n * n
                assert square >= n  # Property: square is always >= original
                assert square > 0   # Property: square is always positive
            
            # Run the property test
            test_square_property()
            
        except ImportError:
            pytest.skip("Hypothesis not available")
