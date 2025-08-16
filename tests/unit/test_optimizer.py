"""
Comprehensive unit tests for optimizer functionality.

This module tests the ModelOptimizer class and related optimization components
for hardware-software co-optimization.
"""

import pytest
import time
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

from codesign_playground.core.optimizer import ModelOptimizer, OptimizationResult
from codesign_playground.core.accelerator import Accelerator, ModelProfile
from codesign_playground.utils.exceptions import OptimizationError, ValidationError


class TestModelOptimizer:
    """Test ModelOptimizer class functionality."""
    
    @pytest.fixture
    def mock_model(self):
        """Mock model for testing."""
        model = Mock()
        model.name = "test_model"
        model.framework = "pytorch"
        model.parameters = 1000000
        model.size_mb = 64.0
        model.precision = "fp32"
        return model
    
    @pytest.fixture
    def sample_accelerator(self):
        """Sample accelerator for testing."""
        return Accelerator(
            compute_units=64,
            memory_hierarchy=["sram_64kb", "dram"],
            dataflow="weight_stationary",
            frequency_mhz=200.0,
            precision="int8"
        )
    
    @pytest.fixture
    def optimizer(self, mock_model, sample_accelerator):
        """ModelOptimizer instance for testing."""
        return ModelOptimizer(mock_model, sample_accelerator)
    
    def test_optimizer_initialization(self, mock_model, sample_accelerator):
        """Test ModelOptimizer initialization."""
        optimizer = ModelOptimizer(mock_model, sample_accelerator)
        
        assert optimizer.model == mock_model
        assert optimizer.accelerator == sample_accelerator
        assert optimizer.optimization_history == []
        assert optimizer.current_iteration == 0
    
    def test_invalid_model_initialization(self, sample_accelerator):
        """Test initialization with invalid model."""
        with pytest.raises((TypeError, ValueError)):
            ModelOptimizer(None, sample_accelerator)
    
    def test_invalid_accelerator_initialization(self, mock_model):
        """Test initialization with invalid accelerator."""
        with pytest.raises((TypeError, ValueError)):
            ModelOptimizer(mock_model, None)
    
    def test_basic_co_optimization(self, optimizer):
        """Test basic co-optimization functionality."""
        result = optimizer.co_optimize(
            target_fps=30.0,
            power_budget=5.0,
            iterations=3
        )
        
        assert isinstance(result, OptimizationResult)
        assert result.target_fps == 30.0
        assert result.power_budget == 5.0
        assert result.iterations == 3
        assert result.optimization_time > 0
        assert result.metrics is not None
        assert len(optimizer.optimization_history) == 3
    
    def test_co_optimization_with_area_constraint(self, optimizer):
        """Test co-optimization with area constraint."""
        result = optimizer.co_optimize(
            target_fps=60.0,
            power_budget=10.0,
            area_budget=200.0,
            iterations=5
        )
        
        assert result.area_budget == 200.0
        assert result.final_metrics["estimated_area_mm2"] <= 200.0
        assert len(optimizer.optimization_history) == 5
    
    def test_co_optimization_with_accuracy_constraint(self, optimizer):
        """Test co-optimization with accuracy preservation."""
        result = optimizer.co_optimize(
            target_fps=30.0,
            power_budget=5.0,
            accuracy_threshold=0.95,
            iterations=3
        )
        
        assert result.accuracy_threshold == 0.95
        assert result.final_metrics["estimated_accuracy"] >= 0.95
    
    def test_hardware_constraint_application(self, optimizer, mock_model):
        """Test application of hardware constraints."""
        constraints = {
            "precision": "int8",
            "memory_limit_mb": 32,
            "compute_units": 32,
            "frequency_mhz": 400.0
        }
        
        optimized_model = optimizer.apply_hardware_constraints(mock_model, constraints)
        
        assert optimized_model is not None
        assert optimized_model.precision == "int8"
        assert hasattr(optimized_model, 'memory_optimized')
        assert optimized_model.memory_optimized
    
    def test_quantization_optimization(self, optimizer, mock_model):
        """Test quantization optimization."""
        quantization_config = {
            "target_precision": "int8",
            "calibration_samples": 100,
            "preserve_accuracy": True,
            "quantization_scheme": "post_training"
        }
        
        quantized_model = optimizer.apply_quantization(mock_model, quantization_config)
        
        assert quantized_model is not None
        assert quantized_model.precision == "int8"
        assert hasattr(quantized_model, 'quantization_config')
        assert quantized_model.quantization_config == quantization_config
    
    def test_pruning_optimization(self, optimizer, mock_model):
        """Test pruning optimization."""
        pruning_config = {
            "sparsity_ratio": 0.5,
            "pruning_method": "magnitude",
            "structured": False,
            "preserve_accuracy": True
        }
        
        pruned_model = optimizer.apply_pruning(mock_model, pruning_config)
        
        assert pruned_model is not None
        assert hasattr(pruned_model, 'sparsity_ratio')
        assert pruned_model.sparsity_ratio == 0.5
        assert pruned_model.parameters < mock_model.parameters  # Fewer parameters after pruning
    
    def test_knowledge_distillation(self, optimizer, mock_model):
        """Test knowledge distillation optimization."""
        distillation_config = {
            "teacher_model": mock_model,
            "student_size_ratio": 0.25,
            "temperature": 4.0,
            "alpha": 0.7
        }
        
        student_model = optimizer.apply_knowledge_distillation(mock_model, distillation_config)
        
        assert student_model is not None
        assert student_model.parameters < mock_model.parameters
        assert hasattr(student_model, 'distillation_config')
        assert student_model.distillation_config == distillation_config
    
    def test_optimize_for_accelerator(self, optimizer, mock_model, sample_accelerator):
        """Test optimization for specific accelerator."""
        optimization_config = {
            "target_precision": "int8",
            "optimization_level": "aggressive",
            "preserve_accuracy": True,
            "enable_fusion": True
        }
        
        optimized_model = optimizer.optimize_for_accelerator(
            mock_model, sample_accelerator, optimization_config
        )
        
        assert optimized_model is not None
        assert optimized_model.precision == "int8"
        assert hasattr(optimized_model, 'accelerator_optimized')
        assert optimized_model.accelerator_optimized
    
    def test_check_compatibility(self, optimizer, mock_model, sample_accelerator):
        """Test model-accelerator compatibility checking."""
        compatibility_score = optimizer.check_compatibility(mock_model, sample_accelerator)
        
        assert isinstance(compatibility_score, float)
        assert 0.0 <= compatibility_score <= 1.0
    
    def test_multi_objective_optimization(self, optimizer):
        """Test multi-objective optimization."""
        model_profile = ModelProfile(
            peak_gflops=20.0,
            bandwidth_gb_s=40.0,
            operations={"conv2d": 4000, "dense": 1200},
            parameters=2000000,
            memory_mb=32.0,
            compute_intensity=0.5,
            layer_types=["conv2d", "dense"],
            model_size_mb=32.0
        )
        
        constraints = {
            "power_budget": 8.0,
            "area_budget": 120.0,
            "latency_ms": 20.0,
            "accuracy_threshold": 0.95
        }
        
        result = optimizer.multi_objective_optimize(
            model_profile=model_profile,
            constraints=constraints,
            objectives=["latency", "power", "accuracy"],
            num_iterations=3
        )
        
        assert isinstance(result, dict)
        assert "pareto_solutions" in result
        assert "best_compromise" in result
        assert "optimization_history" in result
        assert len(result["pareto_solutions"]) > 0
    
    def test_optimization_convergence(self, optimizer):
        """Test optimization convergence detection."""
        # Run optimization with convergence detection
        result = optimizer.co_optimize(
            target_fps=30.0,
            power_budget=5.0,
            iterations=10,
            convergence_threshold=0.01,
            early_stopping=True
        )
        
        assert result.converged or result.iterations <= 10
        if result.converged:
            assert result.iterations < 10  # Should stop early if converged
    
    def test_optimization_with_invalid_constraints(self, optimizer):
        """Test optimization with invalid constraints."""
        # Impossible constraints
        with pytest.raises(OptimizationError):
            optimizer.co_optimize(
                target_fps=1000.0,  # Unrealistic
                power_budget=0.1,   # Too low
                area_budget=1.0,    # Too small
                iterations=3
            )
    
    def test_optimization_rollback(self, optimizer):
        """Test optimization rollback on failure."""
        original_model = optimizer.model
        
        # Simulate optimization failure
        with patch.object(optimizer, '_apply_optimization_step', side_effect=OptimizationError("Optimization failed")):
            with pytest.raises(OptimizationError):
                optimizer.co_optimize(
                    target_fps=30.0,
                    power_budget=5.0,
                    iterations=3,
                    rollback_on_failure=True
                )
        
        # Model should be rolled back to original state
        assert optimizer.model == original_model
    
    def test_parallel_optimization_strategies(self, optimizer):
        """Test parallel optimization with multiple strategies."""
        strategies = ["quantization", "pruning", "distillation"]
        
        results = optimizer.optimize_parallel_strategies(
            strategies=strategies,
            target_fps=30.0,
            power_budget=5.0,
            max_workers=2
        )
        
        assert len(results) == len(strategies)
        for strategy, result in results.items():
            assert strategy in strategies
            assert isinstance(result, OptimizationResult)
    
    def test_optimization_metrics_tracking(self, optimizer):
        """Test optimization metrics tracking."""
        result = optimizer.co_optimize(
            target_fps=30.0,
            power_budget=5.0,
            iterations=3,
            track_metrics=True
        )
        
        assert hasattr(result, 'metrics_history')
        assert len(result.metrics_history) == 3
        
        for metrics in result.metrics_history:
            assert "latency" in metrics
            assert "power" in metrics
            assert "accuracy" in metrics
    
    def test_optimization_caching(self, optimizer):
        """Test optimization result caching."""
        # First optimization
        result1 = optimizer.co_optimize(
            target_fps=30.0,
            power_budget=5.0,
            iterations=3,
            use_cache=True
        )
        
        # Second optimization with same parameters (should use cache)
        start_time = time.time()
        result2 = optimizer.co_optimize(
            target_fps=30.0,
            power_budget=5.0,
            iterations=3,
            use_cache=True
        )
        cache_time = time.time() - start_time
        
        # Cached result should be much faster
        assert cache_time < result1.optimization_time / 2
        assert result2.final_metrics == result1.final_metrics
    
    def test_optimization_reproducibility(self, optimizer):
        """Test optimization reproducibility with seed."""
        seed = 42
        
        result1 = optimizer.co_optimize(
            target_fps=30.0,
            power_budget=5.0,
            iterations=3,
            random_seed=seed
        )
        
        result2 = optimizer.co_optimize(
            target_fps=30.0,
            power_budget=5.0,
            iterations=3,
            random_seed=seed
        )
        
        # Results should be identical with same seed
        assert result1.final_metrics == result2.final_metrics
    
    def test_optimization_progress_callback(self, optimizer):
        """Test optimization with progress callback."""
        progress_updates = []
        
        def progress_callback(iteration, metrics, message):
            progress_updates.append((iteration, metrics, message))
        
        result = optimizer.co_optimize(
            target_fps=30.0,
            power_budget=5.0,
            iterations=3,
            progress_callback=progress_callback
        )
        
        assert len(progress_updates) == 3
        for i, (iteration, metrics, message) in enumerate(progress_updates):
            assert iteration == i + 1
            assert isinstance(metrics, dict)
            assert isinstance(message, str)


class TestOptimizationResult:
    """Test OptimizationResult class."""
    
    def test_optimization_result_creation(self):
        """Test OptimizationResult creation."""
        metrics = {
            "latency": 15.0,
            "power": 4.5,
            "accuracy": 0.96,
            "area": 80.0
        }
        
        result = OptimizationResult(
            target_fps=30.0,
            power_budget=5.0,
            iterations=5,
            optimization_time=10.5,
            final_metrics=metrics,
            converged=True
        )
        
        assert result.target_fps == 30.0
        assert result.power_budget == 5.0
        assert result.iterations == 5
        assert result.optimization_time == 10.5
        assert result.final_metrics == metrics
        assert result.converged
    
    def test_optimization_result_to_dict(self):
        """Test OptimizationResult serialization."""
        metrics = {"latency": 15.0, "power": 4.5}
        result = OptimizationResult(
            target_fps=30.0,
            power_budget=5.0,
            iterations=3,
            optimization_time=5.0,
            final_metrics=metrics
        )
        
        result_dict = result.to_dict()
        
        assert isinstance(result_dict, dict)
        assert result_dict["target_fps"] == 30.0
        assert result_dict["final_metrics"] == metrics
        assert "optimization_time" in result_dict
    
    def test_optimization_result_comparison(self):
        """Test OptimizationResult comparison methods."""
        result1 = OptimizationResult(
            target_fps=30.0,
            power_budget=5.0,
            iterations=3,
            optimization_time=5.0,
            final_metrics={"latency": 15.0, "power": 4.0}
        )
        
        result2 = OptimizationResult(
            target_fps=30.0,
            power_budget=5.0,
            iterations=3,
            optimization_time=7.0,
            final_metrics={"latency": 20.0, "power": 3.5}
        )
        
        # Test comparison based on specific metrics
        assert result1.is_better_than(result2, objective="latency")  # Lower latency is better
        assert result2.is_better_than(result1, objective="power")   # Lower power is better


class TestOptimizationIntegration:
    """Integration tests for optimization components."""
    
    def test_end_to_end_optimization_pipeline(self):
        """Test complete optimization pipeline."""
        # Create mock components
        model = Mock()
        model.name = "integration_test_model"
        model.framework = "pytorch"
        model.parameters = 5000000
        model.precision = "fp32"
        
        accelerator = Accelerator(
            compute_units=128,
            memory_hierarchy=["sram_128kb", "dram"],
            dataflow="weight_stationary",
            frequency_mhz=400.0
        )
        
        optimizer = ModelOptimizer(model, accelerator)
        
        # Run complete optimization pipeline
        # Step 1: Apply hardware constraints
        constrained_model = optimizer.apply_hardware_constraints(model, {
            "precision": "int8",
            "memory_limit_mb": 64
        })
        
        # Step 2: Apply quantization
        quantized_model = optimizer.apply_quantization(constrained_model, {
            "target_precision": "int8",
            "calibration_samples": 100
        })
        
        # Step 3: Apply pruning
        pruned_model = optimizer.apply_pruning(quantized_model, {
            "sparsity_ratio": 0.3,
            "pruning_method": "magnitude"
        })
        
        # Step 4: Run co-optimization
        result = optimizer.co_optimize(
            target_fps=60.0,
            power_budget=8.0,
            iterations=5
        )
        
        # Verify pipeline results
        assert pruned_model.precision == "int8"
        assert pruned_model.sparsity_ratio == 0.3
        assert result.target_fps == 60.0
        assert result.iterations == 5
        assert result.optimization_time > 0
    
    def test_optimization_with_multiple_objectives(self):
        """Test optimization balancing multiple objectives."""
        model = Mock()
        model.parameters = 2000000
        
        accelerator = Accelerator(compute_units=64, dataflow="output_stationary")
        optimizer = ModelOptimizer(model, accelerator)
        
        # Define conflicting objectives
        objectives = ["latency", "power", "accuracy", "area"]
        
        result = optimizer.multi_objective_optimize(
            model_profile=ModelProfile(
                peak_gflops=30.0,
                bandwidth_gb_s=60.0,
                operations={"conv2d": 6000, "dense": 2000},
                parameters=2000000,
                memory_mb=48.0,
                compute_intensity=0.5,
                layer_types=["conv2d", "dense"],
                model_size_mb=48.0
            ),
            constraints={
                "power_budget": 12.0,
                "area_budget": 150.0,
                "latency_ms": 16.0,
                "accuracy_threshold": 0.93
            },
            objectives=objectives,
            num_iterations=5
        )
        
        # Verify multi-objective results
        assert len(result["pareto_solutions"]) > 0
        assert "best_compromise" in result
        
        # Each solution should have all objective metrics
        for solution in result["pareto_solutions"]:
            for objective in objectives:
                assert objective in solution["metrics"]


@pytest.fixture
def performance_test_data():
    """Fixture for performance testing data."""
    return {
        "models": [
            {"name": "small_model", "parameters": 100000},
            {"name": "medium_model", "parameters": 1000000},
            {"name": "large_model", "parameters": 10000000}
        ],
        "accelerators": [
            {"compute_units": 16, "frequency": 200},
            {"compute_units": 64, "frequency": 400},
            {"compute_units": 256, "frequency": 800}
        ]
    }


class TestOptimizationPerformance:
    """Performance tests for optimization algorithms."""
    
    def test_optimization_scalability(self, performance_test_data):
        """Test optimization scalability with different model sizes."""
        optimization_times = []
        
        for model_data in performance_test_data["models"]:
            model = Mock()
            model.parameters = model_data["parameters"]
            model.name = model_data["name"]
            
            accelerator = Accelerator(compute_units=64, dataflow="weight_stationary")
            optimizer = ModelOptimizer(model, accelerator)
            
            start_time = time.time()
            result = optimizer.co_optimize(
                target_fps=30.0,
                power_budget=5.0,
                iterations=3
            )
            optimization_time = time.time() - start_time
            
            optimization_times.append(optimization_time)
            
            # Verify optimization completed successfully
            assert result.optimization_time > 0
        
        # Optimization time should scale reasonably with model size
        # (though this is a simplified test with mocked components)
        assert all(t > 0 for t in optimization_times)
    
    def test_parallel_optimization_performance(self, performance_test_data):
        """Test parallel optimization performance."""
        model = Mock()
        model.parameters = 1000000
        
        accelerator = Accelerator(compute_units=64, dataflow="weight_stationary")
        optimizer = ModelOptimizer(model, accelerator)
        
        strategies = ["quantization", "pruning", "distillation"]
        
        # Sequential optimization
        start_time = time.time()
        sequential_results = {}
        for strategy in strategies:
            sequential_results[strategy] = optimizer.co_optimize(
                target_fps=30.0,
                power_budget=5.0,
                iterations=2
            )
        sequential_time = time.time() - start_time
        
        # Parallel optimization
        start_time = time.time()
        parallel_results = optimizer.optimize_parallel_strategies(
            strategies=strategies,
            target_fps=30.0,
            power_budget=5.0,
            max_workers=2
        )
        parallel_time = time.time() - start_time
        
        # Parallel should be faster (though with mocked components, improvement may be minimal)
        assert len(parallel_results) == len(strategies)
        assert parallel_time <= sequential_time * 1.2  # Allow some overhead


if __name__ == "__main__":
    pytest.main([__file__, "-v"])