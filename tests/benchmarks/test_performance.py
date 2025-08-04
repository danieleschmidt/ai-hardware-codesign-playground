"""
Performance benchmarks for AI Hardware Co-Design Playground.

This module provides performance benchmarks to validate system performance
and detect regressions.
"""

import pytest
import time
import numpy as np
from unittest.mock import Mock

from codesign_playground.core.accelerator import AcceleratorDesigner, Accelerator
from codesign_playground.core.optimizer import ModelOptimizer
from codesign_playground.core.explorer import DesignSpaceExplorer
from codesign_playground.core.cache import cached, AdaptiveCache
from codesign_playground.core.performance import get_profiler, profile_operation


class TestAcceleratorPerformance:
    """Performance benchmarks for accelerator design."""
    
    def test_accelerator_design_performance(self, benchmark):
        """Benchmark accelerator design performance."""
        designer = AcceleratorDesigner()
        
        def design_accelerator():
            return designer.design(
                compute_units=64,
                dataflow="weight_stationary",
                frequency_mhz=200.0
            )
        
        result = benchmark(design_accelerator)
        
        assert result is not None
        assert result.compute_units == 64
    
    def test_model_profiling_performance(self, benchmark):
        """Benchmark model profiling performance."""
        designer = AcceleratorDesigner()
        mock_model = {"type": "test", "complexity": 1.0}
        input_shape = (224, 224, 3)
        
        def profile_model():
            return designer.profile_model(mock_model, input_shape)
        
        result = benchmark(profile_model)
        
        assert result is not None
        assert result.peak_gflops > 0
    
    def test_rtl_generation_performance(self, benchmark):
        """Benchmark RTL generation performance."""
        accelerator = Accelerator(
            compute_units=32,
            memory_hierarchy=["sram_64kb", "dram"],
            dataflow="weight_stationary"
        )
        
        def generate_rtl():
            # Generate RTL in memory (don't write to file for benchmark)
            return accelerator._generate_verilog_code()
        
        result = benchmark(generate_rtl)
        
        assert result is not None
        assert "module accelerator" in result
    
    def test_performance_estimation(self, benchmark):
        """Benchmark performance estimation."""
        accelerator = Accelerator(
            compute_units=64,
            memory_hierarchy=["sram_64kb", "dram"],
            dataflow="weight_stationary"
        )
        
        def estimate_performance():
            return accelerator.estimate_performance()
        
        result = benchmark(estimate_performance)
        
        assert result is not None
        assert "throughput_ops_s" in result


class TestOptimizerPerformance:
    """Performance benchmarks for optimization algorithms."""
    
    def test_co_optimization_performance(self, benchmark):
        """Benchmark co-optimization performance."""
        mock_model = {"type": "test", "complexity": 1.0}
        accelerator = Accelerator(
            compute_units=32,
            memory_hierarchy=["sram_64kb", "dram"],
            dataflow="weight_stationary"
        )
        
        optimizer = ModelOptimizer(mock_model, accelerator)
        
        def run_co_optimization():
            return optimizer.co_optimize(
                target_fps=30.0,
                power_budget=5.0,
                iterations=3  # Reduced for benchmark
            )
        
        result = benchmark(run_co_optimization)
        
        assert result is not None
        assert result.iterations == 3
    
    def test_hardware_constraint_application(self, benchmark):
        """Benchmark hardware constraint application."""
        mock_model = {"type": "test", "complexity": 1.0}
        accelerator = Accelerator(
            compute_units=32,
            memory_hierarchy=["sram_64kb", "dram"],
            dataflow="weight_stationary"
        )
        
        optimizer = ModelOptimizer(mock_model, accelerator)
        constraints = {
            "precision": "int8",
            "memory_limit_mb": 64,
            "compute_units": 32
        }
        
        def apply_constraints():
            return optimizer.apply_hardware_constraints(mock_model, constraints)
        
        result = benchmark(apply_constraints)
        
        assert result is not None


class TestExplorerPerformance:
    """Performance benchmarks for design space exploration."""
    
    def test_design_space_exploration_performance(self, benchmark):
        """Benchmark design space exploration."""
        explorer = DesignSpaceExplorer(parallel_workers=2)  # Reduced for benchmark
        mock_model = {"type": "test", "complexity": 1.0}
        
        design_space = {
            "compute_units": [16, 32, 64],
            "dataflow": ["weight_stationary", "output_stationary"],
            "frequency_mhz": [100, 200],
            "precision": ["int8", "fp16"]
        }
        
        def run_exploration():
            return explorer.explore(
                model=mock_model,
                design_space=design_space,
                objectives=["latency", "power"],
                num_samples=12,  # Reduced for benchmark
                strategy="random"
            )
        
        result = benchmark(run_exploration)
        
        assert result is not None
        assert result.total_evaluations > 0
        assert len(result.design_points) > 0
    
    def test_pareto_frontier_computation(self, benchmark):
        """Benchmark Pareto frontier computation."""
        explorer = DesignSpaceExplorer()
        
        # Create mock design points
        from codesign_playground.core.explorer import DesignPoint
        design_points = []
        
        for i in range(100):  # Create 100 points for benchmark
            config = {"compute_units": 16 + i % 64, "dataflow": "weight_stationary"}
            metrics = {
                "latency": 10 + np.random.random() * 20,
                "power": 2 + np.random.random() * 8
            }
            design_points.append(DesignPoint(config=config, metrics=metrics))
        
        def compute_pareto():
            return explorer._compute_pareto_frontier(design_points, ["latency", "power"])
        
        result = benchmark(compute_pareto)
        
        assert result is not None
        assert len(result) > 0


class TestCachePerformance:
    """Performance benchmarks for caching system."""
    
    def test_cache_put_performance(self, benchmark):
        """Benchmark cache put operations."""
        cache = AdaptiveCache(max_size=1000, max_memory_mb=10.0)
        
        test_data = {"result": list(range(100)), "metadata": {"size": 100}}
        
        def cache_put():
            for i in range(100):
                cache.put(f"key_{i}", test_data)
        
        benchmark(cache_put)
    
    def test_cache_get_performance(self, benchmark):
        """Benchmark cache get operations."""
        cache = AdaptiveCache(max_size=1000, max_memory_mb=10.0)
        
        # Pre-populate cache
        test_data = {"result": list(range(100)), "metadata": {"size": 100}}
        for i in range(100):
            cache.put(f"key_{i}", test_data)
        
        def cache_get():
            for i in range(100):
                cache.get(f"key_{i}")
        
        benchmark(cache_get)
    
    def test_cached_function_performance(self, benchmark):
        """Benchmark cached function decorator."""
        call_count = 0
        
        @cached(cache_type="default", ttl=3600.0)
        def expensive_function(x, y):
            nonlocal call_count
            call_count += 1
            time.sleep(0.001)  # Simulate work
            return x * y + x + y
        
        def run_cached_calls():
            # First call will compute, second will hit cache
            result1 = expensive_function(10, 20)
            result2 = expensive_function(10, 20)  # Should hit cache
            return result1, result2
        
        result = benchmark(run_cached_calls)
        
        assert result[0] == result[1]  # Same result
        assert call_count <= 1  # Should only compute once due to caching


class TestSystemPerformance:
    """System-level performance benchmarks."""
    
    def test_profiler_overhead(self, benchmark):
        """Benchmark profiler overhead."""
        profiler = get_profiler()
        
        def operation_with_profiling():
            op_id = profiler.start_operation("test_operation")
            # Simulate work
            result = sum(range(1000))
            profiler.end_operation(op_id, success=True)
            return result
        
        result = benchmark(operation_with_profiling)
        
        assert result is not None
    
    def test_decorator_profiling_overhead(self, benchmark):
        """Benchmark profiling decorator overhead."""
        
        @profile_operation("benchmark_operation")
        def profiled_function():
            return sum(range(1000))
        
        result = benchmark(profiled_function)
        
        assert result is not None
    
    def test_concurrent_operations(self, benchmark):
        """Benchmark concurrent operations."""
        from concurrent.futures import ThreadPoolExecutor
        
        def compute_heavy_task(n):
            return sum(i * i for i in range(n))
        
        def run_concurrent_tasks():
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(compute_heavy_task, 1000) for _ in range(10)]
                results = [f.result() for f in futures]
            return results
        
        results = benchmark(run_concurrent_tasks)
        
        assert len(results) == 10
        assert all(r > 0 for r in results)


class TestMemoryEfficiency:
    """Memory efficiency benchmarks."""
    
    def test_memory_usage_during_design(self, benchmark):
        """Benchmark memory usage during accelerator design."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        def memory_intensive_design():
            designer = AcceleratorDesigner()
            accelerators = []
            
            # Create multiple accelerators
            for i in range(50):
                acc = designer.design(
                    compute_units=32 + i % 32,
                    dataflow="weight_stationary"
                )
                accelerators.append(acc)
            
            return len(accelerators)
        
        initial_memory = process.memory_info().rss
        result = benchmark(memory_intensive_design)
        final_memory = process.memory_info().rss
        
        memory_increase_mb = (final_memory - initial_memory) / (1024 * 1024)
        
        assert result == 50
        # Memory increase should be reasonable (less than 100MB for this test)
        assert memory_increase_mb < 100
    
    def test_cache_memory_efficiency(self, benchmark):
        """Benchmark cache memory efficiency."""
        cache = AdaptiveCache(max_size=1000, max_memory_mb=5.0)
        
        def fill_cache_efficiently():
            # Fill cache with varying sizes of data
            for i in range(100):
                data_size = 100 + i % 500  # Varying data sizes
                data = list(range(data_size))
                cache.put(f"data_{i}", data)
            
            return cache.get_stats()
        
        stats = benchmark(fill_cache_efficiently)
        
        assert stats["size"] > 0
        assert stats["memory_usage_mb"] <= 5.0  # Should respect limit


# Benchmark configuration
@pytest.fixture(scope="session")
def benchmark_config():
    """Configuration for benchmarks."""
    return {
        "min_rounds": 3,
        "max_time": 10.0,  # Maximum time per benchmark
        "warmup": False,   # Skip warmup for faster testing
    }


# Performance thresholds (for regression testing)
class PerformanceThresholds:
    """Define performance thresholds for regression detection."""
    
    # Maximum acceptable times (in seconds)
    ACCELERATOR_DESIGN_MAX = 0.1
    MODEL_PROFILING_MAX = 0.05
    RTL_GENERATION_MAX = 0.02
    PERFORMANCE_ESTIMATION_MAX = 0.01
    
    CO_OPTIMIZATION_MAX = 1.0  # 3 iterations
    CONSTRAINT_APPLICATION_MAX = 0.01
    
    DESIGN_EXPLORATION_MAX = 2.0  # 12 samples
    PARETO_COMPUTATION_MAX = 0.1
    
    CACHE_PUT_MAX = 0.1  # 100 operations
    CACHE_GET_MAX = 0.05  # 100 operations
    
    PROFILER_OVERHEAD_MAX = 0.001
    MEMORY_DESIGN_MAX = 1.0  # 50 accelerators


@pytest.mark.performance
class TestPerformanceRegression:
    """Performance regression tests."""
    
    def test_no_accelerator_design_regression(self):
        """Test for accelerator design performance regression."""
        designer = AcceleratorDesigner()
        
        start_time = time.time()
        result = designer.design(compute_units=64, dataflow="weight_stationary")
        duration = time.time() - start_time
        
        assert duration < PerformanceThresholds.ACCELERATOR_DESIGN_MAX
        assert result is not None
    
    def test_no_cache_regression(self):
        """Test for cache performance regression."""
        cache = AdaptiveCache(max_size=100, max_memory_mb=1.0)
        test_data = {"test": "data"}
        
        # Test put performance
        start_time = time.time()
        for i in range(50):
            cache.put(f"key_{i}", test_data)
        put_duration = time.time() - start_time
        
        # Test get performance
        start_time = time.time()
        for i in range(50):
            cache.get(f"key_{i}")
        get_duration = time.time() - start_time
        
        assert put_duration < PerformanceThresholds.CACHE_PUT_MAX / 2  # 50 ops vs 100
        assert get_duration < PerformanceThresholds.CACHE_GET_MAX / 2  # 50 ops vs 100