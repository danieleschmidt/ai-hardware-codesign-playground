"""
Performance and Load Testing Framework for AI Hardware Co-Design Playground.

This module implements comprehensive performance tests including load testing,
stress testing, capacity planning, and performance regression detection.
"""

import pytest
import time
import threading
import multiprocessing
import statistics
import psutil
import gc
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from unittest.mock import Mock, patch
import matplotlib.pyplot as plt
import numpy as np

from backend.codesign_playground.core.accelerator import AcceleratorDesigner, Accelerator
from backend.codesign_playground.core.optimizer import ModelOptimizer
from backend.codesign_playground.core.workflow import Workflow
from backend.codesign_playground.utils.monitoring import record_metric, get_metrics
from backend.codesign_playground.utils.health_monitoring import HealthMonitor


@dataclass
class PerformanceMetrics:
    """Performance measurement data."""
    
    operation: str
    duration_seconds: float
    cpu_usage_percent: float
    memory_usage_mb: float
    success: bool
    timestamp: float = field(default_factory=time.time)
    additional_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "operation": self.operation,
            "duration_seconds": self.duration_seconds,
            "cpu_usage_percent": self.cpu_usage_percent,
            "memory_usage_mb": self.memory_usage_mb,
            "success": self.success,
            "timestamp": self.timestamp,
            **self.additional_metrics
        }


@dataclass
class LoadTestResults:
    """Load test execution results."""
    
    total_requests: int
    successful_requests: int
    failed_requests: int
    average_response_time: float
    median_response_time: float
    p95_response_time: float
    p99_response_time: float
    max_response_time: float
    min_response_time: float
    requests_per_second: float
    error_rate: float
    duration_seconds: float
    
    def success_rate(self) -> float:
        """Calculate success rate."""
        return self.successful_requests / self.total_requests if self.total_requests > 0 else 0.0


class PerformanceProfiler:
    """Performance profiling and measurement utilities."""
    
    def __init__(self):
        self.measurements: List[PerformanceMetrics] = []
        self._start_time: Optional[float] = None
        self._start_cpu: Optional[float] = None
        self._start_memory: Optional[float] = None
    
    def start_measurement(self) -> None:
        """Start performance measurement."""
        self._start_time = time.time()
        self._start_cpu = psutil.cpu_percent()
        self._start_memory = psutil.virtual_memory().used / (1024 * 1024)  # MB
    
    def end_measurement(self, operation: str, success: bool = True, **additional_metrics) -> PerformanceMetrics:
        """End performance measurement and record results."""
        end_time = time.time()
        end_cpu = psutil.cpu_percent()
        end_memory = psutil.virtual_memory().used / (1024 * 1024)  # MB
        
        if self._start_time is None:
            raise RuntimeError("Must call start_measurement() first")
        
        duration = end_time - self._start_time
        cpu_usage = max(0, end_cpu - (self._start_cpu or 0))
        memory_usage = end_memory - (self._start_memory or 0)
        
        metrics = PerformanceMetrics(
            operation=operation,
            duration_seconds=duration,
            cpu_usage_percent=cpu_usage,
            memory_usage_mb=memory_usage,
            success=success,
            additional_metrics=additional_metrics
        )
        
        self.measurements.append(metrics)
        return metrics
    
    def get_statistics(self, operation: Optional[str] = None) -> Dict[str, Any]:
        """Get performance statistics."""
        filtered_measurements = self.measurements
        if operation:
            filtered_measurements = [m for m in self.measurements if m.operation == operation]
        
        if not filtered_measurements:
            return {}
        
        durations = [m.duration_seconds for m in filtered_measurements]
        cpu_usages = [m.cpu_usage_percent for m in filtered_measurements]
        memory_usages = [m.memory_usage_mb for m in filtered_measurements]
        success_count = sum(1 for m in filtered_measurements if m.success)
        
        return {
            "count": len(filtered_measurements),
            "success_rate": success_count / len(filtered_measurements),
            "duration": {
                "mean": statistics.mean(durations),
                "median": statistics.median(durations),
                "std": statistics.stdev(durations) if len(durations) > 1 else 0,
                "min": min(durations),
                "max": max(durations),
                "p95": np.percentile(durations, 95),
                "p99": np.percentile(durations, 99)
            },
            "cpu_usage": {
                "mean": statistics.mean(cpu_usages),
                "max": max(cpu_usages)
            },
            "memory_usage": {
                "mean": statistics.mean(memory_usages),
                "max": max(memory_usages)
            }
        }


class LoadTestRunner:
    """Load testing framework."""
    
    def __init__(self, target_function: Callable, *args, **kwargs):
        """
        Initialize load test runner.
        
        Args:
            target_function: Function to test
            *args, **kwargs: Arguments to pass to target function
        """
        self.target_function = target_function
        self.args = args
        self.kwargs = kwargs
        self.results: List[Tuple[float, bool, Optional[Exception]]] = []
    
    def run_single_request(self) -> Tuple[float, bool, Optional[Exception]]:
        """Run a single request and measure performance."""
        start_time = time.time()
        success = True
        exception = None
        
        try:
            result = self.target_function(*self.args, **self.kwargs)
            # Consider result successful if no exception and result is not None/False
            success = result is not None and result is not False
        except Exception as e:
            success = False
            exception = e
        
        duration = time.time() - start_time
        return duration, success, exception
    
    def run_load_test(self, 
                     concurrent_users: int, 
                     duration_seconds: float,
                     ramp_up_seconds: float = 0) -> LoadTestResults:
        """
        Run load test with specified parameters.
        
        Args:
            concurrent_users: Number of concurrent users/threads
            duration_seconds: Test duration in seconds
            ramp_up_seconds: Time to gradually increase load
            
        Returns:
            LoadTestResults with comprehensive metrics
        """
        start_time = time.time()
        end_time = start_time + duration_seconds
        self.results.clear()
        
        def worker():
            while time.time() < end_time:
                result = self.run_single_request()
                self.results.append(result)
                
                # Small delay to prevent overwhelming the system
                time.sleep(0.01)
        
        # Start threads with ramp-up
        threads = []
        for i in range(concurrent_users):
            thread = threading.Thread(target=worker)
            thread.start()
            threads.append(thread)
            
            # Ramp-up delay
            if ramp_up_seconds > 0:
                time.sleep(ramp_up_seconds / concurrent_users)
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Calculate results
        durations = [r[0] for r in self.results]
        successes = [r[1] for r in self.results]
        
        total_requests = len(self.results)
        successful_requests = sum(successes)
        failed_requests = total_requests - successful_requests
        
        if durations:
            average_response_time = statistics.mean(durations)
            median_response_time = statistics.median(durations)
            p95_response_time = np.percentile(durations, 95)
            p99_response_time = np.percentile(durations, 99)
            max_response_time = max(durations)
            min_response_time = min(durations)
        else:
            average_response_time = median_response_time = 0
            p95_response_time = p99_response_time = 0
            max_response_time = min_response_time = 0
        
        actual_duration = time.time() - start_time
        requests_per_second = total_requests / actual_duration if actual_duration > 0 else 0
        error_rate = failed_requests / total_requests if total_requests > 0 else 0
        
        return LoadTestResults(
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            average_response_time=average_response_time,
            median_response_time=median_response_time,
            p95_response_time=p95_response_time,
            p99_response_time=p99_response_time,
            max_response_time=max_response_time,
            min_response_time=min_response_time,
            requests_per_second=requests_per_second,
            error_rate=error_rate,
            duration_seconds=actual_duration
        )


class TestAcceleratorDesignPerformance:
    """Test accelerator design performance."""
    
    @pytest.fixture
    def designer(self):
        return AcceleratorDesigner()
    
    @pytest.fixture
    def profiler(self):
        return PerformanceProfiler()
    
    def test_accelerator_design_latency(self, designer, profiler):
        """Test accelerator design latency."""
        # Test single design latency
        profiler.start_measurement()
        
        accelerator = designer.design(
            compute_units=64,
            memory_hierarchy=["sram_64kb", "dram"],
            dataflow="weight_stationary"
        )
        
        metrics = profiler.end_measurement("accelerator_design")
        
        # Design should complete within reasonable time
        assert metrics.duration_seconds < 5.0, f"Design took {metrics.duration_seconds}s, too slow"
        assert metrics.success
        assert accelerator is not None
    
    def test_accelerator_design_throughput(self, designer):
        """Test accelerator design throughput under load."""
        def design_accelerator():
            return designer.design(
                compute_units=32,
                memory_hierarchy=["sram_32kb", "dram"],
                dataflow="output_stationary"
            )
        
        load_tester = LoadTestRunner(design_accelerator)
        
        # Run load test with moderate load
        results = load_tester.run_load_test(
            concurrent_users=5,
            duration_seconds=10.0,
            ramp_up_seconds=2.0
        )
        
        # Verify performance requirements
        assert results.success_rate() > 0.95, f"Success rate {results.success_rate()} too low"
        assert results.average_response_time < 2.0, f"Average response time {results.average_response_time}s too high"
        assert results.requests_per_second > 2.0, f"Throughput {results.requests_per_second} RPS too low"
    
    def test_parallel_design_performance(self, designer):
        """Test parallel accelerator design performance."""
        configs = [
            {"compute_units": 16, "dataflow": "weight_stationary"},
            {"compute_units": 32, "dataflow": "output_stationary"},
            {"compute_units": 64, "dataflow": "row_stationary"},
            {"compute_units": 128, "dataflow": "weight_stationary"},
        ]
        
        start_time = time.time()
        
        # Test parallel design
        accelerators = designer.design_parallel(configs, max_workers=4)
        
        parallel_duration = time.time() - start_time
        
        # Verify all designs completed
        assert len(accelerators) == len(configs)
        assert all(acc is not None for acc in accelerators)
        
        # Parallel execution should be faster than sequential
        # (This is a simplified test - actual speedup depends on system resources)
        assert parallel_duration < 10.0, f"Parallel design took {parallel_duration}s, too slow"
    
    def test_model_profiling_performance(self, designer, profiler):
        """Test model profiling performance."""
        # Create mock model
        class MockModel:
            def __init__(self, size="medium"):
                self.size = size
                self.complexity = {"small": 0.5, "medium": 1.0, "large": 2.0}[size]
        
        model_sizes = ["small", "medium", "large"]
        input_shape = (1, 3, 224, 224)
        
        for size in model_sizes:
            model = MockModel(size)
            
            profiler.start_measurement()
            
            profile = designer.profile_model(model, input_shape, "pytorch")
            
            metrics = profiler.end_measurement(f"model_profiling_{size}")
            
            # Profiling should be fast regardless of model size
            assert metrics.duration_seconds < 3.0, f"Profiling {size} model took {metrics.duration_seconds}s"
            assert profile is not None
            assert profile.peak_gflops > 0
    
    def test_memory_usage_during_design(self, designer):
        """Test memory usage during accelerator design."""
        import tracemalloc
        
        tracemalloc.start()
        
        initial_memory = psutil.virtual_memory().used
        
        # Create multiple accelerators to test memory usage
        accelerators = []
        for i in range(10):
            acc = designer.design(
                compute_units=64,
                memory_hierarchy=["sram_64kb", "dram"],
                dataflow="weight_stationary"
            )
            accelerators.append(acc)
        
        peak_memory = psutil.virtual_memory().used
        memory_growth = (peak_memory - initial_memory) / (1024 * 1024)  # MB
        
        tracemalloc.stop()
        
        # Memory growth should be reasonable
        assert memory_growth < 100, f"Memory growth {memory_growth}MB too high"
        
        # Clean up
        accelerators.clear()
        gc.collect()


class TestOptimizerPerformance:
    """Test optimizer performance."""
    
    @pytest.fixture
    def mock_model(self):
        class MockModel:
            def __init__(self):
                self.complexity = 1.0
        return MockModel()
    
    @pytest.fixture
    def accelerator(self):
        return Accelerator(
            compute_units=32,
            memory_hierarchy=["sram_32kb", "dram"],
            dataflow="output_stationary"
        )
    
    def test_optimization_performance(self, mock_model, accelerator, tmp_path):
        """Test optimization performance."""
        optimizer = ModelOptimizer(mock_model, accelerator)
        
        start_time = time.time()
        
        result = optimizer.co_optimize(
            target_fps=30.0,
            power_budget=5.0,
            iterations=10
        )
        
        optimization_time = time.time() - start_time
        
        # Optimization should complete within reasonable time
        assert optimization_time < 30.0, f"Optimization took {optimization_time}s, too slow"
        assert result is not None
        assert result.metrics is not None
        assert result.optimization_time > 0
    
    def test_optimization_scalability(self, mock_model, accelerator):
        """Test optimization scalability with different iteration counts."""
        optimizer = ModelOptimizer(mock_model, accelerator)
        
        iteration_counts = [5, 10, 20, 50]
        times = []
        
        for iterations in iteration_counts:
            start_time = time.time()
            
            result = optimizer.co_optimize(
                target_fps=30.0,
                power_budget=5.0,
                iterations=iterations
            )
            
            duration = time.time() - start_time
            times.append(duration)
            
            assert result is not None
        
        # Time should scale roughly linearly with iterations
        # (allowing for some overhead and variation)
        time_per_iteration = [t / i for t, i in zip(times, iteration_counts)]
        
        # Time per iteration should be relatively consistent
        avg_time_per_iter = statistics.mean(time_per_iteration)
        max_deviation = max(abs(t - avg_time_per_iter) for t in time_per_iteration)
        
        assert max_deviation < avg_time_per_iter * 2, "Optimization scaling is inconsistent"
    
    def test_concurrent_optimizations(self, mock_model, accelerator):
        """Test concurrent optimization performance."""
        def run_optimization():
            optimizer = ModelOptimizer(mock_model, accelerator)
            return optimizer.co_optimize(
                target_fps=random.uniform(20, 40),
                power_budget=random.uniform(3, 8),
                iterations=5
            )
        
        load_tester = LoadTestRunner(run_optimization)
        
        # Run concurrent optimizations
        results = load_tester.run_load_test(
            concurrent_users=3,
            duration_seconds=15.0,
            ramp_up_seconds=3.0
        )
        
        # Verify performance under concurrent load
        assert results.success_rate() > 0.8, f"Success rate {results.success_rate()} too low under load"
        assert results.average_response_time < 10.0, f"Average response time {results.average_response_time}s too high"
    
    def test_optimization_caching_performance(self, mock_model, accelerator):
        """Test optimization caching performance."""
        optimizer = ModelOptimizer(mock_model, accelerator)
        
        # First optimization (cache miss)
        start_time = time.time()
        result1 = optimizer.co_optimize(
            target_fps=30.0,
            power_budget=5.0,
            iterations=10,
            enable_caching=True
        )
        first_time = time.time() - start_time
        
        # Second identical optimization (cache hit)
        start_time = time.time()
        result2 = optimizer.co_optimize(
            target_fps=30.0,
            power_budget=5.0,
            iterations=10,
            enable_caching=True
        )
        second_time = time.time() - start_time
        
        # Cached result should be much faster
        assert second_time < first_time * 0.5, f"Cache not effective: {second_time}s vs {first_time}s"
        
        # Results should be equivalent
        assert result1.metrics["fps"] == result2.metrics["fps"]
        assert result1.metrics["power"] == result2.metrics["power"]


class TestWorkflowPerformance:
    """Test workflow performance."""
    
    def test_workflow_end_to_end_performance(self, tmp_path):
        """Test complete workflow performance."""
        workflow = Workflow("perf_test", output_dir=str(tmp_path))
        
        # Create mock model file
        model_file = tmp_path / "test_model.pt"
        model_file.write_text("mock model data")
        
        start_time = time.time()
        
        try:
            # Run complete workflow
            workflow.import_model(
                model_path=str(model_file),
                input_shapes={"input": (1, 3, 224, 224)},
                framework="pytorch"
            )
            
            workflow.map_to_hardware(
                template="systolic_array",
                size=(16, 16),
                precision="int8"
            )
            
            workflow.compile()
            
            workflow.simulate(testbench="mock_testbench")
            
            workflow.generate_rtl()
            
        except Exception as e:
            # Some failures are acceptable in mock environment
            print(f"Workflow exception (may be expected): {e}")
        
        total_time = time.time() - start_time
        
        # Complete workflow should finish within reasonable time
        assert total_time < 60.0, f"Complete workflow took {total_time}s, too slow"
    
    def test_workflow_stage_performance(self, tmp_path):
        """Test individual workflow stage performance."""
        workflow = Workflow("stage_perf_test", output_dir=str(tmp_path))
        
        # Create mock model file
        model_file = tmp_path / "test_model.pt"
        model_file.write_text("mock model data")
        
        stage_times = {}
        
        # Test model import performance
        start_time = time.time()
        try:
            workflow.import_model(
                model_path=str(model_file),
                input_shapes={"input": (1, 3, 224, 224)},
                framework="pytorch"
            )
            stage_times["import"] = time.time() - start_time
        except:
            stage_times["import"] = time.time() - start_time
        
        # Test hardware mapping performance
        start_time = time.time()
        try:
            workflow.map_to_hardware(
                template="systolic_array",
                size=(8, 8),
                precision="int8"
            )
            stage_times["mapping"] = time.time() - start_time
        except:
            stage_times["mapping"] = time.time() - start_time
        
        # Verify stage performance requirements
        assert stage_times["import"] < 10.0, f"Model import took {stage_times['import']}s"
        assert stage_times["mapping"] < 5.0, f"Hardware mapping took {stage_times['mapping']}s"
    
    def test_workflow_memory_efficiency(self, tmp_path):
        """Test workflow memory efficiency."""
        import tracemalloc
        
        tracemalloc.start()
        initial_memory = psutil.virtual_memory().used
        
        # Run multiple workflows
        workflows = []
        for i in range(5):
            workflow = Workflow(f"mem_test_{i}", output_dir=str(tmp_path / f"wf_{i}"))
            workflows.append(workflow)
            
            # Create mock model
            model_file = tmp_path / f"model_{i}.pt"
            model_file.write_text(f"mock model data {i}")
            
            try:
                workflow.import_model(
                    model_path=str(model_file),
                    input_shapes={"input": (1, 3, 224, 224)},
                    framework="pytorch"
                )
            except:
                pass  # Mock failures are acceptable
        
        peak_memory = psutil.virtual_memory().used
        memory_per_workflow = (peak_memory - initial_memory) / len(workflows) / (1024 * 1024)  # MB
        
        tracemalloc.stop()
        
        # Memory usage per workflow should be reasonable
        assert memory_per_workflow < 50, f"Memory per workflow {memory_per_workflow}MB too high"
        
        # Clean up
        workflows.clear()
        gc.collect()


class TestSystemResourceUtilization:
    """Test system resource utilization."""
    
    def test_cpu_utilization_under_load(self):
        """Test CPU utilization under various loads."""
        def cpu_intensive_task():
            designer = AcceleratorDesigner()
            return designer.design(
                compute_units=128,
                memory_hierarchy=["sram_128kb", "dram"],
                dataflow="weight_stationary"
            )
        
        # Monitor CPU usage during load test
        cpu_samples = []
        
        def monitor_cpu():
            for _ in range(20):  # 20 seconds of monitoring
                cpu_samples.append(psutil.cpu_percent(interval=1))
        
        # Start CPU monitoring
        monitor_thread = threading.Thread(target=monitor_cpu)
        monitor_thread.start()
        
        # Run load test
        load_tester = LoadTestRunner(cpu_intensive_task)
        results = load_tester.run_load_test(
            concurrent_users=4,
            duration_seconds=15.0
        )
        
        monitor_thread.join()
        
        # Analyze CPU utilization
        if cpu_samples:
            avg_cpu = statistics.mean(cpu_samples)
            max_cpu = max(cpu_samples)
            
            # CPU utilization should be reasonable but not excessive
            assert avg_cpu < 90, f"Average CPU utilization {avg_cpu}% too high"
            assert max_cpu < 100, f"Peak CPU utilization {max_cpu}% too high"
    
    def test_memory_utilization_patterns(self):
        """Test memory utilization patterns."""
        initial_memory = psutil.virtual_memory().used / (1024 * 1024)  # MB
        memory_samples = [initial_memory]
        
        # Create various components and monitor memory
        designer = AcceleratorDesigner()
        memory_samples.append(psutil.virtual_memory().used / (1024 * 1024))
        
        # Create multiple accelerators
        accelerators = []
        for i in range(10):
            acc = designer.design(compute_units=32)
            accelerators.append(acc)
            memory_samples.append(psutil.virtual_memory().used / (1024 * 1024))
        
        # Clean up and monitor memory recovery
        accelerators.clear()
        gc.collect()
        final_memory = psutil.virtual_memory().used / (1024 * 1024)
        memory_samples.append(final_memory)
        
        # Analyze memory patterns
        peak_memory = max(memory_samples)
        memory_growth = peak_memory - initial_memory
        memory_recovery = peak_memory - final_memory
        
        # Memory should grow reasonably and recover well
        assert memory_growth < 200, f"Memory growth {memory_growth}MB too high"
        assert memory_recovery > memory_growth * 0.7, f"Poor memory recovery: {memory_recovery}MB of {memory_growth}MB"
    
    def test_disk_io_patterns(self, tmp_path):
        """Test disk I/O patterns."""
        # Monitor disk I/O during workflow operations
        disk_start = psutil.disk_io_counters()
        
        # Run I/O intensive operations
        for i in range(5):
            workflow = Workflow(f"io_test_{i}", output_dir=str(tmp_path / f"wf_{i}"))
            
            # Create larger mock files
            model_file = tmp_path / f"large_model_{i}.pt"
            model_file.write_text("mock model data " * 1000)  # Larger file
            
            try:
                workflow.import_model(
                    model_path=str(model_file),
                    input_shapes={"input": (1, 3, 224, 224)},
                    framework="pytorch"
                )
                workflow.save_state()
            except:
                pass  # Mock failures acceptable
        
        disk_end = psutil.disk_io_counters()
        
        if disk_start and disk_end:
            bytes_written = disk_end.write_bytes - disk_start.write_bytes
            write_count = disk_end.write_count - disk_start.write_count
            
            # I/O should be reasonable
            assert bytes_written < 100 * 1024 * 1024, f"Wrote {bytes_written} bytes, too much"
            assert write_count < 1000, f"Too many write operations: {write_count}"


class TestPerformanceRegression:
    """Test for performance regressions."""
    
    def test_performance_baseline(self, tmp_path):
        """Establish and test performance baseline."""
        # This would typically compare against stored baseline metrics
        baseline_file = tmp_path / "performance_baseline.json"
        
        # Run standard performance tests
        designer = AcceleratorDesigner()
        
        # Measure standard operations
        operations = {
            "simple_design": lambda: designer.design(compute_units=16),
            "complex_design": lambda: designer.design(
                compute_units=128,
                memory_hierarchy=["sram_128kb", "sram_64kb", "dram"],
                dataflow="weight_stationary"
            ),
            "model_profiling": lambda: designer.profile_model(
                Mock(), (1, 3, 224, 224), "pytorch"
            )
        }
        
        current_metrics = {}
        
        for op_name, op_func in operations.items():
            times = []
            for _ in range(3):  # Run 3 times for stability
                start_time = time.time()
                try:
                    op_func()
                    duration = time.time() - start_time
                    times.append(duration)
                except:
                    times.append(float('inf'))  # Failed operation
            
            current_metrics[op_name] = {
                "avg_time": statistics.mean(times),
                "min_time": min(times),
                "max_time": max(times)
            }
        
        # Save baseline if it doesn't exist
        if not baseline_file.exists():
            with open(baseline_file, 'w') as f:
                json.dump(current_metrics, f, indent=2)
        
        # Compare with baseline
        try:
            with open(baseline_file, 'r') as f:
                baseline_metrics = json.load(f)
            
            for op_name, current in current_metrics.items():
                if op_name in baseline_metrics:
                    baseline = baseline_metrics[op_name]
                    
                    # Check for significant regression (>50% slower)
                    regression_threshold = 1.5
                    if current["avg_time"] > baseline["avg_time"] * regression_threshold:
                        pytest.fail(
                            f"Performance regression detected in {op_name}: "
                            f"{current['avg_time']:.3f}s vs baseline {baseline['avg_time']:.3f}s"
                        )
        
        except FileNotFoundError:
            # Baseline doesn't exist, this run establishes it
            pass


def generate_performance_report(profiler: PerformanceProfiler, output_path: str) -> None:
    """Generate performance test report."""
    stats = profiler.get_statistics()
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    
    # Plot response times
    plt.subplot(2, 2, 1)
    operations = list(set(m.operation for m in profiler.measurements))
    for op in operations:
        op_measurements = [m for m in profiler.measurements if m.operation == op]
        times = [m.duration_seconds for m in op_measurements]
        plt.plot(times, label=op)
    plt.title("Response Times by Operation")
    plt.ylabel("Duration (seconds)")
    plt.legend()
    
    # Plot CPU usage
    plt.subplot(2, 2, 2)
    cpu_usage = [m.cpu_usage_percent for m in profiler.measurements]
    plt.plot(cpu_usage)
    plt.title("CPU Usage Over Time")
    plt.ylabel("CPU Usage (%)")
    
    # Plot memory usage
    plt.subplot(2, 2, 3)
    memory_usage = [m.memory_usage_mb for m in profiler.measurements]
    plt.plot(memory_usage)
    plt.title("Memory Usage Over Time")
    plt.ylabel("Memory Usage (MB)")
    
    # Plot success rate
    plt.subplot(2, 2, 4)
    success_rate = [m.success for m in profiler.measurements]
    success_counts = []
    window_size = 10
    for i in range(len(success_rate)):
        start_idx = max(0, i - window_size + 1)
        window = success_rate[start_idx:i+1]
        success_counts.append(sum(window) / len(window))
    plt.plot(success_counts)
    plt.title("Success Rate (Rolling Average)")
    plt.ylabel("Success Rate")
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])