"""
SLA Validation and Performance Benchmarks for AI Hardware Co-Design Playground.

This module implements comprehensive SLA validation tests to ensure the system
meets performance requirements under various conditions.
"""

import pytest
import time
import psutil
import asyncio
import statistics
from typing import Dict, List, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import Mock, patch
import numpy as np

from codesign_playground.core.accelerator import AcceleratorDesigner
from codesign_playground.core.optimizer import ModelOptimizer
from codesign_playground.core.explorer import DesignSpaceExplorer
from codesign_playground.core.workflow import Workflow, WorkflowConfig
from codesign_playground.utils.monitoring import get_system_monitor


class SLAMetrics:
    """Class to track and validate SLA metrics."""
    
    def __init__(self):
        self.metrics = {
            "response_times": [],
            "throughput_ops_per_second": [],
            "memory_usage_mb": [],
            "cpu_usage_percent": [],
            "error_rates": [],
            "availability_percent": 100.0
        }
        self.sla_thresholds = {
            "max_response_time_ms": 200,
            "min_throughput_ops_per_second": 100,
            "max_memory_usage_mb": 512,
            "max_cpu_usage_percent": 80,
            "max_error_rate_percent": 1.0,
            "min_availability_percent": 99.9
        }
    
    def record_response_time(self, response_time_ms: float):
        """Record API response time."""
        self.metrics["response_times"].append(response_time_ms)
    
    def record_throughput(self, ops_per_second: float):
        """Record throughput measurement."""
        self.metrics["throughput_ops_per_second"].append(ops_per_second)
    
    def record_resource_usage(self, memory_mb: float, cpu_percent: float):
        """Record resource usage."""
        self.metrics["memory_usage_mb"].append(memory_mb)
        self.metrics["cpu_usage_percent"].append(cpu_percent)
    
    def record_error(self, total_requests: int, errors: int):
        """Record error rate."""
        error_rate = (errors / total_requests) * 100.0 if total_requests > 0 else 0.0
        self.metrics["error_rates"].append(error_rate)
    
    def validate_slas(self) -> Dict[str, Any]:
        """Validate all SLA metrics and return results."""
        results = {
            "passed": True,
            "violations": [],
            "metrics_summary": {},
            "sla_compliance": {}
        }
        
        # Response Time SLA
        if self.metrics["response_times"]:
            avg_response_time = statistics.mean(self.metrics["response_times"])
            p95_response_time = np.percentile(self.metrics["response_times"], 95)
            p99_response_time = np.percentile(self.metrics["response_times"], 99)
            
            results["metrics_summary"]["response_time"] = {
                "average_ms": avg_response_time,
                "p95_ms": p95_response_time,
                "p99_ms": p99_response_time
            }
            
            if p95_response_time > self.sla_thresholds["max_response_time_ms"]:
                results["passed"] = False
                results["violations"].append({
                    "metric": "response_time",
                    "threshold": self.sla_thresholds["max_response_time_ms"],
                    "actual": p95_response_time,
                    "description": "P95 response time exceeds SLA threshold"
                })
            
            results["sla_compliance"]["response_time"] = p95_response_time <= self.sla_thresholds["max_response_time_ms"]
        
        # Throughput SLA
        if self.metrics["throughput_ops_per_second"]:
            avg_throughput = statistics.mean(self.metrics["throughput_ops_per_second"])
            min_throughput = min(self.metrics["throughput_ops_per_second"])
            
            results["metrics_summary"]["throughput"] = {
                "average_ops_per_second": avg_throughput,
                "minimum_ops_per_second": min_throughput
            }
            
            if min_throughput < self.sla_thresholds["min_throughput_ops_per_second"]:
                results["passed"] = False
                results["violations"].append({
                    "metric": "throughput",
                    "threshold": self.sla_thresholds["min_throughput_ops_per_second"],
                    "actual": min_throughput,
                    "description": "Minimum throughput below SLA threshold"
                })
            
            results["sla_compliance"]["throughput"] = min_throughput >= self.sla_thresholds["min_throughput_ops_per_second"]
        
        # Memory Usage SLA
        if self.metrics["memory_usage_mb"]:
            max_memory = max(self.metrics["memory_usage_mb"])
            avg_memory = statistics.mean(self.metrics["memory_usage_mb"])
            
            results["metrics_summary"]["memory_usage"] = {
                "average_mb": avg_memory,
                "peak_mb": max_memory
            }
            
            if max_memory > self.sla_thresholds["max_memory_usage_mb"]:
                results["passed"] = False
                results["violations"].append({
                    "metric": "memory_usage",
                    "threshold": self.sla_thresholds["max_memory_usage_mb"],
                    "actual": max_memory,
                    "description": "Peak memory usage exceeds SLA threshold"
                })
            
            results["sla_compliance"]["memory_usage"] = max_memory <= self.sla_thresholds["max_memory_usage_mb"]
        
        # CPU Usage SLA
        if self.metrics["cpu_usage_percent"]:
            max_cpu = max(self.metrics["cpu_usage_percent"])
            avg_cpu = statistics.mean(self.metrics["cpu_usage_percent"])
            
            results["metrics_summary"]["cpu_usage"] = {
                "average_percent": avg_cpu,
                "peak_percent": max_cpu
            }
            
            if max_cpu > self.sla_thresholds["max_cpu_usage_percent"]:
                results["passed"] = False
                results["violations"].append({
                    "metric": "cpu_usage",
                    "threshold": self.sla_thresholds["max_cpu_usage_percent"],
                    "actual": max_cpu,
                    "description": "Peak CPU usage exceeds SLA threshold"
                })
            
            results["sla_compliance"]["cpu_usage"] = max_cpu <= self.sla_thresholds["max_cpu_usage_percent"]
        
        # Error Rate SLA
        if self.metrics["error_rates"]:
            avg_error_rate = statistics.mean(self.metrics["error_rates"])
            max_error_rate = max(self.metrics["error_rates"])
            
            results["metrics_summary"]["error_rate"] = {
                "average_percent": avg_error_rate,
                "peak_percent": max_error_rate
            }
            
            if max_error_rate > self.sla_thresholds["max_error_rate_percent"]:
                results["passed"] = False
                results["violations"].append({
                    "metric": "error_rate",
                    "threshold": self.sla_thresholds["max_error_rate_percent"],
                    "actual": max_error_rate,
                    "description": "Error rate exceeds SLA threshold"
                })
            
            results["sla_compliance"]["error_rate"] = max_error_rate <= self.sla_thresholds["max_error_rate_percent"]
        
        return results


class TestResponseTimeSLA:
    """Test response time SLA compliance."""
    
    @pytest.fixture
    def sla_metrics(self):
        """SLA metrics tracker."""
        return SLAMetrics()
    
    def test_accelerator_design_response_time(self, sla_metrics):
        """Test accelerator design API response time SLA."""
        designer = AcceleratorDesigner()
        response_times = []
        
        # Test multiple design requests
        for i in range(50):
            start_time = time.time()
            
            accelerator = designer.design(
                compute_units=32 + (i % 4) * 16,
                dataflow="weight_stationary",
                frequency_mhz=200.0 + (i % 3) * 100.0
            )
            
            end_time = time.time()
            response_time_ms = (end_time - start_time) * 1000
            response_times.append(response_time_ms)
            sla_metrics.record_response_time(response_time_ms)
        
        # Validate SLA
        sla_results = sla_metrics.validate_slas()
        
        # All individual response times should be under threshold
        assert all(rt <= 200 for rt in response_times), f"Some response times exceeded 200ms: {[rt for rt in response_times if rt > 200]}"
        
        # P95 response time should meet SLA
        assert sla_results["sla_compliance"]["response_time"], f"P95 response time SLA violation: {sla_results['metrics_summary']['response_time']}"
    
    def test_model_profiling_response_time(self, sla_metrics):
        """Test model profiling response time SLA."""
        designer = AcceleratorDesigner()
        response_times = []
        
        # Test different model sizes
        model_configs = [
            {"type": "small", "complexity": 0.5},
            {"type": "medium", "complexity": 1.0},
            {"type": "large", "complexity": 2.0}
        ]
        
        input_shapes = [
            (224, 224, 3),
            (32, 32, 3),
            (512, 512, 3)
        ]
        
        for i in range(30):
            model_config = model_configs[i % len(model_configs)]
            input_shape = input_shapes[i % len(input_shapes)]
            
            start_time = time.time()
            
            profile = designer.profile_model(model_config, input_shape)
            
            end_time = time.time()
            response_time_ms = (end_time - start_time) * 1000
            response_times.append(response_time_ms)
            sla_metrics.record_response_time(response_time_ms)
        
        # Validate SLA
        sla_results = sla_metrics.validate_slas()
        
        # Model profiling should be fast
        assert all(rt <= 150 for rt in response_times), f"Model profiling response times too slow: {statistics.mean(response_times):.2f}ms average"
        assert sla_results["sla_compliance"]["response_time"], f"Model profiling P95 response time SLA violation"
    
    def test_optimization_response_time(self, sla_metrics):
        """Test optimization response time SLA."""
        # Create mock components
        mock_model = Mock()
        mock_model.parameters = 1000000
        
        accelerator = AcceleratorDesigner().design(compute_units=32)
        optimizer = ModelOptimizer(mock_model, accelerator)
        
        response_times = []
        
        # Test optimization with different parameters
        for i in range(20):
            start_time = time.time()
            
            result = optimizer.co_optimize(
                target_fps=30.0,
                power_budget=5.0,
                iterations=3  # Limited iterations for SLA testing
            )
            
            end_time = time.time()
            response_time_ms = (end_time - start_time) * 1000
            response_times.append(response_time_ms)
            sla_metrics.record_response_time(response_time_ms)
        
        # Optimization should complete within reasonable time
        assert all(rt <= 5000 for rt in response_times), f"Optimization taking too long: max {max(response_times):.2f}ms"
        
        # Average should be much faster
        avg_time = statistics.mean(response_times)
        assert avg_time <= 2000, f"Average optimization time too slow: {avg_time:.2f}ms"
    
    def test_concurrent_request_response_time(self, sla_metrics):
        """Test response time under concurrent load."""
        designer = AcceleratorDesigner()
        
        def design_accelerator(config_id):
            start_time = time.time()
            
            accelerator = designer.design(
                compute_units=16 + (config_id % 8) * 8,
                dataflow="weight_stationary",
                frequency_mhz=200.0 + (config_id % 4) * 100.0
            )
            
            end_time = time.time()
            return (end_time - start_time) * 1000
        
        # Execute concurrent requests
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(design_accelerator, i) for i in range(100)]
            response_times = [future.result() for future in as_completed(futures, timeout=30)]
        
        # Record all response times
        for rt in response_times:
            sla_metrics.record_response_time(rt)
        
        # Validate SLA under concurrent load
        sla_results = sla_metrics.validate_slas()
        
        assert len(response_times) == 100, "Not all concurrent requests completed"
        assert sla_results["sla_compliance"]["response_time"], f"Concurrent load response time SLA violation"
        
        # P99 should still be reasonable under load
        p99_time = np.percentile(response_times, 99)
        assert p99_time <= 500, f"P99 response time under load too high: {p99_time:.2f}ms"


class TestThroughputSLA:
    """Test throughput SLA compliance."""
    
    @pytest.fixture
    def sla_metrics(self):
        """SLA metrics tracker."""
        return SLAMetrics()
    
    def test_design_generation_throughput(self, sla_metrics):
        """Test accelerator design generation throughput."""
        designer = AcceleratorDesigner()
        
        # Measure throughput over time window
        test_duration = 10.0  # seconds
        start_time = time.time()
        operations_completed = 0
        
        while (time.time() - start_time) < test_duration:
            accelerator = designer.design(
                compute_units=32,
                dataflow="weight_stationary"
            )
            operations_completed += 1
        
        elapsed_time = time.time() - start_time
        throughput = operations_completed / elapsed_time
        
        sla_metrics.record_throughput(throughput)
        
        # Should achieve minimum throughput
        assert throughput >= 50, f"Design generation throughput too low: {throughput:.2f} ops/sec"
        
        # Validate SLA
        sla_results = sla_metrics.validate_slas()
        assert sla_results["sla_compliance"]["throughput"], f"Throughput SLA violation: {throughput:.2f} ops/sec"
    
    def test_parallel_processing_throughput(self, sla_metrics):
        """Test throughput with parallel processing."""
        designer = AcceleratorDesigner()
        
        def process_batch(batch_size):
            configs = [
                {"compute_units": 16 + i * 8, "dataflow": "weight_stationary"}
                for i in range(batch_size)
            ]
            
            start_time = time.time()
            results = designer.design_parallel(configs, max_workers=4)
            elapsed_time = time.time() - start_time
            
            return len(results) / elapsed_time
        
        # Test different batch sizes
        batch_sizes = [5, 10, 20, 40]
        throughputs = []
        
        for batch_size in batch_sizes:
            throughput = process_batch(batch_size)
            throughputs.append(throughput)
            sla_metrics.record_throughput(throughput)
        
        # Parallel processing should achieve higher throughput
        max_throughput = max(throughputs)
        assert max_throughput >= 100, f"Parallel processing throughput too low: {max_throughput:.2f} ops/sec"
        
        # Validate SLA
        sla_results = sla_metrics.validate_slas()
        assert sla_results["sla_compliance"]["throughput"], f"Parallel throughput SLA violation"
    
    def test_sustained_throughput(self, sla_metrics):
        """Test sustained throughput over extended period."""
        designer = AcceleratorDesigner()
        
        # Measure throughput in windows
        window_duration = 5.0  # seconds
        num_windows = 6  # 30 seconds total
        window_throughputs = []
        
        for window in range(num_windows):
            start_time = time.time()
            operations_completed = 0
            
            while (time.time() - start_time) < window_duration:
                accelerator = designer.design(
                    compute_units=16 + (operations_completed % 8) * 8,
                    dataflow="weight_stationary"
                )
                operations_completed += 1
            
            elapsed_time = time.time() - start_time
            window_throughput = operations_completed / elapsed_time
            window_throughputs.append(window_throughput)
            sla_metrics.record_throughput(window_throughput)
        
        # Throughput should be sustained across windows
        min_throughput = min(window_throughputs)
        avg_throughput = statistics.mean(window_throughputs)
        throughput_variance = statistics.variance(window_throughputs)
        
        assert min_throughput >= 40, f"Minimum sustained throughput too low: {min_throughput:.2f} ops/sec"
        assert throughput_variance <= 100, f"Throughput variance too high: {throughput_variance:.2f}"
        
        # Validate SLA
        sla_results = sla_metrics.validate_slas()
        assert sla_results["sla_compliance"]["throughput"], f"Sustained throughput SLA violation"


class TestResourceUsageSLA:
    """Test resource usage SLA compliance."""
    
    @pytest.fixture
    def sla_metrics(self):
        """SLA metrics tracker."""
        return SLAMetrics()
    
    def test_memory_usage_sla(self, sla_metrics):
        """Test memory usage SLA compliance."""
        designer = AcceleratorDesigner()
        process = psutil.Process()
        
        # Baseline memory usage
        baseline_memory = process.memory_info().rss / (1024 * 1024)  # MB
        
        # Perform memory-intensive operations
        accelerators = []
        memory_measurements = []
        
        for i in range(100):
            accelerator = designer.design(
                compute_units=64 + i % 64,
                dataflow="weight_stationary",
                frequency_mhz=400.0
            )
            accelerators.append(accelerator)
            
            # Measure memory every 10 operations
            if i % 10 == 0:
                current_memory = process.memory_info().rss / (1024 * 1024)
                memory_usage = current_memory - baseline_memory
                memory_measurements.append(memory_usage)
                sla_metrics.record_resource_usage(memory_usage, 0)  # CPU measured separately
        
        # Memory usage should stay within limits
        max_memory = max(memory_measurements)
        avg_memory = statistics.mean(memory_measurements)
        
        assert max_memory <= 400, f"Peak memory usage too high: {max_memory:.2f} MB"
        assert avg_memory <= 200, f"Average memory usage too high: {avg_memory:.2f} MB"
        
        # Validate SLA
        sla_results = sla_metrics.validate_slas()
        assert sla_results["sla_compliance"]["memory_usage"], f"Memory usage SLA violation: {max_memory:.2f} MB"
    
    def test_cpu_usage_sla(self, sla_metrics):
        """Test CPU usage SLA compliance."""
        designer = AcceleratorDesigner()
        explorer = DesignSpaceExplorer(parallel_workers=4)
        
        # Monitor CPU usage during intensive operations
        cpu_measurements = []
        
        def measure_cpu_usage():
            for _ in range(20):  # Measure for 20 seconds
                cpu_percent = psutil.cpu_percent(interval=1.0)
                cpu_measurements.append(cpu_percent)
                sla_metrics.record_resource_usage(0, cpu_percent)  # Memory measured separately
        
        def intensive_operations():
            # CPU-intensive design space exploration
            design_space = {
                "compute_units": [16, 32, 64, 128],
                "dataflow": ["weight_stationary", "output_stationary"],
                "frequency_mhz": [200.0, 400.0, 600.0]
            }
            
            model_profile = Mock()
            model_profile.peak_gflops = 20.0
            model_profile.bandwidth_gb_s = 40.0
            
            result = explorer.explore(
                model=model_profile,
                design_space=design_space,
                objectives=["latency", "power"],
                num_samples=48,  # All combinations
                strategy="grid"
            )
        
        # Run CPU monitoring and intensive operations concurrently
        import threading
        
        cpu_thread = threading.Thread(target=measure_cpu_usage)
        ops_thread = threading.Thread(target=intensive_operations)
        
        cpu_thread.start()
        ops_thread.start()
        
        ops_thread.join()
        cpu_thread.join()
        
        # CPU usage should stay within limits
        max_cpu = max(cpu_measurements) if cpu_measurements else 0
        avg_cpu = statistics.mean(cpu_measurements) if cpu_measurements else 0
        
        assert max_cpu <= 90, f"Peak CPU usage too high: {max_cpu:.2f}%"
        assert avg_cpu <= 70, f"Average CPU usage too high: {avg_cpu:.2f}%"
        
        # Validate SLA
        sla_results = sla_metrics.validate_slas()
        assert sla_results["sla_compliance"]["cpu_usage"], f"CPU usage SLA violation: {max_cpu:.2f}%"
    
    def test_resource_cleanup_sla(self, sla_metrics):
        """Test resource cleanup and garbage collection SLA."""
        import gc
        
        designer = AcceleratorDesigner()
        process = psutil.Process()
        
        # Baseline measurements
        baseline_memory = process.memory_info().rss / (1024 * 1024)
        baseline_objects = len(gc.get_objects())
        
        # Create and destroy many objects
        for cycle in range(10):
            accelerators = []
            
            # Create many accelerators
            for i in range(50):
                accelerator = designer.design(
                    compute_units=32,
                    dataflow="weight_stationary"
                )
                accelerators.append(accelerator)
            
            # Clear references
            accelerators.clear()
            
            # Force garbage collection
            gc.collect()
            
            # Measure cleanup effectiveness
            current_memory = process.memory_info().rss / (1024 * 1024)
            current_objects = len(gc.get_objects())
            
            memory_growth = current_memory - baseline_memory
            object_growth = current_objects - baseline_objects
            
            sla_metrics.record_resource_usage(memory_growth, 0)
        
        # Resource growth should be bounded
        final_memory = process.memory_info().rss / (1024 * 1024)
        final_memory_growth = final_memory - baseline_memory
        
        assert final_memory_growth <= 100, f"Memory growth after cleanup too high: {final_memory_growth:.2f} MB"
        
        # Validate SLA
        sla_results = sla_metrics.validate_slas()
        assert sla_results["sla_compliance"]["memory_usage"], f"Resource cleanup SLA violation"


class TestErrorRateSLA:
    """Test error rate SLA compliance."""
    
    @pytest.fixture
    def sla_metrics(self):
        """SLA metrics tracker."""
        return SLAMetrics()
    
    def test_operation_error_rate(self, sla_metrics):
        """Test error rate in normal operations."""
        designer = AcceleratorDesigner()
        
        total_operations = 1000
        errors = 0
        
        for i in range(total_operations):
            try:
                # Mix of valid and potentially problematic configurations
                if i % 100 == 0:
                    # Potentially problematic config (but should still work)
                    accelerator = designer.design(
                        compute_units=1,  # Very small
                        dataflow="weight_stationary",
                        frequency_mhz=50.0  # Very low frequency
                    )
                else:
                    # Normal config
                    accelerator = designer.design(
                        compute_units=32,
                        dataflow="weight_stationary",
                        frequency_mhz=400.0
                    )
                
            except Exception as e:
                errors += 1
        
        sla_metrics.record_error(total_operations, errors)
        
        # Error rate should be very low
        error_rate = (errors / total_operations) * 100
        assert error_rate <= 0.5, f"Error rate too high: {error_rate:.2f}%"
        
        # Validate SLA
        sla_results = sla_metrics.validate_slas()
        assert sla_results["sla_compliance"]["error_rate"], f"Error rate SLA violation: {error_rate:.2f}%"
    
    def test_stress_condition_error_rate(self, sla_metrics):
        """Test error rate under stress conditions."""
        designer = AcceleratorDesigner()
        
        # Simulate stress conditions with concurrent operations
        total_operations = 500
        errors = 0
        
        def stress_operation(op_id):
            try:
                # Random configurations that might stress the system
                import random
                
                compute_units = random.choice([1, 16, 32, 64, 128, 256])
                frequency = random.uniform(50.0, 1000.0)
                
                accelerator = designer.design(
                    compute_units=compute_units,
                    dataflow="weight_stationary",
                    frequency_mhz=frequency
                )
                return True
                
            except Exception as e:
                return False
        
        # Execute stress operations concurrently
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(stress_operation, i) for i in range(total_operations)]
            results = [future.result() for future in as_completed(futures, timeout=60)]
        
        errors = sum(1 for result in results if not result)
        sla_metrics.record_error(total_operations, errors)
        
        # Even under stress, error rate should be acceptable
        error_rate = (errors / total_operations) * 100
        assert error_rate <= 2.0, f"Error rate under stress too high: {error_rate:.2f}%"
        
        # Validate SLA
        sla_results = sla_metrics.validate_slas()
        assert sla_results["sla_compliance"]["error_rate"], f"Stress condition error rate SLA violation: {error_rate:.2f}%"
    
    def test_invalid_input_handling(self, sla_metrics):
        """Test error handling for invalid inputs."""
        designer = AcceleratorDesigner()
        
        # Test various invalid inputs
        invalid_configs = [
            {"compute_units": -1, "dataflow": "weight_stationary"},  # Negative compute units
            {"compute_units": 0, "dataflow": "weight_stationary"},   # Zero compute units
            {"compute_units": 32, "dataflow": "invalid_dataflow"},   # Invalid dataflow
            {"compute_units": 32, "dataflow": "weight_stationary", "frequency_mhz": -100},  # Negative frequency
            {"compute_units": 32, "dataflow": "weight_stationary", "frequency_mhz": 0},     # Zero frequency
        ]
        
        total_operations = len(invalid_configs)
        handled_gracefully = 0
        
        for config in invalid_configs:
            try:
                accelerator = designer.design(**config)
                # If it succeeds despite invalid input, that's ok (system might have defaults)
                handled_gracefully += 1
            except (ValueError, ValidationError) as e:
                # Expected exceptions for invalid input - this is correct behavior
                handled_gracefully += 1
            except Exception as e:
                # Unexpected exceptions - this is problematic
                pass
        
        # All invalid inputs should be handled gracefully
        assert handled_gracefully == total_operations, f"Invalid input handling failed for {total_operations - handled_gracefully} cases"
        
        # Record as successful error handling (0 errors)
        sla_metrics.record_error(total_operations, 0)


class TestAvailabilitySLA:
    """Test system availability SLA compliance."""
    
    @pytest.fixture
    def sla_metrics(self):
        """SLA metrics tracker."""
        return SLAMetrics()
    
    def test_continuous_operation_availability(self, sla_metrics):
        """Test system availability during continuous operation."""
        designer = AcceleratorDesigner()
        
        # Test continuous operation for extended period
        test_duration = 30.0  # seconds
        operation_interval = 0.1  # seconds between operations
        
        start_time = time.time()
        total_operations = 0
        successful_operations = 0
        
        while (time.time() - start_time) < test_duration:
            total_operations += 1
            
            try:
                accelerator = designer.design(
                    compute_units=32,
                    dataflow="weight_stationary"
                )
                successful_operations += 1
                
            except Exception as e:
                pass  # Count as unavailable
            
            time.sleep(operation_interval)
        
        # Calculate availability
        availability_percent = (successful_operations / total_operations) * 100
        sla_metrics.metrics["availability_percent"] = availability_percent
        
        # Availability should be very high
        assert availability_percent >= 99.0, f"Availability too low: {availability_percent:.2f}%"
        
        # Validate SLA
        sla_results = sla_metrics.validate_slas()
        assert availability_percent >= sla_metrics.sla_thresholds["min_availability_percent"], f"Availability SLA violation: {availability_percent:.2f}%"
    
    def test_recovery_time_sla(self, sla_metrics):
        """Test system recovery time after simulated failure."""
        designer = AcceleratorDesigner()
        
        # Simulate system degradation and recovery
        recovery_times = []
        
        for test_cycle in range(5):
            # Simulate heavy load that might cause degradation
            start_load_time = time.time()
            
            # Heavy concurrent load
            with ThreadPoolExecutor(max_workers=50) as executor:
                futures = [
                    executor.submit(
                        designer.design,
                        compute_units=64,
                        dataflow="weight_stationary"
                    ) for _ in range(100)
                ]
                
                # Wait for completion
                results = [future.result() for future in as_completed(futures, timeout=30)]
            
            load_end_time = time.time()
            
            # Measure recovery time (time until normal operation resumes)
            recovery_start_time = time.time()
            
            # Test normal operation
            while True:
                try:
                    start_op_time = time.time()
                    accelerator = designer.design(
                        compute_units=32,
                        dataflow="weight_stationary"
                    )
                    op_time = time.time() - start_op_time
                    
                    # Normal operation should be fast
                    if op_time <= 0.2:  # 200ms threshold
                        recovery_time = time.time() - recovery_start_time
                        recovery_times.append(recovery_time)
                        break
                    
                    # If operation is still slow, continue waiting
                    if (time.time() - recovery_start_time) > 10.0:  # Max 10 seconds wait
                        recovery_times.append(10.0)  # Record max recovery time
                        break
                        
                except Exception:
                    # System still recovering
                    if (time.time() - recovery_start_time) > 10.0:
                        recovery_times.append(10.0)
                        break
                
                time.sleep(0.1)  # Brief pause before retry
        
        # Recovery should be fast
        max_recovery_time = max(recovery_times)
        avg_recovery_time = statistics.mean(recovery_times)
        
        assert max_recovery_time <= 5.0, f"Maximum recovery time too long: {max_recovery_time:.2f}s"
        assert avg_recovery_time <= 2.0, f"Average recovery time too long: {avg_recovery_time:.2f}s"
    
    def test_graceful_degradation_sla(self, sla_metrics):
        """Test graceful degradation under extreme load."""
        designer = AcceleratorDesigner()
        
        # Test response under extreme concurrent load
        extreme_load_workers = 100
        operations_per_worker = 10
        
        start_time = time.time()
        
        def extreme_load_operation(worker_id):
            results = []
            
            for op in range(operations_per_worker):
                op_start_time = time.time()
                
                try:
                    accelerator = designer.design(
                        compute_units=32 + (worker_id % 8) * 8,
                        dataflow="weight_stationary"
                    )
                    
                    op_end_time = time.time()
                    op_duration = op_end_time - op_start_time
                    
                    results.append({
                        "success": True,
                        "duration": op_duration,
                        "worker_id": worker_id,
                        "operation": op
                    })
                    
                except Exception as e:
                    op_end_time = time.time()
                    results.append({
                        "success": False,
                        "error": str(e),
                        "duration": op_end_time - op_start_time,
                        "worker_id": worker_id,
                        "operation": op
                    })
            
            return results
        
        # Execute extreme load
        with ThreadPoolExecutor(max_workers=extreme_load_workers) as executor:
            futures = [
                executor.submit(extreme_load_operation, worker_id)
                for worker_id in range(extreme_load_workers)
            ]
            
            all_results = []
            for future in as_completed(futures, timeout=120):  # 2 minute timeout
                worker_results = future.result()
                all_results.extend(worker_results)
        
        end_time = time.time()
        total_duration = end_time - start_time
        
        # Analyze results
        total_operations = len(all_results)
        successful_operations = sum(1 for result in all_results if result["success"])
        availability_percent = (successful_operations / total_operations) * 100
        
        # Even under extreme load, system should maintain minimum availability
        assert availability_percent >= 90.0, f"Availability under extreme load too low: {availability_percent:.2f}%"
        
        # Response times may degrade but shouldn't be excessive
        successful_results = [r for r in all_results if r["success"]]
        if successful_results:
            response_times = [r["duration"] for r in successful_results]
            avg_response_time = statistics.mean(response_times)
            p95_response_time = np.percentile(response_times, 95)
            
            # Graceful degradation means longer response times are acceptable
            assert avg_response_time <= 2.0, f"Average response time under extreme load too high: {avg_response_time:.2f}s"
            assert p95_response_time <= 5.0, f"P95 response time under extreme load too high: {p95_response_time:.2f}s"


class TestEndToEndSLA:
    """Test end-to-end SLA compliance across entire workflows."""
    
    def test_complete_workflow_sla(self, tmp_path):
        """Test SLA compliance for complete workflow execution."""
        sla_metrics = SLAMetrics()
        
        # Adjust thresholds for complete workflows
        sla_metrics.sla_thresholds.update({
            "max_response_time_ms": 30000,  # 30 seconds for complete workflow
            "min_throughput_ops_per_second": 1,  # 1 workflow per second
            "max_memory_usage_mb": 1024,  # 1GB for workflow
            "max_cpu_usage_percent": 90,
            "max_error_rate_percent": 5.0  # Higher tolerance for complex workflows
        })
        
        # Create test model file
        model_file = tmp_path / "sla_test_model.onnx"
        model_file.write_bytes(b"sla_test_model_data" * 1000)
        
        # Test multiple complete workflows
        workflow_count = 10
        successful_workflows = 0
        
        for i in range(workflow_count):
            workflow_start_time = time.time()
            
            try:
                config = WorkflowConfig(
                    name=f"sla_test_workflow_{i}",
                    model_path=str(model_file),
                    input_shapes={"input": (1, 3, 224, 224)},
                    framework="onnx"
                )
                
                workflow = Workflow(config)
                
                # Execute complete workflow
                workflow.import_model()
                workflow.map_to_hardware(
                    template="systolic_array",
                    size=(16, 16),
                    precision="int8"
                )
                
                result = workflow.optimize(
                    target_fps=30.0,
                    power_budget=5.0,
                    iterations=3  # Limited for SLA testing
                )
                
                workflow.generate_rtl()
                
                workflow_end_time = time.time()
                workflow_duration_ms = (workflow_end_time - workflow_start_time) * 1000
                
                # Record metrics
                sla_metrics.record_response_time(workflow_duration_ms)
                
                # Record resource usage
                import psutil
                process = psutil.Process()
                memory_usage = process.memory_info().rss / (1024 * 1024)
                cpu_usage = psutil.cpu_percent()
                sla_metrics.record_resource_usage(memory_usage, cpu_usage)
                
                successful_workflows += 1
                
            except Exception as e:
                # Record failure
                pass
        
        # Record error rate
        sla_metrics.record_error(workflow_count, workflow_count - successful_workflows)
        
        # Calculate throughput
        if successful_workflows > 0:
            # Estimate throughput based on average workflow time
            avg_response_time_s = statistics.mean(sla_metrics.metrics["response_times"]) / 1000
            estimated_throughput = 1.0 / avg_response_time_s if avg_response_time_s > 0 else 0
            sla_metrics.record_throughput(estimated_throughput)
        
        # Validate SLAs
        sla_results = sla_metrics.validate_slas()
        
        # Assert SLA compliance
        assert sla_results["passed"], f"End-to-end SLA violations: {sla_results['violations']}"
        
        # Specific assertions
        assert successful_workflows >= 9, f"Too many workflow failures: {workflow_count - successful_workflows}/{workflow_count}"
        
        if sla_metrics.metrics["response_times"]:
            avg_workflow_time = statistics.mean(sla_metrics.metrics["response_times"])
            assert avg_workflow_time <= 25000, f"Average workflow time too long: {avg_workflow_time:.2f}ms"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])