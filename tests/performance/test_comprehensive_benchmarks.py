"""
Comprehensive Performance Benchmarks for AI Hardware Co-Design Playground.

This module implements detailed performance benchmarks to measure and validate
system performance characteristics across different scenarios and workloads.
"""

import pytest
import time
import psutil
import statistics
import numpy as np
from typing import Dict, List, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import Mock, patch
import tempfile
from pathlib import Path

from codesign_playground.core.accelerator import AcceleratorDesigner, Accelerator
from codesign_playground.core.optimizer import ModelOptimizer
from codesign_playground.core.explorer import DesignSpaceExplorer
from codesign_playground.core.workflow import Workflow, WorkflowConfig
from codesign_playground.core.cache import AdaptiveCache


class BenchmarkResults:
    """Class to collect and analyze benchmark results."""
    
    def __init__(self, test_name: str):
        self.test_name = test_name
        self.measurements = []
        self.metadata = {}
        self.start_time = time.time()
        self.end_time = None
    
    def add_measurement(self, name: str, value: float, unit: str = "", metadata: Dict = None):
        """Add a performance measurement."""
        self.measurements.append({
            "name": name,
            "value": value,
            "unit": unit,
            "metadata": metadata or {},
            "timestamp": time.time()
        })
    
    def finalize(self):
        """Finalize benchmark and calculate summary statistics."""
        self.end_time = time.time()
        self.metadata["total_duration_seconds"] = self.end_time - self.start_time
        
        # Calculate statistics for each measurement type
        measurement_groups = {}
        for measurement in self.measurements:
            name = measurement["name"]
            if name not in measurement_groups:
                measurement_groups[name] = []
            measurement_groups[name].append(measurement["value"])
        
        self.metadata["statistics"] = {}
        for name, values in measurement_groups.items():
            if values:
                self.metadata["statistics"][name] = {
                    "count": len(values),
                    "mean": statistics.mean(values),
                    "median": statistics.median(values),
                    "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                    "min": min(values),
                    "max": max(values),
                    "p95": np.percentile(values, 95),
                    "p99": np.percentile(values, 99)
                }
    
    def get_summary(self) -> Dict[str, Any]:
        """Get benchmark summary."""
        return {
            "test_name": self.test_name,
            "total_duration_seconds": self.metadata.get("total_duration_seconds", 0),
            "measurement_count": len(self.measurements),
            "statistics": self.metadata.get("statistics", {}),
            "metadata": {k: v for k, v in self.metadata.items() if k != "statistics"}
        }


class TestAcceleratorDesignBenchmarks:
    """Benchmarks for accelerator design performance."""
    
    @pytest.mark.benchmark
    def test_accelerator_design_latency_benchmark(self, benchmark):
        """Benchmark accelerator design latency."""
        designer = AcceleratorDesigner()
        results = BenchmarkResults("accelerator_design_latency")
        
        # Test different configuration complexities
        configs = [
            {"compute_units": 16, "dataflow": "weight_stationary", "frequency_mhz": 200.0},
            {"compute_units": 32, "dataflow": "weight_stationary", "frequency_mhz": 400.0},
            {"compute_units": 64, "dataflow": "output_stationary", "frequency_mhz": 600.0},
            {"compute_units": 128, "dataflow": "row_stationary", "frequency_mhz": 800.0}
        ]
        
        def run_design_benchmark():
            latencies = []
            
            for i, config in enumerate(configs * 25):  # 100 total designs
                start_time = time.time()
                
                accelerator = designer.design(**config)
                
                end_time = time.time()
                latency = (end_time - start_time) * 1000  # ms
                latencies.append(latency)
                
                results.add_measurement("design_latency", latency, "ms", {
                    "compute_units": config["compute_units"],
                    "dataflow": config["dataflow"],
                    "iteration": i
                })
            
            return latencies
        
        # Benchmark using pytest-benchmark
        latencies = benchmark(run_design_benchmark)
        results.finalize()
        
        # Performance assertions
        avg_latency = statistics.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        
        assert avg_latency <= 50.0, f"Average design latency too high: {avg_latency:.2f}ms"
        assert p95_latency <= 100.0, f"P95 design latency too high: {p95_latency:.2f}ms"
        
        print(f"\nAccelerator Design Latency Benchmark:")
        print(f"  Average: {avg_latency:.2f}ms")
        print(f"  P95: {p95_latency:.2f}ms")
        print(f"  P99: {np.percentile(latencies, 99):.2f}ms")
    
    @pytest.mark.benchmark
    def test_accelerator_design_throughput_benchmark(self, benchmark):
        """Benchmark accelerator design throughput."""
        designer = AcceleratorDesigner()
        results = BenchmarkResults("accelerator_design_throughput")
        
        def measure_throughput():
            test_duration = 10.0  # seconds
            start_time = time.time()
            designs_completed = 0
            
            while (time.time() - start_time) < test_duration:
                accelerator = designer.design(
                    compute_units=32,
                    dataflow="weight_stationary",
                    frequency_mhz=400.0
                )
                designs_completed += 1
            
            elapsed_time = time.time() - start_time
            throughput = designs_completed / elapsed_time
            
            results.add_measurement("throughput", throughput, "designs/sec")
            return throughput
        
        throughput = benchmark(measure_throughput)
        results.finalize()
        
        # Throughput should be reasonable
        assert throughput >= 50.0, f"Design throughput too low: {throughput:.2f} designs/sec"
        
        print(f"\nAccelerator Design Throughput: {throughput:.2f} designs/sec")
    
    @pytest.mark.benchmark
    def test_parallel_design_scaling_benchmark(self, benchmark):
        """Benchmark parallel design scaling."""
        designer = AcceleratorDesigner()
        results = BenchmarkResults("parallel_design_scaling")
        
        # Test different worker counts
        worker_counts = [1, 2, 4, 8]
        batch_size = 32
        
        def measure_parallel_scaling():
            scaling_results = {}
            
            for workers in worker_counts:
                configs = [
                    {"compute_units": 16 + i * 4, "dataflow": "weight_stationary"}
                    for i in range(batch_size)
                ]
                
                start_time = time.time()
                accelerators = designer.design_parallel(configs, max_workers=workers)
                end_time = time.time()
                
                duration = end_time - start_time
                throughput = len(accelerators) / duration
                
                scaling_results[workers] = {
                    "duration": duration,
                    "throughput": throughput
                }
                
                results.add_measurement("parallel_throughput", throughput, "designs/sec", {
                    "workers": workers,
                    "batch_size": batch_size
                })
            
            return scaling_results
        
        scaling_results = benchmark(measure_parallel_scaling)
        results.finalize()
        
        # Verify scaling efficiency
        single_worker_throughput = scaling_results[1]["throughput"]
        multi_worker_throughput = scaling_results[max(worker_counts)]["throughput"]
        
        scaling_factor = multi_worker_throughput / single_worker_throughput
        
        # Should achieve some parallelization benefit
        assert scaling_factor >= 1.5, f"Parallel scaling too low: {scaling_factor:.2f}x"
        
        print(f"\nParallel Design Scaling:")
        for workers, result in scaling_results.items():
            print(f"  {workers} workers: {result['throughput']:.2f} designs/sec")
        print(f"  Scaling factor: {scaling_factor:.2f}x")
    
    @pytest.mark.benchmark
    def test_memory_efficiency_benchmark(self, benchmark):
        """Benchmark memory efficiency during design operations."""
        designer = AcceleratorDesigner()
        results = BenchmarkResults("memory_efficiency")
        process = psutil.Process()
        
        def measure_memory_efficiency():
            baseline_memory = process.memory_info().rss
            memory_measurements = []
            
            # Create many accelerators and measure memory growth
            accelerators = []
            for i in range(200):
                accelerator = designer.design(
                    compute_units=32 + i % 64,
                    dataflow="weight_stationary"
                )
                accelerators.append(accelerator)
                
                if i % 20 == 0:
                    current_memory = process.memory_info().rss
                    memory_growth = (current_memory - baseline_memory) / (1024 * 1024)  # MB
                    memory_measurements.append(memory_growth)
                    
                    results.add_measurement("memory_growth", memory_growth, "MB", {
                        "accelerators_created": i + 1
                    })
            
            # Final memory measurement
            final_memory = process.memory_info().rss
            total_growth = (final_memory - baseline_memory) / (1024 * 1024)
            
            # Clear references and measure cleanup
            accelerators.clear()
            import gc
            gc.collect()
            
            cleanup_memory = process.memory_info().rss
            cleanup_growth = (cleanup_memory - baseline_memory) / (1024 * 1024)
            
            results.add_measurement("final_memory_growth", total_growth, "MB")
            results.add_measurement("post_cleanup_growth", cleanup_growth, "MB")
            
            return {
                "total_growth": total_growth,
                "cleanup_growth": cleanup_growth,
                "memory_per_accelerator": total_growth / 200
            }
        
        memory_results = benchmark(measure_memory_efficiency)
        results.finalize()
        
        # Memory usage should be reasonable
        memory_per_accelerator = memory_results["memory_per_accelerator"]
        cleanup_efficiency = 1.0 - (memory_results["cleanup_growth"] / memory_results["total_growth"])
        
        assert memory_per_accelerator <= 2.0, f"Memory per accelerator too high: {memory_per_accelerator:.2f}MB"
        assert cleanup_efficiency >= 0.7, f"Memory cleanup efficiency too low: {cleanup_efficiency:.2f}"
        
        print(f"\nMemory Efficiency Benchmark:")
        print(f"  Memory per accelerator: {memory_per_accelerator:.2f}MB")
        print(f"  Cleanup efficiency: {cleanup_efficiency:.2%}")


class TestOptimizerBenchmarks:
    """Benchmarks for optimization performance."""
    
    @pytest.mark.benchmark
    def test_optimization_convergence_benchmark(self, benchmark):
        """Benchmark optimization convergence speed."""
        mock_model = Mock()
        mock_model.parameters = 1000000
        
        accelerator = AcceleratorDesigner().design(compute_units=64)
        optimizer = ModelOptimizer(mock_model, accelerator)
        
        results = BenchmarkResults("optimization_convergence")
        
        def measure_convergence():
            convergence_data = []
            
            # Test different optimization scenarios
            scenarios = [
                {"target_fps": 30.0, "power_budget": 5.0, "iterations": 10},
                {"target_fps": 60.0, "power_budget": 8.0, "iterations": 15},
                {"target_fps": 120.0, "power_budget": 12.0, "iterations": 20}
            ]
            
            for scenario in scenarios:
                start_time = time.time()
                
                result = optimizer.co_optimize(
                    target_fps=scenario["target_fps"],
                    power_budget=scenario["power_budget"],
                    iterations=scenario["iterations"]
                )
                
                end_time = time.time()
                optimization_time = end_time - start_time
                
                convergence_data.append({
                    "scenario": scenario,
                    "optimization_time": optimization_time,
                    "iterations": result.iterations,
                    "converged": result.converged if hasattr(result, 'converged') else True
                })
                
                results.add_measurement("optimization_time", optimization_time, "seconds", {
                    "target_fps": scenario["target_fps"],
                    "power_budget": scenario["power_budget"],
                    "max_iterations": scenario["iterations"]
                })
            
            return convergence_data
        
        convergence_data = benchmark(measure_convergence)
        results.finalize()
        
        # Analyze convergence performance
        optimization_times = [data["optimization_time"] for data in convergence_data]
        avg_optimization_time = statistics.mean(optimization_times)
        
        # Optimization should converge reasonably quickly
        assert avg_optimization_time <= 2.0, f"Average optimization time too long: {avg_optimization_time:.2f}s"
        assert all(data["optimization_time"] <= 5.0 for data in convergence_data), "Some optimizations took too long"
        
        print(f"\nOptimization Convergence Benchmark:")
        print(f"  Average optimization time: {avg_optimization_time:.2f}s")
        for data in convergence_data:
            print(f"  FPS {data['scenario']['target_fps']:3.0f}: {data['optimization_time']:.2f}s")
    
    @pytest.mark.benchmark
    def test_multi_objective_optimization_benchmark(self, benchmark):
        """Benchmark multi-objective optimization performance."""
        from codesign_playground.core.accelerator import ModelProfile
        
        mock_model = ModelProfile(
            peak_gflops=25.0,
            bandwidth_gb_s=50.0,
            operations={"conv2d": 5000, "dense": 1500},
            parameters=2500000,
            memory_mb=40.0,
            compute_intensity=0.5,
            layer_types=["conv2d", "dense"],
            model_size_mb=40.0
        )
        
        accelerator = AcceleratorDesigner().design(compute_units=64)
        optimizer = ModelOptimizer(mock_model, accelerator)
        
        results = BenchmarkResults("multi_objective_optimization")
        
        def measure_multi_objective_performance():
            performance_data = []
            
            # Test different objective combinations
            objective_sets = [
                ["latency", "power"],
                ["latency", "power", "area"],
                ["latency", "power", "area", "accuracy"]
            ]
            
            constraints = {
                "power_budget": 8.0,
                "area_budget": 150.0,
                "latency_ms": 20.0,
                "accuracy_threshold": 0.95
            }
            
            for objectives in objective_sets:
                start_time = time.time()
                
                result = optimizer.multi_objective_optimize(
                    model_profile=mock_model,
                    constraints=constraints,
                    objectives=objectives,
                    num_iterations=5  # Limited for benchmarking
                )
                
                end_time = time.time()
                optimization_time = end_time - start_time
                
                performance_data.append({
                    "objectives": objectives,
                    "optimization_time": optimization_time,
                    "pareto_solutions": len(result.get("pareto_solutions", [])),
                    "objective_count": len(objectives)
                })
                
                results.add_measurement("multi_objective_time", optimization_time, "seconds", {
                    "objective_count": len(objectives),
                    "pareto_solutions": len(result.get("pareto_solutions", []))
                })
            
            return performance_data
        
        performance_data = benchmark(measure_multi_objective_performance)
        results.finalize()
        
        # Analyze multi-objective performance
        for data in performance_data:
            obj_count = data["objective_count"]
            opt_time = data["optimization_time"]
            
            # Time should scale reasonably with objective count
            expected_max_time = obj_count * 2.0  # 2 seconds per objective
            assert opt_time <= expected_max_time, f"Multi-objective optimization too slow for {obj_count} objectives: {opt_time:.2f}s"
        
        print(f"\nMulti-Objective Optimization Benchmark:")
        for data in performance_data:
            print(f"  {data['objective_count']} objectives: {data['optimization_time']:.2f}s, {data['pareto_solutions']} solutions")


class TestExplorerBenchmarks:
    """Benchmarks for design space exploration performance."""
    
    @pytest.mark.benchmark
    def test_design_space_exploration_scaling(self, benchmark):
        """Benchmark design space exploration scaling."""
        explorer = DesignSpaceExplorer(parallel_workers=4)
        
        model_profile = Mock()
        model_profile.peak_gflops = 20.0
        model_profile.bandwidth_gb_s = 40.0
        
        results = BenchmarkResults("design_space_exploration_scaling")
        
        def measure_exploration_scaling():
            scaling_data = []
            
            # Test different design space sizes
            design_spaces = [
                {  # Small space
                    "compute_units": [16, 32, 64],
                    "dataflow": ["weight_stationary", "output_stationary"],
                    "frequency_mhz": [200.0, 400.0]
                },
                {  # Medium space
                    "compute_units": [16, 32, 64, 128],
                    "dataflow": ["weight_stationary", "output_stationary", "row_stationary"],
                    "frequency_mhz": [200.0, 400.0, 600.0],
                    "precision": ["int8", "fp16"]
                },
                {  # Large space
                    "compute_units": [16, 32, 64, 128, 256],
                    "dataflow": ["weight_stationary", "output_stationary", "row_stationary"],
                    "frequency_mhz": [200.0, 400.0, 600.0, 800.0],
                    "precision": ["int8", "fp16", "fp32"],
                    "memory_hierarchy": [["sram_64kb", "dram"], ["sram_128kb", "dram"]]
                }
            ]
            
            for i, design_space in enumerate(design_spaces):
                # Calculate design space size
                space_size = 1
                for dimension, values in design_space.items():
                    space_size *= len(values)
                
                # Sample a portion of the space
                num_samples = min(space_size, 20)  # Cap for benchmarking
                
                start_time = time.time()
                
                result = explorer.explore(
                    model=model_profile,
                    design_space=design_space,
                    objectives=["latency", "power"],
                    num_samples=num_samples,
                    strategy="random"
                )
                
                end_time = time.time()
                exploration_time = end_time - start_time
                
                scaling_data.append({
                    "space_size": space_size,
                    "samples": num_samples,
                    "exploration_time": exploration_time,
                    "time_per_sample": exploration_time / num_samples if num_samples > 0 else 0
                })
                
                results.add_measurement("exploration_time", exploration_time, "seconds", {
                    "space_size": space_size,
                    "samples": num_samples
                })
                
                results.add_measurement("time_per_sample", exploration_time / num_samples, "seconds", {
                    "space_size": space_size
                })
            
            return scaling_data
        
        scaling_data = benchmark(measure_exploration_scaling)
        results.finalize()
        
        # Analyze scaling performance
        times_per_sample = [data["time_per_sample"] for data in scaling_data]
        avg_time_per_sample = statistics.mean(times_per_sample)
        
        # Time per sample should be reasonable and not increase dramatically with space size
        assert avg_time_per_sample <= 1.0, f"Average time per sample too high: {avg_time_per_sample:.2f}s"
        
        # Verify scaling doesn't degrade significantly
        if len(times_per_sample) > 1:
            scaling_factor = max(times_per_sample) / min(times_per_sample)
            assert scaling_factor <= 3.0, f"Exploration scaling too poor: {scaling_factor:.2f}x degradation"
        
        print(f"\nDesign Space Exploration Scaling:")
        for data in scaling_data:
            print(f"  Space size {data['space_size']:4d}: {data['time_per_sample']:.3f}s per sample")
    
    @pytest.mark.benchmark
    def test_pareto_frontier_computation_benchmark(self, benchmark):
        """Benchmark Pareto frontier computation performance."""
        explorer = DesignSpaceExplorer()
        results = BenchmarkResults("pareto_frontier_computation")
        
        def measure_pareto_computation():
            pareto_data = []
            
            # Test different numbers of design points
            point_counts = [50, 100, 200, 500]
            
            for point_count in point_counts:
                # Generate random design points
                from codesign_playground.core.explorer import DesignPoint
                
                design_points = []
                for i in range(point_count):
                    config = {
                        "compute_units": 16 + (i % 64) * 4,
                        "dataflow": "weight_stationary"
                    }
                    
                    # Random metrics with some correlation
                    latency = 10 + np.random.exponential(10)
                    power = 3 + latency * 0.1 + np.random.normal(0, 1)
                    area = 50 + latency * 2 + np.random.normal(0, 10)
                    
                    metrics = {
                        "latency": latency,
                        "power": max(1, power),
                        "area": max(20, area)
                    }
                    
                    design_points.append(DesignPoint(config=config, metrics=metrics))
                
                # Measure Pareto frontier computation time
                start_time = time.time()
                
                pareto_frontier = explorer._compute_pareto_frontier(
                    design_points, 
                    ["latency", "power", "area"]
                )
                
                end_time = time.time()
                computation_time = end_time - start_time
                
                pareto_data.append({
                    "point_count": point_count,
                    "computation_time": computation_time,
                    "pareto_size": len(pareto_frontier),
                    "time_per_point": computation_time / point_count
                })
                
                results.add_measurement("pareto_computation_time", computation_time, "seconds", {
                    "point_count": point_count,
                    "pareto_size": len(pareto_frontier)
                })
            
            return pareto_data
        
        pareto_data = benchmark(measure_pareto_computation)
        results.finalize()
        
        # Analyze Pareto computation performance
        computation_times = [data["computation_time"] for data in pareto_data]
        max_computation_time = max(computation_times)
        
        # Pareto computation should be efficient even for large point sets
        assert max_computation_time <= 2.0, f"Pareto computation too slow: {max_computation_time:.2f}s"
        
        # Time should scale reasonably
        if len(pareto_data) > 1:
            largest_dataset = pareto_data[-1]
            assert largest_dataset["time_per_point"] <= 0.01, f"Time per point too high: {largest_dataset['time_per_point']:.4f}s"
        
        print(f"\nPareto Frontier Computation Benchmark:")
        for data in pareto_data:
            print(f"  {data['point_count']:3d} points: {data['computation_time']:.3f}s, {data['pareto_size']:2d} Pareto points")


class TestWorkflowBenchmarks:
    """Benchmarks for complete workflow performance."""
    
    @pytest.mark.benchmark
    def test_end_to_end_workflow_benchmark(self, benchmark, tmp_path):
        """Benchmark complete end-to-end workflow performance."""
        results = BenchmarkResults("end_to_end_workflow")
        
        # Create test model file
        model_file = tmp_path / "benchmark_model.onnx"
        model_file.write_bytes(b"benchmark_model_data" * 1000)
        
        def measure_workflow_performance():
            workflow_data = []
            
            # Test different workflow configurations
            configurations = [
                {
                    "name": "small_workflow",
                    "hardware": {"template": "systolic_array", "size": (8, 8), "precision": "int8"},
                    "optimization": {"target_fps": 15.0, "power_budget": 3.0, "iterations": 3}
                },
                {
                    "name": "medium_workflow",
                    "hardware": {"template": "systolic_array", "size": (16, 16), "precision": "int8"},
                    "optimization": {"target_fps": 30.0, "power_budget": 5.0, "iterations": 5}
                },
                {
                    "name": "large_workflow",
                    "hardware": {"template": "systolic_array", "size": (32, 32), "precision": "fp16"},
                    "optimization": {"target_fps": 60.0, "power_budget": 10.0, "iterations": 8}
                }
            ]
            
            for config in configurations:
                workflow_config = WorkflowConfig(
                    name=config["name"],
                    model_path=str(model_file),
                    input_shapes={"input": (1, 3, 224, 224)},
                    framework="onnx"
                )
                
                workflow = Workflow(workflow_config)
                
                # Measure each stage
                stage_times = {}
                
                # Model import
                start_time = time.time()
                workflow.import_model()
                stage_times["import"] = time.time() - start_time
                
                # Hardware mapping
                start_time = time.time()
                workflow.map_to_hardware(**config["hardware"])
                stage_times["hardware_mapping"] = time.time() - start_time
                
                # Optimization
                start_time = time.time()
                result = workflow.optimize(**config["optimization"])
                stage_times["optimization"] = time.time() - start_time
                
                # RTL generation
                start_time = time.time()
                rtl_file = workflow.generate_rtl()
                stage_times["rtl_generation"] = time.time() - start_time
                
                total_time = sum(stage_times.values())
                
                workflow_data.append({
                    "config_name": config["name"],
                    "total_time": total_time,
                    "stage_times": stage_times
                })
                
                # Record measurements
                results.add_measurement("total_workflow_time", total_time, "seconds", {
                    "config_name": config["name"]
                })
                
                for stage, time_taken in stage_times.items():
                    results.add_measurement(f"{stage}_time", time_taken, "seconds", {
                        "config_name": config["name"]
                    })
            
            return workflow_data
        
        workflow_data = benchmark(measure_workflow_performance)
        results.finalize()
        
        # Analyze workflow performance
        total_times = [data["total_time"] for data in workflow_data]
        max_total_time = max(total_times)
        avg_total_time = statistics.mean(total_times)
        
        # Workflows should complete within reasonable time
        assert max_total_time <= 30.0, f"Maximum workflow time too long: {max_total_time:.2f}s"
        assert avg_total_time <= 20.0, f"Average workflow time too long: {avg_total_time:.2f}s"
        
        print(f"\nEnd-to-End Workflow Benchmark:")
        print(f"  Average total time: {avg_total_time:.2f}s")
        for data in workflow_data:
            print(f"  {data['config_name']}: {data['total_time']:.2f}s")
            for stage, time_taken in data['stage_times'].items():
                print(f"    {stage}: {time_taken:.2f}s")
    
    @pytest.mark.benchmark
    def test_concurrent_workflow_benchmark(self, benchmark, tmp_path):
        """Benchmark concurrent workflow execution."""
        results = BenchmarkResults("concurrent_workflow")
        
        # Create multiple model files
        model_files = []
        for i in range(5):
            model_file = tmp_path / f"concurrent_model_{i}.onnx"
            model_file.write_bytes(f"concurrent_model_data_{i}".encode() * 500)
            model_files.append(model_file)
        
        def measure_concurrent_performance():
            concurrent_data = []
            
            # Test different concurrency levels
            concurrency_levels = [1, 2, 4]
            
            for concurrency in concurrency_levels:
                configs = []
                for i in range(concurrency):
                    config = WorkflowConfig(
                        name=f"concurrent_workflow_{i}",
                        model_path=str(model_files[i % len(model_files)]),
                        input_shapes={"input": (1, 3, 224, 224)},
                        framework="onnx"
                    )
                    configs.append(config)
                
                # Measure concurrent execution
                start_time = time.time()
                
                def execute_workflow(config):
                    workflow = Workflow(config)
                    workflow.import_model()
                    workflow.map_to_hardware(template="systolic_array", size=(16, 16))
                    result = workflow.optimize(target_fps=30.0, power_budget=5.0, iterations=3)
                    return workflow.state.stage.value == "OPTIMIZED"
                
                with ThreadPoolExecutor(max_workers=concurrency) as executor:
                    futures = [executor.submit(execute_workflow, config) for config in configs]
                    results_list = [future.result() for future in as_completed(futures, timeout=60)]
                
                end_time = time.time()
                concurrent_time = end_time - start_time
                
                successful_workflows = sum(results_list)
                
                concurrent_data.append({
                    "concurrency": concurrency,
                    "concurrent_time": concurrent_time,
                    "successful_workflows": successful_workflows,
                    "throughput": successful_workflows / concurrent_time
                })
                
                results.add_measurement("concurrent_time", concurrent_time, "seconds", {
                    "concurrency": concurrency,
                    "successful_workflows": successful_workflows
                })
                
                results.add_measurement("concurrent_throughput", successful_workflows / concurrent_time, "workflows/sec", {
                    "concurrency": concurrency
                })
            
            return concurrent_data
        
        concurrent_data = benchmark(measure_concurrent_performance)
        results.finalize()
        
        # Analyze concurrent performance
        throughputs = [data["throughput"] for data in concurrent_data]
        max_throughput = max(throughputs)
        
        # Concurrent execution should improve throughput
        single_threaded_throughput = concurrent_data[0]["throughput"]
        best_throughput = max_throughput
        
        scaling_factor = best_throughput / single_threaded_throughput
        assert scaling_factor >= 1.3, f"Concurrent scaling too low: {scaling_factor:.2f}x"
        
        print(f"\nConcurrent Workflow Benchmark:")
        for data in concurrent_data:
            print(f"  {data['concurrency']} concurrent: {data['throughput']:.2f} workflows/sec")
        print(f"  Scaling factor: {scaling_factor:.2f}x")


class TestCacheBenchmarks:
    """Benchmarks for caching system performance."""
    
    @pytest.mark.benchmark
    def test_cache_performance_benchmark(self, benchmark):
        """Benchmark cache performance characteristics."""
        cache = AdaptiveCache(max_size=1000, max_memory_mb=10.0)
        results = BenchmarkResults("cache_performance")
        
        def measure_cache_performance():
            cache_data = {}
            
            # Test data of varying sizes
            test_datasets = [
                {"name": "small", "size": 100, "data": list(range(100))},
                {"name": "medium", "size": 1000, "data": list(range(1000))},
                {"name": "large", "size": 5000, "data": list(range(5000))}
            ]
            
            for dataset in test_datasets:
                dataset_results = {}
                
                # Measure PUT operations
                put_times = []
                for i in range(100):
                    key = f"{dataset['name']}_key_{i}"
                    
                    start_time = time.time()
                    cache.put(key, dataset["data"])
                    end_time = time.time()
                    
                    put_time = (end_time - start_time) * 1000  # ms
                    put_times.append(put_time)
                
                dataset_results["put_avg_ms"] = statistics.mean(put_times)
                dataset_results["put_p95_ms"] = np.percentile(put_times, 95)
                
                # Measure GET operations
                get_times = []
                for i in range(100):
                    key = f"{dataset['name']}_key_{i}"
                    
                    start_time = time.time()
                    value = cache.get(key)
                    end_time = time.time()
                    
                    get_time = (end_time - start_time) * 1000  # ms
                    get_times.append(get_time)
                
                dataset_results["get_avg_ms"] = statistics.mean(get_times)
                dataset_results["get_p95_ms"] = np.percentile(get_times, 95)
                
                # Cache hit rate
                stats = cache.get_stats()
                dataset_results["hit_rate"] = stats.get("hit_rate", 0.0)
                
                cache_data[dataset["name"]] = dataset_results
                
                # Record measurements
                results.add_measurement("put_time", dataset_results["put_avg_ms"], "ms", {
                    "dataset": dataset["name"],
                    "data_size": dataset["size"]
                })
                
                results.add_measurement("get_time", dataset_results["get_avg_ms"], "ms", {
                    "dataset": dataset["name"],
                    "data_size": dataset["size"]
                })
            
            return cache_data
        
        cache_data = benchmark(measure_cache_performance)
        results.finalize()
        
        # Analyze cache performance
        for dataset_name, dataset_results in cache_data.items():
            put_avg = dataset_results["put_avg_ms"]
            get_avg = dataset_results["get_avg_ms"]
            
            # Cache operations should be fast
            assert put_avg <= 5.0, f"PUT operations too slow for {dataset_name}: {put_avg:.2f}ms"
            assert get_avg <= 1.0, f"GET operations too slow for {dataset_name}: {get_avg:.2f}ms"
        
        print(f"\nCache Performance Benchmark:")
        for dataset_name, dataset_results in cache_data.items():
            print(f"  {dataset_name}:")
            print(f"    PUT avg: {dataset_results['put_avg_ms']:.3f}ms")
            print(f"    GET avg: {dataset_results['get_avg_ms']:.3f}ms")
            print(f"    Hit rate: {dataset_results['hit_rate']:.2%}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--benchmark-only"])