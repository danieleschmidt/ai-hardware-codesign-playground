"""
Comprehensive Comparative Study Framework for AI Hardware Co-Design Research.

This module implements a robust framework for conducting comparative studies,
benchmarking algorithms, and validating research claims with statistical rigor.
"""

import asyncio
import time
import json
import statistics
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import defaultdict, Counter
import concurrent.futures
import numpy as np
from ..utils.statistical_validation import StatisticalValidator
from .novel_algorithms import (
    AlgorithmType, ExperimentConfig, ExperimentResult, 
    BaselineComparison, run_comparative_study
)
from .breakthrough_algorithms import (
    BreakthroughResearchManager, ResearchHypothesis, ExperimentalResult
)

logger = logging.getLogger(__name__)


class StudyType(Enum):
    """Types of comparative studies."""
    ALGORITHM_COMPARISON = "algorithm_comparison"
    BASELINE_VALIDATION = "baseline_validation"
    ABLATION_STUDY = "ablation_study"
    SCALABILITY_ANALYSIS = "scalability_analysis"
    ROBUSTNESS_EVALUATION = "robustness_evaluation"
    CROSS_DOMAIN_VALIDATION = "cross_domain_validation"
    REPRODUCIBILITY_STUDY = "reproducibility_study"
    META_ANALYSIS = "meta_analysis"


class EvaluationMetric(Enum):
    """Evaluation metrics for comparative studies."""
    CONVERGENCE_SPEED = "convergence_speed"
    SOLUTION_QUALITY = "solution_quality"
    COMPUTATIONAL_EFFICIENCY = "computational_efficiency"
    MEMORY_USAGE = "memory_usage"
    SCALABILITY = "scalability"
    ROBUSTNESS = "robustness"
    REPRODUCIBILITY = "reproducibility"
    STATISTICAL_SIGNIFICANCE = "statistical_significance"


@dataclass
class StudyConfiguration:
    """Configuration for comparative studies."""
    
    study_id: str
    study_type: StudyType
    algorithms_to_compare: List[str]
    evaluation_metrics: List[EvaluationMetric]
    number_of_runs: int = 30
    significance_level: float = 0.05
    effect_size_threshold: float = 0.5
    timeout_per_run: int = 3600  # seconds
    random_seeds: Optional[List[int]] = None
    problem_instances: List[str] = field(default_factory=list)
    study_parameters: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.random_seeds is None:
            self.random_seeds = list(range(self.number_of_runs))


@dataclass
class StudyResult:
    """Results from comparative study."""
    
    study_id: str
    study_type: StudyType
    algorithms_results: Dict[str, List[Dict[str, Any]]]
    statistical_analysis: Dict[str, Any]
    performance_rankings: List[Tuple[str, float]]
    effect_sizes: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]
    study_duration: float
    validated_claims: List[str]
    rejected_claims: List[str]
    recommendations: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "study_id": self.study_id,
            "study_type": self.study_type.value,
            "algorithms_results": self.algorithms_results,
            "statistical_analysis": self.statistical_analysis,
            "performance_rankings": self.performance_rankings,
            "effect_sizes": self.effect_sizes,
            "confidence_intervals": self.confidence_intervals,
            "study_duration": self.study_duration,
            "validated_claims": self.validated_claims,
            "rejected_claims": self.rejected_claims,
            "recommendations": self.recommendations,
            "metadata": self.metadata
        }


class BenchmarkSuite:
    """Comprehensive benchmark suite for hardware co-design algorithms."""
    
    def __init__(self):
        """Initialize benchmark suite."""
        self.benchmark_problems: Dict[str, Callable] = {}
        self.problem_characteristics: Dict[str, Dict[str, Any]] = {}
        self.ground_truth_solutions: Dict[str, Any] = {}
        
        # Initialize standard benchmark problems
        self._initialize_standard_benchmarks()
        
        logger.info("Benchmark suite initialized")
    
    def _initialize_standard_benchmarks(self) -> None:
        """Initialize standard benchmark problems."""
        
        # Neural Architecture Search Benchmarks
        self.benchmark_problems["nas_cifar10"] = self._nas_cifar10_benchmark
        self.problem_characteristics["nas_cifar10"] = {
            "domain": "neural_architecture_search",
            "complexity": "medium",
            "search_space_size": 10**15,
            "evaluation_time": "moderate",
            "known_optimum": False
        }
        
        # Hardware Accelerator Design Benchmarks  
        self.benchmark_problems["accelerator_design_cnn"] = self._accelerator_cnn_benchmark
        self.problem_characteristics["accelerator_design_cnn"] = {
            "domain": "accelerator_design",
            "complexity": "high",
            "search_space_size": 10**12,
            "evaluation_time": "high",
            "known_optimum": False
        }
        
        # Memory Hierarchy Optimization
        self.benchmark_problems["memory_hierarchy_opt"] = self._memory_hierarchy_benchmark
        self.problem_characteristics["memory_hierarchy_opt"] = {
            "domain": "memory_optimization",
            "complexity": "high",
            "search_space_size": 10**8,
            "evaluation_time": "moderate",
            "known_optimum": False
        }
        
        # Dataflow Optimization
        self.benchmark_problems["dataflow_systolic"] = self._dataflow_systolic_benchmark
        self.problem_characteristics["dataflow_systolic"] = {
            "domain": "dataflow_optimization",
            "complexity": "medium",
            "search_space_size": 10**6,
            "evaluation_time": "low",
            "known_optimum": True
        }
        
        # Multi-objective Hardware-Software Co-design
        self.benchmark_problems["multiobjective_codesign"] = self._multiobjective_codesign_benchmark
        self.problem_characteristics["multiobjective_codesign"] = {
            "domain": "hardware_software_codesign",
            "complexity": "very_high",
            "search_space_size": 10**20,
            "evaluation_time": "very_high",
            "known_optimum": False
        }
        
        logger.info(f"Initialized {len(self.benchmark_problems)} benchmark problems")
    
    def _nas_cifar10_benchmark(self, architecture_config: Dict[str, Any]) -> float:
        """Neural Architecture Search benchmark for CIFAR-10."""
        # Simulate architecture evaluation
        layers = architecture_config.get("layers", 10)
        channels = architecture_config.get("channels", [32, 64, 128])
        operations = architecture_config.get("operations", ["conv3x3", "conv5x5"])
        
        # Mock accuracy based on architecture complexity
        complexity_score = layers * len(channels) * len(operations)
        base_accuracy = 0.85
        
        # Add noise and complexity effects
        accuracy = base_accuracy + 0.1 * np.tanh(complexity_score / 1000) + np.random.normal(0, 0.02)
        
        # Penalty for overly complex architectures
        if complexity_score > 5000:
            accuracy *= 0.95
        
        return min(0.98, max(0.60, accuracy))
    
    def _accelerator_cnn_benchmark(self, accelerator_config: Dict[str, Any]) -> float:
        """Accelerator design benchmark for CNN workloads."""
        pe_count = accelerator_config.get("pe_count", 64)
        memory_size = accelerator_config.get("memory_size_kb", 512)
        dataflow = accelerator_config.get("dataflow", "weight_stationary")
        frequency = accelerator_config.get("frequency_mhz", 200)
        
        # Calculate performance metrics
        compute_throughput = pe_count * frequency * 1e6  # ops/sec
        memory_bandwidth = memory_size * 1000 * frequency  # bytes/sec
        
        # Dataflow efficiency
        dataflow_efficiency = {
            "weight_stationary": 0.85,
            "output_stationary": 0.80,
            "row_stationary": 0.75
        }.get(dataflow, 0.70)
        
        # Overall performance score
        performance = (compute_throughput * 0.4 + memory_bandwidth * 0.3 + 
                      dataflow_efficiency * frequency * 0.3) / 1e9
        
        # Add noise
        performance += np.random.normal(0, performance * 0.05)
        
        return max(0.1, performance)
    
    def _memory_hierarchy_benchmark(self, memory_config: Dict[str, Any]) -> float:
        """Memory hierarchy optimization benchmark."""
        l1_size = memory_config.get("l1_cache_kb", 32)
        l2_size = memory_config.get("l2_cache_kb", 256)
        l3_size = memory_config.get("l3_cache_kb", 2048)
        bandwidth = memory_config.get("memory_bandwidth_gb_s", 25.6)
        
        # Calculate memory hierarchy efficiency
        cache_hierarchy_efficiency = (
            l1_size * 10 +     # L1 has highest impact
            l2_size * 3 +      # L2 medium impact
            l3_size * 1        # L3 lower impact
        ) / 10000
        
        bandwidth_efficiency = min(1.0, bandwidth / 100.0)  # Normalize
        
        # Combined efficiency score
        efficiency = (cache_hierarchy_efficiency * 0.6 + bandwidth_efficiency * 0.4)
        
        # Add realistic noise
        efficiency += np.random.normal(0, 0.02)
        
        return max(0.1, min(1.0, efficiency))
    
    def _dataflow_systolic_benchmark(self, dataflow_config: Dict[str, Any]) -> float:
        """Dataflow optimization benchmark for systolic arrays."""
        array_size = dataflow_config.get("array_size", (16, 16))
        tiling_strategy = dataflow_config.get("tiling_strategy", "square")
        buffer_size = dataflow_config.get("buffer_size_kb", 64)
        
        # Calculate utilization efficiency
        array_utilization = min(1.0, (array_size[0] * array_size[1]) / 256)  # Normalize
        
        # Tiling efficiency
        tiling_efficiency = {
            "square": 0.95,
            "rectangular": 0.88,
            "irregular": 0.75
        }.get(tiling_strategy, 0.70)
        
        # Buffer efficiency
        buffer_efficiency = min(1.0, buffer_size / 128)  # Optimal at 128KB
        
        # Overall efficiency (this benchmark has a known near-optimal solution)
        efficiency = array_utilization * 0.5 + tiling_efficiency * 0.3 + buffer_efficiency * 0.2
        
        # Known optimum is around 0.92
        efficiency += np.random.normal(0, 0.01)
        
        return max(0.2, min(0.95, efficiency))
    
    def _multiobjective_codesign_benchmark(self, codesign_config: Dict[str, Any]) -> float:
        """Multi-objective hardware-software co-design benchmark."""
        # Software configuration
        model_complexity = codesign_config.get("model_complexity", 1.0)
        optimization_level = codesign_config.get("optimization_level", "O2")
        quantization_bits = codesign_config.get("quantization_bits", 8)
        
        # Hardware configuration
        compute_units = codesign_config.get("compute_units", 64)
        memory_bandwidth = codesign_config.get("memory_bandwidth", 25.6)
        power_budget = codesign_config.get("power_budget_w", 10.0)
        
        # Performance objectives
        performance = (compute_units * 1e6 / model_complexity) / 1e9  # GOPS
        
        # Power efficiency
        power_efficiency = performance / power_budget if power_budget > 0 else 0
        
        # Accuracy (affected by quantization)
        accuracy = 0.95 - (16 - quantization_bits) * 0.01
        
        # Combined score (Pareto optimal solutions)
        combined_score = (performance * 0.4 + power_efficiency * 0.3 + accuracy * 0.3)
        
        # Add significant noise due to complexity
        combined_score += np.random.normal(0, combined_score * 0.1)
        
        return max(0.1, combined_score)
    
    def get_benchmark_problem(self, problem_name: str) -> Optional[Callable]:
        """Get benchmark problem by name."""
        return self.benchmark_problems.get(problem_name)
    
    def list_available_benchmarks(self) -> List[str]:
        """List all available benchmark problems."""
        return list(self.benchmark_problems.keys())
    
    def get_problem_characteristics(self, problem_name: str) -> Optional[Dict[str, Any]]:
        """Get characteristics of benchmark problem."""
        return self.problem_characteristics.get(problem_name)
    
    def validate_benchmark_solution(self, problem_name: str, solution: Dict[str, Any]) -> bool:
        """Validate if solution is feasible for benchmark problem."""
        # Basic validation - could be more sophisticated
        characteristics = self.get_problem_characteristics(problem_name)
        
        if not characteristics:
            return False
        
        # Check if solution has required parameters
        required_params = self._get_required_parameters(problem_name)
        
        return all(param in solution for param in required_params)
    
    def _get_required_parameters(self, problem_name: str) -> List[str]:
        """Get required parameters for benchmark problem."""
        param_map = {
            "nas_cifar10": ["layers", "channels", "operations"],
            "accelerator_design_cnn": ["pe_count", "memory_size_kb", "dataflow", "frequency_mhz"],
            "memory_hierarchy_opt": ["l1_cache_kb", "l2_cache_kb", "l3_cache_kb", "memory_bandwidth_gb_s"],
            "dataflow_systolic": ["array_size", "tiling_strategy", "buffer_size_kb"],
            "multiobjective_codesign": ["model_complexity", "compute_units", "power_budget_w"]
        }
        
        return param_map.get(problem_name, [])


class ComparativeStudyEngine:
    """Engine for conducting comprehensive comparative studies."""
    
    def __init__(self):
        """Initialize comparative study engine."""
        self.benchmark_suite = BenchmarkSuite()
        self.baseline_comparison = BaselineComparison()
        self.statistical_validator = StatisticalValidator()
        self.active_studies: Dict[str, StudyConfiguration] = {}
        self.completed_studies: Dict[str, StudyResult] = {}
        
        logger.info("Comparative study engine initialized")
    
    async def conduct_comparative_study(
        self,
        study_config: StudyConfiguration,
        algorithms: Dict[str, Any],
        benchmark_problems: Optional[List[str]] = None
    ) -> StudyResult:
        """Conduct comprehensive comparative study."""
        logger.info(f"Starting comparative study: {study_config.study_id}")
        
        start_time = time.time()
        self.active_studies[study_config.study_id] = study_config
        
        try:
            # Select benchmark problems
            if benchmark_problems is None:
                benchmark_problems = ["nas_cifar10", "accelerator_design_cnn", "memory_hierarchy_opt"]
            
            # Run experiments for each algorithm
            algorithms_results = {}
            
            for algorithm_name in study_config.algorithms_to_compare:
                if algorithm_name in algorithms:
                    algorithm_results = await self._evaluate_algorithm_comprehensive(
                        algorithm_name,
                        algorithms[algorithm_name],
                        benchmark_problems,
                        study_config
                    )
                    algorithms_results[algorithm_name] = algorithm_results
                else:
                    logger.warning(f"Algorithm {algorithm_name} not found in provided algorithms")
            
            # Conduct statistical analysis
            statistical_analysis = await self._conduct_statistical_analysis(
                algorithms_results, study_config
            )
            
            # Calculate performance rankings
            performance_rankings = self._calculate_performance_rankings(
                algorithms_results, study_config.evaluation_metrics
            )
            
            # Calculate effect sizes
            effect_sizes = self._calculate_effect_sizes(algorithms_results)
            
            # Calculate confidence intervals
            confidence_intervals = self._calculate_confidence_intervals(algorithms_results)
            
            # Validate research claims
            validated_claims, rejected_claims = self._validate_research_claims(
                algorithms_results, statistical_analysis, study_config
            )
            
            # Generate recommendations
            recommendations = self._generate_study_recommendations(
                algorithms_results, statistical_analysis, performance_rankings
            )
            
            study_duration = time.time() - start_time
            
            # Create study result
            study_result = StudyResult(
                study_id=study_config.study_id,
                study_type=study_config.study_type,
                algorithms_results=algorithms_results,
                statistical_analysis=statistical_analysis,
                performance_rankings=performance_rankings,
                effect_sizes=effect_sizes,
                confidence_intervals=confidence_intervals,
                study_duration=study_duration,
                validated_claims=validated_claims,
                rejected_claims=rejected_claims,
                recommendations=recommendations,
                metadata={
                    "benchmark_problems": benchmark_problems,
                    "number_of_runs": study_config.number_of_runs,
                    "significance_level": study_config.significance_level,
                    "total_evaluations": sum(
                        len(results) for results in algorithms_results.values()
                    )
                }
            )
            
            # Store completed study
            self.completed_studies[study_config.study_id] = study_result
            
            logger.info(f"Completed comparative study: {study_config.study_id} in {study_duration:.2f}s")
            return study_result
            
        finally:
            # Remove from active studies
            if study_config.study_id in self.active_studies:
                del self.active_studies[study_config.study_id]
    
    async def _evaluate_algorithm_comprehensive(
        self,
        algorithm_name: str,
        algorithm: Any,
        benchmark_problems: List[str],
        study_config: StudyConfiguration
    ) -> List[Dict[str, Any]]:
        """Evaluate algorithm comprehensively across all benchmarks."""
        all_results = []
        
        for problem_name in benchmark_problems:
            benchmark_func = self.benchmark_suite.get_benchmark_problem(problem_name)
            if not benchmark_func:
                logger.warning(f"Benchmark problem {problem_name} not found")
                continue
            
            # Run multiple trials
            problem_results = []
            
            for run_idx in range(study_config.number_of_runs):
                seed = study_config.random_seeds[run_idx] if study_config.random_seeds else run_idx
                
                try:
                    # Set seed for reproducibility
                    np.random.seed(seed)
                    
                    # Generate problem-specific search space
                    search_space = self._generate_search_space(problem_name)
                    
                    # Run algorithm
                    result = await self._run_algorithm_with_timeout(
                        algorithm, 
                        benchmark_func, 
                        search_space,
                        study_config.timeout_per_run,
                        seed
                    )
                    
                    # Add metadata
                    result.update({
                        "problem": problem_name,
                        "run_id": run_idx,
                        "seed": seed,
                        "algorithm": algorithm_name
                    })
                    
                    problem_results.append(result)
                    
                except asyncio.TimeoutError:
                    logger.warning(f"Algorithm {algorithm_name} timed out on {problem_name}, run {run_idx}")
                    problem_results.append({
                        "problem": problem_name,
                        "run_id": run_idx,
                        "seed": seed,
                        "algorithm": algorithm_name,
                        "best_fitness": 0.0,
                        "timeout": True,
                        "execution_time": study_config.timeout_per_run
                    })
                except Exception as e:
                    logger.error(f"Error running {algorithm_name} on {problem_name}: {e}")
                    problem_results.append({
                        "problem": problem_name,
                        "run_id": run_idx,
                        "seed": seed,
                        "algorithm": algorithm_name,
                        "best_fitness": 0.0,
                        "error": str(e),
                        "execution_time": 0.0
                    })
            
            all_results.extend(problem_results)
        
        return all_results
    
    def _generate_search_space(self, problem_name: str) -> Dict[str, Any]:
        """Generate appropriate search space for benchmark problem."""
        search_spaces = {
            "nas_cifar10": {
                "layers": list(range(5, 20)),
                "channels": [[16, 32], [32, 64], [32, 64, 128], [64, 128, 256]],
                "operations": [["conv3x3"], ["conv5x5"], ["conv3x3", "conv5x5"], ["conv1x1", "conv3x3"]]
            },
            "accelerator_design_cnn": {
                "pe_count": list(range(16, 129, 16)),
                "memory_size_kb": [64, 128, 256, 512, 1024],
                "dataflow": ["weight_stationary", "output_stationary", "row_stationary"],
                "frequency_mhz": list(range(100, 501, 50))
            },
            "memory_hierarchy_opt": {
                "l1_cache_kb": [16, 32, 64],
                "l2_cache_kb": [128, 256, 512, 1024],
                "l3_cache_kb": [1024, 2048, 4096, 8192],
                "memory_bandwidth_gb_s": [12.8, 25.6, 51.2, 102.4]
            },
            "dataflow_systolic": {
                "array_size": [(8, 8), (16, 16), (32, 32), (64, 64)],
                "tiling_strategy": ["square", "rectangular", "irregular"],
                "buffer_size_kb": [32, 64, 128, 256]
            },
            "multiobjective_codesign": {
                "model_complexity": [0.5, 0.8, 1.0, 1.5, 2.0],
                "compute_units": list(range(32, 257, 32)),
                "power_budget_w": [3.0, 5.0, 10.0, 15.0, 25.0],
                "quantization_bits": [4, 8, 16],
                "optimization_level": ["O0", "O1", "O2", "O3"]
            }
        }
        
        return search_spaces.get(problem_name, {})
    
    async def _run_algorithm_with_timeout(
        self,
        algorithm: Any,
        objective_function: Callable,
        search_space: Dict[str, Any],
        timeout: int,
        seed: int
    ) -> Dict[str, Any]:
        """Run algorithm with timeout handling."""
        
        async def run_algorithm():
            if hasattr(algorithm, 'optimize'):
                # For novel algorithms with async optimize method
                result = await algorithm.optimize(objective_function, search_space)
                return result.to_dict() if hasattr(result, 'to_dict') else result
            elif hasattr(algorithm, 'run_baseline_comparison'):
                # For baseline algorithms
                result = algorithm._random_search(objective_function, search_space, seed)
                return result
            else:
                # Generic algorithm interface
                return {"best_fitness": 0.5, "execution_time": 1.0}
        
        try:
            return await asyncio.wait_for(run_algorithm(), timeout=timeout)
        except asyncio.TimeoutError:
            raise
    
    async def _conduct_statistical_analysis(
        self,
        algorithms_results: Dict[str, List[Dict[str, Any]]],
        study_config: StudyConfiguration
    ) -> Dict[str, Any]:
        """Conduct comprehensive statistical analysis."""
        statistical_results = {}
        
        # Extract performance metrics for each algorithm
        algorithm_metrics = {}
        for algorithm_name, results in algorithms_results.items():
            metrics = defaultdict(list)
            
            for result in results:
                if "best_fitness" in result and not result.get("timeout", False) and not result.get("error"):
                    metrics["best_fitness"].append(result["best_fitness"])
                    metrics["execution_time"].append(result.get("execution_time", 0.0))
            
            algorithm_metrics[algorithm_name] = dict(metrics)
        
        # Pairwise statistical comparisons
        algorithms = list(algorithm_metrics.keys())
        comparisons = {}
        
        for i in range(len(algorithms)):
            for j in range(i + 1, len(algorithms)):
                alg1, alg2 = algorithms[i], algorithms[j]
                
                # Compare best fitness
                if ("best_fitness" in algorithm_metrics[alg1] and 
                    "best_fitness" in algorithm_metrics[alg2]):
                    
                    fitness1 = algorithm_metrics[alg1]["best_fitness"]
                    fitness2 = algorithm_metrics[alg2]["best_fitness"]
                    
                    comparison_result = self.statistical_validator.compare_algorithms(
                        fitness1, fitness2, alpha=study_config.significance_level
                    )
                    
                    comparisons[f"{alg1}_vs_{alg2}"] = comparison_result
        
        statistical_results["pairwise_comparisons"] = comparisons
        
        # Descriptive statistics for each algorithm
        descriptive_stats = {}
        for algorithm_name, metrics in algorithm_metrics.items():
            alg_stats = {}
            
            for metric_name, values in metrics.items():
                if values:
                    alg_stats[metric_name] = {
                        "mean": statistics.mean(values),
                        "median": statistics.median(values),
                        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                        "min": min(values),
                        "max": max(values),
                        "count": len(values)
                    }
            
            descriptive_stats[algorithm_name] = alg_stats
        
        statistical_results["descriptive_statistics"] = descriptive_stats
        
        # Overall significance test (ANOVA or Kruskal-Wallis)
        all_fitness_values = []
        algorithm_labels = []
        
        for algorithm_name, metrics in algorithm_metrics.items():
            if "best_fitness" in metrics:
                all_fitness_values.extend(metrics["best_fitness"])
                algorithm_labels.extend([algorithm_name] * len(metrics["best_fitness"]))
        
        if len(set(algorithm_labels)) > 2:  # Multiple algorithms
            overall_test = self.statistical_validator.kruskal_wallis_test(
                all_fitness_values, algorithm_labels
            )
            statistical_results["overall_significance"] = overall_test
        
        return statistical_results
    
    def _calculate_performance_rankings(
        self,
        algorithms_results: Dict[str, List[Dict[str, Any]]],
        evaluation_metrics: List[EvaluationMetric]
    ) -> List[Tuple[str, float]]:
        """Calculate performance rankings across algorithms."""
        algorithm_scores = {}
        
        for algorithm_name, results in algorithms_results.items():
            # Extract valid results (no timeout, no error)
            valid_results = [
                r for r in results 
                if not r.get("timeout", False) and not r.get("error") and "best_fitness" in r
            ]
            
            if not valid_results:
                algorithm_scores[algorithm_name] = 0.0
                continue
            
            scores = []
            
            # Calculate scores based on evaluation metrics
            for metric in evaluation_metrics:
                if metric == EvaluationMetric.SOLUTION_QUALITY:
                    fitness_values = [r["best_fitness"] for r in valid_results]
                    scores.append(statistics.mean(fitness_values))
                
                elif metric == EvaluationMetric.CONVERGENCE_SPEED:
                    # Inverse of execution time (faster is better)
                    times = [r.get("execution_time", float('inf')) for r in valid_results]
                    avg_time = statistics.mean(times)
                    scores.append(1.0 / (avg_time + 1e-6))
                
                elif metric == EvaluationMetric.ROBUSTNESS:
                    # Inverse of standard deviation (lower variance is better)
                    fitness_values = [r["best_fitness"] for r in valid_results]
                    if len(fitness_values) > 1:
                        std_dev = statistics.stdev(fitness_values)
                        scores.append(1.0 / (std_dev + 1e-6))
                    else:
                        scores.append(1.0)
                
                elif metric == EvaluationMetric.REPRODUCIBILITY:
                    # Success rate (fraction of successful runs)
                    total_runs = len(results)
                    successful_runs = len(valid_results)
                    scores.append(successful_runs / total_runs)
            
            # Combined score (weighted average)
            combined_score = statistics.mean(scores) if scores else 0.0
            algorithm_scores[algorithm_name] = combined_score
        
        # Sort by score (descending)
        rankings = sorted(algorithm_scores.items(), key=lambda x: x[1], reverse=True)
        
        return rankings
    
    def _calculate_effect_sizes(self, algorithms_results: Dict[str, List[Dict[str, Any]]]) -> Dict[str, float]:
        """Calculate effect sizes between algorithms."""
        effect_sizes = {}
        algorithms = list(algorithms_results.keys())
        
        for i in range(len(algorithms)):
            for j in range(i + 1, len(algorithms)):
                alg1, alg2 = algorithms[i], algorithms[j]
                
                # Extract fitness values
                fitness1 = [
                    r["best_fitness"] for r in algorithms_results[alg1]
                    if "best_fitness" in r and not r.get("timeout", False) and not r.get("error")
                ]
                fitness2 = [
                    r["best_fitness"] for r in algorithms_results[alg2]
                    if "best_fitness" in r and not r.get("timeout", False) and not r.get("error")
                ]
                
                if fitness1 and fitness2:
                    effect_size = self.statistical_validator.effect_size(fitness1, fitness2)
                    effect_sizes[f"{alg1}_vs_{alg2}"] = effect_size
        
        return effect_sizes
    
    def _calculate_confidence_intervals(
        self, 
        algorithms_results: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, Tuple[float, float]]:
        """Calculate confidence intervals for algorithm performance."""
        confidence_intervals = {}
        
        for algorithm_name, results in algorithms_results.items():
            fitness_values = [
                r["best_fitness"] for r in results
                if "best_fitness" in r and not r.get("timeout", False) and not r.get("error")
            ]
            
            if len(fitness_values) >= 2:
                ci = self.statistical_validator.confidence_interval(fitness_values)
                confidence_intervals[algorithm_name] = ci
        
        return confidence_intervals
    
    def _validate_research_claims(
        self,
        algorithms_results: Dict[str, List[Dict[str, Any]]],
        statistical_analysis: Dict[str, Any],
        study_config: StudyConfiguration
    ) -> Tuple[List[str], List[str]]:
        """Validate research claims based on statistical evidence."""
        validated_claims = []
        rejected_claims = []
        
        # Check for significant improvements
        pairwise_comparisons = statistical_analysis.get("pairwise_comparisons", {})
        
        for comparison_name, comparison_result in pairwise_comparisons.items():
            if comparison_result.get("significant", False):
                algorithms = comparison_name.split("_vs_")
                if len(algorithms) == 2:
                    alg1, alg2 = algorithms
                    
                    # Determine which algorithm is better
                    desc_stats = statistical_analysis.get("descriptive_statistics", {})
                    if alg1 in desc_stats and alg2 in desc_stats:
                        mean1 = desc_stats[alg1].get("best_fitness", {}).get("mean", 0)
                        mean2 = desc_stats[alg2].get("best_fitness", {}).get("mean", 0)
                        
                        if mean1 > mean2:
                            validated_claims.append(f"{alg1} significantly outperforms {alg2}")
                        else:
                            validated_claims.append(f"{alg2} significantly outperforms {alg1}")
            else:
                algorithms = comparison_name.split("_vs_")
                if len(algorithms) == 2:
                    rejected_claims.append(f"No significant difference between {algorithms[0]} and {algorithms[1]}")
        
        # Check effect sizes
        effect_sizes = self._calculate_effect_sizes(algorithms_results)
        for comparison, effect_size in effect_sizes.items():
            if abs(effect_size) >= study_config.effect_size_threshold:
                validated_claims.append(f"Large effect size ({effect_size:.2f}) found for {comparison}")
            else:
                rejected_claims.append(f"Small effect size ({effect_size:.2f}) for {comparison}")
        
        return validated_claims, rejected_claims
    
    def _generate_study_recommendations(
        self,
        algorithms_results: Dict[str, List[Dict[str, Any]]],
        statistical_analysis: Dict[str, Any],
        performance_rankings: List[Tuple[str, float]]
    ) -> List[str]:
        """Generate recommendations based on study results."""
        recommendations = []
        
        # Best performing algorithm
        if performance_rankings:
            best_algorithm = performance_rankings[0][0]
            recommendations.append(f"Recommend {best_algorithm} for best overall performance")
        
        # Statistical significance findings
        pairwise_comparisons = statistical_analysis.get("pairwise_comparisons", {})
        significant_comparisons = [
            comp for comp, result in pairwise_comparisons.items()
            if result.get("significant", False)
        ]
        
        if significant_comparisons:
            recommendations.append(f"Found {len(significant_comparisons)} statistically significant differences")
        else:
            recommendations.append("No statistically significant differences found - consider larger sample sizes")
        
        # Robustness analysis
        desc_stats = statistical_analysis.get("descriptive_statistics", {})
        most_robust_algorithm = None
        lowest_variance = float('inf')
        
        for algorithm, stats in desc_stats.items():
            if "best_fitness" in stats:
                variance = stats["best_fitness"].get("std", float('inf'))**2
                if variance < lowest_variance:
                    lowest_variance = variance
                    most_robust_algorithm = algorithm
        
        if most_robust_algorithm:
            recommendations.append(f"Most robust algorithm: {most_robust_algorithm} (lowest variance)")
        
        # Success rate analysis
        success_rates = {}
        for algorithm, results in algorithms_results.items():
            total_runs = len(results)
            successful_runs = len([r for r in results if not r.get("timeout", False) and not r.get("error")])
            success_rates[algorithm] = successful_runs / total_runs
        
        most_reliable = max(success_rates.items(), key=lambda x: x[1])
        if most_reliable[1] < 1.0:
            recommendations.append(f"Reliability concerns: {most_reliable[0]} has {most_reliable[1]:.1%} success rate")
        
        # Computational efficiency
        for algorithm, stats in desc_stats.items():
            if "execution_time" in stats:
                avg_time = stats["execution_time"].get("mean", 0)
                if avg_time > 300:  # 5 minutes
                    recommendations.append(f"{algorithm} has high computational cost ({avg_time:.1f}s average)")
        
        return recommendations
    
    def get_study_summary(self, study_id: str) -> Optional[Dict[str, Any]]:
        """Get summary of completed study."""
        if study_id not in self.completed_studies:
            return None
        
        study = self.completed_studies[study_id]
        
        return {
            "study_id": study_id,
            "study_type": study.study_type.value,
            "algorithms_compared": len(study.algorithms_results),
            "performance_rankings": study.performance_rankings[:5],  # Top 5
            "key_findings": study.validated_claims[:3],  # Top 3
            "recommendations": study.recommendations[:3],  # Top 3
            "study_duration": study.study_duration,
            "total_evaluations": study.metadata.get("total_evaluations", 0)
        }
    
    def generate_comparative_study_report(self, study_id: str) -> Dict[str, Any]:
        """Generate comprehensive comparative study report."""
        if study_id not in self.completed_studies:
            return {"error": f"Study {study_id} not found"}
        
        study = self.completed_studies[study_id]
        
        report = {
            "title": f"Comparative Study Report: {study_id}",
            "study_configuration": {
                "study_type": study.study_type.value,
                "algorithms_compared": len(study.algorithms_results),
                "number_of_runs": study.metadata.get("number_of_runs", 0),
                "benchmark_problems": study.metadata.get("benchmark_problems", [])
            },
            "executive_summary": {
                "best_performing_algorithm": study.performance_rankings[0][0] if study.performance_rankings else "None",
                "significant_findings": len(study.validated_claims),
                "overall_conclusion": self._generate_overall_conclusion(study)
            },
            "detailed_results": {
                "performance_rankings": study.performance_rankings,
                "statistical_analysis": study.statistical_analysis,
                "effect_sizes": study.effect_sizes,
                "confidence_intervals": study.confidence_intervals
            },
            "validated_claims": study.validated_claims,
            "rejected_claims": study.rejected_claims,
            "recommendations": study.recommendations,
            "study_metadata": {
                "duration": study.study_duration,
                "timestamp": time.time(),
                "total_evaluations": study.metadata.get("total_evaluations", 0)
            }
        }
        
        return report
    
    def _generate_overall_conclusion(self, study: StudyResult) -> str:
        """Generate overall conclusion for study."""
        if not study.performance_rankings:
            return "No conclusive results - insufficient valid data"
        
        best_algorithm = study.performance_rankings[0][0]
        best_score = study.performance_rankings[0][1]
        
        if len(study.validated_claims) > len(study.rejected_claims):
            return f"{best_algorithm} demonstrates superior performance with statistically significant improvements"
        elif best_score > 0.8:
            return f"{best_algorithm} shows strong performance but with limited statistical evidence"
        else:
            return "Results inconclusive - algorithms show similar performance across benchmarks"


# Global comparative study engine instance
_comparative_study_engine: Optional[ComparativeStudyEngine] = None


def get_comparative_study_engine() -> ComparativeStudyEngine:
    """Get comparative study engine instance."""
    global _comparative_study_engine
    
    if _comparative_study_engine is None:
        _comparative_study_engine = ComparativeStudyEngine()
    
    return _comparative_study_engine


async def run_algorithm_benchmark_study(
    algorithms: Dict[str, Any],
    study_type: StudyType = StudyType.ALGORITHM_COMPARISON,
    number_of_runs: int = 20,
    benchmark_problems: Optional[List[str]] = None
) -> StudyResult:
    """Run algorithm benchmark study with default configuration."""
    
    study_config = StudyConfiguration(
        study_id=f"benchmark_study_{int(time.time())}",
        study_type=study_type,
        algorithms_to_compare=list(algorithms.keys()),
        evaluation_metrics=[
            EvaluationMetric.SOLUTION_QUALITY,
            EvaluationMetric.CONVERGENCE_SPEED,
            EvaluationMetric.ROBUSTNESS,
            EvaluationMetric.REPRODUCIBILITY
        ],
        number_of_runs=number_of_runs
    )
    
    engine = get_comparative_study_engine()
    return await engine.conduct_comparative_study(
        study_config, algorithms, benchmark_problems
    )