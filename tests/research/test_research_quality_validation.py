"""
Research Quality Validation Framework for AI Hardware Co-Design Playground.

This module implements comprehensive validation of research quality including
statistical significance, reproducibility, experimental methodology, and
scientific rigor validation.
"""

import pytest
import numpy as np
import scipy.stats as stats
import time
import json
import hashlib
from typing import Dict, List, Any, Tuple, Optional
from unittest.mock import Mock, patch
from pathlib import Path
import tempfile

from codesign_playground.core.accelerator import AcceleratorDesigner, ModelProfile
from codesign_playground.core.optimizer import ModelOptimizer
from codesign_playground.core.explorer import DesignSpaceExplorer
from codesign_playground.utils.statistical_validation import StatisticalValidator


class ResearchQualityFramework:
    """Framework for validating research quality and scientific rigor."""
    
    def __init__(self):
        self.experiments = []
        self.validation_results = {}
        self.quality_metrics = {
            "statistical_significance": False,
            "reproducibility_score": 0.0,
            "experimental_rigor": 0.0,
            "peer_review_readiness": 0.0
        }
    
    def register_experiment(self, name: str, description: str, methodology: Dict[str, Any]):
        """Register an experiment for quality validation."""
        experiment = {
            "name": name,
            "description": description,
            "methodology": methodology,
            "results": [],
            "replications": [],
            "validated": False,
            "quality_score": 0.0
        }
        self.experiments.append(experiment)
        return len(self.experiments) - 1  # Return experiment ID
    
    def add_experiment_result(self, experiment_id: int, result: Dict[str, Any], run_metadata: Dict = None):
        """Add a result to an experiment."""
        if 0 <= experiment_id < len(self.experiments):
            self.experiments[experiment_id]["results"].append({
                "data": result,
                "metadata": run_metadata or {},
                "timestamp": time.time()
            })
    
    def validate_statistical_significance(self, experiment_id: int, alpha: float = 0.05) -> Dict[str, Any]:
        """Validate statistical significance of experimental results."""
        if experiment_id >= len(self.experiments):
            raise ValueError("Invalid experiment ID")
        
        experiment = self.experiments[experiment_id]
        results = experiment["results"]
        
        if len(results) < 3:
            return {
                "significant": False,
                "reason": "Insufficient samples for statistical analysis",
                "required_samples": 3,
                "actual_samples": len(results)
            }
        
        # Extract numerical metrics for analysis
        metrics = {}
        for result in results:
            for key, value in result["data"].items():
                if isinstance(value, (int, float)):
                    if key not in metrics:
                        metrics[key] = []
                    metrics[key].append(value)
        
        significance_results = {}
        
        for metric_name, values in metrics.items():
            if len(values) >= 3:
                # Test for normality
                if len(values) >= 8:
                    normality_stat, normality_p = stats.shapiro(values)
                    is_normal = normality_p > alpha
                else:
                    is_normal = True  # Assume normal for small samples
                
                # Calculate confidence interval
                mean_val = np.mean(values)
                std_val = np.std(values, ddof=1)
                n = len(values)
                
                if is_normal:
                    # Use t-distribution for small samples
                    t_critical = stats.t.ppf(1 - alpha/2, df=n-1)
                    margin_error = t_critical * (std_val / np.sqrt(n))
                else:
                    # Use bootstrap for non-normal data
                    bootstrap_means = []
                    for _ in range(1000):
                        bootstrap_sample = np.random.choice(values, size=n, replace=True)
                        bootstrap_means.append(np.mean(bootstrap_sample))
                    
                    margin_error = np.std(bootstrap_means) * 1.96  # 95% CI
                
                ci_lower = mean_val - margin_error
                ci_upper = mean_val + margin_error
                
                # Effect size (Cohen's d if we have baseline)
                effect_size = None
                if len(values) > 1:
                    baseline = values[0]
                    effect_size = (mean_val - baseline) / std_val if std_val > 0 else 0
                
                significance_results[metric_name] = {
                    "mean": mean_val,
                    "std": std_val,
                    "n": n,
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                    "margin_error": margin_error,
                    "effect_size": effect_size,
                    "is_normal": is_normal,
                    "coefficient_of_variation": (std_val / mean_val) if mean_val != 0 else float('inf')
                }
        
        # Overall significance assessment
        significant_metrics = 0
        total_metrics = len(significance_results)
        
        for metric_name, stats_data in significance_results.items():
            # Consider significant if CV < 0.1 (low variability) and effect size > 0.2
            cv = stats_data["coefficient_of_variation"]
            effect = abs(stats_data["effect_size"]) if stats_data["effect_size"] is not None else 0
            
            if cv < 0.2 and effect > 0.2:  # Reasonable thresholds
                significant_metrics += 1
        
        overall_significant = (significant_metrics / max(1, total_metrics)) >= 0.5
        
        return {
            "significant": overall_significant,
            "alpha": alpha,
            "metrics_analyzed": total_metrics,
            "significant_metrics": significant_metrics,
            "significance_ratio": significant_metrics / max(1, total_metrics),
            "detailed_results": significance_results
        }
    
    def validate_reproducibility(self, experiment_id: int, tolerance: float = 0.1) -> Dict[str, Any]:
        """Validate reproducibility across multiple runs."""
        if experiment_id >= len(self.experiments):
            raise ValueError("Invalid experiment ID")
        
        experiment = self.experiments[experiment_id]
        results = experiment["results"]
        
        if len(results) < 3:
            return {
                "reproducible": False,
                "reason": "Insufficient runs for reproducibility analysis",
                "required_runs": 3,
                "actual_runs": len(results)
            }
        
        # Extract metrics for reproducibility analysis
        metrics = {}
        for result in results:
            for key, value in result["data"].items():
                if isinstance(value, (int, float)):
                    if key not in metrics:
                        metrics[key] = []
                    metrics[key].append(value)
        
        reproducibility_results = {}
        
        for metric_name, values in metrics.items():
            if len(values) >= 3:
                mean_val = np.mean(values)
                std_val = np.std(values, ddof=1)
                cv = std_val / mean_val if mean_val != 0 else float('inf')
                
                # Calculate relative standard deviation
                rsd = cv * 100  # As percentage
                
                # Reproducibility criteria
                is_reproducible = rsd <= (tolerance * 100)  # Convert tolerance to percentage
                
                reproducibility_results[metric_name] = {
                    "mean": mean_val,
                    "std": std_val,
                    "coefficient_of_variation": cv,
                    "relative_std_deviation_percent": rsd,
                    "reproducible": is_reproducible,
                    "tolerance_percent": tolerance * 100,
                    "runs": len(values)
                }
        
        # Overall reproducibility score
        reproducible_metrics = sum(1 for r in reproducibility_results.values() if r["reproducible"])
        total_metrics = len(reproducibility_results)
        reproducibility_score = reproducible_metrics / max(1, total_metrics)
        
        return {
            "reproducible": reproducibility_score >= 0.8,  # 80% of metrics must be reproducible
            "reproducibility_score": reproducibility_score,
            "tolerance": tolerance,
            "metrics_analyzed": total_metrics,
            "reproducible_metrics": reproducible_metrics,
            "detailed_results": reproducibility_results
        }
    
    def validate_experimental_methodology(self, experiment_id: int) -> Dict[str, Any]:
        """Validate experimental methodology and design quality."""
        if experiment_id >= len(self.experiments):
            raise ValueError("Invalid experiment ID")
        
        experiment = self.experiments[experiment_id]
        methodology = experiment["methodology"]
        
        methodology_score = 0.0
        max_score = 0.0
        validation_details = {}
        
        # Check for proper experimental design
        design_criteria = [
            ("hypothesis", "Clear hypothesis defined", 10),
            ("control_conditions", "Control conditions specified", 15),
            ("variables_controlled", "Controlled variables identified", 10),
            ("sample_size_justification", "Sample size justified", 10),
            ("randomization", "Randomization strategy", 8),
            ("blinding", "Blinding procedures", 5),
            ("replication_plan", "Replication plan defined", 10)
        ]
        
        for criterion, description, points in design_criteria:
            max_score += points
            if criterion in methodology:
                methodology_score += points
                validation_details[criterion] = {"present": True, "points": points}
            else:
                validation_details[criterion] = {"present": False, "points": 0}
        
        # Check for statistical power analysis
        if "power_analysis" in methodology:
            methodology_score += 15
            max_score += 15
            validation_details["power_analysis"] = {"present": True, "points": 15}
        else:
            max_score += 15
            validation_details["power_analysis"] = {"present": False, "points": 0}
        
        # Check for confounding variable control
        if "confounding_control" in methodology:
            methodology_score += 12
            max_score += 12
            validation_details["confounding_control"] = {"present": True, "points": 12}
        else:
            max_score += 12
            validation_details["confounding_control"] = {"present": False, "points": 0}
        
        # Assess data collection procedures
        data_collection_criteria = [
            ("measurement_procedures", "Measurement procedures defined", 8),
            ("data_quality_checks", "Data quality checks specified", 6),
            ("missing_data_handling", "Missing data handling plan", 5)
        ]
        
        for criterion, description, points in data_collection_criteria:
            max_score += points
            if criterion in methodology:
                methodology_score += points
                validation_details[criterion] = {"present": True, "points": points}
            else:
                validation_details[criterion] = {"present": False, "points": 0}
        
        methodology_quality = methodology_score / max_score
        
        return {
            "methodology_valid": methodology_quality >= 0.7,  # 70% threshold
            "quality_score": methodology_quality,
            "points_scored": methodology_score,
            "max_points": max_score,
            "detailed_validation": validation_details,
            "recommendations": self._generate_methodology_recommendations(validation_details)
        }
    
    def _generate_methodology_recommendations(self, validation_details: Dict) -> List[str]:
        """Generate methodology improvement recommendations."""
        recommendations = []
        
        missing_criteria = [k for k, v in validation_details.items() if not v["present"]]
        
        if "hypothesis" in missing_criteria:
            recommendations.append("Define clear, testable hypothesis")
        
        if "control_conditions" in missing_criteria:
            recommendations.append("Specify control conditions and baseline comparisons")
        
        if "sample_size_justification" in missing_criteria:
            recommendations.append("Provide statistical justification for sample size")
        
        if "power_analysis" in missing_criteria:
            recommendations.append("Conduct statistical power analysis")
        
        if "randomization" in missing_criteria:
            recommendations.append("Implement randomization strategy to reduce bias")
        
        return recommendations
    
    def assess_peer_review_readiness(self, experiment_id: int) -> Dict[str, Any]:
        """Assess readiness for peer review submission."""
        if experiment_id >= len(self.experiments):
            raise ValueError("Invalid experiment ID")
        
        experiment = self.experiments[experiment_id]
        
        # Run all validation checks
        significance_results = self.validate_statistical_significance(experiment_id)
        reproducibility_results = self.validate_reproducibility(experiment_id)
        methodology_results = self.validate_experimental_methodology(experiment_id)
        
        # Calculate overall readiness score
        readiness_components = {
            "statistical_significance": {
                "weight": 0.3,
                "score": 1.0 if significance_results["significant"] else 0.0
            },
            "reproducibility": {
                "weight": 0.3,
                "score": reproducibility_results["reproducibility_score"]
            },
            "methodology": {
                "weight": 0.4,
                "score": methodology_results["quality_score"]
            }
        }
        
        overall_score = sum(
            comp["weight"] * comp["score"] 
            for comp in readiness_components.values()
        )
        
        # Generate readiness assessment
        readiness_level = "Not Ready"
        if overall_score >= 0.9:
            readiness_level = "Excellent - Ready for High-Tier Venues"
        elif overall_score >= 0.8:
            readiness_level = "Good - Ready for Peer Review"
        elif overall_score >= 0.7:
            readiness_level = "Acceptable - Minor Revisions Needed"
        elif overall_score >= 0.6:
            readiness_level = "Needs Improvement - Major Revisions Required"
        
        return {
            "ready_for_peer_review": overall_score >= 0.7,
            "overall_score": overall_score,
            "readiness_level": readiness_level,
            "component_scores": readiness_components,
            "detailed_assessments": {
                "statistical_significance": significance_results,
                "reproducibility": reproducibility_results,
                "methodology": methodology_results
            },
            "improvement_priorities": self._generate_improvement_priorities(readiness_components)
        }
    
    def _generate_improvement_priorities(self, components: Dict) -> List[str]:
        """Generate prioritized improvement recommendations."""
        priorities = []
        
        # Sort components by (weight * (1 - score)) to prioritize high-weight, low-score items
        sorted_components = sorted(
            components.items(),
            key=lambda x: x[1]["weight"] * (1 - x[1]["score"]),
            reverse=True
        )
        
        for component_name, component_data in sorted_components:
            if component_data["score"] < 0.8:
                if component_name == "statistical_significance":
                    priorities.append("Increase sample size and ensure statistical power")
                elif component_name == "reproducibility":
                    priorities.append("Improve experimental control and reduce variance")
                elif component_name == "methodology":
                    priorities.append("Strengthen experimental design and documentation")
        
        return priorities


class TestStatisticalSignificance:
    """Test statistical significance validation."""
    
    @pytest.fixture
    def research_framework(self):
        """Research quality framework instance."""
        return ResearchQualityFramework()
    
    def test_accelerator_performance_significance(self, research_framework):
        """Test statistical significance of accelerator performance comparisons."""
        # Register experiment
        methodology = {
            "hypothesis": "Larger compute arrays improve throughput with statistical significance",
            "control_conditions": "16x16 systolic array baseline",
            "variables_controlled": ["frequency", "precision", "memory_hierarchy"],
            "sample_size_justification": "Power analysis for detecting 20% effect size at α=0.05, β=0.2",
            "randomization": "Random order of configurations tested",
            "replication_plan": "Each configuration tested 10 times"
        }
        
        exp_id = research_framework.register_experiment(
            "accelerator_performance_scaling",
            "Statistical analysis of accelerator performance scaling with array size",
            methodology
        )
        
        designer = AcceleratorDesigner()
        
        # Test different array sizes
        array_sizes = [(16, 16), (32, 32), (64, 64)]
        
        for array_size in array_sizes:
            for run in range(10):  # Multiple runs for statistical power
                accelerator = designer.design(
                    compute_units=array_size[0] * array_size[1],
                    dataflow="weight_stationary",
                    frequency_mhz=400.0,
                    precision="int8"
                )
                
                performance = accelerator.estimate_performance()
                
                # Add noise to simulate real measurement variance
                throughput = performance["throughput_ops_s"] * (1 + np.random.normal(0, 0.05))
                latency = performance["latency_ms"] * (1 + np.random.normal(0, 0.03))
                power = performance["power_w"] * (1 + np.random.normal(0, 0.08))
                
                research_framework.add_experiment_result(
                    exp_id,
                    {
                        "array_size": array_size[0] * array_size[1],
                        "throughput_ops_s": throughput,
                        "latency_ms": latency,
                        "power_w": power,
                        "efficiency_ops_w": throughput / power
                    },
                    {"run": run, "array_dimensions": array_size}
                )
        
        # Validate statistical significance
        significance_results = research_framework.validate_statistical_significance(exp_id)
        
        # Should have sufficient data for statistical analysis
        assert significance_results["metrics_analyzed"] > 0, "No metrics analyzed"
        assert significance_results["significance_ratio"] > 0.5, f"Too few significant metrics: {significance_results['significance_ratio']}"
        
        # Check specific metrics
        detailed_results = significance_results["detailed_results"]
        
        if "throughput_ops_s" in detailed_results:
            throughput_stats = detailed_results["throughput_ops_s"]
            # Should have reasonable confidence intervals
            assert throughput_stats["margin_error"] / throughput_stats["mean"] < 0.2, "Confidence interval too wide"
            assert throughput_stats["coefficient_of_variation"] < 0.15, "Too much variance in throughput"
        
        print(f"\nStatistical Significance Results:")
        print(f"  Significant: {significance_results['significant']}")
        print(f"  Metrics analyzed: {significance_results['metrics_analyzed']}")
        print(f"  Significance ratio: {significance_results['significance_ratio']:.2f}")
    
    def test_optimization_algorithm_comparison(self, research_framework):
        """Test statistical significance of optimization algorithm comparisons."""
        methodology = {
            "hypothesis": "Advanced optimization algorithms achieve better convergence",
            "control_conditions": "Baseline gradient descent optimization",
            "variables_controlled": ["model_complexity", "target_constraints", "initial_conditions"],
            "sample_size_justification": "30 runs per algorithm for detecting medium effect sizes",
            "randomization": "Random initialization seeds for each run",
            "power_analysis": "Power = 0.8, α = 0.05, effect size = 0.5"
        }
        
        exp_id = research_framework.register_experiment(
            "optimization_algorithm_comparison",
            "Comparative analysis of optimization algorithm performance",
            methodology
        )
        
        # Mock model and accelerator for testing
        mock_model = Mock()
        mock_model.parameters = 1000000
        
        accelerator = AcceleratorDesigner().design(compute_units=64)
        optimizer = ModelOptimizer(mock_model, accelerator)
        
        # Test different optimization strategies
        strategies = ["baseline", "advanced", "multi_objective"]
        
        for strategy in strategies:
            for run in range(30):  # Sufficient sample size
                # Simulate optimization with different strategies
                if strategy == "baseline":
                    convergence_iterations = np.random.normal(15, 3)
                    final_accuracy = np.random.normal(0.85, 0.02)
                elif strategy == "advanced":
                    convergence_iterations = np.random.normal(10, 2)  # Better convergence
                    final_accuracy = np.random.normal(0.90, 0.015)  # Better accuracy
                else:  # multi_objective
                    convergence_iterations = np.random.normal(12, 2.5)
                    final_accuracy = np.random.normal(0.88, 0.02)
                
                # Ensure positive values
                convergence_iterations = max(1, convergence_iterations)
                final_accuracy = max(0.1, min(1.0, final_accuracy))
                
                research_framework.add_experiment_result(
                    exp_id,
                    {
                        "strategy": strategy,
                        "convergence_iterations": convergence_iterations,
                        "final_accuracy": final_accuracy,
                        "optimization_time": convergence_iterations * 0.5 + np.random.normal(0, 0.1)
                    },
                    {"run": run, "strategy": strategy}
                )
        
        # Validate statistical significance
        significance_results = research_framework.validate_statistical_significance(exp_id)
        
        # Should detect significant differences between strategies
        assert significance_results["significant"], "Failed to detect significant differences between algorithms"
        assert significance_results["metrics_analyzed"] >= 3, "Not enough metrics analyzed"
        
        # Specific validation for convergence metrics
        detailed_results = significance_results["detailed_results"]
        
        if "convergence_iterations" in detailed_results:
            conv_stats = detailed_results["convergence_iterations"]
            # Should have good statistical properties
            assert conv_stats["coefficient_of_variation"] < 0.3, "Convergence too variable"
            assert conv_stats["n"] >= 90, "Insufficient sample size"  # 30 per strategy * 3 strategies
        
        print(f"\nOptimization Algorithm Comparison:")
        print(f"  Statistically significant: {significance_results['significant']}")
        print(f"  Effect sizes detected: {len([r for r in detailed_results.values() if r.get('effect_size', 0) > 0.2])}")


class TestReproducibility:
    """Test reproducibility validation."""
    
    @pytest.fixture
    def research_framework(self):
        """Research quality framework instance."""
        return ResearchQualityFramework()
    
    def test_design_space_exploration_reproducibility(self, research_framework):
        """Test reproducibility of design space exploration results."""
        methodology = {
            "hypothesis": "Design space exploration produces consistent Pareto frontiers",
            "control_conditions": "Fixed random seed and design space parameters",
            "variables_controlled": ["random_seed", "exploration_strategy", "sample_size"],
            "replication_plan": "5 independent runs with different random seeds",
            "measurement_procedures": "Pareto frontier quality metrics"
        }
        
        exp_id = research_framework.register_experiment(
            "design_space_reproducibility",
            "Reproducibility analysis of design space exploration",
            methodology
        )
        
        explorer = DesignSpaceExplorer(parallel_workers=2)
        
        # Fixed design space for reproducibility
        design_space = {
            "compute_units": [16, 32, 64, 128],
            "dataflow": ["weight_stationary", "output_stationary"],
            "frequency_mhz": [200.0, 400.0, 600.0],
            "precision": ["int8", "fp16"]
        }
        
        model_profile = Mock()
        model_profile.peak_gflops = 20.0
        model_profile.bandwidth_gb_s = 40.0
        
        # Multiple independent runs
        for run in range(5):
            # Set random seed for reproducibility
            np.random.seed(42 + run)
            
            result = explorer.explore(
                model=model_profile,
                design_space=design_space,
                objectives=["latency", "power"],
                num_samples=24,  # Full factorial
                strategy="grid"  # Deterministic strategy
            )
            
            # Extract reproducibility metrics
            design_points = result.design_points
            latencies = [p.metrics["latency"] for p in design_points]
            powers = [p.metrics["power"] for p in design_points]
            
            # Pareto frontier analysis
            pareto_points = explorer._compute_pareto_frontier(design_points, ["latency", "power"])
            
            research_framework.add_experiment_result(
                exp_id,
                {
                    "num_design_points": len(design_points),
                    "num_pareto_points": len(pareto_points),
                    "avg_latency": np.mean(latencies),
                    "avg_power": np.mean(powers),
                    "latency_std": np.std(latencies),
                    "power_std": np.std(powers),
                    "pareto_ratio": len(pareto_points) / len(design_points)
                },
                {"run": run, "seed": 42 + run}
            )
        
        # Validate reproducibility
        reproducibility_results = research_framework.validate_reproducibility(exp_id, tolerance=0.05)
        
        # Should be highly reproducible with fixed seeds and deterministic strategy
        assert reproducibility_results["reproducible"], "Design space exploration not reproducible"
        assert reproducibility_results["reproducibility_score"] >= 0.9, f"Low reproducibility score: {reproducibility_results['reproducibility_score']}"
        
        # Check specific metrics
        detailed_results = reproducibility_results["detailed_results"]
        
        # Number of design points should be exactly reproducible
        if "num_design_points" in detailed_results:
            design_points_repro = detailed_results["num_design_points"]
            assert design_points_repro["reproducible"], "Design point count not reproducible"
            assert design_points_repro["relative_std_deviation_percent"] < 1.0, "Too much variance in design point count"
        
        print(f"\nDesign Space Exploration Reproducibility:")
        print(f"  Reproducible: {reproducibility_results['reproducible']}")
        print(f"  Score: {reproducibility_results['reproducibility_score']:.3f}")
        print(f"  Reproducible metrics: {reproducibility_results['reproducible_metrics']}/{reproducibility_results['metrics_analyzed']}")
    
    def test_model_optimization_reproducibility(self, research_framework):
        """Test reproducibility of model optimization results."""
        methodology = {
            "hypothesis": "Model optimization produces consistent results under controlled conditions",
            "control_conditions": "Fixed model, accelerator, and optimization parameters",
            "variables_controlled": ["random_initialization", "optimization_algorithm", "convergence_criteria"],
            "replication_plan": "10 independent optimization runs",
            "measurement_procedures": "Final optimization metrics and convergence behavior"
        }
        
        exp_id = research_framework.register_experiment(
            "model_optimization_reproducibility",
            "Reproducibility analysis of model optimization",
            methodology
        )
        
        # Fixed components for reproducibility
        mock_model = Mock()
        mock_model.parameters = 1000000
        
        accelerator = AcceleratorDesigner().design(
            compute_units=64,
            dataflow="weight_stationary",
            frequency_mhz=400.0
        )
        
        optimizer = ModelOptimizer(mock_model, accelerator)
        
        # Multiple optimization runs
        for run in range(10):
            # Set random seed for each run
            np.random.seed(123 + run)
            
            result = optimizer.co_optimize(
                target_fps=30.0,
                power_budget=5.0,
                iterations=8,
                random_seed=123 + run  # Explicit seed for reproducibility
            )
            
            research_framework.add_experiment_result(
                exp_id,
                {
                    "final_latency": result.final_metrics.get("latency", 0),
                    "final_power": result.final_metrics.get("power", 0),
                    "final_accuracy": result.final_metrics.get("accuracy", 0),
                    "optimization_time": result.optimization_time,
                    "iterations_completed": result.iterations,
                    "convergence_achieved": getattr(result, 'converged', True)
                },
                {"run": run, "seed": 123 + run}
            )
        
        # Validate reproducibility
        reproducibility_results = research_framework.validate_reproducibility(exp_id, tolerance=0.08)
        
        # Optimization should be reasonably reproducible
        assert reproducibility_results["reproducibility_score"] >= 0.7, f"Low optimization reproducibility: {reproducibility_results['reproducibility_score']}"
        
        # Key metrics should be reproducible
        detailed_results = reproducibility_results["detailed_results"]
        
        critical_metrics = ["final_latency", "final_power", "final_accuracy"]
        reproducible_critical = sum(
            1 for metric in critical_metrics
            if metric in detailed_results and detailed_results[metric]["reproducible"]
        )
        
        assert reproducible_critical >= 2, f"Too few critical metrics reproducible: {reproducible_critical}/3"
        
        print(f"\nModel Optimization Reproducibility:")
        print(f"  Overall reproducible: {reproducibility_results['reproducible']}")
        print(f"  Reproducibility score: {reproducibility_results['reproducibility_score']:.3f}")
        print(f"  Critical metrics reproducible: {reproducible_critical}/3")


class TestExperimentalMethodology:
    """Test experimental methodology validation."""
    
    @pytest.fixture
    def research_framework(self):
        """Research quality framework instance."""
        return ResearchQualityFramework()
    
    def test_comprehensive_methodology_validation(self, research_framework):
        """Test validation of comprehensive experimental methodology."""
        # Example of well-designed methodology
        comprehensive_methodology = {
            "hypothesis": "Multi-objective optimization outperforms single-objective optimization for hardware design",
            "control_conditions": "Single-objective optimization baseline (latency only)",
            "variables_controlled": ["model_complexity", "hardware_constraints", "optimization_budget"],
            "sample_size_justification": "Power analysis: n=50 per condition for detecting d=0.5 at α=0.05, β=0.2",
            "randomization": "Randomized order of optimization runs and initial parameter selection",
            "blinding": "Automated evaluation prevents experimenter bias",
            "replication_plan": "Each condition replicated 50 times across 3 independent sessions",
            "power_analysis": "Statistical power = 0.8, α = 0.05, minimum detectable effect d = 0.5",
            "confounding_control": "Hardware platform, software version, and environmental conditions controlled",
            "measurement_procedures": "Standardized performance metrics with automated data collection",
            "data_quality_checks": "Outlier detection using IQR method, missing data < 5%",
            "missing_data_handling": "Multiple imputation for missing values, sensitivity analysis performed"
        }
        
        exp_id = research_framework.register_experiment(
            "methodology_validation_comprehensive",
            "Comprehensive experimental design validation",
            comprehensive_methodology
        )
        
        # Validate methodology
        methodology_results = research_framework.validate_experimental_methodology(exp_id)
        
        # Should score highly on methodology validation
        assert methodology_results["methodology_valid"], "Comprehensive methodology not validated"
        assert methodology_results["quality_score"] >= 0.9, f"Low methodology score: {methodology_results['quality_score']:.3f}"
        
        # Check specific criteria
        detailed_validation = methodology_results["detailed_validation"]
        
        # Critical criteria should be present
        critical_criteria = ["hypothesis", "control_conditions", "sample_size_justification", "power_analysis"]
        missing_critical = [c for c in critical_criteria if not detailed_validation[c]["present"]]
        
        assert len(missing_critical) == 0, f"Missing critical criteria: {missing_critical}"
        
        # Should have minimal recommendations
        recommendations = methodology_results["recommendations"]
        assert len(recommendations) <= 2, f"Too many methodology recommendations: {len(recommendations)}"
        
        print(f"\nMethodology Validation (Comprehensive):")
        print(f"  Valid: {methodology_results['methodology_valid']}")
        print(f"  Quality score: {methodology_results['quality_score']:.3f}")
        print(f"  Points: {methodology_results['points_scored']}/{methodology_results['max_points']}")
    
    def test_incomplete_methodology_validation(self, research_framework):
        """Test validation of incomplete experimental methodology."""
        # Example of poorly designed methodology
        incomplete_methodology = {
            "hypothesis": "Some optimization is better than others",
            # Missing many critical elements
        }
        
        exp_id = research_framework.register_experiment(
            "methodology_validation_incomplete",
            "Incomplete experimental design validation",
            incomplete_methodology
        )
        
        # Validate methodology
        methodology_results = research_framework.validate_experimental_methodology(exp_id)
        
        # Should score poorly
        assert not methodology_results["methodology_valid"], "Incomplete methodology incorrectly validated"
        assert methodology_results["quality_score"] < 0.3, f"Methodology score too high: {methodology_results['quality_score']:.3f}"
        
        # Should have many recommendations
        recommendations = methodology_results["recommendations"]
        assert len(recommendations) >= 5, f"Too few recommendations for incomplete methodology: {len(recommendations)}"
        
        # Check that key missing elements are identified
        detailed_validation = methodology_results["detailed_validation"]
        missing_elements = [k for k, v in detailed_validation.items() if not v["present"]]
        
        assert "control_conditions" in missing_elements, "Missing control conditions not detected"
        assert "sample_size_justification" in missing_elements, "Missing sample size justification not detected"
        
        print(f"\nMethodology Validation (Incomplete):")
        print(f"  Valid: {methodology_results['methodology_valid']}")
        print(f"  Quality score: {methodology_results['quality_score']:.3f}")
        print(f"  Recommendations: {len(recommendations)}")


class TestPeerReviewReadiness:
    """Test peer review readiness assessment."""
    
    @pytest.fixture
    def research_framework(self):
        """Research quality framework instance."""
        return ResearchQualityFramework()
    
    def test_high_quality_research_assessment(self, research_framework):
        """Test peer review assessment for high-quality research."""
        # Create high-quality experiment
        excellent_methodology = {
            "hypothesis": "Adaptive memory hierarchy optimization reduces energy consumption by >20%",
            "control_conditions": "Fixed memory hierarchy baseline",
            "variables_controlled": ["workload_characteristics", "memory_sizes", "access_patterns"],
            "sample_size_justification": "Power analysis: n=40 per condition for detecting 20% reduction at α=0.05",
            "randomization": "Latin square design for workload-condition assignment",
            "blinding": "Double-blind evaluation with automated metrics collection",
            "replication_plan": "3 independent replications of full experiment",
            "power_analysis": "Power = 0.9, α = 0.05, effect size = 0.8",
            "confounding_control": "Temperature, voltage, and process variation controlled",
            "measurement_procedures": "Calibrated power measurement with 1% accuracy",
            "data_quality_checks": "Real-time validation, redundant measurements",
            "missing_data_handling": "Less than 2% missing data, conservative imputation"
        }
        
        exp_id = research_framework.register_experiment(
            "high_quality_energy_optimization",
            "Energy optimization through adaptive memory hierarchy",
            excellent_methodology
        )
        
        # Generate high-quality experimental data
        np.random.seed(42)  # For reproducibility
        
        conditions = ["baseline", "adaptive_opt"]
        
        for condition in conditions:
            for run in range(40):  # Good sample size
                if condition == "baseline":
                    energy_consumption = np.random.normal(100, 5)  # mJ
                    latency = np.random.normal(50, 3)  # ms
                    throughput = np.random.normal(1000, 50)  # ops/s
                else:  # adaptive_opt
                    energy_consumption = np.random.normal(75, 4)  # 25% reduction
                    latency = np.random.normal(48, 3)  # Slight improvement
                    throughput = np.random.normal(1050, 45)  # Slight improvement
                
                research_framework.add_experiment_result(
                    exp_id,
                    {
                        "condition": condition,
                        "energy_consumption_mj": max(1, energy_consumption),
                        "latency_ms": max(1, latency),
                        "throughput_ops_s": max(100, throughput),
                        "energy_efficiency": max(100, throughput) / max(1, energy_consumption)
                    },
                    {"run": run, "condition": condition}
                )
        
        # Assess peer review readiness
        readiness_results = research_framework.assess_peer_review_readiness(exp_id)
        
        # Should be ready for high-tier publication
        assert readiness_results["ready_for_peer_review"], "High-quality research not deemed ready"
        assert readiness_results["overall_score"] >= 0.85, f"Overall score too low: {readiness_results['overall_score']:.3f}"
        assert "Excellent" in readiness_results["readiness_level"], f"Readiness level not excellent: {readiness_results['readiness_level']}"
        
        # Check component scores
        component_scores = readiness_results["component_scores"]
        
        assert component_scores["statistical_significance"]["score"] >= 0.9, "Statistical significance score too low"
        assert component_scores["reproducibility"]["score"] >= 0.9, "Reproducibility score too low"
        assert component_scores["methodology"]["score"] >= 0.9, "Methodology score too low"
        
        # Should have minimal improvement priorities
        improvement_priorities = readiness_results["improvement_priorities"]
        assert len(improvement_priorities) <= 1, f"Too many improvement priorities: {improvement_priorities}"
        
        print(f"\nHigh-Quality Research Assessment:")
        print(f"  Ready for peer review: {readiness_results['ready_for_peer_review']}")
        print(f"  Overall score: {readiness_results['overall_score']:.3f}")
        print(f"  Readiness level: {readiness_results['readiness_level']}")
    
    def test_low_quality_research_assessment(self, research_framework):
        """Test peer review assessment for low-quality research."""
        # Create low-quality experiment
        poor_methodology = {
            "hypothesis": "Optimization helps performance",
            # Missing many critical elements
            "replication_plan": "Run once and see what happens"
        }
        
        exp_id = research_framework.register_experiment(
            "low_quality_experiment",
            "Poorly designed optimization study",
            poor_methodology
        )
        
        # Generate poor-quality data (small sample, high variance)
        np.random.seed(123)
        
        for run in range(3):  # Very small sample
            # High variance, unclear effects
            performance = np.random.normal(100, 30)  # 30% CV - very noisy
            power = np.random.normal(10, 5)  # 50% CV - extremely noisy
            
            research_framework.add_experiment_result(
                exp_id,
                {
                    "performance_metric": max(10, performance),
                    "power_metric": max(1, power),
                    "unclear_metric": np.random.random() * 1000
                },
                {"run": run}
            )
        
        # Assess peer review readiness
        readiness_results = research_framework.assess_peer_review_readiness(exp_id)
        
        # Should not be ready for publication
        assert not readiness_results["ready_for_peer_review"], "Low-quality research incorrectly deemed ready"
        assert readiness_results["overall_score"] < 0.5, f"Overall score too high: {readiness_results['overall_score']:.3f}"
        assert "Not Ready" in readiness_results["readiness_level"], f"Incorrect readiness level: {readiness_results['readiness_level']}"
        
        # Should have many improvement priorities
        improvement_priorities = readiness_results["improvement_priorities"]
        assert len(improvement_priorities) >= 2, f"Too few improvement priorities: {improvement_priorities}"
        
        print(f"\nLow-Quality Research Assessment:")
        print(f"  Ready for peer review: {readiness_results['ready_for_peer_review']}")
        print(f"  Overall score: {readiness_results['overall_score']:.3f}")
        print(f"  Readiness level: {readiness_results['readiness_level']}")
        print(f"  Improvement priorities: {len(improvement_priorities)}")


class TestResearchQualityReport:
    """Generate comprehensive research quality reports."""
    
    def test_generate_research_quality_report(self, tmp_path):
        """Generate comprehensive research quality validation report."""
        framework = ResearchQualityFramework()
        
        # Create a realistic research scenario
        methodology = {
            "hypothesis": "Hardware-software co-optimization achieves superior Pareto frontiers compared to sequential optimization",
            "control_conditions": "Sequential optimization (hardware then software)",
            "variables_controlled": ["model_architecture", "hardware_constraints", "optimization_budget"],
            "sample_size_justification": "Power analysis for two-sample t-test: n=30 per group, α=0.05, power=0.8, d=0.75",
            "randomization": "Block randomization by model type",
            "replication_plan": "Full experiment replicated across 3 independent hardware platforms",
            "power_analysis": "Minimum detectable effect size = 0.75, achieved power = 0.81",
            "confounding_control": "Hardware platform, compiler version, measurement environment",
            "measurement_procedures": "Automated Pareto frontier evaluation with hypervolume indicator",
            "data_quality_checks": "Outlier detection, convergence validation, measurement repeatability",
            "missing_data_handling": "Less than 5% missing data, intention-to-treat analysis"
        }
        
        exp_id = framework.register_experiment(
            "co_optimization_superiority_study",
            "Comprehensive evaluation of hardware-software co-optimization effectiveness",
            methodology
        )
        
        # Generate realistic experimental data
        np.random.seed(42)
        
        optimization_types = ["sequential", "co_optimization"]
        
        for opt_type in optimization_types:
            for run in range(30):
                if opt_type == "sequential":
                    hypervolume = np.random.normal(0.65, 0.08)  # Lower performance
                    convergence_time = np.random.normal(45, 8)
                    pareto_points = np.random.poisson(12)
                else:  # co_optimization
                    hypervolume = np.random.normal(0.82, 0.06)  # Better performance
                    convergence_time = np.random.normal(38, 6)  # Faster convergence
                    pareto_points = np.random.poisson(18)  # More solutions
                
                framework.add_experiment_result(
                    exp_id,
                    {
                        "optimization_type": opt_type,
                        "hypervolume_indicator": max(0.1, hypervolume),
                        "convergence_time_minutes": max(5, convergence_time),
                        "pareto_points_found": max(1, pareto_points),
                        "solution_quality": max(0.1, hypervolume * np.random.normal(1.0, 0.05))
                    },
                    {"run": run, "optimization_type": opt_type}
                )
        
        # Run all quality validations
        significance_results = framework.validate_statistical_significance(exp_id)
        reproducibility_results = framework.validate_reproducibility(exp_id)
        methodology_results = framework.validate_experimental_methodology(exp_id)
        readiness_results = framework.assess_peer_review_readiness(exp_id)
        
        # Generate comprehensive report
        quality_report = {
            "experiment_summary": {
                "name": framework.experiments[exp_id]["name"],
                "description": framework.experiments[exp_id]["description"],
                "total_runs": len(framework.experiments[exp_id]["results"]),
                "methodology_score": methodology_results["quality_score"]
            },
            "statistical_analysis": significance_results,
            "reproducibility_analysis": reproducibility_results,
            "methodology_validation": methodology_results,
            "peer_review_readiness": readiness_results,
            "overall_quality_assessment": {
                "research_quality_grade": self._calculate_quality_grade(readiness_results["overall_score"]),
                "publication_recommendation": self._get_publication_recommendation(readiness_results),
                "key_strengths": self._identify_strengths(significance_results, reproducibility_results, methodology_results),
                "areas_for_improvement": readiness_results["improvement_priorities"]
            }
        }
        
        # Save detailed report
        report_file = tmp_path / "research_quality_report.json"
        with open(report_file, 'w') as f:
            json.dump(quality_report, f, indent=2, default=str)
        
        # Generate executive summary
        summary_file = tmp_path / "research_quality_summary.md"
        with open(summary_file, 'w') as f:
            self._write_executive_summary(f, quality_report)
        
        # Validate report quality
        overall_score = readiness_results["overall_score"]
        
        # This should be a high-quality research study
        assert overall_score >= 0.8, f"Research quality too low: {overall_score:.3f}"
        assert readiness_results["ready_for_peer_review"], "Research not ready for peer review"
        assert significance_results["significant"], "No statistical significance detected"
        assert reproducibility_results["reproducible"], "Results not reproducible"
        
        print(f"\n📊 Research Quality Validation Complete")
        print(f"🎯 Overall Quality Score: {overall_score:.3f}")
        print(f"📈 Statistical Significance: {significance_results['significant']}")
        print(f"🔄 Reproducibility Score: {reproducibility_results['reproducibility_score']:.3f}")
        print(f"📋 Methodology Score: {methodology_results['quality_score']:.3f}")
        print(f"📝 Peer Review Ready: {readiness_results['ready_for_peer_review']}")
        print(f"📁 Reports saved to: {tmp_path}")
    
    def _calculate_quality_grade(self, score: float) -> str:
        """Calculate letter grade for research quality."""
        if score >= 0.95:
            return "A+"
        elif score >= 0.90:
            return "A"
        elif score >= 0.85:
            return "A-"
        elif score >= 0.80:
            return "B+"
        elif score >= 0.75:
            return "B"
        elif score >= 0.70:
            return "B-"
        elif score >= 0.65:
            return "C+"
        elif score >= 0.60:
            return "C"
        else:
            return "D"
    
    def _get_publication_recommendation(self, readiness_results: Dict) -> str:
        """Get publication venue recommendation."""
        score = readiness_results["overall_score"]
        
        if score >= 0.90:
            return "Ready for top-tier venues (Nature, Science, CACM)"
        elif score >= 0.85:
            return "Ready for high-impact conferences (ISCA, MICRO, ASPLOS)"
        elif score >= 0.80:
            return "Ready for specialized conferences (DATE, ICCAD, FPL)"
        elif score >= 0.70:
            return "Ready for workshops or minor revisions needed"
        else:
            return "Major revisions required before submission"
    
    def _identify_strengths(self, significance, reproducibility, methodology) -> List[str]:
        """Identify key research strengths."""
        strengths = []
        
        if significance["significant"]:
            strengths.append("Strong statistical significance with appropriate power")
        
        if reproducibility["reproducibility_score"] >= 0.9:
            strengths.append("Excellent reproducibility across multiple runs")
        
        if methodology["quality_score"] >= 0.85:
            strengths.append("Rigorous experimental methodology and design")
        
        if significance.get("metrics_analyzed", 0) >= 4:
            strengths.append("Comprehensive metrics analysis")
        
        return strengths
    
    def _write_executive_summary(self, file, report: Dict):
        """Write executive summary of research quality assessment."""
        file.write("# Research Quality Assessment - Executive Summary\n\n")
        
        summary = report["experiment_summary"]
        overall = report["overall_quality_assessment"]
        
        file.write(f"## Experiment: {summary['name']}\n\n")
        file.write(f"**Description:** {summary['description']}\n\n")
        file.write(f"**Total Experimental Runs:** {summary['total_runs']}\n\n")
        
        file.write(f"## Overall Assessment\n\n")
        file.write(f"- **Quality Grade:** {overall['research_quality_grade']}\n")
        file.write(f"- **Publication Recommendation:** {overall['publication_recommendation']}\n")
        file.write(f"- **Peer Review Ready:** {'Yes' if report['peer_review_readiness']['ready_for_peer_review'] else 'No'}\n\n")
        
        file.write(f"## Key Findings\n\n")
        file.write(f"### Statistical Significance\n")
        sig = report["statistical_analysis"]
        file.write(f"- **Significant:** {sig['significant']}\n")
        file.write(f"- **Metrics Analyzed:** {sig['metrics_analyzed']}\n")
        file.write(f"- **Significance Ratio:** {sig['significance_ratio']:.2f}\n\n")
        
        file.write(f"### Reproducibility\n")
        repro = report["reproducibility_analysis"]
        file.write(f"- **Reproducible:** {repro['reproducible']}\n")
        file.write(f"- **Reproducibility Score:** {repro['reproducibility_score']:.3f}\n")
        file.write(f"- **Reproducible Metrics:** {repro['reproducible_metrics']}/{repro['metrics_analyzed']}\n\n")
        
        file.write(f"### Methodology Quality\n")
        method = report["methodology_validation"]
        file.write(f"- **Methodology Valid:** {method['methodology_valid']}\n")
        file.write(f"- **Quality Score:** {method['quality_score']:.3f}\n")
        file.write(f"- **Points Scored:** {method['points_scored']}/{method['max_points']}\n\n")
        
        file.write(f"## Strengths\n\n")
        for strength in overall["key_strengths"]:
            file.write(f"- {strength}\n")
        
        file.write(f"\n## Areas for Improvement\n\n")
        for improvement in overall["areas_for_improvement"]:
            file.write(f"- {improvement}\n")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])