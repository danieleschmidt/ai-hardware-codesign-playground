"""
Comprehensive unit tests for design space exploration functionality.

This module tests the DesignSpaceExplorer class and related components
for automated hardware design space exploration.
"""

import pytest
import numpy as np
import time
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any, Tuple
from concurrent.futures import ThreadPoolExecutor

from codesign_playground.core.explorer import (
    DesignSpaceExplorer, DesignPoint, ExplorationResult, 
    ParetoFrontier, DesignSpace
)
from codesign_playground.core.accelerator import Accelerator, ModelProfile
from codesign_playground.utils.exceptions import ExplorationError, ValidationError


class TestDesignPoint:
    """Test DesignPoint class functionality."""
    
    def test_design_point_creation(self):
        """Test DesignPoint creation with valid data."""
        config = {
            "compute_units": 64,
            "dataflow": "weight_stationary",
            "frequency_mhz": 400.0,
            "precision": "int8"
        }
        
        metrics = {
            "latency": 15.0,
            "power": 4.5,
            "area": 120.0,
            "throughput": 2000.0,
            "accuracy": 0.95
        }
        
        point = DesignPoint(config=config, metrics=metrics)
        
        assert point.config == config
        assert point.metrics == metrics
        assert point.config["compute_units"] == 64
        assert point.metrics["latency"] == 15.0
    
    def test_design_point_dominance(self):
        """Test design point dominance relationships."""
        # Point 1: Lower latency, higher power
        point1 = DesignPoint(
            config={"compute_units": 64},
            metrics={"latency": 10.0, "power": 6.0, "area": 100.0}
        )
        
        # Point 2: Higher latency, lower power
        point2 = DesignPoint(
            config={"compute_units": 32},
            metrics={"latency": 15.0, "power": 4.0, "area": 80.0}
        )
        
        # Point 3: Dominated by point 1 (worse in all objectives)
        point3 = DesignPoint(
            config={"compute_units": 16},
            metrics={"latency": 20.0, "power": 8.0, "area": 120.0}
        )
        
        objectives = ["latency", "power", "area"]  # All minimize
        
        # Point 1 should not dominate point 2 (trade-off)
        assert not point1.dominates(point2, objectives)
        assert not point2.dominates(point1, objectives)
        
        # Point 1 should dominate point 3
        assert point1.dominates(point3, objectives)
        assert not point3.dominates(point1, objectives)
    
    def test_design_point_distance(self):
        """Test distance calculation between design points."""
        point1 = DesignPoint(
            config={"compute_units": 64},
            metrics={"latency": 10.0, "power": 5.0}
        )
        
        point2 = DesignPoint(
            config={"compute_units": 32},
            metrics={"latency": 15.0, "power": 3.0}
        )
        
        distance = point1.distance_to(point2, ["latency", "power"])
        
        # Distance should be positive and calculated correctly
        expected_distance = np.sqrt((10.0 - 15.0)**2 + (5.0 - 3.0)**2)
        assert abs(distance - expected_distance) < 1e-6
    
    def test_design_point_serialization(self):
        """Test design point serialization and deserialization."""
        config = {"compute_units": 64, "dataflow": "weight_stationary"}
        metrics = {"latency": 15.0, "power": 4.5}
        
        point = DesignPoint(config=config, metrics=metrics)
        point_dict = point.to_dict()
        
        assert isinstance(point_dict, dict)
        assert point_dict["config"] == config
        assert point_dict["metrics"] == metrics
        
        # Test recreation from dict
        recreated_point = DesignPoint.from_dict(point_dict)
        assert recreated_point.config == point.config
        assert recreated_point.metrics == point.metrics


class TestDesignSpace:
    """Test DesignSpace class functionality."""
    
    @pytest.fixture
    def sample_design_space(self):
        """Sample design space for testing."""
        return DesignSpace({
            "compute_units": [16, 32, 64, 128, 256],
            "memory_hierarchy": [
                ["sram_32kb", "dram"],
                ["sram_64kb", "dram"],
                ["sram_128kb", "dram"],
                ["sram_64kb", "sram_256kb", "dram"]
            ],
            "dataflow": ["weight_stationary", "output_stationary", "row_stationary"],
            "frequency_mhz": [200.0, 400.0, 600.0, 800.0],
            "precision": ["int8", "fp16", "fp32"]
        })
    
    def test_design_space_creation(self, sample_design_space):
        """Test design space creation and properties."""
        assert len(sample_design_space.dimensions) == 5
        assert "compute_units" in sample_design_space.dimensions
        assert sample_design_space.size() == 5 * 4 * 3 * 4 * 3  # Total combinations
    
    def test_design_space_sampling(self, sample_design_space):
        """Test design space sampling strategies."""
        # Random sampling
        random_samples = sample_design_space.sample(10, strategy="random")
        assert len(random_samples) == 10
        
        for sample in random_samples:
            assert "compute_units" in sample
            assert sample["compute_units"] in sample_design_space.dimensions["compute_units"]
            assert sample["dataflow"] in sample_design_space.dimensions["dataflow"]
        
        # Grid sampling
        grid_samples = sample_design_space.sample(8, strategy="grid")
        assert len(grid_samples) <= 8
        
        # Latin hypercube sampling
        lhs_samples = sample_design_space.sample(12, strategy="latin_hypercube")
        assert len(lhs_samples) == 12
    
    def test_design_space_constraints(self):
        """Test design space with constraints."""
        design_space = DesignSpace({
            "compute_units": [16, 32, 64, 128],
            "frequency_mhz": [200.0, 400.0, 600.0],
            "power_budget": [2.0, 5.0, 10.0]
        })
        
        # Add constraint: power budget increases with compute units
        def power_constraint(config):
            return config["power_budget"] >= config["compute_units"] / 32 * 2.0
        
        design_space.add_constraint(power_constraint)
        
        # Sample with constraints
        constrained_samples = design_space.sample(20, strategy="random", apply_constraints=True)
        
        # All samples should satisfy the constraint
        for sample in constrained_samples:
            assert power_constraint(sample)
    
    def test_design_space_validation(self):
        """Test design space validation."""
        # Valid design space
        valid_space = DesignSpace({
            "compute_units": [16, 32, 64],
            "dataflow": ["weight_stationary", "output_stationary"]
        })
        
        assert valid_space.validate()
        
        # Invalid design space (empty dimension)
        with pytest.raises(ValidationError):
            DesignSpace({
                "compute_units": [],
                "dataflow": ["weight_stationary"]
            })


class TestDesignSpaceExplorer:
    """Test DesignSpaceExplorer class functionality."""
    
    @pytest.fixture
    def explorer(self):
        """DesignSpaceExplorer instance for testing."""
        return DesignSpaceExplorer(parallel_workers=2, cache_size=100)
    
    @pytest.fixture
    def sample_model_profile(self):
        """Sample model profile for testing."""
        return ModelProfile(
            peak_gflops=25.0,
            bandwidth_gb_s=50.0,
            operations={"conv2d": 5000, "dense": 1500},
            parameters=2500000,
            memory_mb=40.0,
            compute_intensity=0.5,
            layer_types=["conv2d", "dense", "activation"],
            model_size_mb=40.0
        )
    
    def test_explorer_initialization(self):
        """Test DesignSpaceExplorer initialization."""
        explorer = DesignSpaceExplorer(parallel_workers=4, cache_size=200)
        
        assert explorer.parallel_workers == 4
        assert explorer.cache_size == 200
        assert explorer.exploration_history == []
        assert explorer.cache_hits == 0
        assert explorer.cache_misses == 0
    
    def test_basic_exploration(self, explorer, sample_model_profile):
        """Test basic design space exploration."""
        design_space = {
            "compute_units": [32, 64, 128],
            "dataflow": ["weight_stationary", "output_stationary"],
            "frequency_mhz": [200.0, 400.0],
            "precision": ["int8", "fp16"]
        }
        
        result = explorer.explore(
            model=sample_model_profile,
            design_space=design_space,
            objectives=["latency", "power"],
            num_samples=12,
            strategy="random"
        )
        
        assert isinstance(result, ExplorationResult)
        assert len(result.design_points) == 12
        assert result.total_evaluations == 12
        assert result.exploration_time > 0
        
        # All design points should have the required metrics
        for point in result.design_points:
            assert "latency" in point.metrics
            assert "power" in point.metrics
            assert point.config["compute_units"] in [32, 64, 128]
    
    def test_pareto_frontier_exploration(self, explorer, sample_model_profile):
        """Test Pareto frontier exploration."""
        design_space = {
            "compute_units": [16, 32, 64, 128],
            "dataflow": ["weight_stationary", "output_stationary"],
            "frequency_mhz": [200.0, 400.0, 600.0],
            "precision": ["int8", "fp16"]
        }
        
        pareto_points = explorer.explore_pareto_frontier(
            design_space=design_space,
            target_profile=sample_model_profile,
            max_samples=24,
            objectives=["latency", "power", "area"]
        )
        
        assert len(pareto_points) > 0
        assert len(pareto_points) <= 24
        
        # Verify Pareto frontier properties
        objectives = ["latency", "power", "area"]
        for i, point1 in enumerate(pareto_points):
            for j, point2 in enumerate(pareto_points):
                if i != j:
                    # No point should dominate another in Pareto frontier
                    assert not point1.dominates(point2, objectives)
    
    def test_multi_objective_exploration(self, explorer, sample_model_profile):
        """Test multi-objective exploration with different strategies."""
        design_space = {
            "compute_units": [32, 64, 128],
            "dataflow": ["weight_stationary", "output_stationary"],
            "precision": ["int8", "fp16"]
        }
        
        objectives = ["latency", "power", "area", "accuracy"]
        
        # Test different strategies
        strategies = ["random", "grid", "evolutionary"]
        
        for strategy in strategies:
            result = explorer.explore(
                model=sample_model_profile,
                design_space=design_space,
                objectives=objectives,
                num_samples=8,
                strategy=strategy
            )
            
            assert len(result.design_points) == 8
            
            # All objectives should be evaluated
            for point in result.design_points:
                for objective in objectives:
                    assert objective in point.metrics
    
    def test_exploration_with_constraints(self, explorer, sample_model_profile):
        """Test exploration with design constraints."""
        design_space = {
            "compute_units": [32, 64, 128, 256],
            "frequency_mhz": [200.0, 400.0, 600.0],
            "power_budget": [3.0, 6.0, 12.0]
        }
        
        constraints = {
            "max_power": 8.0,
            "max_area": 150.0,
            "min_throughput": 1000.0
        }
        
        result = explorer.explore(
            model=sample_model_profile,
            design_space=design_space,
            objectives=["latency", "power"],
            constraints=constraints,
            num_samples=15,
            strategy="random"
        )
        
        # All design points should satisfy constraints
        for point in result.design_points:
            assert point.metrics["power"] <= constraints["max_power"]
            assert point.metrics.get("area", 0) <= constraints["max_area"]
            assert point.metrics.get("throughput", float('inf')) >= constraints["min_throughput"]
    
    def test_parallel_exploration(self, explorer, sample_model_profile):
        """Test parallel exploration performance."""
        design_space = {
            "compute_units": [16, 32, 64, 128, 256],
            "dataflow": ["weight_stationary", "output_stationary", "row_stationary"],
            "frequency_mhz": [200.0, 400.0, 600.0, 800.0]
        }
        
        # Sequential exploration
        explorer_sequential = DesignSpaceExplorer(parallel_workers=1)
        start_time = time.time()
        result_sequential = explorer_sequential.explore(
            model=sample_model_profile,
            design_space=design_space,
            objectives=["latency", "power"],
            num_samples=12,
            strategy="random"
        )
        sequential_time = time.time() - start_time
        
        # Parallel exploration
        start_time = time.time()
        result_parallel = explorer.explore(
            model=sample_model_profile,
            design_space=design_space,
            objectives=["latency", "power"],
            num_samples=12,
            strategy="random"
        )
        parallel_time = time.time() - start_time
        
        # Both should produce same number of results
        assert len(result_sequential.design_points) == len(result_parallel.design_points)
        
        # Parallel should be faster (though with mocked evaluations, difference may be minimal)
        assert parallel_time <= sequential_time * 1.5  # Allow some overhead
    
    def test_exploration_caching(self, explorer, sample_model_profile):
        """Test exploration result caching."""
        design_space = {
            "compute_units": [32, 64],
            "dataflow": ["weight_stationary"],
            "frequency_mhz": [400.0]
        }
        
        # First exploration
        result1 = explorer.explore(
            model=sample_model_profile,
            design_space=design_space,
            objectives=["latency", "power"],
            num_samples=2,
            strategy="grid"  # Deterministic sampling
        )
        
        initial_cache_misses = explorer.cache_misses
        
        # Second exploration with same parameters
        result2 = explorer.explore(
            model=sample_model_profile,
            design_space=design_space,
            objectives=["latency", "power"],
            num_samples=2,
            strategy="grid"
        )
        
        # Should have cache hits
        assert explorer.cache_hits > 0
        assert explorer.cache_misses == initial_cache_misses  # No new misses
        
        # Results should be identical
        assert len(result1.design_points) == len(result2.design_points)
    
    def test_evolutionary_exploration(self, explorer, sample_model_profile):
        """Test evolutionary exploration strategy."""
        design_space = {
            "compute_units": [16, 32, 64, 128, 256],
            "dataflow": ["weight_stationary", "output_stationary"],
            "frequency_mhz": [200.0, 400.0, 600.0, 800.0],
            "precision": ["int8", "fp16"]
        }
        
        result = explorer.explore(
            model=sample_model_profile,
            design_space=design_space,
            objectives=["latency", "power", "area"],
            num_samples=20,
            strategy="evolutionary",
            evolutionary_config={
                "population_size": 8,
                "generations": 3,
                "mutation_rate": 0.1,
                "crossover_rate": 0.8
            }
        )
        
        assert len(result.design_points) == 20
        assert hasattr(result, 'evolutionary_history')
        assert len(result.evolutionary_history) == 3  # 3 generations
    
    def test_adaptive_exploration(self, explorer, sample_model_profile):
        """Test adaptive exploration strategy."""
        design_space = {
            "compute_units": [32, 64, 128, 256],
            "dataflow": ["weight_stationary", "output_stationary"],
            "frequency_mhz": [200.0, 400.0, 600.0]
        }
        
        result = explorer.explore(
            model=sample_model_profile,
            design_space=design_space,
            objectives=["latency", "power"],
            num_samples=15,
            strategy="adaptive",
            adaptive_config={
                "initial_samples": 5,
                "refinement_ratio": 0.3,
                "exploration_ratio": 0.7
            }
        )
        
        assert len(result.design_points) == 15
        assert hasattr(result, 'adaptive_phases')
        assert len(result.adaptive_phases) > 1  # Multiple phases
    
    def test_exploration_sensitivity_analysis(self, explorer, sample_model_profile):
        """Test sensitivity analysis during exploration."""
        design_space = {
            "compute_units": [32, 64, 128],
            "frequency_mhz": [200.0, 400.0, 600.0],
            "memory_size": [32, 64, 128]  # MB
        }
        
        sensitivity_result = explorer.sensitivity_analysis(
            model=sample_model_profile,
            design_space=design_space,
            objectives=["latency", "power"],
            base_config={"compute_units": 64, "frequency_mhz": 400.0, "memory_size": 64},
            perturbation_ratio=0.2
        )
        
        assert "sensitivity_scores" in sensitivity_result
        assert len(sensitivity_result["sensitivity_scores"]) == len(design_space)
        
        # All design parameters should have sensitivity scores
        for param in design_space.keys():
            assert param in sensitivity_result["sensitivity_scores"]
            assert isinstance(sensitivity_result["sensitivity_scores"][param], float)
    
    def test_exploration_convergence(self, explorer, sample_model_profile):
        """Test exploration convergence detection."""
        design_space = {
            "compute_units": [32, 64, 128],
            "dataflow": ["weight_stationary", "output_stationary"]
        }
        
        result = explorer.explore(
            model=sample_model_profile,
            design_space=design_space,
            objectives=["latency", "power"],
            num_samples=20,
            strategy="adaptive",
            convergence_config={
                "patience": 5,
                "improvement_threshold": 0.01,
                "enable_early_stopping": True
            }
        )
        
        assert hasattr(result, 'converged')
        assert hasattr(result, 'convergence_iteration')
        
        if result.converged:
            assert result.convergence_iteration < 20
    
    def test_exploration_with_model_variations(self, explorer):
        """Test exploration with different model profiles."""
        models = [
            ModelProfile(
                peak_gflops=10.0, bandwidth_gb_s=20.0, operations={"conv2d": 2000},
                parameters=500000, memory_mb=16.0, compute_intensity=0.5,
                layer_types=["conv2d"], model_size_mb=16.0
            ),
            ModelProfile(
                peak_gflops=50.0, bandwidth_gb_s=100.0, operations={"dense": 5000},
                parameters=5000000, memory_mb=80.0, compute_intensity=0.5,
                layer_types=["dense"], model_size_mb=80.0
            )
        ]
        
        design_space = {
            "compute_units": [32, 64, 128],
            "dataflow": ["weight_stationary", "output_stationary"]
        }
        
        results = []
        for model in models:
            result = explorer.explore(
                model=model,
                design_space=design_space,
                objectives=["latency", "power"],
                num_samples=6,
                strategy="random"
            )
            results.append(result)
        
        # Different models should produce different optimal designs
        assert len(results) == 2
        
        # Extract best designs for each model
        best_designs = []
        for result in results:
            best_design = min(result.design_points, key=lambda p: p.metrics["latency"])
            best_designs.append(best_design)
        
        # Best designs may be different for different models
        assert len(best_designs) == 2


class TestExplorationResult:
    """Test ExplorationResult class functionality."""
    
    def test_exploration_result_creation(self):
        """Test ExplorationResult creation."""
        design_points = [
            DesignPoint(
                config={"compute_units": 32},
                metrics={"latency": 20.0, "power": 3.0}
            ),
            DesignPoint(
                config={"compute_units": 64},
                metrics={"latency": 15.0, "power": 5.0}
            )
        ]
        
        result = ExplorationResult(
            design_points=design_points,
            total_evaluations=2,
            exploration_time=5.0,
            strategy="random",
            objectives=["latency", "power"]
        )
        
        assert len(result.design_points) == 2
        assert result.total_evaluations == 2
        assert result.exploration_time == 5.0
        assert result.strategy == "random"
        assert result.objectives == ["latency", "power"]
    
    def test_exploration_result_analysis(self):
        """Test exploration result analysis methods."""
        design_points = [
            DesignPoint(
                config={"compute_units": 32, "frequency": 200},
                metrics={"latency": 20.0, "power": 3.0, "area": 80.0}
            ),
            DesignPoint(
                config={"compute_units": 64, "frequency": 400},
                metrics={"latency": 15.0, "power": 5.0, "area": 120.0}
            ),
            DesignPoint(
                config={"compute_units": 128, "frequency": 600},
                metrics={"latency": 10.0, "power": 8.0, "area": 200.0}
            )
        ]
        
        result = ExplorationResult(
            design_points=design_points,
            total_evaluations=3,
            exploration_time=10.0,
            strategy="random",
            objectives=["latency", "power", "area"]
        )
        
        # Test best design extraction
        best_latency = result.get_best_design("latency")
        assert best_latency.metrics["latency"] == 10.0
        
        best_power = result.get_best_design("power")
        assert best_power.metrics["power"] == 3.0
        
        # Test Pareto frontier extraction
        pareto_frontier = result.get_pareto_frontier(["latency", "power"])
        assert len(pareto_frontier) > 0
        
        # Test statistics
        stats = result.get_statistics()
        assert "latency" in stats
        assert "power" in stats
        assert "area" in stats
        
        for objective in ["latency", "power", "area"]:
            assert "mean" in stats[objective]
            assert "std" in stats[objective]
            assert "min" in stats[objective]
            assert "max" in stats[objective]
    
    def test_exploration_result_serialization(self):
        """Test exploration result serialization."""
        design_points = [
            DesignPoint(
                config={"compute_units": 32},
                metrics={"latency": 20.0, "power": 3.0}
            )
        ]
        
        result = ExplorationResult(
            design_points=design_points,
            total_evaluations=1,
            exploration_time=2.0,
            strategy="random",
            objectives=["latency", "power"]
        )
        
        result_dict = result.to_dict()
        
        assert isinstance(result_dict, dict)
        assert "design_points" in result_dict
        assert "total_evaluations" in result_dict
        assert "exploration_time" in result_dict
        assert len(result_dict["design_points"]) == 1


class TestExplorationIntegration:
    """Integration tests for exploration components."""
    
    def test_end_to_end_exploration_pipeline(self):
        """Test complete exploration pipeline."""
        # Define model profile
        model_profile = ModelProfile(
            peak_gflops=30.0,
            bandwidth_gb_s=60.0,
            operations={"conv2d": 6000, "dense": 2000},
            parameters=3000000,
            memory_mb=50.0,
            compute_intensity=0.5,
            layer_types=["conv2d", "dense"],
            model_size_mb=50.0
        )
        
        # Define design space
        design_space = {
            "compute_units": [32, 64, 128, 256],
            "memory_hierarchy": [
                ["sram_64kb", "dram"],
                ["sram_128kb", "dram"],
                ["sram_256kb", "dram"]
            ],
            "dataflow": ["weight_stationary", "output_stationary"],
            "frequency_mhz": [200.0, 400.0, 600.0],
            "precision": ["int8", "fp16"]
        }
        
        # Create explorer
        explorer = DesignSpaceExplorer(parallel_workers=2)
        
        # Phase 1: Initial exploration
        initial_result = explorer.explore(
            model=model_profile,
            design_space=design_space,
            objectives=["latency", "power", "area"],
            num_samples=20,
            strategy="random"
        )
        
        # Phase 2: Pareto frontier refinement
        pareto_points = explorer.explore_pareto_frontier(
            design_space=design_space,
            target_profile=model_profile,
            max_samples=15,
            objectives=["latency", "power", "area"]
        )
        
        # Phase 3: Sensitivity analysis on best design
        best_design = min(initial_result.design_points, key=lambda p: p.metrics["latency"])
        
        sensitivity_result = explorer.sensitivity_analysis(
            model=model_profile,
            design_space=design_space,
            objectives=["latency", "power"],
            base_config=best_design.config,
            perturbation_ratio=0.1
        )
        
        # Verify pipeline results
        assert len(initial_result.design_points) == 20
        assert len(pareto_points) > 0
        assert len(pareto_points) <= 15
        assert "sensitivity_scores" in sensitivity_result
        
        # Verify exploration quality
        all_points = initial_result.design_points + pareto_points
        latencies = [p.metrics["latency"] for p in all_points]
        powers = [p.metrics["power"] for p in all_points]
        
        # Should have diversity in results
        assert max(latencies) > min(latencies)
        assert max(powers) > min(powers)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])