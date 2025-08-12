"""
Integration tests for core AI Hardware Co-Design Playground functionality.

This module tests the integration between different components of the platform,
ensuring they work together correctly in realistic scenarios.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from codesign_playground.core.accelerator import AcceleratorDesigner, ModelProfile
from codesign_playground.core.optimizer import ModelOptimizer
from codesign_playground.core.explorer import DesignSpaceExplorer


class TestCoreIntegration:
    """Integration tests for core platform components."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.designer = AcceleratorDesigner()
        # Create mock model and accelerator for optimizer
        mock_model = Mock()
        mock_accelerator = self.designer.design(compute_units=32)
        self.optimizer = ModelOptimizer(mock_model, mock_accelerator)
        self.explorer = DesignSpaceExplorer()
    
    def test_model_to_accelerator_pipeline(self):
        """Test complete pipeline from model profiling to accelerator design."""
        # Step 1: Profile a mock model
        mock_model = {"name": "test_cnn", "framework": "pytorch"}
        input_shape = (224, 224, 3)
        
        profile = self.designer.profile_model(mock_model, input_shape)
        
        # Verify profile was created
        assert isinstance(profile, ModelProfile)
        assert profile.peak_gflops > 0
        assert profile.parameters > 0
        assert "conv2d" in profile.layer_types
        
        # Step 2: Design an accelerator for this model
        constraints = {
            "target_fps": 30.0,
            "power_budget": 5.0,
            "area_budget": 100.0
        }
        
        accelerator = self.designer.optimize_for_model(profile, constraints)
        
        # Verify accelerator was designed
        assert accelerator.compute_units > 0
        assert accelerator.dataflow in ["weight_stationary", "output_stationary", "row_stationary"]
        assert accelerator.power_budget_w == constraints["power_budget"]
        
        # Step 3: Generate RTL for the accelerator
        with tempfile.NamedTemporaryFile(mode='w', suffix='.v', delete=False) as f:
            rtl_path = f.name
        
        try:
            accelerator.generate_rtl(rtl_path)
            
            # Verify RTL was generated
            rtl_file = Path(rtl_path)
            assert rtl_file.exists()
            
            content = rtl_file.read_text()
            assert "module accelerator" in content
            assert f"compute_unit" in content
            
        finally:
            Path(rtl_path).unlink(missing_ok=True)
    
    def test_design_space_exploration_integration(self):
        """Test integration of design space exploration with accelerator design."""
        # Define design space
        design_space = {
            "compute_units": [16, 32, 64, 128],
            "memory_hierarchy": [
                ["sram_32kb", "dram"],
                ["sram_64kb", "dram"],
                ["sram_128kb", "dram"]
            ],
            "dataflow": ["weight_stationary", "output_stationary"],
            "frequency_mhz": [200.0, 400.0, 600.0]
        }
        
        # Create model profile for exploration target
        target_profile = ModelProfile(
            peak_gflops=25.0,
            bandwidth_gb_s=50.0,
            operations={"conv2d": 5000, "dense": 1000},
            parameters=2000000,
            memory_mb=32.0,
            compute_intensity=0.5,
            layer_types=["conv2d", "dense"],
            model_size_mb=32.0
        )
        
        # Explore design space
        pareto_designs = self.explorer.explore_pareto_frontier(
            design_space=design_space,
            target_profile=target_profile,
            max_samples=12,  # Reduced for faster testing
            objectives=["performance", "power", "area"]
        )
        
        # Verify exploration results
        assert len(pareto_designs) > 0
        assert len(pareto_designs) <= 12
        
        # All designs should be valid accelerators
        for design_point in pareto_designs:
            assert hasattr(design_point, 'accelerator')
            assert hasattr(design_point, 'metrics')
            assert design_point.accelerator.compute_units > 0
            assert "performance" in design_point.metrics
            assert "power" in design_point.metrics
            assert "area" in design_point.metrics
    
    def test_model_optimization_integration(self):
        """Test integration of model optimization with accelerator design."""
        # Create mock model for optimization
        mock_model = Mock()
        mock_model.name = "test_model"
        mock_model.parameters = 1000000
        
        # Define target accelerator
        target_accelerator = self.designer.design(
            compute_units=64,
            dataflow="weight_stationary",
            precision="int8"
        )
        
        # Optimize model for target accelerator
        optimization_config = {
            "target_precision": "int8",
            "optimization_level": "moderate",
            "preserve_accuracy": True
        }
        
        optimized_model = self.optimizer.optimize_for_accelerator(
            model=mock_model,
            accelerator=target_accelerator,
            config=optimization_config
        )
        
        # Verify optimization results
        assert optimized_model is not None
        assert hasattr(optimized_model, 'precision')
        assert optimized_model.precision == "int8"
        
        # The optimized model should be compatible with the target accelerator
        compatibility_score = self.optimizer.check_compatibility(
            optimized_model, target_accelerator
        )
        assert compatibility_score > 0.5  # Should be reasonably compatible
    
    def test_multi_objective_optimization_integration(self):
        """Test integration of multi-objective optimization across components."""
        # Create model profile
        model_profile = ModelProfile(
            peak_gflops=15.0,
            bandwidth_gb_s=30.0,
            operations={"conv2d": 3000, "dense": 800},
            parameters=1500000,
            memory_mb=24.0,
            compute_intensity=0.4,
            layer_types=["conv2d", "dense"],
            model_size_mb=24.0
        )
        
        # Define constraints
        constraints = {
            "power_budget": 10.0,
            "area_budget": 150.0,
            "latency_ms": 25.0,
            "accuracy_threshold": 0.95
        }
        
        # Run multi-objective optimization
        optimization_results = self.optimizer.multi_objective_optimize(
            model_profile=model_profile,
            constraints=constraints,
            objectives=["latency", "power", "accuracy"],
            num_iterations=5  # Reduced for testing
        )
        
        # Verify optimization results
        assert isinstance(optimization_results, dict)
        assert "pareto_solutions" in optimization_results
        assert "best_compromise" in optimization_results
        assert "optimization_history" in optimization_results
        
        pareto_solutions = optimization_results["pareto_solutions"]
        assert len(pareto_solutions) > 0
        
        # Each solution should have model and accelerator
        for solution in pareto_solutions:
            assert "model_config" in solution
            assert "accelerator_config" in solution
            assert "metrics" in solution
            assert "latency" in solution["metrics"]
            assert "power" in solution["metrics"]
    
    def test_parallel_design_exploration(self):
        """Test parallel execution of design exploration."""
        # Create design configurations for parallel execution
        design_configs = [
            {"compute_units": 32, "dataflow": "weight_stationary"},
            {"compute_units": 64, "dataflow": "weight_stationary"},
            {"compute_units": 128, "dataflow": "output_stationary"},
            {"compute_units": 64, "dataflow": "row_stationary"},
        ]
        
        # Execute designs in parallel
        accelerators = self.designer.design_parallel(design_configs, max_workers=2)
        
        # Verify all designs completed
        assert len(accelerators) == len(design_configs)
        
        # Each accelerator should match its configuration
        for i, accelerator in enumerate(accelerators):
            expected_config = design_configs[i]
            assert accelerator.compute_units == expected_config["compute_units"]
            assert accelerator.dataflow == expected_config["dataflow"]
            assert accelerator.performance_model is not None
    
    def test_caching_integration(self):
        """Test that caching works correctly across components."""
        # Profile the same model multiple times
        mock_model = {"name": "cache_test", "id": "12345"}
        input_shape = (224, 224, 3)
        
        # First profiling
        profile1 = self.designer.profile_model(mock_model, input_shape)
        
        # Second profiling (should use cache)
        profile2 = self.designer.profile_model(mock_model, input_shape)
        
        # Profiles should be identical (same cached result)
        assert profile1.peak_gflops == profile2.peak_gflops
        assert profile1.parameters == profile2.parameters
        assert profile1.operations == profile2.operations
        
        # Design the same accelerator multiple times
        design_params = {
            "compute_units": 64,
            "dataflow": "weight_stationary",
            "precision": "int8"
        }
        
        # First design
        accelerator1 = self.designer.design(**design_params)
        
        # Second design (should use cache)
        accelerator2 = self.designer.design(**design_params)
        
        # Accelerators should be identical
        assert accelerator1.compute_units == accelerator2.compute_units
        assert accelerator1.dataflow == accelerator2.dataflow
        assert accelerator1.precision == accelerator2.precision
    
    def test_error_handling_integration(self):
        """Test error handling across integrated components."""
        # Test invalid model profiling
        invalid_model = None
        
        with pytest.raises((TypeError, ValueError)):
            self.designer.profile_model(invalid_model, (224, 224, 3))
        
        # Test invalid accelerator design
        with pytest.raises(ValueError):
            self.designer.design(
                compute_units=-1,  # Invalid parameter
                dataflow="invalid_dataflow"
            )
        
        # Test invalid optimization config
        mock_model = Mock()
        valid_accelerator = self.designer.design(compute_units=32)
        
        invalid_config = {
            "target_precision": "invalid_precision",
            "optimization_level": "invalid_level"
        }
        
        with pytest.raises((ValueError, KeyError)):
            self.optimizer.optimize_for_accelerator(
                mock_model, valid_accelerator, invalid_config
            )


@pytest.fixture
def integration_test_data():
    """Fixture providing test data for integration tests."""
    return {
        "models": [
            {"name": "resnet18", "framework": "pytorch", "size_mb": 45.0},
            {"name": "mobilenet_v2", "framework": "tensorflow", "size_mb": 14.0},
            {"name": "efficientnet_b0", "framework": "onnx", "size_mb": 20.0}
        ],
        "input_shapes": [
            (224, 224, 3),  # Standard ImageNet
            (32, 32, 3),    # CIFAR-10
            (28, 28, 1)     # MNIST
        ],
        "constraint_sets": [
            {"power_budget": 2.0, "latency_ms": 10.0},  # Mobile
            {"power_budget": 10.0, "latency_ms": 5.0},   # Edge
            {"power_budget": 50.0, "latency_ms": 1.0}    # Server
        ]
    }


class TestIntegrationWithFixtures:
    """Integration tests using shared fixtures."""
    
    def test_multiple_models_integration(self, integration_test_data):
        """Test integration with multiple different models."""
        designer = AcceleratorDesigner()
        models = integration_test_data["models"]
        input_shape = integration_test_data["input_shapes"][0]
        
        profiles = []
        accelerators = []
        
        # Profile all models
        for model_info in models:
            profile = designer.profile_model(model_info, input_shape)
            profiles.append(profile)
            
            # Design accelerator for each model
            constraints = {"target_fps": 30.0, "power_budget": 5.0}
            accelerator = designer.optimize_for_model(profile, constraints)
            accelerators.append(accelerator)
        
        # Verify all models were processed
        assert len(profiles) == len(models)
        assert len(accelerators) == len(models)
        
        # Each model should have different characteristics
        gflops_values = [p.peak_gflops for p in profiles]
        assert len(set(gflops_values)) > 1  # Should have different GFLOPS
    
    def test_constraint_variation_integration(self, integration_test_data):
        """Test integration with varying constraint sets."""
        designer = AcceleratorDesigner()
        explorer = DesignSpaceExplorer()
        
        model_info = integration_test_data["models"][0]  # Use first model
        input_shape = integration_test_data["input_shapes"][0]
        
        # Profile the model
        profile = designer.profile_model(model_info, input_shape)
        
        accelerators = []
        
        # Design accelerators for each constraint set
        for constraints in integration_test_data["constraint_sets"]:
            accelerator = designer.optimize_for_model(profile, constraints)
            accelerators.append(accelerator)
        
        # Verify different constraints produce different designs
        compute_units = [acc.compute_units for acc in accelerators]
        power_budgets = [acc.power_budget_w for acc in accelerators]
        
        # Should have variation in designs
        assert len(set(compute_units)) > 1 or len(set(power_budgets)) > 1


class TestSystemIntegration:
    """System-level integration tests."""
    
    def test_full_system_workflow(self):
        """Test complete system workflow from model to deployment."""
        # Initialize all components
        designer = AcceleratorDesigner()
        optimizer = ModelOptimizer()
        explorer = DesignSpaceExplorer()
        
        # Step 1: Model profiling
        model_info = {"name": "test_system", "framework": "pytorch"}
        profile = designer.profile_model(model_info, (224, 224, 3))
        
        # Step 2: Initial accelerator design
        initial_accelerator = designer.design(compute_units=64)
        
        # Step 3: Model optimization
        optimization_config = {"target_precision": "int8"}
        optimized_model = optimizer.optimize_for_accelerator(
            model_info, initial_accelerator, optimization_config
        )
        
        # Step 4: Design space exploration
        design_space = {
            "compute_units": [32, 64, 128],
            "dataflow": ["weight_stationary", "output_stationary"]
        }
        
        pareto_designs = explorer.explore_pareto_frontier(
            design_space=design_space,
            target_profile=profile,
            max_samples=6,
            objectives=["performance", "power"]
        )
        
        # Step 5: Select best design
        best_design = min(pareto_designs, 
                         key=lambda d: d.metrics["power"] + d.metrics.get("latency", 0))
        
        # Verify complete workflow
        assert profile.peak_gflops > 0
        assert optimized_model.precision == "int8"
        assert len(pareto_designs) > 0
        assert best_design.accelerator.compute_units > 0
        
        # System should be ready for deployment
        assert best_design.accelerator.performance_model is not None
        assert best_design.metrics["performance"] > 0