"""
Unit tests for accelerator design functionality.

This module tests the AcceleratorDesigner class and related components
for hardware accelerator design and analysis.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import tempfile
from pathlib import Path

from codesign_playground.core.accelerator import (
    AcceleratorDesigner, 
    Accelerator, 
    ModelProfile
)
from codesign_playground.utils.exceptions import ValidationError, HardwareError


class TestModelProfile:
    """Test ModelProfile dataclass."""
    
    def test_model_profile_creation(self):
        """Test ModelProfile creation with valid data."""
        profile = ModelProfile(
            peak_gflops=10.5,
            bandwidth_gb_s=25.6,
            operations={"conv2d": 1000, "dense": 500},
            parameters=1000000,
            memory_mb=64.0,
            compute_intensity=0.41,
            layer_types=["conv2d", "dense"],
            model_size_mb=64.0
        )
        
        assert profile.peak_gflops == 10.5
        assert profile.bandwidth_gb_s == 25.6
        assert profile.operations["conv2d"] == 1000
        assert profile.parameters == 1000000
        assert "conv2d" in profile.layer_types
    
    def test_model_profile_to_dict(self):
        """Test ModelProfile serialization to dictionary."""
        profile = ModelProfile(
            peak_gflops=10.5,
            bandwidth_gb_s=25.6,
            operations={"conv2d": 1000},
            parameters=1000000,
            memory_mb=64.0,
            compute_intensity=0.41,
            layer_types=["conv2d"],
            model_size_mb=64.0
        )
        
        result = profile.to_dict()
        
        assert isinstance(result, dict)
        assert result["peak_gflops"] == 10.5
        assert result["bandwidth_gb_s"] == 25.6
        assert result["operations"]["conv2d"] == 1000
        assert result["parameters"] == 1000000


class TestAccelerator:
    """Test Accelerator class."""
    
    def test_accelerator_creation(self):
        """Test Accelerator creation with default parameters."""
        accelerator = Accelerator(
            compute_units=64,
            memory_hierarchy=["sram_64kb", "dram"],
            dataflow="weight_stationary"
        )
        
        assert accelerator.compute_units == 64
        assert accelerator.memory_hierarchy == ["sram_64kb", "dram"]
        assert accelerator.dataflow == "weight_stationary"
        assert accelerator.frequency_mhz == 200.0  # default
        assert accelerator.data_width == 8  # default
    
    def test_accelerator_custom_parameters(self):
        """Test Accelerator creation with custom parameters."""
        accelerator = Accelerator(
            compute_units=128,
            memory_hierarchy=["sram_128kb", "dram"],
            dataflow="output_stationary",
            frequency_mhz=400.0,
            data_width=16,
            precision="fp16",
            power_budget_w=10.0
        )
        
        assert accelerator.compute_units == 128
        assert accelerator.frequency_mhz == 400.0
        assert accelerator.data_width == 16
        assert accelerator.precision == "fp16"
        assert accelerator.power_budget_w == 10.0
    
    def test_accelerator_performance_estimation(self):
        """Test accelerator performance estimation."""
        accelerator = Accelerator(
            compute_units=64,
            memory_hierarchy=["sram_64kb", "dram"],
            dataflow="weight_stationary",
            frequency_mhz=200.0
        )
        
        performance = accelerator.estimate_performance()
        
        assert isinstance(performance, dict)
        assert "throughput_ops_s" in performance
        assert "latency_cycles" in performance
        assert "latency_ms" in performance
        assert "power_w" in performance
        assert "efficiency_ops_w" in performance
        assert "area_mm2" in performance
        
        # Check reasonable values
        assert performance["throughput_ops_s"] > 0
        assert performance["latency_cycles"] > 0
        assert performance["power_w"] > 0
        assert performance["area_mm2"] > 0
    
    def test_accelerator_rtl_generation(self):
        """Test RTL code generation."""
        accelerator = Accelerator(
            compute_units=16,
            memory_hierarchy=["sram_32kb", "dram"],
            dataflow="weight_stationary",
            data_width=8
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.v', delete=False) as f:
            rtl_path = f.name
        
        try:
            accelerator.generate_rtl(rtl_path)
            
            # Check that RTL was generated
            assert accelerator.rtl_code is not None
            assert len(accelerator.rtl_code) > 0
            
            # Check file exists and has content
            rtl_file = Path(rtl_path)
            assert rtl_file.exists()
            
            content = rtl_file.read_text()
            assert "module accelerator" in content
            assert "compute_unit" in content
            assert f"DATA_WIDTH({accelerator.data_width})" in content
            
        finally:
            Path(rtl_path).unlink(missing_ok=True)
    
    def test_accelerator_to_dict(self):
        """Test Accelerator serialization to dictionary."""
        accelerator = Accelerator(
            compute_units=64,
            memory_hierarchy=["sram_64kb", "dram"],
            dataflow="weight_stationary",
            frequency_mhz=200.0,
            precision="int8"
        )
        
        # Generate performance model first
        accelerator.estimate_performance()
        
        result = accelerator.to_dict()
        
        assert isinstance(result, dict)
        assert result["compute_units"] == 64
        assert result["memory_hierarchy"] == ["sram_64kb", "dram"]
        assert result["dataflow"] == "weight_stationary"
        assert result["frequency_mhz"] == 200.0
        assert result["precision"] == "int8"
        assert "performance_model" in result


class TestAcceleratorDesigner:
    """Test AcceleratorDesigner class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.designer = AcceleratorDesigner()
    
    def test_designer_initialization(self):
        """Test AcceleratorDesigner initialization."""
        assert self.designer.supported_frameworks == ["pytorch", "tensorflow", "onnx"]
        assert "weight_stationary" in self.designer.dataflow_options
        assert "output_stationary" in self.designer.dataflow_options
        assert "sram_64kb" in self.designer.memory_options
        assert "dram" in self.designer.memory_options
    
    def test_profile_model_basic(self):
        """Test basic model profiling."""
        mock_model = {"type": "test", "name": "test_model"}
        input_shape = (224, 224, 3)
        
        profile = self.designer.profile_model(mock_model, input_shape)
        
        assert isinstance(profile, ModelProfile)
        assert profile.peak_gflops > 0
        assert profile.bandwidth_gb_s > 0
        assert profile.parameters > 0
        assert profile.memory_mb > 0
        assert len(profile.layer_types) > 0
        assert len(profile.operations) > 0
    
    def test_profile_model_different_shapes(self):
        """Test model profiling with different input shapes."""
        mock_model = {"type": "test"}
        
        # Test different input shapes
        shapes = [
            (32, 32, 3),
            (224, 224, 3),
            (512, 512, 3),
            (28, 28, 1)  # MNIST-like
        ]
        
        for shape in shapes:
            profile = self.designer.profile_model(mock_model, shape)
            assert isinstance(profile, ModelProfile)
            assert profile.peak_gflops > 0
            
            # Larger inputs should generally require more computation
            if shape[0] > 32:
                assert profile.peak_gflops > 0.1
    
    def test_design_basic_accelerator(self):
        """Test basic accelerator design."""
        accelerator = self.designer.design(
            compute_units=32,
            dataflow="weight_stationary"
        )
        
        assert isinstance(accelerator, Accelerator)
        assert accelerator.compute_units == 32
        assert accelerator.dataflow == "weight_stationary"
        assert accelerator.memory_hierarchy == ["sram_64kb", "dram"]  # default
        assert accelerator.performance_model is not None
    
    def test_design_custom_accelerator(self):
        """Test accelerator design with custom parameters."""
        accelerator = self.designer.design(
            compute_units=128,
            memory_hierarchy=["sram_128kb", "sram_256kb", "dram"],
            dataflow="output_stationary",
            frequency_mhz=400.0,
            precision="fp16",
            power_budget_w=15.0
        )
        
        assert accelerator.compute_units == 128
        assert accelerator.memory_hierarchy == ["sram_128kb", "sram_256kb", "dram"]
        assert accelerator.dataflow == "output_stationary"
        assert accelerator.frequency_mhz == 400.0
        assert accelerator.precision == "fp16"
        assert accelerator.power_budget_w == 15.0
    
    def test_design_invalid_dataflow(self):
        """Test accelerator design with invalid dataflow."""
        with pytest.raises(ValueError, match="Unsupported dataflow"):
            self.designer.design(
                compute_units=64,
                dataflow="invalid_dataflow"
            )
    
    def test_optimize_for_model(self):
        """Test accelerator optimization for specific model profile."""
        # Create a model profile
        profile = ModelProfile(
            peak_gflops=50.0,
            bandwidth_gb_s=100.0,
            operations={"conv2d": 10000, "dense": 1000},
            parameters=5000000,
            memory_mb=128.0,
            compute_intensity=0.5,
            layer_types=["conv2d", "dense", "activation"],
            model_size_mb=128.0
        )
        
        constraints = {
            "target_fps": 60.0,
            "power_budget": 8.0
        }
        
        accelerator = self.designer.optimize_for_model(profile, constraints)
        
        assert isinstance(accelerator, Accelerator)
        assert accelerator.compute_units > 16  # Should scale for high FPS target
        assert accelerator.power_budget_w == 8.0
        assert accelerator.dataflow == "weight_stationary"  # Conv2D optimized
    
    def test_optimize_for_small_model(self):
        """Test optimization for small model."""
        profile = ModelProfile(
            peak_gflops=1.0,
            bandwidth_gb_s=5.0,
            operations={"dense": 1000},
            parameters=100000,
            memory_mb=8.0,
            compute_intensity=0.2,
            layer_types=["dense"],
            model_size_mb=8.0
        )
        
        constraints = {
            "target_fps": 30.0,
            "power_budget": 2.0
        }
        
        accelerator = self.designer.optimize_for_model(profile, constraints)
        
        assert accelerator.memory_hierarchy == ["sram_64kb", "dram"]  # Small model
        assert accelerator.dataflow == "output_stationary"  # Dense-optimized
    
    def test_optimize_for_large_model(self):
        """Test optimization for large model."""
        profile = ModelProfile(
            peak_gflops=100.0,
            bandwidth_gb_s=200.0,
            operations={"conv2d": 50000, "dense": 5000},
            parameters=50000000,
            memory_mb=500.0,
            compute_intensity=0.5,
            layer_types=["conv2d", "dense"],
            model_size_mb=500.0
        )
        
        constraints = {
            "target_fps": 30.0,
            "power_budget": 20.0
        }
        
        accelerator = self.designer.optimize_for_model(profile, constraints)
        
        assert accelerator.memory_hierarchy == ["sram_128kb", "dram"]  # Large model
        assert accelerator.compute_units >= 16


class TestAcceleratorIntegration:
    """Integration tests for accelerator components."""
    
    def test_end_to_end_design_flow(self):
        """Test complete design flow from model to accelerator."""
        designer = AcceleratorDesigner()
        
        # Step 1: Profile model
        mock_model = {"type": "cnn", "layers": 20}
        input_shape = (224, 224, 3)
        
        profile = designer.profile_model(mock_model, input_shape)
        
        # Step 2: Design accelerator
        accelerator = designer.design(
            compute_units=64,
            dataflow="weight_stationary",
            precision="int8"
        )
        
        # Step 3: Optimize for model
        constraints = {"target_fps": 30, "power_budget": 5.0}
        optimized_accelerator = designer.optimize_for_model(profile, constraints)
        
        # Step 4: Generate RTL
        with tempfile.NamedTemporaryFile(mode='w', suffix='.v', delete=False) as f:
            rtl_path = f.name
        
        try:
            optimized_accelerator.generate_rtl(rtl_path)
            
            # Verify all steps completed successfully
            assert profile.peak_gflops > 0
            assert accelerator.compute_units == 64
            assert optimized_accelerator.compute_units > 0
            assert Path(rtl_path).exists()
            
        finally:
            Path(rtl_path).unlink(missing_ok=True)
    
    def test_performance_scaling(self):
        """Test performance scaling with different configurations."""
        designer = AcceleratorDesigner()
        
        # Test different compute unit counts
        compute_units = [16, 32, 64, 128]
        performances = []
        
        for units in compute_units:
            accelerator = designer.design(compute_units=units)
            perf = accelerator.estimate_performance()
            performances.append(perf["throughput_ops_s"])
        
        # Throughput should generally increase with more compute units
        for i in range(1, len(performances)):
            assert performances[i] >= performances[i-1] * 0.9  # Allow for some variance
    
    def test_power_area_tradeoffs(self):
        """Test power and area tradeoffs."""
        designer = AcceleratorDesigner()
        
        # Small accelerator
        small_acc = designer.design(compute_units=16, frequency_mhz=100)
        small_perf = small_acc.estimate_performance()
        
        # Large accelerator
        large_acc = designer.design(compute_units=128, frequency_mhz=400)
        large_perf = large_acc.estimate_performance()
        
        # Large accelerator should have higher throughput but also higher power/area
        assert large_perf["throughput_ops_s"] > small_perf["throughput_ops_s"]
        assert large_perf["power_w"] > small_perf["power_w"]
        assert large_perf["area_mm2"] > small_perf["area_mm2"]


@pytest.fixture
def mock_model():
    """Fixture for mock model."""
    return {
        "type": "cnn",
        "layers": 10,
        "parameters": 1000000
    }


@pytest.fixture
def sample_accelerator():
    """Fixture for sample accelerator."""
    return Accelerator(
        compute_units=32,
        memory_hierarchy=["sram_64kb", "dram"],
        dataflow="weight_stationary",
        frequency_mhz=200.0,
        precision="int8"
    )


class TestAcceleratorFixtures:
    """Test accelerator functionality with fixtures."""
    
    def test_with_mock_model(self, mock_model):
        """Test using mock model fixture."""
        designer = AcceleratorDesigner()
        profile = designer.profile_model(mock_model, (224, 224, 3))
        
        assert profile.parameters > 0
        assert len(profile.operations) > 0
    
    def test_with_sample_accelerator(self, sample_accelerator):
        """Test using sample accelerator fixture."""
        assert sample_accelerator.compute_units == 32
        assert sample_accelerator.dataflow == "weight_stationary"
        
        perf = sample_accelerator.estimate_performance()
        assert perf["throughput_ops_s"] > 0
        assert perf["power_w"] > 0