"""
Workflow management for end-to-end hardware-software co-design.

This module provides high-level workflow orchestration for the complete
design process from model import to hardware generation.
"""

from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path
import json
import time
from enum import Enum

from .accelerator import AcceleratorDesigner, Accelerator, ModelProfile
from .optimizer import ModelOptimizer, OptimizationResult
from .explorer import DesignSpaceExplorer, DesignSpaceResult


class WorkflowStage(Enum):
    """Workflow execution stages."""
    INIT = "initialization"
    MODEL_IMPORT = "model_import"
    HARDWARE_MAPPING = "hardware_mapping"
    COMPILATION = "compilation"
    SIMULATION = "simulation"
    RTL_GENERATION = "rtl_generation"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class WorkflowMetrics:
    """Performance metrics from workflow execution."""
    
    images_per_second: float = 0.0
    average_power: float = 0.0
    peak_power: float = 0.0
    tops_per_watt: float = 0.0
    latency_ms: float = 0.0
    accuracy: float = 0.0
    cycles: int = 0
    area_mm2: float = 0.0
    memory_usage_mb: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        """Convert metrics to dictionary."""
        return {
            "images_per_second": self.images_per_second,
            "average_power": self.average_power,
            "peak_power": self.peak_power,
            "tops_per_watt": self.tops_per_watt,
            "latency_ms": self.latency_ms,
            "accuracy": self.accuracy,
            "cycles": self.cycles,
            "area_mm2": self.area_mm2,
            "memory_usage_mb": self.memory_usage_mb,
        }


@dataclass
class WorkflowState:
    """Current state of workflow execution."""
    
    stage: WorkflowStage = WorkflowStage.INIT
    progress: float = 0.0
    messages: List[str] = field(default_factory=list)
    metrics: Optional[WorkflowMetrics] = None
    artifacts: Dict[str, str] = field(default_factory=dict)
    start_time: float = field(default_factory=time.time)
    
    def add_message(self, message: str) -> None:
        """Add a message to the workflow log."""
        timestamp = time.time() - self.start_time
        self.messages.append(f"[{timestamp:.2f}s] {message}")
    
    def update_progress(self, stage: WorkflowStage, progress: float) -> None:
        """Update workflow progress."""
        self.stage = stage
        self.progress = progress
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary."""
        return {
            "stage": self.stage.value,
            "progress": self.progress,
            "messages": self.messages,
            "metrics": self.metrics.to_dict() if self.metrics else None,
            "artifacts": self.artifacts,
            "elapsed_time": time.time() - self.start_time,
        }


class Workflow:
    """End-to-end workflow orchestration for hardware-software co-design."""
    
    def __init__(self, name: str, output_dir: Optional[str] = None):
        """
        Initialize workflow.
        
        Args:
            name: Workflow identifier
            output_dir: Directory for output artifacts
        """
        self.name = name
        self.output_dir = Path(output_dir or f"./workflows/{name}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.designer = AcceleratorDesigner()
        self.explorer = DesignSpaceExplorer()
        
        # State management
        self.state = WorkflowState()
        self.model = None
        self.model_profile = None
        self.accelerator = None
        self.optimizer = None
        
        self.state.add_message(f"Initialized workflow '{name}'")
    
    def import_model(
        self, 
        model_path: str, 
        input_shapes: Dict[str, tuple],
        framework: str = "auto"
    ) -> None:
        """
        Import neural network model.
        
        Args:
            model_path: Path to model file
            input_shapes: Input tensor shapes
            framework: ML framework ("pytorch", "tensorflow", "onnx", "auto")
        """
        self.state.update_progress(WorkflowStage.MODEL_IMPORT, 0.1)
        self.state.add_message(f"Importing model from {model_path}")
        
        # Mock model import - in practice would load actual model
        if framework == "auto":
            framework = self._detect_framework(model_path)
        
        # Create mock model object
        class MockModel:
            def __init__(self, path, shapes, framework):
                self.path = path
                self.input_shapes = shapes
                self.framework = framework
                self.complexity = 1.0
        
        self.model = MockModel(model_path, input_shapes, framework)
        
        # Profile the model
        self.state.update_progress(WorkflowStage.MODEL_IMPORT, 0.5)
        self.state.add_message("Profiling model computational requirements")
        
        # Use first input shape for profiling
        primary_input_shape = list(input_shapes.values())[0]
        self.model_profile = self.designer.profile_model(self.model, primary_input_shape)
        
        self.state.update_progress(WorkflowStage.MODEL_IMPORT, 1.0)
        self.state.add_message(
            f"Model imported: {self.model_profile.peak_gflops:.2f} GFLOPS, "
            f"{self.model_profile.parameters:,} parameters"
        )
        
        # Save model profile
        profile_path = self.output_dir / "model_profile.json"
        with open(profile_path, 'w') as f:
            json.dump(self.model_profile.to_dict(), f, indent=2)
        self.state.artifacts["model_profile"] = str(profile_path)
    
    def map_to_hardware(
        self,
        template: str = "systolic_array",
        size: tuple = (16, 16),
        precision: str = "int8",
        **kwargs
    ) -> None:
        """
        Map model to hardware accelerator template.
        
        Args:
            template: Hardware template ("systolic_array", "vector_processor")
            size: Hardware dimensions
            precision: Numerical precision
            **kwargs: Additional template parameters
        """
        if not self.model_profile:
            raise RuntimeError("Must import model before hardware mapping")
        
        self.state.update_progress(WorkflowStage.HARDWARE_MAPPING, 0.1)
        self.state.add_message(f"Mapping to {template} hardware template")
        
        # Configure accelerator based on template and model requirements
        if template == "systolic_array":
            compute_units = size[0] * size[1]
            dataflow = "weight_stationary"
            memory_hierarchy = ["sram_64kb", "dram"]
        elif template == "vector_processor":
            compute_units = size[0]  # Vector lanes
            dataflow = "output_stationary"
            memory_hierarchy = ["sram_32kb", "dram"]
        else:
            raise ValueError(f"Unknown hardware template: {template}")
        
        self.state.update_progress(WorkflowStage.HARDWARE_MAPPING, 0.5)
        
        # Create accelerator
        self.accelerator = self.designer.design(
            compute_units=compute_units,
            dataflow=dataflow,
            memory_hierarchy=memory_hierarchy,
            precision=precision,
            **kwargs
        )
        
        self.state.update_progress(WorkflowStage.HARDWARE_MAPPING, 1.0)
        self.state.add_message(
            f"Hardware mapped: {compute_units} compute units, {dataflow} dataflow"
        )
        
        # Save accelerator config
        config_path = self.output_dir / "accelerator_config.json"
        with open(config_path, 'w') as f:
            json.dump(self.accelerator.to_dict(), f, indent=2)
        self.state.artifacts["accelerator_config"] = str(config_path)
    
    def compile(
        self,
        optimizer: str = "tvm",
        target: str = "custom_accelerator",
        optimizations: Optional[List[str]] = None
    ) -> None:
        """
        Compile model for target accelerator.
        
        Args:
            optimizer: Compiler framework ("tvm", "mlir", "custom")
            target: Target hardware description
            optimizations: List of optimization passes
        """
        if not self.accelerator:
            raise RuntimeError("Must map to hardware before compilation")
        
        self.state.update_progress(WorkflowStage.COMPILATION, 0.1)
        self.state.add_message(f"Compiling with {optimizer} for {target}")
        
        optimizations = optimizations or ["layer_fusion", "tensorization"]
        
        # Initialize optimizer
        self.optimizer = ModelOptimizer(self.model, self.accelerator)
        
        self.state.update_progress(WorkflowStage.COMPILATION, 0.5)
        
        # Apply hardware constraints
        constraints = {
            "precision": self.accelerator.precision,
            "memory_limit_mb": 64,  # Based on memory hierarchy
            "compute_units": self.accelerator.compute_units,
        }
        
        optimized_model = self.optimizer.apply_hardware_constraints(self.model, constraints)
        self.model = optimized_model
        
        self.state.update_progress(WorkflowStage.COMPILATION, 1.0)
        self.state.add_message(f"Compilation completed with optimizations: {optimizations}")
        
        # Save compilation artifacts
        compile_info = {
            "optimizer": optimizer,
            "target": target,
            "optimizations": optimizations,
            "constraints": constraints,
        }
        compile_path = self.output_dir / "compilation_info.json"
        with open(compile_path, 'w') as f:
            json.dump(compile_info, f, indent=2)
        self.state.artifacts["compilation_info"] = str(compile_path)
    
    def simulate(
        self,
        testbench: str,
        cycles_limit: int = 1000000,
        **kwargs
    ) -> WorkflowMetrics:
        """
        Run performance simulation.
        
        Args:
            testbench: Test input dataset
            cycles_limit: Maximum simulation cycles
            **kwargs: Additional simulation parameters
            
        Returns:
            Performance metrics
        """
        if not self.model or not self.accelerator:
            raise RuntimeError("Must compile before simulation")
        
        self.state.update_progress(WorkflowStage.SIMULATION, 0.1)
        self.state.add_message(f"Starting simulation with {testbench}")
        
        # Run performance simulation
        perf = self.accelerator.estimate_performance()
        
        self.state.update_progress(WorkflowStage.SIMULATION, 0.5)
        
        # Create detailed metrics
        metrics = WorkflowMetrics(
            images_per_second=perf["throughput_ops_s"] / 1e6,  # Rough estimate
            average_power=perf["power_w"],
            peak_power=perf["power_w"] * 1.2,
            tops_per_watt=perf["efficiency_ops_w"] / 1e12,
            latency_ms=perf["latency_ms"],
            accuracy=0.95,  # Mock accuracy
            cycles=int(perf["latency_cycles"]),
            area_mm2=perf["area_mm2"],
            memory_usage_mb=self.model_profile.memory_mb,
        )
        
        self.state.metrics = metrics
        self.state.update_progress(WorkflowStage.SIMULATION, 1.0)
        self.state.add_message(
            f"Simulation completed: {metrics.images_per_second:.1f} img/s, "
            f"{metrics.average_power:.2f}W, {metrics.tops_per_watt:.3f} TOPS/W"
        )
        
        # Save metrics
        metrics_path = self.output_dir / "simulation_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics.to_dict(), f, indent=2)
        self.state.artifacts["simulation_metrics"] = str(metrics_path)
        
        return metrics
    
    def generate_rtl(
        self,
        output_dir: Optional[str] = None,
        include_testbench: bool = True,
        **options
    ) -> None:
        """
        Generate RTL code.
        
        Args:
            output_dir: RTL output directory
            include_testbench: Whether to include testbench
            **options: Additional RTL generation options
        """
        if not self.accelerator:
            raise RuntimeError("Must create accelerator before RTL generation")
        
        rtl_dir = Path(output_dir or self.output_dir / "rtl")
        rtl_dir.mkdir(parents=True, exist_ok=True)
        
        self.state.update_progress(WorkflowStage.RTL_GENERATION, 0.1)
        self.state.add_message(f"Generating RTL to {rtl_dir}")
        
        # Generate main RTL
        rtl_file = rtl_dir / f"{self.name}_accelerator.v"
        self.accelerator.generate_rtl(str(rtl_file))
        
        self.state.update_progress(WorkflowStage.RTL_GENERATION, 0.7)
        
        # Generate testbench if requested
        if include_testbench:
            testbench_file = rtl_dir / f"{self.name}_testbench.sv"
            self._generate_testbench(testbench_file)
        
        self.state.update_progress(WorkflowStage.RTL_GENERATION, 1.0)
        self.state.add_message(f"RTL generation completed")
        
        self.state.artifacts["rtl_main"] = str(rtl_file)
        if include_testbench:
            self.state.artifacts["rtl_testbench"] = str(testbench_file)
        
        # Mark workflow as completed
        self.state.update_progress(WorkflowStage.COMPLETED, 1.0)
        self.state.add_message("Workflow completed successfully")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current workflow status."""
        return self.state.to_dict()
    
    def save_state(self, filepath: Optional[str] = None) -> None:
        """Save workflow state to file."""
        filepath = filepath or self.output_dir / "workflow_state.json"
        with open(filepath, 'w') as f:
            json.dump(self.state.to_dict(), f, indent=2)
    
    def _detect_framework(self, model_path: str) -> str:
        """Auto-detect ML framework from model file."""
        path = Path(model_path)
        
        if path.suffix == '.onnx':
            return "onnx"
        elif path.suffix == '.pb':
            return "tensorflow"
        elif path.suffix in ['.pt', '.pth']:
            return "pytorch"
        elif 'tensorflow' in model_path.lower():
            return "tensorflow"
        elif 'pytorch' in model_path.lower():
            return "pytorch"
        else:
            return "onnx"  # Default fallback
    
    def _generate_testbench(self, testbench_file: Path) -> None:
        """Generate SystemVerilog testbench."""
        testbench_code = f"""
// Generated testbench for {self.name} accelerator
`timescale 1ns/1ps

module {self.name}_testbench;

    // Clock and reset
    reg clk;
    reg rst_n;
    
    // Test signals
    reg [{self.accelerator.data_width-1}:0] data_in;
    reg data_valid;
    wire [{self.accelerator.data_width-1}:0] data_out;
    wire data_ready;
    
    // Instantiate accelerator
    accelerator dut (
        .clk(clk),
        .rst_n(rst_n),
        .data_in(data_in),
        .data_valid(data_valid),
        .data_out(data_out),
        .data_ready(data_ready)
    );
    
    // Clock generation
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end
    
    // Test sequence
    initial begin
        // Initialize
        rst_n = 0;
        data_in = 0;
        data_valid = 0;
        
        // Reset release
        #100 rst_n = 1;
        
        // Test data
        #20;
        repeat (100) begin
            data_in = $random;
            data_valid = 1;
            #10;
            data_valid = 0;
            #10;
        end
        
        // Finish
        #1000;
        $display("Testbench completed");
        $finish;
    end
    
    // Monitor
    always @(posedge clk) begin
        if (data_ready) begin
            $display("Time: %0d, Input: %h, Output: %h", $time, data_in, data_out);
        end
    end
    
endmodule
"""
        
        with open(testbench_file, 'w') as f:
            f.write(testbench_code)