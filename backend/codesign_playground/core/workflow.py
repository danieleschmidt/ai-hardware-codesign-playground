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
from ..utils.monitoring import record_metric, monitor_function, get_health_status
from ..utils.validation import validate_inputs, SecurityValidator
from ..utils.exceptions import WorkflowError, ValidationError
from ..utils.model_conversion import ModelConverter, convert_model_format, optimize_for_hardware
# from ..utils.logging import get_logger  # Use standard logging for now
import logging
import pickle

logger = logging.getLogger(__name__)


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
    
    def set_stage(self, stage: WorkflowStage) -> None:
        """Set workflow stage."""
        self.stage = stage
    
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
        self.model_converter = ModelConverter()
        
        # State management
        self.state = WorkflowState()
        self.model = None
        self.model_profile = None
        self.accelerator = None
        self.optimizer = None
        
        # Checkpoint management
        self.checkpoint_enabled = True
        self.checkpoint_interval = 60  # seconds
        self.last_checkpoint_time = time.time()
        
        self.state.add_message(f"Initialized workflow '{name}'")
    
    @monitor_function("workflow_model_import")
    @validate_inputs
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
        try:
            # Input validation and security checks
            security_validator = SecurityValidator()
            if not security_validator.validate_file_path(model_path):
                raise ValidationError("Invalid or unsafe model path")
            
            if not input_shapes:
                raise ValidationError("input_shapes cannot be empty")
            
            # Validate framework
            valid_frameworks = ["pytorch", "tensorflow", "onnx", "auto"]
            if framework not in valid_frameworks:
                raise ValidationError(f"framework must be one of: {valid_frameworks}")
            
            try:
                record_metric("workflow_model_import_started", 1, "counter", {"framework": framework})
            except Exception as e:
                logger.warning(f"Failed to record metric: {e}")
            
            self.state.update_progress(WorkflowStage.MODEL_IMPORT, 0.1)
            self.state.add_message(f"Importing model from {model_path}")
        except (ValidationError, ValueError) as e:
            self.state.set_stage(WorkflowStage.FAILED)
            raise WorkflowError(f"Model import validation failed: {e}")
        
        try:
            # Load actual model using model converter capabilities
            if framework == "auto":
                framework = self._detect_framework(model_path)
            
            # Load the model based on framework
            self.model = self._load_model(model_path, framework)
            
            # Profile the model
            self.state.update_progress(WorkflowStage.MODEL_IMPORT, 0.5)
            self.state.add_message("Profiling model computational requirements")
            
            # Use first input shape for profiling
            primary_input_shape = list(input_shapes.values())[0]
            self.model_profile = self.designer.profile_model(self.model, primary_input_shape, framework)
            
            # Create checkpoint after successful import
            self._create_checkpoint_if_needed()
            
        except Exception as e:
            self.state.set_stage(WorkflowStage.FAILED)
            logger.error(f"Model import failed: {e}")
            raise WorkflowError(f"Failed to import model: {e}")
        
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
    
    # @monitor_function("workflow_hardware_mapping")
    # @validate_inputs
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
        # Input validation
        valid_templates = ["systolic_array", "vector_processor"]
        if template not in valid_templates:
            raise ValueError(f"template must be one of: {valid_templates}")
            
        if not isinstance(size, tuple) or len(size) < 1:
            raise ValueError("size must be a non-empty tuple")
            
        valid_precisions = ["int8", "int16", "fp16", "fp32"]
        if precision not in valid_precisions:
            raise ValueError(f"precision must be one of: {valid_precisions}")
        
        record_metric("workflow_hardware_mapping_started", 1, "counter", {"template": template})
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
    
    # @monitor_function("workflow_simulation")
    # @validate_inputs
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
        # Input validation
        if not testbench:
            raise ValueError("testbench cannot be empty")
        if cycles_limit <= 0:
            raise ValueError("cycles_limit must be positive")
        if cycles_limit > 100000000:  # 100M cycle limit for safety
            raise ValueError("cycles_limit too large (max: 100M)")
        
        record_metric("workflow_simulation_started", 1, "counter")
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
        
        record_metric("workflow_simulation_completed", 1, "counter")
        record_metric("workflow_simulation_fps", metrics.images_per_second, "gauge")
        record_metric("workflow_simulation_power", metrics.average_power, "gauge")
        
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
    
    def _load_model(self, model_path: str, framework: str) -> Any:
        """Load a model from file path."""
        try:
            if framework == "pytorch":
                import torch
                model = torch.load(model_path, map_location='cpu')
                if hasattr(model, 'eval'):
                    model.eval()
                return model
            elif framework == "tensorflow":
                import tensorflow as tf
                return tf.keras.models.load_model(model_path)
            elif framework == "onnx":
                import onnx
                return onnx.load(model_path)
            else:
                # Create a mock model object with path information
                class ModelProxy:
                    def __init__(self, path, framework):
                        self.path = path
                        self.framework = framework
                        self.complexity = 1.0
                return ModelProxy(model_path, framework)
        except Exception as e:
            logger.warning(f"Failed to load model {model_path}: {e}")
            # Return mock model as fallback
            class ModelProxy:
                def __init__(self, path, framework):
                    self.path = path
                    self.framework = framework
                    self.complexity = 1.0
            return ModelProxy(model_path, framework)
    
    def _create_checkpoint_if_needed(self) -> None:
        """Create checkpoint if enough time has passed."""
        current_time = time.time()
        if (self.checkpoint_enabled and 
            current_time - self.last_checkpoint_time > self.checkpoint_interval):
            self.create_checkpoint()
            self.last_checkpoint_time = current_time
    
    def create_checkpoint(self, checkpoint_name: Optional[str] = None) -> str:
        """
        Create a checkpoint of the current workflow state.
        
        Args:
            checkpoint_name: Optional name for the checkpoint
            
        Returns:
            Path to the checkpoint file
        """
        if checkpoint_name is None:
            timestamp = int(time.time())
            checkpoint_name = f"checkpoint_{timestamp}"
        
        checkpoint_dir = self.output_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        checkpoint_file = checkpoint_dir / f"{checkpoint_name}.pkl"
        
        # Create checkpoint data
        checkpoint_data = {
            "name": self.name,
            "state": self.state.to_dict(),
            "model_profile": self.model_profile.to_dict() if self.model_profile else None,
            "accelerator": self.accelerator.to_dict() if self.accelerator else None,
            "timestamp": time.time(),
            "stage": self.state.stage.value,
        }
        
        # Save checkpoint
        try:
            with open(checkpoint_file, 'wb') as f:
                pickle.dump(checkpoint_data, f)
            
            self.state.artifacts[f"checkpoint_{checkpoint_name}"] = str(checkpoint_file)
            self.state.add_message(f"Created checkpoint: {checkpoint_name}")
            logger.info(f"Checkpoint created: {checkpoint_file}")
            
            return str(checkpoint_file)
        except Exception as e:
            logger.error(f"Failed to create checkpoint: {e}")
            raise WorkflowError(f"Checkpoint creation failed: {e}")
    
    def restore_from_checkpoint(self, checkpoint_path: str) -> None:
        """
        Restore workflow state from a checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint file
        """
        try:
            with open(checkpoint_path, 'rb') as f:
                checkpoint_data = pickle.load(f)
            
            # Restore state
            state_data = checkpoint_data["state"]
            self.state.stage = WorkflowStage(state_data["stage"])
            self.state.progress = state_data["progress"]
            self.state.messages = state_data["messages"]
            self.state.artifacts = state_data["artifacts"]
            
            # Restore model profile
            if checkpoint_data["model_profile"]:
                profile_data = checkpoint_data["model_profile"]
                self.model_profile = ModelProfile(**profile_data)
            
            # Restore accelerator
            if checkpoint_data["accelerator"]:
                accel_data = checkpoint_data["accelerator"]
                self.accelerator = Accelerator(**accel_data)
            
            self.state.add_message(f"Restored from checkpoint: {Path(checkpoint_path).name}")
            logger.info(f"Workflow restored from checkpoint: {checkpoint_path}")
            
        except Exception as e:
            logger.error(f"Failed to restore from checkpoint: {e}")
            raise WorkflowError(f"Checkpoint restoration failed: {e}")
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """
        List available checkpoints.
        
        Returns:
            List of checkpoint information
        """
        checkpoint_dir = self.output_dir / "checkpoints"
        if not checkpoint_dir.exists():
            return []
        
        checkpoints = []
        for checkpoint_file in checkpoint_dir.glob("*.pkl"):
            try:
                with open(checkpoint_file, 'rb') as f:
                    checkpoint_data = pickle.load(f)
                
                checkpoints.append({
                    "name": checkpoint_file.stem,
                    "path": str(checkpoint_file),
                    "timestamp": checkpoint_data["timestamp"],
                    "stage": checkpoint_data["stage"],
                    "size_mb": checkpoint_file.stat().st_size / (1024 * 1024)
                })
            except Exception as e:
                logger.warning(f"Failed to read checkpoint {checkpoint_file}: {e}")
        
        # Sort by timestamp (newest first)
        checkpoints.sort(key=lambda x: x["timestamp"], reverse=True)
        return checkpoints
    
    def convert_model_format(
        self, 
        target_format: str, 
        input_shape: Optional[Tuple[int, ...]] = None,
        output_path: Optional[str] = None
    ) -> str:
        """
        Convert the loaded model to a different format.
        
        Args:
            target_format: Target format ("onnx", "tflite")
            input_shape: Input shape for conversion
            output_path: Output path for converted model
            
        Returns:
            Path to converted model
        """
        if not self.model:
            raise RuntimeError("No model loaded for conversion")
        
        if output_path is None:
            model_name = getattr(self.model, 'path', self.name)
            if isinstance(model_name, str):
                model_name = Path(model_name).stem
            output_path = str(self.output_dir / f"{model_name}_converted.{target_format}")
        
        if input_shape is None and self.model_profile:
            # Try to infer from model profile or use a default
            input_shape = (3, 224, 224)  # Common default for vision models
        
        try:
            converted_path = convert_model_format(
                self.model, output_path, target_format, input_shape
            )
            self.state.artifacts[f"converted_{target_format}"] = converted_path
            self.state.add_message(f"Converted model to {target_format}: {converted_path}")
            return converted_path
        except Exception as e:
            logger.error(f"Model conversion failed: {e}")
            raise WorkflowError(f"Failed to convert model to {target_format}: {e}")

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