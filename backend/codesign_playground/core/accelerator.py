"""
Accelerator design and profiling functionality.

This module provides the core AcceleratorDesigner class for analyzing neural network
models and generating matching hardware accelerator architectures.
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import json
import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
# Optional dependency with fallback
try:
    import numpy as np
except ImportError:
    np = None
from .cache import cached, get_thread_pool
from ..utils.monitoring import record_metric
from ..utils.logging import get_logger
from ..utils.exceptions import HardwareError, ValidationError
from ..utils.validation import validate_inputs, SecurityValidator

logger = get_logger(__name__)


@dataclass
class ModelProfile:
    """Profile data extracted from a neural network model."""
    
    peak_gflops: float
    bandwidth_gb_s: float
    operations: Dict[str, int]
    parameters: int
    memory_mb: float
    compute_intensity: float
    layer_types: List[str]
    model_size_mb: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert profile to dictionary representation."""
        return {
            "peak_gflops": self.peak_gflops,
            "bandwidth_gb_s": self.bandwidth_gb_s,
            "operations": self.operations,
            "parameters": self.parameters,
            "memory_mb": self.memory_mb,
            "compute_intensity": self.compute_intensity,
            "layer_types": self.layer_types,
            "model_size_mb": self.model_size_mb,
        }


@dataclass
class Accelerator:
    """Hardware accelerator specification and configuration."""
    
    compute_units: int
    memory_hierarchy: List[str]
    dataflow: str
    frequency_mhz: float = 200.0
    data_width: int = 8
    precision: str = "int8"
    power_budget_w: float = 5.0
    area_budget_mm2: float = 10.0
    
    # Generated artifacts
    rtl_code: Optional[str] = None
    performance_model: Optional[Dict[str, float]] = None
    resource_estimates: Optional[Dict[str, int]] = None
    
    def generate_rtl(self, output_path: str) -> None:
        """Generate RTL code for the accelerator."""
        rtl = self._generate_verilog_code()
        self.rtl_code = rtl
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(rtl)
    
    def _generate_verilog_code(self) -> str:
        """Generate basic Verilog code template."""
        return f"""
// Generated Accelerator RTL
// Configuration: {self.compute_units} compute units, {self.dataflow} dataflow
module accelerator (
    input wire clk,
    input wire rst_n,
    input wire [{self.data_width-1}:0] data_in,
    input wire data_valid,
    output wire [{self.data_width-1}:0] data_out,
    output wire data_ready
);

    // Compute units instantiation
    genvar i;
    generate
        for (i = 0; i < {self.compute_units}; i = i + 1) begin : compute_array
            compute_unit #(
                .DATA_WIDTH({self.data_width}),
                .DATAFLOW("{self.dataflow}")
            ) cu_inst (
                .clk(clk),
                .rst_n(rst_n),
                .data_in(data_in),
                .data_valid(data_valid),
                .data_out(data_out),
                .data_ready(data_ready)
            );
        end
    endgenerate

endmodule

// Basic compute unit
module compute_unit #(
    parameter DATA_WIDTH = 8,
    parameter DATAFLOW = "weight_stationary"
) (
    input wire clk,
    input wire rst_n,
    input wire [DATA_WIDTH-1:0] data_in,
    input wire data_valid,
    output reg [DATA_WIDTH-1:0] data_out,
    output reg data_ready
);

    // Simple MAC operation
    reg [DATA_WIDTH*2-1:0] accumulator;
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            accumulator <= 0;
            data_out <= 0;
            data_ready <= 0;
        end else if (data_valid) begin
            accumulator <= accumulator + (data_in * data_in);
            data_out <= accumulator[DATA_WIDTH-1:0];
            data_ready <= 1;
        end else begin
            data_ready <= 0;
        end
    end

endmodule
"""
    
    def estimate_performance(self) -> Dict[str, float]:
        """Estimate accelerator performance metrics."""
        # Simple performance model with error handling
        throughput_ops_s = self.compute_units * self.frequency_mhz * 1e6
        latency_cycles = 100  # Basic estimate
        power_w = max(self.compute_units * 0.1 + 1.0, 0.1)  # Ensure non-zero power
        
        performance = {
            "throughput_ops_s": throughput_ops_s,
            "latency_cycles": latency_cycles,
            "latency_ms": latency_cycles / (self.frequency_mhz * 1000) if self.frequency_mhz > 0 else 0,
            "power_w": power_w,
            "efficiency_ops_w": throughput_ops_s / power_w if power_w > 0 else 0,
            "area_mm2": self.compute_units * 0.1 + 2.0,  # Estimate
        }
        
        self.performance_model = performance
        return performance
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert accelerator to dictionary representation."""
        return {
            "compute_units": self.compute_units,
            "memory_hierarchy": self.memory_hierarchy,
            "dataflow": self.dataflow,
            "frequency_mhz": self.frequency_mhz,
            "data_width": self.data_width,
            "precision": self.precision,
            "power_budget_w": self.power_budget_w,
            "area_budget_mm2": self.area_budget_mm2,
            "performance_model": self.performance_model,
            "resource_estimates": self.resource_estimates,
        }


class AcceleratorDesigner:
    """Main class for designing hardware accelerators based on neural network models."""
    
    def __init__(self):
        """Initialize the accelerator designer."""
        self.supported_frameworks = ["pytorch", "tensorflow", "onnx"]
        self.dataflow_options = ["weight_stationary", "output_stationary", "row_stationary"]
        self.memory_options = ["sram_32kb", "sram_64kb", "sram_128kb", "dram"]
        
        # Performance optimization using shared thread pool
        self._cache = {}
    
    def validate_design_parameters(self, compute_units: int, memory_hierarchy: List[str], dataflow: str) -> None:
        """Validate design parameters before creating accelerator."""
        if compute_units <= 0:
            raise ValueError(f"Compute units must be positive, got {compute_units}")
        
        if compute_units > 1024:
            raise ValueError(f"Compute units cannot exceed 1024, got {compute_units}")
            
        if dataflow not in self.dataflow_options:
            raise ValueError(f"Unsupported dataflow: {dataflow}. Supported: {self.dataflow_options}")
            
        if not memory_hierarchy or not isinstance(memory_hierarchy, list):
            raise ValueError("Memory hierarchy must be a non-empty list")
        self._cache_lock = threading.RLock()
        
        # Design optimization statistics
        self.design_stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "parallel_designs": 0,
            "total_designs": 0
        }
    
    @cached(cache_type="model", ttl=3600.0)
    def profile_model(self, model: Any, input_shape: Tuple[int, ...], framework: str = "auto") -> ModelProfile:
        """
        Profile a neural network model to extract computational requirements.
        
        Args:
            model: Neural network model (supports PyTorch, TensorFlow, ONNX)
            input_shape: Input tensor shape
            framework: ML framework ("pytorch", "tensorflow", "onnx", "auto")
            
        Returns:
            ModelProfile with computational characteristics
        """
        # Create cache key from model and input shape
        model_hash = self._hash_model_and_shape(model, input_shape)
        
        with self._cache_lock:
            if model_hash in self._cache:
                self.design_stats["cache_hits"] += 1
                return self._cache[model_hash]
            else:
                self.design_stats["cache_misses"] += 1
        
        # Try real model profiling first, fall back to mock if needed
        try:
            operations, parameters, layer_types = self._profile_real_model(model, input_shape, framework)
        except Exception as e:
            logger.warning(f"Real model profiling failed, using estimates: {e}")
            operations = self._estimate_operations(input_shape)
            parameters = self._estimate_parameters(input_shape)
            layer_types = ["conv2d", "dense", "activation"]
        
        # Calculate derived metrics
        peak_gflops = sum(operations.values()) / 1e9
        memory_mb = parameters * 4 / (1024 * 1024)  # Assume 32-bit weights
        bandwidth_gb_s = peak_gflops * 4 / 1024  # Rough estimate
        compute_intensity = peak_gflops / bandwidth_gb_s if bandwidth_gb_s > 0 else 0
        
        profile = ModelProfile(
            peak_gflops=peak_gflops,
            bandwidth_gb_s=bandwidth_gb_s,
            operations=operations,
            parameters=parameters,
            memory_mb=memory_mb,
            compute_intensity=compute_intensity,
            layer_types=layer_types,
            model_size_mb=memory_mb,
        )
        
        # Cache the result
        with self._cache_lock:
            self._cache[model_hash] = profile
        
        return profile
    
    @cached(cache_type="accelerator", ttl=1800.0)
    def design(
        self,
        compute_units: int = 64,
        memory_hierarchy: Optional[List[str]] = None,
        dataflow: str = "weight_stationary",
        **kwargs
    ) -> Accelerator:
        """
        Design an accelerator with specified parameters.
        
        Args:
            compute_units: Number of processing elements
            memory_hierarchy: Memory system configuration
            dataflow: Data movement pattern
            **kwargs: Additional accelerator parameters
            
        Returns:
            Configured Accelerator instance
        """
        if memory_hierarchy is None:
            memory_hierarchy = ["sram_64kb", "dram"]
        
        # Validate parameters
        self.validate_design_parameters(compute_units, memory_hierarchy, dataflow)
        
        accelerator = Accelerator(
            compute_units=compute_units,
            memory_hierarchy=memory_hierarchy,
            dataflow=dataflow,
            **kwargs
        )
        
        # Generate initial performance estimates
        accelerator.estimate_performance()
        
        return accelerator
    
    def optimize_for_model(self, model_profile: ModelProfile, constraints: Dict[str, float]) -> Accelerator:
        """
        Optimize accelerator design for a specific model profile.
        
        Args:
            model_profile: Target model characteristics
            constraints: Performance/power/area constraints
            
        Returns:
            Optimized Accelerator design
        """
        # Simple optimization heuristic
        target_throughput = constraints.get("target_fps", 30) * model_profile.peak_gflops
        power_budget = constraints.get("power_budget", 5.0)
        
        # Scale compute units based on throughput requirements
        base_units = 16
        scaling_factor = max(1, target_throughput / (base_units * 200e6))  # 200 MHz base
        compute_units = int(base_units * scaling_factor)
        
        # Adjust memory hierarchy based on model size
        if model_profile.memory_mb > 64:
            memory_hierarchy = ["sram_128kb", "dram"]
        else:
            memory_hierarchy = ["sram_64kb", "dram"]
        
        # Choose dataflow based on model characteristics
        if "conv2d" in model_profile.layer_types:
            dataflow = "weight_stationary"
        else:
            dataflow = "output_stationary"
        
        return self.design(
            compute_units=compute_units,
            memory_hierarchy=memory_hierarchy,
            dataflow=dataflow,
            power_budget_w=power_budget,
        )
    
    def _estimate_operations(self, input_shape: Tuple[int, ...]) -> Dict[str, int]:
        """Estimate operation counts based on input shape."""
        h, w = input_shape[-2:]  # Assume HW at end
        
        # Mock operation counts for a typical CNN
        return {
            "conv2d": h * w * 32 * 9,  # 32 filters, 3x3 kernels
            "pooling": h * w * 4,
            "dense": 1000 * 512,  # Final classification layer
            "activation": h * w * 32,
        }
    
    def _estimate_parameters(self, input_shape: Tuple[int, ...]) -> int:
        """Estimate parameter count based on input shape."""
        # Mock parameter count for a typical CNN
        return (
            3 * 32 * 9 +        # First conv layer
            32 * 64 * 9 +       # Second conv layer  
            64 * 128 * 9 +      # Third conv layer
            128 * 512 +         # Dense layer
            512 * 1000          # Classification layer
        )
    
    def _profile_real_model(self, model: Any, input_shape: Tuple[int, ...], framework: str) -> Tuple[Dict[str, int], int, List[str]]:
        """
        Profile a real neural network model to extract computational requirements.
        
        Args:
            model: Neural network model object
            input_shape: Input tensor shape
            framework: ML framework type
            
        Returns:
            Tuple of (operations dict, parameter count, layer types)
        """
        operations = {}
        parameters = 0
        layer_types = []
        
        # Auto-detect framework if needed
        if framework == "auto":
            framework = self._detect_model_framework(model)
        
        if framework == "pytorch":
            operations, parameters, layer_types = self._profile_pytorch_model(model, input_shape)
        elif framework == "tensorflow":
            operations, parameters, layer_types = self._profile_tensorflow_model(model, input_shape)
        elif framework == "onnx":
            operations, parameters, layer_types = self._profile_onnx_model(model, input_shape)
        else:
            raise ValueError(f"Unsupported framework: {framework}")
        
        return operations, parameters, layer_types
    
    def _detect_model_framework(self, model: Any) -> str:
        """Detect the ML framework of a model."""
        model_type = str(type(model))
        
        if "torch" in model_type.lower():
            return "pytorch"
        elif "tensorflow" in model_type.lower() or "keras" in model_type.lower():
            return "tensorflow"
        elif "onnx" in model_type.lower():
            return "onnx"
        elif hasattr(model, 'path') and isinstance(model.path, str):
            # Check file extension
            if model.path.endswith('.onnx'):
                return "onnx"
            elif model.path.endswith(('.pt', '.pth')):
                return "pytorch"
            elif model.path.endswith('.pb'):
                return "tensorflow"
        
        # Default fallback
        return "pytorch"
    
    def _profile_pytorch_model(self, model: Any, input_shape: Tuple[int, ...]) -> Tuple[Dict[str, int], int, List[str]]:
        """Profile PyTorch model."""
        try:
            import torch
            import torch.nn as nn
            
            # If model is a path, load it
            if isinstance(model, str) or hasattr(model, 'path'):
                model_path = getattr(model, 'path', model) if hasattr(model, 'path') else model
                model = torch.load(model_path, map_location='cpu')
            
            # Ensure model is in eval mode
            model.eval()
            
            operations = {"conv2d": 0, "linear": 0, "activation": 0, "pooling": 0, "normalization": 0}
            parameters = 0
            layer_types = []
            
            # Count parameters and operations
            for name, module in model.named_modules():
                if isinstance(module, nn.Conv2d):
                    layer_types.append("conv2d")
                    # Calculate FLOPS for convolution
                    if len(input_shape) >= 4:
                        h, w = input_shape[-2:]
                        kernel_ops = module.kernel_size[0] * module.kernel_size[1]
                        output_elements = h * w * module.out_channels
                        operations["conv2d"] += kernel_ops * module.in_channels * output_elements
                    params = sum(p.numel() for p in module.parameters())
                    parameters += params
                    
                elif isinstance(module, nn.Linear):
                    layer_types.append("linear")
                    operations["linear"] += module.in_features * module.out_features
                    params = sum(p.numel() for p in module.parameters())
                    parameters += params
                    
                elif isinstance(module, (nn.ReLU, nn.Sigmoid, nn.Tanh, nn.GELU)):
                    layer_types.append("activation")
                    # Assume same size as input
                    operations["activation"] += np.prod(input_shape)
                    
                elif isinstance(module, (nn.MaxPool2d, nn.AvgPool2d)):
                    layer_types.append("pooling")
                    operations["pooling"] += np.prod(input_shape) // 4  # Rough estimate
                    
                elif isinstance(module, (nn.BatchNorm2d, nn.LayerNorm)):
                    layer_types.append("normalization")
                    operations["normalization"] += np.prod(input_shape)
                    params = sum(p.numel() for p in module.parameters())
                    parameters += params
            
            return operations, parameters, layer_types
            
        except ImportError:
            logger.warning("PyTorch not available, falling back to estimates")
            raise
        except Exception as e:
            logger.error(f"PyTorch model profiling failed: {e}")
            raise
    
    def _profile_tensorflow_model(self, model: Any, input_shape: Tuple[int, ...]) -> Tuple[Dict[str, int], int, List[str]]:
        """Profile TensorFlow/Keras model."""
        try:
            import tensorflow as tf
            
            # If model is a path, load it
            if isinstance(model, str) or hasattr(model, 'path'):
                model_path = getattr(model, 'path', model) if hasattr(model, 'path') else model
                model = tf.keras.models.load_model(model_path)
            
            operations = {"conv2d": 0, "dense": 0, "activation": 0, "pooling": 0, "normalization": 0}
            parameters = 0
            layer_types = []
            
            for layer in model.layers:
                layer_type = type(layer).__name__.lower()
                
                if 'conv2d' in layer_type:
                    layer_types.append("conv2d")
                    if hasattr(layer, 'kernel_size') and len(input_shape) >= 4:
                        h, w = input_shape[-2:]
                        kernel_ops = layer.kernel_size[0] * layer.kernel_size[1]
                        output_elements = h * w * layer.filters
                        operations["conv2d"] += kernel_ops * layer.input_spec.axes[-1] * output_elements
                    parameters += layer.count_params()
                    
                elif 'dense' in layer_type:
                    layer_types.append("dense")
                    if hasattr(layer, 'units'):
                        operations["dense"] += layer.input_spec.axes[-1] * layer.units
                    parameters += layer.count_params()
                    
                elif any(act in layer_type for act in ['relu', 'sigmoid', 'tanh', 'activation']):
                    layer_types.append("activation")
                    operations["activation"] += np.prod(input_shape)
                    
                elif 'pool' in layer_type:
                    layer_types.append("pooling")
                    operations["pooling"] += np.prod(input_shape) // 4
                    
                elif any(norm in layer_type for norm in ['batchnorm', 'layernorm']):
                    layer_types.append("normalization")
                    operations["normalization"] += np.prod(input_shape)
                    parameters += layer.count_params()
            
            return operations, parameters, layer_types
            
        except ImportError:
            logger.warning("TensorFlow not available, falling back to estimates")
            raise
        except Exception as e:
            logger.error(f"TensorFlow model profiling failed: {e}")
            raise
    
    def _profile_onnx_model(self, model: Any, input_shape: Tuple[int, ...]) -> Tuple[Dict[str, int], int, List[str]]:
        """Profile ONNX model."""
        try:
            import onnx
            
            # If model is a path, load it
            if isinstance(model, str) or hasattr(model, 'path'):
                model_path = getattr(model, 'path', model) if hasattr(model, 'path') else model
                model = onnx.load(model_path)
            
            operations = {"conv": 0, "matmul": 0, "add": 0, "relu": 0, "pool": 0}
            parameters = 0
            layer_types = []
            
            # Analyze ONNX graph
            for node in model.graph.node:
                op_type = node.op_type.lower()
                
                if op_type == "conv":
                    layer_types.append("conv2d")
                    # Rough estimate based on typical conv operations
                    operations["conv"] += np.prod(input_shape) * 9  # 3x3 kernel estimate
                elif op_type in ["matmul", "gemm"]:
                    layer_types.append("dense")
                    operations["matmul"] += np.prod(input_shape) * 1000  # Estimate
                elif op_type == "add":
                    operations["add"] += np.prod(input_shape)
                elif op_type == "relu":
                    layer_types.append("activation")
                    operations["relu"] += np.prod(input_shape)
                elif op_type in ["maxpool", "averagepool"]:
                    layer_types.append("pooling")
                    operations["pool"] += np.prod(input_shape) // 4
            
            # Count parameters from initializers
            for initializer in model.graph.initializer:
                tensor_shape = [dim.dim_value for dim in initializer.type.tensor_type.shape.dim]
                parameters += np.prod(tensor_shape)
            
            return operations, parameters, layer_types
            
        except ImportError:
            logger.warning("ONNX not available, falling back to estimates")
            raise
        except Exception as e:
            logger.error(f"ONNX model profiling failed: {e}")
            raise

    def _hash_model_and_shape(self, model: Any, input_shape: Tuple[int, ...]) -> str:
        """Generate cache key from model and input shape."""
        # Create a hash from model characteristics and input shape
        model_str = str(getattr(model, 'path', repr(model)))
        shape_str = str(input_shape)
        combined = f"{model_str}_{shape_str}".encode('utf-8')
        return hashlib.md5(combined).hexdigest()
    
    def design_parallel(
        self,
        configurations: List[Dict[str, Any]],
        max_workers: Optional[int] = None,
        batch_size: int = 10
    ) -> List[Accelerator]:
        """
        Design multiple accelerators in parallel with enhanced performance.
        
        Args:
            configurations: List of design configurations
            max_workers: Maximum parallel workers (auto-scaled)
            batch_size: Batch size for processing
            
        Returns:
            List of designed accelerators
        """
        import time
        
        self.design_stats["parallel_designs"] += 1
        start_time = time.time()
        
        # Auto-scale workers based on system resources and workload
        if max_workers is None:
            try:
                import psutil
                cpu_count = psutil.cpu_count(logical=False) or 2
                system_load = psutil.cpu_percent(interval=0.1)
                
                # Adaptive worker scaling
                if system_load < 50:
                    max_workers = min(len(configurations), cpu_count * 2)
                elif system_load < 80:
                    max_workers = min(len(configurations), cpu_count)
                else:
                    max_workers = min(len(configurations), max(2, cpu_count // 2))
            except ImportError:
                # Fallback if psutil not available
                max_workers = min(len(configurations), 4)
        
        logger.info(f"Parallel design with {max_workers} workers, {len(configurations)} configs")
        
        # Process in batches for memory efficiency
        all_results = []
        for i in range(0, len(configurations), batch_size):
            batch = configurations[i:i + batch_size]
            batch_results = self._process_design_batch(batch, max_workers)
            all_results.extend(batch_results)
            
            # Brief pause between batches to prevent resource exhaustion
            if i + batch_size < len(configurations):
                time.sleep(0.1)
        
        successful_designs = [acc for acc in all_results if acc is not None]
        duration = time.time() - start_time
        
        logger.info(f"Completed parallel design: {len(successful_designs)}/{len(configurations)} successful in {duration:.2f}s")
        record_metric("parallel_design_duration", duration, "timer")
        record_metric("parallel_design_success_rate", len(successful_designs) / len(configurations), "gauge")
        
        return successful_designs
    
    def _process_design_batch(self, batch: List[Dict[str, Any]], max_workers: int) -> List[Optional[Accelerator]]:
        """Process a batch of design configurations."""
        results = []
        executor = get_thread_pool(max_workers)
        
        # Submit all tasks
        future_to_config = {
            executor.submit(self._design_single_config, config): config
            for config in batch
        }
        
        # Collect results with timeout handling
        for future in as_completed(future_to_config, timeout=300):  # 5 minute timeout
            try:
                accelerator = future.result(timeout=60)  # 1 minute per design
                results.append(accelerator)
                record_metric("accelerator_design_success", 1, "counter")
            except Exception as e:
                logger.error(f"Design failed: {e}")
                record_metric("accelerator_design_failure", 1, "counter")
                results.append(None)
        
        return results
    
    def _design_single_config(self, config: Dict[str, Any]) -> Accelerator:
        """Design single accelerator from configuration."""
        return self.design(
            compute_units=config.get("compute_units", 64),
            memory_hierarchy=config.get("memory_hierarchy", ["sram_64kb", "dram"]),
            dataflow=config.get("dataflow", "weight_stationary"),
            frequency_mhz=config.get("frequency_mhz", 200.0),
            data_width=config.get("data_width", 8),
            precision=config.get("precision", "int8"),
            power_budget_w=config.get("power_budget_w", 5.0)
        )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        total_requests = self.design_stats["cache_hits"] + self.design_stats["cache_misses"]
        cache_hit_rate = (self.design_stats["cache_hits"] / total_requests) if total_requests > 0 else 0
        
        return {
            **self.design_stats,
            "cache_hit_rate": cache_hit_rate,
            "cache_size": len(self._cache)
        }
    
    def clear_cache(self) -> None:
        """Clear design cache."""
        with self._cache_lock:
            self._cache.clear()
        self.design_stats["cache_hits"] = 0
        self.design_stats["cache_misses"] = 0