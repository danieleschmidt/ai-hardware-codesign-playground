# AI Hardware Co-Design Playground User Guide

Welcome to the AI Hardware Co-Design Playground! This comprehensive guide will help you get started with co-optimizing neural networks and hardware accelerators.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Key Concepts](#key-concepts)
3. [Model Profiling](#model-profiling)
4. [Accelerator Design](#accelerator-design)
5. [Design Space Exploration](#design-space-exploration)
6. [Co-Optimization](#co-optimization)
7. [Advanced Features](#advanced-features)
8. [Best Practices](#best-practices)
9. [Troubleshooting](#troubleshooting)

## Getting Started

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (optional, for acceleration)
- Docker (for containerized deployment)
- 8GB+ RAM recommended

### Installation

#### Option 1: Python Package

```bash
pip install ai-hardware-codesign-playground
```

#### Option 2: Docker

```bash
docker-compose up -d
```

#### Option 3: From Source

```bash
git clone https://github.com/terragon-labs/codesign-playground.git
cd codesign-playground
pip install -r requirements.txt
python -m codesign_playground.server
```

### First Steps

1. **Start the Server**:
   ```bash
   python -m codesign_playground.server
   ```

2. **Access the Web Interface**:
   Open `http://localhost:8000` in your browser

3. **Run Your First Optimization**:
   ```python
   from codesign_playground import CodesignClient
   
   client = CodesignClient()
   
   # Profile a model
   profile = client.profile_model("resnet18", input_shape=[224, 224, 3])
   
   # Design an accelerator
   accelerator = client.design_accelerator(compute_units=64)
   
   print(f"Model GFLOPS: {profile.peak_gflops}")
   print(f"Accelerator throughput: {accelerator.throughput_ops_s}")
   ```

## Key Concepts

### Neural Network Profiling

**Model profiling** analyzes the computational characteristics of neural networks:

- **Operations Count**: FLOPs, memory accesses, parameter count
- **Compute Patterns**: Layer types, data flow, memory usage
- **Performance Metrics**: Theoretical peak performance, bandwidth requirements

### Hardware Accelerator Design

**Accelerator design** creates custom hardware optimized for neural networks:

- **Compute Units**: Parallel processing elements (PEs)
- **Memory Hierarchy**: Cache levels, bandwidth, latency
- **Dataflow**: How data moves through the accelerator
- **Precision**: Numerical precision (FP32, FP16, INT8)

### Co-Optimization

**Co-optimization** jointly optimizes both the neural network and hardware:

- **Model Optimization**: Quantization, pruning, architecture changes
- **Hardware Optimization**: Resource allocation, frequency scaling
- **Joint Optimization**: Simultaneous model and hardware tuning

## Model Profiling

### Supported Frameworks

- **PyTorch**: `.pt`, `.pth`, `.onnx` files
- **TensorFlow**: `.pb`, `.h5`, `.tflite` files  
- **ONNX**: `.onnx` universal format

### Profiling Your Model

#### Basic Profiling

```python
from codesign_playground import ModelProfiler

profiler = ModelProfiler()

# Profile a PyTorch model
profile = profiler.profile_model(
    model_path="model.pt",
    input_shape=[1, 3, 224, 224],
    framework="pytorch"
)

print(f"Peak GFLOPS: {profile.peak_gflops}")
print(f"Parameters: {profile.parameters:,}")
print(f"Memory: {profile.memory_mb:.1f} MB")
print(f"Layer types: {profile.layer_types}")
```

#### Advanced Profiling

```python
# Profile with detailed analysis
profile = profiler.profile_model(
    model_path="model.pt",
    input_shape=[1, 3, 224, 224],
    batch_sizes=[1, 4, 8, 16],  # Multi-batch analysis
    analyze_bottlenecks=True,   # Find performance bottlenecks
    memory_optimization=True,   # Memory usage analysis
    precision_analysis=True     # Mixed precision opportunities
)

# Access detailed results
print("Bottlenecks:", profile.bottlenecks)
print("Memory hotspots:", profile.memory_hotspots)
print("Quantization candidates:", profile.quantization_opportunities)
```

### Understanding Profile Results

#### Operations Breakdown
```python
for op_type, count in profile.operations.items():
    percentage = (count / sum(profile.operations.values())) * 100
    print(f"{op_type}: {count:,} ops ({percentage:.1f}%)")
```

#### Performance Metrics
- **Peak GFLOPS**: Theoretical maximum performance
- **Compute Intensity**: FLOPS per byte of memory access
- **Bandwidth Requirements**: Memory bandwidth needed
- **Parallelizability**: How well operations can be parallelized

## Accelerator Design

### Design Parameters

#### Core Parameters
- **Compute Units**: Number of parallel processing elements
- **Dataflow**: Data movement pattern (`weight_stationary`, `output_stationary`, `row_stationary`)
- **Memory Hierarchy**: Cache sizes and organization
- **Frequency**: Clock frequency in MHz

#### Example Design

```python
from codesign_playground import AcceleratorDesigner

designer = AcceleratorDesigner()

# Basic accelerator design
accelerator = designer.design(
    compute_units=64,
    memory_hierarchy=["sram_64kb", "dram"],
    dataflow="weight_stationary",
    frequency_mhz=200.0,
    precision="int8"
)

print(f"Throughput: {accelerator.performance_model['throughput_ops_s']:,} ops/s")
print(f"Power: {accelerator.performance_model['power_w']:.1f} W")
print(f"Area: {accelerator.performance_model['area_mm2']:.1f} mm²")
```

### Dataflow Patterns

#### Weight Stationary
- **Best for**: Convolution-heavy networks
- **Characteristics**: Weights stay in compute units, activations flow
- **Efficiency**: High for CNNs, moderate for other layers

#### Output Stationary  
- **Best for**: Dense/fully-connected layers
- **Characteristics**: Output accumulations stay local
- **Efficiency**: High for matrix multiplications

#### Row Stationary
- **Best for**: Balanced workloads
- **Characteristics**: Hybrid approach with flexible data movement
- **Efficiency**: Good general-purpose choice

### Performance Estimation

The platform provides detailed performance models:

```python
perf = accelerator.performance_model

print("Performance Metrics:")
print(f"  Throughput: {perf['throughput_ops_s']:,} ops/s")
print(f"  Latency: {perf['latency_ms']:.2f} ms")
print(f"  Power: {perf['power_w']:.1f} W")
print(f"  Efficiency: {perf['efficiency_ops_w']:,} ops/W")
print(f"  Area: {perf['area_mm2']:.1f} mm²")

print("Resource Estimates:")
resources = accelerator.resource_estimates
print(f"  LUTs: {resources['luts']:,}")
print(f"  DSP blocks: {resources['dsp']:,}")
print(f"  Block RAM: {resources['bram']:,}")
```

## Design Space Exploration

### Pareto Frontier Analysis

Find optimal trade-offs between performance, power, and area:

```python
from codesign_playground import DesignSpaceExplorer

explorer = DesignSpaceExplorer()

# Define design space
design_space = {
    "compute_units": [16, 32, 64, 128, 256],
    "memory_hierarchy": [
        ["sram_32kb", "dram"],
        ["sram_64kb", "dram"], 
        ["sram_128kb", "dram"]
    ],
    "dataflow": ["weight_stationary", "output_stationary"],
    "frequency_mhz": [100, 200, 400, 600]
}

# Explore for a specific model
pareto_designs = explorer.explore_pareto_frontier(
    design_space=design_space,
    target_profile=profile,
    objectives=["performance", "power", "area"],
    max_samples=100
)

# Analyze results
for i, design in enumerate(pareto_designs[:5]):
    print(f"Design {i+1}:")
    print(f"  Compute Units: {design.accelerator.compute_units}")
    print(f"  Performance: {design.metrics['performance']:.2f}")
    print(f"  Power: {design.metrics['power']:.1f} W")
    print(f"  Area: {design.metrics['area']:.1f} mm²")
```

### Visualization

Generate interactive plots of the design space:

```python
# Create Pareto frontier plot
explorer.plot_pareto_frontier(
    designs=pareto_designs,
    objectives=["power", "performance"],
    save_path="pareto_plot.html"
)

# Create 3D design space visualization  
explorer.plot_design_space_3d(
    designs=pareto_designs,
    dimensions=["compute_units", "frequency_mhz", "power"],
    save_path="design_space_3d.html"
)
```

### Multi-Objective Optimization

```python
# Advanced multi-objective optimization
optimization_config = {
    "objectives": ["latency", "power", "accuracy"],
    "constraints": {
        "max_power_w": 10.0,
        "min_accuracy": 0.95,
        "max_latency_ms": 20.0
    },
    "weights": {
        "latency": 0.4,
        "power": 0.3, 
        "accuracy": 0.3
    }
}

optimized_designs = explorer.multi_objective_optimize(
    model_profile=profile,
    config=optimization_config,
    max_iterations=50
)
```

## Co-Optimization

### Joint Model-Hardware Optimization

Simultaneously optimize both the neural network and accelerator:

```python
from codesign_playground import CoOptimizer

co_optimizer = CoOptimizer()

# Define optimization objectives and constraints
objectives = {
    "accuracy": {"weight": 0.4, "direction": "maximize"},
    "latency": {"weight": 0.3, "direction": "minimize"}, 
    "power": {"weight": 0.3, "direction": "minimize"}
}

constraints = {
    "min_accuracy": 0.95,
    "max_power_w": 8.0,
    "max_area_mm2": 150.0
}

# Run co-optimization
result = co_optimizer.optimize(
    model=original_model,
    initial_accelerator=base_accelerator,
    objectives=objectives,
    constraints=constraints,
    max_iterations=100
)

print("Optimization Results:")
print(f"Final accuracy: {result.final_metrics['accuracy']:.3f}")
print(f"Final latency: {result.final_metrics['latency_ms']:.2f} ms")  
print(f"Final power: {result.final_metrics['power_w']:.1f} W")
print(f"Iterations: {result.iterations}")
print(f"Convergence: {result.converged}")
```

### Model Optimization Techniques

#### Quantization
```python
# Apply quantization to reduce precision
quantized_model = co_optimizer.quantize_model(
    model=original_model,
    target_precision="int8",
    calibration_data=calibration_dataset,
    accuracy_threshold=0.95
)
```

#### Pruning
```python
# Remove redundant connections
pruned_model = co_optimizer.prune_model(
    model=original_model,
    sparsity_ratio=0.7,
    structured=False,  # Unstructured pruning
    fine_tune_epochs=10
)
```

#### Architecture Search
```python
# Find optimal architecture modifications
optimized_arch = co_optimizer.optimize_architecture(
    base_model=original_model,
    search_space="mobilenet_v3",
    hardware_constraints=accelerator_constraints,
    search_iterations=50
)
```

## Advanced Features

### Custom Hardware Templates

Create custom accelerator templates:

```python
from codesign_playground.templates import CustomTemplate

class MyAcceleratorTemplate(CustomTemplate):
    def __init__(self):
        super().__init__(name="my_custom_accelerator")
        
    def generate_rtl(self, config):
        # Custom RTL generation logic
        return self._generate_verilog_code(config)
    
    def estimate_performance(self, config):
        # Custom performance modeling
        return {
            "throughput_ops_s": self._compute_throughput(config),
            "power_w": self._estimate_power(config),
            "area_mm2": self._estimate_area(config)
        }

# Register and use custom template
designer.register_template(MyAcceleratorTemplate())
```

### Plugin System

Extend functionality with plugins:

```python
from codesign_playground.plugins import OptimizationPlugin

class MyOptimizationPlugin(OptimizationPlugin):
    def __init__(self):
        super().__init__(name="my_optimizer")
    
    def optimize(self, model, accelerator, constraints):
        # Custom optimization algorithm
        return optimized_model, optimized_accelerator

# Load plugin
co_optimizer.load_plugin(MyOptimizationPlugin())
```

### Batch Processing

Process multiple models efficiently:

```python
from codesign_playground import BatchProcessor

processor = BatchProcessor(max_workers=4)

# Define batch job
models = [
    {"path": "model1.pt", "name": "resnet18"},
    {"path": "model2.pt", "name": "mobilenet_v2"},
    {"path": "model3.pt", "name": "efficientnet_b0"}
]

# Process batch
results = processor.process_batch(
    models=models,
    input_shape=[224, 224, 3],
    accelerator_configs=[
        {"compute_units": 64, "dataflow": "weight_stationary"},
        {"compute_units": 128, "dataflow": "output_stationary"}
    ]
)

# Analyze batch results
for model_name, result in results.items():
    print(f"{model_name}: {result.best_design.metrics}")
```

## Best Practices

### Model Profiling Best Practices

1. **Use Representative Input Shapes**: Profile with realistic input dimensions
2. **Consider Batch Sizes**: Different batch sizes can reveal different bottlenecks
3. **Profile Multiple Models**: Compare similar architectures to understand trade-offs
4. **Validate Profiles**: Cross-check with actual hardware measurements when possible

### Accelerator Design Best Practices

1. **Start Simple**: Begin with basic configurations and iterate
2. **Consider Memory Bandwidth**: Often the limiting factor in performance
3. **Balance Resources**: Don't over-provision any single resource
4. **Validate Timing**: Ensure designs meet timing requirements

### Optimization Best Practices

1. **Set Realistic Constraints**: Use achievable accuracy and power targets
2. **Use Multiple Objectives**: Single-objective optimization may miss good solutions
3. **Iterate Gradually**: Make incremental improvements rather than large jumps
4. **Validate Results**: Test optimized models on real hardware

### Performance Tuning

```python
# Enable performance optimizations
config = {
    "enable_caching": True,
    "parallel_exploration": True,
    "max_workers": 8,
    "memory_limit_mb": 4096,
    "gpu_acceleration": True
}

# Apply configuration
client.configure(config)
```

## Troubleshooting

### Common Issues

#### Model Loading Errors
```python
# Issue: Model fails to load
try:
    profile = profiler.profile_model("model.pt")
except ModelLoadError as e:
    print(f"Model loading failed: {e}")
    # Try converting model format
    converted_model = profiler.convert_model("model.pt", target_format="onnx")
    profile = profiler.profile_model(converted_model)
```

#### Memory Issues
```python
# Issue: Out of memory during optimization
try:
    result = optimizer.optimize(large_model)
except MemoryError:
    # Reduce batch size or enable gradient checkpointing
    optimizer.configure(
        batch_size=1,
        gradient_checkpointing=True,
        memory_efficient=True
    )
    result = optimizer.optimize(large_model)
```

#### Convergence Problems
```python
# Issue: Optimization doesn't converge
if not result.converged:
    print(f"Optimization stopped at iteration {result.iterations}")
    print("Suggestions:")
    print("- Increase max_iterations")
    print("- Adjust learning rate")
    print("- Relax constraints")
    print("- Try different initialization")
```

### Performance Issues

#### Slow Profiling
- **Solution**: Enable caching, use GPU acceleration
- **Command**: `profiler.configure(cache=True, use_gpu=True)`

#### Slow Exploration
- **Solution**: Reduce design space size, use parallel processing
- **Command**: `explorer.configure(max_workers=8, early_stopping=True)`

### Getting Help

1. **Documentation**: Check the full documentation at `docs/`
2. **Examples**: See `examples/` directory for code samples  
3. **Community**: Join our Discord community for questions
4. **Issues**: Report bugs on GitHub
5. **Support**: Contact support@terragon-labs.com for enterprise support

### Debug Mode

Enable debug mode for detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Or configure the platform logger
from codesign_playground.utils.logging import configure_logging
configure_logging(level="DEBUG", format="detailed")
```

## Next Steps

1. **Try the Examples**: Start with `examples/quickstart.py`
2. **Join the Community**: Connect with other users and contributors
3. **Contribute**: Help improve the platform by contributing code or documentation
4. **Stay Updated**: Follow our blog for the latest features and research

Happy co-optimizing! 🚀