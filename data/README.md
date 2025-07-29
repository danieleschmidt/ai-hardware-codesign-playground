# Data Directory

This directory contains datasets, sample files, and other data assets used by the AI Hardware Co-Design Playground.

## Directory Structure

```
data/
├── datasets/          # Training and test datasets
├── models/           # Pre-trained model files
├── samples/          # Sample input files for testing
├── benchmarks/       # Benchmark datasets and results
└── cache/           # Cached data files
```

## Dataset Sources

### Neural Network Models
- ONNX model files for common architectures
- PyTorch and TensorFlow saved models
- Quantized model variants

### Hardware Specifications
- Reference accelerator configurations
- FPGA and ASIC design templates
- Performance benchmark data

### Synthetic Data
- Generated test cases for validation
- Synthetic workloads for performance testing
- Simulated hardware metrics

## Data Management

- Large files should be stored using Git LFS
- Temporary cache files are gitignored
- Sample data should be kept small (<10MB per file)
- Use compressed formats when appropriate

## Adding New Data

1. Place files in appropriate subdirectory
2. Update relevant documentation
3. Add entry to `.gitattributes` for large files
4. Consider using compressed formats