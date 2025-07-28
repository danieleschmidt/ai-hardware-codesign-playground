# Test Fixtures

This directory contains standardized test data and fixtures for consistent testing across the AI Hardware Co-Design Playground.

## Directory Structure

```
fixtures/
├── models/              # Sample ML models for testing
│   ├── small/           # Lightweight models for unit tests
│   │   ├── simple_linear.onnx
│   │   ├── basic_cnn.pt
│   │   └── toy_transformer.onnx
│   ├── medium/          # Realistic models for integration tests
│   │   ├── resnet18.onnx
│   │   ├── mobilenet_v2.pt
│   │   └── bert_base.onnx
│   └── large/           # Complex models for performance tests
│       ├── resnet50.onnx
│       ├── gpt2_medium.pt
│       └── vit_large.onnx
├── designs/            # Hardware design configurations
│   ├── templates/       # Template configurations
│   │   ├── systolic_configs.json
│   │   ├── vector_processor_configs.json
│   │   └── transformer_accelerator_configs.json
│   ├── generated/       # Pre-generated designs for testing
│   │   ├── sample_systolic.v
│   │   ├── sample_vector_proc.sv
│   │   └── sample_testbench.sv
│   └── reference/       # Reference implementations
│       ├── reference_designs.json
│       └── performance_baselines.json
└── data/               # Test datasets and inputs
    ├── inputs/          # Sample input data
    │   ├── image_samples/
    │   ├── text_samples/
    │   └── audio_samples/
    ├── outputs/         # Expected output data
    │   ├── model_outputs/
    │   ├── simulation_results/
    │   └── optimization_results/
    └── datasets/        # Complete test datasets
        ├── mnist_subset/
        ├── cifar10_subset/
        └── imagenet_subset/
```

## Model Fixtures

### Small Models (`models/small/`)
Lightweight models for unit tests that execute quickly:
- **simple_linear.onnx**: Basic linear layer for fundamental testing
- **basic_cnn.pt**: Simple CNN with 2-3 layers
- **toy_transformer.onnx**: Minimal transformer with 1-2 attention blocks

### Medium Models (`models/medium/`)
Realistic models for integration testing:
- **resnet18.onnx**: Standard ResNet-18 for image classification
- **mobilenet_v2.pt**: MobileNet v2 for mobile deployment testing
- **bert_base.onnx**: BERT-base for NLP testing

### Large Models (`models/large/`)
Complex models for performance and stress testing:
- **resnet50.onnx**: ResNet-50 for performance benchmarking
- **gpt2_medium.pt**: GPT-2 medium for language model testing
- **vit_large.onnx**: Vision Transformer large for transformer testing

## Design Fixtures

### Template Configurations (`designs/templates/`)
Standardized configurations for hardware templates:

```json
// systolic_configs.json
{
  "small_systolic": {
    "rows": 8,
    "cols": 8,
    "data_width": 8,
    "accumulator_width": 32
  },
  "medium_systolic": {
    "rows": 16,
    "cols": 16,
    "data_width": 8,
    "accumulator_width": 32
  },
  "large_systolic": {
    "rows": 32,
    "cols": 32,
    "data_width": 8,
    "accumulator_width": 32
  }
}
```

### Generated Designs (`designs/generated/`)
Pre-generated RTL for testing without full generation:
- **sample_systolic.v**: Complete systolic array implementation
- **sample_vector_proc.sv**: Vector processor with custom instructions
- **sample_testbench.sv**: Universal testbench for simulation

### Reference Designs (`designs/reference/`)
Reference implementations and performance baselines:
- **reference_designs.json**: Known-good design configurations
- **performance_baselines.json**: Expected performance metrics

## Data Fixtures

### Input Data (`data/inputs/`)
Sample input data for testing:
- **image_samples/**: JPEG/PNG images for vision models
- **text_samples/**: Text files for NLP models
- **audio_samples/**: WAV files for audio models

### Output Data (`data/outputs/`)
Expected outputs for verification:
- **model_outputs/**: Expected model inference results
- **simulation_results/**: Expected simulation outputs
- **optimization_results/**: Expected optimization outcomes

### Datasets (`data/datasets/`)
Complete datasets for comprehensive testing:
- **mnist_subset/**: 1000 MNIST samples for quick testing
- **cifar10_subset/**: 1000 CIFAR-10 samples for vision testing
- **imagenet_subset/**: 100 ImageNet samples for complex vision testing

## Usage in Tests

### Loading Model Fixtures
```python
import os
from pathlib import Path

def load_test_model(model_name, size="small"):
    """Load a test model fixture."""
    fixtures_dir = Path(__file__).parent / "fixtures"
    model_path = fixtures_dir / "models" / size / model_name
    
    if model_path.suffix == ".onnx":
        import onnx
        return onnx.load(str(model_path))
    elif model_path.suffix == ".pt":
        import torch
        return torch.load(str(model_path))
    else:
        raise ValueError(f"Unsupported model format: {model_path.suffix}")

# Usage in tests
@pytest.fixture
def small_cnn_model():
    return load_test_model("basic_cnn.pt", "small")
```

### Loading Design Fixtures
```python
import json
from pathlib import Path

def load_design_config(config_name):
    """Load a design configuration fixture."""
    fixtures_dir = Path(__file__).parent / "fixtures"
    config_path = fixtures_dir / "designs" / "templates" / f"{config_name}.json"
    
    with open(config_path) as f:
        return json.load(f)

# Usage in tests
@pytest.fixture
def systolic_configs():
    return load_design_config("systolic_configs")
```

### Loading Test Data
```python
import numpy as np
from pathlib import Path

def load_test_dataset(dataset_name):
    """Load a test dataset fixture."""
    fixtures_dir = Path(__file__).parent / "fixtures"
    dataset_path = fixtures_dir / "data" / "datasets" / dataset_name
    
    # Load based on dataset structure
    if dataset_name.startswith("mnist"):
        return load_mnist_subset(dataset_path)
    elif dataset_name.startswith("cifar"):
        return load_cifar_subset(dataset_path)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

# Usage in tests
@pytest.fixture
def mnist_test_data():
    return load_test_dataset("mnist_subset")
```

## Creating New Fixtures

### Guidelines
1. **Size Appropriateness**: Choose appropriate sizes for different test categories
2. **Deterministic**: Ensure fixtures produce consistent results across runs
3. **Documented**: Include clear descriptions and usage examples
4. **Versioned**: Track fixture versions for reproducibility
5. **Licensed**: Ensure proper licensing for all fixtures

### Adding Model Fixtures
```python
# Script to generate model fixtures
import torch
import torch.nn as nn
import onnx

def create_simple_cnn():
    """Create a simple CNN for testing."""
    class SimpleCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.fc = nn.Linear(32 * 8 * 8, 10)
        
        def forward(self, x):
            x = self.pool(torch.relu(self.conv1(x)))
            x = self.pool(torch.relu(self.conv2(x)))
            x = x.view(-1, 32 * 8 * 8)
            x = self.fc(x)
            return x
    
    return SimpleCNN()

# Save fixture
model = create_simple_cnn()
torch.save(model.state_dict(), "fixtures/models/small/basic_cnn.pt")
```

### Adding Design Fixtures
```python
# Script to generate design fixtures
import json

def create_systolic_configs():
    """Create systolic array configurations."""
    configs = {
        "test_small": {
            "rows": 4,
            "cols": 4,
            "data_width": 8,
            "frequency_mhz": 100
        },
        "test_medium": {
            "rows": 8,
            "cols": 8,
            "data_width": 8,
            "frequency_mhz": 200
        }
    }
    return configs

# Save fixture
configs = create_systolic_configs()
with open("fixtures/designs/templates/systolic_configs.json", "w") as f:
    json.dump(configs, f, indent=2)
```

## Maintenance

### Regular Tasks
- **Update models**: Refresh models with latest formats and architectures
- **Verify integrity**: Check that all fixtures load correctly
- **Performance check**: Ensure fixtures don't become too large
- **Documentation**: Keep usage examples up to date

### Fixture Lifecycle
1. **Creation**: Develop fixture following guidelines
2. **Validation**: Test fixture with representative use cases
3. **Integration**: Add to test suite and documentation
4. **Maintenance**: Regular updates and integrity checks
5. **Deprecation**: Remove obsolete fixtures with migration path

## Best Practices

### For Test Authors
- Use appropriate fixture size for test category
- Prefer parameterized tests over multiple similar fixtures
- Clean up temporary files after tests
- Document fixture dependencies and requirements

### For Fixture Maintainers
- Keep fixtures minimal but representative
- Ensure cross-platform compatibility
- Provide clear error messages for fixture issues
- Maintain backwards compatibility when possible

## Troubleshooting

### Common Issues
1. **Missing fixtures**: Ensure fixture generation scripts have been run
2. **Path issues**: Use `Path(__file__).parent` for relative paths
3. **Format issues**: Verify model/data formats are correct
4. **Size issues**: Check if fixtures are too large for test environment

### Regenerating Fixtures
```bash
# Regenerate all fixtures
python scripts/generate_fixtures.py --all

# Regenerate specific category
python scripts/generate_fixtures.py --models --size small

# Verify fixture integrity
python scripts/verify_fixtures.py
```

For questions about test fixtures, please refer to the [testing documentation](../docs/guides/developer/testing.md) or reach out through our community channels.