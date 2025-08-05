"""ML model-related test fixtures for AI Hardware Co-Design Playground."""

import pytest
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, List, Tuple
from pathlib import Path
import tempfile

@pytest.fixture
def simple_cnn_model() -> nn.Module:
    """Simple CNN model for testing."""
    class SimpleCNN(nn.Module):
        def __init__(self, num_classes=10):
            super(SimpleCNN, self).__init__()
            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(64 * 8 * 8, 128)
            self.fc2 = nn.Linear(128, num_classes)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.5)
        
        def forward(self, x):
            x = self.relu(self.conv1(x))
            x = self.pool(x)
            x = self.relu(self.conv2(x))
            x = self.pool(x)
            x = x.view(-1, 64 * 8 * 8)
            x = self.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            return x
    
    return SimpleCNN()

@pytest.fixture
def transformer_block_model() -> nn.Module:
    """Simple transformer block for testing."""
    class TransformerBlock(nn.Module):
        def __init__(self, embed_dim=256, num_heads=8, ff_dim=1024):
            super(TransformerBlock, self).__init__()
            self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
            self.norm1 = nn.LayerNorm(embed_dim)
            self.norm2 = nn.LayerNorm(embed_dim)
            self.feed_forward = nn.Sequential(
                nn.Linear(embed_dim, ff_dim),
                nn.ReLU(),
                nn.Linear(ff_dim, embed_dim)
            )
            self.dropout = nn.Dropout(0.1)
        
        def forward(self, x):
            # Self-attention
            attn_out, _ = self.attention(x, x, x)
            x = self.norm1(x + self.dropout(attn_out))
            
            # Feed forward
            ff_out = self.feed_forward(x)
            x = self.norm2(x + self.dropout(ff_out))
            
            return x
    
    return TransformerBlock()

@pytest.fixture
def resnet_block_model() -> nn.Module:
    """Simple ResNet block for testing."""
    class ResNetBlock(nn.Module):
        def __init__(self, in_channels, out_channels, stride=1):
            super(ResNetBlock, self).__init__()
            self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
            self.bn2 = nn.BatchNorm2d(out_channels)
            self.relu = nn.ReLU(inplace=True)
            
            self.shortcut = nn.Sequential()
            if stride != 1 or in_channels != out_channels:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                    nn.BatchNorm2d(out_channels)
                )
        
        def forward(self, x):
            out = self.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out += self.shortcut(x)
            out = self.relu(out)
            return out
    
    return ResNetBlock(64, 64)

@pytest.fixture
def quantized_model() -> nn.Module:
    """Quantized model for testing quantization-aware training."""
    model = nn.Sequential(
        nn.Conv2d(3, 32, 3),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(32, 10)
    )
    
    # Apply quantization
    torch.quantization.fuse_modules(model, [['0', '1']], inplace=True)
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    torch.quantization.prepare_qat(model, inplace=True)
    
    return model

@pytest.fixture
def sample_input_tensors() -> Dict[str, torch.Tensor]:
    """Sample input tensors for different model types."""
    return {
        "image_32x32": torch.randn(1, 3, 32, 32),
        "image_224x224": torch.randn(1, 3, 224, 224),
        "sequence_128": torch.randn(1, 128, 256),  # batch, seq_len, embed_dim
        "sequence_512": torch.randn(1, 512, 768),
        "tabular_small": torch.randn(1, 10),
        "tabular_large": torch.randn(1, 100),
    }

@pytest.fixture
def model_profiling_data() -> Dict[str, Any]:
    """Sample model profiling data for testing."""
    return {
        "total_params": 1234567,
        "trainable_params": 1234567,
        "model_size_mb": 4.7,
        "inference_time_ms": 15.3,
        "peak_memory_mb": 256,
        "flops": 2.5e9,
        "macs": 1.25e9,
        "layer_analysis": [
            {
                "name": "conv1",
                "type": "Conv2d",
                "input_shape": [1, 3, 224, 224],
                "output_shape": [1, 64, 112, 112],
                "params": 9408,
                "flops": 118013952,
                "memory_mb": 9.4,
                "compute_intensity": 12.5,
            },
            {
                "name": "fc1",
                "type": "Linear", 
                "input_shape": [1, 512],
                "output_shape": [1, 10],
                "params": 5130,
                "flops": 5120,
                "memory_mb": 0.02,
                "compute_intensity": 256.0,
            }
        ],
        "bottlenecks": [
            {
                "layer": "conv2",
                "type": "memory_bound",
                "severity": "high",
                "description": "Low arithmetic intensity"
            }
        ]
    }

@pytest.fixture
def optimization_targets() -> Dict[str, Any]:
    """Sample optimization targets for model optimization."""
    return {
        "accuracy_threshold": 0.95,
        "latency_target_ms": 10.0,
        "memory_budget_mb": 100,
        "energy_budget_mj": 1.0,
        "model_size_budget_mb": 5.0,
        "throughput_target_fps": 30,
        "quantization_constraints": {
            "min_bits_weights": 4,
            "min_bits_activations": 8,
            "exclude_layers": ["classifier"],
        },
        "pruning_constraints": {
            "max_sparsity": 0.9,
            "structured_pruning": True,
            "exclude_layers": ["bn", "bias"],
        }
    }

@pytest.fixture
def sample_dataset() -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample dataset for testing."""
    # Generate synthetic image classification dataset
    num_samples = 100
    num_classes = 10
    
    X = torch.randn(num_samples, 3, 32, 32)
    y = torch.randint(0, num_classes, (num_samples,))
    
    return X, y

@pytest.fixture
def model_compression_results() -> Dict[str, Any]:
    """Sample model compression results for testing."""
    return {
        "original_model": {
            "accuracy": 0.956,
            "model_size_mb": 10.2,
            "inference_time_ms": 25.4,
            "energy_per_inference_mj": 2.1,
        },
        "compressed_model": {
            "accuracy": 0.948,
            "model_size_mb": 2.8,
            "inference_time_ms": 8.7,
            "energy_per_inference_mj": 0.6,
        },
        "compression_metrics": {
            "size_reduction": 0.725,
            "speedup": 2.92,
            "energy_reduction": 0.714,
            "accuracy_loss": 0.008,
        },
        "applied_techniques": [
            "quantization_int8",
            "structured_pruning_50%",
            "knowledge_distillation",
        ]
    }

@pytest.fixture
def onnx_model_path(temp_dir, simple_cnn_model) -> Path:
    """Create a temporary ONNX model file for testing."""
    model_path = temp_dir / "test_model.onnx"
    
    # Export PyTorch model to ONNX
    dummy_input = torch.randn(1, 3, 32, 32)
    torch.onnx.export(
        simple_cnn_model,
        dummy_input,
        str(model_path),
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    return model_path

@pytest.fixture
def model_zoo() -> List[Dict[str, Any]]:
    """Collection of pre-defined models for testing."""
    return [
        {
            "name": "mobilenet_v2",
            "type": "image_classification",
            "input_shape": [1, 3, 224, 224],
            "num_classes": 1000,
            "params": 3504872,
            "flops": 300e6,
            "accuracy_imagenet": 0.72,
        },
        {
            "name": "resnet18",
            "type": "image_classification", 
            "input_shape": [1, 3, 224, 224],
            "num_classes": 1000,
            "params": 11689512,
            "flops": 1.8e9,
            "accuracy_imagenet": 0.70,
        },
        {
            "name": "bert_base",
            "type": "text_classification",
            "input_shape": [1, 512],
            "vocab_size": 30522,
            "params": 110e6,
            "flops": 22.5e9,
            "accuracy_glue": 0.85,
        },
        {
            "name": "efficientnet_b0",
            "type": "image_classification",
            "input_shape": [1, 3, 224, 224], 
            "num_classes": 1000,
            "params": 5288548,
            "flops": 390e6,
            "accuracy_imagenet": 0.77,
        }
    ]

@pytest.fixture
def hardware_aware_model_variants() -> List[Dict[str, Any]]:
    """Hardware-optimized model variants for testing."""
    return [
        {
            "name": "cnn_systolic_optimized",
            "base_model": "simple_cnn",
            "optimizations": [
                "conv_tiling_16x16",
                "weight_stationary_dataflow",
                "int8_quantization",
            ],
            "target_hardware": "systolic_array_16x16",
            "expected_speedup": 8.5,
            "expected_energy_reduction": 0.75,
        },
        {
            "name": "transformer_vector_optimized", 
            "base_model": "transformer_block",
            "optimizations": [
                "attention_sparsity_90%",
                "feedforward_vectorization",
                "fp16_mixed_precision",
            ],
            "target_hardware": "vector_processor_256",
            "expected_speedup": 12.3,
            "expected_energy_reduction": 0.68,
        }
    ]

@pytest.fixture
def mock_training_metrics() -> Dict[str, List[float]]:
    """Mock training metrics for testing."""
    epochs = 50
    return {
        "train_loss": [3.0 - 2.5 * (i / epochs) + 0.1 * np.random.randn() for i in range(epochs)],
        "train_accuracy": [0.1 + 0.85 * (i / epochs) + 0.02 * np.random.randn() for i in range(epochs)],
        "val_loss": [3.2 - 2.3 * (i / epochs) + 0.15 * np.random.randn() for i in range(epochs)],
        "val_accuracy": [0.08 + 0.82 * (i / epochs) + 0.03 * np.random.randn() for i in range(epochs)],
        "learning_rate": [0.01 * (0.95 ** (i // 10)) for i in range(epochs)],
        "hardware_efficiency": [0.3 + 0.5 * (i / epochs) + 0.05 * np.random.randn() for i in range(epochs)],
    }