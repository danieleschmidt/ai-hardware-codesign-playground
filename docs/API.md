# AI Hardware Co-Design Playground API Documentation

## Overview

The AI Hardware Co-Design Playground provides a comprehensive REST API for neural network and hardware accelerator co-optimization. This API enables developers to profile models, design custom accelerators, and explore design spaces programmatically.

## Base URL

```
https://api.codesign-playground.com/v1
```

For local development:
```
http://localhost:8000/v1
```

## Authentication

The API uses JWT (JSON Web Tokens) for authentication. Include your token in the Authorization header:

```http
Authorization: Bearer <your-jwt-token>
```

### Get Access Token

```http
POST /auth/token
Content-Type: application/json

{
  "username": "your-username",
  "password": "your-password"
}
```

Response:
```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "token_type": "bearer",
  "expires_in": 1800
}
```

## Core Endpoints

### Model Profiling

#### Profile a Neural Network Model

Analyze the computational characteristics of a neural network model.

```http
POST /models/profile
Content-Type: application/json
Authorization: Bearer <token>

{
  "model_info": {
    "name": "resnet18",
    "framework": "pytorch",
    "path": "/path/to/model.pt"
  },
  "input_shape": [224, 224, 3],
  "batch_size": 1
}
```

Response:
```json
{
  "profile_id": "prof_123456",
  "model_profile": {
    "peak_gflops": 1.82,
    "bandwidth_gb_s": 4.51,
    "operations": {
      "conv2d": 1814073344,
      "batch_norm": 4718592,
      "relu": 4718592,
      "adaptive_avg_pool2d": 512,
      "linear": 1049088
    },
    "parameters": 11689512,
    "memory_mb": 44.59,
    "compute_intensity": 0.403,
    "layer_types": ["conv2d", "batch_norm", "relu", "linear"],
    "model_size_mb": 44.59
  },
  "profiling_time_ms": 245
}
```

#### Get Model Profile

```http
GET /models/profile/{profile_id}
Authorization: Bearer <token>
```

#### List Model Profiles

```http
GET /models/profiles?limit=20&offset=0
Authorization: Bearer <token>
```

### Accelerator Design

#### Design Hardware Accelerator

Create a custom hardware accelerator with specified parameters.

```http
POST /accelerators/design
Content-Type: application/json
Authorization: Bearer <token>

{
  "compute_units": 64,
  "memory_hierarchy": ["sram_64kb", "dram"],
  "dataflow": "weight_stationary",
  "frequency_mhz": 200.0,
  "data_width": 8,
  "precision": "int8",
  "power_budget_w": 5.0,
  "area_budget_mm2": 100.0
}
```

Response:
```json
{
  "accelerator_id": "acc_789012",
  "accelerator": {
    "compute_units": 64,
    "memory_hierarchy": ["sram_64kb", "dram"],
    "dataflow": "weight_stationary",
    "frequency_mhz": 200.0,
    "data_width": 8,
    "precision": "int8",
    "power_budget_w": 5.0,
    "area_budget_mm2": 100.0,
    "performance_model": {
      "throughput_ops_s": 819200000,
      "latency_cycles": 78125,
      "latency_ms": 0.391,
      "power_w": 4.92,
      "efficiency_ops_w": 166504065,
      "area_mm2": 89.6
    },
    "resource_estimates": {
      "luts": 45678,
      "dsp": 64,
      "bram": 32,
      "registers": 123456
    }
  },
  "design_time_ms": 1205
}
```

#### Optimize Accelerator for Model

Design an accelerator optimized for a specific model profile.

```http
POST /accelerators/optimize
Content-Type: application/json
Authorization: Bearer <token>

{
  "model_profile_id": "prof_123456",
  "constraints": {
    "target_fps": 30.0,
    "power_budget": 5.0,
    "area_budget": 100.0,
    "latency_ms": 33.3
  },
  "optimization_objectives": ["performance", "power", "area"]
}
```

#### Generate RTL Code

Generate hardware description language (HDL) code for an accelerator.

```http
POST /accelerators/{accelerator_id}/rtl
Content-Type: application/json
Authorization: Bearer <token>

{
  "target_language": "verilog",
  "synthesis_tool": "vivado",
  "include_testbench": true
}
```

Response:
```json
{
  "rtl_id": "rtl_345678",
  "files": {
    "accelerator.v": "module accelerator(\n  input clk,\n  input rst,\n  // ... RTL code",
    "testbench.sv": "module tb_accelerator();\n  // ... testbench code",
    "constraints.xdc": "# Timing constraints\n..."
  },
  "synthesis_report": {
    "resource_utilization": {
      "luts": 45678,
      "dsp": 64,
      "bram": 32
    },
    "timing": {
      "max_frequency_mhz": 250.0,
      "critical_path_ns": 4.0
    }
  }
}
```

### Design Space Exploration

#### Explore Design Space

Explore multiple accelerator designs across a defined parameter space.

```http
POST /exploration/pareto-frontier
Content-Type: application/json
Authorization: Bearer <token>

{
  "design_space": {
    "compute_units": [16, 32, 64, 128],
    "memory_hierarchy": [
      ["sram_32kb", "dram"],
      ["sram_64kb", "dram"],
      ["sram_128kb", "dram"]
    ],
    "dataflow": ["weight_stationary", "output_stationary"],
    "frequency_mhz": [200.0, 400.0, 600.0]
  },
  "target_model_profile_id": "prof_123456",
  "objectives": ["performance", "power", "area"],
  "max_samples": 50,
  "exploration_method": "pareto_optimal"
}
```

Response:
```json
{
  "exploration_id": "exp_456789",
  "pareto_designs": [
    {
      "design_point_id": "dp_001",
      "accelerator": { /* accelerator config */ },
      "metrics": {
        "performance": 0.95,
        "power": 4.2,
        "area": 78.5,
        "latency_ms": 12.5
      },
      "pareto_rank": 1
    }
    // ... more designs
  ],
  "exploration_stats": {
    "total_designs_evaluated": 48,
    "pareto_optimal_count": 12,
    "exploration_time_ms": 15420
  }
}
```

### Model Optimization

#### Co-optimize Model and Accelerator

Jointly optimize a neural network model and its target accelerator.

```http
POST /optimization/co-optimize
Content-Type: application/json
Authorization: Bearer <token>

{
  "model_profile_id": "prof_123456",
  "initial_accelerator_id": "acc_789012",
  "optimization_config": {
    "objectives": ["accuracy", "latency", "power"],
    "constraints": {
      "min_accuracy": 0.95,
      "max_latency_ms": 20.0,
      "max_power_w": 8.0
    },
    "max_iterations": 100,
    "convergence_threshold": 0.001
  }
}
```

Response:
```json
{
  "optimization_id": "opt_567890",
  "results": {
    "optimized_model": {
      "quantization": "int8",
      "pruning_ratio": 0.15,
      "accuracy": 0.967
    },
    "optimized_accelerator": {
      "accelerator_id": "acc_new_123",
      "improvements": {
        "latency_reduction": 0.35,
        "power_reduction": 0.22
      }
    },
    "final_metrics": {
      "accuracy": 0.967,
      "latency_ms": 14.2,
      "power_w": 6.1,
      "area_mm2": 92.3
    },
    "optimization_history": [
      {
        "iteration": 1,
        "objectives": [0.95, 18.5, 7.2],
        "pareto_rank": 3
      }
      // ... more iterations
    ]
  },
  "convergence_info": {
    "converged": true,
    "iterations": 47,
    "optimization_time_ms": 45230
  }
}
```

## Utility Endpoints

### Health Check

```http
GET /health
```

Response:
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "version": "1.0.0",
  "services": {
    "database": "healthy",
    "cache": "healthy",
    "worker": "healthy"
  }
}
```

### System Metrics

```http
GET /metrics
Authorization: Bearer <token>
```

Response:
```json
{
  "system": {
    "cpu_usage_percent": 25.4,
    "memory_usage_percent": 67.8,
    "disk_usage_percent": 45.2
  },
  "application": {
    "total_requests": 15420,
    "active_sessions": 23,
    "cache_hit_rate": 0.87,
    "average_response_time_ms": 245
  },
  "processing": {
    "models_profiled": 156,
    "accelerators_designed": 342,
    "optimizations_completed": 78
  }
}
```

## Error Handling

The API uses standard HTTP status codes and returns detailed error information:

### Error Response Format

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid input parameters",
    "details": {
      "field": "compute_units",
      "reason": "must be positive integer"
    },
    "request_id": "req_123456789"
  }
}
```

### Common Error Codes

| HTTP Status | Error Code | Description |
|------------|------------|-------------|
| 400 | VALIDATION_ERROR | Invalid request parameters |
| 401 | UNAUTHORIZED | Missing or invalid authentication |
| 403 | FORBIDDEN | Insufficient permissions |
| 404 | NOT_FOUND | Resource not found |
| 409 | CONFLICT | Resource already exists |
| 422 | UNPROCESSABLE_ENTITY | Request cannot be processed |
| 429 | RATE_LIMIT_EXCEEDED | Too many requests |
| 500 | INTERNAL_ERROR | Server internal error |

## Rate Limiting

API requests are rate limited to ensure fair usage:

- **Standard users**: 100 requests per minute
- **Premium users**: 1000 requests per minute
- **Enterprise users**: 10000 requests per minute

Rate limit information is included in response headers:

```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1642251600
```

## WebSocket API

For real-time updates during long-running operations:

```javascript
const ws = new WebSocket('ws://localhost:8000/ws');

ws.onmessage = function(event) {
  const update = JSON.parse(event.data);
  console.log('Progress:', update.progress);
};

// Start optimization
ws.send(JSON.stringify({
  type: 'start_optimization',
  optimization_id: 'opt_567890'
}));
```

## SDKs and Libraries

Official SDKs are available for popular programming languages:

- **Python**: `pip install codesign-playground-python`
- **JavaScript/Node.js**: `npm install codesign-playground-js`
- **Java**: Maven/Gradle dependencies available
- **Go**: `go get github.com/terragon-labs/codesign-go`

## Examples

### Python SDK Example

```python
from codesign_playground import CodesignClient

client = CodesignClient(
    base_url="https://api.codesign-playground.com/v1",
    api_key="your-api-key"
)

# Profile a model
profile = client.models.profile(
    model_info={
        "name": "resnet18",
        "framework": "pytorch",
        "path": "model.pt"
    },
    input_shape=[224, 224, 3]
)

# Design an accelerator
accelerator = client.accelerators.design(
    compute_units=64,
    dataflow="weight_stationary"
)

# Optimize for the model
optimized = client.accelerators.optimize_for_model(
    model_profile_id=profile.id,
    constraints={"power_budget": 5.0}
)

print(f"Optimized accelerator: {optimized.performance_model}")
```

## Support

For API support and questions:

- **Documentation**: https://docs.codesign-playground.com
- **Community**: https://community.codesign-playground.com  
- **Issues**: https://github.com/terragon-labs/codesign-playground/issues
- **Email**: api-support@terragon-labs.com