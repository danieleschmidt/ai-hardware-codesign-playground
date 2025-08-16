# AI Hardware Co-Design Playground - Complete Developer Guide

## Table of Contents

1. [Development Environment Setup](#development-environment-setup)
2. [Project Structure](#project-structure)
3. [Coding Standards](#coding-standards)
4. [Architecture Guidelines](#architecture-guidelines)
5. [Testing Framework](#testing-framework)
6. [Contribution Guidelines](#contribution-guidelines)
7. [API Development](#api-development)
8. [Frontend Development](#frontend-development)
9. [Quality Assurance](#quality-assurance)
10. [Deployment and CI/CD](#deployment-and-cicd)
11. [Performance Guidelines](#performance-guidelines)
12. [Security Guidelines](#security-guidelines)

## Development Environment Setup

### Prerequisites

```bash
# Required tools
- Python 3.11+
- Node.js 18+
- Docker & Docker Compose
- Git
- Make

# Optional but recommended
- VS Code with Python extension
- PyCharm Professional
- Postman for API testing
```

### Development Installation

```bash
# 1. Clone repository with submodules
git clone --recursive https://github.com/your-org/ai-hardware-codesign-playground.git
cd ai-hardware-codesign-playground

# 2. Set up Python environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install development dependencies
pip install -e ".[dev,test]"

# 4. Install pre-commit hooks
pre-commit install

# 5. Set up environment variables
cp .env.example .env
# Edit .env with your configuration

# 6. Initialize database
python scripts/init_db.py

# 7. Verify installation
make test
make lint
```

### IDE Configuration

#### VS Code Setup

**.vscode/settings.json**:
```json
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "python.linting.flake8Enabled": true,
    "python.linting.mypyEnabled": true,
    "python.formatting.provider": "black",
    "python.formatting.blackArgs": ["--line-length", "88"],
    "python.sortImports.args": ["--profile", "black"],
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.organizeImports": true
    },
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": [
        "tests"
    ]
}
```

**.vscode/extensions.json**:
```json
{
    "recommendations": [
        "ms-python.python",
        "ms-python.black-formatter",
        "ms-python.isort",
        "ms-python.pylint",
        "ms-python.mypy-type-checker",
        "ms-vscode.test-adapter-converter",
        "redhat.vscode-yaml",
        "ms-vscode.vscode-json"
    ]
}
```

#### PyCharm Configuration

1. **Import project**: Select existing environment
2. **Configure interpreter**: Point to `venv/bin/python`
3. **Set code style**: Use Black formatter with 88-character line length
4. **Enable inspections**: Type hints, unused imports, PEP 8
5. **Configure test runner**: pytest with coverage

### Docker Development Environment

```bash
# Build development container
docker-compose -f docker-compose.dev.yml build

# Start development environment
docker-compose -f docker-compose.dev.yml up -d

# Access development container
docker-compose -f docker-compose.dev.yml exec app bash

# Run tests in container
docker-compose -f docker-compose.dev.yml exec app pytest

# Stop development environment
docker-compose -f docker-compose.dev.yml down
```

## Project Structure

### Directory Layout

```
ai-hardware-codesign-playground/
├── backend/                    # Backend Python application
│   ├── codesign_playground/   # Main package
│   │   ├── core/              # Core functionality
│   │   │   ├── accelerator.py      # Accelerator design
│   │   │   ├── optimizer.py        # Model optimization
│   │   │   ├── explorer.py         # Design space exploration
│   │   │   ├── workflow.py         # Workflow management
│   │   │   ├── auto_scaling.py     # Auto-scaling system
│   │   │   ├── cache.py            # Caching system
│   │   │   └── performance.py      # Performance monitoring
│   │   ├── utils/             # Utility modules
│   │   │   ├── validation.py       # Input validation
│   │   │   ├── security.py         # Security utilities
│   │   │   ├── monitoring.py       # System monitoring
│   │   │   ├── logging.py          # Logging configuration
│   │   │   └── exceptions.py       # Custom exceptions
│   │   ├── templates/         # Hardware templates
│   │   │   ├── systolic_array.py   # Systolic array template
│   │   │   ├── vector_processor.py # Vector processor template
│   │   │   └── transformer_accelerator.py
│   │   ├── research/          # Research components
│   │   │   └── novel_algorithms.py
│   │   ├── cli.py             # Command-line interface
│   │   ├── server.py          # FastAPI server
│   │   └── worker.py          # Background worker
│   ├── Dockerfile             # Backend container
│   └── requirements.txt       # Python dependencies
├── frontend/                  # Frontend application
│   ├── src/                   # React/TypeScript source
│   ├── public/                # Static assets
│   ├── package.json           # Node.js dependencies
│   └── Dockerfile             # Frontend container
├── docs/                      # Documentation
│   ├── api/                   # API documentation
│   ├── user/                  # User guides
│   ├── developer/             # Developer documentation
│   └── operations/            # Operational guides
├── tests/                     # Test suite
│   ├── unit/                  # Unit tests
│   ├── integration/           # Integration tests
│   ├── e2e/                   # End-to-end tests
│   ├── performance/           # Performance tests
│   └── security/              # Security tests
├── deployment/                # Deployment configurations
│   ├── kubernetes/            # K8s manifests
│   ├── terraform/             # Infrastructure as code
│   └── docker-compose/        # Docker Compose files
├── scripts/                   # Development scripts
│   ├── setup.sh              # Environment setup
│   ├── test.sh               # Test runner
│   ├── lint.sh               # Code linting
│   └── deploy.sh             # Deployment script
├── monitoring/                # Monitoring configuration
│   ├── prometheus/            # Prometheus rules
│   ├── grafana/               # Grafana dashboards
│   └── alerts/                # Alert definitions
├── .github/                   # GitHub workflows
│   └── workflows/             # CI/CD pipelines
├── pyproject.toml             # Python project configuration
├── Makefile                   # Build automation
├── docker-compose.yml         # Local development
├── .gitignore                 # Git ignore rules
├── .pre-commit-config.yaml    # Pre-commit hooks
└── README.md                  # Project overview
```

### Module Organization

#### Core Modules

**accelerator.py**: Accelerator design and profiling
```python
class AcceleratorDesigner:
    """Main class for accelerator design and model profiling."""
    
    def profile_model(self, model_data: Dict) -> ModelProfile:
        """Profile neural network model."""
        
    def design(self, compute_units: int, memory_hierarchy: List[str], 
               dataflow: str) -> AcceleratorConfig:
        """Design hardware accelerator."""
        
    def generate_rtl(self, config: AcceleratorConfig) -> str:
        """Generate RTL code for accelerator."""

@dataclass
class ModelProfile:
    """Model profiling results."""
    peak_gflops: float
    bandwidth_gb_s: float
    operations: Dict[str, int]
    parameters: int
    memory_mb: float
    compute_intensity: float
```

**optimizer.py**: Model and hardware optimization
```python
class ModelOptimizer:
    """Joint model-hardware optimization."""
    
    def co_optimize(self, target_fps: float, power_budget: float) -> OptimizationResult:
        """Co-optimize model and hardware."""
        
    def apply_quantization(self, model: Model, quantization_scheme: Dict) -> Model:
        """Apply quantization to model."""
        
    def optimize_model_for_hardware(self, model: Model, hardware: Hardware) -> Model:
        """Optimize model for specific hardware."""
```

#### Utility Modules

**validation.py**: Input validation and security
```python
class SecurityValidator:
    """Security validation for all inputs."""
    
    @staticmethod
    def validate_input(value: Any, input_type: str) -> bool:
        """Validate input against security threats."""
        
    @staticmethod
    def sanitize_path(path: str) -> str:
        """Sanitize file paths."""
        
    @staticmethod
    def validate_model_data(model_data: Dict) -> ValidationResult:
        """Validate model data structure."""
```

**monitoring.py**: System monitoring and metrics
```python
class SystemMonitor:
    """System monitoring and metrics collection."""
    
    def record_metric(self, name: str, value: float, metric_type: str):
        """Record system metric."""
        
    def get_system_health(self) -> HealthStatus:
        """Get current system health."""
        
    def get_performance_metrics(self) -> PerformanceMetrics:
        """Get performance metrics."""
```

## Coding Standards

### Python Code Style

We follow PEP 8 with specific modifications:

```python
# Line length: 88 characters (Black default)
# String quotes: Double quotes preferred
# Import order: isort with Black profile

# Example of good code style:
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
import logging

from .utils.exceptions import ValidationError
from .utils.validation import validate_inputs


logger = logging.getLogger(__name__)


@dataclass
class AcceleratorConfig:
    """Configuration for hardware accelerator.
    
    Attributes:
        compute_units: Number of processing units
        memory_hierarchy: List of memory levels
        dataflow: Data movement pattern
        precision: Numeric precision (int8, fp16, etc.)
    """
    
    compute_units: int
    memory_hierarchy: List[str]
    dataflow: str
    precision: str = "int8"
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.compute_units <= 0:
            raise ValidationError("compute_units must be positive")


class AcceleratorDesigner:
    """Design hardware accelerators for neural networks.
    
    This class provides methods for analyzing neural network models
    and generating matching hardware accelerator architectures.
    
    Example:
        >>> designer = AcceleratorDesigner()
        >>> profile = designer.profile_model(model_data)
        >>> accelerator = designer.design(
        ...     compute_units=64,
        ...     memory_hierarchy=["sram_64kb", "dram"],
        ...     dataflow="weight_stationary"
        ... )
    """
    
    def __init__(self, cache_enabled: bool = True):
        """Initialize accelerator designer.
        
        Args:
            cache_enabled: Whether to enable result caching
        """
        self.cache_enabled = cache_enabled
        self._cache = {} if cache_enabled else None
        logger.info("AcceleratorDesigner initialized")
    
    @validate_inputs
    def profile_model(self, model_data: Dict[str, Any]) -> ModelProfile:
        """Profile neural network model characteristics.
        
        Args:
            model_data: Model definition with layers and metadata
            
        Returns:
            ModelProfile containing computational analysis
            
        Raises:
            ValidationError: If model_data is invalid
            
        Example:
            >>> model_data = {
            ...     "layers": [...],
            ...     "input_shape": [224, 224, 3],
            ...     "framework": "tensorflow"
            ... }
            >>> profile = designer.profile_model(model_data)
            >>> print(f"Peak GFLOPS: {profile.peak_gflops}")
        """
        # Implementation here
        pass
```

### Code Documentation

#### Docstring Standards

Use Google-style docstrings:

```python
def complex_function(
    param1: int,
    param2: str,
    optional_param: Optional[bool] = None
) -> Tuple[int, str]:
    """Short description of the function.
    
    Longer description that explains the function's behavior,
    algorithms used, and any important implementation details.
    
    Args:
        param1: Description of first parameter
        param2: Description of second parameter
        optional_param: Description of optional parameter.
            Defaults to None.
    
    Returns:
        Tuple containing:
            - int: Description of first return value
            - str: Description of second return value
    
    Raises:
        ValueError: If param1 is negative
        ValidationError: If param2 is empty
    
    Example:
        >>> result = complex_function(42, "hello")
        >>> print(result)
        (42, "hello")
    
    Note:
        This function has side effects on the global cache.
    """
    pass
```

#### Type Hints

Use comprehensive type hints:

```python
from typing import (
    Dict, List, Optional, Union, Tuple, Callable, 
    TypeVar, Generic, Protocol, Any
)
from pathlib import Path

# Type variables
T = TypeVar('T')
ModelType = TypeVar('ModelType', bound='BaseModel')

# Complex types
ConfigDict = Dict[str, Union[str, int, float, List[str]]]
OptimizationCallback = Callable[[int, float], bool]

# Generic classes
class Cache(Generic[T]):
    def get(self, key: str) -> Optional[T]:
        pass
    
    def set(self, key: str, value: T) -> None:
        pass

# Protocols for duck typing
class Optimizable(Protocol):
    def optimize(self, target: float) -> float:
        pass

def optimize_model(model: Optimizable) -> float:
    return model.optimize(1.0)
```

### Error Handling

#### Exception Hierarchy

```python
class CodesignException(Exception):
    """Base exception for codesign playground."""
    
    def __init__(
        self, 
        message: str, 
        details: Optional[Dict[str, Any]] = None,
        suggestions: Optional[List[str]] = None
    ):
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.suggestions = suggestions or []
        self.timestamp = datetime.utcnow()

class ValidationError(CodesignException):
    """Input validation errors."""
    pass

class DesignError(CodesignException):
    """Hardware design errors."""
    pass

class OptimizationError(CodesignException):
    """Optimization process errors."""
    pass

class SecurityError(CodesignException):
    """Security-related errors."""
    pass
```

#### Error Handling Patterns

```python
def safe_operation(data: Dict[str, Any]) -> Result:
    """Example of proper error handling."""
    try:
        # Validate inputs
        if not data:
            raise ValidationError(
                "Input data cannot be empty",
                suggestions=["Provide valid model data"]
            )
        
        # Perform operation
        result = perform_complex_operation(data)
        
        # Validate results
        if not result.is_valid():
            raise DesignError(
                "Generated design is invalid",
                details={"validation_errors": result.errors},
                suggestions=[
                    "Adjust design parameters",
                    "Check input constraints"
                ]
            )
        
        return result
        
    except ValidationError:
        # Re-raise validation errors
        raise
    except DesignError:
        # Re-raise design errors
        raise
    except Exception as e:
        # Wrap unexpected errors
        raise CodesignException(
            f"Unexpected error in safe_operation: {str(e)}",
            details={"original_error": str(e)},
            suggestions=["Check system logs", "Report this issue"]
        ) from e
```

### Logging Standards

```python
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

def example_function(param: str) -> int:
    """Example function with proper logging."""
    logger.info("Starting example_function", extra={
        "param": param,
        "function": "example_function"
    })
    
    try:
        # Log performance metrics
        start_time = time.time()
        
        result = complex_computation(param)
        
        duration = time.time() - start_time
        logger.info("Function completed successfully", extra={
            "duration_seconds": duration,
            "result_size": len(str(result)),
            "function": "example_function"
        })
        
        return result
        
    except Exception as e:
        logger.error("Function failed", extra={
            "error": str(e),
            "error_type": type(e).__name__,
            "param": param,
            "function": "example_function"
        }, exc_info=True)
        raise

# Structured logging for security events
def log_security_event(event_type: str, details: Dict[str, Any]):
    """Log security-related events."""
    logger.warning("Security event detected", extra={
        "event_type": event_type,
        "timestamp": datetime.utcnow().isoformat(),
        "details": details,
        "severity": "HIGH" if event_type in ["malicious_input", "rate_limit_exceeded"] else "MEDIUM"
    })
```

## Architecture Guidelines

### Design Principles

#### 1. Single Responsibility Principle

Each class and function should have one clear responsibility:

```python
# Good: Single responsibility
class ModelProfiler:
    """Analyzes neural network model characteristics."""
    
    def analyze_operations(self, model: Model) -> Dict[str, int]:
        """Count operations by type."""
        pass
    
    def calculate_memory_requirements(self, model: Model) -> float:
        """Calculate memory requirements in MB."""
        pass

class AcceleratorGenerator:
    """Generates hardware accelerator configurations."""
    
    def generate_systolic_array(self, size: Tuple[int, int]) -> SystemArray:
        """Generate systolic array configuration."""
        pass

# Bad: Multiple responsibilities
class ModelAnalyzerAndGenerator:
    """Does everything - violates SRP."""
    pass
```

#### 2. Dependency Injection

Use dependency injection for testability and flexibility:

```python
from abc import ABC, abstractmethod

class CacheInterface(ABC):
    """Abstract cache interface."""
    
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        pass
    
    @abstractmethod
    def set(self, key: str, value: Any, ttl: int = 3600) -> None:
        pass

class RedisCache(CacheInterface):
    """Redis cache implementation."""
    
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
    
    def get(self, key: str) -> Optional[Any]:
        # Redis implementation
        pass

class AcceleratorDesigner:
    """Accelerator designer with injected dependencies."""
    
    def __init__(
        self, 
        cache: CacheInterface,
        profiler: ModelProfiler,
        validator: SecurityValidator
    ):
        self.cache = cache
        self.profiler = profiler
        self.validator = validator
```

#### 3. Factory Pattern for Template Creation

```python
class TemplateFactory:
    """Factory for creating hardware templates."""
    
    _templates = {
        "systolic_array": SystolicArrayTemplate,
        "vector_processor": VectorProcessorTemplate,
        "transformer_accelerator": TransformerAcceleratorTemplate
    }
    
    @classmethod
    def create_template(cls, template_type: str, **kwargs) -> BaseTemplate:
        """Create hardware template by type."""
        if template_type not in cls._templates:
            raise ValueError(f"Unknown template type: {template_type}")
        
        template_class = cls._templates[template_type]
        return template_class(**kwargs)
    
    @classmethod
    def register_template(cls, name: str, template_class: type) -> None:
        """Register new template type."""
        cls._templates[name] = template_class
```

### Async Programming

Use async/await for I/O-bound operations:

```python
import asyncio
from typing import List
import aiohttp

class AsyncModelAnalyzer:
    """Async model analysis with concurrent processing."""
    
    async def analyze_models_concurrent(
        self, 
        models: List[Model]
    ) -> List[ModelProfile]:
        """Analyze multiple models concurrently."""
        tasks = [self._analyze_single_model(model) for model in models]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        profiles = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Failed to analyze model {i}: {result}")
                continue
            profiles.append(result)
        
        return profiles
    
    async def _analyze_single_model(self, model: Model) -> ModelProfile:
        """Analyze single model asynchronously."""
        # Simulate async I/O (e.g., external API call)
        await asyncio.sleep(0.1)
        
        # Perform CPU-intensive work in thread pool
        loop = asyncio.get_event_loop()
        profile = await loop.run_in_executor(
            None, 
            self._cpu_intensive_analysis, 
            model
        )
        
        return profile
    
    def _cpu_intensive_analysis(self, model: Model) -> ModelProfile:
        """CPU-intensive analysis in thread pool."""
        # Actual analysis implementation
        pass
```

### Configuration Management

```python
from pydantic import BaseSettings, Field
from typing import List, Optional

class Settings(BaseSettings):
    """Application settings with validation."""
    
    # Database settings
    database_url: str = Field(..., env="DATABASE_URL")
    database_pool_size: int = Field(10, env="DATABASE_POOL_SIZE")
    
    # Cache settings  
    redis_url: str = Field("redis://localhost:6379", env="REDIS_URL")
    cache_ttl_seconds: int = Field(3600, env="CACHE_TTL_SECONDS")
    
    # API settings
    api_host: str = Field("0.0.0.0", env="API_HOST")
    api_port: int = Field(8000, env="API_PORT")
    api_workers: int = Field(4, env="API_WORKERS")
    
    # Security settings
    secret_key: str = Field(..., env="SECRET_KEY")
    allowed_hosts: List[str] = Field(["localhost"], env="ALLOWED_HOSTS")
    rate_limit_per_minute: int = Field(60, env="RATE_LIMIT_PER_MINUTE")
    
    # Feature flags
    enable_auto_scaling: bool = Field(True, env="ENABLE_AUTO_SCALING")
    enable_security_validation: bool = Field(True, env="ENABLE_SECURITY_VALIDATION")
    
    class Config:
        env_file = ".env"
        case_sensitive = False

# Global settings instance
settings = Settings()
```

## Testing Framework

### Test Structure

```python
# tests/unit/test_accelerator.py
import pytest
from unittest.mock import Mock, patch
from codesign_playground.core.accelerator import AcceleratorDesigner, ModelProfile

class TestAcceleratorDesigner:
    """Test suite for AcceleratorDesigner."""
    
    @pytest.fixture
    def designer(self):
        """Create AcceleratorDesigner instance for testing."""
        return AcceleratorDesigner(cache_enabled=False)
    
    @pytest.fixture
    def sample_model_data(self):
        """Sample model data for testing."""
        return {
            "layers": [
                {
                    "type": "conv2d",
                    "input_shape": [224, 224, 3],
                    "output_shape": [224, 224, 32],
                    "kernel_size": 3,
                    "stride": 1,
                    "parameters": 864
                }
            ],
            "input_shape": [224, 224, 3],
            "framework": "tensorflow"
        }
    
    def test_profile_model_success(self, designer, sample_model_data):
        """Test successful model profiling."""
        profile = designer.profile_model(sample_model_data)
        
        assert isinstance(profile, ModelProfile)
        assert profile.peak_gflops > 0
        assert profile.parameters > 0
        assert profile.memory_mb > 0
    
    def test_profile_model_invalid_input(self, designer):
        """Test model profiling with invalid input."""
        with pytest.raises(ValidationError):
            designer.profile_model({})  # Empty model data
    
    def test_profile_model_malicious_input(self, designer):
        """Test security validation with malicious input."""
        malicious_data = {
            "layers": ["<script>alert('xss')</script>"],
            "input_shape": "../../etc/passwd",
            "framework": "evil"
        }
        
        with pytest.raises(SecurityError):
            designer.profile_model(malicious_data)
    
    @patch('codesign_playground.core.accelerator.logger')
    def test_profile_model_logging(self, mock_logger, designer, sample_model_data):
        """Test that profiling logs appropriate messages."""
        designer.profile_model(sample_model_data)
        
        assert mock_logger.info.called
        assert "profiling" in mock_logger.info.call_args[0][0].lower()

# Integration tests
class TestAcceleratorIntegration:
    """Integration tests for accelerator design workflow."""
    
    @pytest.fixture
    def full_workflow_setup(self):
        """Set up complete workflow for integration testing."""
        return {
            "designer": AcceleratorDesigner(),
            "optimizer": ModelOptimizer(),
            "explorer": DesignSpaceExplorer()
        }
    
    def test_end_to_end_workflow(self, full_workflow_setup, sample_model_data):
        """Test complete design workflow."""
        designer = full_workflow_setup["designer"]
        optimizer = full_workflow_setup["optimizer"]
        
        # Profile model
        profile = designer.profile_model(sample_model_data)
        assert profile is not None
        
        # Design accelerator
        accelerator = designer.design(
            compute_units=64,
            memory_hierarchy=["sram_64kb", "dram"],
            dataflow="weight_stationary"
        )
        assert accelerator is not None
        
        # Optimize design
        optimization_result = optimizer.co_optimize(
            target_fps=30,
            power_budget_watts=5.0
        )
        assert optimization_result.converged
```

### Test Configuration

**pytest.ini**:
```ini
[tool:pytest]
testpaths = tests
python_files = test_*.py *_test.py
python_classes = Test*
python_functions = test_*
addopts = 
    --strict-markers
    --strict-config
    --verbose
    --cov=codesign_playground
    --cov-report=term-missing
    --cov-report=html:htmlcov
    --cov-report=xml
markers =
    unit: Unit tests
    integration: Integration tests
    e2e: End-to-end tests
    slow: Slow running tests
    security: Security tests
    performance: Performance tests
```

**conftest.py** (shared fixtures):
```python
import pytest
import asyncio
from unittest.mock import Mock
from codesign_playground import AcceleratorDesigner

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def mock_cache():
    """Mock cache for testing."""
    cache = Mock()
    cache.get.return_value = None
    cache.set.return_value = None
    return cache

@pytest.fixture
def test_model_data():
    """Standard test model data."""
    return {
        "layers": [
            {
                "type": "conv2d",
                "input_shape": [224, 224, 3],
                "output_shape": [224, 224, 32],
                "kernel_size": 3,
                "parameters": 864
            }
        ],
        "input_shape": [224, 224, 3],
        "framework": "tensorflow"
    }

@pytest.fixture
def designer_with_mock_cache(mock_cache):
    """AcceleratorDesigner with mocked cache."""
    return AcceleratorDesigner(cache=mock_cache)
```

### Performance Testing

```python
# tests/performance/test_performance.py
import pytest
import time
import psutil
from concurrent.futures import ThreadPoolExecutor
from codesign_playground import AcceleratorDesigner

class TestPerformance:
    """Performance tests for core functionality."""
    
    @pytest.mark.performance
    def test_model_profiling_performance(self, test_model_data):
        """Test model profiling performance."""
        designer = AcceleratorDesigner()
        
        start_time = time.time()
        profile = designer.profile_model(test_model_data)
        duration = time.time() - start_time
        
        # Performance requirements
        assert duration < 1.0, f"Profiling took {duration:.2f}s, should be < 1.0s"
        assert profile is not None
    
    @pytest.mark.performance
    def test_concurrent_profiling_performance(self, test_model_data):
        """Test concurrent model profiling performance."""
        designer = AcceleratorDesigner()
        
        def profile_model():
            return designer.profile_model(test_model_data)
        
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(profile_model) for _ in range(10)]
            results = [f.result() for f in futures]
        duration = time.time() - start_time
        
        # Should complete 10 concurrent profiles in reasonable time
        assert duration < 5.0, f"Concurrent profiling took {duration:.2f}s"
        assert len(results) == 10
        assert all(r is not None for r in results)
    
    @pytest.mark.performance
    def test_memory_usage(self, test_model_data):
        """Test memory usage during operations."""
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        designer = AcceleratorDesigner()
        
        # Perform multiple operations
        for _ in range(100):
            profile = designer.profile_model(test_model_data)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable
        assert memory_increase < 100, f"Memory increased by {memory_increase:.1f}MB"
```

### Security Testing

```python
# tests/security/test_security.py
import pytest
from codesign_playground.utils.validation import SecurityValidator
from codesign_playground.utils.exceptions import SecurityError

class TestSecurityValidation:
    """Security validation tests."""
    
    @pytest.fixture
    def validator(self):
        return SecurityValidator()
    
    @pytest.mark.security
    def test_xss_prevention(self, validator):
        """Test XSS attack prevention."""
        malicious_inputs = [
            "<script>alert('xss')</script>",
            "javascript:alert(1)",
            "<img src=x onerror=alert(1)>",
            "<iframe src='javascript:alert(1)'></iframe>"
        ]
        
        for malicious_input in malicious_inputs:
            assert not validator.validate_input(malicious_input, "string")
    
    @pytest.mark.security
    def test_path_traversal_prevention(self, validator):
        """Test path traversal attack prevention."""
        malicious_paths = [
            "../../etc/passwd",
            "..\\..\\windows\\system32",
            "/etc/shadow",
            "../../../root/.ssh/id_rsa"
        ]
        
        for malicious_path in malicious_paths:
            with pytest.raises(SecurityError):
                validator.sanitize_path(malicious_path)
    
    @pytest.mark.security
    def test_code_injection_prevention(self, validator):
        """Test code injection prevention."""
        malicious_code = [
            "eval('malicious code')",
            "__import__('os').system('rm -rf /')",
            "exec('import os; os.system(\"ls\")')",
            "compile('print(\"hacked\")', '<string>', 'exec')"
        ]
        
        for code in malicious_code:
            assert not validator.validate_input(code, "string")
```

## Contribution Guidelines

### Git Workflow

#### Branch Naming

- **Feature branches**: `feature/description-of-feature`
- **Bug fixes**: `bugfix/description-of-bug`
- **Hot fixes**: `hotfix/critical-issue`
- **Documentation**: `docs/documentation-update`
- **Refactoring**: `refactor/component-name`

#### Commit Messages

Follow Conventional Commits specification:

```
type(scope): description

body (optional)

footer (optional)
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Adding tests
- `chore`: Maintenance tasks

**Examples**:
```
feat(accelerator): add systolic array template

Add new systolic array hardware template with configurable
dimensions and data width support.

Closes #123
```

```
fix(security): prevent path traversal in file uploads

Add path sanitization to prevent directory traversal attacks
in model file upload functionality.

BREAKING CHANGE: File upload API now validates paths
```

#### Pull Request Process

1. **Create feature branch**: `git checkout -b feature/new-feature`
2. **Make changes**: Implement feature with tests
3. **Run quality checks**: `make test lint`
4. **Commit changes**: Use conventional commit format
5. **Push branch**: `git push origin feature/new-feature`
6. **Create PR**: Use PR template
7. **Review process**: Address feedback
8. **Merge**: Squash and merge after approval

#### Pull Request Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] This change requires a documentation update

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Manual testing completed
- [ ] Performance impact assessed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Code is commented appropriately
- [ ] Documentation updated
- [ ] No new warnings introduced

## Related Issues
Closes #(issue number)
```

### Code Review Guidelines

#### For Authors

1. **Self-review first**: Review your own PR before requesting review
2. **Write clear descriptions**: Explain what and why, not just what
3. **Keep PRs focused**: One feature/fix per PR
4. **Add tests**: Include comprehensive test coverage
5. **Update documentation**: Keep docs in sync with code changes

#### For Reviewers

1. **Be constructive**: Provide specific, actionable feedback
2. **Check functionality**: Verify the code works as intended
3. **Review tests**: Ensure adequate test coverage
4. **Security review**: Look for security vulnerabilities
5. **Performance review**: Consider performance implications

#### Review Checklist

**Functionality**:
- [ ] Code works as intended
- [ ] Edge cases are handled
- [ ] Error handling is appropriate
- [ ] No obvious bugs

**Code Quality**:
- [ ] Code is readable and maintainable
- [ ] Functions are appropriately sized
- [ ] Variable names are descriptive
- [ ] Comments explain complex logic

**Testing**:
- [ ] Unit tests cover new functionality
- [ ] Integration tests verify end-to-end behavior
- [ ] Test cases cover edge cases
- [ ] Tests are reliable and not flaky

**Security**:
- [ ] Input validation is present
- [ ] No sensitive data exposure
- [ ] Authentication/authorization checks
- [ ] SQL injection prevention

**Performance**:
- [ ] No obvious performance issues
- [ ] Efficient algorithms used
- [ ] Memory usage is reasonable
- [ ] Database queries are optimized

## API Development

### FastAPI Best Practices

#### Endpoint Definition

```python
from fastapi import FastAPI, HTTPException, Depends, status
from pydantic import BaseModel, Field
from typing import List, Optional

app = FastAPI()

class ModelProfileRequest(BaseModel):
    """Request model for model profiling."""
    
    model_data: Dict[str, Any] = Field(
        ..., 
        description="Neural network model definition",
        example={
            "layers": [...],
            "input_shape": [224, 224, 3],
            "framework": "tensorflow"
        }
    )
    analysis_options: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional analysis options"
    )

class ModelProfileResponse(BaseModel):
    """Response model for model profiling."""
    
    profile: ModelProfile = Field(..., description="Model analysis results")
    metadata: Dict[str, Any] = Field(..., description="Request metadata")

@app.post(
    "/api/v1/profile",
    response_model=ModelProfileResponse,
    status_code=status.HTTP_200_OK,
    summary="Profile Neural Network Model",
    description="Analyze neural network computational characteristics",
    tags=["Model Analysis"]
)
async def profile_model(
    request: ModelProfileRequest,
    designer: AcceleratorDesigner = Depends(get_designer)
) -> ModelProfileResponse:
    """Profile neural network model characteristics."""
    try:
        profile = designer.profile_model(request.model_data)
        
        return ModelProfileResponse(
            profile=profile,
            metadata={
                "request_id": generate_request_id(),
                "timestamp": datetime.utcnow(),
                "processing_time_ms": 0  # TODO: Add timing
            }
        )
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "error": "Validation failed",
                "message": str(e),
                "suggestions": e.suggestions
            }
        )
    except Exception as e:
        logger.exception("Unexpected error in profile_model")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )
```

#### Dependency Injection

```python
from fastapi import Depends
from functools import lru_cache

@lru_cache()
def get_settings() -> Settings:
    """Get application settings."""
    return Settings()

def get_cache(settings: Settings = Depends(get_settings)) -> CacheInterface:
    """Get cache instance."""
    return RedisCache(settings.redis_url)

def get_designer(cache: CacheInterface = Depends(get_cache)) -> AcceleratorDesigner:
    """Get AcceleratorDesigner instance."""
    return AcceleratorDesigner(cache=cache)

# Usage in endpoints
@app.post("/api/v1/design")
async def design_accelerator(
    request: DesignRequest,
    designer: AcceleratorDesigner = Depends(get_designer)
):
    # Use designer instance
    pass
```

#### Error Handling

```python
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse

@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    """Handle validation errors."""
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "type": "validation_error",
                "message": exc.message,
                "details": exc.details,
                "suggestions": exc.suggestions,
                "timestamp": datetime.utcnow().isoformat()
            }
        }
    )

@app.exception_handler(SecurityError)
async def security_exception_handler(request: Request, exc: SecurityError):
    """Handle security errors."""
    # Log security incident
    logger.warning("Security violation", extra={
        "client_ip": request.client.host,
        "user_agent": request.headers.get("user-agent"),
        "error": exc.message
    })
    
    return JSONResponse(
        status_code=403,
        content={
            "error": {
                "type": "security_error",
                "message": "Request blocked for security reasons",
                "timestamp": datetime.utcnow().isoformat()
            }
        }
    )
```

### API Versioning

```python
from fastapi import APIRouter

# Version 1 router
v1_router = APIRouter(prefix="/api/v1", tags=["v1"])

@v1_router.post("/profile")
async def profile_model_v1(request: ModelProfileRequestV1):
    # V1 implementation
    pass

# Version 2 router
v2_router = APIRouter(prefix="/api/v2", tags=["v2"])

@v2_router.post("/profile")
async def profile_model_v2(request: ModelProfileRequestV2):
    # V2 implementation with new features
    pass

# Include routers in main app
app.include_router(v1_router)
app.include_router(v2_router)
```

## Quality Assurance

### Pre-commit Hooks

**.pre-commit-config.yaml**:
```yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-merge-conflict
      - id: check-yaml
      - id: check-json
      - id: check-added-large-files
      - id: check-case-conflict
      - id: check-toml
      - id: debug-statements

  - repo: https://github.com/psf/black
    rev: 23.1.0
    hooks:
      - id: black
        language_version: python3.11

  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
        args: ["--profile", "black"]

  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
        additional_dependencies: [flake8-docstrings, flake8-type-checking]

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.0.1
    hooks:
      - id: mypy
        additional_dependencies: [types-requests, types-redis]

  - repo: https://github.com/pycqa/bandit
    rev: 1.7.4
    hooks:
      - id: bandit
        args: ["-c", "pyproject.toml"]

  - repo: https://github.com/Lucas-C/pre-commit-hooks-safety
    rev: v1.3.2
    hooks:
      - id: python-safety-dependencies-check
```

### Makefile for Development

```makefile
.PHONY: help install test lint format type-check security-check clean build

help:  ## Show help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install development dependencies
	pip install -e ".[dev,test]"
	pre-commit install

test:  ## Run tests
	pytest tests/ -v --cov=codesign_playground --cov-report=term-missing

test-quick:  ## Run quick tests (unit only)
	pytest tests/unit/ -v

test-integration:  ## Run integration tests
	pytest tests/integration/ -v

test-e2e:  ## Run end-to-end tests
	pytest tests/e2e/ -v

test-security:  ## Run security tests
	pytest tests/security/ -v

test-performance:  ## Run performance tests
	pytest tests/performance/ -v -m performance

lint:  ## Run linting
	flake8 backend/
	mypy backend/
	bandit -r backend/ -c pyproject.toml

format:  ## Format code
	black backend/
	isort backend/

type-check:  ## Run type checking
	mypy backend/

security-check:  ## Run security checks
	bandit -r backend/ -c pyproject.toml
	safety check

clean:  ## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf htmlcov/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

build:  ## Build package
	python -m build

docker-build:  ## Build Docker image
	docker build -t codesign-playground:latest .

docker-test:  ## Run tests in Docker
	docker run --rm codesign-playground:latest pytest

ci:  ## Run full CI pipeline locally
	make lint
	make type-check
	make security-check
	make test
```

### Continuous Integration

**.github/workflows/ci.yml**:
```yaml
name: CI

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.9, 3.10, 3.11]

    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Cache dependencies
      uses: actions/cache@v3
      with:
        path: ~/.cache/pip
        key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements*.txt') }}
        restore-keys: |
          ${{ runner.os }}-pip-
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev,test]"
    
    - name: Lint with flake8
      run: |
        flake8 backend/
    
    - name: Type check with mypy
      run: |
        mypy backend/
    
    - name: Security check with bandit
      run: |
        bandit -r backend/ -c pyproject.toml
    
    - name: Test with pytest
      run: |
        pytest tests/ --cov=codesign_playground --cov-report=xml
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        flags: unittests
        name: codecov-umbrella

  build:
    runs-on: ubuntu-latest
    needs: test
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: 3.11
    
    - name: Build package
      run: |
        python -m pip install --upgrade pip build
        python -m build
    
    - name: Test package installation
      run: |
        pip install dist/*.whl
        python -c "import codesign_playground; print('Package installed successfully')"

  docker:
    runs-on: ubuntu-latest
    needs: test
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v2
    
    - name: Build Docker image
      uses: docker/build-push-action@v4
      with:
        context: .
        file: ./Dockerfile
        push: false
        tags: codesign-playground:latest
        cache-from: type=gha
        cache-to: type=gha,mode=max
    
    - name: Test Docker image
      run: |
        docker run --rm codesign-playground:latest python -c "import codesign_playground; print('Docker image works')"
```

This comprehensive developer guide provides the foundation for maintaining high code quality, security, and performance standards while enabling efficient collaboration and development workflows.