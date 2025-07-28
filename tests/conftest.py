"""Pytest configuration and shared fixtures."""

import asyncio
import os
import tempfile
from typing import AsyncGenerator, Generator
from unittest.mock import Mock, patch

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Test environment setup
os.environ["ENVIRONMENT"] = "test"
os.environ["DATABASE_URL"] = "sqlite:///:memory:"
os.environ["REDIS_URL"] = "redis://localhost:6379/15"
os.environ["CELERY_TASK_ALWAYS_EAGER"] = "true"
os.environ["LOG_LEVEL"] = "WARNING"
os.environ["TESTING"] = "true"


# Pytest configuration
pytest_plugins = [
    "pytest_asyncio",
    "pytest_mock",
    "pytest_cov",
    "pytest_benchmark",
    "pytest_xdist",
    "pytest_timeout",
]


# Async test configuration
@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# Database fixtures
@pytest.fixture(scope="session")
def test_engine():
    """Create test database engine."""
    from codesign_playground.database import Base
    
    engine = create_engine(
        "sqlite:///:memory:",
        echo=False,
        connect_args={"check_same_thread": False}
    )
    
    # Create all tables
    Base.metadata.create_all(bind=engine)
    
    yield engine
    
    # Cleanup
    Base.metadata.drop_all(bind=engine)
    engine.dispose()


@pytest.fixture
def db_session(test_engine):
    """Create database session for tests."""
    SessionLocal = sessionmaker(bind=test_engine)
    session = SessionLocal()
    
    try:
        yield session
    finally:
        session.rollback()
        session.close()


# API client fixtures
@pytest.fixture
def api_client():
    """Create FastAPI test client."""
    from codesign_playground.main import app
    
    with TestClient(app) as client:
        yield client


@pytest.fixture
async def async_api_client():
    """Create async FastAPI test client."""
    from httpx import AsyncClient
    from codesign_playground.main import app
    
    async with AsyncClient(app=app, base_url="http://testserver") as client:
        yield client


# Authentication fixtures
@pytest.fixture
def auth_headers():
    """Create authentication headers for API tests."""
    return {
        "Authorization": "Bearer test-token",
        "Content-Type": "application/json"
    }


@pytest.fixture
def mock_auth_user():
    """Mock authenticated user."""
    user = Mock()
    user.id = "test-user-123"
    user.email = "test@example.com"
    user.is_active = True
    user.is_admin = False
    return user


# File system fixtures
@pytest.fixture
def temp_dir():
    """Create temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir


@pytest.fixture
def temp_file():
    """Create temporary file for tests."""
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        yield tmp_file.name
    os.unlink(tmp_file.name)


# Model fixtures
@pytest.fixture
def sample_onnx_model():
    """Create sample ONNX model for testing."""
    import onnx
    import onnx.helper as helper
    import numpy as np
    
    # Create simple linear model
    input_tensor = helper.make_tensor_value_info(
        "input", onnx.TensorProto.FLOAT, [1, 784]
    )
    output_tensor = helper.make_tensor_value_info(
        "output", onnx.TensorProto.FLOAT, [1, 10]
    )
    
    # Create weight tensor
    weight = helper.make_tensor(
        "weight",
        onnx.TensorProto.FLOAT,
        [784, 10],
        np.random.randn(784, 10).astype(np.float32).tolist()
    )
    
    # Create MatMul node
    matmul_node = helper.make_node(
        "MatMul",
        inputs=["input", "weight"],
        outputs=["output"]
    )
    
    # Create graph
    graph = helper.make_graph(
        [matmul_node],
        "simple_linear",
        [input_tensor],
        [output_tensor],
        [weight]
    )
    
    # Create model
    model = helper.make_model(graph)
    return model


@pytest.fixture
def sample_pytorch_model():
    """Create sample PyTorch model for testing."""
    import torch
    import torch.nn as nn
    
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(784, 10)
        
        def forward(self, x):
            return self.linear(x)
    
    model = SimpleModel()
    model.eval()
    return model


# Hardware design fixtures
@pytest.fixture
def sample_systolic_config():
    """Sample systolic array configuration."""
    return {
        "template": "systolic_array",
        "rows": 16,
        "cols": 16,
        "data_width": 8,
        "accumulator_width": 32,
        "frequency_mhz": 200
    }


@pytest.fixture
def sample_vector_processor_config():
    """Sample vector processor configuration."""
    return {
        "template": "vector_processor",
        "vector_length": 512,
        "num_lanes": 8,
        "supported_ops": ["add", "mul", "mac", "relu"],
        "frequency_mhz": 400
    }


# Optimization fixtures
@pytest.fixture
def sample_design_space():
    """Sample design space for optimization."""
    return {
        "compute_units": [16, 32, 64, 128],
        "memory_size_kb": [32, 64, 128, 256],
        "frequency_mhz": [100, 200, 400],
        "dataflow": ["weight_stationary", "output_stationary"],
        "precision": ["int8", "fp16", "mixed"]
    }


@pytest.fixture
def sample_optimization_objectives():
    """Sample optimization objectives."""
    return {
        "latency": {"target": 10.0, "weight": 0.4},  # ms
        "power": {"target": 5.0, "weight": 0.3},     # W
        "area": {"target": 10.0, "weight": 0.3}      # mm²
    }


# Mock external services
@pytest.fixture
def mock_tvm_compiler():
    """Mock TVM compiler for testing."""
    with patch("codesign_playground.compilers.tvm.TVMCompiler") as mock:
        mock_instance = Mock()
        mock_instance.compile.return_value = {
            "success": True,
            "optimized_graph": "mocked_graph",
            "performance_estimate": {"latency_ms": 5.0, "throughput_ops": 1000}
        }
        mock.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_verilator():
    """Mock Verilator simulator for testing."""
    with patch("codesign_playground.simulation.verilator.VerilatorSim") as mock:
        mock_instance = Mock()
        mock_instance.run_simulation.return_value = {
            "cycles": 1000,
            "success": True,
            "performance_metrics": {
                "latency_cycles": 1000,
                "throughput_ops_per_cycle": 1.0,
                "utilization_percent": 85.0
            }
        }
        mock.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_cloud_storage():
    """Mock cloud storage for testing."""
    with patch("codesign_playground.storage.CloudStorage") as mock:
        mock_instance = Mock()
        mock_instance.upload.return_value = "https://storage.example.com/file.bin"
        mock_instance.download.return_value = b"mocked_file_content"
        mock.return_value = mock_instance
        yield mock_instance


# Performance testing fixtures
@pytest.fixture
def performance_baseline():
    """Performance baseline metrics for comparison."""
    return {
        "model_analysis_time_ms": 100,
        "hardware_generation_time_ms": 500,
        "optimization_time_s": 30,
        "simulation_time_s": 60,
        "memory_usage_mb": 512
    }


@pytest.fixture
def benchmark_config():
    """Configuration for benchmark tests."""
    return {
        "rounds": 5,
        "warmup_rounds": 2,
        "timer": "time.perf_counter",
        "disable_gc": True
    }


# Data generation fixtures
@pytest.fixture
def random_seed():
    """Set random seed for reproducible tests."""
    import random
    import numpy as np
    import torch
    
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return seed


@pytest.fixture
def model_factory():
    """Factory for creating test models."""
    class ModelFactory:
        @staticmethod
        def create_cnn(input_shape=(1, 3, 224, 224), num_classes=10):
            """Create CNN model for testing."""
            import torch.nn as nn
            
            class TestCNN(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
                    self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
                    self.pool = nn.MaxPool2d(2, 2)
                    self.fc1 = nn.Linear(64 * 56 * 56, 128)
                    self.fc2 = nn.Linear(128, num_classes)
                    
                def forward(self, x):
                    x = self.pool(torch.relu(self.conv1(x)))
                    x = self.pool(torch.relu(self.conv2(x)))
                    x = x.view(-1, 64 * 56 * 56)
                    x = torch.relu(self.fc1(x))
                    x = self.fc2(x)
                    return x
            
            return TestCNN()
        
        @staticmethod
        def create_transformer(vocab_size=10000, d_model=512, nhead=8):
            """Create transformer model for testing."""
            import torch.nn as nn
            
            class TestTransformer(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.embedding = nn.Embedding(vocab_size, d_model)
                    self.transformer = nn.TransformerEncoder(
                        nn.TransformerEncoderLayer(d_model, nhead),
                        num_layers=6
                    )
                    self.classifier = nn.Linear(d_model, vocab_size)
                    
                def forward(self, x):
                    x = self.embedding(x)
                    x = self.transformer(x)
                    x = self.classifier(x)
                    return x
            
            return TestTransformer()
    
    return ModelFactory


# Cleanup fixtures
@pytest.fixture(autouse=True)
def cleanup_temp_files():
    """Automatically cleanup temporary files after each test."""
    import glob
    import shutil
    
    yield
    
    # Clean up common temporary files
    temp_patterns = [
        "*.tmp",
        "*.temp",
        "test_*.onnx",
        "test_*.pt",
        "test_*.v",
        "test_*.sv"
    ]
    
    for pattern in temp_patterns:
        for file_path in glob.glob(pattern):
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except OSError:
                pass  # Ignore cleanup errors


# Pytest markers
pytest.mark.unit = pytest.mark.mark("unit")
pytest.mark.integration = pytest.mark.mark("integration")
pytest.mark.e2e = pytest.mark.mark("e2e")
pytest.mark.performance = pytest.mark.mark("performance")
pytest.mark.slow = pytest.mark.mark("slow")
pytest.mark.gpu = pytest.mark.mark("gpu")
pytest.mark.hardware = pytest.mark.mark("hardware")
pytest.mark.cloud = pytest.mark.mark("cloud")


# Custom pytest hooks
def pytest_configure(config):
    """Configure pytest markers and settings."""
    config.addinivalue_line(
        "markers",
        "unit: marks tests as unit tests (fast, isolated)"
    )
    config.addinivalue_line(
        "markers",
        "integration: marks tests as integration tests (slower, component interaction)"
    )
    config.addinivalue_line(
        "markers",
        "e2e: marks tests as end-to-end tests (slowest, full system)"
    )
    config.addinivalue_line(
        "markers",
        "performance: marks tests as performance tests"
    )
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow running (deselect with '-m "not slow"')"
    )
    config.addinivalue_line(
        "markers",
        "gpu: marks tests that require GPU hardware"
    )
    config.addinivalue_line(
        "markers",
        "hardware: marks tests that require hardware simulation tools"
    )
    config.addinivalue_line(
        "markers",
        "cloud: marks tests that require cloud services"
    )


def pytest_collection_modifyitems(config, items):
    """Automatically mark tests based on location."""
    for item in items:
        # Add unit marker for tests in unit directory
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        
        # Add integration marker for tests in integration directory
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        
        # Add e2e marker for tests in e2e directory
        elif "e2e" in str(item.fspath):
            item.add_marker(pytest.mark.e2e)
        
        # Add performance marker for tests in performance directory
        elif "performance" in str(item.fspath):
            item.add_marker(pytest.mark.performance)
        
        # Mark slow tests automatically
        if any(keyword in item.name.lower() for keyword in ["slow", "large", "heavy"]):
            item.add_marker(pytest.mark.slow)


def pytest_runtest_setup(item):
    """Setup for each test run."""
    # Skip GPU tests if no GPU available
    if "gpu" in item.keywords:
        try:
            import torch
            if not torch.cuda.is_available():
                pytest.skip("GPU not available")
        except ImportError:
            pytest.skip("PyTorch not available")
    
    # Skip hardware tests if tools not available
    if "hardware" in item.keywords:
        import shutil
        if not shutil.which("verilator"):
            pytest.skip("Verilator not available")
    
    # Skip cloud tests in CI unless explicitly enabled
    if "cloud" in item.keywords:
        if os.getenv("CI") and not os.getenv("ENABLE_CLOUD_TESTS"):
            pytest.skip("Cloud tests disabled in CI")


def pytest_benchmark_update_machine_info(config, machine_info):
    """Add custom machine info for benchmarks."""
    machine_info["python_version"] = f"{os.sys.version_info.major}.{os.sys.version_info.minor}"
    machine_info["test_environment"] = "ci" if os.getenv("CI") else "local"
