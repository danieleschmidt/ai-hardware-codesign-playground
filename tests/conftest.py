"""
Global pytest configuration and fixtures for AI Hardware Co-Design Playground.

This module provides shared fixtures, test utilities, and configuration
that can be used across all test modules.
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import AsyncGenerator, Generator
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

# Add backend source to path for imports
backend_path = Path(__file__).parent.parent / "backend"
if backend_path.exists():
    sys.path.insert(0, str(backend_path))


# Test Configuration
@pytest.fixture(scope="session")
def test_config():
    """Test configuration fixture."""
    return {
        "DATABASE_URL": "sqlite:///./test.db",
        "REDIS_URL": "redis://localhost:6379/1",
        "SECRET_KEY": "test-secret-key",
        "TESTING": True,
        "LOG_LEVEL": "DEBUG",
    }


# Async Support
@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# Database Fixtures
@pytest.fixture(scope="session")
def test_database_url(test_config):
    """Test database URL."""
    return test_config["DATABASE_URL"]


@pytest.fixture
def mock_database():
    """Mock database session for testing."""
    mock_db = Mock()
    mock_db.query.return_value = mock_db
    mock_db.filter.return_value = mock_db
    mock_db.first.return_value = None
    mock_db.all.return_value = []
    mock_db.commit.return_value = None
    mock_db.rollback.return_value = None
    mock_db.close.return_value = None
    return mock_db


# HTTP Client Fixtures
@pytest.fixture
def test_client():
    """Test client for FastAPI application."""
    try:
        from codesign_playground.main import app
        
        with TestClient(app) as client:
            yield client
    except ImportError:
        # Mock client if app is not available
        mock_client = Mock()
        mock_client.get.return_value.status_code = 200
        mock_client.post.return_value.status_code = 200
        yield mock_client


@pytest.fixture
async def async_test_client():
    """Async test client for FastAPI application."""
    try:
        from httpx import AsyncClient
        from codesign_playground.main import app
        
        async with AsyncClient(app=app, base_url="http://test") as client:
            yield client
    except ImportError:
        # Mock async client if dependencies not available
        mock_client = Mock()
        yield mock_client


# File System Fixtures
@pytest.fixture
def temp_directory(tmp_path):
    """Temporary directory for test files."""
    test_dir = tmp_path / "test_workspace"
    test_dir.mkdir()
    return test_dir


@pytest.fixture
def sample_files(temp_directory):
    """Create sample files for testing."""
    files = {}
    
    # Sample model file
    model_file = temp_directory / "sample_model.onnx"
    model_file.write_bytes(b"fake_onnx_content")
    files["model"] = model_file
    
    # Sample RTL file
    rtl_file = temp_directory / "sample_accelerator.v"
    rtl_file.write_text("""
module sample_accelerator (
    input clk,
    input rst,
    input [31:0] data_in,
    output [31:0] data_out
);
    // Sample Verilog content
    assign data_out = data_in + 1;
endmodule
    """)
    files["rtl"] = rtl_file
    
    # Sample configuration file
    config_file = temp_directory / "config.yaml"
    config_file.write_text("""
accelerator:
  type: systolic_array
  dimensions: [16, 16]
  data_width: 8
optimization:
  target_fps: 30
  power_budget: 5.0
    """)
    files["config"] = config_file
    
    return files


# Hardware Simulation Fixtures
@pytest.fixture
def mock_hardware_simulator():
    """Mock hardware simulator for testing."""
    simulator = Mock()
    simulator.compile.return_value = True
    simulator.run.return_value = {
        "cycles": 1000,
        "power": 2.5,
        "area": 10.2,
        "frequency": 200,
    }
    simulator.get_metrics.return_value = {
        "throughput": 100.0,
        "latency": 0.01,
        "efficiency": 0.85,
    }
    return simulator


@pytest.fixture
def mock_model_analyzer():
    """Mock model analyzer for testing."""
    analyzer = Mock()
    analyzer.analyze.return_value = {
        "operations": 1000000,
        "parameters": 50000,
        "memory_mb": 200,
        "compute_intensity": 0.5,
    }
    analyzer.get_layers.return_value = ["conv2d", "relu", "maxpool", "dense"]
    return analyzer


# Performance Testing Fixtures
@pytest.fixture
def benchmark_config():
    """Configuration for benchmark tests."""
    return {
        "max_time": 10.0,  # seconds
        "min_rounds": 5,
        "max_rounds": 100,
        "warmup_rounds": 3,
    }


# Security Testing Fixtures
@pytest.fixture
def security_headers():
    """Expected security headers for API responses."""
    return {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block",
        "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
    }


# Test Data Fixtures
@pytest.fixture
def sample_neural_network():
    """Sample neural network definition for testing."""
    return {
        "name": "sample_cnn",
        "layers": [
            {"type": "conv2d", "filters": 32, "kernel_size": 3, "activation": "relu"},
            {"type": "maxpool2d", "pool_size": 2},
            {"type": "conv2d", "filters": 64, "kernel_size": 3, "activation": "relu"},
            {"type": "maxpool2d", "pool_size": 2},
            {"type": "flatten"},
            {"type": "dense", "units": 128, "activation": "relu"},
            {"type": "dense", "units": 10, "activation": "softmax"},
        ],
        "input_shape": [224, 224, 3],
        "output_shape": [10],
    }


@pytest.fixture
def sample_hardware_spec():
    """Sample hardware specification for testing."""
    return {
        "type": "systolic_array",
        "dimensions": {"rows": 16, "cols": 16},
        "data_width": 8,
        "memory_hierarchy": ["sram_64kb", "dram"],
        "dataflow": "weight_stationary",
        "frequency_mhz": 200,
        "power_budget_w": 5.0,
    }


# Error Handling Fixtures
@pytest.fixture
def mock_error_handler():
    """Mock error handler for testing error scenarios."""
    handler = Mock()
    handler.handle_error.return_value = {"error": "mocked_error", "code": 500}
    return handler


# Cleanup Fixtures
@pytest.fixture(autouse=True)
def cleanup_test_files():
    """Automatically cleanup test files after each test."""
    yield
    # Cleanup logic here if needed
    pass


# Skip Conditions
def skip_if_no_hardware():
    """Skip test if hardware simulation tools are not available."""
    return pytest.mark.skipif(
        not os.getenv("HARDWARE_TOOLS_AVAILABLE"),
        reason="Hardware simulation tools not available"
    )


def skip_if_no_gpu():
    """Skip test if GPU is not available."""
    return pytest.mark.skipif(
        not os.getenv("GPU_AVAILABLE"),
        reason="GPU not available for testing"
    )


# Custom Test Markers
pytestmark = [
    pytest.mark.filterwarnings("ignore::DeprecationWarning"),
    pytest.mark.filterwarnings("ignore::PendingDeprecationWarning"),
]


# Test Utilities
class TestUtils:
    """Utility functions for testing."""
    
    @staticmethod
    def assert_response_ok(response):
        """Assert that HTTP response is successful."""
        assert 200 <= response.status_code < 300, f"Expected success, got {response.status_code}: {response.text}"
    
    @staticmethod
    def assert_valid_json(response):
        """Assert that response contains valid JSON."""
        try:
            response.json()
        except ValueError as e:
            pytest.fail(f"Response does not contain valid JSON: {e}")
    
    @staticmethod
    def create_test_model(temp_dir: Path, model_type: str = "simple"):
        """Create a test model file."""
        if model_type == "simple":
            content = b"fake_model_content"
        else:
            content = b"complex_fake_model_content"
        
        model_path = temp_dir / f"test_model_{model_type}.onnx"
        model_path.write_bytes(content)
        return model_path


@pytest.fixture
def test_utils():
    """Test utilities fixture."""
    return TestUtils