"""
Shared pytest configuration and fixtures for AI Hardware Co-Design Playground tests.

This module provides common fixtures, test utilities, and configuration
for the entire test suite.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock
import os
import sys

# Add backend to Python path for imports
backend_path = Path(__file__).parent.parent / "backend"
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))


@pytest.fixture(scope="session")
def test_data_dir():
    """Get the test data directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def mock_model():
    """Mock model for testing."""
    mock = Mock()
    mock.name = "test_model"
    mock.framework = "pytorch"
    mock.path = "/path/to/model.pt"
    mock.parameters = 1000000
    return mock


@pytest.fixture
def sample_input_shapes():
    """Common input shapes for testing."""
    return [
        (224, 224, 3),  # ImageNet
        (32, 32, 3),    # CIFAR-32
        (28, 28, 1),    # MNIST
        (512, 512, 3),  # High-res
    ]


@pytest.fixture
def sample_constraints():
    """Sample optimization constraints."""
    return {
        "target_fps": 30.0,
        "power_budget": 5.0,
        "area_budget": 100.0,
        "latency_ms": 33.0
    }


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Set up test environment before each test."""
    # Set test environment variable
    os.environ["TEST_MODE"] = "true"
    
    yield
    
    # Cleanup after test
    if "TEST_MODE" in os.environ:
        del os.environ["TEST_MODE"]


@pytest.fixture
def mock_accelerator_designer():
    """Mock AcceleratorDesigner for testing."""
    designer = Mock()
    designer.profile_model.return_value = Mock(
        peak_gflops=10.0,
        bandwidth_gb_s=25.0,
        parameters=1000000,
        memory_mb=64.0
    )
    designer.design.return_value = Mock(
        compute_units=64,
        dataflow="weight_stationary",
        performance_model={"throughput_ops_s": 1e9}
    )
    return designer


# Test markers
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "hardware: marks tests requiring hardware simulation"
    )
    config.addinivalue_line(
        "markers", "benchmark: marks tests as performance benchmarks"
    )


# Test collection customization
def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Add unit marker to all tests in unit/ directory
        if "unit/" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        
        # Add integration marker to all tests in integration/ directory
        elif "integration/" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        
        # Add benchmark marker to performance tests
        elif "benchmark" in item.name or "performance" in str(item.fspath):
            item.add_marker(pytest.mark.benchmark)
            item.add_marker(pytest.mark.slow)


# Session-scoped fixtures for expensive setup
@pytest.fixture(scope="session")
def test_cache():
    """Session-scoped cache for test data."""
    cache = {}
    yield cache
    cache.clear()


# Cleanup fixtures
@pytest.fixture(autouse=True, scope="function")
def cleanup_test_artifacts():
    """Clean up test artifacts after each test."""
    yield
    
    # Clean up any temp files in current directory
    temp_files = [
        f for f in Path.cwd().iterdir() 
        if f.name.startswith("test_") and f.suffix in [".log", ".tmp", ".cache"]
    ]
    
    for temp_file in temp_files:
        try:
            temp_file.unlink()
        except:
            pass  # Ignore cleanup errors