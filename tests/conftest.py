"""Global test configuration and fixtures for AI Hardware Co-Design Playground."""

import os
import tempfile
from pathlib import Path
from typing import Generator, Dict, Any
import pytest
import numpy as np
import torch
import json

# Test configuration
pytest_plugins = [
    "tests.fixtures.hardware_fixtures",
    "tests.fixtures.model_fixtures", 
    "tests.fixtures.simulation_fixtures",
]

def pytest_configure(config):
    """Configure pytest with custom markers and settings."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "e2e: marks tests as end-to-end tests"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU"
    )
    config.addinivalue_line(
        "markers", "fpga: marks tests that require FPGA tools"
    )
    config.addinivalue_line(
        "markers", "cloud: marks tests that require cloud resources"
    )
    config.addinivalue_line(
        "markers", "hardware_tools: marks tests that require hardware design tools"
    )

@pytest.fixture(scope="session")
def test_config() -> Dict[str, Any]:
    """Global test configuration."""
    return {
        "simulation_timeout": 60,  # seconds
        "max_test_models": 5,
        "enable_gpu_tests": torch.cuda.is_available(),
        "enable_hardware_tools": os.getenv("ENABLE_HARDWARE_TOOLS", "false").lower() == "true",
        "test_data_dir": Path(__file__).parent / "fixtures" / "data",
        "temp_output_dir": None,  # Will be set by temp_output_dir fixture
    }

@pytest.fixture(scope="session")
def temp_output_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory(prefix="codesign_test_") as temp_dir:
        yield Path(temp_dir)

@pytest.fixture(scope="function")
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for individual test."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)

@pytest.fixture(scope="session")
def sample_data_dir(test_config) -> Path:
    """Path to sample test data directory."""
    data_dir = test_config["test_data_dir"]
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir

@pytest.fixture
def mock_environment_variables(monkeypatch):
    """Mock environment variables for testing."""
    test_env_vars = {
        "CODESIGN_PLAYGROUND_DEV": "true",
        "LOG_LEVEL": "DEBUG",
        "DATABASE_URL": "sqlite:///:memory:",
        "REDIS_URL": "redis://localhost:6379/1",
        "ENABLE_CACHING": "false",
        "SIMULATION_TIMEOUT": "30",
        "MAX_PARALLEL_SIMULATIONS": "1",
    }
    
    for key, value in test_env_vars.items():
        monkeypatch.setenv(key, value)

@pytest.fixture
def skip_if_no_gpu():
    """Skip test if no GPU is available."""
    if not torch.cuda.is_available():
        pytest.skip("GPU not available")

@pytest.fixture  
def skip_if_no_hardware_tools():
    """Skip test if hardware tools are not available."""
    # Check for common hardware tools
    tools_available = any([
        os.system("which verilator >/dev/null 2>&1") == 0,
        os.system("which yosys >/dev/null 2>&1") == 0,
        os.system("which vivado >/dev/null 2>&1") == 0,
    ])
    
    if not tools_available:
        pytest.skip("Hardware design tools not available")

@pytest.fixture
def numpy_random_seed():
    """Set random seed for reproducible tests."""
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

@pytest.fixture
def sample_test_vectors() -> Dict[str, np.ndarray]:
    """Generate sample test vectors for hardware testing."""
    return {
        "input_data": np.random.randint(0, 256, (8, 8), dtype=np.uint8),
        "weights": np.random.randint(-128, 127, (8, 8), dtype=np.int8),
        "expected_output": np.random.randint(0, 65536, (8, 8), dtype=np.uint16),
    }

@pytest.fixture
def mock_simulation_results():
    """Mock simulation results for testing."""
    return {
        "cycles": 1000,
        "latency_ns": 5000,
        "throughput_ops_per_sec": 1000000,
        "power_mw": 250.5,
        "area_mm2": 0.5,
        "utilization": 0.85,
        "memory_bandwidth_gb_s": 12.8,
    }

# Pytest collection hooks
def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test location."""
    for item in items:
        # Add markers based on test file location
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "e2e" in str(item.fspath):
            item.add_marker(pytest.mark.e2e)
        elif "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        
        # Add slow marker for tests that typically take longer
        if any(keyword in item.name.lower() for keyword in ["simulation", "synthesis", "optimization"]):
            item.add_marker(pytest.mark.slow)
        
        # Add hardware_tools marker for tests requiring external tools
        if any(keyword in item.name.lower() for keyword in ["verilator", "yosys", "vivado", "quartus"]):
            item.add_marker(pytest.mark.hardware_tools)

# Test discovery hooks
def pytest_runtest_setup(item):
    """Setup hook for individual tests."""
    # Skip tests based on markers and available resources
    if item.get_closest_marker("gpu") and not torch.cuda.is_available():
        pytest.skip("GPU not available")
    
    if item.get_closest_marker("hardware_tools"):
        # Check if hardware tools are available
        tools_available = any([
            os.system("which verilator >/dev/null 2>&1") == 0,
            os.system("which yosys >/dev/null 2>&1") == 0,
        ])
        if not tools_available:
            pytest.skip("Hardware design tools not available")
    
    if item.get_closest_marker("cloud"):
        # Skip cloud tests unless specifically enabled
        if not os.getenv("ENABLE_CLOUD_TESTS", "false").lower() == "true":
            pytest.skip("Cloud tests disabled")

# Cleanup hooks
def pytest_runtest_teardown(item, nextitem):
    """Cleanup hook after each test."""
    # Clean up any CUDA memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Clean up any temporary files created during test
    # (Most cleanup should be handled by temp_dir fixtures)
    pass

# Session-level setup/teardown
def pytest_sessionstart(session):
    """Called after the Session object has been created."""
    print("\n🧪 Starting AI Hardware Co-Design Playground test session")
    
    # Setup test environment
    os.environ["CODESIGN_PLAYGROUND_TEST"] = "true"
    
    # Ensure test directories exist
    test_dirs = [
        "tests/fixtures/data",
        "tests/fixtures/models", 
        "tests/fixtures/hardware",
        "tests/fixtures/results",
    ]
    
    for dir_path in test_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

def pytest_sessionfinish(session, exitstatus):
    """Called after whole test run finished."""
    print(f"\n✅ Test session finished with exit status: {exitstatus}")
    
    # Cleanup test environment
    if "CODESIGN_PLAYGROUND_TEST" in os.environ:
        del os.environ["CODESIGN_PLAYGROUND_TEST"]

# Custom test report
def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Add custom summary information to test report."""
    if exitstatus == 0:
        terminalreporter.write_sep("=", "🎉 All tests passed!")
    else:
        terminalreporter.write_sep("=", "❌ Some tests failed")
    
    # Print resource information
    terminalreporter.write_line(f"GPU available: {torch.cuda.is_available()}")
    
    # Count tests by category
    try:
        stats = terminalreporter.stats
        if hasattr(stats, 'get'):
            passed = len(stats.get('passed', []))
            failed = len(stats.get('failed', []))
            skipped = len(stats.get('skipped', []))
            
            terminalreporter.write_line(f"Test summary: {passed} passed, {failed} failed, {skipped} skipped")
    except:
        pass