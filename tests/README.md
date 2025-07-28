# AI Hardware Co-Design Playground - Testing Infrastructure

This directory contains comprehensive test suites for the AI Hardware Co-Design Playground.

## Test Structure

```
tests/
├── unit/                 # Unit tests for individual components
│   ├── core/            # Core functionality tests
│   ├── hardware/        # Hardware template tests
│   ├── optimization/    # Optimization algorithm tests
│   ├── simulation/      # Simulation engine tests
│   └── api/             # API endpoint tests
├── integration/         # Integration tests for component interactions
│   ├── workflows/       # End-to-end workflow tests
│   ├── database/        # Database integration tests
│   └── external/        # External tool integration tests
├── e2e/                 # End-to-end system tests
│   ├── web/             # Web interface tests
│   ├── cli/             # Command-line interface tests
│   └── sdk/             # Python SDK tests
├── performance/         # Performance and load tests
│   ├── benchmarks/      # Performance benchmarks
│   ├── load/            # Load testing scenarios
│   └── stress/          # Stress testing scenarios
├── fixtures/            # Test data and fixtures
│   ├── models/          # Sample ML models
│   ├── designs/         # Sample hardware designs
│   └── data/            # Test datasets
├── conftest.py          # Pytest configuration and fixtures
├── pytest.ini          # Pytest settings (legacy support)
└── README.md            # This file
```

## Test Categories

### Unit Tests (`tests/unit/`)
Test individual components in isolation with mocked dependencies.

**Coverage Target**: >95%

**Key Areas**:
- Model analysis and profiling
- Hardware template generation
- Optimization algorithms
- Simulation engines
- API business logic
- Utility functions

**Example**:
```python
def test_systolic_array_generation():
    """Test systolic array hardware generation."""
    designer = AcceleratorDesigner()
    accelerator = designer.design(
        template="systolic_array",
        rows=16,
        cols=16,
        data_width=8
    )
    
    assert accelerator.dimensions == (16, 16)
    assert accelerator.data_width == 8
    assert accelerator.rtl_code is not None
    assert "module systolic_array" in accelerator.rtl_code
```

### Integration Tests (`tests/integration/`)
Test component interactions and data flow between modules.

**Coverage Target**: >90%

**Key Areas**:
- Database operations and migrations
- Message queue processing
- External tool integrations (TVM, Verilator)
- API endpoint workflows
- File I/O operations

**Example**:
```python
@pytest.mark.integration
def test_model_to_hardware_workflow():
    """Test complete model analysis to hardware generation workflow."""
    # Import and analyze model
    model = ModelImporter.load("fixtures/models/resnet18.onnx")
    profile = ModelAnalyzer.profile(model)
    
    # Generate hardware
    designer = AcceleratorDesigner()
    accelerator = designer.design_for_model(model, profile)
    
    # Verify integration
    assert accelerator.is_compatible_with(model)
    assert accelerator.estimated_performance > 0
```

### End-to-End Tests (`tests/e2e/`)
Test complete user workflows from start to finish.

**Coverage Target**: >80%

**Key Areas**:
- Web UI user journeys
- CLI command sequences
- SDK usage patterns
- Complete design flows
- Export and deployment

**Example**:
```python
@pytest.mark.e2e
def test_complete_design_flow(browser):
    """Test complete design flow through web interface."""
    # Navigate to design page
    page = browser.new_page()
    page.goto("http://localhost:3000/design")
    
    # Upload model
    page.locator("input[type=file]").set_input_files("fixtures/models/model.onnx")
    page.click("button:text('Analyze Model')")
    
    # Select hardware template
    page.click("button:text('Systolic Array')")
    page.fill("input[name=rows]", "16")
    page.fill("input[name=cols]", "16")
    
    # Generate design
    page.click("button:text('Generate Hardware')")
    
    # Verify results
    assert page.locator(".design-results").is_visible()
    assert page.locator(".performance-metrics").is_visible()
```

### Performance Tests (`tests/performance/`)
Test system performance, scalability, and resource usage.

**Key Areas**:
- Simulation speed benchmarks
- Optimization algorithm convergence
- Memory usage profiling
- Concurrent user handling
- Large model processing

**Example**:
```python
@pytest.mark.performance
def test_optimization_performance(benchmark):
    """Benchmark optimization algorithm performance."""
    model = load_test_model("large_transformer")
    optimizer = GeneticOptimizer(population_size=100)
    
    result = benchmark(
        optimizer.optimize,
        model,
        generations=50,
        objectives=["latency", "power", "area"]
    )
    
    # Verify performance requirements
    assert result.total_time < 300  # Under 5 minutes
    assert result.convergence_generation < 40
```

## Test Configuration

### Pytest Configuration (`conftest.py`)
Centralized test configuration and shared fixtures.

**Key Fixtures**:
- Database setup/teardown
- Mock external services
- Test data generators
- Performance measurement
- Browser automation

### Environment Setup
Tests use separate configuration for isolation:

```bash
# Test environment variables
ENVIRONMENT=test
DATABASE_URL=postgresql://test_user:test_pass@localhost:5432/test_db
REDIS_URL=redis://localhost:6379/15
CELERY_TASK_ALWAYS_EAGER=true
LOG_LEVEL=WARNING
```

## Running Tests

### All Tests
```bash
# Run complete test suite
npm run test

# Run with coverage
npm run test:coverage

# Run with parallel execution
pytest -n auto
```

### Specific Test Categories
```bash
# Unit tests only
pytest tests/unit/

# Integration tests
pytest tests/integration/ -m integration

# End-to-end tests
pytest tests/e2e/ -m e2e

# Performance tests
pytest tests/performance/ -m performance
```

### Test Filtering
```bash
# Run specific test file
pytest tests/unit/test_model_analyzer.py

# Run tests matching pattern
pytest -k "test_systolic"

# Skip slow tests
pytest -m "not slow"

# Run only GPU tests
pytest -m gpu
```

## Test Data Management

### Fixtures Directory (`tests/fixtures/`)
Standardized test data for consistent testing.

**Structure**:
```
fixtures/
├── models/
│   ├── small/           # Lightweight models for unit tests
│   ├── medium/          # Realistic models for integration tests
│   └── large/           # Complex models for performance tests
├── designs/
│   ├── templates/       # Hardware template configurations
│   ├── generated/       # Pre-generated designs
│   └── reference/       # Reference implementations
└── data/
    ├── inputs/          # Sample input data
    ├── outputs/         # Expected output data
    └── datasets/        # Test datasets
```

### Dynamic Test Data
Generate test data programmatically for edge cases:

```python
@pytest.fixture
def random_model():
    """Generate random neural network model for testing."""
    return ModelGenerator.create_random(
        layers=random.randint(5, 20),
        input_shape=(1, 3, 224, 224),
        num_classes=random.randint(10, 1000)
    )
```

## Continuous Integration

### GitHub Actions Integration
Automated testing on every push and pull request.

**Test Matrix**:
- Python versions: 3.9, 3.10, 3.11, 3.12
- Operating systems: Ubuntu, macOS, Windows
- Dependencies: Minimal, full

### Quality Gates
- Code coverage >80% (unit tests >95%)
- All tests must pass
- Performance regression detection
- Security vulnerability scanning

## Test Maintenance

### Regular Tasks
- Update test fixtures with new model formats
- Refresh performance baselines
- Review and update test coverage
- Clean up obsolete test cases

### Test Review Process
- Peer review for test changes
- Performance impact assessment
- Documentation updates
- Integration with CI/CD pipeline

## Debugging Tests

### Local Debugging
```bash
# Run with verbose output
pytest -v -s

# Stop on first failure
pytest -x

# Drop into debugger on failure
pytest --pdb

# Profile test execution
pytest --profile
```

### Test Isolation
```bash
# Run single test in isolation
pytest tests/unit/test_model_analyzer.py::test_specific_function -v

# Clear caches between runs
pytest --cache-clear
```

## Contributing to Tests

### Guidelines
1. **Test Naming**: Use descriptive names that explain what is being tested
2. **Test Structure**: Follow Arrange-Act-Assert pattern
3. **Test Independence**: Each test should be independent and idempotent
4. **Coverage**: Aim for high coverage but focus on meaningful tests
5. **Performance**: Keep test execution time reasonable

### Code Review Checklist
- [ ] Tests cover both happy path and edge cases
- [ ] Appropriate test category (unit/integration/e2e)
- [ ] Clear test documentation and comments
- [ ] Proper use of fixtures and mocks
- [ ] Reasonable execution time
- [ ] No hardcoded values or brittle assertions

For questions about testing infrastructure, please refer to the [testing documentation](../docs/guides/developer/testing.md) or reach out through our community channels.