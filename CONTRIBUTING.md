# Contributing to AI Hardware Co-Design Playground

🎉 Thank you for your interest in contributing to the AI Hardware Co-Design Playground! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Ways to Contribute](#ways-to-contribute)
- [Development Setup](#development-setup)
- [Contribution Workflow](#contribution-workflow)
- [Coding Guidelines](#coding-guidelines)
- [Testing Requirements](#testing-requirements)
- [Documentation](#documentation)
- [Review Process](#review-process)
- [Community and Support](#community-and-support)

## Code of Conduct

This project and everyone participating in it is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to [community@codesign-playground.com](mailto:community@codesign-playground.com).

## Getting Started

### Prerequisites

- Python 3.9+ 
- Git
- Docker (recommended for development)
- Basic knowledge of machine learning and/or hardware design

### First Contribution

1. **Explore the project**: Read the [README](README.md), [Architecture](ARCHITECTURE.md), and [Documentation](https://docs.codesign-playground.com)
2. **Find an issue**: Look for issues labeled `good first issue` or `help wanted`
3. **Join the community**: Join our [Discord server](https://discord.gg/ai-hardware-codesign) to ask questions
4. **Set up development environment**: Follow the [Development Setup](#development-setup) instructions

## Ways to Contribute

### 🐛 Bug Reports
- Use the [bug report template](.github/ISSUE_TEMPLATE/bug_report.md)
- Include clear reproduction steps
- Provide environment details
- Add relevant logs and error messages

### 💡 Feature Requests
- Use the [feature request template](.github/ISSUE_TEMPLATE/feature_request.md)
- Describe the problem you're solving
- Explain your proposed solution
- Consider implementation complexity

### 📝 Documentation
- Fix typos and improve clarity
- Add examples and tutorials
- Create API documentation
- Translate content

### 🔧 Code Contributions
- Fix bugs and implement features
- Add new hardware templates
- Improve optimization algorithms
- Enhance testing coverage

### 🎨 Design and UX
- Improve web interface design
- Create visualizations
- Design logos and graphics
- Enhance user experience

### 🧪 Testing
- Add unit tests
- Create integration tests
- Develop benchmarks
- Test on different platforms

## Development Setup

### Option 1: Docker Development (Recommended)

```bash
# Clone the repository
git clone https://github.com/terragon-labs/ai-hardware-codesign-playground.git
cd ai-hardware-codesign-playground

# Start development environment
docker-compose up -d

# Access the container
docker-compose exec app bash

# Run tests
pytest tests/
```

### Option 2: Local Development

```bash
# Clone the repository
git clone https://github.com/terragon-labs/ai-hardware-codesign-playground.git
cd ai-hardware-codesign-playground

# Run setup script
./scripts/setup.sh

# Activate virtual environment
source venv/bin/activate

# Install development dependencies
pip install -e ".[dev,test,docs]"

# Setup pre-commit hooks
pre-commit install
```

### Option 3: VS Code Dev Containers

1. Install the [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)
2. Open the project in VS Code
3. Click "Reopen in Container" when prompted
4. Wait for the container to build and start

## Contribution Workflow

### 1. Fork and Clone

```bash
# Fork the repository on GitHub
# Clone your fork
git clone https://github.com/YOUR_USERNAME/ai-hardware-codesign-playground.git
cd ai-hardware-codesign-playground

# Add upstream remote
git remote add upstream https://github.com/terragon-labs/ai-hardware-codesign-playground.git
```

### 2. Create a Branch

```bash
# Update main branch
git checkout main
git pull upstream main

# Create a feature branch
git checkout -b feature/your-feature-name
# or for bug fixes
git checkout -b fix/issue-number-short-description
```

### 3. Make Changes

- Follow the [coding guidelines](#coding-guidelines)
- Write tests for your changes
- Update documentation as needed
- Ensure all tests pass

### 4. Commit Changes

```bash
# Stage your changes
git add .

# Commit with a descriptive message
git commit -m "feat: add new systolic array template

- Implement configurable systolic array generator
- Add support for different data widths
- Include comprehensive tests and documentation
- Closes #123"
```

#### Commit Message Format

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

Examples:
```
feat(hardware): add transformer accelerator template
fix(simulation): resolve memory leak in verilator integration
docs: update installation instructions for macOS
test: add integration tests for optimization engine
```

### 5. Push and Create PR

```bash
# Push your branch
git push origin feature/your-feature-name

# Create a pull request on GitHub
```

## Coding Guidelines

### Python Code Style

We follow [PEP 8](https://pep8.org/) with some modifications:

- **Line length**: 88 characters (Black formatter default)
- **Imports**: Use `isort` for import sorting
- **Type hints**: Required for all public APIs
- **Docstrings**: Google style docstrings

#### Code Formatting

```bash
# Format code with Black
black src tests

# Sort imports
isort src tests

# Lint with Ruff
ruff check src tests

# Type check with MyPy
mypy src
```

#### Example Code

```python
"""Example module demonstrating coding style."""

from typing import Dict, List, Optional, Union
import numpy as np
import torch


class HardwareTemplate:
    """Base class for hardware templates.
    
    Args:
        name: Template name
        parameters: Configuration parameters
        
    Attributes:
        name: Template name
        parameters: Configuration parameters
    """
    
    def __init__(self, name: str, parameters: Dict[str, Union[int, float, str]]) -> None:
        self.name = name
        self.parameters = parameters
    
    def generate_rtl(self, output_path: Optional[str] = None) -> str:
        """Generate RTL code from template.
        
        Args:
            output_path: Optional path to save RTL file
            
        Returns:
            Generated RTL code as string
            
        Raises:
            ValueError: If template parameters are invalid
        """
        if not self.parameters:
            raise ValueError("Template parameters are required")
        
        # Implementation here
        return "// Generated RTL code"
```

### Hardware Description Language (HDL)

For Verilog/SystemVerilog code:

- **Indentation**: 2 spaces
- **Naming**: `snake_case` for signals, `PascalCase` for modules
- **Comments**: Explain complex logic and interfaces
- **Clocking**: Use consistent clock and reset naming

#### Example HDL

```verilog
// Systolic array processing element
module systolic_pe #(
    parameter DATA_WIDTH = 8,
    parameter WEIGHT_WIDTH = 8,
    parameter ACCUM_WIDTH = 32
) (
    input  wire                    clk,
    input  wire                    rst_n,
    input  wire [DATA_WIDTH-1:0]   data_in,
    input  wire [WEIGHT_WIDTH-1:0] weight_in,
    input  wire [ACCUM_WIDTH-1:0]  partial_sum_in,
    output reg  [DATA_WIDTH-1:0]   data_out,
    output reg  [ACCUM_WIDTH-1:0]  partial_sum_out
);

// Processing logic
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        data_out <= 0;
        partial_sum_out <= 0;
    end else begin
        data_out <= data_in;
        partial_sum_out <= partial_sum_in + (data_in * weight_in);
    end
end

endmodule
```

## Testing Requirements

### Test Coverage

- **Minimum coverage**: 80% for new code
- **Test types**: Unit, integration, and end-to-end tests
- **Test naming**: `test_<functionality>_<condition>_<expected_result>`

### Test Structure

```python
"""Test module for hardware templates."""

import pytest
from codesign_playground.templates import SystolicArray


class TestSystolicArray:
    """Test cases for systolic array template."""
    
    def test_create_valid_array_succeeds(self):
        """Test creating a valid systolic array succeeds."""
        array = SystolicArray(rows=8, cols=8, data_width=8)
        assert array.rows == 8
        assert array.cols == 8
    
    def test_invalid_dimensions_raises_error(self):
        """Test invalid dimensions raise ValueError."""
        with pytest.raises(ValueError, match="Dimensions must be positive"):
            SystolicArray(rows=0, cols=8, data_width=8)
    
    @pytest.mark.slow
    def test_large_array_generation_completes(self):
        """Test large array generation completes successfully."""
        array = SystolicArray(rows=128, cols=128, data_width=16)
        rtl = array.generate_rtl()
        assert len(rtl) > 1000  # Basic sanity check
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/unit/test_templates.py

# Run tests matching pattern
pytest -k "test_systolic"

# Run slow tests
pytest -m slow

# Skip slow tests
pytest -m "not slow"
```

## Documentation

### Types of Documentation

1. **API Documentation**: Docstrings in code
2. **User Guides**: Step-by-step tutorials
3. **Developer Docs**: Architecture and design decisions
4. **Examples**: Jupyter notebooks and scripts

### Writing Guidelines

- **Clear and concise**: Use simple language
- **Examples**: Include code examples and outputs
- **Structure**: Use consistent formatting and organization
- **Updates**: Keep documentation in sync with code changes

### Documentation Structure

```
docs/
├── guides/          # User guides and tutorials
├── api/             # API reference (auto-generated)
├── development/     # Developer documentation
├── examples/        # Example notebooks and scripts
└── deployment/      # Deployment and operations
```

### Building Documentation

```bash
# Install documentation dependencies
pip install -e ".[docs]"

# Build documentation
mkdocs build

# Serve documentation locally
mkdocs serve

# Documentation will be available at http://localhost:8000
```

## Review Process

### Pull Request Requirements

Before submitting a PR, ensure:

- [ ] All tests pass
- [ ] Code follows style guidelines
- [ ] Documentation is updated
- [ ] Commit messages follow convention
- [ ] PR description is clear and complete

### PR Template

Use the provided [PR template](.github/pull_request_template.md):

```markdown
## Summary
Brief description of changes

## Changes
- List of changes made
- Use bullet points

## Testing
- How you tested the changes
- Include test results if applicable

## Documentation
- Documentation updates made
- Links to relevant docs

## Breaking Changes
- List any breaking changes
- Migration guide if needed

## Checklist
- [ ] Tests pass
- [ ] Documentation updated
- [ ] Code style follows guidelines
```

### Review Criteria

Reviewers will check:

1. **Functionality**: Does the code work as intended?
2. **Quality**: Is the code well-written and maintainable?
3. **Testing**: Are there adequate tests?
4. **Documentation**: Is documentation clear and complete?
5. **Security**: Are there any security concerns?
6. **Performance**: Are there performance implications?

### Review Timeline

- **Initial review**: Within 2-3 business days
- **Follow-up reviews**: Within 1-2 business days
- **Merge**: After approval from at least one maintainer

## Community and Support

### Communication Channels

- **Discord**: [AI Hardware Co-Design Community](https://discord.gg/ai-hardware-codesign)
- **GitHub Discussions**: [Project Discussions](https://github.com/terragon-labs/ai-hardware-codesign-playground/discussions)
- **Email**: [community@codesign-playground.com](mailto:community@codesign-playground.com)

### Getting Help

1. **Search existing issues** and discussions first
2. **Ask in Discord** for quick questions
3. **Create a GitHub issue** for bugs or feature requests
4. **Join community calls** (announced in Discord)

### Recognition

Contributors are recognized through:

- **Contributors page** on our website
- **Changelog credits** in releases
- **Community highlights** in newsletters
- **Conference presentations** (with permission)

## License

By contributing to AI Hardware Co-Design Playground, you agree that your contributions will be licensed under the [MIT License](LICENSE).

---

## Quick Reference

### Useful Commands

```bash
# Setup development environment
./scripts/setup.sh

# Run tests
pytest tests/

# Format code
black src tests && isort src tests

# Lint code
ruff check src tests

# Type check
mypy src

# Build documentation
mkdocs build

# Start development services
docker-compose up -d
```

### Helpful Links

- [Project Homepage](https://codesign-playground.com)
- [Documentation](https://docs.codesign-playground.com)
- [Discord Community](https://discord.gg/ai-hardware-codesign)
- [GitHub Issues](https://github.com/terragon-labs/ai-hardware-codesign-playground/issues)
- [GitHub Discussions](https://github.com/terragon-labs/ai-hardware-codesign-playground/discussions)

---

Thank you for contributing to AI Hardware Co-Design Playground! Your contributions help make AI hardware design more accessible to everyone. 🚀