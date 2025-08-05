# Contributing to AI Hardware Co-Design Playground

🎉 Thank you for your interest in contributing to the AI Hardware Co-Design Playground! This project aims to democratize AI hardware design through an open and collaborative community.

## 🌟 How to Contribute

We welcome contributions of all types:
- 🐛 Bug reports and fixes
- ✨ New features and enhancements
- 📚 Documentation improvements
- 🧪 Tests and benchmarks
- 🏗️ Hardware templates
- 🔧 Tool integrations
- 🎨 Design and UX improvements

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Environment](#development-environment)
- [Contribution Workflow](#contribution-workflow)
- [Code Standards](#code-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Hardware Templates](#hardware-templates)
- [Community Guidelines](#community-guidelines)
- [Recognition](#recognition)

## Code of Conduct

This project and everyone participating in it is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to [community@codesign-playground.com](mailto:community@codesign-playground.com).

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- Node.js 18+ (for frontend development)
- Git
- Docker (optional but recommended)
- Basic knowledge of machine learning and/or hardware design

### First Contribution

1. **Explore the project**: Read the [README](README.md), [Architecture](ARCHITECTURE.md), and [Documentation](https://docs.codesign-playground.com)
2. **Find an issue**: Look for issues labeled `good first issue` or `help wanted`
3. **Join the community**: Join our [Discord server](https://discord.gg/ai-hardware-codesign) to ask questions
4. **Set up development environment**: Follow the [Development Environment](#development-environment) instructions

### Ways to Contribute

#### 🐛 Bug Reports
- Use the [bug report template](.github/ISSUE_TEMPLATE/bug_report.md)
- Include clear reproduction steps
- Provide environment details
- Add relevant logs and error messages

#### 💡 Feature Requests
- Use the [feature request template](.github/ISSUE_TEMPLATE/feature_request.md)
- Describe the problem you're solving
- Explain your proposed solution
- Consider implementation complexity

#### 📝 Documentation
- Fix typos and improve clarity
- Add examples and tutorials
- Create API documentation
- Translate content

#### 🔧 Code Contributions
- Fix bugs and implement features
- Add new hardware templates
- Improve optimization algorithms
- Enhance testing coverage

## 🔧 Development Environment

### Quick Setup

```bash
# Install dependencies and set up development environment
npm run setup

# Start development servers
npm run dev
```

### Option 1: Docker Development (Recommended)

```bash
# Clone the repository
git clone https://github.com/terragon-labs/ai-hardware-codesign-playground.git
cd ai-hardware-codesign-playground

# Start development environment
docker-compose -f docker-compose.dev.yml up --build

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

# Backend setup
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e ".[dev,test,docs]"

# Frontend setup
cd ../frontend
npm install

# Install pre-commit hooks
pre-commit install
```

### Option 3: VS Code Dev Containers

1. Install the [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)
2. Open the project in VS Code
3. Click "Reopen in Container" when prompted
4. Wait for the container to build and start

### Development Tools

- **Code Formatting**: Black (Python), Prettier (JS/TS)
- **Linting**: Ruff, ESLint
- **Type Checking**: MyPy, TypeScript
- **Testing**: Pytest, Jest
- **Documentation**: Sphinx, MkDocs, Storybook

## 🔄 Contribution Workflow

### 1. Create an Issue

Before starting work, create an issue to:
- Report bugs with reproduction steps
- Propose new features with use cases
- Discuss architectural changes
- Ask questions or seek clarification

### 2. Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/ai-hardware-codesign-playground.git
cd ai-hardware-codesign-playground

# Add upstream remote
git remote add upstream https://github.com/terragon-labs/ai-hardware-codesign-playground.git
```

### 3. Create a Branch

```bash
# Update main branch
git checkout main
git pull upstream main

# Create a feature branch
git checkout -b feature/your-feature-name
# or for bug fixes
git checkout -b fix/issue-number-short-description
```

### 4. Make Changes

- Follow the [coding guidelines](#code-standards)
- Write tests for your changes
- Update documentation as needed
- Ensure all tests pass
- Commit frequently with clear messages

### 5. Test Your Changes

```bash
# Run all tests
npm run test

# Run specific test suites
npm run test:backend
npm run test:frontend
npm run test:e2e

# Run linting and type checking
npm run lint
npm run typecheck

# Check security
npm run security
```

### 6. Commit Changes

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

### 7. Push and Create PR

```bash
# Push your branch
git push origin feature/your-feature-name

# Create a pull request on GitHub
```

## 📝 Code Standards

### Commit Message Format

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

**Types**: 
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Examples**:
```
feat(hardware): add transformer accelerator template
fix(simulation): resolve memory leak in verilator integration
docs(api): update hardware design API documentation
test: add integration tests for optimization engine
```

### Python Code Style

We follow [PEP 8](https://pep8.org/) with some modifications:

- **Line length**: 88 characters (Black formatter default)
- **Imports**: Use `isort` for import sorting
- **Type hints**: Required for all public APIs
- **Docstrings**: Google style docstrings

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
            
        Example:
            >>> template = HardwareTemplate("systolic", {"rows": 8, "cols": 8})
            >>> rtl = template.generate_rtl()
        """
        if not self.parameters:
            raise ValueError("Template parameters are required")
        
        # Implementation here
        return "// Generated RTL code"
```

### TypeScript/JavaScript Style

```typescript
// Use Prettier formatting (automatic)
// Prefer interfaces over types for objects
// Use async/await over Promises

interface AcceleratorConfig {
  computeUnits: number;
  memoryHierarchy: string[];
  dataflow: 'weight_stationary' | 'output_stationary';
}

export class AcceleratorDesigner {
  constructor(private config: AcceleratorConfig) {}

  async design(): Promise<Accelerator | null> {
    try {
      // Implementation here
      return accelerator;
    } catch (error) {
      console.error('Design failed:', error);
      return null;
    }
  }
}
```

### Hardware Description Language (HDL)

For Verilog/SystemVerilog code:

- **Indentation**: 2 spaces
- **Naming**: `snake_case` for signals, `PascalCase` for modules
- **Comments**: Explain complex logic and interfaces
- **Clocking**: Use consistent clock and reset naming

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

### Code Formatting

```bash
# Format Python code with Black
black src tests

# Sort imports
isort src tests

# Lint with Ruff
ruff check src tests

# Type check with MyPy
mypy src

# Format JavaScript/TypeScript
npm run prettier
```

## 🧪 Testing Guidelines

### Test Coverage

- **Minimum coverage**: 80% for new code
- **Aim for**: >90% coverage on critical paths
- **Test types**: Unit, integration, and end-to-end tests
- **Test naming**: `test_<functionality>_<condition>_<expected_result>`

### Writing Tests

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

```typescript
// frontend/src/components/__tests__/AcceleratorDesigner.test.tsx
import { render, screen, fireEvent } from '@testing-library/react';
import { AcceleratorDesigner } from '../AcceleratorDesigner';

describe('AcceleratorDesigner', () => {
  it('renders design form correctly', () => {
    render(<AcceleratorDesigner />);
    expect(screen.getByText('Compute Units')).toBeInTheDocument();
  });

  it('submits form with valid data', async () => {
    const onSubmit = jest.fn();
    render(<AcceleratorDesigner onSubmit={onSubmit} />);
    
    fireEvent.change(screen.getByLabelText('Compute Units'), {
      target: { value: '64' }
    });
    fireEvent.click(screen.getByText('Design'));
    
    expect(onSubmit).toHaveBeenCalledWith({ computeUnits: 64 });
  });
});
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

## 📚 Documentation

### Types of Documentation

1. **API Documentation**: Docstrings in code (auto-generated)
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

## 🏗️ Hardware Templates

### Creating New Templates

```python
# src/templates/custom_accelerator.py
from .base_template import BaseTemplate
from typing import Dict, Any

class CustomAccelerator(BaseTemplate):
    """Custom accelerator template."""
    
    def __init__(self, **params):
        super().__init__()
        self.params = self.validate_params(params)
    
    def generate_rtl(self) -> str:
        """Generate SystemVerilog RTL."""
        # Template-specific RTL generation
        return rtl_code
    
    def estimate_resources(self) -> Dict[str, Any]:
        """Estimate hardware resources."""
        return {
            'luts': self.calculate_luts(),
            'dsps': self.calculate_dsps(),
            'bram_kb': self.calculate_memory()
        }
    
    @staticmethod
    def validate_params(params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate template parameters."""
        required = ['compute_units', 'data_width']
        for param in required:
            if param not in params:
                raise ValueError(f"Missing required parameter: {param}")
        return params
```

### Template Guidelines

- Inherit from `BaseTemplate`
- Implement required methods: `generate_rtl()`, `estimate_resources()`
- Include parameter validation
- Provide comprehensive documentation
- Add unit tests and integration tests
- Include example usage in README

## 👥 Community Guidelines

### Communication Channels

- **GitHub Issues**: Bug reports, feature requests
- **GitHub Discussions**: Questions, ideas, showcase
- **Discord**: [AI Hardware Co-Design Community](https://discord.gg/ai-hardware-codesign)
- **Email**: [community@codesign-playground.com](mailto:community@codesign-playground.com)
- **Mailing List**: Announcements and updates

### Getting Help

1. **Search existing issues** and discussions first
2. **Ask in Discord** for quick questions
3. **Create a GitHub issue** for bugs or feature requests
4. **Join community calls** (announced in Discord)
5. **Attend office hours** (schedule in Discord)

### Review Process

#### Pull Request Requirements

Before submitting a PR, ensure:

- [ ] All tests pass
- [ ] Code follows style guidelines
- [ ] Documentation is updated
- [ ] Commit messages follow convention
- [ ] PR description is clear and complete

#### PR Template

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

#### Review Criteria

Reviewers will check:

1. **Functionality**: Does the code work as intended?
2. **Quality**: Is the code well-written and maintainable?
3. **Testing**: Are there adequate tests?
4. **Documentation**: Is documentation clear and complete?
5. **Security**: Are there any security concerns?
6. **Performance**: Are there performance implications?

#### Review Timeline

- **Initial review**: Within 2-3 business days
- **Follow-up reviews**: Within 1-2 business days
- **Merge**: After approval from at least two maintainers

## 🏆 Recognition

### Contributor Types

- **Core Maintainers**: Long-term project stewards
- **Regular Contributors**: Frequent, high-quality contributions
- **Domain Experts**: Specialists in AI, hardware, or tooling
- **Community Champions**: Help others and improve docs

### Recognition Programs

- **Contributors page** on our website
- **Contributor spotlight** in monthly newsletters
- **Changelog credits** in releases
- **Community highlights** in newsletters
- **Conference speaking opportunities** (with permission)
- **Co-authorship on research papers**
- **Invitation to maintainer team**

### Hall of Fame

Top contributors are recognized in our [CONTRIBUTORS.md](CONTRIBUTORS.md) file.

## 🔧 Development Tips

### Debugging

```bash
# Backend debugging
cd backend
python -m debugpy --listen 5678 --wait-for-client -m uvicorn main:app --reload

# Frontend debugging
cd frontend
npm run dev:debug
```

### Performance Profiling

```bash
# Profile Python code
python -m cProfile -o profile.stats src/main.py
python -c "import pstats; pstats.Stats('profile.stats').sort_stats('cumulative').print_stats(20)"

# Profile Node.js
cd frontend
npm run build:analyze
```

### Database Migrations

```bash
# Create new migration
npm run migrate:create

# Apply migrations
npm run migrate
```

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

## License

By contributing to AI Hardware Co-Design Playground, you agree that your contributions will be licensed under the [MIT License](LICENSE).

---

## 📞 Questions?

If you have questions not covered here:

1. Check existing [GitHub Discussions](https://github.com/terragon-labs/ai-hardware-codesign-playground/discussions)
2. Join our [Discord community](https://discord.gg/ai-hardware-codesign)
3. Email us at: contribute@terragon-labs.com

Thank you for contributing to the AI Hardware Co-Design Playground! Your contributions help make AI hardware design more accessible to everyone. 🚀
