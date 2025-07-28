# Contributing to AI Hardware Co-Design Playground

Thank you for your interest in contributing to the AI Hardware Co-Design Playground! This project aims to democratize AI hardware design through an open and collaborative community.

## 🌟 How to Contribute

We welcome contributions of all types:
- 🐛 Bug reports and fixes
- ✨ New features and enhancements
- 📚 Documentation improvements
- 🧪 Tests and benchmarks
- 🏗️ Hardware templates
- 🔧 Tool integrations

## 📋 Table of Contents

- [Getting Started](#getting-started)
- [Development Environment](#development-environment)
- [Contribution Workflow](#contribution-workflow)
- [Code Standards](#code-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Hardware Templates](#hardware-templates)
- [Community Guidelines](#community-guidelines)
- [Recognition](#recognition)

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- Node.js 18+
- Git
- Docker (optional)

### Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR-USERNAME/ai-hardware-codesign-playground.git
cd ai-hardware-codesign-playground

# Add upstream remote
git remote add upstream https://github.com/terragon-labs/ai-hardware-codesign-playground.git
```

## 🔧 Development Environment

### Quick Setup

```bash
# Install dependencies and set up development environment
npm run setup

# Start development servers
npm run dev
```

### Manual Setup

```bash
# Backend setup
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e .[dev,test,docs]

# Frontend setup
cd ../frontend
npm install

# Install pre-commit hooks
npm run postinstall
```

### Docker Development

```bash
# Start full development environment
npm run docker:dev

# Or build and run manually
docker-compose -f docker-compose.dev.yml up --build
```

### Development Tools

- **Code Formatting**: Black (Python), Prettier (JS/TS)
- **Linting**: Pylint, ESLint
- **Type Checking**: MyPy, TypeScript
- **Testing**: Pytest, Jest
- **Documentation**: Sphinx, Storybook

## 🔄 Contribution Workflow

### 1. Create an Issue

Before starting work, create an issue to:
- Report bugs with reproduction steps
- Propose new features with use cases
- Discuss architectural changes
- Ask questions or seek clarification

### 2. Create a Branch

```bash
# Update your main branch
git checkout main
git pull upstream main

# Create a feature branch
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-number-description
```

### 3. Make Changes

- Write clean, readable code
- Follow established patterns
- Add tests for new functionality
- Update documentation as needed
- Commit frequently with clear messages

### 4. Test Your Changes

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

### 5. Submit a Pull Request

```bash
# Push your branch
git push origin feature/your-feature-name

# Create a pull request on GitHub with:
# - Clear title and description
# - Link to related issues
# - Screenshots/demos if applicable
# - Checklist completion
```

## 📝 Code Standards

### Python Code Style

```python
# Use Black formatting (automatic)
# Follow PEP 8 conventions
# Use type hints

from typing import Dict, List, Optional
import numpy as np

class AcceleratorDesigner:
    """Design custom AI accelerators."""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
    
    def design(
        self, 
        compute_units: int,
        memory_hierarchy: List[str]
    ) -> Optional[Accelerator]:
        """Design an accelerator with specified parameters."""
        if compute_units <= 0:
            raise ValueError("Compute units must be positive")
        
        # Implementation here
        return accelerator
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

### Commit Message Format

```
type(scope): brief description

Longer description if needed.

Fixes #123
```

**Types**: feat, fix, docs, style, refactor, test, chore  
**Scopes**: frontend, backend, docs, ci, templates, etc.

Examples:
```
feat(templates): add transformer accelerator template
fix(backend): resolve memory leak in simulation engine
docs(api): update hardware design API documentation
```

## 🧪 Testing Guidelines

### Writing Tests

```python
# backend/tests/test_accelerator_designer.py
import pytest
from src.accelerator_designer import AcceleratorDesigner

class TestAcceleratorDesigner:
    def test_design_with_valid_params(self):
        designer = AcceleratorDesigner(config={})
        accelerator = designer.design(
            compute_units=64,
            memory_hierarchy=['sram_64kb', 'dram']
        )
        assert accelerator is not None
        assert accelerator.compute_units == 64
    
    def test_design_with_invalid_params(self):
        designer = AcceleratorDesigner(config={})
        with pytest.raises(ValueError):
            designer.design(compute_units=-1, memory_hierarchy=[])
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

### Test Coverage

- Aim for >90% coverage on new code
- Include edge cases and error conditions
- Test both happy path and failure scenarios
- Use integration tests for complex workflows

## 📚 Documentation

### Code Documentation

```python
def generate_rtl(
    self, 
    output_file: str,
    include_testbench: bool = True
) -> None:
    """Generate RTL code for the accelerator.
    
    Args:
        output_file: Path to output Verilog file
        include_testbench: Whether to include verification testbench
        
    Raises:
        ValueError: If output_file path is invalid
        RuntimeError: If RTL generation fails
        
    Example:
        >>> designer = AcceleratorDesigner(config)
        >>> designer.generate_rtl('accelerator.v')
    """
```

### API Documentation

- Use docstrings for all public methods
- Include parameter types and descriptions
- Provide usage examples
- Document exceptions and error conditions

### User Documentation

- Update README.md for user-facing changes
- Add tutorials for new features
- Include code examples and screenshots
- Update API reference documentation

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

### Code of Conduct

We are committed to providing a welcoming and inclusive experience for everyone. Please read our [Code of Conduct](CODE_OF_CONDUCT.md).

### Communication Channels

- **GitHub Issues**: Bug reports, feature requests
- **GitHub Discussions**: Questions, ideas, showcase
- **Discord**: Real-time chat and collaboration
- **Mailing List**: Announcements and updates

### Getting Help

- Search existing issues and discussions
- Ask questions in GitHub Discussions
- Join our Discord community
- Attend office hours (schedule in Discord)

### Review Process

1. **Automated Checks**: CI/CD runs tests, linting, security scans
2. **Code Review**: Core maintainers review all contributions
3. **Testing**: Contributors verify changes work as expected
4. **Documentation**: Updates are reviewed for clarity and completeness
5. **Approval**: Two maintainer approvals required for merge

## 🏆 Recognition

### Contributor Types

- **Core Maintainers**: Long-term project stewards
- **Regular Contributors**: Frequent, high-quality contributions
- **Domain Experts**: Specialists in AI, hardware, or tooling
- **Community Champions**: Help others and improve docs

### Recognition Programs

- Contributor spotlight in monthly newsletters
- Conference speaking opportunities
- Co-authorship on research papers
- Invitation to maintainer team

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

## 📞 Questions?

If you have questions not covered here:

1. Check existing [GitHub Discussions](https://github.com/terragon-labs/ai-hardware-codesign-playground/discussions)
2. Join our [Discord community](https://discord.gg/ai-hardware-codesign)
3. Email us at: contribute@terragon-labs.com

Thank you for contributing to the AI Hardware Co-Design Playground! 🚀