# Contributing to AI Hardware Co-Design Playground

Thank you for your interest in contributing to the AI Hardware Co-Design Playground! This project aims to democratize AI hardware design through an accessible, open-source platform. We welcome contributions from developers, researchers, educators, and enthusiasts at all levels.

## 📋 Table of Contents

- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Contribution Types](#contribution-types)
- [Code Guidelines](#code-guidelines)
- [Testing Requirements](#testing-requirements)
- [Documentation Standards](#documentation-standards)
- [Review Process](#review-process)
- [Community Guidelines](#community-guidelines)
- [Recognition](#recognition)

## 🚀 Getting Started

### Prerequisites

Before contributing, ensure you have:
- Python 3.9+ with pip
- Node.js 18+ with npm
- Git with conventional commit support
- Basic understanding of AI/ML and hardware design concepts

### Development Setup

```bash
# 1. Fork and clone the repository
git clone https://github.com/your-username/ai-hardware-codesign-playground.git
cd ai-hardware-codesign-playground

# 2. Set up development environment
./scripts/setup_dev.sh

# 3. Install dependencies
npm run setup

# 4. Verify installation
npm run validate

# 5. Run development servers
npm run dev
```

### Development Environment

We recommend using the provided devcontainer for consistent development:

```bash
# Using VS Code with Dev Containers extension
code .
# Then: Cmd/Ctrl+Shift+P -> "Dev Containers: Reopen in Container"

# Or using Docker directly
docker-compose -f docker-compose.dev.yml up
```

## 🔄 Development Workflow

### 1. Issue Creation

Before starting work:
- Check existing issues to avoid duplication
- Create a detailed issue describing the problem or enhancement
- Wait for maintainer approval for significant changes
- Use issue templates for consistency

### 2. Branch Strategy

```bash
# Create feature branch from main
git checkout main
git pull upstream main
git checkout -b feature/your-feature-name

# For bug fixes
git checkout -b fix/issue-description

# For documentation
git checkout -b docs/improvement-description
```

### 3. Development Process

```bash
# Make your changes
# ...

# Run tests continuously during development
npm run test:watch

# Check code quality
npm run lint
npm run typecheck

# Run full validation before committing
npm run validate
```

### 4. Commit Standards

We use [Conventional Commits](https://www.conventionalcommits.org/):

```bash
# Commit format
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]

# Examples
feat(hardware): add systolic array template
fix(api): resolve memory leak in simulation engine
docs(readme): update installation instructions
test(optimization): add unit tests for genetic algorithm
```

**Commit Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks
- `perf`: Performance improvements
- `ci`: CI/CD changes

### 5. Pull Request Process

```bash
# Push your branch
git push origin feature/your-feature-name

# Create pull request through GitHub UI
# Follow the PR template
```

## 🎯 Contribution Types

### Code Contributions

#### High-Priority Areas
- **Hardware Templates**: New accelerator architectures
- **Optimization Algorithms**: Advanced co-optimization methods
- **Tool Integrations**: Support for additional EDA tools
- **Performance Improvements**: Simulation speed optimizations
- **Bug Fixes**: Resolving issues from the issue tracker

#### Hardware Templates
```python
# Example: Adding a new hardware template
class CustomAccelerator(HardwareTemplate):
    """Custom accelerator template with specific optimizations."""
    
    def __init__(self, config: CustomConfig):
        super().__init__()
        self.config = config
    
    def generate_rtl(self) -> str:
        """Generate RTL for the custom accelerator."""
        # Implementation here
        pass
    
    def estimate_resources(self) -> ResourceEstimate:
        """Estimate hardware resource requirements."""
        # Implementation here
        pass
```

#### Optimization Algorithms
```python
# Example: Adding a new optimization algorithm
class CustomOptimizer(BaseOptimizer):
    """Custom optimization algorithm for co-design."""
    
    def optimize(self, design_space: DesignSpace) -> OptimizationResult:
        """Perform optimization over the design space."""
        # Implementation here
        pass
```

### Documentation Contributions

#### Tutorials and Guides
- Step-by-step tutorials for beginners
- Advanced usage examples
- Best practices documentation
- API reference improvements

#### Educational Content
- Course materials and assignments
- Video tutorials and presentations
- Interactive Jupyter notebooks
- Conference talks and papers

### Testing Contributions

#### Test Categories
- Unit tests for individual components
- Integration tests for workflows
- Performance benchmarks
- End-to-end system tests

```python
# Example: Adding unit tests
import pytest
from codesign_playground import AcceleratorDesigner

class TestAcceleratorDesigner:
    def test_systolic_array_generation(self):
        """Test systolic array hardware generation."""
        designer = AcceleratorDesigner()
        accelerator = designer.design(
            template="systolic_array",
            rows=16,
            cols=16
        )
        
        assert accelerator.dimensions == (16, 16)
        assert accelerator.rtl_code is not None
        assert accelerator.resource_estimate.dsps > 0
```

### Community Contributions

- Bug reports with detailed reproduction steps
- Feature requests with clear use cases
- Community support in discussions and forums
- Code reviews for other contributors
- Translation of documentation

## 📝 Code Guidelines

### Python Code Style

```python
# Use type hints for all function signatures
def analyze_model(model: torch.nn.Module, 
                 input_shape: Tuple[int, ...]) -> ModelProfile:
    """Analyze model computational requirements.
    
    Args:
        model: PyTorch model to analyze
        input_shape: Input tensor shape
        
    Returns:
        ModelProfile containing analysis results
        
    Raises:
        ValueError: If input_shape is invalid
    """
    # Implementation here
    pass

# Use dataclasses for configuration objects
@dataclass
class HardwareConfig:
    """Configuration for hardware generation."""
    compute_units: int
    memory_size_kb: int
    frequency_mhz: float
    precision: str = "int8"
```

### Code Quality Standards

- **Type Hints**: Required for all public APIs
- **Docstrings**: Google-style docstrings for all public functions
- **Error Handling**: Explicit exception handling with descriptive messages
- **Logging**: Structured logging with appropriate levels
- **Performance**: Profile code for performance-critical paths

### Project Structure

```
src/
├── codesign_playground/
│   ├── core/              # Core functionality
│   ├── hardware/          # Hardware templates
│   ├── optimization/      # Optimization algorithms
│   ├── simulation/        # Simulation engines
│   ├── utils/             # Utility functions
│   └── api/               # REST API endpoints
├── tests/
│   ├── unit/              # Unit tests
│   ├── integration/       # Integration tests
│   └── fixtures/          # Test data
└── docs/
    ├── api/               # API documentation
    ├── tutorials/         # User tutorials
    └── guides/            # Developer guides
```

## 🧪 Testing Requirements

### Test Coverage
- Minimum 90% code coverage for new contributions
- 100% coverage for critical paths (optimization, simulation)
- All public APIs must have corresponding tests

### Test Categories

```bash
# Unit tests (fast, isolated)
npm run test:unit

# Integration tests (slower, multiple components)
npm run test:integration

# End-to-end tests (full workflows)
npm run test:e2e

# Performance tests
npm run test:performance

# Run all tests
npm run test
```

### Test Guidelines

```python
# Test naming convention
def test_component_action_expected_result():
    """Test description explaining what is being tested."""
    # Arrange
    input_data = create_test_data()
    
    # Act
    result = component.action(input_data)
    
    # Assert
    assert result.success
    assert result.value == expected_value
```

## 📚 Documentation Standards

### Documentation Types

1. **API Documentation**: Auto-generated from docstrings
2. **User Guides**: Step-by-step instructions for common tasks
3. **Developer Guides**: Technical implementation details
4. **Tutorials**: Learning-oriented, hands-on examples
5. **Reference**: Comprehensive technical specifications

### Writing Guidelines

- Use clear, concise language
- Include practical examples
- Provide both conceptual explanations and implementation details
- Keep documentation up-to-date with code changes
- Use consistent terminology throughout

### Documentation Structure

```markdown
# Title

## Overview
Brief description and motivation

## Prerequisites
Required knowledge and setup

## Step-by-Step Guide
1. Clear, actionable steps
2. Code examples with explanations
3. Expected outputs

## Advanced Usage
Optional advanced topics

## Troubleshooting
Common issues and solutions

## References
Links to related documentation
```

## 🔍 Review Process

### Automated Checks

All contributions must pass:
- Code linting and formatting
- Type checking
- Unit and integration tests
- Security vulnerability scanning
- Documentation builds

### Human Review

1. **Technical Review**: Code quality, architecture, performance
2. **Domain Review**: Hardware design correctness, optimization validity
3. **Documentation Review**: Clarity, completeness, accuracy
4. **User Experience Review**: API usability, error messages

### Review Criteria

- ✅ Follows coding standards and best practices
- ✅ Includes comprehensive tests
- ✅ Updates relevant documentation
- ✅ Maintains backward compatibility
- ✅ Addresses security considerations
- ✅ Provides clear commit messages

### Reviewer Guidelines

- Provide constructive, specific feedback
- Suggest improvements rather than just pointing out problems
- Ask questions to understand the approach
- Appreciate good solutions and clean code
- Be responsive to contributor questions

## 🏆 Recognition

### Contributor Levels

**🌟 First-time Contributors**
- Welcome package with project stickers
- Mentorship from experienced contributors
- Recognition in monthly newsletter

**🚀 Regular Contributors**
- Invitation to contributor Discord channel
- Early access to new features
- Conference speaking opportunities

**💎 Core Contributors**
- Commit access to repository
- Participation in architectural decisions
- Recognition at conferences and events

### Contribution Tracking

We track and celebrate contributions:
- Code contributions (commits, reviews)
- Documentation improvements
- Bug reports and feature requests
- Community support and mentoring
- Educational content creation

## 🤝 Community Guidelines

### Code of Conduct

All contributors must follow our [Code of Conduct](CODE_OF_CONDUCT.md). We are committed to providing a welcoming and inclusive environment for everyone.

### Communication Channels

- **GitHub Issues**: Bug reports, feature requests
- **GitHub Discussions**: General questions, ideas
- **Discord**: Real-time community chat
- **Monthly Calls**: Open community meetings

### Getting Help

- Check existing documentation and issues first
- Use appropriate communication channels
- Provide detailed context when asking questions
- Be patient and respectful with community members

### Mentorship Program

New contributors can request mentorship:
1. Comment on a "good first issue"
2. Mention you'd like mentorship
3. A mentor will be assigned within 48 hours
4. Regular check-ins and code review support

## 📞 Contact

- **Maintainers**: @daniel-schmidt, @team-terragon
- **Email**: contributors@terragon-labs.com
- **Discord**: [Join our community](https://discord.gg/terragon-labs)
- **Office Hours**: Thursdays 2-3 PM UTC

Thank you for contributing to democratizing AI hardware design! Your efforts help make advanced hardware design accessible to researchers, educators, and innovators worldwide. 🎉