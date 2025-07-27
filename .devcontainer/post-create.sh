#!/bin/bash

# AI Hardware Co-Design Playground Development Environment Setup

set -e

echo "🚀 Setting up AI Hardware Co-Design Playground development environment..."

# Update system packages
echo "📦 Updating system packages..."
sudo apt-get update

# Install system dependencies for hardware tools
echo "🔧 Installing system dependencies..."
sudo apt-get install -y \
    build-essential \
    cmake \
    ninja-build \
    clang \
    llvm \
    verilator \
    gtkwave \
    iverilog \
    yosys \
    graphviz \
    pkg-config \
    libssl-dev \
    libffi-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    libncurses5-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    libffi-dev \
    liblzma-dev

# Install Python dependencies
echo "🐍 Installing Python dependencies..."
pip install --upgrade pip setuptools wheel

# Install development dependencies
pip install -e ".[dev,test,docs]" || echo "Warning: Could not install package in development mode yet"

# Alternative: Install common dependencies if setup.py doesn't exist yet
pip install \
    numpy \
    scipy \
    pandas \
    matplotlib \
    plotly \
    seaborn \
    jupyter \
    jupyterlab \
    notebook \
    ipywidgets \
    torch \
    torchvision \
    tensorflow \
    onnx \
    onnxruntime \
    pytest \
    pytest-cov \
    pytest-xdist \
    pytest-mock \
    hypothesis \
    black \
    mypy \
    ruff \
    pre-commit \
    sphinx \
    sphinx-rtd-theme \
    mkdocs \
    mkdocs-material \
    fastapi \
    uvicorn \
    pydantic \
    sqlalchemy \
    alembic \
    redis \
    ray \
    pybind11 \
    click \
    typer \
    rich \
    tqdm

# Install Jupyter extensions
echo "📓 Setting up Jupyter extensions..."
jupyter labextension install @jupyter-widgets/jupyterlab-manager || true
jupyter labextension install plotlywidget || true

# Setup pre-commit hooks
echo "🪝 Setting up pre-commit hooks..."
if [ -f ".pre-commit-config.yaml" ]; then
    pre-commit install
    pre-commit install --hook-type commit-msg
else
    echo "Warning: .pre-commit-config.yaml not found, skipping pre-commit setup"
fi

# Setup git configuration for development
echo "🔧 Configuring git..."
git config --global core.autocrlf input
git config --global pull.rebase false
git config --global init.defaultBranch main

# Create necessary directories
echo "📁 Creating project directories..."
mkdir -p \
    src/codesign_playground \
    tests/unit \
    tests/integration \
    tests/e2e \
    docs/guides \
    docs/tutorials \
    docs/api \
    examples \
    benchmarks \
    scripts \
    data \
    models \
    outputs \
    logs

# Create initial Python package structure
echo "🏗️ Creating Python package structure..."
touch src/codesign_playground/__init__.py
touch tests/__init__.py
touch tests/unit/__init__.py
touch tests/integration/__init__.py
touch tests/e2e/__init__.py

# Set up example notebooks
echo "📚 Setting up example notebooks..."
mkdir -p notebooks/examples
mkdir -p notebooks/tutorials
mkdir -p notebooks/benchmarks

# Download sample models for testing (if internet available)
echo "📥 Setting up sample models..."
mkdir -p models/sample
# Note: Add model downloads here if needed

# Setup environment variables
echo "🌍 Setting up environment variables..."
if [ ! -f ".env" ]; then
    cp .env.example .env 2>/dev/null || echo "Warning: .env.example not found"
fi

# Initialize database (if applicable)
echo "🗄️ Initializing database..."
# Add database initialization here if needed

# Build documentation (if available)
echo "📖 Building initial documentation..."
if [ -f "docs/requirements.txt" ]; then
    pip install -r docs/requirements.txt
fi

if [ -f "mkdocs.yml" ]; then
    mkdocs build || echo "Warning: Could not build docs yet"
fi

# Install additional development tools
echo "🛠️ Installing additional development tools..."
pip install \
    ipdb \
    memory-profiler \
    line-profiler \
    py-spy \
    snakeviz \
    scalene

# Setup shell aliases and functions
echo "💡 Setting up development aliases..."
cat >> ~/.bashrc << 'EOF'

# AI Hardware Co-Design Playground aliases
alias pytest-cov='pytest --cov=src --cov-report=html --cov-report=term'
alias pytest-fast='pytest -x -v'
alias pytest-parallel='pytest -n auto'
alias format-code='black src tests && ruff check --fix src tests'
alias type-check='mypy src'
alias docs-serve='mkdocs serve'
alias jupyter-lab='jupyter lab --ip=0.0.0.0 --allow-root --no-browser'

# Quick development functions
codesign-test() {
    echo "🧪 Running full test suite..."
    pytest tests/ --cov=src --cov-report=term-missing
}

codesign-lint() {
    echo "🔍 Running code quality checks..."
    ruff check src tests
    black --check src tests
    mypy src
}

codesign-clean() {
    echo "🧹 Cleaning up build artifacts..."
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
    find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
    find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
    find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
    rm -rf build/ dist/ htmlcov/ .coverage
}

EOF

# Make the post-create script executable
chmod +x .devcontainer/post-create.sh

# Final setup verification
echo "✅ Verifying installation..."
python --version
pip --version
pytest --version || echo "Warning: pytest not available"
black --version || echo "Warning: black not available"
ruff --version || echo "Warning: ruff not available"
mypy --version || echo "Warning: mypy not available"
verilator --version || echo "Warning: verilator not available"

echo ""
echo "🎉 Development environment setup complete!"
echo ""
echo "📋 Next steps:"
echo "  1. Run 'codesign-test' to verify everything works"
echo "  2. Start Jupyter Lab with 'jupyter-lab'"
echo "  3. Check out the examples in notebooks/examples/"
echo "  4. Read the documentation in docs/"
echo ""
echo "🛠️ Available commands:"
echo "  - codesign-test: Run the full test suite"
echo "  - codesign-lint: Run code quality checks"
echo "  - codesign-clean: Clean up build artifacts"
echo "  - pytest-cov: Run tests with coverage"
echo "  - format-code: Format code with black and ruff"
echo "  - docs-serve: Serve documentation locally"
echo ""
echo "Happy coding! 🚀"