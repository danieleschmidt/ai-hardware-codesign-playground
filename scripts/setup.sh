#!/bin/bash

# AI Hardware Co-Design Playground Setup Script
# This script sets up the development environment for the project

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running on supported OS
check_os() {
    log_info "Checking operating system..."
    
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        OS="linux"
        log_success "Detected Linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macos"
        log_success "Detected macOS"
    elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
        OS="windows"
        log_success "Detected Windows (WSL/Cygwin)"
    else
        log_error "Unsupported operating system: $OSTYPE"
        exit 1
    fi
}

# Check Python version
check_python() {
    log_info "Checking Python version..."
    
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version | cut -d ' ' -f 2)
        PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d '.' -f 1)
        PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d '.' -f 2)
        
        if [[ $PYTHON_MAJOR -eq 3 && $PYTHON_MINOR -ge 9 ]]; then
            log_success "Python $PYTHON_VERSION found"
            PYTHON_CMD="python3"
        else
            log_error "Python 3.9+ required, found $PYTHON_VERSION"
            exit 1
        fi
    else
        log_error "Python 3 not found. Please install Python 3.9+"
        exit 1
    fi
}

# Install system dependencies
install_system_deps() {
    log_info "Installing system dependencies..."
    
    case $OS in
        "linux")
            if command -v apt-get &> /dev/null; then
                sudo apt-get update
                sudo apt-get install -y \
                    build-essential \
                    cmake \
                    ninja-build \
                    clang \
                    llvm \
                    pkg-config \
                    libssl-dev \
                    libffi-dev \
                    python3-dev \
                    python3-pip \
                    python3-venv \
                    git \
                    curl \
                    wget
                
                # Try to install hardware tools (optional)
                sudo apt-get install -y verilator yosys gtkwave iverilog || log_warning "Some hardware tools failed to install"
                
            elif command -v dnf &> /dev/null; then
                sudo dnf install -y \
                    gcc gcc-c++ \
                    cmake \
                    ninja-build \
                    clang \
                    llvm \
                    pkg-config \
                    openssl-devel \
                    libffi-devel \
                    python3-devel \
                    python3-pip \
                    git \
                    curl \
                    wget
            else
                log_warning "Unknown package manager. Please install dependencies manually."
            fi
            ;;
        "macos")
            if command -v brew &> /dev/null; then
                brew install \
                    cmake \
                    ninja \
                    llvm \
                    pkg-config \
                    openssl \
                    libffi \
                    python@3.11 \
                    git \
                    curl \
                    wget
                
                # Try to install hardware tools (optional)
                brew install verilator yosys || log_warning "Some hardware tools failed to install"
            else
                log_error "Homebrew not found. Please install Homebrew first: https://brew.sh/"
                exit 1
            fi
            ;;
        "windows")
            log_warning "Windows detected. Please ensure you have the following installed:"
            log_warning "  - Build Tools for Visual Studio 2019/2022"
            log_warning "  - CMake"
            log_warning "  - Git"
            log_warning "  - Python 3.9+"
            ;;
    esac
    
    log_success "System dependencies installation complete"
}

# Create virtual environment
setup_venv() {
    log_info "Setting up Python virtual environment..."
    
    if [[ ! -d "venv" ]]; then
        $PYTHON_CMD -m venv venv
        log_success "Virtual environment created"
    else
        log_info "Virtual environment already exists"
    fi
    
    # Activate virtual environment
    source venv/bin/activate || source venv/Scripts/activate
    
    # Upgrade pip
    pip install --upgrade pip setuptools wheel
    
    log_success "Virtual environment setup complete"
}

# Install Python dependencies
install_python_deps() {
    log_info "Installing Python dependencies..."
    
    # Install package in development mode
    pip install -e ".[dev,test,docs]"
    
    log_success "Python dependencies installation complete"
}

# Setup pre-commit hooks
setup_precommit() {
    log_info "Setting up pre-commit hooks..."
    
    if [[ -f ".pre-commit-config.yaml" ]]; then
        pre-commit install
        pre-commit install --hook-type commit-msg
        log_success "Pre-commit hooks installed"
    else
        log_warning ".pre-commit-config.yaml not found, skipping pre-commit setup"
    fi
}

# Initialize project structure
init_structure() {
    log_info "Initializing project structure..."
    
    # Create necessary directories
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
        data \
        models \
        outputs \
        logs \
        notebooks/examples \
        notebooks/tutorials \
        notebooks/benchmarks
    
    # Create __init__.py files
    touch src/codesign_playground/__init__.py
    touch tests/__init__.py
    touch tests/unit/__init__.py
    touch tests/integration/__init__.py
    touch tests/e2e/__init__.py
    
    # Copy environment file
    if [[ ! -f ".env" && -f ".env.example" ]]; then
        cp .env.example .env
        log_success "Environment file created from template"
    fi
    
    log_success "Project structure initialized"
}

# Verify installation
verify_installation() {
    log_info "Verifying installation..."
    
    # Check Python packages
    python -c "import numpy, scipy, torch, tensorflow" 2>/dev/null && log_success "Core ML packages available" || log_warning "Some ML packages missing"
    
    # Check development tools
    black --version &>/dev/null && log_success "Black formatter available" || log_warning "Black formatter missing"
    mypy --version &>/dev/null && log_success "MyPy type checker available" || log_warning "MyPy type checker missing"
    pytest --version &>/dev/null && log_success "Pytest available" || log_warning "Pytest missing"
    
    # Check hardware tools (optional)
    verilator --version &>/dev/null && log_success "Verilator available" || log_warning "Verilator not available"
    yosys -V &>/dev/null && log_success "Yosys available" || log_warning "Yosys not available"
    
    log_success "Installation verification complete"
}

# Main setup function
main() {
    echo "🚀 AI Hardware Co-Design Playground Setup"
    echo "========================================"
    
    check_os
    check_python
    install_system_deps
    setup_venv
    install_python_deps
    setup_precommit
    init_structure
    verify_installation
    
    echo ""
    echo "🎉 Setup complete!"
    echo ""
    echo "📋 Next steps:"
    echo "  1. Activate the virtual environment: source venv/bin/activate"
    echo "  2. Run tests: pytest"
    echo "  3. Start Jupyter: jupyter lab"
    echo "  4. Check out examples in notebooks/examples/"
    echo ""
    echo "🛠️ Development commands:"
    echo "  - Run tests: pytest"
    echo "  - Format code: black src tests"
    echo "  - Type check: mypy src"
    echo "  - Lint code: ruff check src tests"
    echo "  - Build docs: mkdocs build"
    echo ""
    echo "📖 Documentation: https://docs.codesign-playground.com"
    echo "💬 Community: https://github.com/terragon-labs/ai-hardware-codesign-playground/discussions"
    echo ""
    echo "Happy coding! 🚀"
}

# Run main function if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi