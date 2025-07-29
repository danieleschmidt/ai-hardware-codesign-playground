#!/bin/bash
# Setup script for AI Hardware Co-Design Playground development environment

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
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
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        OS="linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macos"
    else
        log_error "Unsupported operating system: $OSTYPE"
        exit 1
    fi
    log_info "Detected OS: $OS"
}

# Check system requirements
check_requirements() {
    log_info "Checking system requirements..."
    
    # Check Python
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
        log_success "Python $PYTHON_VERSION found"
    else
        log_error "Python 3 is required but not installed"
        exit 1
    fi
    
    # Check Node.js
    if command -v node &> /dev/null; then
        NODE_VERSION=$(node --version)
        log_success "Node.js $NODE_VERSION found"
    else
        log_error "Node.js is required but not installed"
        exit 1
    fi
    
    # Check Git
    if command -v git &> /dev/null; then
        GIT_VERSION=$(git --version | cut -d' ' -f3)
        log_success "Git $GIT_VERSION found"
    else
        log_error "Git is required but not installed"
        exit 1
    fi
    
    # Check Docker (optional)
    if command -v docker &> /dev/null; then
        DOCKER_VERSION=$(docker --version | cut -d' ' -f3 | tr -d ',')
        log_success "Docker $DOCKER_VERSION found"
    else
        log_warning "Docker not found (optional for development)"
    fi
}

# Install Python dependencies
install_python_deps() {
    log_info "Installing Python dependencies..."
    
    # Upgrade pip
    python3 -m pip install --upgrade pip
    
    # Install development dependencies
    if [ -f "requirements-dev.txt" ]; then
        python3 -m pip install -r requirements-dev.txt
        log_success "Python development dependencies installed"
    else
        log_warning "requirements-dev.txt not found, installing from pyproject.toml"
        python3 -m pip install -e ".[dev]"
    fi
}

# Install Node.js dependencies
install_node_deps() {
    log_info "Installing Node.js dependencies..."
    
    # Install root dependencies
    npm install
    log_success "Root Node.js dependencies installed"
    
    # Install frontend dependencies if frontend exists
    if [ -d "frontend" ]; then
        cd frontend
        npm install
        cd ..
        log_success "Frontend dependencies installed"
    fi
}

# Setup pre-commit hooks
setup_precommit() {
    log_info "Setting up pre-commit hooks..."
    
    if command -v pre-commit &> /dev/null; then
        pre-commit install
        pre-commit install --hook-type commit-msg
        log_success "Pre-commit hooks installed"
    else
        log_warning "pre-commit not found, installing..."
        python3 -m pip install pre-commit
        pre-commit install
        pre-commit install --hook-type commit-msg
        log_success "Pre-commit hooks installed"
    fi
}

# Create necessary directories
create_directories() {
    log_info "Creating necessary directories..."
    
    DIRS=(
        "logs"
        "uploads"
        "generated"
        "data"
        "notebooks"
        "cache"
        "temp"
        "artifacts"
    )
    
    for dir in "${DIRS[@]}"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            log_success "Created directory: $dir"
        fi
    done
}

# Setup environment files
setup_environment() {
    log_info "Setting up environment configuration..."
    
    if [ ! -f ".env" ]; then
        if [ -f ".env.example" ]; then
            cp .env.example .env
            log_success "Created .env from .env.example"
            log_warning "Please update .env with your configuration"
        else
            log_warning ".env.example not found, creating basic .env"
            cat > .env << EOF
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=DEBUG
SECRET_KEY=dev-secret-key-change-in-production
DATABASE_URL=postgresql://username:password@localhost:5432/codesign_playground_dev
REDIS_URL=redis://localhost:6379/0
EOF
            log_success "Created basic .env file"
        fi
    else
        log_info ".env file already exists"
    fi
}

# Install optional hardware tools
install_hardware_tools() {
    log_info "Installing optional hardware simulation tools..."
    
    if [[ "$OS" == "linux" ]]; then
        # Try to install Verilator on Linux
        if command -v apt-get &> /dev/null; then
            if ! command -v verilator &> /dev/null; then
                log_info "Installing Verilator via apt..."
                sudo apt-get update
                sudo apt-get install -y verilator
                log_success "Verilator installed"
            else
                log_info "Verilator already installed"
            fi
        elif command -v yum &> /dev/null; then
            log_warning "Please install Verilator manually on RHEL/CentOS"
        fi
    elif [[ "$OS" == "macos" ]]; then
        # Try to install Verilator on macOS
        if command -v brew &> /dev/null; then
            if ! command -v verilator &> /dev/null; then
                log_info "Installing Verilator via Homebrew..."
                brew install verilator
                log_success "Verilator installed"
            else
                log_info "Verilator already installed"
            fi
        else
            log_warning "Please install Homebrew and then run: brew install verilator"
        fi
    fi
}

# Run tests to verify setup
verify_setup() {
    log_info "Verifying setup..."
    
    # Test Python imports
    if python3 -c "import pytest; import black; import isort; import ruff" 2>/dev/null; then
        log_success "Python development tools working"
    else
        log_error "Python development tools not working properly"
        return 1
    fi
    
    # Test Node.js
    if node -e "console.log('Node.js working')" 2>/dev/null; then
        log_success "Node.js working"
    else
        log_error "Node.js not working properly"
        return 1
    fi
    
    # Test pre-commit
    if pre-commit --version &> /dev/null; then
        log_success "Pre-commit working"
    else
        log_warning "Pre-commit not working"
    fi
    
    log_success "Setup verification completed"
}

# Print next steps
print_next_steps() {
    echo ""
    log_success "✅ Development environment setup completed!"
    echo ""
    echo -e "${BLUE}Next steps:${NC}"
    echo "1. Update .env file with your configuration"
    echo "2. Start the development server: npm run dev"
    echo "3. Run tests: npm test"
    echo "4. Check code quality: npm run lint"
    echo ""
    echo -e "${BLUE}Useful commands:${NC}"
    echo "  make help          - Show all available commands"
    echo "  npm run dev        - Start development servers"
    echo "  npm run test       - Run all tests"
    echo "  npm run lint       - Run linting"
    echo "  docker-compose up  - Start with Docker"
    echo ""
    echo -e "${YELLOW}Note:${NC} If you encounter any issues, check the DEVELOPMENT.md guide"
}

# Main setup function
main() {
    echo -e "${BLUE}"
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║        AI Hardware Co-Design Playground Setup Script        ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    
    check_os
    check_requirements
    create_directories
    setup_environment
    install_python_deps
    install_node_deps
    setup_precommit
    
    # Optional hardware tools installation
    read -p "Install optional hardware simulation tools? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        install_hardware_tools
    fi
    
    verify_setup
    print_next_steps
}

# Handle script arguments
case "${1:-}" in
    "python")
        install_python_deps
        ;;
    "node")
        install_node_deps
        ;;
    "precommit")
        setup_precommit
        ;;
    "env")
        setup_environment
        ;;
    "verify")
        verify_setup
        ;;
    "help")
        echo "Usage: $0 [command]"
        echo ""
        echo "Commands:"
        echo "  python     - Install Python dependencies only"
        echo "  node       - Install Node.js dependencies only"
        echo "  precommit  - Setup pre-commit hooks only"
        echo "  env        - Setup environment files only"
        echo "  verify     - Verify setup only"
        echo "  help       - Show this help message"
        echo ""
        echo "Run without arguments for full setup"
        ;;
    *)
        main
        ;;
esac