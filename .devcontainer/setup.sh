#!/bin/bash
set -e

echo "🚀 Setting up AI Hardware Co-Design Playground development environment..."

# Update package lists
sudo apt-get update

# Install system dependencies
echo "📦 Installing system dependencies..."
sudo apt-get install -y \
    build-essential \
    cmake \
    ninja-build \
    verilator \
    gtkwave \
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
    liblzma-dev \
    graphviz \
    graphviz-dev \
    curl \
    wget \
    git-lfs \
    jq

# Install Python dependencies
echo "🐍 Installing Python development tools..."
pip install --upgrade pip setuptools wheel

# Install Poetry for dependency management
curl -sSL https://install.python-poetry.org | python3 -
export PATH="/home/vscode/.local/bin:$PATH"
echo 'export PATH="/home/vscode/.local/bin:$PATH"' >> /home/vscode/.bashrc

# Install pre-commit
pip install pre-commit

# Install Python development packages
pip install \
    black \
    isort \
    pylint \
    mypy \
    flake8 \
    pytest \
    pytest-cov \
    pytest-xdist \
    pytest-mock \
    hypothesis \
    coverage \
    bandit \
    safety \
    jupyterlab \
    notebook \
    ipywidgets \
    matplotlib \
    plotly \
    seaborn \
    pandas \
    numpy \
    scipy \
    scikit-learn \
    torch \
    torchvision \
    onnx \
    onnxruntime

# Install Node.js dependencies globally
echo "📦 Installing Node.js global packages..."
npm install -g \
    @angular/cli \
    @vue/cli \
    create-react-app \
    typescript \
    ts-node \
    nodemon \
    prettier \
    eslint \
    @typescript-eslint/parser \
    @typescript-eslint/eslint-plugin

# Install TVM (if available in package manager or build from source)
echo "🔧 Installing TVM and MLIR dependencies..."
pip install \
    tvm \
    mlir-core \
    apache-tvm

# Install hardware simulation tools
echo "⚡ Setting up hardware simulation environment..."
# Verilator is already installed via apt
# Add any additional simulation tools here

# Install documentation tools
echo "📚 Installing documentation tools..."
pip install \
    sphinx \
    sphinx-rtd-theme \
    sphinx-autodoc-typehints \
    myst-parser \
    mkdocs \
    mkdocs-material \
    mkdocs-mermaid2-plugin

# Set up Git hooks
echo "🔗 Setting up Git hooks..."
git config --global user.name "Development Container"
git config --global user.email "dev@terragon-labs.com"
git config --global init.defaultBranch main
git config --global pull.rebase false

# Install GitHub CLI if not present
if ! command -v gh &> /dev/null; then
    echo "📱 Installing GitHub CLI..."
    curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null
    sudo apt update
    sudo apt install gh
fi

# Create workspace directories
echo "📁 Creating workspace directories..."
mkdir -p \
    /workspace-cache/pip \
    /workspace-cache/npm \
    /workspace-cache/poetry \
    "$WORKSPACE_FOLDER/src" \
    "$WORKSPACE_FOLDER/tests" \
    "$WORKSPACE_FOLDER/docs" \
    "$WORKSPACE_FOLDER/examples" \
    "$WORKSPACE_FOLDER/scripts" \
    "$WORKSPACE_FOLDER/frontend" \
    "$WORKSPACE_FOLDER/backend" \
    "$WORKSPACE_FOLDER/hardware" \
    "$WORKSPACE_FOLDER/notebooks"

# Set up cache directories
export PIP_CACHE_DIR="/workspace-cache/pip"
export NPM_CONFIG_CACHE="/workspace-cache/npm"
export POETRY_CACHE_DIR="/workspace-cache/poetry"

echo 'export PIP_CACHE_DIR="/workspace-cache/pip"' >> /home/vscode/.bashrc
echo 'export NPM_CONFIG_CACHE="/workspace-cache/npm"' >> /home/vscode/.bashrc
echo 'export POETRY_CACHE_DIR="/workspace-cache/poetry"' >> /home/vscode/.bashrc

# Install development environment dependencies if they exist
if [ -f "$WORKSPACE_FOLDER/pyproject.toml" ]; then
    echo "📦 Installing project dependencies with Poetry..."
    cd "$WORKSPACE_FOLDER"
    poetry install --with dev,test,docs
fi

if [ -f "$WORKSPACE_FOLDER/requirements.txt" ]; then
    echo "📦 Installing requirements.txt dependencies..."
    pip install -r "$WORKSPACE_FOLDER/requirements.txt"
fi

if [ -f "$WORKSPACE_FOLDER/requirements-dev.txt" ]; then
    echo "📦 Installing development requirements..."
    pip install -r "$WORKSPACE_FOLDER/requirements-dev.txt"
fi

# Set up frontend dependencies if package.json exists
if [ -f "$WORKSPACE_FOLDER/frontend/package.json" ]; then
    echo "📦 Installing frontend dependencies..."
    cd "$WORKSPACE_FOLDER/frontend"
    npm install
fi

# Install Jupyter extensions
echo "📊 Setting up Jupyter Lab extensions..."
jupyter labextension install \
    @jupyter-widgets/jupyterlab-manager \
    @jupyterlab/git \
    @ryantam626/jupyterlab_code_formatter \
    jupyterlab-plotly \
    @jupyterlab/toc

# Enable Jupyter Lab extensions
jupyter lab build

# Set up development aliases
echo "⚡ Setting up development aliases..."
cat >> /home/vscode/.bashrc << 'EOF'

# Development aliases
alias ll='ls -alF'
alias la='ls -A'
alias l='ls -CF'
alias grep='grep --color=auto'
alias ..='cd ..'
alias ...='cd ../..'

# Project-specific aliases
alias pytest-cov='pytest --cov=src --cov-report=html --cov-report=term'
alias lint='black . && isort . && pylint src/ && mypy src/'
alias format='black . && isort .'
alias test='pytest -v'
alias test-watch='pytest-watch'
alias serve-docs='mkdocs serve'
alias build-docs='mkdocs build'
alias notebook='jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root'

# Git aliases
alias gs='git status'
alias ga='git add'
alias gc='git commit'
alias gp='git push'
alias gl='git pull'
alias gd='git diff'
alias gb='git branch'
alias gco='git checkout'

# Docker aliases
alias dps='docker ps'
alias dpa='docker ps -a'
alias di='docker images'
alias dex='docker exec -it'
alias dlogs='docker logs -f'

EOF

# Create a development script
cat > "$WORKSPACE_FOLDER/dev" << 'EOF'
#!/bin/bash
# Development helper script

case "$1" in
    "setup")
        echo "Setting up development environment..."
        poetry install --with dev,test,docs
        pre-commit install
        echo "Development environment ready!"
        ;;
    "test")
        echo "Running tests..."
        pytest --cov=src --cov-report=html --cov-report=term
        ;;
    "lint")
        echo "Running linters..."
        black .
        isort .
        pylint src/
        mypy src/
        ;;
    "format")
        echo "Formatting code..."
        black .
        isort .
        ;;
    "docs")
        echo "Building documentation..."
        mkdocs build
        ;;
    "serve-docs")
        echo "Serving documentation..."
        mkdocs serve --dev-addr=0.0.0.0:8000
        ;;
    "clean")
        echo "Cleaning up..."
        find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
        find . -type f -name "*.pyc" -delete
        rm -rf .coverage htmlcov/ .pytest_cache/ dist/ build/
        ;;
    "jupyter")
        echo "Starting Jupyter Lab..."
        jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
        ;;
    *)
        echo "Usage: ./dev {setup|test|lint|format|docs|serve-docs|clean|jupyter}"
        echo ""
        echo "Commands:"
        echo "  setup      - Install dependencies and setup pre-commit hooks"
        echo "  test       - Run tests with coverage"
        echo "  lint       - Run all linters and type checkers"
        echo "  format     - Format code with black and isort"
        echo "  docs       - Build documentation"
        echo "  serve-docs - Serve documentation locally"
        echo "  clean      - Clean up generated files"
        echo "  jupyter    - Start Jupyter Lab server"
        ;;
esac
EOF

chmod +x "$WORKSPACE_FOLDER/dev"

# Set up shell completion
echo "🔧 Setting up shell completions..."
# GitHub CLI completion
gh completion -s bash > /tmp/gh_completion
sudo mv /tmp/gh_completion /etc/bash_completion.d/gh

# Poetry completion
poetry completions bash > /tmp/poetry_completion
sudo mv /tmp/poetry_completion /etc/bash_completion.d/poetry

# Create welcome message
cat > /home/vscode/.hushlogin << 'EOF'
EOF

cat >> /home/vscode/.bashrc << 'EOF'

# Welcome message
echo ""
echo "🚀 Welcome to AI Hardware Co-Design Playground Development Environment!"
echo ""
echo "Quick commands:"
echo "  ./dev setup     - Set up development environment"
echo "  ./dev test      - Run tests"
echo "  ./dev lint      - Run linters"
echo "  ./dev jupyter   - Start Jupyter Lab"
echo ""
echo "Happy coding! 🎯"
echo ""
EOF

# Fix permissions
sudo chown -R vscode:vscode /home/vscode
sudo chown -R vscode:vscode "$WORKSPACE_FOLDER" 2>/dev/null || true

echo "✅ Development environment setup complete!"
echo ""
echo "Next steps:"
echo "1. Run './dev setup' to install project dependencies"
echo "2. Run './dev test' to verify everything is working"
echo "3. Start coding! 🎯"