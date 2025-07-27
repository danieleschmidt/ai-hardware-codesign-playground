# Multi-stage Dockerfile for AI Hardware Co-Design Playground
# Optimized for development, testing, and production deployment

# =============================================================================
# Base Stage: Common dependencies and system setup
# =============================================================================
FROM ubuntu:22.04 as base

LABEL maintainer="AI Hardware Co-Design Team <team@codesign-playground.com>"
LABEL description="AI Hardware Co-Design Playground - Interactive environment for co-optimizing neural networks and hardware accelerators"
LABEL version="0.1.0"

# Prevent interactive prompts during apt installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # Build tools
    build-essential \
    cmake \
    ninja-build \
    pkg-config \
    git \
    curl \
    wget \
    unzip \
    # Python
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    # Hardware design tools
    verilator \
    yosys \
    gtkwave \
    iverilog \
    # Libraries
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
    liblzma-dev \
    # Graphics and visualization
    graphviz \
    # Clean up
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create python3 symlink
RUN ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.11 /usr/bin/python

# Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# Create app user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory
WORKDIR /app

# =============================================================================
# Development Stage: Include development tools and dependencies
# =============================================================================
FROM base as development

# Install additional development tools
RUN apt-get update && apt-get install -y \
    # Development tools
    vim \
    nano \
    htop \
    tree \
    jq \
    # Debugging tools
    gdb \
    valgrind \
    strace \
    # Node.js for web development
    nodejs \
    npm \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python development dependencies
COPY requirements-dev.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements-dev.txt

# Copy source code
COPY . /app/

# Install package in development mode
RUN pip install -e ".[dev,test,docs]"

# Setup pre-commit hooks
RUN pre-commit install --install-hooks || true

# Change ownership to app user
RUN chown -R appuser:appuser /app

USER appuser

# Expose ports for development
EXPOSE 8000 8888 6006 3000

# Default command for development
CMD ["bash"]

# =============================================================================
# Production Dependencies Stage: Install only production dependencies
# =============================================================================
FROM base as prod-deps

# Copy requirements
COPY requirements.txt /tmp/
COPY pyproject.toml setup.py /tmp/

# Install production dependencies
RUN pip install --no-cache-dir --no-deps -r /tmp/requirements.txt

# =============================================================================
# Testing Stage: Include test dependencies and run tests
# =============================================================================
FROM development as testing

# Copy test requirements
COPY requirements-test.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements-test.txt

# Copy test files
COPY tests/ /app/tests/
COPY pytest.ini /app/
COPY .coveragerc /app/

# Run tests
RUN python -m pytest tests/ --cov=src --cov-report=html --cov-report=term-missing

# =============================================================================
# Documentation Stage: Build documentation
# =============================================================================
FROM development as docs

# Install documentation dependencies
COPY requirements-docs.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements-docs.txt

# Copy documentation source
COPY docs/ /app/docs/
COPY mkdocs.yml /app/

# Build documentation
RUN mkdocs build

# =============================================================================
# Production Stage: Minimal image for production deployment
# =============================================================================
FROM prod-deps as production

# Security: Don't run as root
USER appuser

# Copy application code
COPY --chown=appuser:appuser src/ /app/src/
COPY --chown=appuser:appuser pyproject.toml setup.py README.md LICENSE /app/

# Install the package
RUN pip install --no-cache-dir .

# Copy configuration files
COPY --chown=appuser:appuser .env.example /app/
COPY --chown=appuser:appuser scripts/entrypoint.sh /app/

# Make entrypoint executable
RUN chmod +x /app/entrypoint.sh

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import codesign_playground; print('OK')" || exit 1

# Expose application port
EXPOSE 8000

# Set entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]

# Default command
CMD ["codesign-playground", "serve", "--host", "0.0.0.0", "--port", "8000"]

# =============================================================================
# Jupyter Stage: Jupyter Lab environment for interactive development
# =============================================================================
FROM development as jupyter

# Install Jupyter and extensions
RUN pip install --no-cache-dir \
    jupyterlab \
    jupyter-server-proxy \
    jupyterlab-git \
    jupyterlab-widgets \
    ipywidgets

# Setup Jupyter configuration
RUN mkdir -p /home/appuser/.jupyter
COPY --chown=appuser:appuser docker/jupyter_lab_config.py /home/appuser/.jupyter/

# Copy example notebooks
COPY --chown=appuser:appuser notebooks/ /app/notebooks/

USER appuser

# Expose Jupyter port
EXPOSE 8888

# Start Jupyter Lab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--notebook-dir=/app"]

# =============================================================================
# GPU Stage: CUDA-enabled environment for GPU acceleration
# =============================================================================
FROM nvidia/cuda:11.8-devel-ubuntu22.04 as gpu

# Copy base setup
COPY --from=base /usr/bin/python3 /usr/bin/python3
COPY --from=base /usr/local/lib/python3.11 /usr/local/lib/python3.11

# Install CUDA-specific dependencies
RUN pip install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 \
    cupy-cuda11x \
    tensorflow[and-cuda]

# Copy application
COPY --from=production /app /app

USER appuser
WORKDIR /app

# Default command
CMD ["codesign-playground", "serve", "--device", "cuda"]

# =============================================================================
# CI/CD Stage: Specialized for continuous integration
# =============================================================================
FROM testing as ci

# Install additional CI tools
RUN pip install --no-cache-dir \
    codecov \
    pytest-cov \
    pytest-xdist \
    pytest-benchmark

# Copy CI scripts
COPY --chown=appuser:appuser scripts/ci/ /app/scripts/ci/

# Make CI scripts executable
RUN chmod +x /app/scripts/ci/*.sh

# Default command for CI
CMD ["/app/scripts/ci/run_tests.sh"]