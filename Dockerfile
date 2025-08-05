```dockerfile
# Multi-stage Dockerfile for AI Hardware Co-Design Playground
# Supports development, testing, and production environments

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
    libpq-dev \
    redis-tools \
    # Graphics and visualization
    graphviz \
    # Clean up
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs \
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

# Copy dependency files
COPY package.json package-lock.json* ./
COPY requirements.txt requirements-dev.txt pyproject.toml ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install Node.js dependencies
RUN npm ci --only=production

# =============================================================================
# Development Stage: Include development tools and dependencies
# =============================================================================
FROM base as development

# Install development dependencies
RUN pip install --no-cache-dir -r requirements-dev.txt \
    && npm install

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
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy source code
COPY . /app/

# Install package in development mode
RUN pip install -e ".[dev,test,docs]"

# Setup pre-commit hooks
RUN pre-commit install --install-hooks || true

# Change ownership to app user
RUN chown -R appuser:appuser /app

USER appuser

# Set environment variables for development
ENV PYTHONPATH=/app/backend:/app/hardware \
    NODE_ENV=development \
    FLASK_ENV=development

# Expose ports for development
EXPOSE 8000 3000 8888 6006

# Default command for development
CMD ["npm", "run", "dev"]

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
# Production Backend Stage
# =============================================================================
FROM base as backend-production

# Copy backend source code
COPY backend/ ./backend/
COPY tests/ ./tests/

# Install package in production mode
RUN cd backend && pip install --no-cache-dir -e .

# Copy configuration files
COPY --chown=appuser:appuser .env.example /app/
COPY --chown=appuser:appuser scripts/entrypoint.sh /app/

# Make entrypoint executable
RUN chmod +x /app/entrypoint.sh

# Change ownership to app user
RUN chown -R appuser:appuser /app

USER appuser

# Set environment variables for production
ENV PYTHONPATH=/app/backend \
    NODE_ENV=production

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Production backend command
CMD ["uvicorn", "codesign_playground.main:app", "--host", "0.0.0.0", "--port", "8000"]

# =============================================================================
# Production Frontend Stage
# =============================================================================
FROM node:18-alpine as frontend-production

WORKDIR /app

# Copy frontend source
COPY frontend/package*.json ./
COPY frontend/ ./

# Install dependencies and build
RUN npm ci --only=production \
    && npm run build

# =============================================================================
# Nginx Stage for serving frontend
# =============================================================================
FROM nginx:alpine as frontend-nginx

# Copy built frontend
COPY --from=frontend-production /app/dist /usr/share/nginx/html

# Copy nginx configuration
COPY docker/nginx.conf /etc/nginx/nginx.conf

# Expose port
EXPOSE 80

# Production frontend command
CMD ["nginx", "-g", "daemon off;"]

# =============================================================================
# Hardware Simulation Stage (with additional tools)
# =============================================================================
FROM base as hardware-simulation

# Install additional Python packages for hardware simulation
RUN pip install --no-cache-dir \
    cocotb \
    cocotb-test \
    pyverilog \
    nmigen \
    litex

# Copy source code
COPY backend/ ./backend/
COPY hardware/ ./hardware/
COPY tests/ ./tests/

# Create hardware tools user
RUN useradd --create-home --shell /bin/bash --uid 1002 hwuser \
    && chown -R hwuser:hwuser /app

USER hwuser

# Set environment for hardware simulation
ENV PYTHONPATH=/app/backend:/app/hardware \
    COCOTB_REDUCED_LOG_FMT=1

# Expose ports for simulation and debugging
EXPOSE 8000 8888 5000

# Hardware simulation command
CMD ["python", "-m", "codesign_playground.simulation.server"]

# =============================================================================
# Jupyter Stage: Jupyter Lab environment for interactive development
# =============================================================================
FROM development as jupyter

# Install Jupyter and extensions
RUN pip install --no-cache-dir \
    jupyterlab \
    jupyter-server-proxy \
    jupyter-lsp \
    jupyterlab-git \
    jupyterlab-code-formatter \
    jupyterlab-widgets \
    ipywidgets

# Create notebooks directory and setup Jupyter configuration
RUN mkdir -p /app/notebooks /home/appuser/.jupyter
COPY --chown=appuser:appuser docker/jupyter_lab_config.py /home/appuser/.jupyter/

# Copy example notebooks
COPY --chown=appuser:appuser notebooks/ /app/notebooks/

# Set Jupyter configuration
ENV JUPYTER_ENABLE_LAB=yes \
    JUPYTER_TOKEN=codesign-playground

USER appuser

# Expose Jupyter port
EXPOSE 8888

# Start Jupyter Lab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--notebook-dir=/app", "--NotebookApp.token=codesign-playground"]

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
COPY --from=backend-production /app /app

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

# =============================================================================
# All-in-one Development Stage
# =============================================================================
FROM development as all-in-one

# Install all optional dependencies
RUN pip install --no-cache-dir \
    torch \
    tensorflow \
    jupyterlab \
    cocotb

# Set comprehensive environment
ENV PYTHONPATH=/app/backend:/app/hardware \
    NODE_ENV=development

# Expose all ports
EXPOSE 3000 8000 8888 6006

# All-in-one command (using docker-compose services)
CMD ["npm", "run", "dev"]
```
