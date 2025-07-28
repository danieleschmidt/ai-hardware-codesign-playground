# Multi-stage Dockerfile for AI Hardware Co-Design Playground
# Optimized for security, performance, and minimal image size

# ============================================================================
# Build Stage - Frontend
# ============================================================================
FROM node:18-alpine AS frontend-builder

# Set working directory
WORKDIR /app/frontend

# Copy package files
COPY frontend/package*.json ./

# Install dependencies with clean install
RUN npm ci --only=production --silent

# Copy frontend source
COPY frontend/ ./

# Build frontend application
RUN npm run build

# ============================================================================
# Build Stage - Backend Dependencies
# ============================================================================
FROM python:3.11-slim AS backend-deps

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY backend/requirements*.txt ./
COPY pyproject.toml ./
COPY README.md ./

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install dependencies
RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# ============================================================================
# Build Stage - Hardware Tools
# ============================================================================
FROM ubuntu:22.04 AS hardware-tools

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install hardware design tools
RUN apt-get update && apt-get install -y \
    # Build tools
    build-essential \
    cmake \
    ninja-build \
    # Verilator for hardware simulation
    verilator \
    # Additional tools
    gtkwave \
    iverilog \
    yosys \
    # Python for tool integration
    python3 \
    python3-pip \
    # Cleanup
    && rm -rf /var/lib/apt/lists/*

# Install TVM (optional - if available)
# RUN pip3 install apache-tvm

# ============================================================================
# Production Stage
# ============================================================================
FROM python:3.11-slim AS production

# Metadata
LABEL maintainer="Terragon Labs <contact@terragon-labs.com>"
LABEL version="0.1.0"
LABEL description="AI Hardware Co-Design Playground"
LABEL org.opencontainers.image.source="https://github.com/terragon-labs/ai-hardware-codesign-playground"
LABEL org.opencontainers.image.documentation="https://docs.terragon-labs.com/ai-hardware-codesign-playground"
LABEL org.opencontainers.image.licenses="MIT"

# Create non-root user
RUN groupadd -r codesign && useradd -r -g codesign -u 1000 codesign

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # Runtime dependencies
    curl \
    wget \
    git \
    # Hardware simulation tools
    verilator \
    # Cleanup
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy virtual environment from builder
COPY --from=backend-deps /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy hardware tools from builder
COPY --from=hardware-tools /usr/bin/verilator* /usr/bin/
COPY --from=hardware-tools /usr/share/verilator /usr/share/verilator

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=codesign:codesign backend/ ./backend/
COPY --chown=codesign:codesign --from=frontend-builder /app/frontend/dist ./frontend/dist/
COPY --chown=codesign:codesign pyproject.toml README.md LICENSE ./

# Create necessary directories
RUN mkdir -p /app/data /app/logs /app/uploads /app/artifacts \
    && chown -R codesign:codesign /app

# Install the application
RUN pip install -e .

# Create startup script
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
# Run database migrations\n\
if [ "$RUN_MIGRATIONS" = "true" ]; then\n\
    echo "Running database migrations..."\n\
    alembic upgrade head\n\
fi\n\
\n\
# Start the application\n\
if [ "$1" = "web" ]; then\n\
    echo "Starting web server..."\n\
    exec uvicorn backend.main:app --host 0.0.0.0 --port 8000\n\
elif [ "$1" = "worker" ]; then\n\
    echo "Starting celery worker..."\n\
    exec celery -A backend.worker worker --loglevel=info\n\
elif [ "$1" = "scheduler" ]; then\n\
    echo "Starting celery beat scheduler..."\n\
    exec celery -A backend.worker beat --loglevel=info\n\
else\n\
    echo "Usage: $0 {web|worker|scheduler}"\n\
    exit 1\n\
fi\n\
' > /app/docker-entrypoint.sh \
    && chmod +x /app/docker-entrypoint.sh

# Security: Switch to non-root user
USER codesign

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Environment variables
ENV PYTHONPATH=/app/backend \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LOG_LEVEL=INFO \
    ENVIRONMENT=production

# Expose port
EXPOSE 8000

# Default command
ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["web"]

# ============================================================================
# Development Stage
# ============================================================================
FROM production AS development

# Switch back to root for development tools installation
USER root

# Install development dependencies
RUN pip install --no-cache-dir \
    pytest \
    pytest-cov \
    pytest-asyncio \
    black \
    isort \
    mypy \
    pylint \
    bandit \
    ipython \
    jupyter \
    jupyterlab

# Install additional development tools
RUN apt-get update && apt-get install -y \
    vim \
    htop \
    tree \
    && rm -rf /var/lib/apt/lists/*

# Switch back to codesign user
USER codesign

# Override environment for development
ENV ENVIRONMENT=development \
    DEBUG=true \
    LOG_LEVEL=DEBUG

# Development command
CMD ["/bin/bash"]

# ============================================================================
# Testing Stage
# ============================================================================
FROM development AS testing

# Copy test files
COPY --chown=codesign:codesign tests/ ./tests/

# Run tests by default
CMD ["pytest", "tests/", "-v", "--cov=backend/src", "--cov-report=html", "--cov-report=term-missing"]

# ============================================================================
# Documentation Stage
# ============================================================================
FROM python:3.11-slim AS docs

# Install documentation dependencies
RUN pip install --no-cache-dir \
    sphinx \
    sphinx-rtd-theme \
    myst-parser \
    sphinx-autodoc-typehints

# Set working directory
WORKDIR /app

# Copy documentation source
COPY docs/ ./docs/
COPY backend/src/ ./backend/src/
COPY README.md pyproject.toml ./

# Build documentation
RUN sphinx-build -b html docs/ docs/_build/html

# Serve documentation
EXPOSE 8080
CMD ["python", "-m", "http.server", "8080", "--directory", "docs/_build/html"]
