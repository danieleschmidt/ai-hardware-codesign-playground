# Multi-stage Dockerfile for AI Hardware Co-Design Playground Backend
# Optimized for production deployment with security and performance

# =============================================================================
# Base Stage - Common dependencies and Python setup
# =============================================================================
FROM python:3.11-slim as base

# Set environment variables for Python
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Create non-root user for security
RUN groupadd -r codesign && useradd -r -g codesign codesign

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libpq-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Create app directory with proper permissions
WORKDIR /app
RUN chown -R codesign:codesign /app

# =============================================================================
# Dependencies Stage - Install Python dependencies
# =============================================================================
FROM base as dependencies

# Copy requirements files
COPY requirements.txt requirements-dev.txt ./

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt && \
    pip cache purge

# =============================================================================
# Development Stage - For local development and testing
# =============================================================================
FROM dependencies as development

# Install development dependencies
RUN pip install -r requirements-dev.txt && \
    pip cache purge

# Copy source code
COPY backend/ ./

# Change ownership to non-root user
RUN chown -R codesign:codesign /app

# Switch to non-root user
USER codesign

# Expose ports
EXPOSE 8000 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command for development
CMD ["python", "-m", "uvicorn", "codesign_playground.server:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# =============================================================================
# Production Stage - Optimized for production deployment
# =============================================================================
FROM dependencies as production

# Install production server (Gunicorn)
RUN pip install gunicorn[gthread] uvicorn[standard] && \
    pip cache purge

# Copy source code
COPY backend/ ./

# Create necessary directories
RUN mkdir -p /app/data/cache /app/data/models /app/logs && \
    chown -R codesign:codesign /app

# Switch to non-root user
USER codesign

# Expose ports
EXPOSE 8000 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Production command with Gunicorn
CMD ["gunicorn", "codesign_playground.server:app", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--workers", "4", \
     "--bind", "0.0.0.0:8000", \
     "--max-requests", "1000", \
     "--max-requests-jitter", "50", \
     "--timeout", "120", \
     "--keepalive", "2", \
     "--access-logfile", "/app/logs/access.log", \
     "--error-logfile", "/app/logs/error.log", \
     "--log-level", "info"]

# =============================================================================
# Worker Stage - For Celery background tasks
# =============================================================================
FROM production as worker

# Install Celery with Redis support
RUN pip install celery[redis] && \
    pip cache purge

# Worker-specific health check
HEALTHCHECK --interval=60s --timeout=30s --start-period=60s --retries=3 \
    CMD celery -A codesign_playground.tasks.celery inspect ping || exit 1

# Worker command
CMD ["celery", "-A", "codesign_playground.tasks.celery", "worker", \
     "--loglevel=info", \
     "--concurrency=4", \
     "--max-tasks-per-child=1000", \
     "--time-limit=3600", \
     "--soft-time-limit=3300"]

# =============================================================================
# Scheduler Stage - For Celery Beat scheduler
# =============================================================================
FROM production as scheduler

# Install Celery with Redis support
RUN pip install celery[redis] && \
    pip cache purge

# Create celerybeat directory
RUN mkdir -p /app/celerybeat && \
    chown codesign:codesign /app/celerybeat

# Scheduler command
CMD ["celery", "-A", "codesign_playground.tasks.celery", "beat", \
     "--loglevel=info", \
     "--schedule=/app/celerybeat/celerybeat-schedule", \
     "--pidfile=/app/celerybeat/celerybeat.pid"]

# =============================================================================
# Security Stage - Hardened for security-critical deployments
# =============================================================================
FROM production as security

# Additional security hardening
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Remove unnecessary packages and files
RUN apt-get purge -y build-essential git && \
    apt-get autoremove -y && \
    rm -rf /var/cache/apt/* /tmp/* /var/tmp/*

# Set strict file permissions
RUN find /app -type d -exec chmod 755 {} \; && \
    find /app -type f -exec chmod 644 {} \; && \
    chmod +x /app/codesign_playground/*.py

# Security labels
LABEL security.scan="true" \
      security.vulnerability-scan="true" \
      security.compliance="SOC2,GDPR" \
      maintainer="Terragon Labs <contact@terragon-labs.com>"

# =============================================================================
# Testing Stage - For CI/CD testing
# =============================================================================
FROM development as testing

# Install additional testing tools
RUN pip install pytest-xvfb pytest-benchmark safety bandit && \
    pip cache purge

# Copy test files
COPY tests/ /app/tests/
COPY pytest.ini pyproject.toml ./

# Test command
CMD ["python", "-m", "pytest", "tests/", "-v", "--cov=codesign_playground", "--cov-report=html", "--cov-report=term-missing"]

# =============================================================================
# Documentation Stage - For generating documentation
# =============================================================================
FROM dependencies as docs

# Install documentation tools
RUN pip install sphinx sphinx-rtd-theme sphinx-autodoc-typehints && \
    pip cache purge

# Copy documentation source
COPY docs/ /app/docs/
COPY backend/ ./

# Documentation build command
CMD ["sphinx-build", "-b", "html", "docs/", "/app/docs/_build/html/"]