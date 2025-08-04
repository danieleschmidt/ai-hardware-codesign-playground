# Production-optimized Dockerfile for AI Hardware Co-Design Playground
# Multi-stage build for minimal production image with security hardening

# ============================================================================
# Build Stage: Compile and prepare application
# ============================================================================
FROM python:3.11-slim as builder

# Build arguments
ARG BUILD_DATE
ARG VCS_REF
ARG VERSION

# Metadata
LABEL maintainer="Terragon Labs <contact@terragon-labs.com>" \
      org.opencontainers.image.title="AI Hardware Co-Design Playground" \
      org.opencontainers.image.description="Production build for neural network and hardware accelerator co-optimization" \
      org.opencontainers.image.url="https://github.com/terragon-labs/ai-hardware-codesign-playground" \
      org.opencontainers.image.source="https://github.com/terragon-labs/ai-hardware-codesign-playground" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.vendor="Terragon Labs" \
      org.opencontainers.image.licenses="MIT"

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    pkg-config \
    libpq-dev \
    libffi-dev \
    libssl-dev \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create build user
RUN groupadd -r builder && useradd -r -g builder builder

# Set working directory
WORKDIR /build

# Copy dependency files
COPY --chown=builder:builder requirements.txt pyproject.toml ./

# Upgrade pip and install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir --user -r requirements.txt

# Copy source code
COPY --chown=builder:builder backend/ ./backend/
COPY --chown=builder:builder README.md LICENSE ./

# Install the package
RUN pip install --no-cache-dir --user -e ./backend/

# ============================================================================
# Production Stage: Minimal runtime image
# ============================================================================
FROM python:3.11-slim as production

# Build arguments for runtime
ARG BUILD_DATE
ARG VCS_REF  
ARG VERSION

# Metadata
LABEL maintainer="Terragon Labs <contact@terragon-labs.com>" \
      org.opencontainers.image.title="AI Hardware Co-Design Playground - Production" \
      org.opencontainers.image.description="Production runtime for neural network and hardware accelerator co-optimization" \
      org.opencontainers.image.url="https://github.com/terragon-labs/ai-hardware-codesign-playground" \
      org.opencontainers.image.source="https://github.com/terragon-labs/ai-hardware-codesign-playground" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.vendor="Terragon Labs" \
      org.opencontainers.image.licenses="MIT"

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    libffi8 \
    libssl3 \
    curl \
    ca-certificates \
    redis-tools \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create application user and group
RUN groupadd -r -g 1001 appuser && \
    useradd -r -g appuser -u 1001 -m -d /home/appuser -s /bin/bash appuser

# Set working directory
WORKDIR /app

# Copy Python packages from builder
COPY --from=builder --chown=appuser:appuser /root/.local /home/appuser/.local

# Copy application code
COPY --from=builder --chown=appuser:appuser /build/backend/ ./backend/
COPY --chown=appuser:appuser scripts/entrypoint.sh ./scripts/
RUN chmod +x ./scripts/entrypoint.sh

# Create necessary directories
RUN mkdir -p /app/data /app/logs /app/tmp && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Add user's local bin to PATH
ENV PATH="/home/appuser/.local/bin:${PATH}" \
    PYTHONPATH="/app/backend" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Security: Remove setuid/setgid permissions from system binaries
USER root
RUN find / -perm /6000 -type f -exec chmod a-s {} \; 2>/dev/null || true
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Use entrypoint script for graceful startup
ENTRYPOINT ["./scripts/entrypoint.sh"]

# Default command
CMD ["uvicorn", "codesign_playground.server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4", "--worker-class", "uvicorn.workers.UvicornWorker"]

# ============================================================================
# Multi-region deployment variant
# ============================================================================
FROM production as multi-region

# Install additional tools for multi-region deployment
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    gettext-base \
    awscli \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

USER appuser

# Copy multi-region configuration templates
COPY --chown=appuser:appuser deployment/multi-region/ ./deployment/

# Environment variables for multi-region support
ENV ENABLE_MULTI_REGION=true \
    REGION_AUTO_DETECT=true \
    COMPLIANCE_MODE=global

# Multi-region startup command
CMD ["./scripts/multi-region-start.sh"]

# ============================================================================
# High-performance variant with optimizations
# ============================================================================
FROM production as high-performance

# Install performance optimization libraries
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    libjemalloc2 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

USER appuser

# Performance environment variables
ENV LD_PRELOAD=libjemalloc.so.2 \
    MALLOC_CONF="background_thread:true,metadata_thp:auto" \
    PYTHONMALLOC=malloc \
    WORKER_CONNECTIONS=2000 \
    MAX_REQUESTS=10000 \
    MAX_REQUESTS_JITTER=1000

# High-performance command with optimized settings
CMD ["uvicorn", "codesign_playground.server:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "8", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--backlog", "2048", \
     "--max-requests", "10000", \
     "--max-requests-jitter", "1000"]

# ============================================================================
# Monitoring and observability variant
# ============================================================================
FROM production as monitoring

USER root

# Install monitoring tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    prometheus-node-exporter \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Install Python monitoring packages
RUN pip install --no-cache-dir \
    prometheus-client \
    opentelemetry-api \
    opentelemetry-sdk \
    opentelemetry-auto-instrumentation \
    opentelemetry-exporter-prometheus \
    opentelemetry-instrumentation-fastapi \
    opentelemetry-instrumentation-psycopg2 \
    opentelemetry-instrumentation-redis

USER appuser

# Copy monitoring configuration
COPY --chown=appuser:appuser monitoring/ ./monitoring/

# Monitoring environment variables
ENV ENABLE_METRICS=true \
    ENABLE_TRACING=true \
    METRICS_PORT=9090 \
    OTEL_SERVICE_NAME=codesign-playground \
    OTEL_RESOURCE_ATTRIBUTES="service.name=codesign-playground,service.version=${VERSION}"

# Expose monitoring ports
EXPOSE 9090 9464

# Monitoring-enabled startup
CMD ["./scripts/monitoring-start.sh"]

# ============================================================================
# Security-hardened variant
# ============================================================================
FROM production as security-hardened

USER root

# Install security tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    apparmor-utils \
    fail2ban \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Apply additional security hardening
RUN echo "net.ipv4.ip_forward = 0" >> /etc/sysctl.conf && \
    echo "net.ipv4.conf.all.send_redirects = 0" >> /etc/sysctl.conf && \
    echo "net.ipv4.conf.default.send_redirects = 0" >> /etc/sysctl.conf && \
    echo "net.ipv4.conf.all.accept_source_route = 0" >> /etc/sysctl.conf && \
    echo "net.ipv4.conf.default.accept_source_route = 0" >> /etc/sysctl.conf && \
    echo "net.ipv4.conf.all.accept_redirects = 0" >> /etc/sysctl.conf && \
    echo "net.ipv4.conf.default.accept_redirects = 0" >> /etc/sysctl.conf && \
    echo "net.ipv4.conf.all.secure_redirects = 0" >> /etc/sysctl.conf && \
    echo "net.ipv4.conf.default.secure_redirects = 0" >> /etc/sysctl.conf

# Copy security policies
COPY --chown=appuser:appuser security/ ./security/

USER appuser

# Security environment variables
ENV SECURITY_HARDENED=true \
    ENABLE_RATE_LIMITING=true \
    ENABLE_REQUEST_VALIDATION=strict \
    MAX_REQUEST_SIZE=10MB \
    ALLOWED_HOSTS="localhost,127.0.0.1"

# Security-hardened startup
CMD ["./scripts/security-start.sh"]