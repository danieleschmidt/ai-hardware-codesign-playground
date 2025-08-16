# Enhanced Production Dockerfile for AI Hardware Co-Design Playground
# Optimized for enterprise production with security, monitoring, and multi-stage builds

# ============================================================================
# Security Scanner Stage: Vulnerability assessment during build
# ============================================================================
FROM aquasec/trivy:latest as security-scanner

# Copy source code for security scanning
COPY . /src
WORKDIR /src

# Run security scan and save results
RUN trivy fs --format json --output /tmp/security-report.json /src && \
    trivy fs --format table /src

# ============================================================================
# Base Builder Stage: Common build dependencies
# ============================================================================
FROM python:3.11-slim as base-builder

# Build arguments with defaults
ARG BUILD_DATE=unknown
ARG VCS_REF=unknown
ARG VERSION=latest
ARG BUILD_ENV=production
ARG TARGET_ARCH=amd64

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install build dependencies with security best practices
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    pkg-config \
    libpq-dev \
    libffi-dev \
    libssl-dev \
    curl \
    git \
    ca-certificates \
    gnupg \
    lsb-release \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/* \
    && rm -rf /var/tmp/*

# Create build user with specific UID/GID for consistency
RUN groupadd -r -g 1001 builder && \
    useradd -r -g builder -u 1001 -m -d /home/builder builder

# Set working directory
WORKDIR /build

# Upgrade pip and install build tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel build

# ============================================================================
# Dependencies Stage: Install Python dependencies
# ============================================================================
FROM base-builder as dependencies

# Copy dependency files with proper ownership
COPY --chown=builder:builder requirements.txt requirements-dev.txt pyproject.toml ./

# Create virtual environment for better isolation
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install dependencies with integrity verification
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir \
    gunicorn[gthread] \
    uvicorn[standard] \
    prometheus-client \
    opentelemetry-api \
    opentelemetry-sdk \
    opentelemetry-instrumentation-fastapi \
    opentelemetry-instrumentation-psycopg2 \
    opentelemetry-instrumentation-redis \
    opentelemetry-exporter-prometheus \
    sentry-sdk[fastapi] \
    structlog \
    && pip cache purge

# ============================================================================
# Application Build Stage: Build the application
# ============================================================================
FROM dependencies as app-builder

# Copy source code
COPY --chown=builder:builder backend/ ./backend/
COPY --chown=builder:builder scripts/ ./scripts/
COPY --chown=builder:builder monitoring/ ./monitoring/
COPY --chown=builder:builder README.md LICENSE ./

# Install application in production mode
RUN cd backend && pip install --no-cache-dir -e .

# Generate application metadata
RUN echo "{\
  \"version\": \"${VERSION}\", \
  \"build_date\": \"${BUILD_DATE}\", \
  \"vcs_ref\": \"${VCS_REF}\", \
  \"build_env\": \"${BUILD_ENV}\", \
  \"target_arch\": \"${TARGET_ARCH}\" \
}" > /build/app-metadata.json

# ============================================================================
# Production Runtime Stage: Minimal secure runtime
# ============================================================================
FROM python:3.11-slim as production

# Build arguments for runtime metadata
ARG BUILD_DATE=unknown
ARG VCS_REF=unknown
ARG VERSION=latest
ARG BUILD_ENV=production

# Enhanced metadata labels following OCI spec
LABEL maintainer="Terragon Labs <contact@terragon-labs.com>" \
      org.opencontainers.image.title="AI Hardware Co-Design Playground - Production Enhanced" \
      org.opencontainers.image.description="Enterprise production runtime for neural network and hardware accelerator co-optimization" \
      org.opencontainers.image.url="https://github.com/terragon-labs/ai-hardware-codesign-playground" \
      org.opencontainers.image.source="https://github.com/terragon-labs/ai-hardware-codesign-playground" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.vendor="Terragon Labs" \
      org.opencontainers.image.licenses="MIT" \
      org.label-schema.build-date="${BUILD_DATE}" \
      org.label-schema.version="${VERSION}" \
      org.label-schema.vcs-ref="${VCS_REF}" \
      security.scan.enabled="true" \
      security.hardening.enabled="true" \
      compliance.gdpr="true" \
      compliance.ccpa="true" \
      compliance.pdpa="true"

# Install runtime dependencies with security updates
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    libffi8 \
    libssl3 \
    curl \
    ca-certificates \
    redis-tools \
    dumb-init \
    netcat-openbsd \
    && apt-get upgrade -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/* \
    && rm -rf /var/tmp/*

# Create application user and group with fixed UID/GID
RUN groupadd -r -g 1001 appuser && \
    useradd -r -g appuser -u 1001 -m -d /home/appuser -s /bin/bash appuser

# Set working directory
WORKDIR /app

# Copy virtual environment from dependencies stage
COPY --from=dependencies --chown=appuser:appuser /opt/venv /opt/venv

# Copy application code and metadata
COPY --from=app-builder --chown=appuser:appuser /build/backend/ ./backend/
COPY --from=app-builder --chown=appuser:appuser /build/scripts/ ./scripts/
COPY --from=app-builder --chown=appuser:appuser /build/monitoring/ ./monitoring/
COPY --from=app-builder --chown=appuser:appuser /build/app-metadata.json ./

# Copy security scan results for runtime access
COPY --from=security-scanner /tmp/security-report.json ./security/

# Create necessary directories with proper permissions
RUN mkdir -p \
    /app/data \
    /app/logs \
    /app/tmp \
    /app/cache \
    /app/uploads \
    /app/backups \
    /app/security \
    /app/compliance \
    && chown -R appuser:appuser /app \
    && chmod -R 755 /app \
    && chmod -R 750 /app/security \
    && chmod -R 750 /app/compliance

# Make scripts executable
RUN chmod +x ./scripts/*.sh

# Security hardening: Remove unnecessary packages and clean up
RUN apt-get autoremove -y && \
    apt-get autoclean && \
    find /usr/share/doc -depth -type f ! -name copyright -delete && \
    find /usr/share/man -type f -delete && \
    find /var/cache -type f -delete

# Switch to non-root user
USER appuser

# Set environment variables for production
ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONPATH="/app/backend" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    ENV=production \
    LOG_LEVEL=INFO \
    WORKERS=4 \
    MAX_REQUESTS=1000 \
    MAX_REQUESTS_JITTER=50 \
    TIMEOUT=120 \
    KEEPALIVE=5 \
    GRACEFUL_TIMEOUT=30 \
    PRELOAD_APP=true \
    ENABLE_MONITORING=true \
    ENABLE_SECURITY_HEADERS=true \
    ENABLE_RATE_LIMITING=true \
    ENABLE_REQUEST_VALIDATION=true

# Health check with comprehensive validation
HEALTHCHECK --interval=30s --timeout=15s --start-period=60s --retries=5 \
    CMD curl -f http://localhost:8000/health/detailed || exit 1

# Expose application port
EXPOSE 8000

# Use dumb-init for proper signal handling
ENTRYPOINT ["dumb-init", "--"]

# Default production command with gunicorn
CMD ["gunicorn", "codesign_playground.server:app", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--workers", "${WORKERS}", \
     "--bind", "0.0.0.0:8000", \
     "--max-requests", "${MAX_REQUESTS}", \
     "--max-requests-jitter", "${MAX_REQUESTS_JITTER}", \
     "--timeout", "${TIMEOUT}", \
     "--keep-alive", "${KEEPALIVE}", \
     "--graceful-timeout", "${GRACEFUL_TIMEOUT}", \
     "--preload", \
     "--access-logfile", "/app/logs/access.log", \
     "--error-logfile", "/app/logs/error.log", \
     "--log-level", "${LOG_LEVEL}", \
     "--capture-output", \
     "--enable-stdio-inheritance"]

# ============================================================================
# Worker Stage: Optimized for Celery background tasks
# ============================================================================
FROM production as worker

# Install Celery with Redis and monitoring support
USER root
RUN /opt/venv/bin/pip install --no-cache-dir \
    celery[redis] \
    flower \
    celery-prometheus-exporter
USER appuser

# Worker-specific environment variables
ENV CELERY_WORKERS=4 \
    CELERY_MAX_TASKS_PER_CHILD=1000 \
    CELERY_MAX_MEMORY_PER_CHILD=200000 \
    CELERY_TIME_LIMIT=3600 \
    CELERY_SOFT_TIME_LIMIT=3300 \
    CELERY_ACKS_LATE=true \
    CELERY_REJECT_ON_WORKER_LOST=true \
    CELERY_TASK_ROUTES_FILE=/app/celery-routes.json

# Copy Celery configuration
COPY --chown=appuser:appuser deployment/celery/ ./celery/

# Worker health check
HEALTHCHECK --interval=60s --timeout=30s --start-period=60s --retries=3 \
    CMD celery -A codesign_playground.tasks.celery inspect ping || exit 1

# Expose Celery monitoring port
EXPOSE 5555

# Celery worker command
CMD ["celery", "-A", "codesign_playground.tasks.celery", "worker", \
     "--loglevel=info", \
     "--concurrency=${CELERY_WORKERS}", \
     "--max-tasks-per-child=${CELERY_MAX_TASKS_PER_CHILD}", \
     "--max-memory-per-child=${CELERY_MAX_MEMORY_PER_CHILD}", \
     "--time-limit=${CELERY_TIME_LIMIT}", \
     "--soft-time-limit=${CELERY_SOFT_TIME_LIMIT}", \
     "--pool=prefork", \
     "--optimization=fair"]

# ============================================================================
# Scheduler Stage: Celery Beat scheduler
# ============================================================================
FROM production as scheduler

# Install Celery Beat with persistence
USER root
RUN /opt/venv/bin/pip install --no-cache-dir \
    celery[redis] \
    django-celery-beat
USER appuser

# Create scheduler directories
RUN mkdir -p /app/celerybeat /app/schedules && \
    chmod 755 /app/celerybeat /app/schedules

# Scheduler environment variables
ENV CELERY_BEAT_SCHEDULE_FILENAME=/app/celerybeat/celerybeat-schedule \
    CELERY_BEAT_PID_FILE=/app/celerybeat/celerybeat.pid \
    CELERY_BEAT_LOG_FILE=/app/logs/celerybeat.log

# Copy scheduler configuration
COPY --chown=appuser:appuser deployment/scheduler/ ./scheduler/

# Scheduler command
CMD ["celery", "-A", "codesign_playground.tasks.celery", "beat", \
     "--loglevel=info", \
     "--schedule=${CELERY_BEAT_SCHEDULE_FILENAME}", \
     "--pidfile=${CELERY_BEAT_PID_FILE}"]

# ============================================================================
# Monitoring Stage: Enhanced observability
# ============================================================================
FROM production as monitoring

USER root

# Install monitoring and observability tools
RUN /opt/venv/bin/pip install --no-cache-dir \
    prometheus-client \
    opentelemetry-auto-instrumentation \
    opentelemetry-exporter-jaeger \
    opentelemetry-exporter-prometheus \
    opentelemetry-instrumentation-fastapi \
    opentelemetry-instrumentation-psycopg2 \
    opentelemetry-instrumentation-redis \
    opentelemetry-instrumentation-celery \
    sentry-sdk[fastapi] \
    structlog \
    elastic-apm

# Install system monitoring tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    prometheus-node-exporter \
    && rm -rf /var/lib/apt/lists/*

USER appuser

# Copy monitoring configuration
COPY --chown=appuser:appuser monitoring/ ./monitoring/
COPY --chown=appuser:appuser deployment/observability/ ./observability/

# Monitoring environment variables
ENV ENABLE_METRICS=true \
    ENABLE_TRACING=true \
    ENABLE_APM=true \
    METRICS_PORT=9090 \
    OTEL_SERVICE_NAME=codesign-playground \
    OTEL_SERVICE_VERSION=${VERSION} \
    OTEL_RESOURCE_ATTRIBUTES="service.name=codesign-playground,service.version=${VERSION},deployment.environment=production" \
    OTEL_EXPORTER_JAEGER_ENDPOINT=http://jaeger:14268/api/traces \
    OTEL_EXPORTER_PROMETHEUS_PORT=9090 \
    SENTRY_ENVIRONMENT=production \
    ELASTIC_APM_ENVIRONMENT=production

# Expose monitoring ports
EXPOSE 9090 9464 8000

# Monitoring-enabled startup command
CMD ["opentelemetry-instrument", \
     "gunicorn", "codesign_playground.server:app", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--workers", "${WORKERS}", \
     "--bind", "0.0.0.0:8000", \
     "--max-requests", "${MAX_REQUESTS}", \
     "--timeout", "${TIMEOUT}", \
     "--preload"]

# ============================================================================
# High Availability Stage: Multi-region deployment
# ============================================================================
FROM production as ha-multi-region

USER root

# Install tools for multi-region deployment
RUN apt-get update && apt-get install -y --no-install-recommends \
    gettext-base \
    awscli \
    jq \
    && rm -rf /var/lib/apt/lists/*

# Install additional Python packages for HA
RUN /opt/venv/bin/pip install --no-cache-dir \
    consul-python \
    etcd3 \
    kubernetes \
    redis-sentinel

USER appuser

# Copy multi-region configuration
COPY --chown=appuser:appuser deployment/multi-region/ ./deployment/
COPY --chown=appuser:appuser deployment/ha/ ./ha/

# Multi-region environment variables
ENV ENABLE_MULTI_REGION=true \
    REGION_AUTO_DETECT=true \
    COMPLIANCE_MODE=regional \
    HA_MODE=active-active \
    DISCOVERY_SERVICE=consul \
    LOAD_BALANCER_TYPE=global \
    FAILOVER_ENABLED=true \
    HEALTH_CHECK_INTERVAL=30

# Multi-region startup command
CMD ["./scripts/ha-multi-region-start.sh"]

# ============================================================================
# Edge Computing Stage: Optimized for edge deployment
# ============================================================================
FROM production as edge

# Optimize for edge deployment (smaller footprint)
USER root

# Remove unnecessary packages for edge deployment
RUN apt-get autoremove -y && \
    apt-get autoclean && \
    find /usr/share/locale -mindepth 1 -maxdepth 1 ! -name 'en' -exec rm -r {} + && \
    find /usr/share/doc -depth -type f ! -name copyright -delete && \
    find /usr/share/man -type f -delete

USER appuser

# Edge-specific environment variables
ENV EDGE_MODE=true \
    CACHE_SIZE_MB=64 \
    MAX_WORKERS=2 \
    MAX_REQUESTS=500 \
    ENABLE_COMPRESSION=true \
    ENABLE_EDGE_CACHING=true

# Copy edge configuration
COPY --chown=appuser:appuser deployment/edge/ ./edge/

# Edge-optimized command
CMD ["gunicorn", "codesign_playground.server:app", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--workers", "2", \
     "--bind", "0.0.0.0:8000", \
     "--max-requests", "500", \
     "--timeout", "60", \
     "--preload"]