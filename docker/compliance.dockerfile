# Compliance Auditor Dockerfile
# GDPR, CCPA, PDPA compliance monitoring and audit service

FROM python:3.11-slim as compliance-auditor

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies for compliance
RUN pip install --no-cache-dir \
    psycopg2-binary \
    redis \
    sqlalchemy \
    alembic \
    fastapi \
    uvicorn \
    structlog \
    cryptography \
    prometheus-client \
    schedule \
    pandas \
    numpy \
    python-dateutil \
    pytz

# Create compliance user
RUN groupadd -r -g 1001 compliance && \
    useradd -r -g compliance -u 1001 -m -d /home/compliance compliance

# Set working directory
WORKDIR /app

# Copy compliance scripts and configuration
COPY backend/codesign_playground/utils/compliance*.py ./utils/
COPY scripts/compliance/ ./scripts/
COPY monitoring/compliance/ ./monitoring/
COPY config/compliance/ ./config/

# Create necessary directories
RUN mkdir -p /app/data /app/logs /app/reports /app/audit && \
    chown -R compliance:compliance /app

# Make scripts executable
RUN chmod +x ./scripts/*.sh

# Switch to compliance user
USER compliance

# Set environment variables
ENV PYTHONPATH=/app \
    COMPLIANCE_MODE=production \
    AUDIT_LEVEL=detailed \
    LOG_LEVEL=INFO

# Health check
HEALTHCHECK --interval=60s --timeout=30s --start-period=30s --retries=3 \
    CMD python ./scripts/compliance_health_check.py

# Expose compliance API port
EXPOSE 8001

# Default command
CMD ["python", "./scripts/compliance_monitor.py"]