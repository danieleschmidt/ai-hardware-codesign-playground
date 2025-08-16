# Backup Service Dockerfile
# Automated backup and disaster recovery service

FROM python:3.11-alpine as backup-service

# Install system dependencies
RUN apk add --no-cache \
    postgresql-client \
    redis \
    aws-cli \
    gnupg \
    curl \
    rsync \
    cron \
    && rm -rf /var/cache/apk/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    psycopg2-binary \
    redis \
    boto3 \
    cryptography \
    schedule \
    structlog \
    prometheus-client

# Create backup user
RUN addgroup -g 1001 backup && \
    adduser -D -u 1001 -G backup backup

# Set working directory
WORKDIR /app

# Copy backup scripts
COPY scripts/backup/ ./scripts/
COPY monitoring/backup/ ./monitoring/

# Create directories
RUN mkdir -p /backups/postgres /backups/redis /backups/compliance /backups/logs && \
    chown -R backup:backup /app /backups

# Make scripts executable
RUN chmod +x ./scripts/*.sh

# Switch to backup user
USER backup

# Health check
HEALTHCHECK --interval=300s --timeout=30s --start-period=60s --retries=3 \
    CMD python ./scripts/health_check.py

# Default command
CMD ["python", "./scripts/backup_scheduler.py"]