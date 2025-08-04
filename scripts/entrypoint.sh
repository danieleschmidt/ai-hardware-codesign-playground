#!/bin/bash

# Production entrypoint script for AI Hardware Co-Design Playground
# Handles initialization, health checks, and graceful startup

set +e  # Allow commands to fail for error handling

# Configuration
APP_NAME="codesign-playground"
LOG_LEVEL="${LOG_LEVEL:-info}"
WORKERS="${WORKERS:-4}"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"

# Colors for logging
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1" >&2
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

# Signal handlers for graceful shutdown
shutdown_handler() {
    log_info "Received shutdown signal, initiating graceful shutdown..."
    
    if [ ! -z "$SERVER_PID" ]; then
        log_info "Stopping server (PID: $SERVER_PID)..."
        kill -TERM "$SERVER_PID" 2>/dev/null
        
        # Wait for graceful shutdown
        local timeout=30
        while kill -0 "$SERVER_PID" 2>/dev/null && [ $timeout -gt 0 ]; do
            sleep 1
            timeout=$((timeout - 1))
        done
        
        if kill -0 "$SERVER_PID" 2>/dev/null; then
            log_warn "Server did not shutdown gracefully, forcing termination..."
            kill -KILL "$SERVER_PID" 2>/dev/null
        else
            log_success "Server shutdown gracefully"
        fi
    fi
    
    log_info "Cleanup completed, exiting..."
    exit 0
}

# Register signal handlers
trap shutdown_handler SIGTERM SIGINT

# Environment validation
validate_environment() {
    log_info "Validating environment configuration..."
    
    # Check required environment variables
    local required_vars=("PYTHONPATH")
    local missing_vars=()
    
    for var in "${required_vars[@]}"; do
        if [ -z "${!var}" ]; then
            missing_vars+=("$var")
        fi
    done
    
    if [ ${#missing_vars[@]} -ne 0 ]; then
        log_error "Missing required environment variables: ${missing_vars[*]}"
        return 1
    fi
    
    # Validate numeric values
    if ! [[ "$WORKERS" =~ ^[0-9]+$ ]] || [ "$WORKERS" -lt 1 ] || [ "$WORKERS" -gt 32 ]; then
        log_error "WORKERS must be a number between 1 and 32, got: $WORKERS"
        return 1
    fi
    
    if ! [[ "$PORT" =~ ^[0-9]+$ ]] || [ "$PORT" -lt 1024 ] || [ "$PORT" -gt 65535 ]; then
        log_error "PORT must be a number between 1024 and 65535, got: $PORT"
        return 1
    fi
    
    log_success "Environment validation passed"
    return 0
}

# Dependency checks
check_dependencies() {
    log_info "Checking dependencies..."
    
    # Check Python dependencies
    local python_deps=("fastapi" "uvicorn" "pydantic")
    for dep in "${python_deps[@]}"; do
        if ! python -c "import $dep" 2>/dev/null; then
            log_error "Missing Python dependency: $dep"
            return 1
        fi
    done
    
    # Check if Redis is accessible (if configured)
    if [ ! -z "$REDIS_URL" ]; then
        log_info "Checking Redis connectivity..."
        if command -v redis-cli >/dev/null 2>&1; then
            if ! redis-cli -u "$REDIS_URL" ping >/dev/null 2>&1; then
                log_warn "Redis is not accessible at $REDIS_URL"
            else
                log_success "Redis connection verified"
            fi
        fi
    fi
    
    # Check database connectivity (if configured)
    if [ ! -z "$DATABASE_URL" ]; then
        log_info "Checking database connectivity..."
        if python -c "
import os
import psycopg2
try:
    conn = psycopg2.connect(os.environ['DATABASE_URL'])
    conn.close()
    print('Database connection successful')
except Exception as e:
    print(f'Database connection failed: {e}')
    exit(1)
" 2>/dev/null; then
            log_success "Database connection verified"
        else
            log_warn "Database connection failed"
        fi
    fi
    
    log_success "Dependency checks completed"
    return 0
}

# Initialize application
initialize_app() {
    log_info "Initializing application..."
    
    # Create necessary directories
    mkdir -p /app/logs /app/tmp /app/data
    
    # Set up logging configuration
    export CODESIGN_PLAYGROUND_LOG_LEVEL="$LOG_LEVEL"
    export CODESIGN_PLAYGROUND_LOG_FILE="/app/logs/app.log"
    
    # Initialize database migrations if needed
    if [ "$RUN_MIGRATIONS" = "true" ] && [ ! -z "$DATABASE_URL" ]; then
        log_info "Running database migrations..."
        if python -c "
from codesign_playground.database import run_migrations
run_migrations()
" 2>/dev/null; then
            log_success "Database migrations completed"
        else
            log_error "Database migrations failed"
            return 1
        fi
    fi
    
    # Warm up caches if enabled
    if [ "$WARM_UP_CACHE" = "true" ]; then
        log_info "Warming up caches..."
        python -c "
from codesign_playground.core.cache import get_cache_stats
stats = get_cache_stats()
print(f'Cache system initialized: {len(stats)} cache types')
" 2>/dev/null || log_warn "Cache warm-up failed"
    fi
    
    log_success "Application initialization completed"
    return 0
}

# Health check function
health_check() {
    local max_attempts=30
    local attempt=1
    
    log_info "Performing health check..."
    
    while [ $attempt -le $max_attempts ]; do
        if curl -f -s "http://$HOST:$PORT/health" >/dev/null 2>&1; then
            log_success "Health check passed"
            return 0
        fi
        
        log_info "Health check attempt $attempt/$max_attempts failed, retrying in 2 seconds..."
        sleep 2
        attempt=$((attempt + 1))
    done
    
    log_error "Health check failed after $max_attempts attempts"
    return 1
}

# Performance optimization
optimize_performance() {
    log_info "Applying performance optimizations..."
    
    # Set memory allocator optimizations
    if [ -f "/usr/lib/x86_64-linux-gnu/libjemalloc.so.2" ]; then
        export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libjemalloc.so.2"
        log_info "Using jemalloc memory allocator"
    fi
    
    # Set Python optimizations
    export PYTHONOPTIMIZE=1
    export PYTHONHASHSEED=random
    
    # Set uvicorn worker optimizations
    export UVICORN_BACKLOG=2048
    export UVICORN_MAX_REQUESTS=10000
    export UVICORN_MAX_REQUESTS_JITTER=1000
    
    log_success "Performance optimizations applied"
}

# Security hardening
apply_security() {
    log_info "Applying security hardening..."
    
    # Set secure environment variables
    export PYTHONHASHSEED=random
    export PYTHONDONTWRITEBYTECODE=1
    
    # Limit resource usage
    ulimit -n 65536  # Max file descriptors
    ulimit -u 4096   # Max processes
    
    # Set secure umask
    umask 0027
    
    log_success "Security hardening applied"
}

# Multi-region setup
setup_multi_region() {
    if [ "$ENABLE_MULTI_REGION" = "true" ]; then
        log_info "Setting up multi-region configuration..."
        
        # Auto-detect region if enabled
        if [ "$REGION_AUTO_DETECT" = "true" ] && [ -z "$AWS_REGION" ]; then
            if command -v curl >/dev/null 2>&1; then
                AWS_REGION=$(curl -s --max-time 2 http://169.254.169.254/latest/meta-data/placement/region 2>/dev/null || echo "us-east-1")
                export AWS_REGION
                log_info "Auto-detected region: $AWS_REGION"
            fi
        fi
        
        # Set region-specific configurations
        export CODESIGN_REGION="${AWS_REGION:-us-east-1}"
        export CODESIGN_MULTI_REGION=true
        
        log_success "Multi-region configuration completed"
    fi
}

# Monitoring setup
setup_monitoring() {
    if [ "$ENABLE_METRICS" = "true" ]; then
        log_info "Setting up monitoring and metrics..."
        
        # Set up Prometheus metrics
        export PROMETHEUS_MULTIPROC_DIR="/tmp/prometheus_multiproc"
        mkdir -p "$PROMETHEUS_MULTIPROC_DIR"
        
        # Set up OpenTelemetry if configured
        if [ "$ENABLE_TRACING" = "true" ]; then
            log_info "Enabling OpenTelemetry tracing..."
            export OTEL_PYTHON_LOGGING_AUTO_INSTRUMENTATION_ENABLED=true
        fi
        
        log_success "Monitoring setup completed"
    fi
}

# Main execution
main() {
    log_info "Starting $APP_NAME entrypoint..."
    log_info "Configuration: Workers=$WORKERS, Port=$PORT, Host=$HOST, Log Level=$LOG_LEVEL"
    
    # Validation and initialization
    validate_environment || exit 1
    check_dependencies || exit 1
    initialize_app || exit 1
    
    # Optional setups
    optimize_performance
    apply_security
    setup_multi_region
    setup_monitoring
    
    # Start the application
    log_info "Starting application server..."
    
    # Build command arguments
    local cmd_args=(
        "uvicorn"
        "codesign_playground.server:app"
        "--host" "$HOST"
        "--port" "$PORT"
        "--workers" "$WORKERS"
        "--worker-class" "uvicorn.workers.UvicornWorker"
        "--log-level" "$LOG_LEVEL"
        "--access-log"
        "--use-colors"
    )
    
    # Add production optimizations
    if [ "$NODE_ENV" = "production" ]; then
        cmd_args+=(
            "--backlog" "2048"
            "--max-requests" "10000"
            "--max-requests-jitter" "1000"
        )
    fi
    
    # Start server in background to handle signals
    "${cmd_args[@]}" &
    SERVER_PID=$!
    
    log_success "Server started with PID: $SERVER_PID"
    
    # Wait a moment for server to initialize
    sleep 3
    
    # Perform health check
    if ! health_check; then
        log_error "Health check failed, shutting down..."
        kill -TERM "$SERVER_PID" 2>/dev/null
        exit 1
    fi
    
    log_success "$APP_NAME is ready and healthy!"
    
    # Wait for server process
    wait "$SERVER_PID"
    local exit_code=$?
    
    log_info "Server process exited with code: $exit_code"
    exit $exit_code
}

# Execute main function with all arguments
main "$@"