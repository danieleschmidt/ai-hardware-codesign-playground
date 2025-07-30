#!/bin/bash

# Health Check Script for AI Hardware Co-Design Playground
# Monitors system health and performs automated recovery

set -euo pipefail

# Configuration
BACKEND_URL="${BACKEND_URL:-http://localhost:8000}"
FRONTEND_URL="${FRONTEND_URL:-http://localhost:3000}"
DATABASE_URL="${DATABASE_URL:-postgresql://postgres:postgres@localhost:5432/postgres}"
REDIS_URL="${REDIS_URL:-redis://localhost:6379/0}"
PROMETHEUS_URL="${PROMETHEUS_URL:-http://localhost:9090}"
GRAFANA_URL="${GRAFANA_URL:-http://localhost:3001}"

# Health check timeout
TIMEOUT=10

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1"
}

# Health check function
check_service() {
    local name=$1
    local url=$2
    local expected_status=${3:-200}
    
    log "Checking $name at $url"
    
    if curl -f -s --max-time $TIMEOUT "$url" > /dev/null; then
        echo -e "${GREEN}✓${NC} $name is healthy"
        return 0
    else
        echo -e "${RED}✗${NC} $name is unhealthy"
        return 1
    fi
}

# Database connectivity check
check_database() {
    log "Checking database connectivity"
    
    if command -v psql > /dev/null; then
        if psql "$DATABASE_URL" -c "SELECT 1;" > /dev/null 2>&1; then
            echo -e "${GREEN}✓${NC} Database is accessible"
            return 0
        else
            echo -e "${RED}✗${NC} Database is not accessible"
            return 1
        fi
    else
        echo -e "${YELLOW}⚠${NC} psql not available, skipping database check"
        return 1
    fi
}

# Redis connectivity check
check_redis() {
    log "Checking Redis connectivity"
    
    if command -v redis-cli > /dev/null; then
        if redis-cli -u "$REDIS_URL" ping > /dev/null 2>&1; then
            echo -e "${GREEN}✓${NC} Redis is accessible"
            return 0
        else
            echo -e "${RED}✗${NC} Redis is not accessible"
            return 1
        fi
    else
        echo -e "${YELLOW}⚠${NC} redis-cli not available, skipping Redis check"
        return 1
    fi
}

# System resource check
check_system_resources() {
    log "Checking system resources"
    
    # Check CPU usage
    cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | awk -F'%' '{print $1}')
    if (( $(echo "$cpu_usage > 80" | bc -l) )); then
        echo -e "${RED}✗${NC} High CPU usage: ${cpu_usage}%"
    else
        echo -e "${GREEN}✓${NC} CPU usage normal: ${cpu_usage}%"
    fi
    
    # Check memory usage
    memory_usage=$(free | grep Mem | awk '{printf "%.2f", $3/$2 * 100.0}')
    if (( $(echo "$memory_usage > 80" | bc -l) )); then
        echo -e "${RED}✗${NC} High memory usage: ${memory_usage}%"
    else
        echo -e "${GREEN}✓${NC} Memory usage normal: ${memory_usage}%"
    fi
    
    # Check disk usage
    disk_usage=$(df -h / | awk 'NR==2 {print $5}' | sed 's/%//')
    if [ "$disk_usage" -gt 80 ]; then
        echo -e "${RED}✗${NC} High disk usage: ${disk_usage}%"
    else
        echo -e "${GREEN}✓${NC} Disk usage normal: ${disk_usage}%"
    fi
}

# Docker container health check
check_docker_containers() {
    log "Checking Docker containers"
    
    if command -v docker > /dev/null; then
        # Get unhealthy containers
        unhealthy=$(docker ps --filter health=unhealthy --format "{{.Names}}" | wc -l)
        stopped=$(docker ps -a --filter status=exited --format "{{.Names}}" | wc -l)
        
        if [ "$unhealthy" -gt 0 ]; then
            echo -e "${RED}✗${NC} $unhealthy unhealthy containers found"
            docker ps --filter health=unhealthy --format "table {{.Names}}\t{{.Status}}"
        else
            echo -e "${GREEN}✓${NC} No unhealthy containers"
        fi
        
        if [ "$stopped" -gt 0 ]; then
            echo -e "${YELLOW}⚠${NC} $stopped stopped containers found"
        fi
    else
        echo -e "${YELLOW}⚠${NC} Docker not available"
    fi
}

# Performance metrics check
check_performance_metrics() {
    log "Checking performance metrics"
    
    # Check if Prometheus is collecting metrics
    if check_service "Prometheus" "$PROMETHEUS_URL/-/healthy" 200; then
        # Query some basic metrics
        response_time=$(curl -s "$PROMETHEUS_URL/api/v1/query?query=http_request_duration_seconds" | jq -r '.data.result[0].value[1]' 2>/dev/null || echo "N/A")
        echo "Average response time: $response_time seconds"
        
        error_rate=$(curl -s "$PROMETHEUS_URL/api/v1/query?query=rate(http_requests_total{status=~\"5..\"}[5m])" | jq -r '.data.result[0].value[1]' 2>/dev/null || echo "N/A")
        echo "Error rate: $error_rate requests/sec"
    fi
}

# Recovery actions
attempt_recovery() {
    local service=$1
    
    log "Attempting recovery for $service"
    
    case $service in
        "backend")
            if command -v docker > /dev/null; then
                docker restart ai-hardware-codesign-backend || true
            else
                systemctl restart codesign-backend || true
            fi
            ;;
        "frontend")
            if command -v docker > /dev/null; then
                docker restart ai-hardware-codesign-frontend || true
            else
                systemctl restart codesign-frontend || true
            fi
            ;;
        "database")
            if command -v docker > /dev/null; then
                docker restart postgres || true
            else
                systemctl restart postgresql || true
            fi
            ;;
        "redis")
            if command -v docker > /dev/null; then
                docker restart redis || true
            else
                systemctl restart redis || true
            fi
            ;;
    esac
    
    sleep 30  # Wait for service to start
}

# Send alert
send_alert() {
    local message=$1
    local severity=${2:-warning}
    
    log "ALERT [$severity]: $message"
    
    # Send to webhook if configured
    if [ -n "${WEBHOOK_URL:-}" ]; then
        curl -X POST "$WEBHOOK_URL" \
            -H "Content-Type: application/json" \
            -d "{\"text\":\"Health Check Alert: $message\", \"severity\":\"$severity\"}" \
            2>/dev/null || true
    fi
    
    # Send email if configured
    if [ -n "${ALERT_EMAIL:-}" ] && command -v mail > /dev/null; then
        echo "$message" | mail -s "Health Check Alert - $severity" "$ALERT_EMAIL" || true
    fi
}

# Main health check function
main() {
    log "Starting comprehensive health check"
    
    local failed_checks=0
    local recovery_attempted=false
    
    # Core service checks
    check_service "Backend Health" "$BACKEND_URL/health" 200 || {
        ((failed_checks++))
        if [ "${AUTO_RECOVERY:-false}" = "true" ]; then
            attempt_recovery "backend"
            recovery_attempted=true
        fi
    }
    
    check_service "Frontend" "$FRONTEND_URL" 200 || {
        ((failed_checks++))
        if [ "${AUTO_RECOVERY:-false}" = "true" ]; then
            attempt_recovery "frontend"
            recovery_attempted=true
        fi
    }
    
    # Infrastructure checks
    check_database || {
        ((failed_checks++))
        if [ "${AUTO_RECOVERY:-false}" = "true" ]; then
            attempt_recovery "database"
            recovery_attempted=true
        fi
    }
    
    check_redis || {
        ((failed_checks++))
        if [ "${AUTO_RECOVERY:-false}" = "true" ]; then
            attempt_recovery "redis"
            recovery_attempted=true
        fi
    }
    
    # Monitoring checks
    check_service "Prometheus" "$PROMETHEUS_URL/-/healthy" 200 || ((failed_checks++))
    check_service "Grafana" "$GRAFANA_URL/api/health" 200 || ((failed_checks++))
    
    # System checks
    check_system_resources
    check_docker_containers
    check_performance_metrics
    
    # If recovery was attempted, wait and re-check
    if [ "$recovery_attempted" = "true" ]; then
        log "Recovery attempted, waiting 60 seconds before re-check"
        sleep 60
        
        # Re-check critical services
        check_service "Backend Health" "$BACKEND_URL/health" 200 || ((failed_checks++))
        check_database || ((failed_checks++))
        check_redis || ((failed_checks++))
    fi
    
    # Final report
    if [ $failed_checks -eq 0 ]; then
        log "All health checks passed ✓"
        exit 0
    else
        log "$failed_checks health check(s) failed"
        send_alert "$failed_checks health check(s) failed" "critical"
        exit 1
    fi
}

# Help function
show_help() {
    cat << EOF
AI Hardware Co-Design Playground Health Check Script

Usage: $0 [OPTIONS]

Options:
    -h, --help              Show this help message
    -r, --recovery          Enable automatic recovery attempts
    -v, --verbose           Enable verbose output
    --backend-url URL       Backend service URL (default: $BACKEND_URL)
    --frontend-url URL      Frontend service URL (default: $FRONTEND_URL)
    --database-url URL      Database connection URL
    --redis-url URL         Redis connection URL
    --webhook-url URL       Webhook URL for alerts
    --alert-email EMAIL     Email address for alerts

Environment Variables:
    AUTO_RECOVERY          Enable automatic recovery (true/false)
    WEBHOOK_URL            Webhook URL for sending alerts
    ALERT_EMAIL            Email address for alert notifications
    
Examples:
    $0                     # Basic health check
    $0 --recovery          # Health check with recovery
    $0 --verbose           # Verbose health check

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -r|--recovery)
            AUTO_RECOVERY=true
            shift
            ;;
        -v|--verbose)
            set -x
            shift
            ;;
        --backend-url)
            BACKEND_URL=$2
            shift 2
            ;;
        --frontend-url)
            FRONTEND_URL=$2
            shift 2
            ;;
        --database-url)
            DATABASE_URL=$2
            shift 2
            ;;
        --redis-url)
            REDIS_URL=$2
            shift 2
            ;;
        --webhook-url)
            WEBHOOK_URL=$2
            shift 2
            ;;
        --alert-email)
            ALERT_EMAIL=$2
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Run main function
main