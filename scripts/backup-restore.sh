#!/bin/bash

# Backup and Restore Script for AI Hardware Co-Design Playground
# Handles database backups, configuration backups, and disaster recovery

set -euo pipefail

# Configuration
BACKUP_DIR="${BACKUP_DIR:-/opt/backups/codesign-playground}"
DATABASE_URL="${DATABASE_URL:-postgresql://postgres:postgres@localhost:5432/postgres}"
REDIS_URL="${REDIS_URL:-redis://localhost:6379/0}"
S3_BUCKET="${S3_BUCKET:-}"
RETENTION_DAYS="${RETENTION_DAYS:-30}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Logging function
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1"
}

error() {
    echo -e "${RED}ERROR:${NC} $1" >&2
}

success() {
    echo -e "${GREEN}SUCCESS:${NC} $1"
}

warning() {
    echo -e "${YELLOW}WARNING:${NC} $1"
}

info() {
    echo -e "${BLUE}INFO:${NC} $1"
}

# Create backup directory
create_backup_dir() {
    local backup_path="$BACKUP_DIR/$(date '+%Y-%m-%d_%H-%M-%S')"
    mkdir -p "$backup_path"
    echo "$backup_path"
}

# Database backup
backup_database() {
    local backup_path=$1
    local db_backup_file="$backup_path/database.sql"
    
    log "Starting database backup"
    
    if command -v pg_dump > /dev/null; then
        pg_dump "$DATABASE_URL" > "$db_backup_file"
        gzip "$db_backup_file"
        success "Database backup completed: $db_backup_file.gz"
    else
        error "pg_dump not found. Cannot backup PostgreSQL database."
        return 1
    fi
}

# Redis backup
backup_redis() {
    local backup_path=$1
    local redis_backup_file="$backup_path/redis.rdb"
    
    log "Starting Redis backup"
    
    if command -v redis-cli > /dev/null; then
        # Get Redis host and port from URL
        redis_host=$(echo "$REDIS_URL" | sed -n 's/.*:\/\/\([^:]*\):.*/\1/p')
        redis_port=$(echo "$REDIS_URL" | sed -n 's/.*:\([0-9]*\)\/.*/\1/p')
        
        redis-cli -h "$redis_host" -p "$redis_port" --rdb "$redis_backup_file"
        gzip "$redis_backup_file"
        success "Redis backup completed: $redis_backup_file.gz"
    else
        warning "redis-cli not found. Skipping Redis backup."
    fi
}

# Configuration files backup
backup_configurations() {
    local backup_path=$1
    local config_backup_dir="$backup_path/configurations"
    
    log "Starting configuration backup"
    mkdir -p "$config_backup_dir"
    
    # Backup important configuration files
    local config_files=(
        ".env"
        "docker-compose.yml"
        "docker-compose.dev.yml"
        "deployment/docker-compose.monitoring.yml"
        "monitoring/prometheus.yml"
        "monitoring/alert_rules.yml"
        "pyproject.toml"
        "package.json"
        ".pre-commit-config.yaml"
        "renovate.json"
    )
    
    for config_file in "${config_files[@]}"; do
        if [ -f "$config_file" ]; then
            cp "$config_file" "$config_backup_dir/"
            log "Backed up: $config_file"
        fi
    done
    
    # Backup entire configuration directories
    local config_dirs=(
        "docs/"
        "scripts/"
        "monitoring/"
        ".github/"
    )
    
    for config_dir in "${config_dirs[@]}"; do
        if [ -d "$config_dir" ]; then
            cp -r "$config_dir" "$config_backup_dir/"
            log "Backed up directory: $config_dir"
        fi
    done
    
    # Create archive
    tar -czf "$backup_path/configurations.tar.gz" -C "$config_backup_dir" .
    rm -rf "$config_backup_dir"
    success "Configuration backup completed: $backup_path/configurations.tar.gz"
}

# Application data backup
backup_application_data() {
    local backup_path=$1
    local data_backup_dir="$backup_path/application_data"
    
    log "Starting application data backup"
    mkdir -p "$data_backup_dir"
    
    # Backup user uploads and generated files
    local data_dirs=(
        "data/"
        "notebooks/"
        "backend/uploads/"
        "backend/generated/"
        "logs/"
    )
    
    for data_dir in "${data_dirs[@]}"; do
        if [ -d "$data_dir" ]; then
            cp -r "$data_dir" "$data_backup_dir/"
            log "Backed up data directory: $data_dir"
        fi
    done
    
    # Create archive
    if [ "$(ls -A "$data_backup_dir" 2>/dev/null)" ]; then
        tar -czf "$backup_path/application_data.tar.gz" -C "$data_backup_dir" .
        rm -rf "$data_backup_dir"
        success "Application data backup completed: $backup_path/application_data.tar.gz"
    else
        warning "No application data found to backup"
        rm -rf "$data_backup_dir"
    fi
}

# Docker volumes backup
backup_docker_volumes() {
    local backup_path=$1
    
    if ! command -v docker > /dev/null; then
        warning "Docker not found. Skipping volume backup."
        return
    fi
    
    log "Starting Docker volumes backup"
    
    # Get list of volumes
    local volumes=$(docker volume ls -q | grep -E "(postgres|redis|grafana|prometheus)" || true)
    
    if [ -n "$volumes" ]; then
        for volume in $volumes; do
            log "Backing up Docker volume: $volume"
            docker run --rm \
                -v "$volume":/source:ro \
                -v "$backup_path":/backup \
                alpine:latest \
                tar -czf "/backup/volume_${volume}.tar.gz" -C /source .
        done
        success "Docker volumes backup completed"
    else
        info "No relevant Docker volumes found"
    fi
}

# Upload to S3 (if configured)
upload_to_s3() {
    local backup_path=$1
    
    if [ -z "$S3_BUCKET" ]; then
        info "S3_BUCKET not configured. Skipping cloud backup."
        return
    fi
    
    if ! command -v aws > /dev/null; then
        warning "AWS CLI not found. Skipping S3 upload."
        return
    fi
    
    local backup_name=$(basename "$backup_path")
    local s3_path="s3://$S3_BUCKET/backups/codesign-playground/$backup_name"
    
    log "Uploading backup to S3: $s3_path"
    
    # Create archive of entire backup
    tar -czf "$backup_path.tar.gz" -C "$(dirname "$backup_path")" "$backup_name"
    
    # Upload to S3
    aws s3 cp "$backup_path.tar.gz" "$s3_path.tar.gz"
    
    # Clean up local archive
    rm "$backup_path.tar.gz"
    
    success "Backup uploaded to S3: $s3_path.tar.gz"
}

# Clean old backups
cleanup_old_backups() {
    log "Cleaning up backups older than $RETENTION_DAYS days"
    
    find "$BACKUP_DIR" -type d -name "*_*-*-*" -mtime +$RETENTION_DAYS -exec rm -rf {} \; 2>/dev/null || true
    
    # Also clean S3 if configured
    if [ -n "$S3_BUCKET" ] && command -v aws > /dev/null; then
        local cutoff_date=$(date -d "$RETENTION_DAYS days ago" '+%Y-%m-%d')
        aws s3 ls "s3://$S3_BUCKET/backups/codesign-playground/" | \
            awk '{print $4}' | \
            while read -r file; do
                if [[ "$file" < "$cutoff_date" ]]; then
                    aws s3 rm "s3://$S3_BUCKET/backups/codesign-playground/$file"
                    log "Deleted old S3 backup: $file"
                fi
            done
    fi
    
    success "Cleanup completed"
}

# Restore database
restore_database() {
    local backup_file=$1
    
    if [ ! -f "$backup_file" ]; then
        error "Database backup file not found: $backup_file"
        return 1
    fi
    
    log "Restoring database from: $backup_file"
    
    # Handle gzipped files
    if [[ "$backup_file" == *.gz ]]; then
        gunzip -c "$backup_file" | psql "$DATABASE_URL"
    else
        psql "$DATABASE_URL" < "$backup_file"
    fi
    
    success "Database restore completed"
}

# Restore Redis
restore_redis() {
    local backup_file=$1
    
    if [ ! -f "$backup_file" ]; then
        error "Redis backup file not found: $backup_file"
        return 1
    fi
    
    log "Restoring Redis from: $backup_file"
    
    # Get Redis host and port from URL
    redis_host=$(echo "$REDIS_URL" | sed -n 's/.*:\/\/\([^:]*\):.*/\1/p')
    redis_port=$(echo "$REDIS_URL" | sed -n 's/.*:\([0-9]*\)\/.*/\1/p')
    
    # Stop Redis, restore backup, start Redis
    if command -v docker > /dev/null && docker ps | grep -q redis; then
        docker stop redis
        
        # Handle gzipped files
        if [[ "$backup_file" == *.gz ]]; then
            gunzip -c "$backup_file" > /tmp/redis_restore.rdb
        else
            cp "$backup_file" /tmp/redis_restore.rdb
        fi
        
        # Copy backup to Redis data directory
        docker run --rm -v redis_data:/data alpine cp /tmp/redis_restore.rdb /data/dump.rdb
        docker start redis
        rm -f /tmp/redis_restore.rdb
    else
        warning "Redis container not found. Manual restore may be required."
    fi
    
    success "Redis restore completed"
}

# Restore configurations
restore_configurations() {
    local backup_file=$1
    
    if [ ! -f "$backup_file" ]; then
        error "Configuration backup file not found: $backup_file"
        return 1
    fi
    
    log "Restoring configurations from: $backup_file"
    
    # Extract to temporary directory
    local temp_dir=$(mktemp -d)
    tar -xzf "$backup_file" -C "$temp_dir"
    
    # Restore configuration files
    cp -r "$temp_dir"/* .
    
    # Clean up
    rm -rf "$temp_dir"
    
    success "Configuration restore completed"
}

# Full backup
perform_full_backup() {
    log "Starting full backup process"
    
    local backup_path=$(create_backup_dir)
    info "Backup location: $backup_path"
    
    # Perform all backup operations
    backup_database "$backup_path" || warning "Database backup failed"
    backup_redis "$backup_path" || warning "Redis backup failed"
    backup_configurations "$backup_path" || warning "Configuration backup failed"
    backup_application_data "$backup_path" || warning "Application data backup failed"
    backup_docker_volumes "$backup_path" || warning "Docker volumes backup failed"
    
    # Upload to cloud storage
    upload_to_s3 "$backup_path" || warning "S3 upload failed"
    
    # Create backup manifest
    cat > "$backup_path/manifest.txt" << EOF
Backup created: $(date)
Backup path: $backup_path
Components:
- Database: $([ -f "$backup_path/database.sql.gz" ] && echo "✓" || echo "✗")
- Redis: $([ -f "$backup_path/redis.rdb.gz" ] && echo "✓" || echo "✗")
- Configurations: $([ -f "$backup_path/configurations.tar.gz" ] && echo "✓" || echo "✗")
- Application Data: $([ -f "$backup_path/application_data.tar.gz" ] && echo "✓" || echo "✗")
- Docker Volumes: $(find "$backup_path" -name "volume_*.tar.gz" | wc -l) volumes
EOF
    
    success "Full backup completed: $backup_path"
    
    # Cleanup old backups
    cleanup_old_backups
}

# Disaster recovery
disaster_recovery() {
    local backup_path=$1
    
    if [ ! -d "$backup_path" ]; then
        error "Backup directory not found: $backup_path"
        return 1
    fi
    
    log "Starting disaster recovery from: $backup_path"
    
    # Stop services
    if command -v docker > /dev/null; then
        docker-compose down || true
    fi
    
    # Restore components
    if [ -f "$backup_path/database.sql.gz" ] || [ -f "$backup_path/database.sql" ]; then
        restore_database "$backup_path/database.sql"*
    fi
    
    if [ -f "$backup_path/redis.rdb.gz" ] || [ -f "$backup_path/redis.rdb" ]; then
        restore_redis "$backup_path/redis.rdb"*
    fi
    
    if [ -f "$backup_path/configurations.tar.gz" ]; then
        restore_configurations "$backup_path/configurations.tar.gz"
    fi
    
    # Restore Docker volumes
    for volume_backup in "$backup_path"/volume_*.tar.gz; do
        if [ -f "$volume_backup" ]; then
            volume_name=$(basename "$volume_backup" | sed 's/^volume_//' | sed 's/.tar.gz$//')
            log "Restoring Docker volume: $volume_name"
            docker volume create "$volume_name" || true
            docker run --rm \
                -v "$volume_name":/target \
                -v "$backup_path":/backup \
                alpine:latest \
                tar -xzf "/backup/$(basename "$volume_backup")" -C /target
        fi
    done
    
    # Start services
    if [ -f "docker-compose.yml" ]; then
        docker-compose up -d
    fi
    
    success "Disaster recovery completed"
}

# Show help
show_help() {
    cat << EOF
AI Hardware Co-Design Playground Backup & Restore Script

Usage: $0 [COMMAND] [OPTIONS]

Commands:
    backup                  Perform full backup
    restore DATABASE        Restore database from backup file
    restore-redis          Restore Redis from backup file
    restore-config         Restore configurations from backup file
    disaster-recovery      Full disaster recovery from backup directory
    cleanup                Clean up old backups
    list                   List available backups

Options:
    -h, --help             Show this help message
    --backup-dir DIR       Backup directory (default: $BACKUP_DIR)
    --retention-days N     Retention period in days (default: $RETENTION_DAYS)
    --s3-bucket BUCKET     S3 bucket for cloud backup
    --database-url URL     Database connection URL
    --redis-url URL        Redis connection URL

Environment Variables:
    BACKUP_DIR            Backup directory path
    DATABASE_URL          PostgreSQL connection string
    REDIS_URL             Redis connection string
    S3_BUCKET             S3 bucket for cloud backup
    RETENTION_DAYS        Backup retention period
    AWS_ACCESS_KEY_ID     AWS credentials for S3
    AWS_SECRET_ACCESS_KEY AWS credentials for S3

Examples:
    $0 backup                                    # Full backup
    $0 restore database backup.sql.gz           # Restore database
    $0 disaster-recovery /opt/backups/2024-01-01_12-00-00
    $0 cleanup                                   # Clean old backups

EOF
}

# Parse command line arguments
case "${1:-}" in
    backup)
        perform_full_backup
        ;;
    restore)
        case "${2:-}" in
            database)
                restore_database "${3:-}"
                ;;
            redis)
                restore_redis "${3:-}"
                ;;
            config)
                restore_configurations "${3:-}"
                ;;
            *)
                error "Invalid restore target. Use: database, redis, or config"
                exit 1
                ;;
        esac
        ;;
    disaster-recovery)
        disaster_recovery "${2:-}"
        ;;
    cleanup)
        cleanup_old_backups
        ;;
    list)
        log "Available backups:"
        ls -la "$BACKUP_DIR" 2>/dev/null || echo "No backups found"
        ;;
    -h|--help)
        show_help
        exit 0
        ;;
    *)
        error "Invalid command. Use --help for usage information."
        exit 1
        ;;
esac