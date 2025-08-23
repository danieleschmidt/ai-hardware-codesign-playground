#!/bin/bash
# PostgreSQL Backup Script for Production

set -e

DB_NAME="codesign_production"
DB_USER="codesign_user"
BACKUP_DIR="/backups"
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="${BACKUP_DIR}/${DB_NAME}_${DATE}.sql.gz"

# Create backup directory
mkdir -p ${BACKUP_DIR}

# Perform backup
pg_dump -h ${DATABASE_HOST:-localhost} -U ${DB_USER} -d ${DB_NAME} | gzip > ${BACKUP_FILE}

# Verify backup
if [ -f "${BACKUP_FILE}" ] && [ -s "${BACKUP_FILE}" ]; then
    echo "Backup completed successfully: ${BACKUP_FILE}"
    
    # Clean up old backups (keep last 30 days)
    find ${BACKUP_DIR} -name "${DB_NAME}_*.sql.gz" -mtime +30 -delete
    
    echo "Old backups cleaned up"
else
    echo "Backup failed!" >&2
    exit 1
fi
