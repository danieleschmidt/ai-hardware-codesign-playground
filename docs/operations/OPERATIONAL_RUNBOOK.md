# AI Hardware Co-Design Playground - Operational Runbook

## Table of Contents

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Monitoring and Alerting](#monitoring-and-alerting)
4. [Routine Operations](#routine-operations)
5. [Incident Response](#incident-response)
6. [Troubleshooting](#troubleshooting)
7. [Maintenance Procedures](#maintenance-procedures)
8. [Security Operations](#security-operations)
9. [Compliance Operations](#compliance-operations)
10. [Performance Optimization](#performance-optimization)

## Overview

This runbook provides comprehensive operational procedures for the AI Hardware Co-Design Playground production environment. It covers monitoring, incident response, troubleshooting, and maintenance procedures for ensuring system reliability, security, and compliance.

### Key Contacts

- **Primary On-Call**: operations@codesign.example.com
- **Security Team**: security@codesign.example.com  
- **Compliance Officer**: compliance@codesign.example.com
- **Engineering Lead**: engineering@codesign.example.com

### System Status

- **Status Page**: https://status.codesign.example.com
- **Monitoring Dashboard**: https://monitoring.codesign.example.com
- **Log Aggregation**: https://logs.codesign.example.com

## System Architecture

### Core Components

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Ingress   │────│  API Gateway │────│  Backend    │
│   (Nginx)   │    │   (Load     │    │   Services  │
│             │    │   Balancer) │    │             │
└─────────────┘    └─────────────┘    └─────────────┘
                                              │
                          ┌───────────────────┼───────────────────┐
                          │                   │                   │
                    ┌─────────┐         ┌─────────┐         ┌─────────┐
                    │  Redis  │         │Postgres│         │ Celery  │
                    │ (Cache) │         │  (DB)   │         │Workers  │
                    └─────────┘         └─────────┘         └─────────┘
```

### Dependencies

- **External APIs**: None (self-contained system)
- **Storage**: PostgreSQL, Redis, File System
- **Monitoring**: Prometheus, Grafana
- **Security**: Built-in authentication, compliance tracking

## Monitoring and Alerting

### Key Metrics

#### Application Metrics
- **Request Rate**: Requests per second
- **Response Time**: P50, P95, P99 latencies
- **Error Rate**: 4xx and 5xx error percentages
- **Circuit Breaker Status**: Open/closed state
- **Cache Hit Rate**: Redis cache effectiveness

#### Infrastructure Metrics
- **CPU Usage**: Per container and node
- **Memory Usage**: Available vs. used memory
- **Disk Usage**: Storage utilization
- **Network I/O**: Bandwidth utilization

#### Business Metrics
- **Active Users**: Current active sessions
- **Workflow Completions**: Successful design workflows
- **Compliance Events**: Data processing activities
- **Security Events**: Authentication failures, access violations

### Alert Thresholds

| Metric | Warning | Critical | Action |
|--------|---------|----------|--------|
| API Response Time (P95) | >2s | >5s | Scale up, investigate |
| Error Rate | >5% | >10% | Immediate investigation |
| CPU Usage | >70% | >90% | Scale up |
| Memory Usage | >80% | >95% | Scale up, investigate leaks |
| Disk Usage | >80% | >90% | Clean up, add storage |
| PostgreSQL Connections | >150 | >190 | Investigate connection leaks |

### Alert Channels

- **Critical Alerts**: PagerDuty → On-call engineer
- **Warning Alerts**: Slack #alerts channel
- **Security Alerts**: Email to security team + Slack
- **Compliance Alerts**: Email to compliance officer

## Routine Operations

### Daily Checks

```bash
# Check system health
kubectl get pods -n codesign-production
kubectl top nodes
kubectl top pods -n codesign-production

# Verify backup completion
kubectl logs -n codesign-production job/backup-job-$(date +%Y%m%d)

# Check alert status
curl -s http://prometheus:9090/api/v1/alerts | jq '.data.alerts[] | select(.state=="firing")'

# Review security logs
tail -f /var/log/security/auth.log | grep FAIL
```

### Weekly Checks

```bash
# Review storage usage trends
df -h
kubectl get pv

# Analyze performance trends
# Access Grafana dashboard and review weekly metrics

# Security scan
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock aquasec/trivy image codesign-playground:production

# Compliance report generation
kubectl exec -it deployment/codesign-api -- python -c "
from backend.codesign_playground.utils.compliance import get_compliance_manager
import time
manager = get_compliance_manager()
report = manager.generate_compliance_report(time.time() - 604800, time.time())
print(report)
"
```

### Monthly Checks

- Review and rotate secrets
- Update dependencies and security patches
- Conduct disaster recovery testing
- Performance capacity planning
- Compliance audit preparation

## Incident Response

### Severity Levels

#### SEV1 - Critical
- **Definition**: Complete system outage, data loss, security breach
- **Response Time**: 15 minutes
- **Escalation**: Immediate PagerDuty alert + Management notification

#### SEV2 - High
- **Definition**: Significant performance degradation, partial outage
- **Response Time**: 1 hour
- **Escalation**: PagerDuty alert

#### SEV3 - Medium  
- **Definition**: Minor performance issues, non-critical component failure
- **Response Time**: 4 hours
- **Escalation**: Slack notification

#### SEV4 - Low
- **Definition**: Monitoring alerts, minor issues
- **Response Time**: Next business day
- **Escalation**: Ticket creation

### Incident Response Process

1. **Detection & Triage**
   ```bash
   # Quick system check
   ./scripts/health-check.sh
   
   # Check recent deployments
   kubectl rollout history deployment/codesign-api -n codesign-production
   
   # Review error logs
   kubectl logs -n codesign-production deployment/codesign-api --tail=100 | grep ERROR
   ```

2. **Initial Response**
   ```bash
   # Scale up if resource constrained
   kubectl scale deployment codesign-api --replicas=5 -n codesign-production
   
   # Restart unhealthy pods
   kubectl delete pod -l app=codesign-api -n codesign-production --field-selector=status.phase!=Running
   
   # Enable circuit breakers if needed
   kubectl set env deployment/codesign-api CIRCUIT_BREAKER_ENABLED=true -n codesign-production
   ```

3. **Communication**
   - Update status page
   - Notify stakeholders via Slack
   - Create incident tracking ticket

4. **Resolution**
   - Implement fix
   - Verify system recovery
   - Document root cause and lessons learned

## Troubleshooting

### Common Issues

#### High Response Times

**Symptoms**: P95 latency > 2 seconds, user complaints

**Diagnosis**:
```bash
# Check resource utilization
kubectl top pods -n codesign-production

# Review application logs for slow queries
kubectl logs -n codesign-production deployment/codesign-api | grep "duration.*[2-9][0-9][0-9][0-9]ms"

# Check database performance
kubectl exec -it postgres-0 -n codesign-production -- psql -U codesign -d codesign_db -c "
SELECT query, mean_time, calls, total_time 
FROM pg_stat_statements 
ORDER BY mean_time DESC 
LIMIT 10;"
```

**Resolution**:
```bash
# Scale up application
kubectl scale deployment codesign-api --replicas=5 -n codesign-production

# Optimize database queries if needed
# Review and apply database indexes

# Enable caching
kubectl set env deployment/codesign-api CACHE_SIZE_MB=512 -n codesign-production
```

#### High Error Rates

**Symptoms**: 5xx errors > 5%, failed health checks

**Diagnosis**:
```bash
# Check application logs
kubectl logs -n codesign-production deployment/codesign-api --tail=500 | grep -E "(ERROR|CRITICAL|500|502|503)"

# Check dependencies
kubectl get pods -n codesign-production | grep -v Running

# Verify circuit breaker status
kubectl exec -it deployment/codesign-api -n codesign-production -- python -c "
from backend.codesign_playground.utils.monitoring import get_health_status
print(get_health_status())
"
```

**Resolution**:
```bash
# Restart failing pods
kubectl rollout restart deployment/codesign-api -n codesign-production

# Check and fix database connectivity
kubectl exec -it postgres-0 -n codesign-production -- pg_isready

# Reset circuit breakers if needed
kubectl exec -it deployment/codesign-api -n codesign-production -- python -c "
from backend.codesign_playground.utils.circuit_breaker import reset_all_circuit_breakers
reset_all_circuit_breakers()
"
```

#### Memory Leaks

**Symptoms**: Gradual memory increase, OOMKilled pods

**Diagnosis**:
```bash
# Monitor memory usage over time
kubectl top pods -n codesign-production --containers

# Check for memory leaks in application
kubectl exec -it deployment/codesign-api -n codesign-production -- python -c "
import gc
import psutil
process = psutil.Process()
print(f'Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB')
print(f'Objects in memory: {len(gc.get_objects())}')
"

# Review memory allocation patterns
kubectl logs -n codesign-production deployment/codesign-api | grep -i memory
```

**Resolution**:
```bash
# Restart affected pods
kubectl delete pod -l app=codesign-api -n codesign-production

# Adjust memory limits if needed
kubectl patch deployment codesign-api -n codesign-production -p '{"spec":{"template":{"spec":{"containers":[{"name":"codesign-api","resources":{"limits":{"memory":"3Gi"}}}]}}}}'

# Enable memory monitoring
kubectl set env deployment/codesign-api MEMORY_MONITORING_ENABLED=true -n codesign-production
```

#### Database Connection Issues

**Symptoms**: Connection timeouts, "too many connections" errors

**Diagnosis**:
```bash
# Check PostgreSQL status
kubectl exec -it postgres-0 -n codesign-production -- psql -U codesign -d codesign_db -c "
SELECT count(*) as active_connections, state 
FROM pg_stat_activity 
GROUP BY state;"

# Check connection pool status
kubectl logs -n codesign-production deployment/codesign-api | grep -i "connection\|pool"

# Verify database health
kubectl exec -it postgres-0 -n codesign-production -- pg_isready -U codesign
```

**Resolution**:
```bash
# Kill idle connections
kubectl exec -it postgres-0 -n codesign-production -- psql -U codesign -d codesign_db -c "
SELECT pg_terminate_backend(pid) 
FROM pg_stat_activity 
WHERE state = 'idle' AND state_change < now() - interval '5 minutes';"

# Restart application to reset connection pools
kubectl rollout restart deployment/codesign-api -n codesign-production

# Adjust connection pool settings if needed
kubectl set env deployment/codesign-api DB_POOL_SIZE=20 -n codesign-production
```

## Maintenance Procedures

### Planned Maintenance

#### Application Updates

```bash
# 1. Create maintenance announcement
echo "Maintenance window: $(date) - Expected duration: 30 minutes" > /tmp/maintenance.txt

# 2. Scale up for redundancy
kubectl scale deployment codesign-api --replicas=4 -n codesign-production

# 3. Update application image
kubectl set image deployment/codesign-api codesign-api=codesign-playground:v2.1.0 -n codesign-production

# 4. Monitor rollout
kubectl rollout status deployment/codesign-api -n codesign-production --timeout=600s

# 5. Verify health
./scripts/health-check.sh

# 6. Scale back to normal
kubectl scale deployment codesign-api --replicas=3 -n codesign-production
```

#### Database Maintenance

```bash
# 1. Create database backup
kubectl exec -it postgres-0 -n codesign-production -- pg_dump -U codesign -d codesign_db > /backup/pre_maintenance_$(date +%Y%m%d).sql

# 2. Run maintenance tasks
kubectl exec -it postgres-0 -n codesign-production -- psql -U codesign -d codesign_db -c "
VACUUM ANALYZE;
REINDEX DATABASE codesign_db;
"

# 3. Update statistics
kubectl exec -it postgres-0 -n codesign-production -- psql -U codesign -d codesign_db -c "
ANALYZE;
"

# 4. Verify database health
kubectl exec -it postgres-0 -n codesign-production -- pg_isready -U codesign
```

#### Certificate Renewal

```bash
# 1. Check certificate expiration
kubectl get certificate -n codesign-production

# 2. Renew certificates (if using cert-manager)
kubectl delete certificate codesign-tls -n codesign-production
# cert-manager will automatically recreate

# 3. Verify new certificate
kubectl describe certificate codesign-tls -n codesign-production

# 4. Test HTTPS connectivity
curl -I https://codesign.example.com
```

### Emergency Procedures

#### Rollback Process

```bash
# 1. Check rollout history
kubectl rollout history deployment/codesign-api -n codesign-production

# 2. Rollback to previous version
kubectl rollout undo deployment/codesign-api -n codesign-production

# 3. Monitor rollback
kubectl rollout status deployment/codesign-api -n codesign-production

# 4. Verify system health
./scripts/health-check.sh
```

#### Scale Down (Emergency)

```bash
# 1. Scale down to minimum viable service
kubectl scale deployment codesign-api --replicas=1 -n codesign-production
kubectl scale deployment celery-worker --replicas=1 -n codesign-production

# 2. Disable non-essential features
kubectl set env deployment/codesign-api CIRCUIT_BREAKER_ENABLED=true -n codesign-production
kubectl set env deployment/codesign-api RATE_LIMITING_ENABLED=true -n codesign-production

# 3. Enable maintenance mode
kubectl set env deployment/codesign-api MAINTENANCE_MODE=true -n codesign-production
```

## Security Operations

### Security Monitoring

```bash
# Check authentication failures
kubectl logs -n codesign-production deployment/codesign-api | grep -i "authentication.*failed\|unauthorized"

# Review compliance events
kubectl exec -it deployment/codesign-api -n codesign-production -- python -c "
from backend.codesign_playground.utils.compliance import get_compliance_manager
manager = get_compliance_manager()
logs = [log for log in manager._audit_logs if log.risk_level in ['high', 'critical']]
for log in logs[-10:]:
    print(f'{log.timestamp}: {log.action} - {log.result} - {log.risk_level}')
"

# Scan for vulnerabilities
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock aquasec/trivy image codesign-playground:production
```

### Incident Response (Security)

```bash
# 1. Isolate affected components
kubectl scale deployment codesign-api --replicas=0 -n codesign-production

# 2. Preserve evidence
kubectl logs -n codesign-production deployment/codesign-api > /security/incident_$(date +%Y%m%d_%H%M%S).log

# 3. Rotate secrets
kubectl delete secret codesign-secrets -n codesign-production
kubectl create secret generic codesign-secrets \
  --from-literal=postgres-password="$(openssl rand -base64 32)" \
  --from-literal=redis-password="$(openssl rand -base64 32)" \
  --from-literal=secret-key="$(openssl rand -base64 64)" \
  -n codesign-production

# 4. Restart with new secrets
kubectl rollout restart deployment/codesign-api -n codesign-production
```

## Compliance Operations

### Data Subject Request Handling

```bash
# Process GDPR access request
kubectl exec -it deployment/codesign-api -n codesign-production -- python -c "
from backend.codesign_playground.utils.compliance import get_compliance_manager
manager = get_compliance_manager()
response = manager.handle_data_subject_request(
    user_id='$USER_ID',
    request_type='access',
    verification_method='email'
)
print(response)
"

# Process data deletion request
kubectl exec -it deployment/codesign-api -n codesign-production -- python -c "
from backend.codesign_playground.utils.compliance import get_compliance_manager
manager = get_compliance_manager()
response = manager.handle_data_subject_request(
    user_id='$USER_ID',
    request_type='erasure'
)
print(f'Deleted {response[\"deleted_records\"]} records')
"
```

### Compliance Reporting

```bash
# Generate monthly compliance report
kubectl exec -it deployment/codesign-api -n codesign-production -- python -c "
from backend.codesign_playground.utils.compliance import get_compliance_manager
import time
manager = get_compliance_manager()
start_time = time.time() - (30 * 24 * 3600)  # 30 days ago
end_time = time.time()
report = manager.generate_compliance_report(start_time, end_time)
import json
print(json.dumps(report, indent=2))
" > /compliance/monthly_report_$(date +%Y%m).json
```

## Performance Optimization

### Performance Tuning

```bash
# Optimize PostgreSQL
kubectl exec -it postgres-0 -n codesign-production -- psql -U codesign -d codesign_db -c "
-- Update statistics
ANALYZE;

-- Check slow queries
SELECT query, mean_time, calls 
FROM pg_stat_statements 
WHERE mean_time > 1000 
ORDER BY mean_time DESC 
LIMIT 10;

-- Check index usage
SELECT schemaname, tablename, attname, n_distinct, correlation 
FROM pg_stats 
WHERE schemaname = 'public' 
ORDER BY n_distinct DESC;
"

# Optimize Redis
kubectl exec -it redis-0 -n codesign-production -- redis-cli --no-auth-warning -a "$REDIS_PASSWORD" info memory

# Application performance tuning
kubectl set env deployment/codesign-api CACHE_SIZE_MB=512 -n codesign-production
kubectl set env deployment/codesign-api MAX_WORKERS=12 -n codesign-production
```

### Capacity Planning

```bash
# Resource utilization analysis
kubectl top nodes
kubectl top pods -n codesign-production

# Growth trend analysis
# Access Grafana dashboard for historical metrics

# Scaling recommendations
echo "Current resource usage and scaling recommendations:"
kubectl describe hpa codesign-api-hpa -n codesign-production
```

---

## Appendix

### Useful Commands

```bash
# Quick health check
alias health-check='kubectl get pods -n codesign-production && kubectl top nodes'

# View logs
alias api-logs='kubectl logs -n codesign-production deployment/codesign-api --tail=100 -f'
alias db-logs='kubectl logs -n codesign-production deployment/postgres --tail=100 -f'

# Scale operations
alias scale-up='kubectl scale deployment codesign-api --replicas=5 -n codesign-production'
alias scale-down='kubectl scale deployment codesign-api --replicas=3 -n codesign-production'
```

### Emergency Contacts

- **Infrastructure Team**: +1-555-INFRA-01
- **Security Team**: +1-555-SEC-911
- **Database Team**: +1-555-DB-HELP
- **Network Team**: +1-555-NET-SOS

### External Dependencies

- **DNS Provider**: CloudFlare
- **SSL Certificates**: Let's Encrypt
- **Container Registry**: Docker Hub
- **Backup Storage**: AWS S3 (if enabled)

---

*Last Updated: 2024-08-16*  
*Version: 2.0*  
*Document Owner: Platform Engineering Team*