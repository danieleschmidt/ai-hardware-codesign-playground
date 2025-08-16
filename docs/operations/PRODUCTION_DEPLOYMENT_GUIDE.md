# AI Hardware Co-Design Playground - Production Deployment Guide

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Infrastructure Setup](#infrastructure-setup)
4. [Security Configuration](#security-configuration)
5. [Application Deployment](#application-deployment)
6. [Monitoring Setup](#monitoring-setup)
7. [Backup Configuration](#backup-configuration)
8. [Testing and Validation](#testing-and-validation)
9. [Go-Live Checklist](#go-live-checklist)
10. [Post-Deployment](#post-deployment)

## Overview

This guide provides step-by-step instructions for deploying the AI Hardware Co-Design Playground to a production environment with enhanced robustness features, comprehensive monitoring, and full compliance capabilities.

### Deployment Architecture

```
Internet
    │
┌───▼────┐     ┌──────────┐     ┌─────────────┐
│  CDN   │────▶│  WAF     │────▶│ Load        │
│        │     │          │     │ Balancer    │
└────────┘     └──────────┘     └─────────────┘
                                        │
                               ┌────────▼────────┐
                               │   Kubernetes    │
                               │    Cluster      │
                               └─────────────────┘
                                        │
                    ┌───────────────────┼───────────────────┐
                    │                   │                   │
            ┌───────▼───────┐  ┌────────▼────────┐  ┌───────▼───────┐
            │   API Pods    │  │  Worker Pods    │  │  Database     │
            │  (3 replicas) │  │  (2 replicas)   │  │   Services    │
            └───────────────┘  └─────────────────┘  └───────────────┘
```

### Key Features

- **High Availability**: Multi-replica deployments with auto-scaling
- **Security**: End-to-end encryption, RBAC, network policies
- **Monitoring**: Comprehensive observability with Prometheus and Grafana
- **Compliance**: GDPR, CCPA, PDPA compliance with audit logging
- **Backup & Recovery**: Automated backups with disaster recovery
- **Performance**: Optimized for production workloads

## Prerequisites

### Infrastructure Requirements

#### Kubernetes Cluster
- **Minimum**: 3 nodes, 8 vCPUs, 32GB RAM each
- **Recommended**: 5 nodes, 16 vCPUs, 64GB RAM each
- **Kubernetes Version**: 1.25+
- **Storage**: SSD-backed persistent volumes
- **Network**: CNI with network policy support

#### External Dependencies
- **Domain**: Registered domain with DNS management
- **SSL Certificates**: Let's Encrypt or commercial CA
- **Container Registry**: Docker Hub, ECR, or GCR
- **Backup Storage**: S3-compatible object storage (optional)

### Software Requirements

```bash
# Required tools
kubectl >= 1.25
helm >= 3.10
docker >= 20.10
git >= 2.30

# Optional tools
terraform >= 1.0  # For infrastructure automation
ansible >= 4.0    # For configuration management
```

### Access Requirements

- Kubernetes cluster admin access
- Docker registry push access
- DNS management access
- SSL certificate management access

## Infrastructure Setup

### 1. Kubernetes Cluster Preparation

```bash
# Verify cluster is ready
kubectl cluster-info
kubectl get nodes
kubectl get storageclasses

# Create namespace
kubectl create namespace codesign-production
kubectl label namespace codesign-production environment=production

# Set default namespace
kubectl config set-context --current --namespace=codesign-production
```

### 2. Storage Classes Configuration

```yaml
# Create storage classes for different performance tiers
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: fast-ssd
provisioner: kubernetes.io/gce-pd  # Adjust for your cloud provider
parameters:
  type: pd-ssd
  replication-type: regional-pd
allowVolumeExpansion: true
volumeBindingMode: WaitForFirstConsumer
---
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: encrypted-ssd
provisioner: kubernetes.io/gce-pd
parameters:
  type: pd-ssd
  encrypted: "true"
allowVolumeExpansion: true
volumeBindingMode: WaitForFirstConsumer
---
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: backup-storage
provisioner: kubernetes.io/gce-pd
parameters:
  type: pd-standard
  replication-type: regional-pd
allowVolumeExpansion: true
volumeBindingMode: WaitForFirstConsumer
```

```bash
# Apply storage classes
kubectl apply -f storage-classes.yaml
```

### 3. Network Policies

```yaml
# Default deny-all network policy
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: default-deny-all
  namespace: codesign-production
spec:
  podSelector: {}
  policyTypes:
  - Ingress
  - Egress
```

```bash
# Apply network policies
kubectl apply -f network-policies.yaml
```

## Security Configuration

### 1. RBAC Setup

```yaml
# Service account for application
apiVersion: v1
kind: ServiceAccount
metadata:
  name: codesign-api-sa
  namespace: codesign-production
---
# Role for application permissions
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: codesign-api-role
  namespace: codesign-production
rules:
- apiGroups: [""]
  resources: ["pods", "configmaps", "secrets"]
  verbs: ["get", "list", "watch"]
- apiGroups: ["apps"]
  resources: ["deployments"]
  verbs: ["get", "list", "watch", "patch"]
---
# Role binding
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: codesign-api-binding
  namespace: codesign-production
subjects:
- kind: ServiceAccount
  name: codesign-api-sa
  namespace: codesign-production
roleRef:
  kind: Role
  name: codesign-api-role
  apiGroup: rbac.authorization.k8s.io
```

### 2. Secrets Management

```bash
# Generate secure passwords
export POSTGRES_PASSWORD=$(openssl rand -base64 32)
export REDIS_PASSWORD=$(openssl rand -base64 32)
export SECRET_KEY=$(openssl rand -base64 64)
export JWT_SECRET=$(openssl rand -base64 64)

# Create secrets
kubectl create secret generic codesign-secrets \
  --from-literal=postgres-password="$POSTGRES_PASSWORD" \
  --from-literal=redis-password="$REDIS_PASSWORD" \
  --from-literal=secret-key="$SECRET_KEY" \
  --from-literal=jwt-secret="$JWT_SECRET" \
  --namespace=codesign-production

# Verify secrets
kubectl get secrets -n codesign-production
```

### 3. Pod Security Standards

```yaml
# Pod Security Policy
apiVersion: policy/v1beta1
kind: PodSecurityPolicy
metadata:
  name: codesign-psp
spec:
  privileged: false
  allowPrivilegeEscalation: false
  requiredDropCapabilities:
    - ALL
  volumes:
    - 'configMap'
    - 'emptyDir'
    - 'projected'
    - 'secret'
    - 'downwardAPI'
    - 'persistentVolumeClaim'
  runAsUser:
    rule: 'MustRunAsNonRoot'
  seLinux:
    rule: 'RunAsAny'
  fsGroup:
    rule: 'RunAsAny'
```

## Application Deployment

### 1. Build and Push Container Images

```bash
# Build production image
docker build -f docker/production.dockerfile \
  --build-arg BUILD_VERSION=$(git describe --tags) \
  --build-arg BUILD_COMMIT=$(git rev-parse HEAD) \
  -t codesign-playground:$(git describe --tags) .

# Tag for registry
docker tag codesign-playground:$(git describe --tags) \
  your-registry.com/codesign-playground:$(git describe --tags)

# Push to registry
docker push your-registry.com/codesign-playground:$(git describe --tags)

# Update latest tag
docker tag codesign-playground:$(git describe --tags) \
  your-registry.com/codesign-playground:latest
docker push your-registry.com/codesign-playground:latest
```

### 2. Configuration Management

```bash
# Create configuration from template
envsubst < deployment/kubernetes/configmap-template.yaml > configmap.yaml

# Apply configuration
kubectl apply -f configmap.yaml
```

### 3. Database Deployment

```bash
# Create PostgreSQL configuration
kubectl create configmap postgres-config \
  --from-file=postgresql.conf=postgres/postgresql.conf \
  --from-file=pg_hba.conf=postgres/pg_hba.conf \
  --namespace=codesign-production

# Create initialization scripts
kubectl create configmap postgres-init-scripts \
  --from-file=scripts/init-db.sql \
  --from-file=scripts/init-compliance.sql \
  --from-file=scripts/init-monitoring.sql \
  --namespace=codesign-production

# Deploy PostgreSQL
kubectl apply -f deployment/kubernetes/postgres.yaml

# Wait for PostgreSQL to be ready
kubectl wait --for=condition=ready pod -l app=postgres --timeout=300s
```

### 4. Redis Deployment

```bash
# Create Redis configuration
kubectl create configmap redis-config \
  --from-file=redis.conf=redis/redis.conf \
  --namespace=codesign-production

# Deploy Redis
kubectl apply -f deployment/kubernetes/redis.yaml

# Wait for Redis to be ready
kubectl wait --for=condition=ready pod -l app=redis --timeout=300s
```

### 5. Application Deployment

```bash
# Update image in deployment manifest
sed -i "s|image: codesign-playground:.*|image: your-registry.com/codesign-playground:$(git describe --tags)|" \
  deployment/kubernetes/production-enhanced.yaml

# Deploy application
kubectl apply -f deployment/kubernetes/production-enhanced.yaml

# Wait for deployment to complete
kubectl rollout status deployment/codesign-api --timeout=600s
kubectl rollout status deployment/celery-worker --timeout=600s

# Verify deployment
kubectl get pods -l app=codesign-api
kubectl get pods -l app=celery-worker
```

### 6. Ingress and Load Balancer

```bash
# Install NGINX Ingress Controller (if not already installed)
helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx
helm repo update
helm install ingress-nginx ingress-nginx/ingress-nginx \
  --namespace ingress-nginx \
  --create-namespace \
  --set controller.metrics.enabled=true \
  --set controller.podAnnotations."prometheus\.io/scrape"="true" \
  --set controller.podAnnotations."prometheus\.io/port"="10254"

# Install cert-manager for SSL certificates
helm repo add jetstack https://charts.jetstack.io
helm repo update
helm install cert-manager jetstack/cert-manager \
  --namespace cert-manager \
  --create-namespace \
  --version v1.12.0 \
  --set installCRDs=true

# Create ClusterIssuer for Let's Encrypt
kubectl apply -f - <<EOF
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-prod
spec:
  acme:
    server: https://acme-v02.api.letsencrypt.org/directory
    email: admin@codesign.example.com
    privateKeySecretRef:
      name: letsencrypt-prod
    solvers:
    - http01:
        ingress:
          class: nginx
EOF

# Deploy ingress
kubectl apply -f deployment/kubernetes/ingress.yaml

# Verify SSL certificate
kubectl get certificate -n codesign-production
kubectl describe certificate codesign-tls -n codesign-production
```

## Monitoring Setup

### 1. Prometheus Stack

```bash
# Add Prometheus helm repository
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

# Install Prometheus stack
helm install prometheus-stack prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --create-namespace \
  --set prometheus.prometheusSpec.retention=30d \
  --set prometheus.prometheusSpec.storageSpec.volumeClaimTemplate.spec.storageClassName=fast-ssd \
  --set prometheus.prometheusSpec.storageSpec.volumeClaimTemplate.spec.resources.requests.storage=50Gi \
  --set grafana.persistence.enabled=true \
  --set grafana.persistence.storageClassName=fast-ssd \
  --set grafana.persistence.size=10Gi \
  --set grafana.adminPassword=admin123 \
  --values monitoring/prometheus-values.yaml

# Wait for deployment
kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=prometheus --timeout=300s -n monitoring
kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=grafana --timeout=300s -n monitoring
```

### 2. Application Monitoring

```bash
# Deploy ServiceMonitor for application metrics
kubectl apply -f monitoring/kubernetes/servicemonitor.yaml

# Deploy PrometheusRule for alerting
kubectl apply -f monitoring/kubernetes/prometheusrule.yaml

# Import Grafana dashboards
./scripts/import-grafana-dashboards.sh
```

### 3. Log Aggregation

```bash
# Install Fluent Bit for log collection
helm repo add fluent https://fluent.github.io/helm-charts
helm install fluent-bit fluent/fluent-bit \
  --namespace logging \
  --create-namespace \
  --set config.outputs='[OUTPUT]\n    Name es\n    Match *\n    Host elasticsearch\n    Port 9200\n    Index codesign-logs'

# Install Elasticsearch and Kibana (optional)
helm repo add elastic https://helm.elastic.co
helm install elasticsearch elastic/elasticsearch \
  --namespace logging \
  --set replicas=1 \
  --set minimumMasterNodes=1

helm install kibana elastic/kibana \
  --namespace logging
```

## Backup Configuration

### 1. Database Backups

```bash
# Create backup CronJob
kubectl apply -f - <<EOF
apiVersion: batch/v1
kind: CronJob
metadata:
  name: postgres-backup
  namespace: codesign-production
spec:
  schedule: "0 2 * * *"  # Daily at 2 AM
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: backup
            image: postgres:15-alpine
            command:
            - /bin/sh
            - -c
            - |
              timestamp=\$(date +%Y%m%d_%H%M%S)
              pg_dump -h postgres -U codesign -d codesign_db > /backup/postgres_backup_\$timestamp.sql
              gzip /backup/postgres_backup_\$timestamp.sql
              # Clean up old backups (keep 30 days)
              find /backup -name "postgres_backup_*.sql.gz" -mtime +30 -delete
              echo "Backup completed: postgres_backup_\$timestamp.sql.gz"
            env:
            - name: PGPASSWORD
              valueFrom:
                secretKeyRef:
                  name: codesign-secrets
                  key: postgres-password
            volumeMounts:
            - name: backup-storage
              mountPath: /backup
          volumes:
          - name: backup-storage
            persistentVolumeClaim:
              claimName: backup-pvc
          restartPolicy: OnFailure
  successfulJobsHistoryLimit: 3
  failedJobsHistoryLimit: 1
EOF

# Verify backup job
kubectl get cronjob postgres-backup -n codesign-production
```

### 2. Application Data Backups

```bash
# Create compliance data backup
kubectl apply -f - <<EOF
apiVersion: batch/v1
kind: CronJob
metadata:
  name: compliance-backup
  namespace: codesign-production
spec:
  schedule: "0 */6 * * *"  # Every 6 hours
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: backup
            image: your-registry.com/codesign-playground:latest
            command:
            - /bin/sh
            - -c
            - |
              timestamp=\$(date +%Y%m%d_%H%M%S)
              python -c "
              from backend.codesign_playground.utils.compliance import get_compliance_manager
              import pickle
              manager = get_compliance_manager()
              with open('/backup/compliance_backup_\$timestamp.pkl', 'wb') as f:
                  pickle.dump({
                      'audit_logs': manager._audit_logs,
                      'processing_records': manager._processing_records,
                      'user_consents': manager._user_consents
                  }, f)
              print('Compliance backup completed')
              "
            env:
            - name: PYTHONPATH
              value: "/app/backend"
            volumeMounts:
            - name: backup-storage
              mountPath: /backup
            - name: compliance-storage
              mountPath: /app/compliance
          volumes:
          - name: backup-storage
            persistentVolumeClaim:
              claimName: backup-pvc
          - name: compliance-storage
            persistentVolumeClaim:
              claimName: compliance-pvc
          restartPolicy: OnFailure
EOF
```

### 3. Offsite Backup (Optional)

```bash
# Configure S3 backup if enabled
if [ "$S3_BACKUP_ENABLED" = "true" ]; then
  kubectl create secret generic s3-credentials \
    --from-literal=access-key-id="$AWS_ACCESS_KEY_ID" \
    --from-literal=secret-access-key="$AWS_SECRET_ACCESS_KEY" \
    --namespace=codesign-production

  # Deploy S3 sync job
  kubectl apply -f deployment/kubernetes/s3-backup.yaml
fi
```

## Testing and Validation

### 1. Health Checks

```bash
# Run comprehensive health check
./scripts/health-check.sh --comprehensive

# Test API endpoints
curl -k https://your-domain.com/health
curl -k https://your-domain.com/health/ready
curl -k https://your-domain.com/health/startup

# Test authentication
curl -X POST https://your-domain.com/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"test","password":"test123"}'
```

### 2. Load Testing

```bash
# Install load testing tools
pip install locust

# Run load test
locust -f tests/performance/locustfile.py \
  --host=https://your-domain.com \
  --users=50 \
  --spawn-rate=5 \
  --run-time=5m \
  --headless

# Monitor during load test
kubectl top pods -n codesign-production
kubectl get hpa -n codesign-production
```

### 3. Security Testing

```bash
# Run security scan
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
  aquasec/trivy image your-registry.com/codesign-playground:latest

# Test SSL configuration
testssl.sh https://your-domain.com

# Verify security headers
curl -I https://your-domain.com
```

### 4. Compliance Testing

```bash
# Test GDPR compliance features
kubectl exec -it deployment/codesign-api -n codesign-production -- python -c "
from backend.codesign_playground.utils.compliance import get_compliance_manager
manager = get_compliance_manager()

# Test data subject request
response = manager.handle_data_subject_request(
    user_id='test_user_gdpr',
    request_type='access'
)
print('GDPR access request test:', response['status'])

# Test data processing recording
success = manager.record_data_processing(
    user_id='test_user_processing',
    data_category=DataCategory.PERSONAL_IDENTIFYING,
    processing_purpose='testing',
    legal_basis='consent'
)
print('Data processing recording test:', success)
"
```

### 5. Disaster Recovery Testing

```bash
# Test backup restoration
./scripts/test-backup-restore.sh --dry-run

# Test failover scenarios
./scripts/test-failover.sh --component=database

# Verify monitoring alerts
./scripts/test-alerting.sh
```

## Go-Live Checklist

### Pre-Go-Live

- [ ] All components deployed and healthy
- [ ] SSL certificates valid and configured
- [ ] Monitoring and alerting operational
- [ ] Backup jobs configured and tested
- [ ] Security scans passed
- [ ] Load testing completed successfully
- [ ] Compliance features verified
- [ ] DNS records updated
- [ ] Runbooks and documentation updated
- [ ] Team training completed

### Go-Live Steps

```bash
# 1. Final deployment verification
kubectl get all -n codesign-production
./scripts/health-check.sh --comprehensive

# 2. Enable production traffic
kubectl patch ingress codesign-ingress -n codesign-production \
  -p '{"metadata":{"annotations":{"nginx.ingress.kubernetes.io/server-snippet":"set $maintenance off;"}}}'

# 3. Monitor initial traffic
kubectl logs -f deployment/codesign-api -n codesign-production

# 4. Verify metrics collection
curl -s http://prometheus:9090/api/v1/query?query=up{job="codesign-api"}

# 5. Test critical user journeys
./scripts/test-critical-user-journeys.sh --production

# 6. Update status page
curl -X POST https://api.status.codesign.example.com/incidents \
  -H "Authorization: Bearer $STATUS_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"title":"Production Deployment Complete","status":"operational"}'
```

### Post-Go-Live Monitoring

```bash
# Monitor for first 4 hours
./scripts/monitor-go-live.sh --duration=4h

# Check error rates
kubectl exec -it deployment/prometheus -n monitoring -- \
  promtool query instant 'rate(http_requests_total{status=~"5.."}[5m])'

# Verify compliance logging
kubectl exec -it deployment/codesign-api -n codesign-production -- python -c "
from backend.codesign_playground.utils.compliance import get_compliance_manager
manager = get_compliance_manager()
print(f'Audit logs created: {len(manager._audit_logs)}')
"
```

## Post-Deployment

### 1. Performance Optimization

```bash
# Review performance metrics
./scripts/analyze-performance-metrics.sh --timeframe=24h

# Optimize database queries
kubectl exec -it deployment/postgres -n codesign-production -- psql -U codesign -d codesign_db -c "
SELECT query, mean_time, calls 
FROM pg_stat_statements 
WHERE mean_time > 100 
ORDER BY mean_time DESC 
LIMIT 10;
"

# Tune application settings based on load
kubectl patch deployment codesign-api -n codesign-production \
  -p '{"spec":{"template":{"spec":{"containers":[{"name":"codesign-api","env":[{"name":"MAX_WORKERS","value":"12"}]}]}}}}'
```

### 2. Security Hardening

```bash
# Enable additional security features
kubectl patch deployment codesign-api -n codesign-production \
  -p '{"spec":{"template":{"spec":{"containers":[{"name":"codesign-api","env":[{"name":"SECURITY_AUDIT_ENABLED","value":"true"}]}]}}}}'

# Configure Web Application Firewall rules
./scripts/configure-waf-rules.sh

# Set up vulnerability scanning schedule
kubectl apply -f security/vulnerability-scan-cronjob.yaml
```

### 3. Operational Procedures

```bash
# Set up automated health checks
kubectl apply -f monitoring/health-check-cronjob.yaml

# Configure log rotation
kubectl apply -f logging/log-rotation-config.yaml

# Set up compliance reporting
kubectl apply -f compliance/reporting-cronjob.yaml
```

### 4. Documentation Updates

```bash
# Update operational runbooks
./scripts/update-runbooks.sh --deployment-date=$(date)

# Generate system documentation
./scripts/generate-system-docs.sh --environment=production

# Create handover documentation
./scripts/create-handover-docs.sh --for-team=operations
```

### 5. Team Training

- Conduct operational training sessions
- Review incident response procedures
- Practice disaster recovery scenarios
- Update on-call rotation
- Share monitoring dashboards and alerts

---

## Troubleshooting

### Common Deployment Issues

#### Pod Stuck in Pending
```bash
# Check resource constraints
kubectl describe pod <pod-name> -n codesign-production
kubectl top nodes

# Check storage
kubectl get pv
kubectl get pvc -n codesign-production
```

#### Image Pull Errors
```bash
# Check image availability
docker pull your-registry.com/codesign-playground:latest

# Verify registry credentials
kubectl get secrets -n codesign-production
kubectl describe secret regcred -n codesign-production
```

#### Database Connection Issues
```bash
# Check PostgreSQL status
kubectl logs deployment/postgres -n codesign-production

# Test connectivity
kubectl exec -it deployment/codesign-api -n codesign-production -- \
  pg_isready -h postgres -p 5432 -U codesign
```

#### SSL Certificate Issues
```bash
# Check certificate status
kubectl describe certificate codesign-tls -n codesign-production

# Check cert-manager logs
kubectl logs -n cert-manager deployment/cert-manager

# Manual certificate request
kubectl apply -f - <<EOF
apiVersion: cert-manager.io/v1
kind: Certificate
metadata:
  name: codesign-tls-manual
  namespace: codesign-production
spec:
  secretName: codesign-tls
  issuerRef:
    name: letsencrypt-prod
    kind: ClusterIssuer
  dnsNames:
  - your-domain.com
EOF
```

---

## Appendix

### Environment Variables Reference

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| ENV | Environment name | production | Yes |
| SECRET_KEY | Application secret key | - | Yes |
| POSTGRES_PASSWORD | Database password | - | Yes |
| REDIS_PASSWORD | Cache password | - | Yes |
| COMPLIANCE_REGION | Compliance region | global | No |
| BACKUP_ENABLED | Enable backups | true | No |
| S3_BACKUP_ENABLED | Enable S3 backups | false | No |

### Resource Requirements

| Component | Min CPU | Min Memory | Min Storage |
|-----------|---------|------------|-------------|
| API Pods (3x) | 3 cores | 3 GB | - |
| Worker Pods (2x) | 1 core | 1 GB | - |
| PostgreSQL | 1 core | 2 GB | 50 GB |
| Redis | 0.5 cores | 1 GB | 10 GB |
| Monitoring | 2 cores | 4 GB | 100 GB |

### Network Ports

| Service | Port | Protocol | Description |
|---------|------|----------|-------------|
| API | 8000 | HTTP | Main application |
| PostgreSQL | 5432 | TCP | Database |
| Redis | 6379 | TCP | Cache |
| Prometheus | 9090 | HTTP | Metrics |
| Grafana | 3000 | HTTP | Dashboards |

---

*Last Updated: 2024-08-16*  
*Version: 2.0*  
*Document Owner: Platform Engineering Team*