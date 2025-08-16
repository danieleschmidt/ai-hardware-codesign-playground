# AI Hardware Co-Design Playground - Complete Operations & Deployment Guide

## Table of Contents

1. [Deployment Overview](#deployment-overview)
2. [Environment Setup](#environment-setup)
3. [Local Development Deployment](#local-development-deployment)
4. [Staging Environment](#staging-environment)
5. [Production Deployment](#production-deployment)
6. [Container Orchestration](#container-orchestration)
7. [Infrastructure as Code](#infrastructure-as-code)
8. [Monitoring and Observability](#monitoring-and-observability)
9. [Backup and Disaster Recovery](#backup-and-disaster-recovery)
10. [Security Operations](#security-operations)
11. [Performance Tuning](#performance-tuning)
12. [Troubleshooting](#troubleshooting)

## Deployment Overview

The AI Hardware Co-Design Playground supports multiple deployment configurations, from local development to enterprise-scale production environments.

### Deployment Architectures

```mermaid
graph TB
    subgraph "Development"
        DevLocal[Local Development]
        DevDocker[Docker Compose]
    end
    
    subgraph "Staging"
        StagingK8s[Kubernetes Staging]
        StagingCloud[Cloud Staging]
    end
    
    subgraph "Production"
        ProdSingle[Single Region]
        ProdMulti[Multi-Region]
        ProdEdge[Edge Deployment]
    end
    
    DevLocal --> StagingK8s
    DevDocker --> StagingCloud
    StagingK8s --> ProdSingle
    StagingCloud --> ProdMulti
    ProdSingle --> ProdEdge
```

### Supported Platforms

| Platform | Environment | Complexity | Scalability |
|----------|-------------|------------|-------------|
| **Docker Compose** | Development/Testing | Low | Limited |
| **Kubernetes** | Staging/Production | Medium | High |
| **AWS EKS** | Production | High | Very High |
| **Azure AKS** | Production | High | Very High |
| **Google GKE** | Production | High | Very High |
| **Edge Computing** | Distributed | Very High | Horizontal |

## Environment Setup

### Prerequisites

#### System Requirements

**Development Environment:**
```bash
# Minimum requirements
CPU: 4 cores
RAM: 8 GB
Storage: 50 GB
OS: Linux, macOS, Windows (WSL2)

# Recommended requirements
CPU: 8+ cores
RAM: 16+ GB
Storage: 100+ GB SSD
Network: Broadband internet
```

**Production Environment:**
```bash
# Minimum production requirements
CPU: 16 cores per node
RAM: 32 GB per node
Storage: 200 GB SSD per node
Network: 10 Gbps

# Recommended production requirements
CPU: 32+ cores per node
RAM: 64+ GB per node
Storage: 500+ GB NVMe per node
Network: 25+ Gbps
```

#### Software Dependencies

```bash
# Core tools
docker >= 20.10
docker-compose >= 2.0
kubectl >= 1.25
helm >= 3.8
terraform >= 1.3 (for IaC)
ansible >= 4.0 (for configuration)

# Optional tools
k9s              # Kubernetes management
stern            # Log streaming
kubectx/kubens   # Context switching
```

### Environment Variables

#### Core Configuration

```bash
# Application settings
export CODESIGN_ENV=production
export CODESIGN_VERSION=0.1.0
export CODESIGN_LOG_LEVEL=INFO

# Database configuration
export DATABASE_URL=postgresql://user:pass@localhost:5432/codesign
export DATABASE_POOL_SIZE=20
export DATABASE_POOL_TIMEOUT=30

# Cache configuration
export REDIS_URL=redis://localhost:6379/0
export CACHE_TTL_SECONDS=3600
export CACHE_MAX_SIZE_MB=1024

# API configuration
export API_HOST=0.0.0.0
export API_PORT=8000
export API_WORKERS=4
export API_WORKER_CLASS=uvicorn.workers.UvicornWorker

# Security configuration
export SECRET_KEY=your-secret-key-here
export ALLOWED_HOSTS=localhost,127.0.0.1
export RATE_LIMIT_PER_MINUTE=60
export RATE_LIMIT_PER_HOUR=1000

# Feature flags
export ENABLE_AUTO_SCALING=true
export ENABLE_SECURITY_VALIDATION=true
export ENABLE_PERFORMANCE_MONITORING=true
export ENABLE_DISTRIBUTED_TRACING=true

# Cloud provider settings (AWS example)
export AWS_REGION=us-west-2
export AWS_ACCESS_KEY_ID=your-access-key
export AWS_SECRET_ACCESS_KEY=your-secret-key
export AWS_S3_BUCKET=codesign-artifacts

# Monitoring configuration
export PROMETHEUS_URL=http://prometheus:9090
export GRAFANA_URL=http://grafana:3000
export JAEGER_ENDPOINT=http://jaeger:14268/api/traces
```

#### Environment-Specific Files

**.env.development**:
```bash
# Development environment
CODESIGN_ENV=development
DEBUG=true
LOG_LEVEL=DEBUG
DATABASE_URL=postgresql://dev:dev@localhost:5432/codesign_dev
REDIS_URL=redis://localhost:6379/0
ENABLE_AUTO_SCALING=false
RATE_LIMIT_PER_MINUTE=1000
```

**.env.staging**:
```bash
# Staging environment
CODESIGN_ENV=staging
DEBUG=false
LOG_LEVEL=INFO
DATABASE_URL=postgresql://staging:staging@db-staging:5432/codesign_staging
REDIS_URL=redis://redis-staging:6379/0
ENABLE_AUTO_SCALING=true
RATE_LIMIT_PER_MINUTE=100
```

**.env.production**:
```bash
# Production environment
CODESIGN_ENV=production
DEBUG=false
LOG_LEVEL=WARNING
DATABASE_URL=postgresql://prod:securepass@db-prod:5432/codesign_prod
REDIS_URL=redis://redis-prod:6379/0
ENABLE_AUTO_SCALING=true
RATE_LIMIT_PER_MINUTE=60
SSL_ENABLED=true
```

## Local Development Deployment

### Docker Compose Setup

#### Basic Development Stack

**docker-compose.dev.yml**:
```yaml
version: '3.8'

services:
  app:
    build:
      context: .
      dockerfile: Dockerfile.dev
    ports:
      - "8000:8000"
      - "5678:5678"  # Debugger port
    volumes:
      - .:/app
      - /app/venv  # Exclude virtual environment
    environment:
      - CODESIGN_ENV=development
      - DATABASE_URL=postgresql://dev:dev@db:5432/codesign_dev
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - db
      - redis
    command: python -m uvicorn codesign_playground.server:app --host 0.0.0.0 --port 8000 --reload
    
  db:
    image: postgres:15-alpine
    environment:
      - POSTGRES_DB=codesign_dev
      - POSTGRES_USER=dev
      - POSTGRES_PASSWORD=dev
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./scripts/init_db.sql:/docker-entrypoint-initdb.d/init.sql
    
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes
    
  worker:
    build:
      context: .
      dockerfile: Dockerfile.dev
    volumes:
      - .:/app
    environment:
      - CODESIGN_ENV=development
      - DATABASE_URL=postgresql://dev:dev@db:5432/codesign_dev
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - db
      - redis
    command: python -m codesign_playground.worker

volumes:
  postgres_data:
  redis_data:
```

#### Development with Monitoring

**docker-compose.dev.monitoring.yml**:
```yaml
version: '3.8'

services:
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--web.enable-lifecycle'
      
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./monitoring/grafana/datasources:/etc/grafana/provisioning/datasources
      
  jaeger:
    image: jaegertracing/all-in-one:latest
    ports:
      - "16686:16686"
      - "14268:14268"
    environment:
      - COLLECTOR_OTLP_ENABLED=true

volumes:
  prometheus_data:
  grafana_data:
```

### Quick Start Commands

```bash
# 1. Clone and setup
git clone https://github.com/your-org/ai-hardware-codesign-playground.git
cd ai-hardware-codesign-playground

# 2. Environment setup
cp .env.example .env.development
# Edit .env.development with your settings

# 3. Start development environment
docker-compose -f docker-compose.dev.yml up -d

# 4. Initialize database
docker-compose -f docker-compose.dev.yml exec app python scripts/init_db.py

# 5. Run tests
docker-compose -f docker-compose.dev.yml exec app pytest

# 6. Access services
# API: http://localhost:8000
# API Docs: http://localhost:8000/docs
# Database: localhost:5432
# Redis: localhost:6379

# 7. View logs
docker-compose -f docker-compose.dev.yml logs -f app

# 8. Stop environment
docker-compose -f docker-compose.dev.yml down
```

### Development Workflow

```bash
# Hot reload development
docker-compose -f docker-compose.dev.yml up app

# Run specific tests
docker-compose -f docker-compose.dev.yml exec app pytest tests/unit/

# Database migrations
docker-compose -f docker-compose.dev.yml exec app alembic upgrade head

# Code formatting
docker-compose -f docker-compose.dev.yml exec app black .
docker-compose -f docker-compose.dev.yml exec app isort .

# Linting
docker-compose -f docker-compose.dev.yml exec app flake8
docker-compose -f docker-compose.dev.yml exec app mypy backend/

# Security scan
docker-compose -f docker-compose.dev.yml exec app bandit -r backend/
```

## Staging Environment

### Kubernetes Staging Deployment

#### Namespace Configuration

**k8s/staging/namespace.yaml**:
```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: codesign-staging
  labels:
    environment: staging
    app: codesign-playground
---
apiVersion: v1
kind: ResourceQuota
metadata:
  name: codesign-staging-quota
  namespace: codesign-staging
spec:
  hard:
    requests.cpu: "8"
    requests.memory: 16Gi
    limits.cpu: "16"
    limits.memory: 32Gi
    persistentvolumeclaims: "10"
```

#### Application Deployment

**k8s/staging/deployment.yaml**:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: codesign-app
  namespace: codesign-staging
  labels:
    app: codesign-app
    environment: staging
spec:
  replicas: 2
  selector:
    matchLabels:
      app: codesign-app
  template:
    metadata:
      labels:
        app: codesign-app
        environment: staging
    spec:
      containers:
      - name: app
        image: codesign-playground:staging
        ports:
        - containerPort: 8000
        env:
        - name: CODESIGN_ENV
          value: "staging"
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: codesign-secrets
              key: database-url
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: codesign-secrets
              key: redis-url
        resources:
          requests:
            cpu: 500m
            memory: 1Gi
          limits:
            cpu: 2
            memory: 4Gi
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
        volumeMounts:
        - name: config
          mountPath: /app/config
          readOnly: true
      volumes:
      - name: config
        configMap:
          name: codesign-config
```

#### Service Configuration

**k8s/staging/service.yaml**:
```yaml
apiVersion: v1
kind: Service
metadata:
  name: codesign-service
  namespace: codesign-staging
  labels:
    app: codesign-app
spec:
  type: ClusterIP
  ports:
  - port: 80
    targetPort: 8000
    protocol: TCP
  selector:
    app: codesign-app
---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: codesign-ingress
  namespace: codesign-staging
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-staging
    nginx.ingress.kubernetes.io/rate-limit: "100"
spec:
  tls:
  - hosts:
    - staging.codesign-playground.com
    secretName: codesign-staging-tls
  rules:
  - host: staging.codesign-playground.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: codesign-service
            port:
              number: 80
```

### Staging Deployment Commands

```bash
# 1. Build and push staging image
docker build -t codesign-playground:staging .
docker tag codesign-playground:staging your-registry/codesign-playground:staging
docker push your-registry/codesign-playground:staging

# 2. Create namespace and secrets
kubectl apply -f k8s/staging/namespace.yaml
kubectl create secret generic codesign-secrets \
  --namespace=codesign-staging \
  --from-literal=database-url="postgresql://..." \
  --from-literal=redis-url="redis://..."

# 3. Deploy application
kubectl apply -f k8s/staging/
kubectl rollout status deployment/codesign-app -n codesign-staging

# 4. Check deployment
kubectl get pods -n codesign-staging
kubectl logs -f deployment/codesign-app -n codesign-staging

# 5. Test deployment
kubectl port-forward svc/codesign-service 8080:80 -n codesign-staging
curl http://localhost:8080/health
```

## Production Deployment

### Production Architecture

```mermaid
graph TB
    subgraph "Load Balancing"
        LB[Load Balancer]
        WAF[Web Application Firewall]
    end
    
    subgraph "Application Tier"
        APP1[App Pod 1]
        APP2[App Pod 2]
        APP3[App Pod N]
    end
    
    subgraph "Data Tier"
        PG_PRIMARY[PostgreSQL Primary]
        PG_REPLICA[PostgreSQL Replica]
        REDIS_PRIMARY[Redis Primary]
        REDIS_REPLICA[Redis Replica]
    end
    
    subgraph "Monitoring"
        PROMETHEUS[Prometheus]
        GRAFANA[Grafana]
        JAEGER[Jaeger]
        ALERTS[AlertManager]
    end
    
    LB --> WAF
    WAF --> APP1
    WAF --> APP2
    WAF --> APP3
    
    APP1 --> PG_PRIMARY
    APP2 --> PG_PRIMARY
    APP3 --> PG_PRIMARY
    
    PG_PRIMARY --> PG_REPLICA
    
    APP1 --> REDIS_PRIMARY
    APP2 --> REDIS_PRIMARY
    APP3 --> REDIS_PRIMARY
    
    REDIS_PRIMARY --> REDIS_REPLICA
    
    PROMETHEUS --> APP1
    PROMETHEUS --> APP2
    PROMETHEUS --> APP3
    GRAFANA --> PROMETHEUS
    ALERTS --> PROMETHEUS
```

### Production Kubernetes Configuration

#### High Availability Deployment

**k8s/production/deployment.yaml**:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: codesign-app
  namespace: codesign-production
  labels:
    app: codesign-app
    environment: production
spec:
  replicas: 6
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxUnavailable: 1
      maxSurge: 2
  selector:
    matchLabels:
      app: codesign-app
  template:
    metadata:
      labels:
        app: codesign-app
        environment: production
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8000"
        prometheus.io/path: "/metrics"
    spec:
      affinity:
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 100
            podAffinityTerm:
              labelSelector:
                matchExpressions:
                - key: app
                  operator: In
                  values:
                  - codesign-app
              topologyKey: kubernetes.io/hostname
      containers:
      - name: app
        image: codesign-playground:v0.1.0
        ports:
        - containerPort: 8000
          name: http
        env:
        - name: CODESIGN_ENV
          value: "production"
        - name: POD_NAME
          valueFrom:
            fieldRef:
              fieldPath: metadata.name
        - name: POD_NAMESPACE
          valueFrom:
            fieldRef:
              fieldPath: metadata.namespace
        envFrom:
        - secretRef:
            name: codesign-secrets
        - configMapRef:
            name: codesign-config
        resources:
          requests:
            cpu: 1
            memory: 2Gi
          limits:
            cpu: 4
            memory: 8Gi
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 30
          timeoutSeconds: 10
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
          timeoutSeconds: 5
          failureThreshold: 3
        securityContext:
          runAsNonRoot: true
          runAsUser: 1000
          readOnlyRootFilesystem: true
          allowPrivilegeEscalation: false
        volumeMounts:
        - name: tmp
          mountPath: /tmp
        - name: cache
          mountPath: /app/cache
      volumes:
      - name: tmp
        emptyDir: {}
      - name: cache
        emptyDir:
          sizeLimit: 1Gi
      securityContext:
        fsGroup: 1000
```

#### Horizontal Pod Autoscaler

**k8s/production/hpa.yaml**:
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: codesign-app-hpa
  namespace: codesign-production
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: codesign-app
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  - type: Pods
    pods:
      metric:
        name: http_requests_per_second
      target:
        type: AverageValue
        averageValue: "100"
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 100
        periodSeconds: 15
      - type: Pods
        value: 4
        periodSeconds: 15
      selectPolicy: Max
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
      selectPolicy: Min
```

#### Production Service with Load Balancer

**k8s/production/service.yaml**:
```yaml
apiVersion: v1
kind: Service
metadata:
  name: codesign-service
  namespace: codesign-production
  labels:
    app: codesign-app
  annotations:
    service.beta.kubernetes.io/aws-load-balancer-type: nlb
    service.beta.kubernetes.io/aws-load-balancer-cross-zone-load-balancing-enabled: "true"
    service.beta.kubernetes.io/aws-load-balancer-backend-protocol: tcp
spec:
  type: LoadBalancer
  ports:
  - port: 80
    targetPort: 8000
    protocol: TCP
    name: http
  - port: 443
    targetPort: 8000
    protocol: TCP
    name: https
  selector:
    app: codesign-app
  sessionAffinity: None
```

### Database Configuration

#### PostgreSQL High Availability

**k8s/production/postgresql.yaml**:
```yaml
apiVersion: postgresql.cnpg.io/v1
kind: Cluster
metadata:
  name: postgres-cluster
  namespace: codesign-production
spec:
  instances: 3
  
  postgresql:
    parameters:
      max_connections: "200"
      shared_buffers: "256MB"
      effective_cache_size: "1GB"
      maintenance_work_mem: "64MB"
      checkpoint_completion_target: "0.9"
      wal_buffers: "16MB"
      default_statistics_target: "100"
      random_page_cost: "1.1"
      effective_io_concurrency: "200"
    
  bootstrap:
    initdb:
      database: codesign_prod
      owner: codesign
      secret:
        name: postgres-credentials
        
  storage:
    size: 500Gi
    storageClass: fast-ssd
    
  monitoring:
    enabled: true
    
  backup:
    retentionPolicy: "30d"
    barmanObjectStore:
      destinationPath: "s3://your-backup-bucket/postgres"
      s3Credentials:
        accessKeyId:
          name: backup-credentials
          key: ACCESS_KEY_ID
        secretAccessKey:
          name: backup-credentials
          key: SECRET_ACCESS_KEY
      wal:
        retention: "7d"
      data:
        retention: "30d"
```

#### Redis Cluster

**k8s/production/redis.yaml**:
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: redis-config
  namespace: codesign-production
data:
  redis.conf: |
    maxmemory 2gb
    maxmemory-policy allkeys-lru
    save 900 1
    save 300 10
    save 60 10000
    appendonly yes
    appendfsync everysec
---
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: redis
  namespace: codesign-production
spec:
  serviceName: redis-service
  replicas: 3
  selector:
    matchLabels:
      app: redis
  template:
    metadata:
      labels:
        app: redis
    spec:
      containers:
      - name: redis
        image: redis:7-alpine
        ports:
        - containerPort: 6379
        command:
        - redis-server
        - /etc/redis/redis.conf
        volumeMounts:
        - name: redis-data
          mountPath: /data
        - name: redis-config
          mountPath: /etc/redis
        resources:
          requests:
            cpu: 100m
            memory: 512Mi
          limits:
            cpu: 500m
            memory: 2Gi
      volumes:
      - name: redis-config
        configMap:
          name: redis-config
  volumeClaimTemplates:
  - metadata:
      name: redis-data
    spec:
      accessModes: ["ReadWriteOnce"]
      storageClassName: "fast-ssd"
      resources:
        requests:
          storage: 20Gi
```

### Production Deployment Commands

```bash
# 1. Create production namespace
kubectl create namespace codesign-production

# 2. Create secrets
kubectl create secret generic codesign-secrets \
  --namespace=codesign-production \
  --from-literal=DATABASE_URL="postgresql://..." \
  --from-literal=REDIS_URL="redis://..." \
  --from-literal=SECRET_KEY="your-secret-key"

# 3. Apply configurations
kubectl apply -f k8s/production/

# 4. Wait for deployment
kubectl rollout status deployment/codesign-app -n codesign-production

# 5. Check pods
kubectl get pods -n codesign-production

# 6. Check HPA status
kubectl get hpa -n codesign-production

# 7. Get external IP
kubectl get svc codesign-service -n codesign-production

# 8. Test deployment
curl http://<EXTERNAL-IP>/health
```

## Container Orchestration

### Helm Charts

#### Chart Structure

```
charts/codesign-playground/
├── Chart.yaml
├── values.yaml
├── values-staging.yaml
├── values-production.yaml
├── templates/
│   ├── deployment.yaml
│   ├── service.yaml
│   ├── ingress.yaml
│   ├── configmap.yaml
│   ├── secret.yaml
│   ├── hpa.yaml
│   ├── pdb.yaml
│   └── servicemonitor.yaml
├── charts/
│   ├── postgresql/
│   └── redis/
└── crds/
```

#### Chart.yaml

```yaml
apiVersion: v2
name: codesign-playground
description: AI Hardware Co-Design Playground Helm Chart
type: application
version: 0.1.0
appVersion: "0.1.0"
keywords:
  - ai
  - hardware
  - machine-learning
  - accelerator
home: https://github.com/your-org/ai-hardware-codesign-playground
sources:
  - https://github.com/your-org/ai-hardware-codesign-playground
maintainers:
  - name: Your Team
    email: team@your-org.com
dependencies:
  - name: postgresql
    version: 12.x.x
    repository: https://charts.bitnami.com/bitnami
    condition: postgresql.enabled
  - name: redis
    version: 17.x.x
    repository: https://charts.bitnami.com/bitnami
    condition: redis.enabled
```

#### values.yaml

```yaml
# Default values for codesign-playground
replicaCount: 3

image:
  repository: codesign-playground
  pullPolicy: IfNotPresent
  tag: ""

imagePullSecrets: []
nameOverride: ""
fullnameOverride: ""

serviceAccount:
  create: true
  annotations: {}
  name: ""

podAnnotations: {}

podSecurityContext:
  fsGroup: 1000

securityContext:
  runAsNonRoot: true
  runAsUser: 1000
  readOnlyRootFilesystem: true
  allowPrivilegeEscalation: false

service:
  type: ClusterIP
  port: 80
  targetPort: 8000

ingress:
  enabled: false
  className: ""
  annotations: {}
  hosts:
    - host: chart-example.local
      paths:
        - path: /
          pathType: ImplementationSpecific
  tls: []

resources:
  limits:
    cpu: 4
    memory: 8Gi
  requests:
    cpu: 1
    memory: 2Gi

autoscaling:
  enabled: true
  minReplicas: 3
  maxReplicas: 20
  targetCPUUtilizationPercentage: 70
  targetMemoryUtilizationPercentage: 80

nodeSelector: {}

tolerations: []

affinity: {}

# Application configuration
config:
  environment: production
  logLevel: INFO
  enableAutoScaling: true
  enableSecurityValidation: true

# Database configuration
postgresql:
  enabled: true
  auth:
    postgresPassword: "changeme"
    username: "codesign"
    password: "changeme"
    database: "codesign"
  primary:
    persistence:
      enabled: true
      size: 500Gi
      storageClass: "fast-ssd"

# Redis configuration
redis:
  enabled: true
  auth:
    enabled: true
    password: "changeme"
  master:
    persistence:
      enabled: true
      size: 20Gi
      storageClass: "fast-ssd"

# Monitoring
monitoring:
  enabled: true
  serviceMonitor:
    enabled: true
    namespace: monitoring
    interval: 30s
    scrapeTimeout: 10s
```

#### Helm Deployment Commands

```bash
# 1. Add Helm repositories
helm repo add bitnami https://charts.bitnami.com/bitnami
helm repo update

# 2. Install dependencies
helm dependency build charts/codesign-playground/

# 3. Deploy to staging
helm upgrade --install codesign-staging charts/codesign-playground/ \
  --namespace codesign-staging \
  --create-namespace \
  --values charts/codesign-playground/values-staging.yaml

# 4. Deploy to production
helm upgrade --install codesign-production charts/codesign-playground/ \
  --namespace codesign-production \
  --create-namespace \
  --values charts/codesign-playground/values-production.yaml

# 5. Check deployment status
helm status codesign-production -n codesign-production

# 6. Rollback if needed
helm rollback codesign-production 1 -n codesign-production

# 7. Uninstall
helm uninstall codesign-production -n codesign-production
```

## Infrastructure as Code

### Terraform Configuration

#### AWS Infrastructure

**terraform/aws/main.tf**:
```hcl
terraform {
  required_version = ">= 1.3"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.16"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.8"
    }
  }
}

provider "aws" {
  region = var.aws_region
  
  default_tags {
    tags = {
      Project     = "codesign-playground"
      Environment = var.environment
      ManagedBy   = "terraform"
    }
  }
}

# VPC Configuration
module "vpc" {
  source = "terraform-aws-modules/vpc/aws"
  version = "~> 3.0"

  name = "${var.project_name}-${var.environment}"
  cidr = var.vpc_cidr

  azs             = var.availability_zones
  private_subnets = var.private_subnet_cidrs
  public_subnets  = var.public_subnet_cidrs

  enable_nat_gateway = true
  enable_vpn_gateway = false
  enable_dns_hostnames = true
  enable_dns_support = true

  tags = {
    Environment = var.environment
  }
}

# EKS Cluster
module "eks" {
  source = "terraform-aws-modules/eks/aws"
  version = "~> 19.0"

  cluster_name    = "${var.project_name}-${var.environment}"
  cluster_version = var.kubernetes_version

  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets

  # EKS Managed Node Groups
  eks_managed_node_groups = {
    main = {
      min_size     = var.node_group_min_size
      max_size     = var.node_group_max_size
      desired_size = var.node_group_desired_size

      instance_types = var.node_instance_types
      capacity_type  = "ON_DEMAND"

      k8s_labels = {
        Environment = var.environment
        NodeGroup   = "main"
      }

      update_config = {
        max_unavailable_percentage = 25
      }
    }
  }

  # aws-auth configmap
  manage_aws_auth_configmap = true

  aws_auth_roles = [
    {
      rolearn  = aws_iam_role.eks_admin.arn
      username = "eks-admin"
      groups   = ["system:masters"]
    },
  ]

  tags = {
    Environment = var.environment
  }
}

# RDS Database
resource "aws_db_subnet_group" "main" {
  name       = "${var.project_name}-${var.environment}"
  subnet_ids = module.vpc.private_subnets

  tags = {
    Name = "${var.project_name}-${var.environment}"
  }
}

resource "aws_db_instance" "main" {
  identifier = "${var.project_name}-${var.environment}"

  engine         = "postgres"
  engine_version = "15.3"
  instance_class = var.db_instance_class

  allocated_storage     = var.db_allocated_storage
  max_allocated_storage = var.db_max_allocated_storage
  storage_encrypted     = true

  db_name  = var.db_name
  username = var.db_username
  password = var.db_password

  vpc_security_group_ids = [aws_security_group.rds.id]
  db_subnet_group_name   = aws_db_subnet_group.main.name

  backup_retention_period = var.db_backup_retention_period
  backup_window          = "03:00-04:00"
  maintenance_window     = "Sun:04:00-Sun:05:00"

  skip_final_snapshot = var.environment != "production"

  tags = {
    Name = "${var.project_name}-${var.environment}"
  }
}

# ElastiCache Redis
resource "aws_elasticache_subnet_group" "main" {
  name       = "${var.project_name}-${var.environment}"
  subnet_ids = module.vpc.private_subnets
}

resource "aws_elasticache_replication_group" "main" {
  replication_group_id       = "${var.project_name}-${var.environment}"
  description                = "Redis cluster for ${var.project_name}"

  node_type            = var.redis_node_type
  port                 = 6379
  parameter_group_name = "default.redis7"

  num_cache_clusters = var.redis_num_cache_nodes

  subnet_group_name = aws_elasticache_subnet_group.main.name
  security_group_ids = [aws_security_group.redis.id]

  at_rest_encryption_enabled = true
  transit_encryption_enabled = true

  tags = {
    Name = "${var.project_name}-${var.environment}"
  }
}
```

#### Security Groups

**terraform/aws/security_groups.tf**:
```hcl
# EKS Node Group Security Group Rules
resource "aws_security_group_rule" "eks_node_ingress_cluster" {
  type                     = "ingress"
  from_port                = 0
  to_port                  = 65535
  protocol                 = "tcp"
  source_security_group_id = module.eks.cluster_security_group_id
  security_group_id        = module.eks.node_security_group_id
}

# RDS Security Group
resource "aws_security_group" "rds" {
  name_prefix = "${var.project_name}-${var.environment}-rds"
  vpc_id      = module.vpc.vpc_id

  ingress {
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [module.eks.node_security_group_id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "${var.project_name}-${var.environment}-rds"
  }
}

# Redis Security Group
resource "aws_security_group" "redis" {
  name_prefix = "${var.project_name}-${var.environment}-redis"
  vpc_id      = module.vpc.vpc_id

  ingress {
    from_port       = 6379
    to_port         = 6379
    protocol        = "tcp"
    security_groups = [module.eks.node_security_group_id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "${var.project_name}-${var.environment}-redis"
  }
}
```

#### Terraform Variables

**terraform/aws/variables.tf**:
```hcl
variable "project_name" {
  description = "Name of the project"
  type        = string
  default     = "codesign-playground"
}

variable "environment" {
  description = "Environment name"
  type        = string
}

variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-west-2"
}

variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "availability_zones" {
  description = "Availability zones"
  type        = list(string)
  default     = ["us-west-2a", "us-west-2b", "us-west-2c"]
}

variable "private_subnet_cidrs" {
  description = "CIDR blocks for private subnets"
  type        = list(string)
  default     = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
}

variable "public_subnet_cidrs" {
  description = "CIDR blocks for public subnets"
  type        = list(string)
  default     = ["10.0.101.0/24", "10.0.102.0/24", "10.0.103.0/24"]
}

variable "kubernetes_version" {
  description = "Kubernetes version"
  type        = string
  default     = "1.28"
}

variable "node_instance_types" {
  description = "EC2 instance types for EKS nodes"
  type        = list(string)
  default     = ["m5.xlarge"]
}

variable "node_group_min_size" {
  description = "Minimum number of nodes in EKS node group"
  type        = number
  default     = 3
}

variable "node_group_max_size" {
  description = "Maximum number of nodes in EKS node group"
  type        = number
  default     = 20
}

variable "node_group_desired_size" {
  description = "Desired number of nodes in EKS node group"
  type        = number
  default     = 6
}

variable "db_instance_class" {
  description = "RDS instance class"
  type        = string
  default     = "db.r6g.large"
}

variable "db_allocated_storage" {
  description = "RDS allocated storage in GB"
  type        = number
  default     = 500
}

variable "db_max_allocated_storage" {
  description = "RDS maximum allocated storage in GB"
  type        = number
  default     = 1000
}

variable "db_name" {
  description = "Database name"
  type        = string
  default     = "codesign"
}

variable "db_username" {
  description = "Database username"
  type        = string
  default     = "codesign"
}

variable "db_password" {
  description = "Database password"
  type        = string
  sensitive   = true
}

variable "db_backup_retention_period" {
  description = "Database backup retention period in days"
  type        = number
  default     = 30
}

variable "redis_node_type" {
  description = "Redis node type"
  type        = string
  default     = "cache.r6g.large"
}

variable "redis_num_cache_nodes" {
  description = "Number of Redis cache nodes"
  type        = number
  default     = 2
}
```

#### Terraform Deployment

```bash
# 1. Initialize Terraform
cd terraform/aws
terraform init

# 2. Plan deployment
terraform plan -var-file="environments/production.tfvars"

# 3. Apply infrastructure
terraform apply -var-file="environments/production.tfvars"

# 4. Configure kubectl
aws eks update-kubeconfig --region us-west-2 --name codesign-playground-production

# 5. Verify cluster access
kubectl get nodes

# 6. Install cluster addons
kubectl apply -f k8s/addons/

# 7. Deploy application with Helm
helm upgrade --install codesign-production charts/codesign-playground/ \
  --namespace codesign-production \
  --create-namespace \
  --values charts/codesign-playground/values-production.yaml
```

This comprehensive operations and deployment guide provides detailed instructions for deploying the AI Hardware Co-Design Playground across various environments, from local development to enterprise-scale production deployments. The guide covers container orchestration, infrastructure as code, monitoring, and operational best practices to ensure reliable and scalable deployments.