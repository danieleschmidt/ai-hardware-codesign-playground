# 🚀 Deployment Guide - AI Hardware Co-Design Playground

This comprehensive guide covers deployment strategies for the AI Hardware Co-Design Playground platform, from development to production-ready global deployments.

## 📋 Table of Contents

- [Quick Start](#quick-start)
- [Development Deployment](#development-deployment)
- [Production Deployment](#production-deployment)
- [Multi-Region Deployment](#multi-region-deployment)
- [Security Hardening](#security-hardening)
- [Monitoring & Observability](#monitoring--observability)
- [Compliance & Regulations](#compliance--regulations)
- [Troubleshooting](#troubleshooting)

## 🏃 Quick Start

### Using Docker Compose (Recommended for Development)

```bash
# Clone repository
git clone https://github.com/terragon-labs/ai-hardware-codesign-playground
cd ai-hardware-codesign-playground

# Start development environment
docker-compose up -d

# Access the application
open http://localhost:8000
```

### Using Docker (Production)

```bash
# Build production image
docker build -f docker/production.dockerfile -t codesign-playground:latest .

# Run production container
docker run -d \
  --name codesign-playground \
  -p 8000:8000 \
  -e NODE_ENV=production \
  -e DATABASE_URL=postgresql://user:pass@db:5432/codesign \
  -e REDIS_URL=redis://redis:6379/0 \
  codesign-playground:latest
```

## 🛠️ Development Deployment

### Prerequisites

- Docker 20.10+
- Docker Compose 2.0+
- Python 3.9+
- Node.js 18+
- 8GB RAM minimum
- 20GB disk space

### Local Development Setup

1. **Environment Configuration**

```bash
# Copy environment template
cp .env.example .env

# Edit configuration
vim .env
```

2. **Development with Docker Compose**

```bash
# Start all services
docker-compose -f docker-compose.dev.yml up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

3. **Native Development Setup**

```bash
# Install Python dependencies
pip install -r requirements-dev.txt

# Install Node.js dependencies
npm install

# Start development servers
npm run dev
```

### Development Services

| Service | Port | Description |
|---------|------|-------------|
| API Server | 8000 | FastAPI backend |
| Frontend | 3000 | React development server |
| Jupyter | 8888 | Jupyter Lab environment |
| Monitoring | 6006 | TensorBoard/monitoring |

## 🏭 Production Deployment

### Architecture Overview

```mermaid
graph TB
    LB[Load Balancer] --> API1[API Server 1]
    LB --> API2[API Server 2]
    LB --> API3[API Server 3]
    
    API1 --> DB[(PostgreSQL)]
    API2 --> DB
    API3 --> DB
    
    API1 --> REDIS[(Redis)]
    API2 --> REDIS
    API3 --> REDIS
    
    API1 --> WORKER1[Worker 1]
    API2 --> WORKER2[Worker 2]
    API3 --> WORKER3[Worker 3]
    
    WORKER1 --> QUEUE[Message Queue]
    WORKER2 --> QUEUE
    WORKER3 --> QUEUE
```

### Docker Production Deployment

1. **Build Production Images**

```bash
# Build multi-stage production image
docker build \
  -f docker/production.dockerfile \
  --target production \
  --build-arg VERSION=$(git describe --tags) \
  --build-arg BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ') \
  --build-arg VCS_REF=$(git rev-parse HEAD) \
  -t terragon/codesign-playground:latest .

# Build specialized variants
docker build --target high-performance -t terragon/codesign-playground:high-perf .
docker build --target security-hardened -t terragon/codesign-playground:secure .
docker build --target monitoring -t terragon/codesign-playground:monitoring .
```

2. **Production Docker Compose**

```yaml
# docker-compose.prod.yml
version: '3.8'
services:
  api:
    image: terragon/codesign-playground:latest
    restart: unless-stopped
    environment:
      - NODE_ENV=production
      - WORKERS=4
      - LOG_LEVEL=info
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
        reservations:
          cpus: '0.5'
          memory: 1G
    networks:
      - internal
    depends_on:
      - postgres
      - redis

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/ssl:ro
    depends_on:
      - api
    networks:
      - internal

  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: codesign_playground
      POSTGRES_USER: codesign
      POSTGRES_PASSWORD_FILE: /run/secrets/postgres_password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    secrets:
      - postgres_password
    networks:
      - internal

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes --maxmemory 256mb
    volumes:
      - redis_data:/data
    networks:
      - internal

volumes:
  postgres_data:
  redis_data:

networks:
  internal:
    driver: bridge

secrets:
  postgres_password:
    file: ./secrets/postgres_password.txt
```

### Kubernetes Deployment

1. **Deploy to Kubernetes**

```bash
# Apply production configuration
kubectl apply -f deployment/kubernetes/production.yaml

# Check deployment status
kubectl get pods -n codesign-playground

# View logs
kubectl logs -f deployment/codesign-playground-api -n codesign-playground
```

2. **Scale Application**

```bash
# Manual scaling
kubectl scale deployment codesign-playground-api --replicas=5 -n codesign-playground

# Enable auto-scaling (HPA is already configured)
kubectl get hpa -n codesign-playground
```

## 🌍 Multi-Region Deployment

### Global Architecture

```mermaid
graph TB
    subgraph "US East (Primary)"
        USE1[API Cluster] --> USEDB[(Database)]
        USE1 --> USEREDIS[(Redis)]
    end
    
    subgraph "EU West (Secondary)"
        EUW1[API Cluster] --> EUWDB[(Database)]
        EUW1 --> EUWREDIS[(Redis)]
    end
    
    subgraph "Asia Pacific (Secondary)"
        AP1[API Cluster] --> APDB[(Database)]
        AP1 --> APREDIS[(Redis)]
    end
    
    CDN[Global CDN] --> USE1
    CDN --> EUW1
    CDN --> AP1
    
    USEDB -.->|Replication| EUWDB
    USEDB -.->|Replication| APDB
```

### Multi-Region Configuration

1. **Environment Variables**

```bash
# Primary region (US-East-1)
export AWS_REGION=us-east-1
export ENABLE_MULTI_REGION=true
export PRIMARY_REGION=true
export REPLICATION_REGIONS=eu-west-1,ap-southeast-1

# Secondary regions
export AWS_REGION=eu-west-1
export ENABLE_MULTI_REGION=true
export PRIMARY_REGION=false
export PRIMARY_REGION_ENDPOINT=https://api-us.codesign-playground.com
```

2. **Deploy to Multiple Regions**

```bash
# Deploy to primary region
helm install codesign-playground ./helm/codesign-playground \
  --namespace codesign-playground \
  --set global.region=us-east-1 \
  --set global.primaryRegion=true

# Deploy to secondary regions
for region in eu-west-1 ap-southeast-1; do
  helm install codesign-playground ./helm/codesign-playground \
    --namespace codesign-playground \
    --set global.region=$region \
    --set global.primaryRegion=false \
    --set global.primaryEndpoint=https://api-us.codesign-playground.com
done
```

### Regional Endpoints

| Region | Endpoint | Primary | Compliance |
|--------|----------|---------|------------|
| US East | api-us.codesign-playground.com | ✅ | CCPA, SOC2 |
| EU West | api-eu.codesign-playground.com | ❌ | GDPR, ISO27001 |
| Asia Pacific | api-ap.codesign-playground.com | ❌ | PDPA |

## 🔒 Security Hardening

### Security-Hardened Deployment

1. **Use Security-Hardened Image**

```bash
# Build and deploy security-hardened variant
docker build --target security-hardened -t codesign-playground:secure .
docker run -d \
  --name codesign-playground-secure \
  --security-opt no-new-privileges:true \
  --cap-drop ALL \
  --read-only \
  --tmpfs /tmp:rw,noexec,nosuid,size=100m \
  -p 8000:8000 \
  codesign-playground:secure
```

2. **Security Configuration**

```yaml
# security-config.yaml
security:
  authentication:
    jwt_expiry: 3600
    refresh_token_expiry: 604800
    max_login_attempts: 5
    lockout_duration: 900
  
  rate_limiting:
    enabled: true
    requests_per_minute: 100
    burst_limit: 200
  
  input_validation:
    max_request_size: 10MB
    allowed_file_types: ['.onnx', '.pb', '.pt', '.json']
    sanitize_inputs: true
  
  headers:
    hsts_max_age: 31536000
    content_security_policy: "default-src 'self'"
    x_frame_options: DENY
    x_content_type_options: nosniff
```

3. **Enable Security Features**

```bash
# Environment variables for security
export SECURITY_HARDENED=true
export ENABLE_RATE_LIMITING=true
export ENABLE_REQUEST_VALIDATION=strict
export MAX_REQUEST_SIZE=10MB
export ALLOWED_HOSTS="codesign-playground.com,*.codesign-playground.com"
```

### SSL/TLS Configuration

1. **Generate SSL Certificates**

```bash
# Using Let's Encrypt
certbot certonly --webroot \
  -w /var/www/html \
  -d codesign-playground.com \
  -d api.codesign-playground.com
```

2. **Nginx SSL Configuration**

```nginx
server {
    listen 443 ssl http2;
    server_name codesign-playground.com;
    
    ssl_certificate /etc/ssl/certs/codesign-playground.com.crt;
    ssl_certificate_key /etc/ssl/private/codesign-playground.com.key;
    
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;
    ssl_prefer_server_ciphers off;
    
    add_header Strict-Transport-Security "max-age=63072000" always;
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header Referrer-Policy "strict-origin-when-cross-origin";
    
    location / {
        proxy_pass http://api:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

## 📊 Monitoring & Observability

### Monitoring Stack

1. **Deploy Monitoring Variant**

```bash
docker build --target monitoring -t codesign-playground:monitoring .
docker run -d \
  --name codesign-playground-monitoring \
  -p 8000:8000 \
  -p 9090:9090 \
  -p 9464:9464 \
  -e ENABLE_METRICS=true \
  -e ENABLE_TRACING=true \
  codesign-playground:monitoring
```

2. **Prometheus Configuration**

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'codesign-playground'
    static_configs:
      - targets: ['localhost:9090']
    metrics_path: /metrics
    scrape_interval: 30s
  
  - job_name: 'system'
    static_configs:
      - targets: ['localhost:9464']
    scrape_interval: 60s
```

3. **Grafana Dashboard**

```bash
# Import pre-built dashboard
curl -X POST \
  http://admin:admin@localhost:3000/api/dashboards/db \
  -H 'Content-Type: application/json' \
  -d @monitoring/grafana-dashboards/backend-performance.json
```

### Key Metrics

| Metric | Description | Alert Threshold |
|--------|-------------|-----------------|
| Response Time | API response latency | > 2 seconds |
| Error Rate | HTTP 5xx error percentage | > 5% |
| CPU Usage | Container CPU utilization | > 80% |
| Memory Usage | Container memory utilization | > 85% |
| Queue Size | Background job queue depth | > 100 jobs |

## 🌐 Compliance & Regulations

### GDPR Compliance (EU)

1. **Data Processing Configuration**

```bash
export COMPLIANCE_REGION=EU
export GDPR_COMPLIANCE=true
export DATA_RETENTION_DAYS=365
export ENABLE_RIGHT_TO_DELETION=true
export ENABLE_DATA_PORTABILITY=true
```

2. **Privacy Controls**

```python
# Privacy API endpoints
POST /api/v1/privacy/consent       # Manage user consent
GET  /api/v1/privacy/data-export   # Export user data
POST /api/v1/privacy/data-deletion # Request data deletion
GET  /api/v1/privacy/policy        # Get privacy policy
```

### CCPA Compliance (California)

```bash
export COMPLIANCE_REGION=US
export CCPA_COMPLIANCE=true
export ENABLE_DO_NOT_SELL=true
export ENABLE_DATA_TRANSPARENCY=true
```

### Multi-Region Compliance Matrix

| Region | Regulations | Data Residency | Consent Required | Features |
|--------|-------------|----------------|------------------|----------|
| EU | GDPR | ✅ Required | ✅ Explicit | Right to deletion, portability |
| US | CCPA, COPPA | ❌ Optional | ✅ Opt-out | Do not sell, transparency |
| APAC | PDPA | ✅ Required | ✅ Explicit | Data localization |
| Global | ISO27001, SOC2 | ❌ Optional | ❌ Optional | Security standards |

## 🔧 Quality Gates & CI/CD

### Quality Gates Script

```bash
# Run comprehensive quality checks
./scripts/run-quality-gates.sh

# Quality gates include:
# - Unit and integration tests (85%+ coverage)
# - Security scanning (0 high-severity issues)
# - Performance benchmarks (no regressions)
# - Code quality checks (linting, formatting)
# - Dependency vulnerability scanning
```

### CI/CD Pipeline

```yaml
# .github/workflows/deploy.yml
name: Deploy to Production

on:
  push:
    tags: ['v*']

jobs:
  quality-gates:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run Quality Gates
        run: ./scripts/run-quality-gates.sh
  
  build-and-deploy:
    needs: quality-gates
    runs-on: ubuntu-latest
    steps:
      - name: Build Production Image
        run: docker build -f docker/production.dockerfile .
      
      - name: Deploy to Kubernetes
        run: kubectl apply -f deployment/kubernetes/production.yaml
```

## 🐛 Troubleshooting

### Common Issues

1. **Container Startup Failures**

```bash
# Check container logs
docker logs codesign-playground

# Check health status
docker exec codesign-playground curl -f http://localhost:8000/health

# Validate environment
docker exec codesign-playground env | grep CODESIGN
```

2. **Database Connection Issues**

```bash
# Test database connectivity
docker exec codesign-playground python -c "
import psycopg2
conn = psycopg2.connect('$DATABASE_URL')
print('Database connection successful')
conn.close()
"
```

3. **Performance Issues**

```bash
# Check resource usage
docker stats codesign-playground

# View performance metrics
curl http://localhost:9090/metrics

# Analyze slow queries
docker exec postgres psql -c "
SELECT query, mean_time, calls 
FROM pg_stat_statements 
ORDER BY mean_time DESC LIMIT 10;
"
```

### Debug Mode

```bash
# Enable debug logging
export LOG_LEVEL=debug
export CODESIGN_DEBUG=true

# Start with debugging
docker run -it --rm \
  -e LOG_LEVEL=debug \
  -e CODESIGN_DEBUG=true \
  codesign-playground:latest \
  bash
```

### Support Channels

- 📧 **Email**: support@terragon-labs.com
- 💬 **Discord**: [Join our community](https://discord.gg/terragon-labs)
- 📖 **Documentation**: [Full deployment docs](https://docs.terragon-labs.com/deployment)
- 🐛 **Issues**: [GitHub Issues](https://github.com/terragon-labs/ai-hardware-codesign-playground/issues)

## 📚 Additional Resources

- [Architecture Decision Records (ADRs)](docs/adr/)
- [Security Best Practices](SECURITY.md)
- [Performance Tuning Guide](docs/PERFORMANCE.md)
- [Monitoring and Alerting](docs/MONITORING.md)
- [API Documentation](docs/API.md)
- [Contributing Guidelines](CONTRIBUTING.md)

---

**Note**: This deployment guide covers production-ready configurations. For development and testing purposes, simpler configurations are available in the main [README.md](README.md).

*Last updated: 2024-01-15 | Version: 0.1.0*