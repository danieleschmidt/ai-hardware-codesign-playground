#!/bin/bash
# Production Deployment Script for AI Hardware Co-Design Platform

set -e

echo "🚀 Starting Production Deployment"
echo "================================="

# Configuration
NAMESPACE="codesign-production"
DOCKER_IMAGE="codesign-playground:latest"
KUBECTL_TIMEOUT="600s"

# Check prerequisites
echo "🔍 Checking prerequisites..."

if ! command -v kubectl &> /dev/null; then
    echo "❌ kubectl is not installed"
    exit 1
fi

if ! command -v docker &> /dev/null; then
    echo "❌ docker is not installed" 
    exit 1
fi

if ! kubectl cluster-info &> /dev/null; then
    echo "❌ kubectl is not connected to a cluster"
    exit 1
fi

echo "✅ Prerequisites check passed"

# Build Docker image
echo "🔨 Building Docker image..."
docker build -t ${DOCKER_IMAGE} -f docker/Dockerfile.production ../..

# Push to registry (if configured)
if [[ -n "${DOCKER_REGISTRY}" ]]; then
    echo "📤 Pushing Docker image to registry..."
    docker tag ${DOCKER_IMAGE} ${DOCKER_REGISTRY}/${DOCKER_IMAGE}
    docker push ${DOCKER_REGISTRY}/${DOCKER_IMAGE}
fi

# Apply Kubernetes manifests
echo "☸️  Applying Kubernetes manifests..."

# Create namespace
kubectl apply -f kubernetes/namespace.yaml

# Apply configurations
kubectl apply -f kubernetes/configmap.yaml
kubectl apply -f kubernetes/secrets.yaml

# Deploy database
kubectl apply -f database/postgres-deployment.yaml
kubectl wait --for=condition=ready pod -l app=postgres -n ${NAMESPACE} --timeout=${KUBECTL_TIMEOUT}

# Deploy Redis
kubectl apply -f database/redis-deployment.yaml
kubectl wait --for=condition=ready pod -l app=redis -n ${NAMESPACE} --timeout=${KUBECTL_TIMEOUT}

# Deploy application
kubectl apply -f kubernetes/deployment.yaml
kubectl apply -f kubernetes/service.yaml
kubectl apply -f kubernetes/hpa.yaml

# Wait for deployment
echo "⏳ Waiting for deployment to be ready..."
kubectl wait --for=condition=available deployment/codesign-app -n ${NAMESPACE} --timeout=${KUBECTL_TIMEOUT}

# Verify deployment
echo "🔍 Verifying deployment..."
kubectl get pods -n ${NAMESPACE}
kubectl get services -n ${NAMESPACE}

# Health check
echo "🏥 Performing health check..."
SERVICE_IP=$(kubectl get service codesign-service -n ${NAMESPACE} -o jsonpath='{.spec.clusterIP}')
if kubectl run health-check --image=curlimages/curl --rm -i --restart=Never -- curl -f http://${SERVICE_IP}/health; then
    echo "✅ Health check passed"
else
    echo "❌ Health check failed"
    exit 1
fi

echo "🎉 Production deployment completed successfully!"
