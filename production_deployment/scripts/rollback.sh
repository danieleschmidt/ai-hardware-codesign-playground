#!/bin/bash
# Rollback Script for AI Hardware Co-Design Platform

set -e

NAMESPACE="codesign-production"
DEPLOYMENT_NAME="codesign-app"

echo "🔄 Starting rollback process..."

# Get rollout history
echo "📊 Current rollout history:"
kubectl rollout history deployment/${DEPLOYMENT_NAME} -n ${NAMESPACE}

# Perform rollback
echo "⏪ Rolling back to previous version..."
kubectl rollout undo deployment/${DEPLOYMENT_NAME} -n ${NAMESPACE}

# Wait for rollback to complete
echo "⏳ Waiting for rollback to complete..."
kubectl rollout status deployment/${DEPLOYMENT_NAME} -n ${NAMESPACE} --timeout=600s

# Verify rollback
echo "🔍 Verifying rollback..."
kubectl get pods -n ${NAMESPACE}

echo "✅ Rollback completed successfully!"
