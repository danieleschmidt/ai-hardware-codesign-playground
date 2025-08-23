
terraform {
  required_version = ">= 1.0"
  required_providers {
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.0"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.0"
    }
  }
}

provider "kubernetes" {
  config_path = "~/.kube/config"
}

provider "helm" {
  kubernetes {
    config_path = "~/.kube/config"
  }
}

# Create namespace
resource "kubernetes_namespace" "codesign_production" {
  metadata {
    name = "codesign-production"
    labels = {
      environment = "production"
      managed-by  = "terraform"
    }
  }
}

# Install Prometheus using Helm
resource "helm_release" "prometheus" {
  name       = "prometheus"
  repository = "https://prometheus-community.github.io/helm-charts"
  chart      = "kube-prometheus-stack"
  namespace  = kubernetes_namespace.codesign_production.metadata[0].name

  values = [
    file("${path.module}/prometheus-values.yaml")
  ]

  depends_on = [kubernetes_namespace.codesign_production]
}

# Install Grafana using Helm
resource "helm_release" "grafana" {
  name       = "grafana"
  repository = "https://grafana.github.io/helm-charts"
  chart      = "grafana"
  namespace  = kubernetes_namespace.codesign_production.metadata[0].name

  set {
    name  = "adminPassword"
    value = var.grafana_admin_password
  }

  depends_on = [kubernetes_namespace.codesign_production]
}

# Variables
variable "grafana_admin_password" {
  description = "Admin password for Grafana"
  type        = string
  sensitive   = true
}

# Outputs
output "namespace_name" {
  value = kubernetes_namespace.codesign_production.metadata[0].name
}
