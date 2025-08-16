# Terraform Variables for AI Hardware Co-Design Playground Infrastructure

# Environment Configuration
variable "environment" {
  description = "Environment name (production, staging, development)"
  type        = string
  default     = "production"
  
  validation {
    condition     = contains(["production", "staging", "development"], var.environment)
    error_message = "Environment must be one of: production, staging, development."
  }
}

variable "primary_region" {
  description = "Primary AWS region"
  type        = string
  default     = "us-east-1"
}

variable "secondary_regions" {
  description = "Secondary regions for multi-region deployment"
  type        = list(string)
  default     = ["us-west-2", "eu-west-1"]
}

# Network Configuration
variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
  
  validation {
    condition     = can(cidrhost(var.vpc_cidr, 0))
    error_message = "VPC CIDR must be a valid IPv4 CIDR block."
  }
}

variable "availability_zones" {
  description = "List of availability zones"
  type        = list(string)
  default     = []
}

# Domain and DNS Configuration
variable "domain_name" {
  description = "Domain name for the application"
  type        = string
  default     = "codesign.example.com"
}

variable "create_dns_zone" {
  description = "Whether to create Route53 hosted zone"
  type        = bool
  default     = true
}

variable "create_ssl_certificate" {
  description = "Whether to create ACM SSL certificate"
  type        = bool
  default     = true
}

# Kubernetes Configuration
variable "kubernetes_version" {
  description = "Kubernetes version for EKS cluster"
  type        = string
  default     = "1.28"
}

variable "node_groups" {
  description = "EKS node group configurations"
  type = map(object({
    instance_types = list(string)
    min_size      = number
    max_size      = number
    desired_size  = number
    disk_size     = number
    ami_type      = string
    capacity_type = string
    labels        = map(string)
    taints = list(object({
      key    = string
      value  = string
      effect = string
    }))
  }))
  default = {
    general = {
      instance_types = ["t3.large", "t3a.large"]
      min_size      = 3
      max_size      = 10
      desired_size  = 3
      disk_size     = 50
      ami_type      = "AL2_x86_64"
      capacity_type = "ON_DEMAND"
      labels = {
        role = "general"
      }
      taints = []
    }
    compute = {
      instance_types = ["c5.xlarge", "c5a.xlarge"]
      min_size      = 0
      max_size      = 20
      desired_size  = 2
      disk_size     = 100
      ami_type      = "AL2_x86_64"
      capacity_type = "SPOT"
      labels = {
        role = "compute"
      }
      taints = [
        {
          key    = "compute-optimized"
          value  = "true"
          effect = "NO_SCHEDULE"
        }
      ]
    }
    memory = {
      instance_types = ["r5.large", "r5a.large"]
      min_size      = 0
      max_size      = 10
      desired_size  = 1
      disk_size     = 50
      ami_type      = "AL2_x86_64"
      capacity_type = "SPOT"
      labels = {
        role = "memory-optimized"
      }
      taints = [
        {
          key    = "memory-optimized"
          value  = "true"
          effect = "NO_SCHEDULE"
        }
      ]
    }
  }
}

# Database Configuration
variable "postgres_version" {
  description = "PostgreSQL version"
  type        = string
  default     = "15.4"
}

variable "rds_instance_class" {
  description = "RDS instance class"
  type        = string
  default     = "db.r6g.large"
}

variable "rds_allocated_storage" {
  description = "RDS allocated storage (GB)"
  type        = number
  default     = 100
}

variable "rds_max_allocated_storage" {
  description = "RDS maximum allocated storage (GB)"
  type        = number
  default     = 1000
}

variable "rds_multi_az" {
  description = "Enable RDS Multi-AZ deployment"
  type        = bool
  default     = true
}

variable "rds_backup_retention_period" {
  description = "RDS backup retention period (days)"
  type        = number
  default     = 30
}

variable "enable_rds_performance_insights" {
  description = "Enable RDS Performance Insights"
  type        = bool
  default     = true
}

# Redis Configuration
variable "redis_node_type" {
  description = "ElastiCache Redis node type"
  type        = string
  default     = "cache.r6g.large"
}

variable "redis_num_nodes" {
  description = "Number of Redis nodes"
  type        = number
  default     = 3
}

variable "redis_parameter_group_family" {
  description = "Redis parameter group family"
  type        = string
  default     = "redis7"
}

variable "enable_redis_auth" {
  description = "Enable Redis authentication"
  type        = bool
  default     = true
}

variable "enable_redis_encryption" {
  description = "Enable Redis encryption at rest and in transit"
  type        = bool
  default     = true
}

# Monitoring Configuration
variable "enable_prometheus" {
  description = "Enable Prometheus monitoring"
  type        = bool
  default     = true
}

variable "enable_grafana" {
  description = "Enable Grafana dashboards"
  type        = bool
  default     = true
}

variable "enable_alertmanager" {
  description = "Enable Alertmanager"
  type        = bool
  default     = true
}

variable "enable_jaeger" {
  description = "Enable Jaeger distributed tracing"
  type        = bool
  default     = true
}

variable "enable_elasticsearch" {
  description = "Enable Elasticsearch for log aggregation"
  type        = bool
  default     = true
}

variable "grafana_admin_password" {
  description = "Grafana admin password"
  type        = string
  default     = "admin123"
  sensitive   = true
}

# Alerting Configuration
variable "slack_webhook_url" {
  description = "Slack webhook URL for alerts"
  type        = string
  default     = ""
  sensitive   = true
}

variable "pagerduty_integration_key" {
  description = "PagerDuty integration key"
  type        = string
  default     = ""
  sensitive   = true
}

variable "email_alerts" {
  description = "Email addresses for alerts"
  type        = list(string)
  default     = []
}

# Security Configuration
variable "enable_vault" {
  description = "Enable HashiCorp Vault for secrets management"
  type        = bool
  default     = true
}

variable "enable_falco" {
  description = "Enable Falco for runtime security"
  type        = bool
  default     = true
}

variable "enable_policy_engine" {
  description = "Enable OPA/Gatekeeper for policy enforcement"
  type        = bool
  default     = true
}

variable "enable_network_policies" {
  description = "Enable Kubernetes network policies"
  type        = bool
  default     = true
}

variable "enable_pod_security_policies" {
  description = "Enable Pod Security Policies"
  type        = bool
  default     = true
}

# Backup Configuration
variable "backup_retention_days" {
  description = "Number of days to retain backups"
  type        = number
  default     = 90
}

variable "enable_cross_region_backup" {
  description = "Enable cross-region backup"
  type        = bool
  default     = true
}

variable "backup_regions" {
  description = "Regions for cross-region backup"
  type        = list(string)
  default     = ["us-west-2", "eu-west-1"]
}

variable "s3_backup_bucket" {
  description = "S3 bucket name for backups"
  type        = string
  default     = ""
}

variable "enable_automated_snapshots" {
  description = "Enable automated EBS snapshots"
  type        = bool
  default     = true
}

# Auto Scaling Configuration
variable "enable_cluster_autoscaler" {
  description = "Enable Kubernetes Cluster Autoscaler"
  type        = bool
  default     = true
}

variable "enable_hpa" {
  description = "Enable Horizontal Pod Autoscaler"
  type        = bool
  default     = true
}

variable "enable_vpa" {
  description = "Enable Vertical Pod Autoscaler"
  type        = bool
  default     = true
}

# Cost Optimization
variable "enable_spot_instances" {
  description = "Enable Spot instances for cost optimization"
  type        = bool
  default     = true
}

variable "spot_instance_pools" {
  description = "Number of Spot instance pools"
  type        = number
  default     = 3
}

variable "enable_scheduled_scaling" {
  description = "Enable scheduled scaling"
  type        = bool
  default     = false
}

# Multi-Region Configuration
variable "enable_multi_region" {
  description = "Enable multi-region deployment"
  type        = bool
  default     = false
}

variable "region_config" {
  description = "Configuration for each region"
  type = map(object({
    enabled           = bool
    is_primary        = bool
    node_groups      = map(any)
    database_replica = bool
    redis_replica    = bool
  }))
  default = {
    "us-east-1" = {
      enabled           = true
      is_primary        = true
      node_groups      = {}
      database_replica = false
      redis_replica    = false
    }
    "us-west-2" = {
      enabled           = false
      is_primary        = false
      node_groups      = {}
      database_replica = true
      redis_replica    = true
    }
    "eu-west-1" = {
      enabled           = false
      is_primary        = false
      node_groups      = {}
      database_replica = true
      redis_replica    = true
    }
  }
}

# Compliance Configuration
variable "enable_compliance_monitoring" {
  description = "Enable compliance monitoring (GDPR, CCPA, etc.)"
  type        = bool
  default     = true
}

variable "data_residency_regions" {
  description = "Regions where data must reside for compliance"
  type        = list(string)
  default     = ["us-east-1", "eu-west-1"]
}

variable "enable_audit_logging" {
  description = "Enable audit logging"
  type        = bool
  default     = true
}

variable "audit_log_retention_days" {
  description = "Audit log retention period (days)"
  type        = number
  default     = 2555  # 7 years
}

# Performance Configuration
variable "enable_performance_monitoring" {
  description = "Enable detailed performance monitoring"
  type        = bool
  default     = true
}

variable "enable_x_ray" {
  description = "Enable AWS X-Ray tracing"
  type        = bool
  default     = true
}

variable "cloudwatch_retention_days" {
  description = "CloudWatch logs retention period (days)"
  type        = number
  default     = 90
}

# Disaster Recovery Configuration
variable "enable_disaster_recovery" {
  description = "Enable disaster recovery setup"
  type        = bool
  default     = true
}

variable "rto_minutes" {
  description = "Recovery Time Objective in minutes"
  type        = number
  default     = 60
}

variable "rpo_minutes" {
  description = "Recovery Point Objective in minutes"
  type        = number
  default     = 15
}

# Edge Computing Configuration
variable "enable_edge_deployment" {
  description = "Enable edge computing deployment"
  type        = bool
  default     = false
}

variable "edge_locations" {
  description = "Edge computing locations"
  type        = list(string)
  default     = []
}

# Development and Testing
variable "enable_load_testing" {
  description = "Enable load testing infrastructure"
  type        = bool
  default     = false
}

variable "enable_chaos_engineering" {
  description = "Enable chaos engineering tools"
  type        = bool
  default     = false
}

# External Integrations
variable "enable_datadog" {
  description = "Enable Datadog integration"
  type        = bool
  default     = false
}

variable "datadog_api_key" {
  description = "Datadog API key"
  type        = string
  default     = ""
  sensitive   = true
}

variable "enable_newrelic" {
  description = "Enable New Relic integration"
  type        = bool
  default     = false
}

variable "newrelic_license_key" {
  description = "New Relic license key"
  type        = string
  default     = ""
  sensitive   = true
}

# Resource Tags
variable "additional_tags" {
  description = "Additional tags to apply to all resources"
  type        = map(string)
  default     = {}
}

variable "cost_center" {
  description = "Cost center for billing"
  type        = string
  default     = "Engineering"
}

variable "owner" {
  description = "Owner of the infrastructure"
  type        = string
  default     = "Platform Engineering"
}

variable "project" {
  description = "Project name"
  type        = string
  default     = "AI Hardware Co-Design Playground"
}