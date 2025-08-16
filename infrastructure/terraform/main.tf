# AI Hardware Co-Design Playground - Terraform Infrastructure
# Production-ready infrastructure for multi-cloud deployment

terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    google = {
      source  = "hashicorp/google"
      version = "~> 4.80"
    }
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.70"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.23"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.11"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.5"
    }
    tls = {
      source  = "hashicorp/tls"
      version = "~> 4.0"
    }
  }

  # Remote state backend configuration
  backend "s3" {
    bucket         = "codesign-terraform-state"
    key            = "production/terraform.tfstate"
    region         = "us-east-1"
    encrypt        = true
    dynamodb_table = "codesign-terraform-locks"
    
    # Enable versioning and backup
    versioning = true
  }
}

# Local variables for common configurations
locals {
  # Project configuration
  project_name = "ai-hardware-codesign-playground"
  environment  = var.environment
  region       = var.primary_region
  
  # Common tags
  common_tags = {
    Project     = local.project_name
    Environment = local.environment
    ManagedBy   = "Terraform"
    Owner       = "Platform Engineering"
    CostCenter  = "Engineering"
    Compliance  = "GDPR,CCPA,PDPA"
    Backup      = "Required"
    Monitoring  = "Enabled"
  }
  
  # Network configuration
  vpc_cidr = var.vpc_cidr
  availability_zones = data.aws_availability_zones.available.names
  
  # Cluster configuration
  cluster_name = "${local.project_name}-${local.environment}"
  
  # Domain configuration
  domain_name = var.domain_name
  subdomain   = "${local.environment}.${var.domain_name}"
}

# Data sources
data "aws_availability_zones" "available" {
  state = "available"
}

data "aws_caller_identity" "current" {}

# Random password generation for databases
resource "random_password" "postgres_password" {
  length  = 32
  special = true
}

resource "random_password" "redis_password" {
  length  = 32
  special = false
}

resource "random_password" "secret_key" {
  length  = 64
  special = true
}

# Main VPC
resource "aws_vpc" "main" {
  cidr_block           = local.vpc_cidr
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = merge(local.common_tags, {
    Name = "${local.cluster_name}-vpc"
    Type = "main-vpc"
  })
}

# Internet Gateway
resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id

  tags = merge(local.common_tags, {
    Name = "${local.cluster_name}-igw"
  })
}

# Public subnets
resource "aws_subnet" "public" {
  count = min(length(local.availability_zones), 3)

  vpc_id                  = aws_vpc.main.id
  cidr_block              = cidrsubnet(local.vpc_cidr, 4, count.index)
  availability_zone       = local.availability_zones[count.index]
  map_public_ip_on_launch = true

  tags = merge(local.common_tags, {
    Name = "${local.cluster_name}-public-${count.index + 1}"
    Type = "public"
    Tier = "public"
    "kubernetes.io/role/elb" = "1"
    "kubernetes.io/cluster/${local.cluster_name}" = "shared"
  })
}

# Private subnets
resource "aws_subnet" "private" {
  count = min(length(local.availability_zones), 3)

  vpc_id            = aws_vpc.main.id
  cidr_block        = cidrsubnet(local.vpc_cidr, 4, count.index + 3)
  availability_zone = local.availability_zones[count.index]

  tags = merge(local.common_tags, {
    Name = "${local.cluster_name}-private-${count.index + 1}"
    Type = "private"
    Tier = "private"
    "kubernetes.io/role/internal-elb" = "1"
    "kubernetes.io/cluster/${local.cluster_name}" = "shared"
  })
}

# Database subnets
resource "aws_subnet" "database" {
  count = min(length(local.availability_zones), 3)

  vpc_id            = aws_vpc.main.id
  cidr_block        = cidrsubnet(local.vpc_cidr, 4, count.index + 6)
  availability_zone = local.availability_zones[count.index]

  tags = merge(local.common_tags, {
    Name = "${local.cluster_name}-database-${count.index + 1}"
    Type = "database"
    Tier = "data"
  })
}

# NAT Gateways
resource "aws_eip" "nat" {
  count = min(length(local.availability_zones), 3)

  domain = "vpc"
  depends_on = [aws_internet_gateway.main]

  tags = merge(local.common_tags, {
    Name = "${local.cluster_name}-nat-eip-${count.index + 1}"
  })
}

resource "aws_nat_gateway" "main" {
  count = min(length(local.availability_zones), 3)

  allocation_id = aws_eip.nat[count.index].id
  subnet_id     = aws_subnet.public[count.index].id

  tags = merge(local.common_tags, {
    Name = "${local.cluster_name}-nat-${count.index + 1}"
  })

  depends_on = [aws_internet_gateway.main]
}

# Route tables
resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.main.id
  }

  tags = merge(local.common_tags, {
    Name = "${local.cluster_name}-public-rt"
    Type = "public"
  })
}

resource "aws_route_table" "private" {
  count = min(length(local.availability_zones), 3)

  vpc_id = aws_vpc.main.id

  route {
    cidr_block     = "0.0.0.0/0"
    nat_gateway_id = aws_nat_gateway.main[count.index].id
  }

  tags = merge(local.common_tags, {
    Name = "${local.cluster_name}-private-rt-${count.index + 1}"
    Type = "private"
  })
}

resource "aws_route_table" "database" {
  vpc_id = aws_vpc.main.id

  tags = merge(local.common_tags, {
    Name = "${local.cluster_name}-database-rt"
    Type = "database"
  })
}

# Route table associations
resource "aws_route_table_association" "public" {
  count = length(aws_subnet.public)

  subnet_id      = aws_subnet.public[count.index].id
  route_table_id = aws_route_table.public.id
}

resource "aws_route_table_association" "private" {
  count = length(aws_subnet.private)

  subnet_id      = aws_subnet.private[count.index].id
  route_table_id = aws_route_table.private[count.index].id
}

resource "aws_route_table_association" "database" {
  count = length(aws_subnet.database)

  subnet_id      = aws_subnet.database[count.index].id
  route_table_id = aws_route_table.database.id
}

# Security Groups
resource "aws_security_group" "eks_cluster" {
  name_prefix = "${local.cluster_name}-eks-cluster"
  vpc_id      = aws_vpc.main.id

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = merge(local.common_tags, {
    Name = "${local.cluster_name}-eks-cluster-sg"
    Type = "eks-cluster"
  })
}

resource "aws_security_group" "eks_nodes" {
  name_prefix = "${local.cluster_name}-eks-nodes"
  vpc_id      = aws_vpc.main.id

  ingress {
    from_port = 0
    to_port   = 65535
    protocol  = "tcp"
    self      = true
  }

  ingress {
    from_port       = 1025
    to_port         = 65535
    protocol        = "tcp"
    security_groups = [aws_security_group.eks_cluster.id]
  }

  ingress {
    from_port       = 443
    to_port         = 443
    protocol        = "tcp"
    security_groups = [aws_security_group.eks_cluster.id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = merge(local.common_tags, {
    Name = "${local.cluster_name}-eks-nodes-sg"
    Type = "eks-nodes"
  })
}

resource "aws_security_group" "rds" {
  name_prefix = "${local.cluster_name}-rds"
  vpc_id      = aws_vpc.main.id

  ingress {
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [aws_security_group.eks_nodes.id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = merge(local.common_tags, {
    Name = "${local.cluster_name}-rds-sg"
    Type = "database"
  })
}

resource "aws_security_group" "elasticache" {
  name_prefix = "${local.cluster_name}-elasticache"
  vpc_id      = aws_vpc.main.id

  ingress {
    from_port       = 6379
    to_port         = 6379
    protocol        = "tcp"
    security_groups = [aws_security_group.eks_nodes.id]
  }

  tags = merge(local.common_tags, {
    Name = "${local.cluster_name}-elasticache-sg"
    Type = "cache"
  })
}

# Include child modules
module "eks_cluster" {
  source = "./modules/eks"

  cluster_name           = local.cluster_name
  cluster_version        = var.kubernetes_version
  vpc_id                 = aws_vpc.main.id
  subnet_ids            = aws_subnet.private[*].id
  security_group_ids    = [aws_security_group.eks_cluster.id]
  node_security_group_ids = [aws_security_group.eks_nodes.id]
  
  node_groups = var.node_groups
  
  tags = local.common_tags
}

module "rds" {
  source = "./modules/rds"

  identifier     = "${local.cluster_name}-postgres"
  engine_version = var.postgres_version
  instance_class = var.rds_instance_class
  
  database_name = "codesign_db"
  username      = "codesign"
  password      = random_password.postgres_password.result
  
  vpc_id     = aws_vpc.main.id
  subnet_ids = aws_subnet.database[*].id
  security_group_ids = [aws_security_group.rds.id]
  
  backup_retention_period = var.backup_retention_days
  backup_window          = "03:00-04:00"
  maintenance_window     = "sun:04:00-sun:05:00"
  
  performance_insights_enabled = true
  monitoring_interval         = 60
  
  tags = local.common_tags
}

module "elasticache" {
  source = "./modules/elasticache"

  cluster_id           = "${local.cluster_name}-redis"
  node_type           = var.redis_node_type
  num_cache_clusters  = var.redis_num_nodes
  
  subnet_group_name   = aws_elasticache_subnet_group.main.name
  security_group_ids  = [aws_security_group.elasticache.id]
  
  at_rest_encryption_enabled = true
  transit_encryption_enabled = true
  auth_token                 = random_password.redis_password.result
  
  tags = local.common_tags
}

module "monitoring" {
  source = "./modules/monitoring"

  cluster_name = local.cluster_name
  vpc_id       = aws_vpc.main.id
  subnet_ids   = aws_subnet.private[*].id
  
  enable_prometheus = var.enable_prometheus
  enable_grafana   = var.enable_grafana
  enable_alertmanager = var.enable_alertmanager
  
  grafana_admin_password = var.grafana_admin_password
  slack_webhook_url     = var.slack_webhook_url
  pagerduty_integration_key = var.pagerduty_integration_key
  
  tags = local.common_tags
}

module "security" {
  source = "./modules/security"

  cluster_name = local.cluster_name
  vpc_id       = aws_vpc.main.id
  
  enable_vault     = var.enable_vault
  enable_falco     = var.enable_falco
  enable_policy_engine = var.enable_policy_engine
  
  vault_storage_encrypted = true
  
  tags = local.common_tags
}

module "backup" {
  source = "./modules/backup"

  cluster_name = local.cluster_name
  
  rds_instance_identifier = module.rds.instance_identifier
  backup_retention_days   = var.backup_retention_days
  
  enable_cross_region_backup = var.enable_cross_region_backup
  backup_regions            = var.backup_regions
  
  s3_backup_bucket = var.s3_backup_bucket
  
  tags = local.common_tags
}

# ElastiCache subnet group
resource "aws_elasticache_subnet_group" "main" {
  name       = "${local.cluster_name}-cache-subnet"
  subnet_ids = aws_subnet.database[*].id

  tags = merge(local.common_tags, {
    Name = "${local.cluster_name}-cache-subnet-group"
  })
}

# KMS keys for encryption
resource "aws_kms_key" "main" {
  description             = "KMS key for ${local.cluster_name}"
  deletion_window_in_days = 7
  enable_key_rotation     = true

  tags = merge(local.common_tags, {
    Name = "${local.cluster_name}-kms-key"
  })
}

resource "aws_kms_alias" "main" {
  name          = "alias/${local.cluster_name}"
  target_key_id = aws_kms_key.main.key_id
}

# S3 bucket for backups
resource "aws_s3_bucket" "backups" {
  bucket = "${local.cluster_name}-backups-${random_id.bucket_suffix.hex}"

  tags = merge(local.common_tags, {
    Name = "${local.cluster_name}-backups"
    Type = "backup"
  })
}

resource "random_id" "bucket_suffix" {
  byte_length = 4
}

resource "aws_s3_bucket_versioning" "backups" {
  bucket = aws_s3_bucket.backups.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_encryption" "backups" {
  bucket = aws_s3_bucket.backups.id

  server_side_encryption_configuration {
    rule {
      apply_server_side_encryption_by_default {
        kms_master_key_id = aws_kms_key.main.arn
        sse_algorithm     = "aws:kms"
      }
    }
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "backups" {
  bucket = aws_s3_bucket.backups.id

  rule {
    id     = "backup_lifecycle"
    status = "Enabled"

    expiration {
      days = var.backup_retention_days
    }

    noncurrent_version_expiration {
      noncurrent_days = 30
    }
  }
}

# CloudWatch Log Groups
resource "aws_cloudwatch_log_group" "eks_cluster" {
  name              = "/aws/eks/${local.cluster_name}/cluster"
  retention_in_days = 30
  kms_key_id        = aws_kms_key.main.arn

  tags = local.common_tags
}

# Route53 hosted zone
resource "aws_route53_zone" "main" {
  count = var.create_dns_zone ? 1 : 0
  
  name = local.domain_name

  tags = merge(local.common_tags, {
    Name = local.domain_name
    Type = "dns-zone"
  })
}

# ACM certificate
resource "aws_acm_certificate" "main" {
  count = var.create_ssl_certificate ? 1 : 0
  
  domain_name               = local.domain_name
  subject_alternative_names = ["*.${local.domain_name}"]
  validation_method         = "DNS"

  lifecycle {
    create_before_destroy = true
  }

  tags = merge(local.common_tags, {
    Name = local.domain_name
    Type = "ssl-certificate"
  })
}

# Outputs
output "cluster_name" {
  description = "Name of the EKS cluster"
  value       = local.cluster_name
}

output "cluster_endpoint" {
  description = "Endpoint of the EKS cluster"
  value       = module.eks_cluster.cluster_endpoint
}

output "cluster_security_group_id" {
  description = "Security group ID of the EKS cluster"
  value       = aws_security_group.eks_cluster.id
}

output "vpc_id" {
  description = "ID of the VPC"
  value       = aws_vpc.main.id
}

output "private_subnet_ids" {
  description = "IDs of the private subnets"
  value       = aws_subnet.private[*].id
}

output "database_subnet_ids" {
  description = "IDs of the database subnets"
  value       = aws_subnet.database[*].id
}

output "rds_endpoint" {
  description = "RDS instance endpoint"
  value       = module.rds.endpoint
  sensitive   = true
}

output "redis_endpoint" {
  description = "Redis cluster endpoint"
  value       = module.elasticache.redis_endpoint
  sensitive   = true
}

output "kms_key_id" {
  description = "KMS key ID for encryption"
  value       = aws_kms_key.main.key_id
}

output "backup_bucket" {
  description = "S3 bucket for backups"
  value       = aws_s3_bucket.backups.id
}

output "secrets" {
  description = "Generated secrets"
  value = {
    postgres_password = random_password.postgres_password.result
    redis_password    = random_password.redis_password.result
    secret_key        = random_password.secret_key.result
  }
  sensitive = true
}