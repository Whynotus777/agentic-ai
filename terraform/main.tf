# terraform/main.tf - Enterprise Multi-Agent AI Infrastructure

terraform {
  required_version = ">= 1.5.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.23"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.11"
    }
  }
  
  backend "s3" {
    bucket         = "agentic-ai-terraform-state"
    key            = "infrastructure/terraform.tfstate"
    region         = "us-east-1"
    dynamodb_table = "terraform-state-lock"
    encrypt        = true
  }
}

provider "aws" {
  region = var.aws_region
  default_tags {
    tags = {
      Environment = var.environment
      ManagedBy   = "Terraform"
      Project     = "AgenticAI"
      CostCenter  = var.cost_center
    }
  }
}

# Variables
variable "aws_region" {
  default = "us-east-1"
}

variable "environment" {
  default = "production"
}

variable "cost_center" {
  default = "ai-platform"
}

variable "cluster_name" {
  default = "agentic-ai-cluster"
}

# VPC Configuration
module "vpc" {
  source = "terraform-aws-modules/vpc/aws"
  version = "5.1.0"

  name = "${var.cluster_name}-vpc"
  cidr = "10.0.0.0/16"

  azs             = data.aws_availability_zones.available.names
  private_subnets = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
  public_subnets  = ["10.0.101.0/24", "10.0.102.0/24", "10.0.103.0/24"]
  database_subnets = ["10.0.201.0/24", "10.0.202.0/24", "10.0.203.0/24"]

  enable_nat_gateway = true
  enable_vpn_gateway = true
  enable_dns_hostnames = true
  enable_dns_support = true

  # Enable VPC flow logs for security
  enable_flow_log = true
  flow_log_destination_type = "cloud-watch-logs"

  tags = {
    "kubernetes.io/cluster/${var.cluster_name}" = "shared"
  }

  private_subnet_tags = {
    "kubernetes.io/cluster/${var.cluster_name}" = "owned"
    "kubernetes.io/role/internal-elb" = "1"
  }

  public_subnet_tags = {
    "kubernetes.io/cluster/${var.cluster_name}" = "owned"
    "kubernetes.io/role/elb" = "1"
  }
}

# EKS Cluster
module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "19.16.0"

  cluster_name    = var.cluster_name
  cluster_version = "1.28"

  vpc_id                   = module.vpc.vpc_id
  subnet_ids               = module.vpc.private_subnets
  control_plane_subnet_ids = module.vpc.private_subnets

  # Security
  cluster_endpoint_private_access = true
  cluster_endpoint_public_access  = true
  cluster_endpoint_public_access_cidrs = ["0.0.0.0/0"] # Restrict in production

  # Enable IRSA (IAM Roles for Service Accounts)
  enable_irsa = true

  # Enable cluster logging
  cluster_enabled_log_types = ["api", "audit", "authenticator", "controllerManager", "scheduler"]

  # Cluster addons
  cluster_addons = {
    coredns = {
      most_recent = true
    }
    kube-proxy = {
      most_recent = true
    }
    vpc-cni = {
      most_recent = true
    }
    aws-ebs-csi-driver = {
      most_recent = true
    }
  }

  # Node Groups
  eks_managed_node_groups = {
    # System nodes for control plane components
    system = {
      desired_size = 3
      min_size     = 3
      max_size     = 5

      instance_types = ["t3.large"]
      
      labels = {
        nodegroup = "system"
        workload  = "system"
      }

      taints = [
        {
          key    = "workload"
          value  = "system"
          effect = "NO_SCHEDULE"
        }
      ]
    }

    # CPU nodes for orchestrators
    orchestrator = {
      desired_size = 3
      min_size     = 2
      max_size     = 10

      instance_types = ["m5.2xlarge"]
      
      labels = {
        nodegroup = "orchestrator"
        workload  = "orchestrator"
      }
    }

    # GPU nodes for AI models
    gpu_inference = {
      desired_size = 2
      min_size     = 1
      max_size     = 5

      instance_types = ["g4dn.xlarge"]
      
      labels = {
        nodegroup = "gpu"
        workload  = "ai-inference"
        "nvidia.com/gpu" = "true"
      }

      taints = [
        {
          key    = "nvidia.com/gpu"
          value  = "true"
          effect = "NO_SCHEDULE"
        }
      ]

      # Install NVIDIA drivers
      user_data = base64encode(<<-EOT
        #!/bin/bash
        curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
        distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
        curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
        sudo apt-get update && sudo apt-get install -y nvidia-docker2
        sudo systemctl restart docker
      EOT
      )
    }

    # High memory nodes for vector operations
    memory_optimized = {
      desired_size = 2
      min_size     = 1
      max_size     = 5

      instance_types = ["r5.2xlarge"]
      
      labels = {
        nodegroup = "memory"
        workload  = "vector-db"
      }
    }
  }
}

# RDS for metadata and state
resource "aws_db_subnet_group" "agentic_ai" {
  name       = "${var.cluster_name}-db-subnet"
  subnet_ids = module.vpc.database_subnets
}

resource "aws_security_group" "rds" {
  name_prefix = "${var.cluster_name}-rds"
  vpc_id      = module.vpc.vpc_id

  ingress {
    from_port   = 5432
    to_port     = 5432
    protocol    = "tcp"
    cidr_blocks = [module.vpc.vpc_cidr_block]
  }
}

resource "aws_db_instance" "metadata" {
  identifier     = "${var.cluster_name}-metadata"
  engine         = "postgres"
  engine_version = "15.4"
  instance_class = "db.r5.xlarge"
  
  allocated_storage     = 100
  storage_encrypted     = true
  storage_type         = "gp3"
  
  db_name  = "agentic_metadata"
  username = "agentic_admin"
  password = random_password.db_password.result

  vpc_security_group_ids = [aws_security_group.rds.id]
  db_subnet_group_name   = aws_db_subnet_group.agentic_ai.name

  backup_retention_period = 30
  backup_window          = "03:00-04:00"
  maintenance_window     = "mon:04:00-mon:05:00"

  enabled_cloudwatch_logs_exports = ["postgresql"]

  deletion_protection = true
  skip_final_snapshot = false
  final_snapshot_identifier = "${var.cluster_name}-final-snapshot-${formatdate("YYYYMMDD-hhmmss", timestamp())}"

  tags = {
    Name = "${var.cluster_name}-metadata"
    Type = "Metadata"
  }
}

resource "random_password" "db_password" {
  length  = 32
  special = true
}

# ElastiCache for Redis
resource "aws_elasticache_subnet_group" "agentic_ai" {
  name       = "${var.cluster_name}-cache-subnet"
  subnet_ids = module.vpc.private_subnets
}

resource "aws_elasticache_parameter_group" "agentic_ai" {
  family = "redis7"
  name   = "${var.cluster_name}-redis-params"

  parameter {
    name  = "maxmemory-policy"
    value = "allkeys-lru"
  }

  parameter {
    name  = "notify-keyspace-events"
    value = "Ex"
  }
}

resource "aws_elasticache_replication_group" "agentic_ai" {
  replication_group_id       = "${var.cluster_name}-redis"
  description                = "Redis cluster for agent state and caching"
  engine                     = "redis"
  node_type                  = "cache.r6g.xlarge"
  port                       = 6379
  parameter_group_name       = aws_elasticache_parameter_group.agentic_ai.name
  automatic_failover_enabled = true
  multi_az_enabled          = true
  
  num_cache_clusters = 3
  
  subnet_group_name = aws_elasticache_subnet_group.agentic_ai.name
  security_group_ids = [aws_security_group.redis.id]

  at_rest_encryption_enabled = true
  transit_encryption_enabled = true
  auth_token                = random_password.redis_auth_token.result

  snapshot_retention_limit = 7
  snapshot_window         = "03:00-05:00"

  log_delivery_configuration {
    destination      = aws_cloudwatch_log_group.redis.name
    destination_type = "cloudwatch-logs"
    log_format       = "json"
    log_type        = "slow-log"
  }
}

resource "aws_security_group" "redis" {
  name_prefix = "${var.cluster_name}-redis"
  vpc_id      = module.vpc.vpc_id

  ingress {
    from_port   = 6379
    to_port     = 6379
    protocol    = "tcp"
    cidr_blocks = [module.vpc.vpc_cidr_block]
  }
}

resource "random_password" "redis_auth_token" {
  length  = 32
  special = false # Redis doesn't like special chars in auth tokens
}

# S3 Buckets
resource "aws_s3_bucket" "artifact_store" {
  bucket = "${var.cluster_name}-artifact-store"
}

resource "aws_s3_bucket_versioning" "artifact_store" {
  bucket = aws_s3_bucket.artifact_store.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_encryption" "artifact_store" {
  bucket = aws_s3_bucket.artifact_store.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket" "model_cache" {
  bucket = "${var.cluster_name}-model-cache"
}

resource "aws_s3_bucket_lifecycle_configuration" "model_cache" {
  bucket = aws_s3_bucket.model_cache.id

  rule {
    id     = "expire-old-models"
    status = "Enabled"

    expiration {
      days = 90
    }
  }
}

# SQS for task queue
resource "aws_sqs_queue" "task_queue" {
  name                       = "${var.cluster_name}-task-queue"
  delay_seconds              = 0
  max_message_size          = 262144
  message_retention_seconds  = 1209600 # 14 days
  receive_wait_time_seconds  = 10
  visibility_timeout_seconds = 300

  redrive_policy = jsonencode({
    deadLetterTargetArn = aws_sqs_queue.dlq.arn
    maxReceiveCount     = 3
  })

  sqs_managed_sse_enabled = true
}

resource "aws_sqs_queue" "dlq" {
  name                      = "${var.cluster_name}-dlq"
  message_retention_seconds = 1209600 # 14 days
  sqs_managed_sse_enabled = true
}

# EventBridge for event bus
resource "aws_cloudwatch_event_bus" "agentic_ai" {
  name = "${var.cluster_name}-event-bus"
}

resource "aws_cloudwatch_event_archive" "agentic_ai" {
  name             = "${var.cluster_name}-events"
  event_source_arn = aws_cloudwatch_event_bus.agentic_ai.arn
  retention_days   = 7
}

# CloudWatch Log Groups
resource "aws_cloudwatch_log_group" "cluster" {
  name              = "/aws/eks/${var.cluster_name}/cluster"
  retention_in_days = 30
}

resource "aws_cloudwatch_log_group" "redis" {
  name              = "/aws/elasticache/${var.cluster_name}"
  retention_in_days = 7
}

# Secrets Manager for storing sensitive data
resource "aws_secretsmanager_secret" "db_credentials" {
  name = "${var.cluster_name}-db-credentials"
  recovery_window_in_days = 7
}

resource "aws_secretsmanager_secret_version" "db_credentials" {
  secret_id = aws_secretsmanager_secret.db_credentials.id
  secret_string = jsonencode({
    username = aws_db_instance.metadata.username
    password = random_password.db_password.result
    host     = aws_db_instance.metadata.endpoint
    port     = aws_db_instance.metadata.port
    database = aws_db_instance.metadata.db_name
  })
}

# IAM roles for service accounts
resource "aws_iam_role" "orchestrator" {
  name = "${var.cluster_name}-orchestrator"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRoleWithWebIdentity"
        Effect = "Allow"
        Principal = {
          Federated = module.eks.oidc_provider_arn
        }
        Condition = {
          StringEquals = {
            "${replace(module.eks.cluster_oidc_issuer_url, "https://", "")}:sub": "system:serviceaccount:agentic-ai:orchestrator"
          }
        }
      }
    ]
  })
}

# Attach policies for orchestrator
resource "aws_iam_role_policy" "orchestrator" {
  role = aws_iam_role.orchestrator.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "bedrock:*",
          "sagemaker:InvokeEndpoint",
          "s3:GetObject",
          "s3:PutObject",
          "sqs:*",
          "secretsmanager:GetSecretValue",
          "kms:Decrypt"
        ]
        Resource = "*"
      }
    ]
  })
}

# Outputs
output "cluster_endpoint" {
  value = module.eks.cluster_endpoint
}

output "cluster_name" {
  value = module.eks.cluster_name
}

output "cluster_certificate" {
  value = module.eks.cluster_certificate_authority_data
}

output "db_endpoint" {
  value = aws_db_instance.metadata.endpoint
}

output "redis_endpoint" {
  value = aws_elasticache_replication_group.agentic_ai.primary_endpoint_address
}

output "task_queue_url" {
  value = aws_sqs_queue.task_queue.url
}

output "artifact_bucket" {
  value = aws_s3_bucket.artifact_store.id
}

data "aws_availability_zones" "available" {
  state = "available"
}