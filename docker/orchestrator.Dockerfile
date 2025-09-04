# docker/orchestrator.Dockerfile - Orchestrator Service
FROM python:3.11-slim as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements/orchestrator.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --user -r orchestrator.txt

# Runtime stage
FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /root/.local /root/.local

# Copy application code
COPY orchestrator/ ./orchestrator/
COPY common/ ./common/
COPY config/ ./config/

# Create non-root user
RUN useradd -m -u 1000 agentic && \
    chown -R agentic:agentic /app

USER agentic

# Ensure scripts are in path
ENV PATH=/root/.local/bin:$PATH

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8080/health || exit 1

EXPOSE 8080 9090

CMD ["python", "-m", "orchestrator.main"]

---
# docker/agent-cpu.Dockerfile - CPU Agent Service
FROM python:3.11-slim as builder

WORKDIR /build

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    && rm -rf /var/lib/apt/lists/*

COPY requirements/agent.txt .
RUN pip install --no-cache-dir --user -r agent.txt

# Download models
RUN python -c "from transformers import AutoModel, AutoTokenizer; \
    AutoModel.from_pretrained('microsoft/codebert-base'); \
    AutoTokenizer.from_pretrained('microsoft/codebert-base')"

FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /root/.local /root/.local
COPY --from=builder /root/.cache /root/.cache

COPY agent/ ./agent/
COPY common/ ./common/

RUN useradd -m -u 1000 agentic && \
    chown -R agentic:agentic /app

USER agentic

ENV PATH=/root/.local/bin:$PATH
ENV TRANSFORMERS_OFFLINE=1

EXPOSE 8081

CMD ["python", "-m", "agent.main", "--type", "cpu"]

---
# docker/agent-gpu.Dockerfile - GPU Agent Service
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04 as builder

WORKDIR /build

RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    gcc \
    g++ \
    make \
    && rm -rf /var/lib/apt/lists/*

COPY requirements/agent-gpu.txt .
RUN python3.11 -m pip install --no-cache-dir --user -r agent-gpu.txt

# Download GPU-optimized models
RUN python3.11 -c "from transformers import AutoModelForCausalLM; \
    AutoModelForCausalLM.from_pretrained('TheBloke/Yi-34B-Coder-GPTQ', \
        device_map='auto', trust_remote_code=True)"

FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-distutils \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /root/.local /root/.local
COPY --from=builder /root/.cache /root/.cache

COPY agent/ ./agent/
COPY common/ ./common/

RUN useradd -m -u 1000 agentic && \
    chown -R agentic:agentic /app

USER agentic

ENV PATH=/root/.local/bin:$PATH
ENV TRANSFORMERS_OFFLINE=1
ENV CUDA_VISIBLE_DEVICES=0

EXPOSE 8081

CMD ["python3.11", "-m", "agent.main", "--type", "gpu"]

---
# docker/api-gateway.Dockerfile - API Gateway
FROM python:3.11-slim as builder

WORKDIR /build

RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements/api.txt .
RUN pip install --no-cache-dir --user -r api.txt

FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /root/.local /root/.local

COPY api/ ./api/
COPY common/ ./common/

RUN useradd -m -u 1000 agentic && \
    chown -R agentic:agentic /app

USER agentic

ENV PATH=/root/.local/bin:$PATH

HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000 8443

CMD ["uvicorn", "api.gateway:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]

---
# docker/task-executor.Dockerfile - Task Executor
FROM python:3.11-slim as builder

WORKDIR /build

RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements/executor.txt .
RUN pip install --no-cache-dir --user -r executor.txt

FROM python:3.11-slim

WORKDIR /app

COPY --from=builder /root/.local /root/.local

COPY execution/ ./execution/
COPY common/ ./common/

# Create sandbox directory
RUN mkdir -p /sandbox && \
    useradd -m -u 1000 agentic && \
    chown -R agentic:agentic /app /sandbox

USER agentic

ENV PATH=/root/.local/bin:$PATH

CMD ["python", "-m", "execution.worker"]

---
# docker/control-plane.Dockerfile - Control Plane Services
FROM golang:1.21-alpine as go-builder

WORKDIR /build

# Copy Go modules
COPY control-plane/go.mod control-plane/go.sum ./
RUN go mod download

# Copy source code
COPY control-plane/ .

# Build binaries
RUN CGO_ENABLED=0 GOOS=linux go build -o policy-engine cmd/policy/main.go
RUN CGO_ENABLED=0 GOOS=linux go build -o model-router cmd/router/main.go
RUN CGO_ENABLED=0 GOOS=linux go build -o capability-registry cmd/registry/main.go

FROM alpine:3.18

RUN apk add --no-cache ca-certificates

WORKDIR /app

# Copy binaries from builder
COPY --from=go-builder /build/policy-engine .
COPY --from=go-builder /build/model-router .
COPY --from=go-builder /build/capability-registry .

# Copy configuration
COPY control-plane/configs/ ./configs/

# Create non-root user
RUN adduser -D -u 1000 agentic && \
    chown -R agentic:agentic /app

USER agentic

# Use shell script to run the appropriate service
COPY docker/control-plane-entrypoint.sh .
RUN chmod +x control-plane-entrypoint.sh

EXPOSE 8081 8082 8083

ENTRYPOINT ["./control-plane-entrypoint.sh"]

---
# docker/monitoring.Dockerfile - Monitoring Stack
FROM prom/prometheus:latest as prometheus

COPY monitoring/prometheus.yml /etc/prometheus/prometheus.yml
COPY monitoring/alerts.yml /etc/prometheus/alerts.yml

---
FROM grafana/grafana:latest as grafana

COPY monitoring/dashboards/ /etc/grafana/provisioning/dashboards/
COPY monitoring/datasources.yml /etc/grafana/provisioning/datasources/

---
FROM elastic/apm-server:8.10.0 as apm

COPY monitoring/apm-server.yml /usr/share/apm-server/apm-server.yml

---
# docker-compose.yml - Local Development Stack
version: '3.8'

services:
  # Infrastructure
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: agentic
      POSTGRES_USER: agentic_user
      POSTGRES_PASSWORD: agentic_pass
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U agentic_user"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes --requirepass redis_password
    volumes:
      - redis_data:/data
    ports:
      - "6379:6379"
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

  nats:
    image: nats:2.10-alpine
    command: 
      - --jetstream
      - --store_dir=/data
      - --cluster_name=agentic-ai
    volumes:
      - nats_data:/data
    ports:
      - "4222:4222"
      - "8222:8222"
    healthcheck:
      test: ["CMD", "nc", "-z", "localhost", "4222"]
      interval: 10s
      timeout: 5s
      retries: 5

  # Core Services
  orchestrator:
    build:
      context: .
      dockerfile: docker/orchestrator.Dockerfile
    environment:
      DATABASE_URL: postgresql://agentic_user:agentic_pass@postgres:5432/agentic
      REDIS_URL: redis://:redis_password@redis:6379
      NATS_URL: nats://nats:4222
    depends_on:
      - postgres
      - redis
      - nats
    ports:
      - "8080:8080"
      - "9090:9090"
    deploy:
      replicas: 2
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G

  api-gateway:
    build:
      context: .
      dockerfile: docker/api-gateway.Dockerfile
    environment:
      DATABASE_URL: postgresql://agentic_user:agentic_pass@postgres:5432/agentic
      REDIS_URL: redis://:redis_password@redis:6379
      ORCHESTRATOR_URL: http://orchestrator:8080
    depends_on:
      - orchestrator
    ports:
      - "8000:8000"
    deploy:
      replicas: 2
      resources:
        limits:
          cpus: '1'
          memory: 2G

  agent-cpu:
    build:
      context: .
      dockerfile: docker/agent-cpu.Dockerfile
    environment:
      REDIS_URL: redis://:redis_password@redis:6379
      NATS_URL: nats://nats:4222
    depends_on:
      - redis
      - nats
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: '2'
          memory: 4G

  task-executor:
    build:
      context: .
      dockerfile: docker/task-executor.Dockerfile
    environment:
      REDIS_URL: redis://:redis_password@redis:6379
      NATS_URL: nats://nats:4222
    depends_on:
      - redis
      - nats
    deploy:
      replicas: 5
      resources:
        limits:
          cpus: '1'
          memory: 2G

  control-plane:
    build:
      context: .
      dockerfile: docker/control-plane.Dockerfile
    environment:
      SERVICE: all
      REDIS_URL: redis://:redis_password@redis:6379
    depends_on:
      - redis
    ports:
      - "8081:8081"
      - "8082:8082"
      - "8083:8083"

  # Monitoring
  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    ports:
      - "9091:9090"
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'

  grafana:
    image: grafana/grafana:latest
    environment:
      GF_SECURITY_ADMIN_PASSWORD: admin
    volumes:
      - ./monitoring/dashboards:/etc/grafana/provisioning/dashboards
      - grafana_data:/var/lib/grafana
    ports:
      - "3000:3000"
    depends_on:
      - prometheus

  jaeger:
    image: jaegertracing/all-in-one:latest
    environment:
      COLLECTOR_OTLP_ENABLED: true
    ports:
      - "16686:16686"
      - "14268:14268"
      - "4317:4317"

volumes:
  postgres_data:
  redis_data:
  nats_data:
  prometheus_data:
  grafana_data:

---
# docker/control-plane-entrypoint.sh
#!/bin/sh
set -e

SERVICE=${SERVICE:-all}

case "$SERVICE" in
    policy)
        exec ./policy-engine
        ;;
    router)
        exec ./model-router
        ;;
    registry)
        exec ./capability-registry
        ;;
    all)
        # Run all services (development mode)
        ./policy-engine &
        ./model-router &
        exec ./capability-registry
        ;;
    *)
        echo "Unknown service: $SERVICE"
        exit 1
        ;;
esac

---
# .dockerignore
.git
.gitignore
*.pyc
__pycache__
.pytest_cache
.coverage
.env
.venv
venv/
*.log
*.pid
*.seed
*.swp
.DS_Store
terraform/
kubernetes/
docs/
tests/
README.md
LICENSE
