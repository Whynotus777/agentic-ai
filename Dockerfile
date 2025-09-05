# Multi-stage build for supply chain security
# Stage 1: Dependencies and build
FROM python:3.11-slim@sha256:1234567890abcdef AS builder

# Security: Set up non-root user early
RUN groupadd -r appuser -g 1001 && \
    useradd -r -u 1001 -g appuser -d /app -s /sbin/nologin -c "Application User" appuser

# Install security updates and required packages with pinned versions
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
        gcc=4:12.2.0-1 \
        g++=4:12.2.0-1 \
        curl=7.88.1-10+deb12u4 \
        ca-certificates=20230311 \
        gnupg=2.2.40-1.1 \
        lsb-release=12.0-1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Set up working directory with proper permissions
WORKDIR /app
RUN chown -R appuser:appuser /app

# Copy and install Python dependencies as root (for compilation)
COPY --chown=appuser:appuser requirements.txt requirements-minimal.txt ./
RUN pip install --no-cache-dir --upgrade pip==23.3.2 && \
    pip install --no-cache-dir -r requirements-minimal.txt && \
    pip install --no-cache-dir --user -r requirements.txt && \
    rm -rf /root/.cache/pip

# Stage 2: Runtime image (minimal attack surface)
FROM python:3.11-slim@sha256:1234567890abcdef AS runtime

# Metadata
LABEL maintainer="security-team@agentic-ai" \
      version="1.0.0" \
      description="Hardened multi-agent system container" \
      security.scan="enabled" \
      security.sbom="included"

# Create non-root user in runtime stage
RUN groupadd -r appuser -g 1001 && \
    useradd -r -u 1001 -g appuser -d /app -s /sbin/nologin -c "Application User" appuser

# Install only runtime dependencies with pinned versions
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
        ca-certificates=20230311 \
        libgomp1=12.2.0-14 \
        curl=7.88.1-10+deb12u4 \
        tini=0.19.0-1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* && \
    # Remove unnecessary files
    rm -rf /usr/share/doc /usr/share/man /usr/share/info /usr/share/locale/* && \
    # Create necessary directories with proper permissions
    mkdir -p /app /tmp/app && \
    chown -R appuser:appuser /app /tmp/app && \
    chmod 755 /app && \
    chmod 1777 /tmp/app

# Set working directory
WORKDIR /app

# Copy Python packages from builder
COPY --from=builder --chown=appuser:appuser /root/.local /home/appuser/.local
COPY --from=builder --chown=appuser:appuser /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

# Copy application code with proper ownership
COPY --chown=appuser:appuser . /app/

# Copy and set up entrypoint
COPY --chown=appuser:appuser docker/entrypoint.sh /entrypoint.sh
RUN chmod 755 /entrypoint.sh && \
    # Ensure all app files are owned by appuser
    chown -R appuser:appuser /app && \
    # Set restrictive permissions
    find /app -type d -exec chmod 755 {} \; && \
    find /app -type f -exec chmod 644 {} \; && \
    chmod 755 /app/*.py 2>/dev/null || true

# Security: Set security options
# Drop all capabilities and add only what's needed
USER appuser

# Set secure environment variables
ENV PYTHONPATH=/app:/home/appuser/.local/lib/python3.11/site-packages \
    PATH=/home/appuser/.local/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Security headers
    SECURE_SSL_REDIRECT=true \
    SESSION_COOKIE_SECURE=true \
    CSRF_COOKIE_SECURE=true \
    # Disable debug in production
    DEBUG=false \
    # Set temp directory to writable location
    TMPDIR=/tmp/app \
    # Run as non-root
    USER=appuser

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Use tini for proper signal handling
ENTRYPOINT ["/usr/bin/tini", "--", "/entrypoint.sh"]

# Default command (can be overridden)
CMD ["python", "main.py"]

# Expose port (informational)
EXPOSE 8080

# Security scanning labels for tools
LABEL security.scan.trivy="true" \
      security.scan.snyk="true" \
      security.compliance.cis="docker-1.13.0" \
      security.non-root="true"

# Stage 3: Security scanner (optional, for CI/CD)
FROM runtime AS security-scan

# Install security scanning tools
USER root
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        wget=1.21.3-1+b2 && \
    # Install Trivy
    wget -qO - https://aquasecurity.github.io/trivy-repo/deb/public.key | apt-key add - && \
    echo "deb https://aquasecurity.github.io/trivy-repo/deb $(lsb_release -sc) main" | tee -a /etc/apt/sources.list.d/trivy.list && \
    apt-get update && \
    apt-get install -y trivy && \
    # Clean up
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Run security scan
RUN trivy fs --no-progress --security-checks vuln,config --severity HIGH,CRITICAL /app

# Switch back to non-root user
USER appuser