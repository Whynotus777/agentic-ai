#!/bin/bash
# Hardened entrypoint script with security best practices
# Set strict error handling: -e (exit on error), -u (undefined vars), -o pipefail (pipe failures)
set -euo pipefail

# Enable additional security features
set -o noclobber  # Prevent file overwrite with >
set -o errtrace   # Inherit trap on ERR
set -o functrace  # Inherit trap on DEBUG and RETURN

# Security: Set IFS to prevent word splitting attacks
IFS=$'\n\t'

# Trap errors and perform cleanup
trap 'echo "Error occurred at line $LINENO with exit code $?"; cleanup_on_exit' ERR
trap 'cleanup_on_exit' EXIT SIGINT SIGTERM

# Cleanup function
cleanup_on_exit() {
    local exit_code=$?
    if [ ${exit_code} -ne 0 ]; then
        echo "[ERROR] Entrypoint script failed with exit code: ${exit_code}" >&2
    fi
    # Add any cleanup operations here
    return ${exit_code}
}

# Logging functions with timestamps
log_info() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] $*"
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [ERROR] $*" >&2
}

log_warn() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [WARN] $*"
}

# Security sanity checks
perform_security_checks() {
    log_info "Performing security sanity checks..."
    
    # Check if running as non-root
    if [ "$(id -u)" -eq 0 ]; then
        log_error "Container is running as root! This is a security risk."
        log_error "Expected to run as user 'appuser' (UID 1001)"
        exit 1
    fi
    
    # Verify running as expected user
    EXPECTED_USER="appuser"
    CURRENT_USER=$(whoami)
    if [ "${CURRENT_USER}" != "${EXPECTED_USER}" ]; then
        log_warn "Running as user '${CURRENT_USER}' instead of expected '${EXPECTED_USER}'"
    else
        log_info "Running as non-root user: ${CURRENT_USER} (UID: $(id -u))"
    fi
    
    # Check write permissions for necessary directories
    if [ ! -w "/tmp/app" ]; then
        log_error "No write permission for /tmp/app directory"
        exit 1
    fi
    
    # Verify critical files exist
    REQUIRED_FILES=(
        "/app/main.py"
        "/app/requirements.txt"
    )
    
    for file in "${REQUIRED_FILES[@]}"; do
        if [ ! -f "${file}" ]; then
            log_error "Required file missing: ${file}"
            exit 1
        fi
    done
    
    log_info "Security checks passed"
}

# Environment validation
validate_environment() {
    log_info "Validating environment variables..."
    
    # Set defaults for optional environment variables
    export LOG_LEVEL="${LOG_LEVEL:-INFO}"
    export APP_PORT="${APP_PORT:-8080}"
    export WORKERS="${WORKERS:-1}"
    export MAX_REQUESTS="${MAX_REQUESTS:-1000}"
    export TIMEOUT="${TIMEOUT:-30}"
    
    # Validate required environment variables
    REQUIRED_VARS=()  # Add any required vars here, e.g., ("API_KEY" "DB_URL")
    
    for var in "${REQUIRED_VARS[@]}"; do
        if [ -z "${!var:-}" ]; then
            log_error "Required environment variable '${var}' is not set"
            exit 1
        fi
    done
    
    # Validate port number
    if ! [[ "${APP_PORT}" =~ ^[0-9]+$ ]] || [ "${APP_PORT}" -lt 1 ] || [ "${APP_PORT}" -gt 65535 ]; then
        log_error "Invalid port number: ${APP_PORT}"
        exit 1
    fi
    
    # Security: Ensure sensitive variables are not logged
    SENSITIVE_VARS=("API_KEY" "SECRET_KEY" "PASSWORD" "TOKEN")
    for var in "${SENSITIVE_VARS[@]}"; do
        if [ -n "${!var:-}" ]; then
            log_info "Environment variable '${var}' is set (value hidden)"
        fi
    done
    
    log_info "Environment validation completed"
}

# Initialize application
initialize_app() {
    log_info "Initializing application..."
    
    # Create necessary runtime directories
    mkdir -p /tmp/app/{logs,cache,uploads}
    
    # Set secure permissions
    chmod 700 /tmp/app/{logs,cache,uploads}
    
    # Export Python path
    export PYTHONPATH="${PYTHONPATH:-/app}"
    
    # Check Python installation
    if ! command -v python3 &> /dev/null; then
        log_error "Python3 is not installed or not in PATH"
        exit 1
    fi
    
    PYTHON_VERSION=$(python3 --version 2>&1)
    log_info "Python version: ${PYTHON_VERSION}"
    
    # Run any initialization scripts
    if [ -f "/app/init.py" ]; then
        log_info "Running initialization script..."
        python3 /app/init.py || {
            log_error "Initialization script failed"
            exit 1
        }
    fi
    
    log_info "Application initialized successfully"
}

# Health check function
health_check() {
    # Simple health check - can be extended
    if [ -f "/tmp/app/.healthy" ]; then
        return 0
    else
        # Create health file on first run
        touch /tmp/app/.healthy
        return 0
    fi
}

# Main execution
main() {
    log_info "Starting agentic-ai container..."
    log_info "Container version: ${VERSION:-unknown}"
    log_info "Build date: ${BUILD_DATE:-unknown}"
    
    # Run security checks
    perform_security_checks
    
    # Validate environment
    validate_environment
    
    # Initialize application
    initialize_app
    
    # Run health check
    if health_check; then
        log_info "Health check passed"
    else
        log_warn "Health check failed, continuing anyway"
    fi
    
    # Handle different command scenarios
    if [ $# -eq 0 ]; then
        # No arguments, run default command
        log_info "Starting application with default command..."
        exec python3 /app/main.py
    else
        # Execute provided command
        log_info "Executing command: $*"
        exec "$@"
    fi
}

# Signal handlers
handle_sigterm() {
    log_info "Received SIGTERM, initiating graceful shutdown..."
    # Add graceful shutdown logic here
    exit 0
}

handle_sigint() {
    log_info "Received SIGINT, initiating graceful shutdown..."
    # Add graceful shutdown logic here  
    exit 0
}

# Register signal handlers
trap handle_sigterm SIGTERM
trap handle_sigint SIGINT

# Run main function with all arguments
main "$@"