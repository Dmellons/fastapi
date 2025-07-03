#!/bin/bash

# deploy.sh - Production deployment script for Melston API
# This script optimizes the deployment for maximum performance

set -e

echo "🚀 Starting Melston API Production Deployment"

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "❌ Error: .env file not found. Please create one from .env.template"
    exit 1
fi

# Source environment variables
source .env

# Set default values if not provided
WORKERS=${WORKERS:-$(nproc)}
HOST=${HOST:-"0.0.0.0"}
PORT=${PORT:-8000}
LOG_LEVEL=${LOG_LEVEL:-"info"}

echo "📋 Configuration:"
echo "  Workers: $WORKERS"
echo "  Host: $HOST"
echo "  Port: $PORT"
echo "  Log Level: $LOG_LEVEL"

# Function to start with uvicorn (single worker)
start_uvicorn() {
    echo "🔧 Starting with Uvicorn (single worker)"
    uvicorn main:app \
        --host $HOST \
        --port $PORT \
        --log-level $LOG_LEVEL \
        --worker-connections 1000 \
        --backlog 2048 \
        --keepalive 2 \
        --no-access-log \
        --loop uvloop
}

# Function to start with gunicorn (multiple workers)
start_gunicorn() {
    echo "🔧 Starting with Gunicorn ($WORKERS workers)"
    gunicorn main:app \
        -w $WORKERS \
        -k uvicorn.workers.UvicornWorker \
        --bind $HOST:$PORT \
        --worker-connections 1000 \
        --backlog 2048 \
        --keepalive 2 \
        --log-level $LOG_LEVEL \
        --access-logfile - \
        --error-logfile - \
        --preload \
        --max-requests 1000 \
        --max-requests-jitter 100 \
        --timeout 30 \
        --graceful-timeout 30
}

# Function to check system resources
check_resources() {
    echo "💻 System Resources:"
    echo "  CPU Cores: $(nproc)"
    echo "  Memory: $(free -h | awk '/^Mem:/ {print $2}')"
    echo "  Load Average: $(uptime | awk -F'load average:' '{ print $2 }')"
}

# Function to optimize system settings
optimize_system() {
    echo "⚙️  Applying system optimizations..."
    
    # Increase file descriptor limits
    ulimit -n 65536
    
    # Set TCP settings for better performance
    if [ -w /proc/sys/net/core/somaxconn ]; then
        echo 65536 > /proc/sys/net/core/somaxconn
    fi
    
    if [ -w /proc/sys/net/ipv4/tcp_max_syn_backlog ]; then
        echo 65536 > /proc/sys/net/ipv4/tcp_max_syn_backlog
    fi
    
    echo "✅ System optimizations applied"
}

# Function to check dependencies
check_dependencies() {
    echo "🔍 Checking dependencies..."
    
    # Check if gunicorn is available
    if command -v gunicorn &> /dev/null; then
        echo "✅ Gunicorn found"
        return 0
    else
        echo "⚠️  Gunicorn not found, will use Uvicorn"
        return 1
    fi
}

# Function to run pre-flight checks
preflight_checks() {
    echo "🔬 Running pre-flight checks..."
    
    # Check if port is available
    if netstat -tuln | grep -q ":$PORT "; then
        echo "❌ Port $PORT is already in use"
        exit 1
    fi
    
    # Test database connection
    python -c "
import sys
sys.path.append('.')
try:
    from appwrite_config import check_appwrite_connection
    import asyncio
    result = asyncio.run(check_appwrite_connection())
    if result:
        print('✅ Database connection successful')
    else:
        print('❌ Database connection failed')
        sys.exit(1)
except Exception as e:
    print(f'❌ Database check failed: {e}')
    sys.exit(1)
"
    
    echo "✅ Pre-flight checks passed"
}

# Function to show monitoring URLs
show_monitoring() {
    echo "📊 Monitoring URLs:"
    echo "  API Documentation: http://$HOST:$PORT/api/v1/docs"
    echo "  Health Check: http://$HOST:$PORT/api/v1/health"
    echo "  System Metrics: http://$HOST:$PORT/api/v1/monitoring/system"
    echo "  Performance Metrics: http://$HOST:$PORT/api/v1/monitoring/performance"
}

# Main deployment logic
main() {
    check_resources
    
    # Only run system optimizations if running as root
    if [ "$EUID" -eq 0 ]; then
        optimize_system
    else
        echo "⚠️  Skipping system optimizations (not running as root)"
    fi
    
    preflight_checks
    
    show_monitoring
    
    # Choose deployment method based on worker count and availability
    if [ "$WORKERS" -gt 1 ] && check_dependencies; then
        start_gunicorn
    else
        start_uvicorn
    fi
}

# Handle script arguments
case "${1:-}" in
    "check")
        echo "🔍 Running health checks only..."
        check_resources
        preflight_checks
        echo "✅ All checks passed"
        ;;
    "dev")
        echo "🔧 Starting in development mode..."
        ENV=development uvicorn main:app --host 127.0.0.1 --port $PORT --reload --log-level debug
        ;;
    "prod"|"")
        echo "🚀 Starting in production mode..."
        main
        ;;
    *)
        echo "Usage: $0 [check|dev|prod]"
        echo "  check - Run health checks only"
        echo "  dev   - Start in development mode"
        echo "  prod  - Start in production mode (default)"
        exit 1
        ;;
esac