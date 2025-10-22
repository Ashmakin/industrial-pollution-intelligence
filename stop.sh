#!/bin/bash

# Stop Industrial Pollution Intelligence System

echo "🛑 Stopping Industrial Pollution Intelligence System..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Stop Docker services if running
if docker-compose ps | grep -q "Up"; then
    print_status "Stopping Docker services..."
    docker-compose down
    print_success "Docker services stopped"
fi

# Stop local services
if [ -f ".backend.pid" ]; then
    BACKEND_PID=$(cat .backend.pid)
    if kill -0 $BACKEND_PID 2>/dev/null; then
        print_status "Stopping backend service (PID: $BACKEND_PID)..."
        kill $BACKEND_PID
        rm .backend.pid
        print_success "Backend service stopped"
    fi
fi

if [ -f ".frontend.pid" ]; then
    FRONTEND_PID=$(cat .frontend.pid)
    if kill -0 $FRONTEND_PID 2>/dev/null; then
        print_status "Stopping frontend service (PID: $FRONTEND_PID)..."
        kill $FRONTEND_PID
        rm .frontend.pid
        print_success "Frontend service stopped"
    fi
fi

print_success "All services stopped successfully!"

