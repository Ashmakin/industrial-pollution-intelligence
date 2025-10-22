#!/bin/bash

# View logs for Industrial Pollution Intelligence System

echo "📋 Industrial Pollution Intelligence System Logs"
echo "================================================"

# Colors for output
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

# Check if Docker services are running
if docker-compose ps | grep -q "Up"; then
    print_status "Showing Docker logs..."
    echo ""
    echo "🔧 Backend logs:"
    docker-compose logs rust-backend
    echo ""
    echo "🌐 Frontend logs:"
    docker-compose logs frontend
    echo ""
    echo "🗄️ Database logs:"
    docker-compose logs postgres
else
    print_status "Docker services not running. Checking local logs..."
    
    # Check if log files exist
    if [ -f "logs/backend.log" ]; then
        echo ""
        echo "🔧 Backend logs:"
        tail -50 logs/backend.log
    fi
    
    if [ -f "logs/frontend.log" ]; then
        echo ""
        echo "🌐 Frontend logs:"
        tail -50 logs/frontend.log
    fi
    
    if [ -f "logs/python.log" ]; then
        echo ""
        echo "🐍 Python logs:"
        tail -50 logs/python.log
    fi
    
    echo ""
    print_status "To view real-time logs, use: docker-compose logs -f"
fi

