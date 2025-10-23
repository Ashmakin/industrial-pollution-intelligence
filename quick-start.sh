




set -e

echo "⚡ Quick Start - Industrial Pollution Intelligence System"
echo "========================================================"


GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}


check_postgres() {
    if ! pg_isready -h localhost -p 5432 &> /dev/null; then
        print_warning "PostgreSQL is not running. Please start PostgreSQL first:"
        echo "  brew services start postgresql@17"
        echo "  or"
        echo "  sudo systemctl start postgresql"
        exit 1
    fi
    print_success "PostgreSQL is running"
}


quick_db_setup() {
    print_status "Quick database setup..."


    if ! psql-17 -h localhost -p 5432 -U postgres -lqt | cut -d \| -f 1 | grep -qw pollution_db; then
        psql-17 -h localhost -p 5432 -U postgres -c "CREATE DATABASE pollution_db;"
        psql-17 -h localhost -p 5432 -U postgres -c "CREATE USER pollution_user WITH PASSWORD 'pollution_pass';"
        psql-17 -h localhost -p 5432 -U postgres -c "GRANT ALL PRIVILEGES ON DATABASE pollution_db TO pollution_user;"
        psql-17 -h localhost -p 5432 -U pollution_user -d pollution_db -f migrations/001_initial_schema.sql
        print_success "Database created and migrated"
    else
        print_warning "Database already exists, skipping..."
    fi
}


start_with_docker() {
    print_status "Starting services with Docker..."

    if command -v docker &> /dev/null && command -v docker-compose &> /dev/null; then
        docker-compose up -d
        print_success "Services started with Docker"


        print_status "Waiting for services to be ready..."
        sleep 10


        if curl -s http://localhost:8080/health > /dev/null; then
            print_success "Backend is ready"
        else
            print_warning "Backend may still be starting..."
        fi

        if curl -s http://localhost:3000 > /dev/null; then
            print_success "Frontend is ready"
        else
            print_warning "Frontend may still be starting..."
        fi

    else
        print_warning "Docker not available. Please install Docker and Docker Compose."
        exit 1
    fi
}


main() {
    check_postgres
    quick_db_setup
    start_with_docker

    echo ""
    print_success "🎉 System is ready!"
    echo ""
    echo "📊 Frontend: http://localhost:3000"
    echo "🔧 Backend API: http://localhost:8080"
    echo "🗄️  Database: localhost:5432/pollution_db"
    echo ""
    echo "📋 Useful commands:"
    echo "  ./logs.sh          - View logs"
    echo "  ./stop.sh          - Stop services"
    echo "  docker-compose logs -f - View real-time logs"
    echo ""


    if command -v open &> /dev/null; then
        open http://localhost:3000
    elif command -v xdg-open &> /dev/null; then
        xdg-open http://localhost:3000
    fi
}

main "$@"

