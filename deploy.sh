




set -e

echo "🚀 Starting Industrial Pollution Intelligence System Deployment"
echo "================================================================"


RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
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

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}


check_dependencies() {
    print_status "Checking dependencies..."


    if ! command -v psql-17 &> /dev/null; then
        print_error "PostgreSQL-17 not found. Please install PostgreSQL-17 first."
        exit 1
    fi


    if ! command -v docker &> /dev/null; then
        print_warning "Docker not found. Will use local deployment."
        USE_DOCKER=false
    else
        USE_DOCKER=true
        print_success "Docker found"
    fi


    if ! command -v python3 &> /dev/null; then
        print_error "Python3 not found. Please install Python3 first."
        exit 1
    fi


    if ! command -v node &> /dev/null; then
        print_error "Node.js not found. Please install Node.js first."
        exit 1
    fi


    if ! command -v cargo &> /dev/null; then
        print_error "Rust not found. Please install Rust first."
        exit 1
    fi

    print_success "All dependencies found"
}


setup_environment() {
    print_status "Setting up environment..."


    if [ ! -f .env ]; then
        cp env.example .env
        print_success "Created .env file from template"
    else
        print_warning ".env file already exists, skipping..."
    fi


    mkdir -p data models logs
    print_success "Created necessary directories"
}


setup_database() {
    print_status "Setting up database..."


    if ! pg_isready -h localhost -p 5432 &> /dev/null; then
        print_error "PostgreSQL is not running. Please start PostgreSQL first."
        exit 1
    fi


    if ! psql-17 -h localhost -p 5432 -U postgres -lqt | cut -d \| -f 1 | grep -qw pollution_db; then
        print_status "Creating database..."
        psql-17 -h localhost -p 5432 -U postgres -c "CREATE DATABASE pollution_db;"
        psql-17 -h localhost -p 5432 -U postgres -c "CREATE USER pollution_user WITH PASSWORD 'pollution_pass';"
        psql-17 -h localhost -p 5432 -U postgres -c "GRANT ALL PRIVILEGES ON DATABASE pollution_db TO pollution_user;"
        print_success "Database created successfully"
    else
        print_warning "Database already exists, skipping..."
    fi


    print_status "Running database migrations..."
    psql-17 -h localhost -p 5432 -U pollution_user -d pollution_db -f migrations/001_initial_schema.sql
    print_success "Database migrations completed"
}


setup_python() {
    print_status "Setting up Python environment..."

    cd python


    if [ ! -d "venv" ]; then
        python3 -m venv venv
        print_success "Created Python virtual environment"
    fi


    source venv/bin/activate


    pip install --upgrade pip
    pip install -e .

    cd ..
    print_success "Python environment setup completed"
}


setup_rust() {
    print_status "Setting up Rust backend..."

    cd rust-backend


    cargo build --release

    cd ..
    print_success "Rust backend setup completed"
}


setup_frontend() {
    print_status "Setting up frontend..."

    cd frontend


    npm install


    npm run build

    cd ..
    print_success "Frontend setup completed"
}


start_services() {
    print_status "Starting services..."

    if [ "$USE_DOCKER" = true ]; then
        print_status "Starting services with Docker..."
        docker-compose up -d
        print_success "Services started with Docker"
    else
        print_status "Starting services locally..."


        cd rust-backend
        cargo run --release &
        BACKEND_PID=$!
        cd ..


        cd frontend
        npm run preview -- --host 0.0.0.0 &
        FRONTEND_PID=$!
        cd ..

        print_success "Services started locally"
        print_status "Backend PID: $BACKEND_PID"
        print_status "Frontend PID: $FRONTEND_PID"


        echo $BACKEND_PID > .backend.pid
        echo $FRONTEND_PID > .frontend.pid
    fi
}


run_data_collection() {
    print_status "Running data collection..."

    cd python
    source venv/bin/activate


    python -c "
import sys
sys.path.append('.')
from scraper.cnemc_collector import CNEMCCollector
import asyncio

async def main():
    collector = CNEMCCollector()
    await collector.collect_data(area_ids=[110000, 310000, 440000])
    print('Data collection completed')

asyncio.run(main())
"

    cd ..
    print_success "Data collection completed"
}


main() {
    echo "================================================================"
    echo "🏭 Industrial Pollution Intelligence System"
    echo "================================================================"

    check_dependencies
    setup_environment
    setup_database
    setup_python
    setup_rust
    setup_frontend

    echo ""
    print_status "Deployment completed successfully!"
    print_status "Starting services..."

    start_services


    sleep 5

    echo ""
    print_success "🎉 System is ready!"
    echo ""
    echo "📊 Frontend: http://localhost:3000"
    echo "🔧 Backend API: http://localhost:8080"
    echo "🗄️  Database: localhost:5432/pollution_db"
    echo ""


    read -p "Do you want to run data collection now? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        run_data_collection
    fi

    echo ""
    print_status "To stop the services, run: ./stop.sh"
    print_status "To view logs, run: ./logs.sh"
}


main "$@"

