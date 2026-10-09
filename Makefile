.PHONY: help install setup-docker setup-neo4j setup-env start stop restart status verify clean test test-e2e test-smoke test-e2e-quick lint format

# Default target
help:
	@echo "╔════════════════════════════════════════════════════════════╗"
	@echo "║          Kosmos AI Scientist - Make Targets               ║"
	@echo "╚════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "Setup Commands:"
	@echo "  make install        - Complete environment setup (Python, deps, config)"
	@echo "  make setup-docker   - Install Docker on WSL2"
	@echo "  make setup-neo4j    - Setup and start Neo4j container"
	@echo "  make setup-env      - Setup Python environment only"
	@echo ""
	@echo "Service Management:"
	@echo "  make start          - Start all services (dev profile)"
	@echo "  make start-prod     - Start all services (production profile)"
	@echo "  make stop           - Stop all services"
	@echo "  make restart        - Restart all services"
	@echo "  make status         - Show status of all containers"
	@echo ""
	@echo "Development:"
	@echo "  make verify         - Run deployment verification checks"
	@echo "  make test           - Run test suite"
	@echo "  make test-unit      - Run unit tests only"
	@echo "  make test-int       - Run integration tests (requires services)"
	@echo "  make lint           - Run code linters"
	@echo "  make format         - Format code with black/isort"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean          - Remove caches and temporary files"
	@echo "  make clean-all      - Remove caches, temp files, and venv"
	@echo "  make logs           - View logs from all services"
	@echo "  make logs-neo4j     - View Neo4j logs only"
	@echo ""
	@echo "Database:"
	@echo "  make db-migrate     - Run database migrations"
	@echo "  make db-reset       - Reset database (DESTRUCTIVE)"
	@echo ""
	@echo "Quick Start:"
	@echo "  1. make setup-docker    # Install Docker (one-time)"
	@echo "  2. make install         # Setup environment"
	@echo "  3. make start           # Start services"
	@echo "  4. make verify          # Verify everything works"
	@echo ""

#==============================================================================
# Setup Targets
#==============================================================================

install:
	@echo "📦 Setting up Kosmos development environment..."
	@./scripts/setup_environment.sh

setup-docker:
	@echo "🐳 Installing Docker on WSL2..."
	@./scripts/setup_docker_wsl2.sh

setup-neo4j:
	@echo "🔷 Setting up Neo4j..."
	@./scripts/setup_neo4j.sh

setup-env:
	@echo "🐍 Setting up Python environment..."
	@./scripts/setup_environment.sh

#==============================================================================
# Service Management
#==============================================================================

start:
	@echo "🚀 Starting Kosmos services (dev profile)..."
	@docker compose --profile dev up -d
	@echo "✓ Services started"
	@make status

start-prod:
	@echo "🚀 Starting Kosmos services (production profile)..."
	@docker compose --profile prod up -d
	@echo "✓ Services started"
	@make status

stop:
	@echo "🛑 Stopping all services..."
	@docker compose down
	@echo "✓ Services stopped"

restart:
	@echo "🔄 Restarting services..."
	@docker compose restart
	@echo "✓ Services restarted"
	@make status

status:
	@echo "📊 Service Status:"
	@echo "════════════════════════════════════════════════════════════"
	@docker compose ps || echo "No services running"
	@echo ""

#==============================================================================
# Development & Testing
#==============================================================================

verify:
	@echo "🔍 Running deployment verification..."
	@./scripts/verify_deployment.sh

test:
	@echo "🧪 Running test suite..."
	@python -m pytest tests/unit tests/integration --no-cov -p no:cacheprovider -q

test-unit:
	@echo "🧪 Running unit tests..."
	@python -m pytest tests/unit --no-cov -p no:cacheprovider -q

test-int:
	@echo "🧪 Running integration tests..."
	@python -m pytest tests/integration --no-cov -p no:cacheprovider -q

test-cov:
	@echo "🧪 Running tests with coverage..."
	@python -m pytest tests/unit tests/integration -p no:cacheprovider --cov=kosmos --cov-report=html --cov-report=term
	@echo "📄 Coverage report generated in htmlcov/index.html"

test-e2e:
	@echo "🧪 Running E2E tests..."
	@python -m pytest tests/e2e/ -v --no-cov -p no:cacheprovider

test-smoke:
	@echo "🧪 Running smoke tests..."
	@python -m pytest tests/e2e/ -m smoke -v --no-cov -p no:cacheprovider

test-e2e-quick:
	@echo "🧪 Running quick E2E tests (excluding slow)..."
	@python -m pytest tests/e2e/ -m "not slow" -v --no-cov -p no:cacheprovider

lint:
	@echo "🔍 Running linters..."
	@echo "Running pylint..."
	@pylint kosmos/ || true
	@echo "Running mypy..."
	@mypy kosmos/ --ignore-missing-imports || true
	@echo "Running flake8..."
	@flake8 kosmos/ --max-line-length=120 || true

format:
	@echo "✨ Formatting code..."
	@echo "Running black..."
	@black kosmos/ tests/
	@echo "Running isort..."
	@isort kosmos/ tests/
	@echo "✓ Code formatted"

#==============================================================================
# Logs
#==============================================================================

logs:
	@echo "📜 Viewing logs from all services..."
	@docker compose logs -f

logs-neo4j:
	@echo "📜 Viewing Neo4j logs..."
	@docker compose logs -f neo4j

logs-postgres:
	@echo "📜 Viewing PostgreSQL logs..."
	@docker compose logs -f postgres

logs-redis:
	@echo "📜 Viewing Redis logs..."
	@docker compose logs -f redis

#==============================================================================
# Database Management
#==============================================================================

db-migrate:
	@echo "🗄️  Running database migrations..."
	@alembic upgrade head
	@echo "✓ Migrations complete"

db-reset:
	@echo "⚠️  WARNING: This will delete all data!"
	@read -p "Are you sure? (yes/NO): " confirm && [ "$$confirm" = "yes" ] || exit 1
	@echo "Stopping services..."
	@docker compose down
	@echo "Removing database volumes..."
	@rm -rf postgres_data/ neo4j_data/
	@echo "Recreating volumes..."
	@mkdir -p postgres_data neo4j_data
	@echo "Starting services..."
	@docker compose up -d
	@echo "Running migrations..."
	@sleep 5
	@alembic upgrade head
	@echo "✓ Database reset complete"

#==============================================================================
# Maintenance
#==============================================================================

clean:
	@echo "🧹 Cleaning caches and temporary files..."
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type f -name "*.pyo" -delete 2>/dev/null || true
	@find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name ".coverage" -delete 2>/dev/null || true
	@rm -rf htmlcov/ 2>/dev/null || true
	@echo "✓ Cleanup complete"

clean-all: clean
	@echo "🧹 Deep cleaning (including venv)..."
	@read -p "Remove virtual environment? (y/N): " confirm && [ "$$confirm" = "y" ] || exit 1
	@rm -rf venv/
	@echo "✓ Deep cleanup complete"

#==============================================================================
# Utility Targets
#==============================================================================

shell:
	@echo "🐚 Starting shell in kosmos container..."
	@docker exec -it kosmos-app /bin/bash

shell-neo4j:
	@echo "🐚 Starting Cypher shell..."
	@docker exec -it kosmos-neo4j cypher-shell -u neo4j -p kosmos-password

shell-postgres:
	@echo "🐚 Starting PostgreSQL shell..."
	@docker exec -it kosmos-postgres psql -U kosmos

ps:
	@docker compose ps

top:
	@docker compose top

stats:
	@docker stats

pull:
	@echo "📥 Pulling latest images..."
	@docker compose pull
	@echo "✓ Images updated"

build:
	@echo "🔨 Building kosmos image..."
	@docker compose build
	@echo "✓ Build complete"

#==============================================================================
# Graph Management
#==============================================================================

graph-stats:
	@echo "📊 Knowledge graph statistics:"
	@source venv/bin/activate && kosmos graph --stats || echo "Activate venv first: source venv/bin/activate"

graph-export:
	@echo "💾 Exporting knowledge graph..."
	@source venv/bin/activate && kosmos graph --export backup_$(shell date +%Y%m%d_%H%M%S).json
	@echo "✓ Export complete"

graph-reset:
	@echo "⚠️  WARNING: This will delete all graph data!"
	@source venv/bin/activate && kosmos graph --reset

#==============================================================================
# Documentation
#==============================================================================

docs:
	@echo "📚 Building documentation..."
	@cd docs && make html
	@echo "✓ Documentation built in docs/_build/html/"

docs-serve:
	@echo "📚 Serving documentation..."
	@python -m http.server -d docs/_build/html 8080

#==============================================================================
# Environment Information
#==============================================================================

info:
	@echo "ℹ️  Kosmos Environment Information"
	@echo "════════════════════════════════════════════════════════════"
	@echo "Python:       $$(python3 --version 2>/dev/null || echo 'Not found')"
	@echo "Docker:       $$(docker --version 2>/dev/null || echo 'Not installed')"
	@echo "Docker Compose: $$(docker compose version 2>/dev/null || echo 'Not installed')"
	@echo "Git:          $$(git --version 2>/dev/null || echo 'Not found')"
	@echo "Kosmos:       $$(source venv/bin/activate 2>/dev/null && kosmos --version 2>/dev/null || echo 'Not installed')"
	@echo ""
	@echo "Virtual Env:  $$([ -d venv ] && echo 'venv/' || echo 'Not created')"
	@echo "Config:       $$([ -f .env ] && echo '.env exists' || echo '.env not found')"
	@echo ""
