# Makefile for AI Hardware Co-Design Playground
# Provides convenient commands for development, testing, and deployment

# ============================================================================
# Configuration
# ============================================================================
.DEFAULT_GOAL := help
.PHONY: help install clean test lint format build docker up down logs shell

# Docker configuration
DOCKER_COMPOSE_DEV := docker-compose -f docker-compose.dev.yml
DOCKER_COMPOSE_PROD := docker-compose -f docker-compose.yml
DOCKER_IMAGE_NAME := ai-hardware-codesign-playground
DOCKER_TAG := latest

# Python configuration
PYTHON := python3
PIP := pip3
PYTEST := pytest

# Node.js configuration
NPM := npm
NODE := node

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[0;33m
BLUE := \033[0;34m
MAGENTA := \033[0;35m
CYAN := \033[0;36m
WHITE := \033[0;37m
RESET := \033[0m

# ============================================================================
# Help
# ============================================================================
help: ## Show this help message
	@echo "$(CYAN)AI Hardware Co-Design Playground - Build Commands$(RESET)"
	@echo ""
	@echo "$(YELLOW)Development Commands:$(RESET)"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  $(GREEN)%-20s$(RESET) %s\n", $$1, $$2}' $(MAKEFILE_LIST) | grep -E "install|clean|test|lint|format|dev"
	@echo ""
	@echo "$(YELLOW)Docker Commands:$(RESET)"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  $(GREEN)%-20s$(RESET) %s\n", $$1, $$2}' $(MAKEFILE_LIST) | grep -E "docker|build|up|down|logs|shell"
	@echo ""
	@echo "$(YELLOW)Production Commands:$(RESET)"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  $(GREEN)%-20s$(RESET) %s\n", $$1, $$2}' $(MAKEFILE_LIST) | grep -E "deploy|prod|release"
	@echo ""

# ============================================================================
# Development Environment Setup
# ============================================================================
install: install-python install-node ## Install all dependencies

install-python: ## Install Python dependencies
	@echo "$(BLUE)Installing Python dependencies...$(RESET)"
	$(PIP) install -e .[dev,test,docs]
	@echo "$(GREEN)Python dependencies installed successfully!$(RESET)"

install-node: ## Install Node.js dependencies
	@echo "$(BLUE)Installing Node.js dependencies...$(RESET)"
	cd frontend && $(NPM) install
	@echo "$(GREEN)Node.js dependencies installed successfully!$(RESET)"

install-hooks: ## Install pre-commit hooks
	@echo "$(BLUE)Installing pre-commit hooks...$(RESET)"
	pre-commit install
	pre-commit install --hook-type commit-msg
	@echo "$(GREEN)Pre-commit hooks installed successfully!$(RESET)"

setup: install install-hooks ## Complete development setup
	@echo "$(GREEN)Development environment setup complete!$(RESET)"

# ============================================================================
# Cleaning
# ============================================================================
clean: clean-python clean-node clean-docker ## Clean all generated files

clean-python: ## Clean Python generated files
	@echo "$(BLUE)Cleaning Python files...$(RESET)"
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete
	find . -type f -name '*.pyo' -delete
	find . -type f -name '*.pyd' -delete
	find . -type f -name '.coverage' -delete
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .tox/
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	@echo "$(GREEN)Python files cleaned!$(RESET)"

clean-node: ## Clean Node.js generated files
	@echo "$(BLUE)Cleaning Node.js files...$(RESET)"
	rm -rf frontend/node_modules/
	rm -rf frontend/dist/
	rm -rf frontend/build/
	rm -rf frontend/.next/
	rm -f frontend/package-lock.json
	@echo "$(GREEN)Node.js files cleaned!$(RESET)"

clean-docker: ## Clean Docker images and volumes
	@echo "$(BLUE)Cleaning Docker resources...$(RESET)"
	$(DOCKER_COMPOSE_DEV) down -v --remove-orphans
	$(DOCKER_COMPOSE_PROD) down -v --remove-orphans
	docker system prune -f
	@echo "$(GREEN)Docker resources cleaned!$(RESET)"

# ============================================================================
# Code Quality
# ============================================================================
lint: lint-python lint-node ## Run all linting

lint-python: ## Run Python linting
	@echo "$(BLUE)Running Python linting...$(RESET)"
	black --check backend/src/ backend/tests/
	isort --check-only backend/src/ backend/tests/
	flake8 backend/src/ backend/tests/
	pylint backend/src/
	mypy backend/src/
	bandit -r backend/src/
	@echo "$(GREEN)Python linting passed!$(RESET)"

lint-node: ## Run Node.js linting
	@echo "$(BLUE)Running Node.js linting...$(RESET)"
	cd frontend && $(NPM) run lint
	@echo "$(GREEN)Node.js linting passed!$(RESET)"

format: format-python format-node ## Format all code

format-python: ## Format Python code
	@echo "$(BLUE)Formatting Python code...$(RESET)"
	black backend/src/ backend/tests/
	isort backend/src/ backend/tests/
	@echo "$(GREEN)Python code formatted!$(RESET)"

format-node: ## Format Node.js code
	@echo "$(BLUE)Formatting Node.js code...$(RESET)"
	cd frontend && $(NPM) run format
	@echo "$(GREEN)Node.js code formatted!$(RESET)"

type-check: ## Run type checking
	@echo "$(BLUE)Running type checking...$(RESET)"
	mypy backend/src/
	cd frontend && $(NPM) run typecheck
	@echo "$(GREEN)Type checking passed!$(RESET)"

# ============================================================================
# Testing
# ============================================================================
test: test-python test-node ## Run all tests

test-python: ## Run Python tests
	@echo "$(BLUE)Running Python tests...$(RESET)"
	$(PYTEST) tests/ -v --cov=backend/src --cov-report=html --cov-report=term-missing
	@echo "$(GREEN)Python tests passed!$(RESET)"

test-node: ## Run Node.js tests
	@echo "$(BLUE)Running Node.js tests...$(RESET)"
	cd frontend && $(NPM) run test
	@echo "$(GREEN)Node.js tests passed!$(RESET)"

test-unit: ## Run only unit tests
	@echo "$(BLUE)Running unit tests...$(RESET)"
	$(PYTEST) tests/unit/ -v
	@echo "$(GREEN)Unit tests passed!$(RESET)"

test-integration: ## Run only integration tests
	@echo "$(BLUE)Running integration tests...$(RESET)"
	$(PYTEST) tests/integration/ -v
	@echo "$(GREEN)Integration tests passed!$(RESET)"

test-e2e: ## Run only end-to-end tests
	@echo "$(BLUE)Running end-to-end tests...$(RESET)"
	$(PYTEST) tests/e2e/ -v
	@echo "$(GREEN)End-to-end tests passed!$(RESET)"

test-performance: ## Run performance tests
	@echo "$(BLUE)Running performance tests...$(RESET)"
	$(PYTEST) tests/performance/ -v --benchmark-only
	@echo "$(GREEN)Performance tests passed!$(RESET)"

test-watch: ## Run tests in watch mode
	@echo "$(BLUE)Running tests in watch mode...$(RESET)"
	ptw tests/ -- -v

coverage: ## Generate coverage report
	@echo "$(BLUE)Generating coverage report...$(RESET)"
	$(PYTEST) tests/ --cov=backend/src --cov-report=html --cov-report=term
	open htmlcov/index.html

# ============================================================================
# Docker Development
# ============================================================================
docker-build: ## Build Docker images
	@echo "$(BLUE)Building Docker images...$(RESET)"
	docker build -t $(DOCKER_IMAGE_NAME):$(DOCKER_TAG) .
	docker build -t $(DOCKER_IMAGE_NAME):dev --target development .
	@echo "$(GREEN)Docker images built successfully!$(RESET)"

docker-build-prod: ## Build production Docker image
	@echo "$(BLUE)Building production Docker image...$(RESET)"
	docker build -t $(DOCKER_IMAGE_NAME):$(DOCKER_TAG) --target production .
	@echo "$(GREEN)Production Docker image built successfully!$(RESET)"

up: ## Start development environment
	@echo "$(BLUE)Starting development environment...$(RESET)"
	$(DOCKER_COMPOSE_DEV) up -d
	@echo "$(GREEN)Development environment started!$(RESET)"
	@echo "$(YELLOW)Web UI: http://localhost:8000$(RESET)"
	@echo "$(YELLOW)Frontend: http://localhost:3000$(RESET)"
	@echo "$(YELLOW)Jupyter: http://localhost:8888$(RESET)"

down: ## Stop development environment
	@echo "$(BLUE)Stopping development environment...$(RESET)"
	$(DOCKER_COMPOSE_DEV) down
	@echo "$(GREEN)Development environment stopped!$(RESET)"

restart: down up ## Restart development environment

logs: ## Show logs from all services
	$(DOCKER_COMPOSE_DEV) logs -f

logs-web: ## Show logs from web service
	$(DOCKER_COMPOSE_DEV) logs -f web

logs-worker: ## Show logs from worker service
	$(DOCKER_COMPOSE_DEV) logs -f worker

shell: ## Open shell in web container
	$(DOCKER_COMPOSE_DEV) exec web bash

shell-db: ## Open PostgreSQL shell
	$(DOCKER_COMPOSE_DEV) exec db psql -U codesign_user -d codesign_dev

shell-redis: ## Open Redis shell
	$(DOCKER_COMPOSE_DEV) exec redis redis-cli

# ============================================================================
# Database Management
# ============================================================================
db-migrate: ## Run database migrations
	@echo "$(BLUE)Running database migrations...$(RESET)"
	$(DOCKER_COMPOSE_DEV) exec web alembic upgrade head
	@echo "$(GREEN)Database migrations completed!$(RESET)"

db-migration: ## Create new database migration
	@echo "$(BLUE)Creating new database migration...$(RESET)"
	$(DOCKER_COMPOSE_DEV) exec web alembic revision --autogenerate -m "$(MSG)"
	@echo "$(GREEN)Database migration created!$(RESET)"

db-reset: ## Reset database
	@echo "$(BLUE)Resetting database...$(RESET)"
	$(DOCKER_COMPOSE_DEV) exec web alembic downgrade base
	$(DOCKER_COMPOSE_DEV) exec web alembic upgrade head
	@echo "$(GREEN)Database reset completed!$(RESET)"

db-seed: ## Seed database with test data
	@echo "$(BLUE)Seeding database...$(RESET)"
	$(DOCKER_COMPOSE_DEV) exec web python scripts/seed_database.py
	@echo "$(GREEN)Database seeded!$(RESET)"

# ============================================================================
# Production Deployment
# ============================================================================
prod-up: ## Start production environment
	@echo "$(BLUE)Starting production environment...$(RESET)"
	$(DOCKER_COMPOSE_PROD) up -d
	@echo "$(GREEN)Production environment started!$(RESET)"

prod-down: ## Stop production environment
	@echo "$(BLUE)Stopping production environment...$(RESET)"
	$(DOCKER_COMPOSE_PROD) down
	@echo "$(GREEN)Production environment stopped!$(RESET)"

prod-logs: ## Show production logs
	$(DOCKER_COMPOSE_PROD) logs -f

prod-deploy: docker-build-prod ## Deploy to production
	@echo "$(BLUE)Deploying to production...$(RESET)"
	$(DOCKER_COMPOSE_PROD) up -d --force-recreate
	@echo "$(GREEN)Production deployment completed!$(RESET)"

# ============================================================================
# Documentation
# ============================================================================
docs: ## Build documentation
	@echo "$(BLUE)Building documentation...$(RESET)"
	cd backend && sphinx-build -b html docs/ docs/_build/html
	@echo "$(GREEN)Documentation built!$(RESET)"

docs-serve: ## Serve documentation locally
	@echo "$(BLUE)Starting documentation server...$(RESET)"
	cd backend/docs/_build/html && python -m http.server 8080

docs-dev: ## Start documentation development server
	@echo "$(BLUE)Starting documentation development server...$(RESET)"
	cd backend && sphinx-autobuild docs/ docs/_build/html --watch ../

# ============================================================================
# Security
# ============================================================================
security: security-python security-node ## Run security checks

security-python: ## Run Python security checks
	@echo "$(BLUE)Running Python security checks...$(RESET)"
	bandit -r backend/src/
	safety check
	@echo "$(GREEN)Python security checks passed!$(RESET)"

security-node: ## Run Node.js security checks
	@echo "$(BLUE)Running Node.js security checks...$(RESET)"
	cd frontend && npm audit
	@echo "$(GREEN)Node.js security checks passed!$(RESET)"

# ============================================================================
# Release Management
# ============================================================================
version-patch: ## Bump patch version
	@echo "$(BLUE)Bumping patch version...$(RESET)"
	npm version patch
	git push && git push --tags
	@echo "$(GREEN)Patch version bumped!$(RESET)"

version-minor: ## Bump minor version
	@echo "$(BLUE)Bumping minor version...$(RESET)"
	npm version minor
	git push && git push --tags
	@echo "$(GREEN)Minor version bumped!$(RESET)"

version-major: ## Bump major version
	@echo "$(BLUE)Bumping major version...$(RESET)"
	npm version major
	git push && git push --tags
	@echo "$(GREEN)Major version bumped!$(RESET)"

release: ## Create release
	@echo "$(BLUE)Creating release...$(RESET)"
	npm run release
	@echo "$(GREEN)Release created!$(RESET)"

# ============================================================================
# Monitoring
# ============================================================================
monitor: ## Start monitoring stack
	@echo "$(BLUE)Starting monitoring stack...$(RESET)"
	$(DOCKER_COMPOSE_PROD) up -d prometheus grafana
	@echo "$(GREEN)Monitoring stack started!$(RESET)"
	@echo "$(YELLOW)Prometheus: http://localhost:9090$(RESET)"
	@echo "$(YELLOW)Grafana: http://localhost:3001$(RESET)"

# ============================================================================
# Utilities
# ============================================================================
ps: ## Show running containers
	$(DOCKER_COMPOSE_DEV) ps

top: ## Show container resource usage
	docker stats

health: ## Check service health
	@echo "$(BLUE)Checking service health...$(RESET)"
	curl -f http://localhost:8000/health || echo "$(RED)Web service unhealthy$(RESET)"
	curl -f http://localhost:3000 || echo "$(RED)Frontend service unhealthy$(RESET)"
	@echo "$(GREEN)Health check completed!$(RESET)"

backup: ## Backup database
	@echo "$(BLUE)Creating database backup...$(RESET)"
	$(DOCKER_COMPOSE_DEV) exec db pg_dump -U codesign_user codesign_dev > backup_$(shell date +%Y%m%d_%H%M%S).sql
	@echo "$(GREEN)Database backup created!$(RESET)"

restore: ## Restore database from backup
	@echo "$(BLUE)Restoring database from backup...$(RESET)"
	$(DOCKER_COMPOSE_DEV) exec -T db psql -U codesign_user codesign_dev < $(BACKUP_FILE)
	@echo "$(GREEN)Database restored!$(RESET)"

# ============================================================================
# CI/CD
# ============================================================================
ci: install lint test security ## Run CI pipeline locally
	@echo "$(GREEN)CI pipeline completed successfully!$(RESET)"

validate: lint type-check test security ## Validate code quality
	@echo "$(GREEN)Code validation completed successfully!$(RESET)"

# ============================================================================
# Environment Info
# ============================================================================
info: ## Show environment information
	@echo "$(CYAN)Environment Information:$(RESET)"
	@echo "Python: $(shell $(PYTHON) --version)"
	@echo "Node.js: $(shell $(NODE) --version)"
	@echo "npm: $(shell $(NPM) --version)"
	@echo "Docker: $(shell docker --version)"
	@echo "Docker Compose: $(shell docker-compose --version)"
	@echo "Git: $(shell git --version)"
	@echo "OS: $(shell uname -s -r)"
	@echo "Architecture: $(shell uname -m)"
