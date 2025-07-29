# Makefile for AI Hardware Co-Design Playground
# Provides convenient commands for development, testing, and deployment

.PHONY: help install install-dev clean test test-unit test-integration test-e2e
.PHONY: lint lint-fix format typecheck security coverage
.PHONY: build build-frontend build-backend docker-build docker-dev docker-prod
.PHONY: serve serve-dev deploy docs docs-serve
.PHONY: setup setup-dev setup-ci setup-hooks
.PHONY: migrate seed reset logs shell

# Default target
.DEFAULT_GOAL := help

# Variables
PYTHON := python3
PIP := pip3
NPM := npm
DOCKER := docker
DOCKER_COMPOSE := docker-compose
DOCKER_COMPOSE_DEV := docker-compose -f docker-compose.dev.yml

# Colors for output
BLUE := \033[34m
GREEN := \033[32m
YELLOW := \033[33m
RED := \033[31m
RESET := \033[0m

## Help
help: ## Show this help message
	@echo "$(BLUE)AI Hardware Co-Design Playground - Development Commands$(RESET)"
	@echo ""
	@echo "$(GREEN)Available commands:$(RESET)"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  $(YELLOW)%-20s$(RESET) %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""

## Installation
install: ## Install production dependencies
	@echo "$(BLUE)Installing production dependencies...$(RESET)"
	$(PIP) install -r requirements.txt
	$(NPM) install --production

install-dev: ## Install development dependencies
	@echo "$(BLUE)Installing development dependencies...$(RESET)"
	$(PIP) install -r requirements-dev.txt
	$(NPM) install
	pre-commit install

install-all: ## Install all dependencies including optional ones
	@echo "$(BLUE)Installing all dependencies...$(RESET)"
	$(PIP) install -e ".[all]"
	$(NPM) install

## Setup
setup: install-dev setup-hooks ## Complete development setup
	@echo "$(GREEN)Development environment setup complete!$(RESET)"

setup-dev: ## Setup development environment
	@echo "$(BLUE)Setting up development environment...$(RESET)"
	mkdir -p logs uploads generated data notebooks
	cp .env.example .env 2>/dev/null || echo "No .env.example found"
	$(MAKE) install-dev

setup-ci: ## Setup CI environment
	@echo "$(BLUE)Setting up CI environment...$(RESET)"
	$(PIP) install -r requirements-dev.txt
	$(NPM) ci

setup-hooks: ## Setup git hooks
	@echo "$(BLUE)Setting up git hooks...$(RESET)"
	pre-commit install
	pre-commit install --hook-type commit-msg

## Cleaning
clean: ## Clean build artifacts and caches
	@echo "$(BLUE)Cleaning build artifacts...$(RESET)"
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .coverage htmlcov/ .pytest_cache/ .mypy_cache/ .ruff_cache/
	rm -rf dist/ build/ *.egg-info/
	rm -rf frontend/dist/ frontend/build/ frontend/.next/
	rm -rf node_modules/.cache/
	$(NPM) run clean 2>/dev/null || true

clean-all: clean ## Clean everything including node_modules
	@echo "$(BLUE)Cleaning everything...$(RESET)"
	rm -rf node_modules/ frontend/node_modules/
	rm -rf .venv/ venv/

## Testing
test: ## Run all tests
	@echo "$(BLUE)Running all tests...$(RESET)"
	$(NPM) run test

test-unit: ## Run unit tests only
	@echo "$(BLUE)Running unit tests...$(RESET)"
	pytest tests/unit/ -v

test-integration: ## Run integration tests only
	@echo "$(BLUE)Running integration tests...$(RESET)"
	pytest tests/integration/ -v

test-e2e: ## Run end-to-end tests
	@echo "$(BLUE)Running end-to-end tests...$(RESET)"
	pytest tests/e2e/ -v

test-benchmark: ## Run benchmark tests
	@echo "$(BLUE)Running benchmark tests...$(RESET)"
	pytest tests/benchmarks/ -v --benchmark-only

test-watch: ## Run tests in watch mode
	@echo "$(BLUE)Running tests in watch mode...$(RESET)"
	$(NPM) run test:watch

test-coverage: ## Run tests with coverage report
	@echo "$(BLUE)Running tests with coverage...$(RESET)"
	pytest --cov=backend/codesign_playground --cov-report=html --cov-report=term-missing
	@echo "$(GREEN)Coverage report generated in htmlcov/$(RESET)"

## Code Quality
lint: ## Run all linters
	@echo "$(BLUE)Running linters...$(RESET)"
	$(NPM) run lint

lint-fix: ## Run linters with auto-fix
	@echo "$(BLUE)Running linters with auto-fix...$(RESET)"
	$(NPM) run lint:fix

format: ## Format code
	@echo "$(BLUE)Formatting code...$(RESET)"
	$(NPM) run format

typecheck: ## Run type checking
	@echo "$(BLUE)Running type checking...$(RESET)"
	$(NPM) run typecheck

security: ## Run security checks
	@echo "$(BLUE)Running security checks...$(RESET)"
	$(NPM) run security

validate: ## Run all validation (lint, typecheck, test, security)
	@echo "$(BLUE)Running all validation...$(RESET)"
	$(NPM) run validate

## Building
build: ## Build all components
	@echo "$(BLUE)Building all components...$(RESET)"
	$(NPM) run build

build-frontend: ## Build frontend only
	@echo "$(BLUE)Building frontend...$(RESET)"
	$(NPM) run build:frontend

build-backend: ## Build backend only
	@echo "$(BLUE)Building backend...$(RESET)"
	$(NPM) run build:backend

## Docker
docker-build: ## Build Docker images
	@echo "$(BLUE)Building Docker images...$(RESET)"
	$(DOCKER) build -t ai-hardware-codesign-playground .

docker-dev: ## Start development environment with Docker
	@echo "$(BLUE)Starting development environment...$(RESET)"
	$(DOCKER_COMPOSE_DEV) up --build

docker-prod: ## Start production environment with Docker
	@echo "$(BLUE)Starting production environment...$(RESET)"
	$(DOCKER_COMPOSE) up --build -d

docker-down: ## Stop Docker containers
	@echo "$(BLUE)Stopping Docker containers...$(RESET)"
	$(DOCKER_COMPOSE_DEV) down
	$(DOCKER_COMPOSE) down

docker-logs: ## View Docker logs
	$(DOCKER_COMPOSE_DEV) logs -f

docker-shell: ## Open shell in backend container
	$(DOCKER_COMPOSE_DEV) exec backend bash

## Development Server
serve: ## Start development servers
	@echo "$(BLUE)Starting development servers...$(RESET)"
	$(NPM) run dev

serve-prod: ## Start production servers locally
	@echo "$(BLUE)Starting production servers...$(RESET)"
	$(NPM) run start

serve-backend: ## Start backend server only
	@echo "$(BLUE)Starting backend server...$(RESET)"
	$(NPM) run dev:backend

serve-frontend: ## Start frontend server only
	@echo "$(BLUE)Starting frontend server...$(RESET)"
	$(NPM) run dev:frontend

## Database
migrate: ## Run database migrations
	@echo "$(BLUE)Running database migrations...$(RESET)"
	$(NPM) run migrate

migrate-create: ## Create new migration
	@echo "$(BLUE)Creating new migration...$(RESET)"
	$(NPM) run migrate:create

seed: ## Seed database with sample data
	@echo "$(BLUE)Seeding database...$(RESET)"
	$(NPM) run seed

reset: clean migrate seed ## Reset database and reseed
	@echo "$(GREEN)Database reset complete!$(RESET)"

## Documentation
docs: ## Build documentation
	@echo "$(BLUE)Building documentation...$(RESET)"
	$(NPM) run docs:build

docs-serve: ## Serve documentation locally
	@echo "$(BLUE)Serving documentation...$(RESET)"
	$(NPM) run docs:serve

## Jupyter
jupyter: ## Start Jupyter Lab
	@echo "$(BLUE)Starting Jupyter Lab...$(RESET)"
	$(NPM) run jupyter

## Utilities
logs: ## View application logs
	tail -f logs/*.log 2>/dev/null || echo "No log files found"

shell: ## Open Python shell with app context
	@echo "$(BLUE)Opening Python shell...$(RESET)"
	$(NPM) run shell

health: ## Check application health
	@echo "$(BLUE)Checking application health...$(RESET)"
	$(NPM) run health || echo "$(RED)Health check failed$(RESET)"

## Release
version-patch: ## Bump patch version
	@echo "$(BLUE)Bumping patch version...$(RESET)"
	$(NPM) run version:patch

version-minor: ## Bump minor version
	@echo "$(BLUE)Bumping minor version...$(RESET)"
	$(NPM) run version:minor

version-major: ## Bump major version
	@echo "$(BLUE)Bumping major version...$(RESET)"
	$(NPM) run version:major

release: ## Create release
	@echo "$(BLUE)Creating release...$(RESET)"
	$(NPM) run release

## Deployment
deploy-staging: ## Deploy to staging
	@echo "$(BLUE)Deploying to staging...$(RESET)"
	$(NPM) run deploy:staging

deploy-production: ## Deploy to production
	@echo "$(BLUE)Deploying to production...$(RESET)"
	$(NPM) run deploy:production

## Benchmarking
benchmark: ## Run performance benchmarks
	@echo "$(BLUE)Running performance benchmarks...$(RESET)"
	$(NPM) run benchmark

profile: ## Run profiling
	@echo "$(BLUE)Running profiling...$(RESET)"
	$(NPM) run profile

## Monitoring
monitor: ## Start monitoring stack
	@echo "$(BLUE)Starting monitoring stack...$(RESET)"
	$(DOCKER_COMPOSE) -f docker-compose.monitoring.yml up -d

monitor-down: ## Stop monitoring stack
	$(DOCKER_COMPOSE) -f docker-compose.monitoring.yml down

## Git hooks
pre-commit: ## Run pre-commit hooks
	@echo "$(BLUE)Running pre-commit hooks...$(RESET)"
	pre-commit run --all-files

pre-commit-update: ## Update pre-commit hooks
	@echo "$(BLUE)Updating pre-commit hooks...$(RESET)"
	pre-commit autoupdate