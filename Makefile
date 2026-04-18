# NOTE: Makefile targets are Windows-oriented dev conveniences. The supported,
# cross-platform way to run the project is `docker compose up -d` (see README).
# Some targets may assume a local Python venv and will not work inside Docker.

.PHONY: help install install-dev setup download transform visualize test test-coverage lint format clean run-all docker-build docker-run type-check

# Variables
PYTHON := python
PIP := pip
PYTEST := pytest
BLACK := black
FLAKE8 := flake8
MYPY := mypy

# Directories
SRC_DIR := src
TEST_DIR := tests
DATA_RAW := data/raw
DATA_PROCESSED := data/processed
LOGS_DIR := logs
CONFIG_DIR := config

# Colors for output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[1;33m
NC := \033[0m # No Color

##@ Help

help: ## Display this help message
	@echo "$(BLUE)Climate Data ETL Pipeline - Makefile Commands$(NC)"
	@echo ""
	@awk 'BEGIN {FS = ":.*##"; printf "Usage: make $(GREEN)<target>$(NC)\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2 } /^##@/ { printf "\n$(BLUE)%s$(NC)\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ Setup

install: ## Install production dependencies
	@echo "$(BLUE)Installing dependencies...$(NC)"
	$(PIP) install -r requirements.txt
	@echo "$(GREEN)✓ Dependencies installed$(NC)"

install-dev: install ## Install development dependencies
	@echo "$(BLUE)Installing development dependencies...$(NC)"
	$(PIP) install -r requirements-dev.txt
	@echo "$(GREEN)✓ Development dependencies installed$(NC)"

setup: ## Set up environment and directories
	@echo "$(BLUE)Setting up environment...$(NC)"
	@if not exist "$(DATA_RAW)" mkdir "$(DATA_RAW)"
	@if not exist "$(DATA_PROCESSED)" mkdir "$(DATA_PROCESSED)"
	@if not exist "$(LOGS_DIR)" mkdir "$(LOGS_DIR)"
	@if not exist ".env" copy ".env.example" ".env" && echo "$(YELLOW)⚠ Created .env file - please edit with your credentials$(NC)"
	@echo "$(GREEN)✓ Environment setup complete$(NC)"

##@ Data Pipeline

run-all: docker-compose-up ## Start all services with docker compose
	@echo "$(GREEN)✓ All services started$(NC)"

##@ Testing

test: ## Run all tests
	@echo "$(BLUE)Running tests...$(NC)"
	$(PYTEST) $(TEST_DIR) -v
	@echo "$(GREEN)✓ Tests complete$(NC)"

test-coverage: ## Run tests with coverage report
	@echo "$(BLUE)Running tests with coverage...$(NC)"
	$(PYTEST) $(TEST_DIR) --cov=$(SRC_DIR) --cov-report=html --cov-report=term
	@echo "$(GREEN)✓ Coverage report generated in htmlcov/$(NC)"

test-formats: ## Test all supported data formats (NetCDF, GeoTIFF, CSV, ZIP)
	@echo "$(BLUE)Testing all data formats...$(NC)"
	@bash test_formats_server.sh
	@echo "$(GREEN)✓ Format tests complete$(NC)"

test-raster: ## Run raster pipeline tests
	@echo "$(BLUE)Running raster pipeline tests...$(NC)"
	$(PYTEST) $(TEST_DIR)/test_raster_pipeline_flow.py -v
	@echo "$(GREEN)✓ Raster tests complete$(NC)"

test-api: ## Run API tests
	@echo "$(BLUE)Running API tests...$(NC)"
	$(PYTEST) $(TEST_DIR)/test_web_api.py -v
	@echo "$(GREEN)✓ API tests complete$(NC)"

test-watch: ## Run tests in watch mode
	$(PYTEST) $(TEST_DIR) -v --looponfail

test-all: test-raster test-api ## Run all test suites
	@echo "$(GREEN)✓ All test suites complete$(NC)"

##@ Code Quality

lint: ## Run code linters (flake8)
	@echo "$(BLUE)Running linters...$(NC)"
	$(FLAKE8) $(SRC_DIR) $(TEST_DIR) --max-line-length=120 --exclude=__pycache__,*.pyc
	@echo "$(GREEN)✓ Linting complete$(NC)"

format: ## Format code with black
	@echo "$(BLUE)Formatting code...$(NC)"
	$(BLACK) $(SRC_DIR) $(TEST_DIR) --line-length=120
	@echo "$(GREEN)✓ Code formatted$(NC)"

type-check: ## Run type checking with mypy
	@echo "$(BLUE)Running type checks...$(NC)"
	$(MYPY) $(SRC_DIR) --ignore-missing-imports
	@echo "$(GREEN)✓ Type checking complete$(NC)"

check-all: lint type-check test ## Run all quality checks

##@ Cleanup

clean: ## Clean generated files and caches
	@echo "$(BLUE)Cleaning generated files...$(NC)"
	@if exist "$(DATA_PROCESSED)\*" del /Q "$(DATA_PROCESSED)\*"
	@if exist "$(LOGS_DIR)\*" del /Q "$(LOGS_DIR)\*"
	@if exist ".pytest_cache" rmdir /S /Q ".pytest_cache"
	@if exist "htmlcov" rmdir /S /Q "htmlcov"
	@if exist ".coverage" del /Q ".coverage"
	@for /d /r . %%d in (__pycache__) do @if exist "%%d" rmdir /S /Q "%%d"
	@for /d /r . %%d in (*.egg-info) do @if exist "%%d" rmdir /S /Q "%%d"
	@echo "$(GREEN)✓ Cleanup complete$(NC)"

clean-docker: ## Clean all Docker images, containers, volumes (use with caution!)
	@echo "$(BLUE)Cleaning Docker resources...$(NC)"
	@bash cleanup_docker.sh
	@echo "$(GREEN)✓ Docker cleaned$(NC)"

clean-rebuild: clean-docker docker-build docker-compose-up ## Clean Docker and rebuild everything
	@echo "$(GREEN)✓ Clean rebuild complete$(NC)"

clean-data: ## Clean all data (raw + processed) - USE WITH CAUTION
	@echo "$(YELLOW)⚠ WARNING: This will delete all data files$(NC)"
	@choice /C YN /M "Are you sure you want to continue?"
	@if errorlevel 2 exit /b 1
	@if exist "$(DATA_RAW)\*" del /Q "$(DATA_RAW)\*"
	@if exist "$(DATA_PROCESSED)\*" del /Q "$(DATA_PROCESSED)\*"
	@echo "$(GREEN)✓ Data cleaned$(NC)"

##@ Docker

docker-build: ## Build Docker image
	@echo "$(BLUE)Building Docker image...$(NC)"
	docker build -t climate-etl-pipeline .
	@echo "$(GREEN)✓ Docker image built$(NC)"

docker-run: ## Run pipeline in Docker container
	@echo "$(BLUE)Running pipeline in Docker...$(NC)"
	docker run --rm -v "%CD%\data:/app/data" -v "%CD%\.env:/app/.env" climate-etl-pipeline
	@echo "$(GREEN)✓ Docker run complete$(NC)"

docker-compose-up: ## Start services with docker compose
	docker compose up -d
	@echo "$(GREEN)✓ Services started$(NC)"

docker-compose-down: ## Stop services with docker compose
	docker compose down
	@echo "$(GREEN)✓ Services stopped$(NC)"

##@ Documentation

docs: ## Generate documentation
	@echo "$(BLUE)Generating documentation...$(NC)"
	@echo "Documentation in docs/ directory"
	@echo "$(GREEN)✓ Documentation ready$(NC)"

docs-serve: ## Serve documentation locally
	@echo "$(BLUE)Serving documentation...$(NC)"
	$(PYTHON) -m http.server --directory docs 8000

##@ Utilities

show-config: ## Display current configuration
	@echo "$(BLUE)Current Configuration:$(NC)"
	@type "$(CONFIG_DIR)\pipeline_config.yaml"

show-status: ## Show pipeline status
	@echo "$(BLUE)Pipeline Status:$(NC)"
	@echo "Raw data files:"
	@dir /B "$(DATA_RAW)" 2>nul || echo "  No files"
	@echo ""
	@echo "Processed data files:"
	@dir /B "$(DATA_PROCESSED)" 2>nul || echo "  No files"
	@echo ""
	@echo "Log files:"
	@dir /B "$(LOGS_DIR)" 2>nul || echo "  No logs"

verify-env: ## Verify environment setup
	@echo "$(BLUE)Verifying environment...$(NC)"
	@$(PYTHON) --version
	@$(PIP) --version
	@if exist ".env" (echo "$(GREEN)✓ .env file exists$(NC)") else (echo "$(YELLOW)⚠ .env file missing$(NC)")
	@if exist "$(DATA_RAW)" (echo "$(GREEN)✓ Data directories exist$(NC)") else (echo "$(YELLOW)⚠ Data directories missing$(NC)")
	@echo "$(GREEN)✓ Environment verification complete$(NC)"

##@ Development

dev-setup: install-dev setup ## Complete development environment setup
	@echo "$(GREEN)✓ Development environment ready$(NC)"

watch-tests: ## Run tests in watch mode
	$(PYTEST) $(TEST_DIR) -v --looponfail

notebook: ## Start Jupyter notebook server
	jupyter notebook

##@ Quick Commands

list-sources: ## List configured data sources via API
	@echo "$(BLUE)Listing data sources...$(NC)"
	@curl -s http://localhost:8000/sources | $(PYTHON) -m json.tool || echo "$(YELLOW)⚠ API not running$(NC)"

check-qdrant: ## Check Qdrant status
	@echo "$(BLUE)Checking Qdrant vector database...$(NC)"
	@curl -s http://localhost:6333/health || echo "$(YELLOW)⚠ Qdrant not running$(NC)"

verify-services: check-qdrant api-health ## Verify all services are running
	@echo "$(GREEN)✓ Service verification complete$(NC)"

show-collections: ## Show Qdrant collections
	@echo "$(BLUE)Qdrant collections:$(NC)"
	@curl -s http://localhost:6333/collections | $(PYTHON) -m json.tool

##@ Phase 4: Orchestration & Web UI

dagster-setup: ## Setup Dagster home directory
	@echo "$(BLUE)Setting up Dagster...$(NC)"
	@if not exist ".dagster_home" mkdir ".dagster_home"
	@if not exist ".dagster_home\storage" mkdir ".dagster_home\storage"
	@if not exist ".dagster_home\compute_logs" mkdir ".dagster_home\compute_logs"
	@if not exist ".dagster_home\history" mkdir ".dagster_home\history"
	@if not exist ".dagster_home\dagster.yaml" type nul > ".dagster_home\dagster.yaml"
	@echo "$(GREEN)✓ Dagster home created at .dagster_home$(NC)"

dagit: dagster-setup ## Start Dagit UI (port 3000)
	@echo "$(BLUE)Starting Dagster dev server (UI + daemon)...$(NC)"
	@echo "Access at: http://localhost:3000"
	@echo "$(YELLOW)Press Ctrl+C to stop$(NC)"
	@start_dagster.bat

dagster-daemon: dagster-setup ## Start Dagster daemon (schedules & sensors)
	@echo "$(BLUE)Starting Dagster daemon...$(NC)"
	@echo "$(YELLOW)Note: Use 'make dagit' to start UI + daemon together$(NC)"
	@cmd /c "set DAGSTER_HOME=%~dp0.dagster_home && $(PYTHON) -m dagster daemon run -w dagster_project/workspace.yaml"

api: ## Start FastAPI web service (port 8000)
	@echo "$(BLUE)Starting FastAPI service...$(NC)"
	@echo "API: http://localhost:8000"
	@echo "Docs: http://localhost:8000/docs"
	$(PYTHON) -m uvicorn web_api.main:app --host 0.0.0.0 --port 8000 --reload --reload-dir web_api --reload-dir src --reload-dir config

dagster-all: ## Start all Dagster services (Docker Compose)
	@echo "$(BLUE)Starting all Dagster services...$(NC)"
	docker compose up -d dagster-postgres dagster-daemon dagit web-api
	@echo "$(GREEN)✓ Services started:$(NC)"
	@echo "  - Dagit UI: http://localhost:3000"
	@echo "  - API Service: http://localhost:8000"
	@echo "  - API Docs: http://localhost:8000/docs"

dagster-stop: ## Stop all Dagster services
	@echo "$(BLUE)Stopping Dagster services...$(NC)"
	docker compose stop dagster-postgres dagster-daemon dagit web-api
	@echo "$(GREEN)✓ Services stopped$(NC)"

dagster-logs: ## Show Dagster service logs
	docker compose logs -f dagit dagster-daemon web-api

api-health: ## Check API health status
	@echo "$(BLUE)Checking API health...$(NC)"
	@curl -s http://localhost:8000/health || echo "API not running"

rag-query: ## Query RAG system via API (example)
	@echo "$(BLUE)Querying RAG system...$(NC)"
	@echo "$(YELLOW)Example: curl -X POST http://localhost:8000/rag/query -H 'Content-Type: application/json' -d '{\"query\":\"What is the temperature trend?\",\"top_k\":5}'$(NC)"

