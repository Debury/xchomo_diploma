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

download: ## Download ERA5 climate data
	@echo "$(BLUE)Downloading ERA5 data...$(NC)"
	$(PYTHON) -m src.data_acquisition.era5_downloader
	@echo "$(GREEN)✓ Download complete$(NC)"

transform: ## Run data transformation pipeline
	@echo "$(BLUE)Running transformation pipeline...$(NC)"
	$(PYTHON) -m src.data_transformation.pipeline $(DATA_RAW)/test_era5_data.nc -o $(DATA_PROCESSED)
	@echo "$(GREEN)✓ Transformation complete$(NC)"

visualize: ## Visualize processed data
	@echo "$(BLUE)Generating visualizations...$(NC)"
	$(PYTHON) -m src.data_acquisition.visualizer $(DATA_PROCESSED)/test_era5_data.nc
	@echo "$(GREEN)✓ Visualization complete$(NC)"

run-all: setup download transform ## Run complete pipeline (download + transform)
	@echo "$(GREEN)✓ Complete pipeline executed successfully$(NC)"

##@ Testing

test: ## Run all tests
	@echo "$(BLUE)Running tests...$(NC)"
	$(PYTEST) $(TEST_DIR) -v
	@echo "$(GREEN)✓ Tests complete$(NC)"

test-coverage: ## Run tests with coverage report
	@echo "$(BLUE)Running tests with coverage...$(NC)"
	$(PYTEST) $(TEST_DIR) --cov=$(SRC_DIR) --cov-report=html --cov-report=term
	@echo "$(GREEN)✓ Coverage report generated in htmlcov/$(NC)"

test-integration: ## Run integration tests only
	@echo "$(BLUE)Running integration tests...$(NC)"
	$(PYTEST) $(TEST_DIR)/test_integration.py -v
	@echo "$(GREEN)✓ Integration tests complete$(NC)"

test-watch: ## Run tests in watch mode
	$(PYTEST) $(TEST_DIR) -v --looponfail

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

docker-compose-up: ## Start services with docker-compose
	docker-compose up -d
	@echo "$(GREEN)✓ Services started$(NC)"

docker-compose-down: ## Stop services with docker-compose
	docker-compose down
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

quick-test: ## Quick test with sample data
	@echo "$(BLUE)Running quick test...$(NC)"
	$(PYTHON) -m src.data_transformation.pipeline data/raw/test_era5_data.nc
	@echo "$(GREEN)✓ Quick test complete$(NC)"

quick-viz: ## Quick visualization of latest processed data
	@echo "$(BLUE)Creating quick visualization...$(NC)"
	$(PYTHON) -m src.data_acquisition.visualizer $(DATA_PROCESSED)/test_era5_data.nc
	@echo "$(GREEN)✓ Visualization generated$(NC)"

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
	$(PYTHON) -m uvicorn web_api.main:app --host 0.0.0.0 --port 8000 --reload

dagster-all: ## Start all Dagster services (Docker Compose)
	@echo "$(BLUE)Starting all Dagster services...$(NC)"
	docker-compose up -d dagster-postgres dagster-daemon dagit web-api
	@echo "$(GREEN)✓ Services started:$(NC)"
	@echo "  - Dagit UI: http://localhost:3000"
	@echo "  - API Service: http://localhost:8000"
	@echo "  - API Docs: http://localhost:8000/docs"

dagster-stop: ## Stop all Dagster services
	@echo "$(BLUE)Stopping Dagster services...$(NC)"
	docker-compose stop dagster-postgres dagster-daemon dagit web-api
	@echo "$(GREEN)✓ Services stopped$(NC)"

dagster-logs: ## Show Dagster service logs
	docker-compose logs -f dagit dagster-daemon web-api

test-dagster: ## Run Phase 4 tests (Dagster + API)
	@echo "$(BLUE)Running Phase 4 tests...$(NC)"
	$(PYTEST) $(TEST_DIR)/test_dagster.py $(TEST_DIR)/test_web_api.py -v
	@echo "$(GREEN)✓ Phase 4 tests complete$(NC)"

verify-phase4: ## Verify Phase 4 implementation
	@echo "$(BLUE)Verifying Phase 4 structure...$(NC)"
	@echo ""
	@echo "Checking Dagster project files:"
	@if exist "dagster_project\__init__.py" (echo "  [✓] __init__.py") else (echo "  [✗] __init__.py")
	@if exist "dagster_project\workspace.yaml" (echo "  [✓] workspace.yaml") else (echo "  [✗] workspace.yaml")
	@if exist "dagster_project\dagster.yaml" (echo "  [✓] dagster.yaml") else (echo "  [✗] dagster.yaml")
	@if exist "dagster_project\resources.py" (echo "  [✓] resources.py") else (echo "  [✗] resources.py")
	@if exist "dagster_project\jobs.py" (echo "  [✓] jobs.py") else (echo "  [✗] jobs.py")
	@if exist "dagster_project\schedules.py" (echo "  [✓] schedules.py") else (echo "  [✗] schedules.py")
	@if exist "dagster_project\repository.py" (echo "  [✓] repository.py") else (echo "  [✗] repository.py")
	@echo ""
	@echo "Checking ops:"
	@if exist "dagster_project\ops\__init__.py" (echo "  [✓] ops\__init__.py") else (echo "  [✗] ops\__init__.py")
	@if exist "dagster_project\ops\data_acquisition_ops.py" (echo "  [✓] ops\data_acquisition_ops.py") else (echo "  [✗] ops\data_acquisition_ops.py")
	@if exist "dagster_project\ops\transformation_ops.py" (echo "  [✓] ops\transformation_ops.py") else (echo "  [✗] ops\transformation_ops.py")
	@if exist "dagster_project\ops\embedding_ops.py" (echo "  [✓] ops\embedding_ops.py") else (echo "  [✗] ops\embedding_ops.py")
	@echo ""
	@echo "Checking Web API:"
	@if exist "web_api\__init__.py" (echo "  [✓] web_api\__init__.py") else (echo "  [✗] web_api\__init__.py")
	@if exist "web_api\main.py" (echo "  [✓] web_api\main.py") else (echo "  [✗] web_api\main.py")
	@echo ""
	@echo "Checking tests:"
	@if exist "tests\test_dagster.py" (echo "  [✓] test_dagster.py") else (echo "  [✗] test_dagster.py")
	@if exist "tests\test_web_api.py" (echo "  [✓] test_web_api.py") else (echo "  [✗] test_web_api.py")
	@echo ""
	@echo "Checking documentation:"
	@if exist "docs\PHASE4_USAGE.md" (echo "  [✓] PHASE4_USAGE.md") else (echo "  [✗] PHASE4_USAGE.md")
	@if exist "docs\PHASE4_SUMMARY.md" (echo "  [✓] PHASE4_SUMMARY.md") else (echo "  [✗] PHASE4_SUMMARY.md")
	@echo ""
	@echo "$(GREEN)✓ Phase 4 verification complete$(NC)"

api-health: ## Check API health status
	@echo "$(BLUE)Checking API health...$(NC)"
	@curl -s http://localhost:8000/health || echo "API not running"

api-list-jobs: ## List available Dagster jobs via API
	@echo "$(BLUE)Listing jobs...$(NC)"
	@curl -s http://localhost:8000/jobs | $(PYTHON) -m json.tool

trigger-etl: ## Trigger daily ETL job via API
	@echo "$(BLUE)Triggering daily_etl_job...$(NC)"
	@curl -X POST http://localhost:8000/jobs/daily_etl_job/run -H "Content-Type: application/json" -d "{}"

trigger-embeddings: ## Trigger embedding job via API
	@echo "$(BLUE)Triggering embedding_job...$(NC)"
	@curl -X POST http://localhost:8000/jobs/embedding_job/run -H "Content-Type: application/json" -d "{}"

