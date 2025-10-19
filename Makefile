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
