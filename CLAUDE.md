# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Climate Data RAG (Retrieval-Augmented Generation) pipeline — a production system for ingesting multi-format climate datasets, generating semantic embeddings, and answering natural language queries via LLM. Diploma thesis project.

## Architecture

```
Caddy (80/443) → FastAPI (8000) + Dagit (3000)
                     ↓                ↓
               Vue.js frontend   Dagster ETL jobs
                     ↓                ↓
              Qdrant (6333)    PostgreSQL (5432)
                     ↓
              OpenRouter (LLM)
```

**Four main subsystems:**
- **`web_api/`** — FastAPI REST API + Vue.js frontend. `main.py` is the central file with source CRUD, RAG endpoints, catalog processing, scheduling, job triggering, and static file serving.
- **`dagster_project/`** — ETL orchestration. `dynamic_jobs.py` defines the main pipeline job. `catalog_jobs.py` defines `batch_catalog_etl_job` and `catalog_metadata_only_job`. `repository.py` is the Dagster entry point.
- **`src/`** — Core logic. `climate_embeddings/` handles multi-format loading, embedding, and RAG. `llm/` contains the OpenRouter client used by the RAG pipeline.
- **`src/catalog/`** — Batch catalog processing module for the D1.1.xlsx Excel catalog (233 entries, 69 unique datasets).

**Key design patterns:**
- Generator-based chunk streaming throughout — never loads full raster files into RAM
- Graceful LLM fallback — system returns raw search results if no LLM is available
- Embedding model: BAAI/bge-large-en-v1.5 (1024-dim, COSINE distance in Qdrant)
- Lazy imports for heavy ML deps (sentence-transformers, torch) inside FastAPI endpoints and Dagster ops

## Catalog Module (`src/catalog/`)

Batch-processes the D1.1.xlsx Excel catalog through 5 phases:

| Module | Purpose |
|--------|---------|
| `excel_reader.py` | Reads Excel with `openpyxl`, produces `CatalogEntry` dataclasses. Hazard column uses `.ffill()` for merged cells. |
| `phase_classifier.py` | Classifies entries into phases 0-4 based on link/access type. |
| `metadata_pipeline.py` | Phase 0: embeds Excel metadata as Qdrant points (no data download). |
| `batch_orchestrator.py` | Orchestrates all phases with resume support, progress tracking, persistent file logging. |
| `location_enricher.py` | Extracts/normalizes location info from region_country and spatial_coverage fields. |
| `portal_adapters.py` | Phase 3: adapters for CDS, ESGF, and other API portals. |

**Phases:**
- **Phase 0** — Metadata-only embedding (all 233 entries). No download, immediate RAG awareness.
- **Phase 1** — Direct HTTP download + raster processing (open-access `.nc`, `.tif`, etc.)
- **Phase 2** — Registration-required downloads (same pipeline, manual auth setup)
- **Phase 3** — API portal downloads (CDS, ESGF) via portal adapters
- **Phase 4** — Manual/restricted sources (metadata-only fallback)

## API Endpoints

### Existing
- `GET/POST/PUT/DELETE /sources` — Source CRUD
- `POST /rag/query` — RAG query (vector search + LLM)
- `GET /health` — Health check

### Catalog
- `GET /catalog` — List all catalog entries from Excel
- `GET /catalog/{row_index}` — Single catalog entry
- `POST /catalog/process` — Start batch processing (`{"phases": [0]}`)
- `GET /catalog/progress` — Current batch progress
- `POST /catalog/classify` — Dry-run phase classification
- `POST /catalog/retry-failed` — Retry failed entries

### Admin
- `GET /schedules` — List Dagster schedules
- `POST /schedules/{name}/toggle` — Enable/disable schedule
- `GET /logs/etl` — Read ETL log files
- `GET /settings/system` — System info (disk, memory, Qdrant stats)

## Vue.js Frontend Pages

- **Dashboard** (`/app`) — Overview with source counts, Qdrant stats, recent activity
- **Catalog** (`/app/catalog`) — Browse Excel catalog, trigger processing, view phase status badges
- **ETL Monitor** (`/app/etl`) — Real-time ETL progress, log viewer
- **Schedules** (`/app/schedules`) — Dagster schedule management
- **Settings** (`/app/settings`) — System info, configuration

All pages use dark theme with Tailwind CSS, `card` class for containers.

## Common Commands

```bash
# Docker (primary way to run)
docker compose up -d              # Start all services
docker compose down               # Stop services

# Local development
make install-dev                  # Install all dependencies
make api                          # Start FastAPI with --reload on port 8000
make dagit                        # Start Dagster UI + daemon on port 3000

# Testing
make test                         # Run all tests (pytest tests/ -v)
make test-raster                  # Raster pipeline tests only
make test-api                     # API endpoint tests only
make test-coverage                # Tests with coverage report
pytest tests/test_raster_pipeline_flow.py::TestClassName::test_name -v  # Single test

# Catalog operations
curl -X POST http://localhost:8000/catalog/process -H "Content-Type: application/json" -d '{"phases": [0]}'
curl http://localhost:8000/catalog/progress

# Code quality
make lint                         # flake8 (max-line-length=120)
make format                       # black (line-length=120)
make type-check                   # mypy
make check-all                    # lint + type-check + test
```

## Configuration

- **`config/pipeline_config.yaml`** — Raster loading, embedding model, Qdrant collection, text generation settings
- **`.env`** (from `.env.example`) — API keys, service hosts, auth credentials, `CATALOG_EXCEL_PATH`
- **`Caddyfile`** — Reverse proxy routing rules
- **`dagster_project/workspace.yaml`** + **`dagster.yaml`** — Dagster workspace and instance config
- **`data/Kopie souboru D1.1.xlsx`** — Excel catalog file (mounted via `./data:/app/data` in Docker)

## Docker Services

Seven services in `docker-compose.yml`: dagster-postgres (:5432), dagster-daemon, dagit (:3000), web-api (:8000), qdrant (:6333/:6334), caddy (:80/:443), jupyter (:8888 optional). All on `climate-net` bridge network.

## Deployment

- **Server**: Digital Ocean droplet at `159.65.207.173`
- **Domain**: `climaterag.online` (Caddy auto-TLS)
- **Deploy**: `git pull && docker compose build --no-cache && docker compose up -d`
- **Logs**: `docker compose logs -f web-api` or `GET /logs/etl` endpoint
- **Persistent catalog log**: `logs/catalog_pipeline.log` (via `setup_logger()`)

## Companion Directory

`../xchomo_latex/` contains the diploma thesis LaTeX source. `0_kontext/` has project state docs (`stav_projektu.md`) and literature review (`literarna_resers.md`) in Czech/Slovak.
