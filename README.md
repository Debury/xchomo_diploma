# Climate Data RAG Pipeline

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production RAG (Retrieval-Augmented Generation) system for multi-format climate datasets.
Ingest NetCDF / GRIB / HDF5 / GeoTIFF / CSV / Zarr, embed into Qdrant, query in natural language
via LLM.

Diploma thesis project. Live at **[climaterag.online](https://climaterag.online)**.

---

## 🚀 Quick Start (5 min)

**Prerequisites:** Docker Desktop 24+, 8 GB free RAM, ports 80 / 3000 / 6333 / 8000 free.

```bash
git clone <repo-url> xchomo_diploma
cd xchomo_diploma/xchomo_diploma

# 1. Create your .env from the example
cp .env.example .env
# Open .env, set AUTH_PASSWORD (required) and OPENROUTER_API_KEY (for LLM answers)

# 2. Boot the full stack (6 services; first build takes ~8 min)
docker compose up -d

# 3. Wait for all services to be healthy (~60 s after first-time build)
docker compose ps
# Expect all rows "Up", dagit + dagster-postgres marked "(healthy)"

# 4. Open the UI
#    http://localhost:8000/   →  redirects to /app/ (login page)
#    Log in with AUTH_USERNAME / AUTH_PASSWORD from your .env
```

Additional web UIs:

| URL | Purpose |
|---|---|
| http://localhost:8000/app/ | Vue SPA (main UI) |
| http://localhost:8000/docs | Swagger / OpenAPI docs |
| http://localhost:3000/ | Dagster orchestration UI |
| http://localhost:6333/dashboard | Qdrant vector DB dashboard |

### First source (curl smoke test)

```bash
# Add an open-access CSV source (no auto-embed)
curl -X POST http://localhost:8000/sources -L \
  -H 'Content-Type: application/json' \
  -d '{
    "source_id": "co2_mauna_loa",
    "url": "https://raw.githubusercontent.com/datasets/co2-ppm/master/data/co2-mm-mlo.csv",
    "format": "csv",
    "description": "Monthly mean CO2 at Mauna Loa",
    "hazard_type": "Atmospheric composition",
    "auto_embed": false
  }'

# Ask a question via RAG
curl -X POST http://localhost:8000/rag/chat \
  -H 'Content-Type: application/json' \
  -d '{"question": "What precipitation data is available for Slovakia?", "top_k": 3}'
```

---

## 🏗️ Architecture

```
Caddy (80/443) ──→ FastAPI (8000) + Dagster webserver (3000)
                       │                   │
                       │                   ▼
                       │         Dagster ETL jobs + sensors
                       ▼                   │
                  Vue 3 SPA                ▼
                       │         PostgreSQL (metadata)
                       ▼
                  Qdrant (6333)  ←──   Embeddings (1024-dim, COSINE)
                       │
                       ▼
         OpenRouter (LLM)
```

**Six Docker services:**

| Service | Role | Port |
|---|---|---|
| `caddy` | Reverse proxy, auto-TLS | 80, 443 |
| `web-api` | FastAPI + Vue SPA | 8000 |
| `dagit` | Dagster orchestration UI | 3000 |
| `dagster-daemon` | Cron sensors + freshness checks | — |
| `dagster-postgres` | Dagster metadata + `climate_app` DB | 5432 (internal) |
| `qdrant` | Vector DB (~1.5 M chunks) | 6333, 6334 |

Optional: `jupyter` (profile `jupyter`) for ad-hoc notebooks.

---

## 📁 Project layout

```
xchomo_diploma/
├── web_api/
│   ├── main.py            # FastAPI app
│   ├── routes/            # auth, sources, catalog, rag, schedules, admin
│   ├── rag_endpoint.py    # /rag/query implementation
│   └── frontend/          # Vue 3 SPA (Pinia, Vue Router, Tailwind)
├── dagster_project/
│   ├── repository.py      # Definitions(jobs, schedules, sensors, resources)
│   ├── dynamic_jobs.py    # Legacy all-source ETL job
│   ├── source_jobs.py     # single_source_etl_job (used by sensors)
│   ├── catalog_jobs.py    # D1.1.xlsx catalog batch jobs
│   └── schedules.py       # Sensors: source_schedule_sensor, data_freshness_sensor
├── src/
│   ├── climate_embeddings/
│   │   ├── loaders/       # raster_pipeline.py (multi-format streaming)
│   │   ├── schema.py      # ClimateChunkMetadata
│   │   └── text_generation.py
│   ├── catalog/           # D1.1.xlsx ingest + phase classifier + portal adapters
│   ├── database/          # SQLAlchemy models, SourceStore (Postgres)
│   ├── embeddings/        # VectorDatabase (Qdrant wrapper)
│   ├── llm/               # OpenRouter client
│   └── sources/           # ClimateDataSource DTO + shim
├── config/
│   └── pipeline_config.yaml
├── docker/
│   ├── dagster.yaml       # Dagster instance config (sqlite storage in DAGSTER_HOME)
│   └── init-db.sh         # Creates climate_app DB in postgres on first boot
├── scripts/qdrant/        # snapshot.sh, restore.sh
├── backups/qdrant/        # .snapshot files + restore README
├── tests/                 # pytest suites (raster, catalog, text-gen, API, RAG eval)
├── Caddyfile
├── docker-compose.yml
├── Dockerfile             # Python 3.11 + GDAL/PROJ/GEOS + Node.js (frontend build)
├── DEMO.md                # Defense day demo script
└── stav_projektu.md       # Project status (Slovak)
```

---

## 🛠️ Common commands

```bash
# Stack lifecycle
docker compose up -d           # start / resume
docker compose down            # stop (volumes persist)
docker compose ps              # status
docker compose logs -f web-api # tail web-api logs

# Local dev (optional, without container rebuild)
make install-dev               # pip install -r requirements*.txt
make api                       # uvicorn on :8000 with reload
make dagit                     # dagster-webserver on :3000
make test                      # pytest tests/ -v
make test-coverage             # pytest --cov=src

# Frontend
cd web_api/frontend
npm run dev                    # vite on :5173 with proxy to :8000
npm run build                  # rebuild dist/ (docker compose serves this)

# Qdrant snapshots (see scripts/qdrant/ + backups/qdrant/README.md)
./scripts/qdrant/snapshot.sh
./scripts/qdrant/restore.sh /qdrant/snapshots/climate_data/<file>.snapshot
```

---

## 🧠 Key design points

- **Memory-safe ingest.** `load_raster_auto()` streams chunks via xarray / dask / rasterio
  generators. No full raster is ever loaded into RAM.
- **Statistical summaries.** Each chunk → 8-dim stats vector (mean, std, min, max, p10, p50,
  p90, range) encoded into a human-readable description, then embedded by
  `BAAI/bge-large-en-v1.5` (1024-dim, COSINE).
- **Two storage systems.** Qdrant is the truth about *what is indexed*. PostgreSQL holds
  *management metadata* (schedules, credentials, processing history, sources).
- **Per-source scheduling.** `source_schedule_sensor` polls the `source_schedules` table
  every 60 s and dispatches `single_source_etl_job` runs. Dataset-level scheduling was
  intentionally removed (advisor decision).
- **LLM backend.** `src/llm/` implements an OpenRouter client; RAG
  gracefully returns raw hits if no LLM is configured.
- **Graceful cold start.** `web-api` waits for `dagit` healthcheck before starting,
  eliminating the "Dagster GraphQL error: All connection attempts failed" boot-time race.

---

## 🔒 Authentication

The Vue SPA is gated by a username/password prompt served at `/app/login`. Credentials
come from the `AUTH_USERNAME` / `AUTH_PASSWORD` environment variables. Successful login
returns a bearer token (stored in `localStorage`); sent back as
`Authorization: Bearer <token>` for protected endpoints.

Auth endpoints:

- `POST /auth/login` — `{username, password}` → `{success, token, username}`
- `POST /auth/logout` — invalidates the token (header-based)
- `GET /auth/verify` — verifies a token (header-based)

`AUTH_PASSWORD` is **required** — if unset, `/auth/login` returns HTTP 503 with a clear
message so you know to configure it.

---

## 🧪 Testing

```bash
make test                  # all suites
make test-raster           # raster pipeline only
make test-api              # FastAPI endpoints
make test-coverage         # coverage report

# Single test
docker compose exec web-api pytest tests/test_raster_pipeline_flow.py::TestNetCDF::test_load_basic -v
```

Coverage target is pragmatic (~21 %) and focused on the raster pipeline, which is the
core academic contribution.

---

## 🗂️ Related documents

- **[DEMO.md](./DEMO.md)** — step-by-step defense demo script
- **[CLAUDE.md](./CLAUDE.md)** — architecture cheat sheet (for LLM assistants)
- **[stav_projektu.md](./stav_projektu.md)** — detailed project status (Slovak)
- **[docs/](./docs/)** — extended research notes, embedding strategy papers
- **[Caddyfile](./Caddyfile)** — reverse-proxy rules
- **[backups/qdrant/README.md](./backups/qdrant/README.md)** — snapshot restore procedure

---

## 🌐 Deployment

Hosted on a Digital Ocean droplet (`159.65.207.173`) behind Caddy with automatic
Let's Encrypt HTTPS for `climaterag.online`.

```bash
# On the server
git pull
docker compose build --no-cache
docker compose up -d
```

---

## 📝 License

MIT — see [LICENSE](LICENSE).

## 👤 Author

Martin Chomoň — Mendel University in Brno, Faculty of Business and Economics.

---

*Status: production-ready for defense, April 2026.*
