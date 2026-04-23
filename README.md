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

# 1a. Pull the rag-mendelu submodule (required — used by all Docker services)
git submodule update --init --recursive

# 1b. Create your .env from the example
cp .env.example .env
# Open .env and set:
#   AUTH_USERNAME / AUTH_PASSWORD_HASH  (generate hash: python scripts/hash_password.py)
#   JWT_SECRET_KEY                      (any random 64+ char string)
#   OPENROUTER_API_KEY                  (for LLM answers)
# AUTH_PASSWORD (plaintext) is accepted as a dev fallback if AUTH_PASSWORD_HASH is unset.

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

Protected endpoints require a bearer JWT. Log in once, then reuse `$TOKEN`:

```bash
TOKEN=$(curl -sX POST http://localhost:8000/auth/login \
  -H 'Content-Type: application/json' \
  -d '{"username":"admin","password":"YOUR_PASSWORD"}' | jq -r .token)

# Add an open-access CSV source (no auto-embed)
curl -X POST http://localhost:8000/sources -L \
  -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{
    "source_id": "co2_mauna_loa",
    "url": "https://raw.githubusercontent.com/datasets/co2-ppm/master/data/co2-mm-mlo.csv",
    "format": "csv",
    "description": "Monthly mean CO2 at Mauna Loa",
    "hazard_type": "Atmospheric composition",
    "auto_embed": false
  }'

# Ask a question via RAG (use_llm=false returns chunks only — no OpenRouter needed)
curl -X POST http://localhost:8000/rag/chat \
  -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"question": "What precipitation data is available for Slovakia?", "top_k": 3, "use_llm": false}'
```

---

## 📦 Git Submodule: rag-mendelu

This project depends on **[rag-mendelu](https://github.com/dvlastnik/rag-mendelu)** (branch `integration-to-climate-rag`) as a git submodule, mounted into all Docker services at `/app/rag-mendelu`.

**After cloning, you must initialise the submodule before running the stack:**

```bash
git submodule update --init --recursive
```

If you already cloned without `--recurse-submodules` and the `rag-mendelu/` directory is empty, run the same command to populate it.

To update the submodule to the latest commit on its tracking branch:

```bash
git submodule update --remote rag-mendelu
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
│   ├── routes/            # auth, sources, catalog, rag (prefix /rag), schedules,
│   │                      # admin, adapters, embeddings, health, qdrant_datasets,
│   │                      # frontend (SPA static serving)
│   ├── rag_endpoint.py    # internal helpers used by web_api/routes/rag.py
│   ├── prompt_builder.py  # RAG prompt assembly (incl. unit-conversion guidance)
│   └── frontend/          # Vue 3 SPA (Pinia, Vue Router, Tailwind)
├── dagster_project/
│   ├── repository.py      # Definitions(jobs, schedules, sensors, resources)
│   ├── dynamic_jobs.py    # Legacy all-source ETL job
│   ├── source_jobs.py     # single_source_etl_job (launched by sensor / manual trigger)
│   ├── catalog_jobs.py    # D1.1.xlsx catalog batch jobs
│   ├── resources.py       # Dagster resources (DB, settings)
│   └── schedules.py       # source_schedule_sensor (30 s), data_freshness_sensor,
│                          # weekly_catalog_refresh
├── src/
│   ├── climate_embeddings/
│   │   ├── loaders/       # raster_pipeline.py (multi-format streaming)
│   │   ├── schema.py      # ClimateChunkMetadata
│   │   └── text_generation.py
│   ├── catalog/           # D1.1.xlsx ingest + phase classifier + portal adapters
│   ├── database/          # SQLAlchemy models, SourceStore, advisory-lock helper
│   ├── embeddings/        # VectorDatabase (Qdrant wrapper)
│   ├── llm/               # OpenRouter client
│   ├── sources/           # ClimateDataSource DTO + connection_tester (SSRF guard)
│   └── utils/             # persisted_creds (UI → os.environ for Dagster ops)
├── config/
│   └── pipeline_config.yaml
├── docker/
│   ├── dagster.yaml       # Dagster instance config (sqlite storage in DAGSTER_HOME)
│   └── init-db.sh         # Creates climate_app DB in postgres on first boot
├── scripts/
│   ├── qdrant/            # snapshot.sh, restore.sh
│   └── hash_password.py   # Generate AUTH_PASSWORD_HASH (argon2id via pwdlib)
├── backups/qdrant/        # .snapshot files + restore README
├── tests/                 # pytest suites (raster, catalog, text-gen, API, RAG eval)
├── Caddyfile
├── docker-compose.yml
├── Dockerfile             # Python 3.11 + GDAL/PROJ/GEOS + Node.js (frontend build)
├── LIMITATIONS.md         # Documented scope decisions / known limits
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
  every 30 s and dispatches `single_source_etl_job` runs. A Postgres advisory lock keyed
  on `source_id` prevents the sensor and a manual `/sources/{id}/trigger` from racing.
  Dataset-level scheduling was intentionally removed (advisor decision).
- **LLM backend.** `src/llm/openrouter_client.py` (default model
  `anthropic/claude-sonnet-4.6`); the chat endpoint can run with `use_llm=false` and
  return raw retrieved chunks if the key is missing or OpenRouter is misbehaving.
- **Graceful cold start.** `web-api` waits for `dagit` healthcheck before starting,
  eliminating the "Dagster GraphQL error: All connection attempts failed" boot-time race.

---

## 🔒 Authentication

The Vue SPA is gated by a username/password prompt served at `/app/login`. Auth is
**stateless JWT** (PyJWT, HS256) signed with `JWT_SECRET_KEY`; passwords are stored as
**argon2id** hashes (`pwdlib`, OWASP-2024 parameters). The token is kept in
`localStorage` and sent as `Authorization: Bearer <token>` for protected endpoints.

Relevant env vars (see `.env.example`):

- `AUTH_USERNAME` — single admin account
- `AUTH_PASSWORD_HASH` — argon2id hash; generate with `python scripts/hash_password.py`
  and wrap in single quotes inside `.env` so docker-compose doesn't expand `$`
- `AUTH_PASSWORD` — plaintext fallback, only used if `AUTH_PASSWORD_HASH` is empty
- `JWT_SECRET_KEY` — HS256 signing key (falls back to `AUTH_PASSWORD_HASH` if unset)
- `AUTH_TOKEN_TTL_SECONDS` — token lifetime, default 24 h
- `AUTH_LOGIN_MAX_FAILURES` / `AUTH_LOGIN_WINDOW_SECONDS` — per-IP rate limit (defaults
  10 / 900 s), failure counters live in the Postgres `login_failures` table
- `AUTH_ALLOW_ANONYMOUS=1` — opt-in bypass for CI / smoke tests; refused in
  `APP_ENV=production`

Endpoints:

- `POST /auth/login` — `{username, password}` → `{success, token, username, ...}`
- `POST /auth/logout` — client-side convenience; JWT remains valid until `exp`
- `GET /auth/verify` — verifies a bearer token

If neither `AUTH_PASSWORD_HASH` nor `AUTH_PASSWORD` is set, `/auth/login` and every
protected endpoint return HTTP 503. Rotate `JWT_SECRET_KEY` to invalidate all live
tokens (there is no server-side revocation list — see `LIMITATIONS.md`).

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

108 tests collected across the suite; line coverage is ~11 % and
deliberately focused on the raster pipeline, which is the core academic
contribution.

---

## 🗂️ Related documents

- **[LIMITATIONS.md](./LIMITATIONS.md)** — documented scope decisions and known limits
- **[DEMO.md](./DEMO.md)** — step-by-step defense demo script
- **[DEMO_URLS.md](./DEMO_URLS.md)** — sample sources to add live during the demo
- **[CLAUDE.md](./CLAUDE.md)** — architecture cheat sheet (for LLM assistants)
- **[stav_projektu.md](./stav_projektu.md)** — detailed project status (Slovak)
- **[docs/](./docs/)** — extended research notes; canonical evaluation is
  `docs/rag_eval_v2_clean_final.md` (other `rag_eval_*` files are iteration history)
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
