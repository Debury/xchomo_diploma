# Stav projektu — Climate Data RAG Pipeline

> **Projekt**: ClimateData AI Assistant
> **Autor**: Martin Chomoň
> **Dátum**: Apríl 2026
> **Status**: Production-ready, nasadené cez Docker Compose na `climaterag.online`

---

## 1. Prehľad systému

Systém umožňuje:
1. **Pridať klimatické dátové zdroje** (URL / súbor) — NetCDF, GRIB, HDF5, GeoTIFF, CSV, Zarr, ASCII Grid, ZIP/TAR archívy
2. **Automaticky spracovať** dáta cez ETL pipeline (stiahnutie → detekcia formátu → chunking → štatistiky → embedding → uloženie)
3. **Sémanticky vyhľadávať a odpovedať** na otázky v prirodzenom jazyku (RAG pattern)

### Architektúra

```
Caddy (80/443) ──→ FastAPI (8000) + Dagit (3000)
                        ↓                ↓
                  Vue 3 SPA        Dagster ETL jobs
                        ↓                ↓
                 Qdrant (6333)    PostgreSQL (5432)
                        ↓
                 OpenRouter (LLM)
```

**Štyri hlavné subsystémy:**
- **`web_api/`** — FastAPI REST API + Vue 3 SPA frontend (Pinia + Vue Router)
- **`dagster_project/`** — ETL orchestrácia (Dagster, asset-orientovaný prístup, sensor pre per-source plány)
- **`src/`** — Jadro: multi-formátový raster pipeline, embedding, RAG, LLM klienti
- **`src/catalog/`** — Batch spracovanie Excel katalógu (233 záznamov, 69 unikátnych datasetov)

---

## 2. Docker mikroservisná architektúra (6 služieb + voliteľný Jupyter)

| Služba | Image | Účel | Port |
|--------|-------|------|------|
| `dagster-postgres` | `postgres:15` | Metadáta Dagster + zdroje, plány, runy | 5432 (internal) |
| `dagster-daemon` | Custom (Dockerfile) | Dagster plánovač + source_schedule_sensor + freshness sensor | — |
| `dagit` | Custom (Dockerfile) | Dagster Web UI pre monitorovanie ETL | 3000 |
| `web-api` | Custom (Dockerfile) | FastAPI REST API + Vue SPA | 8000 |
| `qdrant` | `qdrant/qdrant:latest` | Vektorová databáza (1024-dim, COSINE) | 6333, 6334 |
| `caddy` | `caddy:2-alpine` | Reverzný proxy, automatické HTTPS | 80, 443 |
| `jupyter` | `jupyter/scipy-notebook` | Interaktívna analýza (profile: jupyter) | 8888 |

Sieť: `climate-net` (bridge). Volumes: `dagster-postgres-data`, `dagster-home`, `qdrant-data`, `caddy-data`, `caddy-config`.

---

## 3. Čo funguje (funkčné celky)

### 3.1 ETL Pipeline
- **Multi-formátový raster loader** (`src/climate_embeddings/loaders/raster_pipeline.py`): NetCDF, GRIB, HDF5, GeoTIFF, ASCII Grid, CSV, ZIP, TAR, Zarr (cez xarray fallback)
- **Generator-based chunk streaming** — nikdy nenačíta celý súbor do RAM
- **Štatistické súhrny** per chunk: mean, std, min, max, p10, median, p90, range
- **Text generation** z metadát a štatistík (`text_generation.py`, `schema.py`)
- **Embedding** cez sentence-transformers (BAAI/bge-large-en-v1.5, 1024-dim)
- **Uloženie** do Qdrant s kompletným payload metadát (ClimateChunkMetadata)
- **Dagster orchestrácia**: `dynamic_source_etl_job`, `single_source_etl_job`
- **Batch katalóg**: 5-fázové spracovanie Excel katalógu (Phase 0–4)
- **Format detection**: rozšírené o `.zarr`, `.tsv`, `.asc`, `.tar`, `.grb2`, `.he5`

### 3.2 RAG System
- **Vektorové vyhľadávanie** v Qdrant (kosínusová podobnosť)
- **Payloadové filtrovanie** (čas, lokalita, premenná, zdroj)
- **Kontextová konštrukcia** (`prompt_builder.py`) — formátovanie chunk metadát pre LLM
- **LLM backend**: OpenRouter (Claude Sonnet 4.6 ako default model)
- **Graceful fallback** — systém vráti raw výsledky ak LLM nie je dostupné
- **Počet embeddingov v Qdrant**: ~1.53M chunks v kolekcii `climate_data`

### 3.3 Katalógový modul
- **Excel reader** (`excel_reader.py`) — openpyxl, CatalogEntry, merged cell handling
- **Phase classifier** — klasifikácia 233 záznamov do fáz 0–4
- **Phase 0** — Metadata-only embedding (všetky záznamy, bez sťahovania)
- **Phase 1** — Direct HTTP download + raster spracovanie
- **Phase 3** — API portal adaptery (CDS, ESGF, NASA, CMEMS, CEDA, EIDC, NOAA)
- **Location enricher** — extrakcia/normalizácia lokačných informácií
- **Resume support** a progress tracking

### 3.4 Web Frontend (Vue 3 SPA)
- **Autentifikácia** (`/app/login`): hardcoded admin cez .env (AUTH_USERNAME / AUTH_PASSWORD), token-based session, bezpečné error handling (401 pri zlých credentialoch, nechytá sa na string "undefined" v localStorage)
- **Dashboard** (`/app`) — prehľad zdrojov, Qdrant štatistiky, service health, greeting
- **Chat** (`/app/chat`) — RAG dopytovanie s filtrami (source, variable) a zobrazením retrieved chunks
- **Sources** (`/app/sources`) — prehliadanie, filtre, detail modal, edit modal (metadata + schedule), reprocess, delete embeddings
- **Create Source** (`/app/sources/create`) — 5-krokový wizard: URL → Auth → Config → Schedule → Review; automatická detekcia formátu a portálu; skenovanie metadát z NetCDF headers
- **Catalog** (`/app/catalog`) — prehliadanie Excel katalógu, spúšťanie phase processingu
- **ETL Monitor** (`/app/etl`) — real-time progress, log viewer
- **Schedules** (`/app/schedules`) — **per-source plány** s cron picker, Create/Edit/Remove modal
- **Settings** (`/app/settings`) — LLM config, portal credentials (CDS, NASA, CMEMS, ESGF, NOAA, EIDC), embedding model info, system resources
- **Navigácia** obsahuje všetky hlavné stránky (Dashboard, Chat, Sources, Catalog, ETL Monitor, Schedules, Settings)
- **Light/Dark theme toggle** s Tailwind CSS

### 3.5 Plánovanie (Scheduling)
- **Zrnitosť**: **per-source** cron schedule (dohodnuté so zadávateľom). Žiadne per-dataset schedules — odstránené z backendu aj frontendu.
- **Backend**: `PUT/DELETE /sources/{id}/schedule` cez `SourceStore.set_schedule` (validuje cron cez croniter, ukladá do `source_schedules` tabuľky v PostgreSQL)
- **Dagster sensor** (`source_schedule_sensor`, minimum_interval=60s) polling PostgreSQL pre due schedules a spúšťa `single_source_etl_job` s tagom `source_id`
- **Freshness sensor** detekuje stale sources (>30 dní) a automaticky spúšťa reprocess
- **Per-source schedule UI**: Schedules page (Create, Edit, Remove) a aj Source edit modal (CronPicker komponent)

### 3.6 Infraštruktúra
- **Docker Compose** s multi-stage Dockerfile (Python + GDAL/PROJ/GEOS + Node.js frontend build)
- **Caddy** reverzný proxy s automatickým HTTPS (Let's Encrypt)
- **Nasadenie**: Digital Ocean droplet, doména `climaterag.online`
- **Deploy workflow**: `git pull && docker compose build --no-cache && docker compose up -d`
- **GitHub Actions CI**: kompiluje thesis Typst dokumenty na PRs
- **docker compose up** je čistý — všetkých 6 služieb nabehnie (postgres healthy → daemon/dagit/web-api closed loop), žiadne kritické warnings/errors pri čistom štarte

---

## 4. Najnovšie opravy (Apríl 2026, production readiness pass pre obhajobu)

### Login / Auth
- Backend `/auth/login` teraz vracia **HTTP 401** pri zlých credentialoch (predtým vracal HTTP 200 so `success=false`, čo frontendu uletelo ako "úspech" a uložil `undefined` token do localStorage)
- Autentifikácia používa `hmac.compare_digest` pre timing-safe porovnanie
- Pridaný 503 error ak `AUTH_PASSWORD` nie je nakonfigurovaný
- Frontend auth store (`stores/auth.js`): explicitne kontroluje `data.success && data.token`, filtruje stringy `"undefined"` a `"null"` z localStorage (pre staré sessions)
- Login view: `await router.replace('/')` + `window.location.assign('/app/')` fallback ak Vue Router navigácia zlyhá; error správy sú viditeľné a popisné

### Scheduling zjednodušené na per-source
- Odstránené dataset-level endpointy `/schedules/datasets` (POST/GET/DELETE)
- Odstránená `DatasetSchedule` logika zo `source_store.py` a `source_schedule_sensor`
- Schedules.vue prerobené: jeden "Create Schedule" modal s výberom source + cron + enabled toggle
- CreateSource.vue wizard "Schedule" krok zjednodušený na jeden toggle + CronPicker

### Formát detection
- `connection_tester.EXT_FORMATS` rozšírené o `.zarr`, `.tsv`, `.asc`, `.tar`, `.grb2`, `.he5`

### Navigácia
- Catalog page pridaná do navigačného baru (predtým bola dostupná iba cez priamy URL)
- Orphaned `Embeddings` view odstránené (duplicita s Dashboard/Settings, zobrazovalo fake sample dáta)

### Docker infraštruktúra
- Prechod z deprecated `dagit` CLI na `dagster-webserver` (Dagster 2.0-ready)
- Pridaný `docker/dagster.yaml` (sqlite storage vo `/opt/dagster/dagster_home/` volume) — odstránené warningy "No dagster.yaml found"
- Dagit má healthcheck (`GET :3000`) a `web-api` naň teraz čaká cez `depends_on: service_healthy` — odstránená race condition "Dagster GraphQL error: All connection attempts failed" pri coldstarte
- Odstránené nepotrebné `DAGSTER_POSTGRES_*` env vars (neboli použité keďže dagster-postgres pip package nie je nainštalovaný, čo spôsobovalo ModuleNotFoundError warningy)
- Caddyfile vyčistené (odstránené duplicitné `X-Forwarded-For` / `X-Forwarded-Proto` header_up — reverse_proxy ich posiela defaultne)
- `docker compose down && docker compose up -d` teraz nabehne **bez jedinej chybovej hlášky** okrem NVIDIA CUDA GPU warning (irelevantné na tomto stroji, base image je `nvidia/cuda:12.4.1-runtime`)

### Input validation a error handling
- `POST /sources/` validuje `source_id` + `url` nesmú byť prázdne (400), odmietne duplicitný `source_id` (409)
- `DELETE /sources/{id}` je idempotent — neexistujúci source vráti 204 (predtým 500 kvôli broken shelve fallback)
- `POST /rag/chat` validuje `question` nie je prázdna (400, predtým 500 kvôli re-wrapped HTTPException)
- `PUT /sources/{id}/schedule` validuje cron expression cez croniter (400 ak je invalid)

### Security hardening
- **Full auth middleware**: všetky API routery (sources, rag, catalog, schedules, admin, embeddings, qdrant_datasets) chránené cez `Depends(require_auth)` — bez valid Bearer tokenu vracajú 401. Public sú iba `/health`, `/auth/*`, `/app/*`, `/` a `/docs`.
- `/auth/logout` a `/auth/verify` prijímajú token cez `Authorization: Bearer <token>` header, už NIE cez query param (`?token=...`) — predchádzajúci návrh exponoval tokeny v browser history a access logoch
- Frontend má `web_api/frontend/src/api.js` wrapper nad `fetch()` ktorý automaticky injektuje `Authorization: Bearer` header a redirectuje na /login pri 401 (token expired / server restart).
- Login používa `hmac.compare_digest` pre timing-safe porovnanie
- `/auth/login` vracia 503 ak `AUTH_PASSWORD` env var chýba (predtým silently whitelisted empty passwords)
- Ak je `AUTH_PASSWORD` prázdne, middleware dep `require_auth` no-opuje — local dev bez secrets stále funguje
- Settings credentials endpoint odfiltruje obvious placeholder strings (`your-`, `CHANGE_ME`, `REPLACE_ME`) aby neoznačoval placeholder secrets ako "Configured"

### UX / error handling
- `ErrorBoundary.vue` komponent obaľuje `<router-view>` — ak view hodí exception pri renderovaní, zobrazí kartu s error message + retry / back-to-dashboard buttons namiesto bieleho screenu
- Globálny `app.config.errorHandler` a `window.unhandledrejection` listener logujú exception do console pre demo-day debugging
- `Chat.vue` kontroluje `resp.ok` pred `resp.json()` — už nekrašne na LLM 500 / HTML error pages
- `Chat.vue` má `onUnmounted` cleanup pre loading timer — žiadny leak pri navigácii počas RAG query
- `Catalog.vue` progress poll má hard cap (400 tickov = ~20 min) a zastaví sa pri `thread_crashed`, čistí interval pri unmount — už nespamuje requesty forever pri stuck batchi
- Login view zobrazuje špecifické správy pre 401 (Invalid creds) vs 503 (auth not configured)
- `/catalog/progress` už neleakuje plný Python traceback do `thread_error` — frontend vidí len generic "Batch processing crashed. Check server logs for details.", plný traceback ide len do server logov

### Post-audit security hardening (April 2026)
- CORS middleware: keď `CORS_ORIGINS` nie je explicitne nastavené, defaultuje sa na same-origin (prázdny zoznam) namiesto risky `*` + `credentials=true`; ak je `CORS_ORIGINS` nastavené s explicitnými originmi, credentials sa povolia, inak nie
- Bearer token parser: odmieta čokoľvek iné než `Bearer <token>` formát (predtým akceptoval aj bare strings bez prefixu — auth bypass risk pri mal-configured clientovi)
- Sensor `source_schedule_sensor` používa stabilný `run_key` odvodený zo `next_run_at` (nie `utcnow()`) — Dagster's run-key dedup teraz chráni pred duplicitným spustením ak sa sensor re-evalujuje medzi yield a advance_schedule
- `process_single_source_op` wrappuje `update_processing_status` volania v try/except, aby DB hiccup neorphanol source v stave "processing"

### Performance
- **Qdrant payload indices** pridané pre `source_id`, `dataset_name`, `hazard_type`, `variable` (keyword type). Zásadne zrýchľuje filtrované RAG vyhľadávanie na 1.5M chunks. Index create je idempotent, `_ensure_payload_indices` beží pri `VectorDatabase.__init__`.
- `UVICORN_RELOAD=0` env var umožňuje vypnúť `--reload` v produkcii (defaulty zostávajú `=1` pre thesis iteration)
- **Idle-aware polling**: MainLayout, ETLMonitor, ActivityFeed registrujú `visibilitychange` listener — keď je tab skrytý, polling sa zastaví (šetrí bandwidth a server load pri demo, keď máte na pozadí otvorených 10 tabov)
- **Router re-mount thrash fix**: `<router-view :key="$route.fullPath" />` odstránené — Dashboard/Chat/Sources sa už nere-mountujú na každú navigáciu naspäť (predtým každý návrat na Dashboard fetchoval 3 endpointy znova)
- **Debounced search** na Sources.vue (250ms): filter sa už neprekomputuje na každý stisk klávesy
- **Loading skeleton** v `index.html` (3 bouncing dots + "Loading ClimateRAG…") sa zobrazí hneď pri page-load, pred stiahnutím JS bundle-u; zmizne automaticky keď Vue mountuje na `#app`
- **Vendor chunk splitting**: `vue` + `vue-router` + `pinia` sú teraz v separátnom `vendor-*.js` chunku (41 KB gzip) ktorý sa cachuje cez PR; hlavný app-specific kód má iba 2.5 KB gzip

### TypeScript migrácia (Apríl 2026)
Celý frontend prešiel z JavaScript na TypeScript — štandard pre moderné Vue 3 projekty.

**Zmeny:**
- `package.json`: pridané `typescript`, `vue-tsc`, `@vue/tsconfig`, `@types/node`
- `tsconfig.json`: vychádza z `@vue/tsconfig/tsconfig.dom.json`, `strict: false` + `noImplicitAny: false` (pragmatická konfigurácia — compile-time checks bez blokovania na každý `any`)
- `shims-vue.d.ts`: shim pre .vue imports
- **Všetky infrastruktúrne súbory** premenované: `main.js` → `main.ts`, `router.js` → `router.ts`, `api.js` → `api.ts`, `stores/auth.js` → `stores/auth.ts`, `stores/theme.js` → `stores/theme.ts`, `composables/useCountUp.js` → `useCountUp.ts`, `vite.config.js` → `vite.config.ts`
- **Všetky .vue súbory** (10 views + 11 components + MainLayout + App): `<script setup>` → `<script setup lang="ts">`
- Kľúčové typy pridané:
  - `AuthUser`, `LoginSuccessResponse`, `LoginErrorResponse` v `stores/auth.ts`
  - `Theme` v `stores/theme.ts`
  - `ApiInput`, `ApiInit` v `api.ts`
  - `DonutSegment`, `EditableSettings` pre komponenty
  - `PropType<...>` anotácie pre typed defineProps
- Build skript: `vue-tsc --noEmit && vite build` — strict type-check pred buildom; fallback `npm run build:skip-types` ak treba obísť type-check
- `npm run type-check` samostatný príkaz pre CI/local check

**Výhody pre obhajobu:**
- Ukáže že projekt dodržiava modernú Vue 3 best-practice
- IDE auto-complete a refactor safety
- Compile-time chytená jedna kategórie bugov (napr. `$event.target.value` bez typeguard bolo hneď pri migrácii odhalené)

### Critical bugfix: circular import v DB backend
- `src/database/source_store.py` mal circular import s `src.sources.__init__` — importoval `ClimateDataSource` z `src.sources`, ktorý sa zas inicializoval podľa `APP_DATABASE_URL`
- Výsledkom bol **silent fallback na shelve-based SourceStore** (in-memory dict v kontajneri, NEPRETRVÁVAL cez restart!)
- Sources CRUD cez `/sources` tak nešiel do Postgres → po reštarte kontajnera sa stratili všetky zdroje vytvorené cez frontend (schedules zostali, lebo pg_store sa používal explicitne)
- Fix: `from __future__ import annotations` + lazy import `ClimateDataSource` iba vo `_to_dto()`
- Po fixe všetky sources CRUD operácie idú do Postgres a perzistujú cez reštarty kontajnera

### Docker infraštruktúra
- `qdrant/qdrant:latest` → `qdrant/qdrant:v1.17.1` (pinned na konkrétnu verziu, ochrana pred breaking update)
- `jupyter/scipy-notebook:latest` → `jupyter/scipy-notebook:2024-04-15`
- `.env.example` obsahuje `AUTH_USERNAME`/`AUTH_PASSWORD`, `APP_DATABASE_URL`, `DAGSTER_HOME` (predtým chýbali — user nevedel, čo musí nastaviť)
- `.env` pôvodný `CDS_API_KEY=your-personal-access-token-here` placeholder odstránený (bol označovaný ako "configured" v Settings UI)

### UX vylepšenia
- Chat.vue loading indicator má progresívne fázové texty ("Searching Qdrant…" → "Reranking results…" → "Asking the LLM…") + counter sekúnd + hint o očakávanom trvaní (15–25 s). Bez toho UI vyzeralo zaseknuté pri 20s RAG latencii.
- Settings.vue "Test Connection" má 30s AbortController timeout — ak LLM spadne, button sa neflacne indefinitely
- SourceResponse pydantic model rozšírený o `etl_run_id` a `etl_error` (predtým FastAPI silently filtered ich out)

### Robustness pass pre multi-source flow (Apríl 2026)
Trojitý paralelný audit (lifecycle, concurrency, silent-errors) pred obhajobou identifikoval šesť kritických problémov v reálnom user-adds-sources flow. Všetky opravené:

- **Atomic versioned re-ingest** — každý chunk má v Qdrant payload `ingestion_run_id` (= Dagster `context.run_id`). Po úspešnom upserte sa zmažú všetky stale chunky toho istého `source_id` s iným run_id. Rieši duplikátne chunky pri Reprocess aj pri scheduled refresh (doteraz každý re-run pridával kópiu bez zmazania starej). Queries počas re-ingestu vidia starú verziu; failed run nechá starú verziu nedotknutú. Implementované cez `src/utils/ingestion_context.py` (contextvars) + `VectorDatabase.delete_by_source(source_id, exclude_run_id)`.
- **Portal adapter silent success-with-0-chunks fix** — `source_jobs.py` po návrate z portal adaptera volá `count_by_source(source_id, ingestion_run_id=current)`. Ak adaptér vrátil success ale nezapísal 0 chunks (napr. token refresh error caught internally), raise RuntimeError → outer except označí failed. Predtým sa source označoval "completed" s 0 chunks.
- **Startup reconciliation orphaned sources** — `web_api/main.py` má lifespan hook ktorý pri boote volá `store.reset_orphaned_processing(max_age_minutes=30)`. Sources stuck v "processing" (napr. po kontejner restarte / OOM killed Dagster run) sa označia "failed" s jasnou hláškou. Pre postgres store použitý age-based filter, pre shelve fallback blanket-reset.
- **Persist trigger errors** — v `create_source` keď `launch_dagster_run()` zlyhá (daemon down, GraphQL error), error sa zapíše do DB cez `update_processing_status("failed", error_message=...)`. Predtým bol error viditeľný iba v immediate response DTO a zmizol pri refreshi stránky.
- **Server-side credential validation** — pred spustením Dagster jobu pre portal source (CDS, NASA, MARINE) sa skontroluje, či sú required creds v `app_settings.json` (alebo per-source). Ak chýbajú, job sa nespustí; source sa označí failed s hláškou "Cannot start ETL: {portal} source requires credentials that are not configured". Frontend `PORTAL_CREDENTIAL_KEYS` v CreateSource.vue je synced s backendom.
- **HTTP session cleanup** — direct-HTTP download v `source_jobs.py` wrappe `requests.Session()` do `try/finally: http_session.close()`. Pod repeated add-then-fail cyklmi sa socket pool už nedostáva do FD exhaustion.

### Čistenie dead code / files
- Odstránené `web_api/frontend/spa.html`, `styles.css` (legacy chat UI nahradené Vue SPA)
- Odstránené `/ui`, `/chat` endpointy z `routes/frontend.py` (legacy static files už neexistujú)
- Odstránený static mount `/ui/static` z `main.py`
- Odstránené duplikáty `Kopie souboru D1.1.xlsx` v root a `web_api/` (ponechaná iba kanonická `data/Kopie souboru D1.1.xlsx`)
- Odstránený prázdny folder `data;C/` (artefakt shell escape bugu)
- Root redirect `GET /` → 307 `/app/` (bol JSON alebo FileResponse, teraz jednotný redirect)
- Odstránený `/settings/credentials` dataset_schedule nepoužívaný kód

---

## 5. V procese / zostáva

### Dokončené (2026-04)
- **Evaluation kapitola** — metriky realizované, výsledky v `docs/rag_eval_v2_clean_final.md`:
  Context Relevance 98 %, Faithfulness 99 %, Overall Composite **89 %**, pass-rate 10/10 (top_k=10, reranker=off, Claude Sonnet 4.6)
- **Test set** — 10 golden queries (`tests/fixtures/golden_queries.json`) naprieč hazard kategóriami

### Chýba / TODO
- **Rozšírenie test setu** nad 10 golden queries (plánovaných 50–100 query-answer párov)
- **Baseline porovnania** — BM25, no-RAG LLM, alternatívne embedding modely (plánované)
- **Test coverage** je ~11 % (zámerne zacielené na raster pipeline)
- **Ablation studies** (chunk size, top-k, metadata enrichment)
- **Inter-annotator agreement** pre test set

### Technický dlh / drobnosti na vyčistenie
- Duplicitný súbor `Kopie souboru D1.1.xlsx` v root, `web_api/` a `data/` — použiť iba `data/` verziu
- Prázdny adresár `data;C/` v root (pravdepodobne omyl shellu pri raste)
- `src/database/models.py::DatasetSchedule` model zostal v DB migračných schémach — v kóde sa už nepoužíva, len histó­rická pripomienka; pri novej migrácii ho odstrániť
- Periodické warningy `dagster.code_server - WARNING - No heartbeat received in 20 seconds, shutting down` — interné Dagster správanie gRPC code servera (neblokujúce, sensor beží ďalej), zvážiť `DAGSTER_GRPC_SERVER_HEARTBEAT` override
- Voliteľný `legacy-index.html` + `spa.html` vo frontende — zvážiť odstránenie, live SPA je `dist/index.html`

---

## 6. Tech stack

| Vrstva | Technológie |
|--------|------------|
| Dátové spracovanie | xarray, rasterio, pandas, dask, numpy, cfgrib, netCDF4, h5netcdf |
| ML / Embeddings | sentence-transformers (bge-large-en-v1.5), torch, transformers |
| Vektorová DB | Qdrant latest, qdrant-client |
| Orchestrácia | Dagster, dagit (sensor-driven scheduling) |
| Web API | FastAPI, uvicorn, pydantic, SQLAlchemy |
| Frontend | Vue 3, Vite 5, Pinia, Vue Router 4, Tailwind CSS |
| LLM | OpenRouter (Claude Sonnet 4.6 default) |
| Infraštruktúra | Docker Compose, PostgreSQL 15, Caddy 2, GitHub Actions |

---

## 7. Testovanie

| Test suite | Súbor | Pokrytie |
|------------|-------|----------|
| Raster pipeline | `test_raster_pipeline_flow.py` | Multi-format loading, chunking |
| Catalog | `test_catalog.py` | Excel ingest, phase classifier |
| Text generation | `test_text_generation.py` | Metadata → text |
| Web API | `test_web_api.py` | FastAPI endpoints |
| RAG claims | `test_rag_claims.py` | Grounding / citations |
| RAG evaluation | `test_rag_evaluation.py` | RAGAS metrics (faithfulness, context recall) |

**Spustenie**: `make test` / `pytest tests/ -v`
**Počet testov**: 108 (pytest collect)
**Celkové pokrytie**: ~11 % line coverage (zámerne zacielené na raster pipeline)

### Smoke test (overené apríl 2026, cold start overený)
- `GET /health` → 200 `{"status":"healthy","dagster_available":true}`
- `POST /auth/login` (zlé creds) → 401 `{"detail":"Invalid username or password"}`
- `POST /auth/login` (dobré creds) → 200 `{"success":true,"token":"..."}`
- `POST /sources/analyze-url` pre NetCDF, CSV, GeoTIFF, GRIB, Zarr — všetky vrátia správny `format`
- `POST /sources/` + `DELETE /sources/{id}` — CRUD funkčné
- `PUT /sources/{id}/schedule` s validným cron → 200, s neplatným → 400
- `POST /rag/chat` na Qdrant s ~1.53M chunkami — vráti answer + chunks (~20s kvôli LLM roundtripu, RAG search <1s)
- `GET /schedules/datasets` → 404 (dataset-level schedules sú preč)
- `GET /ui` / `GET /chat` → 404 (legacy dead endpointy odstránené)
- E2E: login → analyze-url → create source → set schedule → update metadata → delete — všetky kroky OK
- `docker compose down && docker compose up -d` → všetkých 6 služieb healthy za ~60s, žiadne error logs (iba benign NVIDIA CUDA GPU warning z base image)

---

## 8. Kľúčové súbory

| Účel | Cesta |
|------|-------|
| Hlavný ETL job | `dagster_project/dynamic_jobs.py` |
| Dagster schedules / sensors | `dagster_project/schedules.py` |
| Raster pipeline | `src/climate_embeddings/loaders/raster_pipeline.py` |
| Format detection | `src/climate_embeddings/loaders/detect_format.py`, `src/sources/connection_tester.py` |
| RAG pipeline | `web_api/rag_endpoint.py` + `src/climate_embeddings/text_generation.py` |
| Schéma metadát | `src/climate_embeddings/schema.py` |
| Text generation | `src/climate_embeddings/text_generation.py` |
| FastAPI app | `web_api/main.py` |
| Auth endpoint | `web_api/routes/auth.py` |
| Sources + schedule endpoints | `web_api/routes/sources.py` |
| Prompt builder | `web_api/prompt_builder.py` |
| RAG endpoint | `web_api/routes/rag.py` |
| SourceStore (Postgres) | `src/database/source_store.py` |
| Vector DB wrapper | `src/embeddings/database.py` |
| Vue frontend vstup | `web_api/frontend/src/main.js` |
| Auth store | `web_api/frontend/src/stores/auth.js` |
| Schedules view | `web_api/frontend/src/views/Schedules.vue` |
| Katalóg orchestrátor | `src/catalog/batch_orchestrator.py` |
| Docker infraštruktúra | `docker-compose.yml` |
| Pipeline konfigurácia | `config/pipeline_config.yaml` |

---

*Aktualizované: 2026-04-18*
