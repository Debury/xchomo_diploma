# Stav projektu — Climate Data RAG Pipeline

> **Projekt**: ClimateData AI Assistant
> **Autor**: Martin Chomoň
> **Dátum**: Február 2026
> **Status**: Production-ready, nasadené cez Docker Compose

---

## 1. Prehľad systému

Systém umožňuje:
1. **Pridať klimatické dátové zdroje** (URL / súbor) — NetCDF, GRIB, HDF5, GeoTIFF, CSV, Zarr
2. **Automaticky spracovať** dáta cez ETL pipeline (stiahnutie → detekcia formátu → chunking → štatistiky → embedding → uloženie)
3. **Sémanticky vyhľadávať a odpovedať** na otázky v prirodzenom jazyku (RAG pattern)

### Architektúra

```
Caddy (80/443) ──→ FastAPI (8000) + Dagit (3000)
                        ↓                ↓
                  Vue.js frontend   Dagster ETL jobs
                        ↓                ↓
                 Qdrant (6333)    PostgreSQL (5432)
                        ↓
                 Ollama / Groq / OpenRouter (LLM)
```

**Štyri hlavné subsystémy:**
- **`web_api/`** — FastAPI REST API + Vue.js SPA frontend
- **`dagster_project/`** — ETL orchestrácia (Dagster, asset-orientovaný prístup)
- **`src/`** — Jadro: multi-formátový raster pipeline, embedding, RAG, LLM klienti
- **`src/catalog/`** — Batch spracovanie Excel katalógu (233 záznamov, 69 unikátnych datasetov)

---

## 2. Docker mikroservisná architektúra (7 služieb)

| Služba | Image | Účel | Port |
|--------|-------|------|------|
| `dagster-postgres` | `postgres:15` | Metadáta Dagster (história behov, plány) | 5432 |
| `dagster-daemon` | Custom (Dockerfile) | Dagster plánovač a senzor daemon | — |
| `dagit` | Custom (Dockerfile) | Dagster Web UI pre monitorovanie ETL | 3000 |
| `web-api` | Custom (Dockerfile) | FastAPI REST API + Vue.js frontend | 8000 |
| `qdrant` | `qdrant/qdrant:v1.11.0` | Vektorová databáza (1024-dim, COSINE) | 6333, 6334 |
| `caddy` | `caddy:2-alpine` | Reverzný proxy, automatické HTTPS | 80, 443 |
| `jupyter` | `jupyter/scipy-notebook` | Interaktívna analýza dát | 8888 |

Sieť: `climate-net` (bridge). Volumes: `dagster-postgres-data`, `dagster-home`, `qdrant-data`, `caddy-data`, `caddy-config`.

---

## 3. Čo funguje (funkčné celky)

### 3.1 ETL Pipeline
- **Multi-formátový raster loader** (`src/climate_embeddings/loaders/raster_pipeline.py`): NetCDF, GRIB, HDF5, GeoTIFF, ASCII Grid, CSV, Zarr, ZIP
- **Generator-based chunk streaming** — nikdy nenačíta celý súbor do RAM
- **Štatistické súhrny** per chunk: mean, std, min, max, p10, median, p90, range
- **Text generation** z metadát a štatistík (`text_generation.py`, `schema.py`)
- **Embedding** cez sentence-transformers (BAAI/bge-large-en-v1.5, 1024-dim)
- **Uloženie** do Qdrant s kompletným payload metadát (ClimateChunkMetadata)
- **Dagster orchestrácia**: `process_all_sources()` job s heartbeat mechanizmom
- **Batch katalóg**: 5-fázové spracovanie Excel katalógu (Phase 0–4)

### 3.2 RAG System
- **Vektorové vyhľadávanie** v Qdrant (kosínusová podobnosť)
- **Payloadové filtrovanie** (čas, lokalita, premenná)
- **Kontextová konštrukcia** (`prompt_builder.py`) — formátovanie chunk metadát pre LLM
- **3 LLM backendy**: Ollama (lokálne, granite4:3b), Groq (cloud), OpenRouter (multi-model)
- **Graceful fallback** — systém vráti raw výsledky ak LLM nie je dostupné

### 3.3 Katalógový modul
- **Excel reader** (`excel_reader.py`) — openpyxl, CatalogEntry, merged cell handling
- **Phase classifier** — klasifikácia 233 záznamov do fáz 0–4
- **Phase 0** — Metadata-only embedding (všetky záznamy, bez sťahovania)
- **Phase 1** — Direct HTTP download + raster spracovanie
- **Phase 3** — API portal adaptery (CDS, ESGF)
- **Location enricher** — extrakcia/normalizácia lokačných informácií
- **Resume support** a progress tracking

### 3.4 Web Frontend
- **Vue 3 SPA** s Vite + Tailwind CSS (dark theme)
- **Dashboard** (`/app`) — prehľad zdrojov, Qdrant štatistiky
- **Katalóg** (`/app/catalog`) — prehliadanie Excel katalógu, spúšťanie spracovania
- **ETL Monitor** (`/app/etl`) — real-time progress, log viewer
- **Chat** — RAG dopytovanie cez interaktívny chat
- **Schedules** (`/app/schedules`) — správa Dagster plánov
- **Settings** (`/app/settings`) — systémové informácie

### 3.5 Infraštruktúra
- **Docker Compose** s multi-stage Dockerfile (Python + GDAL/PROJ/GEOS + Node.js frontend build)
- **Caddy** reverzný proxy s automatickým HTTPS (Let's Encrypt)
- **Nasadenie**: Digital Ocean droplet, doména `climaterag.online`
- **Deploy workflow**: `git pull && docker compose build --no-cache && docker compose up -d`
- **GitHub Actions CI**: kompiluje thesis Typst dokumenty na PRs

---

## 4. V procese / zostáva

### Rozpracované
- **Evaluation kapitola** — definované metriky (Recall@k, MRR, Faithfulness), chýba realizácia experimentov
- **Test set** — navrhnutých 50–100 query-answer párov, zatiaľ nevytvorené
- **Baseline porovnania** — BM25, no-RAG LLM, alternatívne embedding modely (plánované)

### Chýba
- **Kvantitatívne výsledky** experimentov pre evaluation kapitolu
- **Test coverage** je 21% (primárne raster pipeline)
- **Ablation studies** (chunk size, top-k, metadata enrichment)
- **Inter-annotator agreement** pre test set

---

## 5. Tech stack

| Vrstva | Technológie |
|--------|------------|
| Dátové spracovanie | xarray, rasterio, pandas, dask, numpy, cfgrib, netCDF4, h5netcdf |
| ML / Embeddings | sentence-transformers (bge-large-en-v1.5), torch, transformers |
| Vektorová DB | Qdrant v1.11.0, qdrant-client |
| Orchestrácia | Dagster, dagit |
| Web API | FastAPI, uvicorn, pydantic |
| Frontend | Vue 3, Vite, Tailwind CSS |
| LLM | Ollama (granite4:3b), Groq API, OpenRouter API |
| Infraštruktúra | Docker Compose, PostgreSQL 15, Caddy 2, GitHub Actions |

---

## 6. Testovanie

| Test suite | Súbor | Pokrytie |
|------------|-------|----------|
| Raster pipeline | `test_raster_pipeline_flow.py` | Multi-format loading, chunking |
| Embeddings | `test_embeddings.py` | Qdrant integration |
| RAG components | `test_rag_components.py` | Pipeline flow |
| Text generation | `test_text_generation.py` | Metadata → text |
| Web API | `test_web_api.py` | FastAPI endpoints |
| Dagster | `test_dagster.py` | Job and op tests |

**Spustenie**: `make test` / `pytest tests/ -v`
**Celkové pokrytie**: ~21%

---

## 7. Kľúčové súbory

| Účel | Cesta |
|------|-------|
| Hlavný ETL job | `dagster_project/dynamic_jobs.py` |
| Raster pipeline | `src/climate_embeddings/loaders/raster_pipeline.py` |
| RAG pipeline | `src/climate_embeddings/rag/rag_pipeline.py` |
| Schéma metadát | `src/climate_embeddings/schema.py` |
| Text generation | `src/climate_embeddings/text_generation.py` |
| FastAPI app | `web_api/main.py` |
| Prompt builder | `web_api/prompt_builder.py` |
| RAG endpoint | `web_api/rag_endpoint.py` |
| Source management | `src/sources.py` |
| Vector DB wrapper | `src/embeddings/database.py` |
| Katalóg orchestrátor | `src/catalog/batch_orchestrator.py` |
| Docker infraštruktúra | `docker-compose.yml` |
| Pipeline konfigurácia | `config/pipeline_config.yaml` |

---

*Aktualizované: 2026-02-17*
