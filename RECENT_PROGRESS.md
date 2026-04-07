# Recent Development Progress

Summary of the last 7 commits (`132f27f` → `d119872`, Feb 17–22, 2026).
**66 files changed, +15 799 / −2 302 lines.**

---

## 1. What Was Done

### Qdrant Upsert Fix (`132f27f`)
- Large datasets (CMIP6, 500+ chunks) caused "Server disconnected" errors — upserts were sent one point at a time.
- **Fix:** batch upserts in groups of 50 with exponential-backoff retry (3 attempts).

### RAG Evaluation Suite + Batch Pipeline Hardening (`c94edaf`)
- Built a **3-tier RAG evaluation framework** with golden test sets, RAGAS metrics, and embedding space analysis.
- Created a golden test set of 10 seed queries with annotation guidelines.
- Batch orchestrator gained **crash recovery**, retry logic, and a memory guard so the pipeline survives OOM / network drops.
- Raster pipeline got optimized chunk processing with numpy pre-loading.
- Full Vue.js dark-theme UI refresh across all admin views.

### Spatial-Aware Retrieval (`d8c598e`)
- Queries now undergo **spatial intent extraction** — region names and lat/lon bounding boxes are parsed out.
- Qdrant `Range` filters on `latitude_min/max` ensure geographically correct retrieval.
- Fallback: if too few results pass the spatial filter, the system retries without it.
- Payload indexes added on Qdrant (lat/lon float, dataset_name/variable/source_id keyword) for faster filtered search.

### Portal Credentials & Per-Source Auth (`19a8e77`)
- Persistent settings storage (`data/app_settings.json`) for LLM config and portal credentials.
- New credentials API (`GET/PUT /settings/credentials`) with masked values in responses.
- Source CRUD accepts `auth_method`, `auth_credentials`, and `portal` fields.
- Frontend Settings page gets a Portal Credentials card; CreateSource modal gains auth method selector with portal presets.

### Batch Orchestrator & Frontend Overhaul (`15db662`)
- Batch orchestrator: improved progress tracking with per-phase breakdown and unique source status computation.
- Raster pipeline: better error handling for edge-case file formats.
- UI refresh across Catalog, Chat, Dashboard, ETL Monitor, Embeddings, Login, and Schedules views.

### Chunking & Upsert Optimization (`5b7b469`)
- **Temporal chunking:** 1 → 30 timesteps per chunk (~monthly aggregation) — drastically fewer, more meaningful chunks.
- **Spatial chunking:** 100 → 10 grid cells per chunk (~2.5° regional accuracy).
- **Upsert batch size:** 50 → 500 points per call (~10× fewer HTTP round-trips).
- Backed by research from Spatial-RAG, GeoGPT-RAG, and Geo-RAG papers.

### Source Management Refactor + Full-Phase Catalog (`d119872`)
- Fixed RAGAS `context_recall` (column name mismatch: `context_recall` vs `llm_context_recall_without_reference`).
- Fixed ZIP extraction fallback (system `unzip`/`7z` for unsupported compression methods).
- SSL verification fallback for downloads (retry with `verify=False` on SSLError).
- Connection timeout tuning (30 s connect, 600 s read) and skip for oversized datasets.
- Added **grounding/citation instructions** to all RAG prompt types for better faithfulness.
- Refactored source management into `src/database/` (PostgreSQL models, migrations) and `src/sources/` (connection tester).
- Added **cross-encoder reranking** evaluation tier with proper thresholds.
- Re-annotated golden queries against live **1.17 M-point** Qdrant collection.
- Dagster schedule management, source jobs, and resource configuration added.

---

## 2. Testing

### Test Files & What They Cover

| File | Focus | Key Classes |
|------|-------|-------------|
| `test_rag_evaluation.py` | 3-tier RAG quality evaluation | `TestRetrievalQuality`, `TestRetrievalQualityWithReranking`, `TestRAGASMetrics`, `TestEmbeddingSpace` |
| `test_catalog.py` | Catalog module (Excel, phases, metadata, locations) | `TestCatalogEntry`, `TestReadCatalog`, `TestPhaseClassifier`, `TestLocationEnricher`, `TestMetadataPipeline`, `TestBatchProgress`, `TestSchemaExtension` |
| `test_web_api.py` | FastAPI endpoints (async) | `TestBasicEndpoints`, `TestSourceDeletion`, `TestJobEndpoints`, `TestRunStatusEndpoints`, `TestCORS`, `TestSampleDownloads` |
| `test_raster_pipeline_flow.py` | Format auto-detection (NetCDF, GeoTIFF, CSV) | Parametrized test across 3 formats |
| `test_rag_components.py` | Embeddings, vector index, ZIP loader, RAG pipeline | `TestTextEmbeddings`, `TestVectorIndex`, `TestZIPLoader`, `TestRAGPipeline` |
| `test_dagster.py` | Dagster job definitions | Job structure tests |
| `test_embeddings.py` | Embedding model loading | Model initialization |
| `test_text_generation.py` | LLM text generation | Generation output tests |

### New Tests Added in These Commits

- **`test_rag_evaluation.py` (901 lines, brand new)** — the entire 3-tier evaluation suite:
  - **Tier 1 — Retrieval Quality:** Hit@5 ≥ 80%, Hit@10 ≥ 90%, MRR@10 ≥ 0.60, NDCG@10 ≥ 0.50, Recall@3/5/10.
  - **Tier 1b — With Cross-Encoder Reranking:** Same metrics with adjusted thresholds after reranking.
  - **Tier 2 — RAGAS End-to-End:** Faithfulness ≥ 0.30, Context Precision ≥ 0.60, Context Recall ≥ 0.30, Answer Relevancy ≥ 0.50, Numerical Coverage ≥ 0.70.
  - **Tier 3 — Embedding Space Analysis:** Intra-variable similarity, inter-variable separation, nearest-neighbor sanity checks.
- **`tests/fixtures/golden_queries.json` (1 811 lines)** — 10 seed queries annotated against the live 1.17 M-point collection.
- **`test_catalog.py` expanded (+157/−25 lines)** — new tests for location enrichment, batch progress (bulk operations, interruption handling, metadata-only status), and schema backward compatibility.

### How to Run

```bash
make test              # All tests
make test-rag          # RAG component tests
make test-api          # API endpoint tests
make test-dagster      # Dagster job tests
make test-raster       # Raster pipeline tests
make test-coverage     # With coverage report
```

---

## 3. What Got Better (Before → After)

| Area | Before | After |
|------|--------|-------|
| **Qdrant upserts** | One point at a time → timeouts on 500+ chunks | Batched (500/call) with retry — ~10× fewer HTTP calls |
| **Chunking granularity** | 1 timestep × 100 grid cells → millions of tiny, meaningless chunks | 30 timesteps × 10 grid cells → coherent monthly/regional chunks |
| **Retrieval accuracy** | Pure vector similarity, no spatial awareness | Spatial intent extraction + Qdrant range filters on lat/lon |
| **RAG faithfulness** | No grounding instructions — LLM could hallucinate freely | All prompt types include grounding/citation instructions |
| **Evaluation** | No systematic quality measurement | 3-tier test suite (IR metrics, RAGAS, embedding intrinsics) with golden query set |
| **Reranking** | Not available | Cross-encoder reranking with measurable NDCG/MRR improvements |
| **Pipeline resilience** | Crash = start over | Crash recovery, retry logic, memory guard, per-entry status tracking |
| **Source management** | Flat `sources.py` file | Structured `src/database/` (models, migrations, store) + `src/sources/` (connection tester) |
| **Auth / credentials** | No portal auth support | Per-source auth with portal presets, masked credential storage |
| **SSL / network** | Hard failure on SSL errors or slow connections | SSL fallback (retry without verify), timeout tuning (30 s / 600 s) |
| **ZIP handling** | Python-only extraction — failed on some compression methods | Fallback to system `unzip`/`7z` for unsupported methods |
| **Frontend** | Basic functional UI | Polished dark-theme admin with reusable components (StatCard, DonutChart, SparkLine, ProgressRing, CronPicker) |
| **Dagster** | Manual job triggering only | Schedule management, source jobs, resource configuration via API |
