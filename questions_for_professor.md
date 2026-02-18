# Questions for Diploma Thesis Supervisor

## Current State

Our raster pipeline chunks NetCDF/GRIB files as follows:
- **Time dimension**: `chunk_size = 1` (one embedding per timestep)
- **Spatial dimensions (lat/lon)**: `chunk_size = 100` (100 grid cells per chunk)
- **Embedding**: 8-dimensional stats vector per chunk (mean, std, min, max, P10, median, P90, range)
- **Metadata**: each chunk stores `lat_min`, `lat_max`, `lon_min`, `lon_max`, `time_start`, `time_end`
- **Spatial filtering**: Qdrant payload Range filters on `lat_min`/`lat_max` before vector search (Spatial-RAG approach from arXiv:2502.18470)

**Problem case**: SLOCLIM (Slovenian climate dataset) has produced **100,000+ chunks** from a single source and is still running after 8+ hours. The entire Qdrant collection has ~277k embeddings — one source is becoming ~40% of all data.

---

## 1. Temporal Granularity

**Current**: One embedding per timestep (daily data = ~365 chunks/year/variable).

30 years of daily data with 5 variables = **~54,750 chunks per source**.

| Strategy | Chunks for 30yr daily, 5 vars | Processing Time | RAG Precision |
|----------|-------------------------------|-----------------|---------------|
| Daily (current) | ~54,750 | ~8 hours | Exact day match |
| Monthly aggregation | ~1,800 | ~15 min | Monthly averages |
| Yearly aggregation | ~150 | ~1 min | Annual trends only |
| Seasonal (DJF/MAM/JJA/SON) | ~600 | ~5 min | Season-level |

**Questions:**
- Is daily granularity necessary for the thesis use case? What kind of temporal queries do we expect users to ask?
- Would **monthly aggregation** be a good default, with daily as an opt-in for specific sources?
- Should different sources have different temporal strategies? (e.g., daily for station data, monthly for gridded reanalysis)
- For the thesis evaluation, is it more impressive to have precise daily retrieval or broad coverage of more sources?

---

## 2. Spatial Chunking Strategy

**Current**: `lat chunk_size = 100`, `lon chunk_size = 100`. Each chunk covers a **100x100 grid cell block**.

For a 0.25-degree resolution dataset (e.g., ERA5):
- 100 lat cells = **25 degrees of latitude** per chunk
- A single chunk might cover all of Central Europe

For a 0.1-degree resolution dataset:
- 100 lat cells = **10 degrees of latitude** per chunk
- Still a very large area

**The location problem**: When a user asks *"What is the temperature in Brno?"*, we retrieve a chunk whose `lat_min`/`lat_max` spans e.g., 45.0–70.0. The stats (mean, min, max) are computed across that entire 100x100 block. The answer reflects the average temperature of **half of Europe**, not Brno.

| Spatial Strategy | Chunk covers | Chunks per timestep (global 0.25°) | Location accuracy |
|------------------|-------------|-------------------------------------|-------------------|
| `chunk=100` (current) | 25° x 25° block | ~10 | Continental average |
| `chunk=10` | 2.5° x 2.5° block | ~1,000 | Regional (~250km) |
| `chunk=1` | Single grid cell (0.25°) | ~100,000 | ~25 km resolution |
| `chunk=5` | 1.25° x 1.25° | ~4,000 | ~140 km resolution |

**Questions:**
- Is country/regional-level spatial accuracy sufficient, or do we need city-level?
- With `chunk=1` (per grid cell), we get precise locations but chunk count explodes. Is this acceptable for a subset of sources?
- Should we use **adaptive spatial chunking** — coarse for global datasets, fine for regional ones?
- The Spatial-RAG filter already narrows by bounding box. Does that compensate for coarse spatial chunks? (i.e., if we filter to Czech Republic bbox first, the 100x100 chunk within that area is actually a good answer)

---

## 3. Embedding Strategy

**Current**: Pure statistical vector (8 dimensions: mean, std, min, max, P10, P50, P90, range).

This means:
- Two chunks with similar statistical distributions match even if they're from different variables/regions
- A temperature chunk (mean=15°C, std=5) and a precipitation chunk (mean=15mm, std=5mm) have identical embeddings
- Search relies heavily on metadata filters (variable, source_id) to be useful

**Alternative**: Text-based embeddings using BAAI/bge-m3 (already loaded for metadata embeddings). Generate a natural language description per chunk:

> "Temperature (tas) in Central Europe (lat 48-50, lon 14-18), January 2020: mean 2.3°C, range -5.1 to 8.7°C"

Then embed this text with BGE-M3 (1024-dim). This would:
- Allow semantic search ("cold winter in Czechia" matches temperature chunks naturally)
- Distinguish variables by meaning, not just filter
- Be compatible with the existing Qdrant collection (already 1024-dim for metadata embeddings)

**Questions:**
- Is the statistical 8-dim vector approach defensible in the thesis, or should we switch to text embeddings for raster data too?
- If we switch to text embeddings, processing per chunk becomes ~10x slower (embedding model inference). Is the tradeoff worth it?
- Could we use a **hybrid approach** — text embeddings for metadata + statistical vectors as additional payload for numeric queries?

---

## 4. Collection Balance & Source Dominance

**Current state**: 277k embeddings total. SLOCLIM alone will be ~100k+ (36%+).

This means:
- RAG queries about any topic will likely return SLOCLIM data in top results
- Smaller but important sources (CMIP6 projections, ERA5 reanalysis) get drowned out
- Search results are biased toward whatever source has the most chunks

**Possible mitigations:**
1. **Cap chunks per source** (e.g., max 10,000 per source) — aggregate if over limit
2. **Source-balanced retrieval** — retrieve top-K per source, then merge
3. **Temporal aggregation** for large sources (monthly instead of daily)

**Questions:**
- Is source balance important for the thesis evaluation?
- Should we enforce a **maximum chunk budget per source** to keep the collection balanced?
- Or is it fine to let some sources dominate if they have more data?

---

## 5. Processing Time vs. Coverage Tradeoff

With the current strategy:
- Phase 0 (metadata only): **all 233 sources** in ~10 minutes
- Phase 1 (data download + embed): some sources take **8+ hours** each
- At this rate, processing all Phase 1 sources could take **days to weeks**

With monthly aggregation:
- Most sources would process in **5-30 minutes**
- Full Phase 1 could finish in **hours, not weeks**

**Questions:**
- For the thesis deadline, is it better to have **few sources deeply embedded** (daily) or **many sources with monthly aggregation**?
- Should we prioritize breadth (more sources) or depth (more temporal detail)?
- Is a configurable strategy (per-source temporal resolution) over-engineering for the thesis scope?

---

## Summary of Key Decisions Needed

| Decision | Option A | Option B | Recommendation |
|----------|----------|----------|----------------|
| Temporal | Daily (current) | Monthly aggregation | Monthly default, daily opt-in |
| Spatial | 100 cells/chunk (current) | 5-10 cells/chunk | 10 cells for regional accuracy |
| Embedding | 8-dim stats (current) | Text-based 1024-dim | Text-based (matches existing collection) |
| Source limit | No limit (current) | Max 10k chunks/source | Cap at 10k with aggregation |
| Priority | Depth (few sources, daily) | Breadth (many sources, monthly) | Breadth for thesis |
