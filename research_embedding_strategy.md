# What We're Changing and Why (with Research Backing)

## What's Wrong Right Now (in plain words)

We have 4 problems:

### Problem 1: The search vector is useless

Right now, every chunk of climate data gets turned into 8 numbers:
`[mean, std, min, max, P10, median, P90, range]`

This is what Qdrant uses to find "similar" data when you ask a question.

**The issue**: Temperature of 15°C and rainfall of 15mm produce the **exact same 8 numbers**. Qdrant can't tell them apart. The vector — the thing that's supposed to help search — doesn't know what variable it's looking at. Search only works because we manually filter by variable name, not because of the vector.

**Example**: You ask "What was the winter temperature in Czechia?"
- Qdrant compares your question's vector to all stored vectors
- A temperature chunk with mean=5 and a precipitation chunk with mean=5 look equally good
- Only the metadata filter `variable=tas` saves us — the vector itself is doing nothing useful

### Problem 2: Each chunk covers a huge area

Spatial dimensions are chunked into blocks of 100 grid cells. For a typical dataset at 0.25° resolution, one chunk covers **100 x 0.25° = 25 degrees of latitude**. That's roughly from Rome to Stockholm in one chunk.

**Example**: You ask "temperature in Brno" — you get back the average temperature of an area from southern Italy to northern Sweden. That's not useful.

### Problem 3: Too many chunks (daily data)

Every single day gets its own chunk. A dataset with 30 years of daily data with 5 variables = **54,750 chunks**. SLOCLIM has been processing for 8+ hours and produced 100,000+ chunks — it's still going. One source is 40% of our entire database.

Most of those daily chunks are nearly identical (January 5 vs January 6 temperature — basically the same embedding). It's redundant data that slows everything down.

### Problem 4: Two incompatible vector spaces

Phase 0 metadata embeddings use BGE-M3 (1024 dimensions — text-based).
Raster data embeddings use stats vectors (8 dimensions — numbers).

These live in the same Qdrant collection but are completely different vector spaces. You can't meaningfully search across both. It's like having books and recipes in the same library but one is indexed by author and the other by calorie count.

---

## What Research Papers Say We Should Do

### Paper 1: Spatial-RAG

> Yu, D., Bao, R., Ning, R., Peng, J., Mai, G., & Zhao, L. (2025). "Spatial-RAG: Spatial Retrieval Augmented Generation for Real-World Geospatial Reasoning Questions." arXiv:2502.18470.

**What they did**: Built a system that answers questions like "find restaurants near Central Park" by combining two types of search:

1. **Spatial filtering** (sparse) — use the database to filter by actual location coordinates (distance, bounding box). This is the "hard" filter — if something is in the wrong place, throw it out.
2. **Text similarity** (dense) — use text embeddings to match the meaning of the question to the description of the data. This is the "soft" matching — find things that are semantically related.

They combine both scores with a formula: `final_score = weight_A * spatial_score + weight_B * semantic_score`

**Their key result**: When they removed the spatial filtering part, accuracy dropped the most — more than removing any other component. This means location filtering is critical and you can't rely on text similarity alone for geographic questions.

**What this means for us**: We already do the spatial filtering part (Qdrant payload filters on lat_min/lat_max). That part is correct. What we're missing is the text similarity part — our 8-dim stats vector is not a text embedding, so the "dense semantic matching" half of their approach is completely absent in our system.

**Test results from the paper:**
- TourismQA-NYC: Precision@1 = 56.65% (19.9% better than the next best method)
- MapQA-AME: Recall = 90.72% (75.4% better than baseline)

---

### Paper 2: Geo-RAG (RAG for Geoscience)

> Yu, R., Luo, S., Ghosh, R., Li, L., Xie, Y., & Jia, X. (2025). "RAG for Geoscience: What We Expect, Gaps and Opportunities." arXiv:2508.11246.

**What they did**: Wrote a position paper saying that current RAG systems are "text-centric" — they only know how to search through text documents. But geoscience data is not text. It's raster grids, time series, measurements, satellite images. The standard RAG approach of "embed text → search → generate answer" is not enough.

They propose **Geo-RAG** — a loop with 4 steps:
1. **Retrieve** — find relevant data (not just text, but actual Earth observation data)
2. **Reason** — apply physical constraints (e.g., temperature can't be 500°C, conservation laws must hold)
3. **Generate** — produce a science-grade answer (with units, uncertainties, references)
4. **Verify** — check the answer against numerical models or real measurements

**What this means for us**: Our system only does retrieve→generate (steps 1 and 3). We skip reasoning and verification. For the thesis, we can mention this as a limitation and future work. But the key takeaway is: even the researchers defining the future of geoscience RAG say that **text-based retrieval is the starting point** — you need text embeddings as the baseline, then add domain-specific features on top.

---

### Paper 3: GeoGPT-RAG

> Huang, F., Wu, F., Zhang, Z., Wang, Q., Zhang, L., Boquet, G.M., & Chen, H. (2025). "GeoGPT-RAG Technical Report." arXiv:2509.09686.

**What they built**: A complete RAG system for geoscience with custom-trained components:

1. **GeoEmbedding** — a custom embedding model (based on Mistral-7B, 7 billion parameters) trained specifically on geoscience text. They trained it on 360,000 samples from geoscience papers.

2. **GeoReranker** — a reranking model based on **BGE-M3** (the same embedding model we already use!). After the initial search returns top results, the reranker reads each result together with the question and re-scores them for relevance.

3. **Chunking** — they split long documents into pieces of max 512 tokens using semantic segmentation (they use a BERT model to find the best split points so sentences don't get cut in the middle).

4. **Search pipeline**: Query → embed with GeoEmbedding → find top 8 similar chunks in vector DB → rerank with GeoReranker → keep best ones → send to LLM with context

**Their results:**
- GeoEmbedding Recall@1: 90.8% on geoscience questions
- End-to-end answer accuracy: 85.7% (evaluated by domain experts)
- They use **Zilliz Cloud** (managed Milvus) as their vector database

**What this means for us**: The state-of-the-art geoscience RAG system uses **text embeddings**, not statistical vectors. They use BGE-M3 as a reranker backbone — that's literally the same model we have loaded. Nobody in academic research is putting raw statistics into a vector for search. The vector is always a text embedding.

---

### Paper 4: GeoGraphRAG

> Published in International Journal of Applied Earth Observation and Geoinformation (2025). "GeoGraphRAG: A graph-based retrieval-augmented generation approach for empowering large language models in automated geospatial modeling."

**What they did**: Instead of just searching through chunks, they built a **knowledge graph** — a network of connected concepts (e.g., "ERA5" → provides → "temperature" → measured_in → "Czechia"). The LLM acts as an agent that navigates this graph to find relevant information.

**What this means for us**: This is the most advanced approach but also the most complex. For a diploma thesis, this is "future work" material. Worth mentioning as a direction but not implementing.

---

### Paper 5: AlphaEarth Foundations (Google DeepMind)

> Brown, C.F., Kazmierski, M.R., et al. (2025). "AlphaEarth Foundations: An embedding field model for accurate and efficient global mapping from sparse label data." arXiv:2507.22291.

**What they did**: Google built a model that takes satellite images, radar data, climate measurements, and text descriptions — and produces a single embedding vector for any point on Earth at any time. These embeddings capture what a location "looks like" across all data sources.

**What this means for us**: Even Google, with unlimited resources, produces embeddings from **multi-modal data** (combining different data types), not from raw statistics. They learn the embedding from the data rather than hand-crafting 8 statistical features. This confirms that our stats-vector approach is not aligned with how the field is moving.

---

## What We're Actually Changing

### Change 1: Monthly instead of daily (fixes Problem 3)

**Before**: One chunk per day → 365 chunks/year/variable
**After**: One chunk per month → 12 chunks/year/variable

**Why**: 30x fewer chunks. SLOCLIM goes from 100k+ chunks (8 hours) to ~3,000 chunks (~15 min). The daily detail is redundant for RAG — nobody asks "what was the temperature on January 5 vs January 6". They ask "what was the winter temperature" or "what was the January average".

**We don't lose data**: The monthly chunk still stores min, max, percentiles in the metadata payload. If someone asks about extremes, the data is there.

### Change 2: Spatial chunk 100 → 10 (fixes Problem 2)

**Before**: Each chunk covers 100 grid cells = ~25° of latitude (Rome to Stockholm)
**After**: Each chunk covers 10 grid cells = ~2.5° of latitude (~250km, roughly one country)

**Why**: When someone asks "temperature in Czechia", a 2.5° chunk actually covers approximately Czechia. A 25° chunk covers half of Europe — the statistics are meaningless for a specific location.

**Chunk count impact**: Minimal increase because we're also doing monthly aggregation. The time reduction (30x fewer) more than compensates for the spatial increase (~10x more spatial chunks). Net result is still fewer total chunks.

### Change 3: Text embeddings instead of stats vector (fixes Problems 1 and 4)

**Before**: Chunk → compute 8 statistics → store as 8-dim vector
**After**: Chunk → generate text description → embed with BGE-M3 → store as 1024-dim vector

The text description looks like:
```
"Monthly mean temperature (tas) from SLOCLIM, southern Moravia
(lat 48.5-49.5, lon 16.0-17.0), January 2020:
mean 2.3°C, std 1.8, range [-5.1, 8.7°C],
P10=-2.1, median=2.5, P90=6.1. Units: degC."
```

This gets embedded into 1024 dimensions by BGE-M3 (same model we already have loaded for Phase 0 metadata).

**Why this is better**:
- "cold winter in Czechia" now **semantically matches** chunks about low temperature in Czech locations
- Temperature and precipitation chunks are **different** embeddings because the text says different things
- Phase 0 metadata embeddings and Phase 1 raster embeddings are now in the **same 1024-dim vector space** — unified collection, one search finds both
- All the numerical values are still stored in the Qdrant payload — nothing is lost, the numbers are still filterable and readable

**Speed impact**: BGE-M3 embedding takes ~10ms per chunk. For 3,000 chunks (monthly SLOCLIM) that's 30 seconds total. Negligible compared to the download and upsert time.

### What stays the same

- **Qdrant** stays as our vector database — it supports everything we need
- **Spatial filtering** stays (Qdrant payload Range filters on lat/lon) — confirmed as important by Spatial-RAG paper
- **All numerical stats** stay in the Qdrant payload — mean, std, min, max, percentiles are all still there
- **Phase 0 metadata pipeline** stays unchanged — already uses BGE-M3 text embeddings

---

## Before vs After (complete picture)

```
BEFORE (current):
                                                  Qdrant Collection
NetCDF file                                       ┌─────────────────────┐
  │                                               │ Point:              │
  ├─ var: tas                                     │   vector: [8-dim]   │ ← useless for search
  ├─ time: 2020-01-01                             │     [15, 5, 2, 28,  │    (can't tell temp
  ├─ lat: 100 cells (25°)                         │      8, 15, 22, 26] │     from rain)
  ├─ lon: 100 cells (25°)                         │   payload:          │
  │                                               │     variable: tas   │
  └─ compute stats ──────────────────────────────→│     lat_min: 35.0   │ ← covers half Europe
                                                  │     lat_max: 60.0   │
                                                  │     time: 2020-01-01│ ← single day
                                                  └─────────────────────┘
                                                  x 100,000+ chunks per source (!!!)


AFTER (proposed):
                                                  Qdrant Collection
NetCDF file                                       ┌──────────────────────────┐
  │                                               │ Point:                   │
  ├─ var: tas                                     │   vector: [1024-dim]     │ ← BGE-M3 text
  ├─ time: January 2020 (aggregated)              │     (semantic embedding  │    embedding,
  ├─ lat: 10 cells (2.5°)                         │      of text below)      │    understands
  ├─ lon: 10 cells (2.5°)                         │   payload:               │    meaning
  │                                               │     variable: tas        │
  ├─ compute stats ──────────┐                    │     lat_min: 48.5        │ ← covers ~Czechia
  │                          ├───────────────────→│     lat_max: 51.0        │
  └─ generate text ──────────┘                    │     mean: 2.3            │ ← stats still here
    "Monthly mean temperature                     │     std: 1.8             │    for filtering
     (tas) in Czechia,                            │     min: -5.1            │
     January 2020:                                │     max: 8.7             │
     mean 2.3°C ..."                              │     time_start: 2020-01  │ ← monthly
                                                  │     time_end: 2020-01    │
                                                  │     text: "Monthly..."   │
                                                  └──────────────────────────┘
                                                  x ~3,000 chunks per source (manageable)
```

---

## Summary for Professor

| What | Before | After | Why | Backed by |
|------|--------|-------|-----|-----------|
| Time granularity | Daily (1 chunk/day) | Monthly (1 chunk/month) | 30x fewer chunks, no useful info lost for RAG queries | Common practice, reduces noise |
| Spatial granularity | 100 grid cells/chunk (~25°) | 10 grid cells/chunk (~2.5°) | Answers about specific regions become meaningful | Spatial-RAG confirms location precision matters |
| Search vector | 8-dim hand-crafted statistics | 1024-dim BGE-M3 text embedding | Stats can't distinguish variables; text embeddings understand meaning | GeoGPT-RAG, Geo-RAG — all use text embeddings |
| Numerical data | Only in vector (lost after embedding) | In payload (preserved, filterable) | Can still filter by "mean > 20" or "variable = tas" | Standard Qdrant payload filtering |
| Vector space | Two incompatible (8-dim + 1024-dim) | One unified 1024-dim | Phase 0 metadata and Phase 1 raster in same searchable space | GeoGPT-RAG uses single embedding space |
| Processing time | 8+ hours for large sources | ~15-20 min | Feasible to process all 233 catalog sources | - |
| SLOCLIM chunks | 100,000+ | ~3,000 | Collection stays balanced, no single source dominates | - |

---

## References

1. Yu, D., Bao, R., Ning, R., Peng, J., Mai, G., & Zhao, L. (2025). Spatial-RAG: Spatial Retrieval Augmented Generation for Real-World Geospatial Reasoning Questions. *arXiv:2502.18470*. https://arxiv.org/abs/2502.18470

2. Yu, R., Luo, S., Ghosh, R., Li, L., Xie, Y., & Jia, X. (2025). RAG for Geoscience: What We Expect, Gaps and Opportunities. *arXiv:2508.11246*. https://arxiv.org/abs/2508.11246

3. Huang, F., Wu, F., Zhang, Z., Wang, Q., Zhang, L., Boquet, G.M., & Chen, H. (2025). GeoGPT-RAG Technical Report. *arXiv:2509.09686*. https://arxiv.org/abs/2509.09686

4. GeoGraphRAG: A graph-based retrieval-augmented generation approach for empowering large language models in automated geospatial modeling. (2025). *International Journal of Applied Earth Observation and Geoinformation*. https://www.sciencedirect.com/science/article/pii/S1569843225003590

5. Brown, C.F., Kazmierski, M.R., et al. (2025). AlphaEarth Foundations: An embedding field model for accurate and efficient global mapping from sparse label data. *arXiv:2507.22291*. https://arxiv.org/abs/2507.22291

6. Qdrant Documentation: Hybrid Queries. https://qdrant.tech/documentation/concepts/hybrid-queries/
