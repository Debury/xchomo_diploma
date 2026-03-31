# RAG 90%+ Design Spec

**Date:** 2026-03-31
**Goal:** Push V2 composite from 84.5% to 90%+ by fixing retrieval precision and data quality.

## Current State

- 1.5M points in Qdrant, Phase 0+1 processed (71/74 datasets)
- LLM: Claude Sonnet 4.6 via OpenRouter
- V2 composite: 84.5% avg, weakest: T9=75%, T7=79%, T4=82%
- Answer Correctness: 79% avg, Retrieval Precision@5: 64% avg

## Changes

### 1. Retrieval Precision Fix (`web_api/rag_endpoint.py`)

**Problem:** Reranker returns chunks with scores as low as 0.001. T4 gets 4 wind chunks in top-5 for a drought query.

**Changes:**
- After cross-encoder reranking, drop chunks with reranker score < 0.05
- `_enforce_diversity`: reduce `max_per_source` from 4 to 3, `max_per_source_var` from 2 to 1
- This prevents same-variable flooding (e.g. 4x MERRA2 wind for drought query)

**Expected impact:** T4 precision 20%->60%+, T9 20%->40%+

### 2. Fix CO2 Data (new source `noaa_co2`)

**Problem:** `cams_co2` source has `tcco` (carbon monoxide), not CO2 concentration in ppm. T7 retrieves wrong data.

**Solution:** Download NOAA Global Monthly Mean CO2 (free CSV from NOAA GML), process and embed as `noaa_co2` source with actual ppm values.

- URL: `https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_mlo.csv`
- Contains: year, month, decimal_date, monthly_average_ppm, trend_ppm
- Process: read CSV, chunk by year ranges, embed with proper metadata (variable=co2_concentration, unit=ppm, hazard=Atmospheric CO2)

**Expected impact:** T7 79%->88%+

### 3. Enrich Dataset Summary Chunks (`src/climate_embeddings/schema.py`)

**Problem:** Summary chunks (is_dataset_summary=True) have same format as data chunks. Missing interpretive context about what the dataset measures and why it matters.

**Solution:** In `generate_human_readable_text`, when metadata has `is_dataset_summary=True`, append richer text:
- What the dataset measures (from long_name, hazard_type)
- Temporal and spatial scope (from temporal_coverage_text, spatial_coverage)
- Relevance context (from impact_sector)
- Aggregate statistics across the whole dataset

This helps the LLM understand dataset purpose without inventing information — all from existing metadata.

**Expected impact:** +2-3% on Answer Correctness across all tests

## Files to Modify

1. `web_api/rag_endpoint.py` — reranker threshold + diversity params
2. `src/climate_embeddings/schema.py` — enrich summary chunk text
3. New script or inline: download + embed NOAA CO2 data

## Success Criteria

- V2 composite avg >= 90%
- All 10 tests PASS
- T7 >= 85%, T9 >= 80%
- No hallucination — all answers grounded in real data
