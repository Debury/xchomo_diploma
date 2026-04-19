# Limitations

This document records the known limitations of the Climate Data RAG pipeline as
delivered for the diploma defense. They are intentional scope decisions, not
hidden defects.

## Orchestration

- **Single-process Dagster** — `dagit` and `dagster-daemon` run with one worker
  each, so at most one ETL job executes concurrently. Triggering a second run
  while one is active queues it; triggering the same source twice in quick
  succession can race on the same Qdrant `source_id` (last write wins). The
  system is designed for single-operator use, not multi-tenant load.
- **Schedule firing is manual** — per-source `SourceSchedule` rows persist
  `next_run_at`, but no Dagster sensor polls them; the "Trigger" button and the
  catalog batch runner are the production paths.

## Security / credentials

- **Plaintext credentials at rest** — API keys and portal logins live in
  `data/app_settings.json` in plaintext (gitignored). No KMS or OS keyring
  integration. Acceptable for a single-operator deployment; unsuitable for
  shared or untrusted hosts.
- **`/scan-metadata` is synchronous** — the endpoint blocks the worker for the
  duration of a remote NetCDF header read. Under concurrent callers this
  degrades API responsiveness.

## Data coverage

- **Supported portals**: CDS, NASA Earthdata, CMEMS, ESGF, CEDA, EIDC, NOAA.
  Other portals fall back to metadata-only entries.
- **Catalog automation**: 167 of 233 D1.1.xlsx entries are automatically
  ingested end-to-end. The remaining 66 are blocked by authentication,
  JavaScript-rendered landing pages, or non-raster formats and are registered
  as metadata-only points.

## Evaluation

- **Small golden-query set** — the evaluation in `docs/rag_eval_v2_clean_final.md`
  uses 10 hand-written queries. There is no inter-annotator agreement study.
- **No load testing** — the 1.53 M-chunk collection has not been stress-tested
  beyond single-operator query patterns. Latency numbers are measured from a
  warm Qdrant and a warm embedding cache.
