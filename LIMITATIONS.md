# Limitations

This document records the known limitations of the Climate Data RAG pipeline as
delivered for the diploma defense. They are intentional scope decisions, not
hidden defects.

## Orchestration

- **Two concurrent ETL runs max** — the `QueuedRunCoordinator` is configured
  for `max_concurrent_runs: 2` (`docker/dagster.yaml`). Further runs queue in
  Dagster and fire in submission order as earlier runs finish. Acceptable for
  single-operator use; not a multi-tenant setup.
- **Ops serialize within a run** — catalog jobs use
  `executor_def=in_process_executor`, so inside one run all ops execute
  sequentially in a single Python process. Memory available to a single run
  is therefore capped by one process's heap, and CPU-bound ops don't use
  more than one core.
- **Schedule granularity is ±30 s** — `source_schedule_sensor` polls the
  `source_schedules` table every 30 seconds. Cron slots sub-second accurate to
  the minute; no sub-minute scheduling.
- **Cross-container concurrency control is advisory** — both
  `/sources/{id}/trigger` and `source_schedule_sensor` take a Postgres
  session-level advisory lock (`pg_try_advisory_lock` keyed on a 64-bit
  blake2b hash of `source_id`) before launching. API returns 409 when held;
  sensor skips and retries in 30 s. The lock is released when the Dagster
  launch request has been persisted, *not* when the run finishes — so a
  long-running ETL does not block a completely unrelated future trigger.

## Authentication

- **Single admin account** — `AUTH_USERNAME` / `AUTH_PASSWORD[_HASH]` come
  from environment variables. There is no user table, no sign-up, no
  role/permission model, and no password-reset flow. Intentional scope
  decision for a single-operator thesis deployment.
- **In-process token store** — issued bearer tokens live in a Python dict
  (`web_api/routes/auth.py::_valid_tokens`). All sessions are invalidated on
  `web-api` restart, and tokens cannot be shared across replicas. Acceptable
  for a single-node demo; would need Redis or Postgres for horizontal scale.
- **Password hashing is optional but recommended** — plaintext `AUTH_PASSWORD`
  is still accepted for compatibility. Generate a PBKDF2-SHA256 hash with
  `python scripts/hash_password.py` and paste the result into
  `AUTH_PASSWORD_HASH`; when both are set the hash wins.
- **Login throttling is in-memory only** — `/auth/login` counts failures per
  source IP (default 10 failures per 15 min → 429). Counters reset on
  `web-api` restart and are not shared across replicas. A determined attacker
  behind a rotating IP pool bypasses this.

## Credentials and API keys

- **Plaintext at rest in two places** — `data/app_settings.json` (written by
  `POST /settings/credentials`, gitignored) and `.env` (read at process start,
  gitignored). Neither is encrypted; only filesystem permissions and the
  gitignore rule prevent leakage.
- **`.env` wins over `app_settings.json` for overlapping keys** — on both
  startup and per-op refresh, the helper only writes to `os.environ` for
  empty slots (`override=False` in Dagster ops). A key set in `.env` is
  treated as authoritative and the UI-persisted value is ignored. To rotate
  a key that already lives in `.env`, edit `.env` and
  `docker compose up -d --force-recreate dagit dagster-daemon web-api` —
  a UI-only change won't take effect.

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
