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
- **Plaintext `AUTH_PASSWORD` still accepted as fallback** — when
  `AUTH_PASSWORD_HASH` is unset, the server falls back to a plaintext compare
  against `AUTH_PASSWORD` and logs a warning at startup. Generate an
  argon2id hash with `python scripts/hash_password.py` and paste the result
  (wrapped in single quotes in `.env`) into `AUTH_PASSWORD_HASH` to retire the
  plaintext value. Hash verification uses `pwdlib` (argon2-cffi).
- **Stateless JWT means no server-side revocation** — tokens are signed with
  `JWT_SECRET_KEY` and verified by signature + `exp` claim only. `/auth/logout`
  is a client-side convenience; a stolen token stays valid until `exp`
  (default 24 h) unless you rotate `JWT_SECRET_KEY`, which invalidates *all*
  outstanding tokens. Acceptable for the single-operator thesis scope; a
  multi-user production deployment would want a revocation list or reduce
  the TTL.
- **Per-IP login throttling is best-effort** — failures are counted in
  Postgres (`login_failures` table) and shared across workers, but a
  determined attacker behind a rotating IP pool bypasses it. There is no
  global (per-username) rate cap.

## Credentials and API keys

- **Plaintext at rest** — `data/app_settings.json` (written by
  `POST /settings/credentials`, gitignored, `chmod 0600` enforced on every
  write) and `.env` (gitignored) both hold keys in cleartext. There is no
  encryption-at-rest layer; protection is filesystem permissions + gitignore.
- **Clearing a credential in the UI doesn't evict it from a running op** —
  `load_persisted_credentials_into_env` only writes non-empty values into
  `os.environ` (`override=True` — UI wins when present, but empty means
  "skip"). If you remove a key via the UI, the Dagster container may still
  hold the old value in `os.environ` until it restarts. Acceptable for
  rotations; awkward for outright revocation.

## Embedding model changes

- **Switching embedding model requires an explicit collection drop** — the
  Qdrant collection is pinned to a vector size at creation. If you change the
  embedder (e.g. `BAAI/bge-large-en-v1.5` → `sentence-transformers/all-MiniLM-L6-v2`)
  the next op that initialises `VectorDatabase` will refuse to start with a
  `RuntimeError: Vector size mismatch` rather than silently drop and recreate
  the collection. To migrate, export a Qdrant snapshot, `DELETE /collections/climate_data`
  yourself, re-ingest all sources with the new embedder, then optionally
  re-upload the old snapshot to a differently-named collection if you need
  to compare. This replaced a historic auto-drop footgun that could wipe
  the entire collection implicitly.

## Data coverage

- **Seven built-in portal adapters**: CDS, NASA Earthdata, CMEMS, ESGF, CEDA,
  EIDC, NOAA. These have dedicated Python classes in `src/catalog/portal_adapters.py`
  and handle portal-specific flows (CDS API's `cdsapi` client, NASA Earthdata
  URS bearer auth, etc.).
- **Any other portal with a plain HTTP+API-key/bearer/basic auth flow** can be
  added at runtime through `Settings → Add adapter` (POST `/settings/adapters`).
  Credential fields are defined in the UI, then filled through the existing
  `/settings/credentials` endpoint and flow straight into `os.environ` for
  Dagster ops. OAuth-flow portals and portals that require a custom SDK are
  the genuine gap.
- **Catalog automation**: 167 of 233 D1.1.xlsx entries ingest end-to-end. The
  remaining 66 are blocked by authentication, JavaScript-rendered landing
  pages, or non-raster formats and are registered as metadata-only points.

## Evaluation

- **Small golden-query set** — the evaluation in `docs/rag_eval_v2_clean_final.md`
  uses 10 hand-written queries. There is no inter-annotator agreement study.
- **No load testing** — the 1.53 M-chunk collection has not been stress-tested
  beyond single-operator query patterns. Latency numbers are measured from a
  warm Qdrant and a warm embedding cache.
