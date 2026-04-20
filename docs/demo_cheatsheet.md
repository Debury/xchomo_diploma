# Demo cheat-sheet — obhajoba 2026-04-20 13:00

Lokálny stack: `http://localhost/` (Caddy) alebo `http://localhost:8000/app/`

## Login
- Username: `admin`
- Password: `climate2024`

---

## Demo flow (10-12 min)

### 1. Login + dashboard (30 s)
JWT + argon2id, per-IP rate limit, fail-closed.

### 2. Sources list (1 min)
72 datasetov, ~1.53 M chunks v Qdrant. Portály: CDS, ESGF, NASA, CMEMS, CEDA, EIDC, NOAA.

### 3. Create Source wizard — Test connection + Auto-detect (2 min)

**a) SSRF guard demo** — vlož túto URL a klikni **Auto-detect**:
```
http://169.254.169.254/latest/meta-data/
```
Očakávaný výsledok: červený toast `URL check failed: URL rejected: ...`. AWS metadata endpoint je blokovaný SSRF guardom (`src/sources/connection_tester.py::validate_public_url`).

**b) CSV scan** — HadCRUT5 monthly global temperature anomaly (Met Office, 89 KB):
```
https://www.metoffice.gov.uk/hadobs/hadcrut5/data/HadCRUT.5.0.2.0/analysis/diagnostics/HadCRUT.5.0.2.0.analysis.summary_series.global.monthly.csv
```
Klikni Auto-detect → format `csv`, portal blank, suggested name. Potom v Step 3 (Review) klikni **Scan metadata** → ukáže 4 columns (Time, Anomaly, Lower, Upper) + period 1850 → 2025.

**c) NetCDF scan** — E-OBS minimum temperature (KNMI, 152 MB, počítaj 30-60 s):
```
https://knmi-ecad-assets-prd.s3.amazonaws.com/ensembles/data/Grid_0.25deg_reg_ensemble/tn_ens_mean_0.25deg_reg_2011-2025_v32.0e.nc
```
Auto-detect → format `netcdf`, suggested name `tn_ens_mean...`. Scan metadata → variables, lat/lon range (Európa), time range.

**d) Upload demo (voliteľné)** — prepni na "Upload file", drag&drop akýkoľvek lokálny súbor. Auto-detect formátu z extension.

> **NEPOTVRDZUJ** Confirm/Create source ak nechceš spustiť ETL na 152 MB súbor.

### 4. ETL Monitor (1 min)
Otvor ETL Monitor tab. Ak nič nebeží: "No ETL batch running" banner.
Spomeň: Dagster, max 2 paralelné runs, per-source advisory lock (Postgres).

### 5. RAG Chat (4 min) ⭐
Pripravené dotazy (testované, ~20 s response):
- *"What is the projected temperature increase in Europe by 2050?"*
- *"What datasets do you have about precipitation?"*
- *"Compare temperature anomalies between Northern and Southern Hemisphere"*

Ukáž:
- Markdown rendering odpovede
- Chunks panel (vpravo) → cituje konkrétne datasety
- K↔°C konverzia
- Reranker default OFF (rýchlosť, eval ukázal žiadnu stratu kvality)

### 6. Settings → Custom adapters (1 min)
Settings → "+ Add adapter" → fake adapter `MyPortal` → ukáž že sa objaví v `/settings/credentials` immediately bez restartu.

### 7. Schedules (30 s)
Per-source scheduling, Dagster sensor poll 30 s, advisory lock proti race s manual triggermi.

---

## Záložné URL (ak primárne na demu zlyhajú)

**Záložný CSV** — NOAA Climate at a Glance, 2 KB, ročné anomálie 1850-2024:
```
https://www.ncei.noaa.gov/access/monitoring/climate-at-a-glance/global/time-series/globe/land_ocean/12/12/1850-2024.csv
```

**Záložná SSRF URL** (ak `169.254.169.254` čudne odpovedá):
```
http://127.0.0.1:8000/health
```
alebo
```
http://10.0.0.1
```

---

## Highlight pre recenzenta

**`LIMITATIONS.md`** v rooti — transparentnosť o známych obmedzeniach (sensor granularity, lock collisions, embedding model rebuild). Ukáž že vieš o trade-offoch.

**`docs/rag_eval_v2_clean_final.md`** — kanonický eval: composite 89 %, faithfulness 99 %, context relevance 98 %, 10/10 pass.

---

## Troubleshooting počas dema

| Problém | Rýchla akcia |
|---|---|
| Chat hangne >40 s | `docker compose restart web-api` (15 s) |
| Frontend nereaguje | hard reload Ctrl+Shift+R |
| Scan metadata timeout na E-OBS | ukáž že CSV scan funguje a povedz "veľký NetCDF beží na pozadí" |
| Login 401 | over `.env` má `AUTH_PASSWORD_HASH=...` |

`docker ps` musí ukázať 6 containerov: caddy, web-api, dagit, dagster-daemon, qdrant, dagster-postgres.

---

## Po obhajobe TODO (z `memory/project_climate_rag_handin_todo.md`)
1. Rotovať `OPENROUTER_API_KEY`
2. Doplniť AI transparency declaration v `thesis-template-typst-main/content/transparency_ai_tools.typ`
