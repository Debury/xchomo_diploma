# Demo scenár — obhajoba diplomky

> Live demo skript pre prezentáciu Climate RAG Pipeline. Drž sa poradia, ak sa niečo pokazí, skoč na *Záchranný plán* na konci.

---

## 0. Pred začiatkom (5 min pred)

**V termináli v `xchomo_diploma/` adresári:**

```bash
# 1. Spustiť celý stack (ak ešte nebeží)
docker compose up -d

# 2. Počkať kým sú všetky služby healthy (cca 60 s)
docker compose ps          # všetkých 6 musí byť "Up"

# 3. Sanity check
curl -s http://localhost:8000/health
# → {"status":"healthy","dagster_available":true,...}
```

**V browseri otvoriť tieto taby (in this order):**

1. `http://localhost:8000/` → redirect na `/app/` (login page)
2. `http://localhost:3000/` → Dagster UI (v samostatnom tabe pre prípad ak chce komisia vidieť orchestráciu)
3. `http://localhost:6333/dashboard` → Qdrant UI (ukáže vektory v collection)

**Login credentials:** `admin` / `climate2024` (z `.env`)

**Ak si predtým nebol prihlásený alebo máš starú cache, urob hard refresh (Ctrl+Shift+R) na /app/login.**

---

## 1. Dashboard (úvod, 1 min)

- Ukáž hlavnú stránku `/app/`
- Highlight čísla:
  - **1 529 425 vektorov** v Qdrant
  - **72 datasetov** rôznych formátov
  - **LLM provider**: OpenRouter (Claude Sonnet 4.6)
  - **Qdrant**: healthy zelená bodka
- "Live Activity" panel ukazuje health eventy + ETL progress

**Talking points:**
- Všetko beží v Docker Compose, 6 mikroslužieb prepojených cez `climate-net`
- Embedding model: BAAI/bge-large-en-v1.5 (1024-dim, COSINE distance)
- Jednotlivé datasety sú ingestované cez Dagster orchestrator

---

## 2. Chat / RAG dotaz (3 min, ťažisko demo)

Ísť na `/app/chat`. Zadať postupne tieto dotazy (jeden po druhom):

### Dotaz 1 (rýchly, vektorový hit)
```
What precipitation data is available for Slovakia?
```
- Očakávaná odpoveď: zmienka SLOCLIM, prípadne GPCC/SPREAD
- Loading indicator ukáže: "Searching Qdrant…" → "Reranking…" → "Asking the LLM…" + counter
- Trvanie: ~15–25 s (vysvetli že LLM cloud roundtrip)

**Pri čakaní rozprávaj:**
- "Systém teraz robí hybridný retrieval: vektorové vyhľadávanie nájde top-k relevantných chunkov, reranker ich usporiada, LLM syntetizuje odpoveď."
- "Každý chunk má payload s metadátami — dataset, variable, region, hazard — čo umožňuje filtrovanie."
- Po odpovedi rozklikni "N retrieved chunks" → ukáže similarity score + text chunk + dataset origin

### Dotaz 2 (temporal/geographic)
```
Extreme heat events in Europe between 2000 and 2020
```
- Očakávaná odpoveď: E-OBS, SLOCLIM, Iberia01
- Highlight: systém rozumie časovému rozsahu aj geo filtru

### Dotaz 3 (flood/drought multi-hazard, ukáže diverzitu)
```
Drought indicators for the Iberian Peninsula
```
- Očakávaná odpoveď: SPREAD, Iberia01, SEDI

**Talking points:**
- RAG pattern = Retrieval Augmented Generation
- Bez RAG by LLM halucinoval bez dát
- S RAG má groundovanú odpoveď + citácie na zdroje

---

## 3. Data Sources page (2 min)

Ísť na `/app/sources`.

- Ukáž **72 zdrojov** s filtrom na "Hazard" a "Region"
- Klikni na **SLOCLIM** (najväčší dataset, 254 571 chunks)
- V detail modale ukáž:
  - Stats (Qdrant chunk count)
  - Variables: `pcp` (precipitation)
  - Region: Slovenia
  - Metadata fields
- V detaile klikni "Edit" → ukáž že môžeš editovať keywords/hazard/schedule cez Qdrant payload update (bez re-embeddingu!)

**Talking points:**
- Qdrant je "source of truth" pre čo je indexované
- PostgreSQL drží management metadata (schedules, auth, processing history)
- Metadata update ide paralelne do oboch stores

---

## 4. Add New Source (3 min, ukáže flexibilitu)

Ísť na `/app/sources/create`.

### Step 1: URL
Zadať:
```
Source Name: Demo CO2 trend
Data URL: https://raw.githubusercontent.com/datasets/co2-ppm/master/data/co2-mm-mlo.csv
```
Klikni **"Auto-detect"**.
- Ukáže: Format=csv, reachable=true, suggested name

### Step 2: Auth (skip — CSV je public)
Nechaj "None (Open Access)"

### Step 3: Metadata
Klikni **"Scan File"** — toto sa nemusí podariť pre CSV (je to špeciálne pre NetCDF). Ak nie, vyplň manuálne:
- Hazard: Atmospheric composition
- Keywords: CO2, Mauna Loa, atmospheric
- Description: Monthly mean CO2 concentration from Mauna Loa observatory

### Step 4: Schedule
Zaškrtni "Enable per-source schedule":
```
0 3 * * 0    (Weekly Sunday 3am)
```

### Step 5: Review → Create
Klikni **"Create Source"**. Redirect na sources list s novým zdrojom.

**Talking points:**
- Auto-detect používa URL parsing + HTTP HEAD
- Scan File parsuje NetCDF headery remotely (xarray) bez downloadu celého súboru
- Schedule zaregistruje per-source cron — Dagster sensor ho zachytí (`source_schedule_sensor` poll every 60s)

---

## 5. Schedules page (1 min)

Ísť na `/app/schedules`.

- Ukáž novovytvorený schedule pre "Demo CO2 trend"
- "Create Schedule" button pre per-source cron
- Cron reference panel (daily/weekly/monthly examples)

**Talking points:**
- Per-source granularita (dohodnuté so zadávateľom)
- Dagster `source_schedule_sensor` beží každých 60s, kontroluje due schedules, triggruje `single_source_etl_job`

---

## 6. ETL Monitor (1 min)

Ísť na `/app/etl`.

- Ukáž živý log z catalog pipeline
- Progress bar per-phase (ak beží batch)

**Talking points:**
- Catalog modul spracúva D1.1.xlsx (233 zdrojov, 69 datasetov) v 5 fázach
- Phase 0 = metadata-only embedding, Phase 1 = direct download, Phase 3 = API portals (CDS, NASA)

---

## 7. Dagster UI (2 min — volitelné ak zostane čas)

Prejsť na tab `http://localhost:3000/`.

- Tab **"Jobs"**: zoznam jobov (`single_source_etl_job`, `batch_catalog_etl_job`, ...)
- Tab **"Sensors"**: `source_schedule_sensor`, `data_freshness_sensor` (oba RUNNING, check every 60s)
- Tab **"Schedules"**: `weekly_catalog_refresh` (Sunday 3am)

**Talking points:**
- Dagster poskytuje visibility do asset-level dependencies + run history
- PostgreSQL uchováva full event log + schedules state

---

## 8. Settings (1 min)

Ísť na `/app/settings`.

- LLM configuration: OpenRouter key configured, model selection
- Portal adapters: CDS, NASA Earthdata, CMEMS — každý s vlastným credential slotom
- Embedding model info: bge-large-en-v1.5, 1024-dim, COSINE
- System resources (disk usage)

---

## 9. Qdrant dashboard (1 min — volitelné)

Prejsť na tab `http://localhost:6333/dashboard`.

- Collections → `climate_data`
- Ukáž 1.53M points, 1024-dim, indexed
- Search cez native UI pre "temperature anomalies" — vráti top-k chunks

---

## Po demo — Q&A cheatsheet

Ak sa spýtajú:

**"Ako je to zabezpečené?"**
- Frontend má login gate (Vue router guard)
- Token-based session cez `Authorization: Bearer`
- Pre produkciu by sa pridalo Keycloak/OAuth (mimo scope diplomky)

**"Čo ak URL zdroja nefunguje?"**
- `source.processing_status` sa nastaví na `failed` s error message
- Freshness sensor každý hour kontroluje stale sources

**"Aká je presnosť RAG odpovedí?"**
- Evaluation kapitola má definované metriky Recall@k, MRR, Faithfulness
- Test set 50–100 query-answer párov je v pláne (rozpracované)

**"Prečo LLM RAG a nie čistý search?"**
- Search vráti surové chunky, RAG syntetizuje čitateľnú odpoveď s citáciami
- Užívateľ sa opýta prirodzeným jazykom, nemusí vedieť Qdrant query syntax

**"Ako škáluje Qdrant na 100M chunks?"**
- Hybrid dense+sparse, HNSW index, horizontálne cez sharding
- Momentálne je 1.5M bez problémov, collection status = "green"

---

## Záchranný plán — ak sa niečo pokazí

### Docker stack spadol
```bash
docker compose down
docker compose up -d
# počkať 60s, sledovať docker compose ps
```

### Web-api nereaguje
```bash
docker compose restart web-api
curl -s http://localhost:8000/health
```

### Login nefunguje / UI bieli screen
1. Ctrl+Shift+R (hard refresh)
2. Ak stále: F12 → Application → Local Storage → Clear → retry
3. Fallback: ukáž API cez curl alebo Dagster UI

### RAG chat hangol
- LLM provider (OpenRouter) môže byť pomalý. Timeout je 30s.
- Fallback: ukáž "Search" bez LLM cez `/rag/search?query=...` v curl termináli

### Qdrant obsahuje poškodené dáta
```bash
# Restore zo snapshot-u (viď backups/qdrant/README.md)
./scripts/qdrant/restore.sh /qdrant/snapshots/climate_data/climate_data-886361186156628-2026-04-18-11-29-12.snapshot
```

### Kľúčové links pre Q&A
- Thesis text: `../thesis-template-typst-main/` (Typst)
- Stav projektu: `stav_projektu.md`
- Architektúra: `CLAUDE.md`
- Config: `config/pipeline_config.yaml`, `docker-compose.yml`, `docker/dagster.yaml`

---

## Časový rozvrh (celkovo ~15 min)

| Sekcia | Čas |
|---|---|
| 1. Dashboard úvod | 1 min |
| 2. RAG chat (3 dotazy) | 3 min |
| 3. Sources list + detail | 2 min |
| 4. Add new source wizard | 3 min |
| 5. Schedules | 1 min |
| 6. ETL Monitor | 1 min |
| 7. Dagster UI (volitelne) | 2 min |
| 8. Settings | 1 min |
| 9. Qdrant dashboard (volitelne) | 1 min |

Ak máš menej času, skrátiť na: Dashboard → Chat (1 dotaz) → Add source → Schedules.
