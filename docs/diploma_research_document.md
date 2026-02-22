# Diplomová práca — Výskumný a implementačný referenčný dokument

> **Autor**: Bc. [meno]
> **Téma**: Systém pre spracovanie klimatických dát s využitím RAG (Retrieval-Augmented Generation)
> **Dátum poslednej aktualizácie**: 2026-02-22
> **Počet commitov v repozitári**: 161+ (Oct 2025 – Feb 2026)

---

## Obsah dokumentu

1. [Mapovanie na kapitoly diplomovej práce](#1-mapovanie-na-kapitoly-diplomovej-práce)
2. [Literárna rešerš — akademické zdroje](#2-literárna-rešerš)
3. [Implementačné rozhodnutia s odôvodnením](#3-implementačné-rozhodnutia)
4. [Evolúcia projektu (git história)](#4-evolúcia-projektu)
5. [Porovnávacie tabuľky](#5-porovnávacie-tabuľky)
6. [Výkonnostné merania](#6-výkonnostné-merania)
7. [Kompletný zoznam referencií](#7-kompletný-zoznam-referencií)

---

## 1. Mapovanie na kapitoly diplomovej práce

### Kapitola 1 — Úvod (motivácia, definícia problému, ciele, štruktúra)

**Čo sem patrí:**
- Klimatické dáta sú distribuované cez desiatky portálov (CDS, ESGF, NOAA, NASA) v rôznych formátoch (NetCDF, GRIB, GeoTIFF, CSV)
- Vedci nemajú jednotný spôsob prehľadávania — manuálne prechádzanie katalógov
- Problém: Ako umožniť prirodzeno-jazykové dotazovanie nad heterogénnymi klimatickými dátami?
- Riešenie: RAG pipeline — vektorové vyhľadávanie + LLM generácia

**Akademické zdroje:**
- Rolnick et al. (2022): "Tackling Climate Change with Machine Learning" — motivácia pre ML v klimatológii [1]
- Nguyen et al. (2023): "ClimateBERT" — domain-specific NLP pre klimatické texty [2]
- Thulke et al. (2024): "ClimateGPT" — LLM pre klimatické otázky [3]

**Git kontext:**
- Commit `e37ab66` (2025-10-19): "Phase 1 and Phase 2 diploma start" — prvý commit projektu
- Commit `c32eb6e` (2025-10-22): "UI, Qdrant, testing, and RAG" — prvá funkčná RAG pipeline

---

### Kapitola 2 — Literárna rešerš

#### 2.1 Klimatické dáta a ich formáty

**2.1.1 Globálne klimatické datasety**

Projekt spracováva 233 záznamov z katalógu D1.1.xlsx (69 unikátnych datasetov):

| Dataset | Typ | Formát | Rozlíšenie | Zdroj |
|---------|-----|--------|------------|-------|
| ERA5 | Reanalýza | NetCDF/GRIB | 0.25°, hodinové | Copernicus CDS [4] |
| CMIP6 | Projekcie | NetCDF | variabilné | ESGF [5] |
| E-OBS | Pozorovania | NetCDF | 0.25°, denné | Copernicus/ECA&D [6] |
| CRU TS | Pozorovania | NetCDF | 0.5°, mesačné | UEA CRU [7] |
| CHIRPS | Zrážky | GeoTIFF | 0.05°, denné | UCSB [8] |
| WorldClim | Historické | GeoTIFF/NetCDF | 10 min | UC Davis [9] |
| GISTEMP | Anomálie | CSV | globálne | NASA GISS [10] |
| SPREAD | Zrážky (Španielsko) | NetCDF | 5 km, denné | CSIC [11] |

**Akademické zdroje pre formáty:**
- Rew & Davis (1990): NetCDF — Network Common Data Form [12]
- WMO (2003): GRIB (GRIdded Binary) — štandard WMO pre meteorologické dáta [13]
- CF Conventions: Climate and Forecast metadata conventions [14]

**2.1.2 Dátové formáty pre klimatické dáta**

| Formát | Knižnica | Detekcia v projekte | Poznámka |
|--------|----------|---------------------|----------|
| NetCDF-3/4 | xarray + netCDF4 | Magic bytes `CDF\x01`/`\x89HDF` | Primárny formát |
| GRIB/GRIB2 | xarray + cfgrib | Magic bytes `GRIB` | ERA5, predpovede |
| GeoTIFF | rasterio | Magic bytes `II*\x00`/`MM\x00*` | Satelitné snímky |
| HDF5 | xarray + h5netcdf | Magic bytes `\x89HDF` | Vedecké dáta |
| CSV/TSV | pandas | Textový fallback | Stanice, časové rady |
| ZIP/TAR/GZ | zipfile/tarfile/gzip | Kontajner → rekurzívne spracovanie | Balíčky |

Implementácia: `src/climate_embeddings/loaders/raster_pipeline.py` — trojvrstvová detekcia:
1. Prípona súboru → priamy loader
2. Magic bytes → identifikácia binárneho formátu
3. Fallback: xarray → CSV → chyba

**2.1.3 Nástroje na prácu s klimatickými dátami**

| Nástroj | Účel | Referencia |
|---------|------|------------|
| xarray | Labeled N-D arrays | Hoyer & Hamman (2017) [15] |
| Dask | Paralelné výpočty | Rocklin (2015) [16] |
| rasterio | GeoTIFF I/O | Gillies et al. (2013) [17] |
| pandas | Tabuľkové dáta | McKinney (2010) [18] |

---

#### 2.2 ETL procesy a orchestrácia dátových pipeline

**2.2.1 Princípy ETL procesov**

Náš systém implementuje ETL v 5 fázach (Phase 0–4):

| Fáza | Extract | Transform | Load |
|------|---------|-----------|------|
| Phase 0 | Čítanie Excel metadát | Generovanie textu z metadát | Embedding → Qdrant |
| Phase 1 | HTTP download (priamy prístup) | Raster chunking → štatistiky | Embedding → Qdrant |
| Phase 2 | Download s autentifikáciou | Rovnaký pipeline ako Phase 1 | Embedding → Qdrant |
| Phase 3 | API adaptéry (CDS, ESGF) | Rovnaký pipeline | Embedding → Qdrant |
| Phase 4 | Manuálne/kontaktné | Metadata-only (fallback) | Embedding → Qdrant |

**Akademický kontext:**
- Vassiliadis et al. (2002): ETL taxonomy — Extract, Transform, Load ako štandard [19]
- Dagster dokumentácia: Software-defined assets vs. task-based orchestration [20]

**2.2.2 Orchestračné nástroje**

| Nástroj | Výhody | Nevýhody | Prečo sme vybrali/nevybrali |
|---------|--------|----------|------------------------------|
| **Dagster** (vybraný) | Software-defined assets, type checking, webové UI | Zložitejšie nastavenie | Integrovaný monitoring, asset-based model |
| Apache Airflow | Zrelý ekosystém, komunita | Task-based (nie asset-based), ťažší debugging | Príliš generický pre vedecké dáta |
| Prefect | Jednoduchý Python API | Menšia komunita | Menej battle-tested |
| Luigi | Jednoduchý | Zastaralý, bez webového UI | Nedostatočné monitorovanie |

**Git kontext:**
- Commit `23d744c` (2025-12-16): Prechod z legacy standalone ETL na Dagster orchestráciu
- Commit `e375c3e` (2026-02-17): "Add catalog batch processing module with admin UI" — kompletný 5-fázový systém

---

#### 2.3 Vektorové databázy a embeddingy

**2.3.1 Textové embeddingy**

**Bi-encoder model: BAAI/bge-large-en-v1.5** (1024 dimenzií)
- Xiao et al. (2024): "C-Pack: Packaged Resources to Advance General Chinese Embedding", arXiv:2309.07597 [21]
- MTEB benchmark: #1 na retrieval úlohách (v čase výberu)
- Sentence-BERT architektúra: Reimers & Gurevych (2019), EMNLP [22]

**Cross-encoder reranker: BAAI/bge-reranker-v2-m3**
- Dvoj-fázový retrieval: bi-encoder (rýchly recall) → cross-encoder (presné rerankovanie)
- Nogueira & Cho (2019): "Passage Re-ranking with BERT" [23]
- Glass et al. (2022): "Re2G: Retrieve, Rerank, Generate" [24]

**Porovnanie embedding modelov** (akademicky podložené):

| Model | Dimenzie | MTEB Retrieval | Jazyk | Veľkosť | Referencia |
|-------|----------|----------------|-------|---------|------------|
| **BAAI/bge-large-en-v1.5** (vybraný) | 1024 | 63.55 | EN | 335M | Xiao et al. [21] |
| BAAI/bge-m3 | 1024 | 67.19 | Multi | 568M | Chen et al. [25] |
| E5-large-v2 | 1024 | 62.03 | EN | 335M | Wang et al. [26] |
| Sentence-BERT (all-MiniLM-L6) | 384 | 49.54 | EN | 22M | Reimers & Gurevych [22] |
| multilingual-e5-large | 1024 | 60.79 | Multi | 560M | Wang et al. [27] |

**Prečo BGE-large-en-v1.5:**
- Najvyšší retrieval score na MTEB v čase implementácie
- 1024-dim poskytuje dostatočnú sémantickú kapacitu
- Instruction-augmented: query prefix "Represent this sentence..." zlepšuje retrieval
- Normalizované embeddingy → cosine similarity

**2.3.2 Vektorové databázy**

**Qdrant** (vybraný) s HNSW indexom:

| Databáza | Index | Distribúcia | Disk support | gRPC | Prečo vybraný/nevybraný |
|----------|-------|-------------|--------------|------|--------------------------|
| **Qdrant** (vybraný) | HNSW | Áno | Áno | Áno | Najlepší payload filtering, gRPC, open-source |
| Pinecone | Proprietary | Cloud-only | N/A | N/A | Proprietárny, no self-hosting |
| Milvus | HNSW/IVF | Áno | Áno | Áno | Zložitejšia inštalácia |
| ChromaDB | HNSW | Nie | Áno | Nie | Vhodný len pre prototypy |
| Weaviate | HNSW | Áno | Áno | Čiastočne | Menej flexibilný filtering |

**HNSW algoritmus:**
- Malkov & Yashunin (2020): "Efficient and Robust Approximate Nearest Neighbor using Hierarchical Navigable Small World Graphs", IEEE TPAMI [28]
- Parametre v projekte: `m=16` (počet prepojení), `ef_construct=100` (presnosť stavby)
- Optimalizácia: `indexing_threshold=100M` počas bulk ingestion, `20K` po dokončení

---

#### 2.4 Retrieval-Augmented Generation

**2.4.1 Princíp RAG**

- Lewis et al. (2020): "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks", NeurIPS [29]
  - Založil RAG paradigmu: retrieve relevantné dokumenty → augmentuj LLM prompt → generuj odpoveď
- Gao et al. (2024): "Retrieval-Augmented Generation for Large Language Models: A Survey" [30]
  - Comprehensive survey RAG techník, kategorizácia Naive RAG → Advanced RAG → Modular RAG

**Naša implementácia (Advanced RAG):**
```
Query → Bi-encoder embedding → Qdrant semantic search (40 kandidátov)
                                        ↓
                              Cross-encoder reranking (top 10)
                                        ↓
                              Metadata keyword boosting (+0.05/match)
                                        ↓
                              Spatial/temporal filtering
                                        ↓
                              LLM data selection (výber relevantných premenných)
                                        ↓
                              Targeted per-variable search
                                        ↓
                              Compact context → LLM → Answer
```

**Dvoj-fázové LLM prompting (Two-Stage Prompting):**

Kľúčový architektonický vzor — LLM sa používa dvakrát:

| Fáza | Prompt | Vstup | Výstup | Implementácia |
|------|--------|-------|--------|---------------|
| **Stage 1: Data Selection** | `build_data_selection_prompt()` | Query + zoznam dostupných premenných, lokácií, časových období | Štruktúrovaný výber: `VARIABLES:`, `LOCATIONS:`, `TIME_PERIODS:` | `web_api/prompt_builder.py` |
| **Stage 2: Answer Generation** | `build_rag_prompt()` | Query + vybrané dáta + cielené per-variable výsledky | Kompletná odpoveď pre užívateľa | `web_api/prompt_builder.py` |

LLM v Stage 1 funguje ako **inteligentný query planner** — namiesto priameho retrieval na základe query, LLM najprv analyzuje otázku a selektuje relevantné premenné. Toto eliminuje irelevantné výsledky (napr. otázka o teplote nevráti zrážkové dáta).

**Dynamická detekcia typu otázky (`detect_question_type()`):**

Systém klasifikuje užívateľské otázky do 5 kategórií, pričom každá dostane špecializovaný prompt template:

| Typ otázky | Príklad | Špecifické inštrukcie v prompte |
|------------|---------|--------------------------------|
| `variable_list` | "What variables are available?" | Zoznamová odpoveď, žiadna analýza |
| `comparison` | "Compare temperatures in Prague vs Vienna" | Filtrovanie podľa lokácie/času, tabuľková odpoveď |
| `statistical` | "What is the mean precipitation?" | Mapovanie premenných, jednotkové konverzie |
| `temporal` | "How did temperature change 2000-2020?" | Časové filtrovanie, trendová analýza |
| `general` | "Tell me about ERA5 dataset" | Všeobecný popis, metadata-focused |

Ref: Tento prístup je inšpirovaný query decomposition (Shi et al. [32]) a agentic RAG patterns (Gao et al. [30]).

**2.4.2 Pokročilé RAG techniky**

| Technika | Referencia | Implementácia v projekte |
|----------|------------|--------------------------|
| Two-stage retrieval | Nogueira & Cho (2019) [23] | Bi-encoder (40) → cross-encoder rerank (10) |
| Metadata filtering | Qdrant docs [31] | Spatial bounding box, variable name filter |
| Query decomposition | Shi et al. (2024) [32] | LLM selektuje premenné, lokácie, časové obdobia |
| Contextual retrieval | Anthropic (2024) [33] | Metadata-as-prefix embedding (dataset name + variable + stats) |
| Hybrid search | Chen et al. (2024) [25] | Sémantické + metadata keyword boosting |

**2.4.3 RAG pre geopriestorové a klimatické dáta**

Toto je kľúčová sekcia — existuje málo práce na RAG pre klimatické dáta:

| Paper | Rok | Prínos | Relevancia pre náš projekt |
|-------|-----|--------|----------------------------|
| Yu et al.: "Spatial-RAG" | 2025 | Hybridný priestorový + sémantický retrieval | Priama inšpirácia pre spatial filtering [34] |
| Yu et al.: "RAG for Geoscience" | 2025 | Position paper: RAG pre geovedy | Identifikuje rovnaké problémy [35] |
| Huang et al.: "GeoGPT-RAG" | 2025 | Custom GeoEmbedding + GeoReranker (BGE-M3!) | Validácia nášho prístupu (tiež BGE) [36] |
| Vaghefi et al.: "ChatClimate" | 2023 | RAG pre IPCC reporty | Textový RAG, nie dátový [37] |
| Adamu et al.: "ClimatePub4KG" | 2025 | Knowledge graph pre klímu | Alternatívny prístup (KG vs. vektorový) [38] |
| Kong et al.: "CAIRNS" | 2025 | Citation-aware RAG pre klímu | Zodpovedný RAG s citáciami [39] |
| Nguyen et al.: "Responsible RAG" | 2024 | RAG pre klimatické rozhodovanie | Evaluačné aspekty [40] |

**2.4.4 Veľké jazykové modely**

Systém implementuje 3 LLM klientov s rozdielnymi prompt engineering stratégiami:

| Model | Provider | Klient | Default model | Poznámka |
|-------|----------|--------|---------------|----------|
| Grok 4.1 Fast | OpenRouter | `openrouter_client.py` | `x-ai/grok-4.1-fast` | Produkčný (aktuálny) |
| Granite 4 3B | Ollama (lokálny) | `ollama_client.py` (430 riadkov) | `granite4:3b` | Sofistikovaný prompt engineering |
| Llama 3.1 8B | Groq API | `groq_client.py` | `llama-3.1-8b-instant` | 30 req/min free tier |

**Ollama klient — anti-halucinácie:**
- Detekcia zámeny súradníc vs. teplôt (lat 48.5 ≠ teplota 48.5°C)
- Pravidlá pre Fahrenheit/Kelvin konverziu
- Detekcia count premenných (počet dní, mesiacov)
- Model warm-up pri inicializácii

**Fallback stratégia:**
- Ak LLM nie je dostupný → systém vracia raw vyhľadávacie výsledky
- Graceful degradácia: užívateľ stále vidí relevantné dáta
- Kaskádové prepínanie: OpenRouter → Groq → Ollama (konfigurovateľné)

---

#### 2.5 Evaluácia RAG systémov

**2.5.1 Retrieval metriky**

Implementované v `tests/test_rag_evaluation.py`:

| Metrika | Definícia | Cieľ v projekte | Referencia |
|---------|-----------|------------------|------------|
| Hit@K | Aspoň 1 relevantný v top-K | Hit@5 ≥ 0.80 | Manning et al. (2008) [41] |
| MRR@10 | 1/rank prvého relevantného | MRR@10 ≥ 0.60 | Voorhees (1999) [42] |
| NDCG@10 | Normalized Discounted Cumulative Gain | NDCG@10 ≥ 0.50 | Järvelin & Kekäläinen (2002) [43] |
| Recall@K | Podiel nájdených relevantných | Recall@5 ≥ 0.15 | — |

**2.5.2 End-to-End RAG metriky (RAGAS)**

- Shahul Es et al. (2024): "RAGAS: Automated Evaluation of Retrieval Augmented Generation", EACL [44]

| Metrika | Čo meria | Cieľ | LLM-judged? |
|---------|----------|------|-------------|
| Faithfulness | Odpoveď založená na kontexte (nie halucinovanie) | ≥ 0.85 | Áno |
| Context Precision | Relevantnosť retrieval výsledkov | ≥ 0.70 | Áno |
| Context Recall | Kompletnosť retrieval výsledkov | ≥ 0.75 | Áno |
| Answer Relevancy | Odpoveď adresuje otázku | ≥ 0.80 | Áno |
| Numerical Coverage | Kľúčové čísla z golden set v odpovedi | ≥ 0.70 | Nie (regex) |

**Porovnanie evaluačných frameworkov:**

| Framework | Referencia | Výhody | Nevýhody |
|-----------|------------|--------|----------|
| **RAGAS** (vybraný) | Es et al. (2024) [44] | Štandard v komunite, jednoduché API | Závisí na kvalite judge LLM |
| ARES | Saad-Falcon et al. (2024) [45] | Automatická konfidencia | Zložitejšia konfigurácia |
| TruLens | TruEra (2024) [46] | Komerčný, UI | Proprietárny |
| DeepEval | Confident AI (2024) [47] | Mnohé metriky | Menej stabilný |
| ARAGOG | Benchmarking paper [48] | End-to-end benchmark | Fixný dataset |

**2.5.3 Evaluačné aspekty pre klimatické RAG**

Naše špecifické výzvy (nie riešené v štandardných RAG benchmarkoch):
1. **Numerická presnosť**: Teplota 15.3°C vs. 15.8°C — dôležité pre vedcov
2. **Priestorová presnosť**: "Stredná Európa" ≠ "Celá Európa" — lat/lon filtering
3. **Časová relevancia**: Historické dáta vs. projekcie — nesprávne obdobie = zlá odpoveď
4. **Multi-variable**: Otázka môže zahŕňať teplotu AJ zrážky — multi-variable search
5. **Jednotková konzistentnosť**: Kelvin vs. Celsius, mm vs. mm/day

---

#### 2.6 Webové technológie

**2.6.1 Backend — FastAPI**
- Ramírez (2021): FastAPI framework [49]
- Asynchrónny Python web framework (ASGI)
- Automatická OpenAPI dokumentácia
- Pydantic validácia vstupov

**2.6.2 Frontend — Vue.js**
- You (2014): Vue.js progressive framework [50]
- SPA (Single Page Application) s Vue Router
- Tailwind CSS pre styling (tmavá téma)
- 8 stránok: Dashboard, Catalog, Sources, ETL Monitor, Schedules, Settings, Embeddings, Chat

---

#### 2.7 Kontajnerizácia a nasadenie

**2.7.1 Docker**
- Merkel (2014): "Docker: Lightweight Linux Containers for Consistent Development and Deployment" [51]
- 7 služieb v docker-compose.yml:
  1. dagster-postgres (:5432) — PostgreSQL pre Dagster stav + BatchProgress
  2. dagster-daemon — vykonávanie ETL jobov
  3. dagit (:3000) — Dagster webové UI
  4. web-api (:8000) — FastAPI s GPU prístupom (NVIDIA runtime)
  5. qdrant (:6333/:6334) — vektorová databáza (REST/gRPC)
  6. caddy (:80/:443) — reverse proxy s auto-HTTPS
  7. jupyter (:8888) — interaktívna analýza (voliteľné)

**2.7.2 Architektúra nasadenia**
```
Internet → Caddy (80/443, auto-TLS) → FastAPI (:8000) → Qdrant (:6333)
                                     → Dagit (:3000)      ↑
                                                     Dagster daemon → ETL jobs
```
- Server: Digital Ocean droplet (159.65.207.173)
- Doména: climaterag.online
- GPU: NVIDIA RTX 5090 (32 GB VRAM) — lokálny vývoj + embedding inference

---

### Kapitola 3 — Ciele práce a metodika

**3.1 Ciele práce**

**Hlavný cieľ:** Navrhnúť a implementovať systém pre spracovanie heterogénnych klimatických dát s využitím RAG, umožňujúci prirodzeno-jazykové dotazovanie nad vektorovou databázou klimatických embeddingov.

**Čiastkové ciele:**
1. Multi-formátový ETL pipeline (NetCDF, GRIB, GeoTIFF, CSV)
2. Embedding pipeline s GPU akceleráciou (BAAI/bge-large-en-v1.5)
3. Spatial-aware RAG retrieval s metadátovým filtrovaním
4. Webové rozhranie pre správu a dotazovanie
5. Evaluácia pomocou RAGAS frameworku
6. Nasadenie na produkčný server

**3.2 Metodika**

**3.2.1 Výskumná metodika**
- Design Science Research (Hevner et al., 2004) [52]: iteratívny návrh artefaktov
- Experimentálne porovnanie embedding modelov a retrieval stratégií
- Kvantitatívna evaluácia (RAGAS metriky)

**3.2.2 Technologický stack**

| Vrstva | Technológia | Verzia | Účel |
|--------|-------------|--------|------|
| Jazyk | Python | 3.11 | Hlavný vývojový jazyk |
| ML framework | PyTorch + CUDA | 2.10 + 12.4 | GPU-akcelerované embeddingy |
| Embedding model | BAAI/bge-large-en-v1.5 | — | Sémantické vyhľadávanie |
| Reranker | BAAI/bge-reranker-v2-m3 | — | Dvoj-fázový retrieval |
| Vektorová DB | Qdrant | 1.17+ | HNSW index, gRPC |
| Orchestrácia | Dagster | 1.12 | ETL pipeline management |
| Web backend | FastAPI | 0.129 | REST API |
| Web frontend | Vue.js 3 + Tailwind | — | SPA |
| Kontajnerizácia | Docker Compose | — | Multi-service deployment |
| Reverse proxy | Caddy | 2.x | Auto-HTTPS |

---

### Kapitola 4 — Návrh systému

**4.1 Celková architektúra systému**

```
┌──────────────────────────────────────────────────────────────┐
│                      Internet / Klient                        │
└──────────────────────┬───────────────────────────────────────┘
                       │ HTTPS (443)
                ┌──────▼──────┐
                │   Caddy     │  Auto-TLS (Let's Encrypt)
                └──────┬──────┘
           ┌───────────┼───────────┐
           │           │           │
    ┌──────▼──────┐    │    ┌──────▼──────┐
    │  FastAPI    │    │    │   Dagit     │
    │  (:8000)    │    │    │  (:3000)    │
    └──────┬──────┘    │    └──────┬──────┘
           │           │           │
    ┌──────▼──────┐    │    ┌──────▼──────┐
    │   Qdrant    │◄───┘    │   Dagster   │
    │  (:6333)    │         │   Daemon    │
    └─────────────┘         └──────┬──────┘
                                   │
                            ┌──────▼──────┐
                            │ PostgreSQL  │
                            │  (:5432)    │
                            └─────────────┘
```

**4.2 Návrh katalógového spracovania**

5-fázový model inspirovaný prístupnosťou dátových zdrojov:

| Fáza | Trigger | Vstup | Výstup | Čas |
|------|---------|-------|--------|-----|
| 0 | Automatický | Excel metadáta | 233 embedding bodov | <30s |
| 1 | Na požiadanie | HTTP download | Tisíce-státisíce bodov | Minúty-hodiny |
| 2 | Manuálna auth | Registračné portály | Rovnaké ako Phase 1 | Variabilné |
| 3 | API adaptéry | CDS/ESGF/NASA | Rovnaké ako Phase 1 | Variabilné |
| 4 | Manuálny | Kontaktné zdroje | Metadata-only | Okamžité |

**4.3 Návrh dátovej vrstvy**

**4.3.1 Databázový model (PostgreSQL)**

5 tabuliek (SQLAlchemy ORM, `src/database/models.py`):

| Tabuľka | Stĺpce | Účel |
|---------|--------|------|
| `sources` | id, name, url, format, status, is_deleted, created_at | Dátové zdroje (soft delete) |
| `source_credentials` | id, source_id, portal_type, credentials (JSON) | Prístupové údaje (per-source alebo globálne) |
| `source_schedules` | id, source_id, cron_expression, next_run, is_active | Automatické obnovovanie (croniter) |
| `processing_runs` | id, source_id, status, started_at, completed_at, duration_ms, error | História spracovania |
| `catalog_progress` | source_id, phase, status, dataset_name, error, started_at, completed_at | Stav batch spracovania |

**`SourceStore` trieda** (`src/database/source_store.py`):
- Source CRUD s soft delete (is_deleted flag)
- Credential management (per-source a globálne)
- Cron scheduling s `croniter` (výpočet next_run)
- Processing history s duration tracking
- Stale source detekcia (>30 dní od posledného update)

**Connection pooling** (`src/database/connection.py`):
- `pool_size=5`, `max_overflow=10`, `pool_pre_ping=True`
- Context manager `get_db_session()` s automatic commit/rollback

**Resume support**: pri reštarte kontajnera sa `processing` záznamy resetujú na `pending`

**4.3.2 Vektorové úložisko (Qdrant)**
- Kolekcia: `climate_data`
- Vektor: 1024-dim, COSINE distance
- Payload: štruktúrované metadáta (variable, source, time_start/end, lat/lon_min/max, štatistiky)
- HNSW index: m=16, ef_construct=100

**4.4 Návrh ETL pipeline**

**4.4.1 Extract — sťahovanie dát**
- Direct download URLs: 69 overridov pre portálové stránky → priame súbory
- Validácia: Content-Type check, HTML/XML detekcia po stiahnutí
- Timeout: 600s pre download, 30 min pre raster loading
- Disk guard: stop ak < 5 GB voľného miesta
- Memory guard: GC + čakanie ak > 85%, stop ak > 90%

**4.4.2 Transform — spracovanie dát**
- Chunking stratégia (kľúčový dizajnový rozhodnutie):
  - Čas: 30 timestepov/chunk (~1 mesiac pre denné dáta)
  - Priestor: 10 buniek/chunk (~2.5° pri 0.25° rozlíšení)
  - Motivácia: Granulárne RAG odpovede na regionálnej/mesačnej úrovni
- Pre-load stratégia:
  - Malé premenné (< 6 GB): načítanie celého array do RAM → numpy slicing
  - Veľké premenné: dávkový dask.compute() (500 slices naraz)

**4.4.3 Load — ukladanie výsledkov**
- Asynchrónny embed+upsert pipeline:
  1. GPU embedding (3.5s / 2000 textov na RTX 5090)
  2. Paralelný Qdrant upsert v background thread
  3. Backpressure: čaká na predchádzajúci upsert pred novým

**4.5 Návrh RAG pipeline**

Trojfázový retrieval:
1. **Sémantické vyhľadávanie** (bi-encoder): 40 kandidátov
2. **Cross-encoder reranking**: top 10
3. **LLM data selection**: výber relevantných premenných → cielené per-variable vyhľadávanie

Spatial-aware filtering:
- Bounding box: ±2° (~200 km) okolo detegovanej lokácie
- Temporal overlap: hit_start ≤ req_end AND hit_end ≥ req_start

**4.6 Návrh webového rozhrania**

10 Vue.js stránok (`web_api/frontend/src/views/`):

| Stránka | Vue komponent | Účel | Kľúčové komponenty |
|---------|---------------|------|---------------------|
| Dashboard | `Dashboard.vue` | Prehľad systému | Qdrant štatistiky, počet zdrojov, posledná aktivita |
| Catalog | `Catalog.vue` | Prehliadanie Excel katalógu | Phase status badgy, trigger spracovania, filtrovanie |
| Sources | `Sources.vue` | Zoznam dátových zdrojov | Tabuľka s status indikátormi |
| Create Source | `CreateSource.vue` | Pridanie nového zdroja | Formulár s validáciou, format detekcia |
| ETL Monitor | `ETLMonitor.vue` | Real-time monitorovanie | Log viewer, progress bars |
| Schedules | `Schedules.vue` | Dagster schedule management | Enable/disable, cron konfigurácia |
| Settings | `Settings.vue` | Systémová konfigurácia | Disk/memory info, Qdrant stats |
| **Chat** | `Chat.vue` | **RAG dotazovanie** | Konverzačné vlákna, filter bar (source/variable), quick question buttons, chunk detail accordion, spatial filter badge, timing metadata |
| **Embeddings** | `Embeddings.vue` | **Správa vektorového úložiska** | Collection health indikátor, dataset/variable breakdown, distribučné stĺpce, sample records, akcie (regenerate, optimize, export, clear) |
| Login | `Login.vue` | Autentifikácia | MENDELU branding |

---

### Kapitola 5 — Implementácia

#### 5.1 Štruktúra projektu

```
xchomo_diploma/
├── src/
│   ├── catalog/          # 5-fázový batch processing
│   │   ├── excel_reader.py
│   │   ├── phase_classifier.py
│   │   ├── metadata_pipeline.py
│   │   ├── batch_orchestrator.py
│   │   ├── location_enricher.py
│   │   └── portal_adapters.py
│   ├── climate_embeddings/
│   │   ├── loaders/
│   │   │   └── raster_pipeline.py  # Multi-format loader
│   │   ├── embeddings/
│   │   │   └── text_models.py      # BGE embedding + ONNX
│   │   └── schema.py               # ClimateChunkMetadata
│   ├── embeddings/
│   │   └── database.py             # Qdrant wrapper
│   ├── database/
│   │   ├── connection.py           # PostgreSQL session (pooling)
│   │   ├── models.py               # SQLAlchemy models (5 tabuliek)
│   │   └── source_store.py         # Source CRUD, scheduling, credentials
│   └── llm/                        # LLM clients (Ollama, Groq, OpenRouter)
├── dagster_project/                # ETL orchestrácia
├── web_api/
│   ├── main.py                     # FastAPI centrálny endpoint (25+ routes)
│   ├── rag_endpoint.py             # RAG query pipeline
│   ├── prompt_builder.py           # Two-stage prompting + 5 question types
│   └── frontend/                   # Vue.js SPA (10 stránok)
├── tests/
│   ├── test_catalog.py
│   ├── test_rag_evaluation.py      # 3-tier evaluácia
│   └── fixtures/golden_queries.json
├── config/pipeline_config.yaml
└── docker-compose.yml
```

#### 5.2 Katalóg a správa zdrojov

**5.2.1 Čítanie Excel katalógu**
- Vstup: `data/Kopie souboru D1.1.xlsx` (233 riadkov, stĺpce: dataset, hazard, link, access, formát, ...)
- Riešenie merged cells: `pd.DataFrame.ffill()` pre hazard stĺpec
- Výstup: zoznam `CatalogEntry` dataclass objektov

**5.2.2 Klasifikácia do fáz** (`phase_classifier.py`)

Decision tree s `_PORTAL_DOMAINS` registrom — **38 domén** mapovaných na 12 portálových typov:

| Portálový typ | Domény | Phase |
|---------------|--------|-------|
| CDS | cds.climate.copernicus.eu, cds-beta.climate.copernicus.eu | 3 |
| ESGF | esgf-node.llnl.gov, esgf-data.dkrz.de, esg-dn1.nsc.liu.se | 3 |
| NOAA | psl.noaa.gov, ncdc.noaa.gov, ncei.noaa.gov | 3 |
| NASA | disc.gsfc.nasa.gov, earthdata.nasa.gov | 3 |
| ECAD | knmi.nl, surfobs.climate.copernicus.eu | 2 |
| MARINE | data.marine.copernicus.eu | 3 |
| ... | + NCAR, METEO, WMO, PORTAL, JMA, CEDA | 2-3 |

Klasifikačná logika:
1. `DIRECT_DOWNLOAD_URLS` override → Phase 1 (69 overených priamych URL)
2. Multi-value links ("or" separator) → Phase 4
3. "Contact" v access field → Phase 4
4. URL-less záznamy → Phase 4
5. Doménová zhoda → Phase 2 alebo 3
6. Default: Phase 1 (priamy download pokus)

**5.2.4 Priestorové obohatenie** (`location_enricher.py`)

5-úrovňová stratégia obohatenia lokačných dát (bez externých geocoding API — schválené vedúcim):

| Úroveň | Zdroj | Príklad | Fallback |
|--------|-------|---------|----------|
| 1 | Excel Region/Country | "Czech Republic" → bbox [12.1, 48.6, 18.9, 51.1] | → Úroveň 2 |
| 2 | Station name | "Berlin-Tempelhof" → [13.40, 52.47, 13.41, 52.48] | → Úroveň 3 |
| 3 | Spatial coverage text | "48.5N-51.1N, 12.1E-18.9E" → parsed bbox | → Úroveň 4 |
| 4 | Dataset name inference | "e-obs" → Europe, "CERRA" → Europe | → Úroveň 5 |
| 5 | Geografická zóna | 8 kurátorských zón (Europe, North America, Arctic, ...) | → Global |

Kurátorské bounding boxy (`_BBOX_ZONES`): Europe [-25, 34, 45, 72], North America [-170, 15, -50, 85], Arctic [-180, 66, 180, 90], atď.

**5.2.5 Portálové adaptéry** (`portal_adapters.py`) — Phase 3

5 konkrétnych adaptérov zdieľajúcich spoločnú `_process_file()` metódu:

| Adaptér | Počet dataset configs | Metóda prístupu | Kľúčový detail |
|---------|----------------------|-----------------|----------------|
| **CDSAdapter** | 6 CDS API configs | `cdsapi` Python klient | URL parsing → dataset ID, name-to-ID mapping |
| **ESGFAdapter** | 4 OpenDAP URL | xarray OpenDAP slicing (prvých 12 timestepov) | HTTP fallback s 500 MB limitom |
| **NOAAAdapter** | 5 priamych URL | HTTP download | HTML content-type detekcia |
| **NASAAdapter** | 5 dataset URL | OAuth bearer token (`NASA_EARTHDATA_TOKEN`) | Earthdata autentifikácia |
| **MarineCopernicusAdapter** | — | `copernicusmarine` Python toolbox | Product ID extrakcia z URL |

**5.2.3 Batch orchestrátor**

Kľúčové implementačné detaily:

```python
# Resume support (PostgreSQL-backed)
completed_set = progress.get_completed_set(phase=1)  # 1 DB query namiesto 233
to_process = [e for e in entries if e.source_id not in completed_set]

# Smallest-first ordering (HEAD requests)
download_candidates = _prefetch_sizes(candidates, phase)  # Parallel HEAD

# HNSW optimization
db.disable_indexing()   # indexing_threshold = 100M
try:
    for entry in download_candidates:
        # Download → Load → Chunk → Embed → Upsert
finally:
    db.enable_indexing()  # indexing_threshold = 20K → single rebuild
```

**Git evolúcia orchestrátora:**
| Commit | Dátum | Zmena |
|--------|-------|-------|
| `e375c3e` | 2026-02-17 | Prvá implementácia batch spracovania |
| `ce3d2dc` | 2026-02-17 | Per-phase status tracking |
| `af7cd84` | 2026-02-17 | URL validácia, SKIP_PHASE1 |
| `fc7bd2a` | 2026-02-18 | Batch upsert 50→500 s retry |
| `fcf261b` | 2026-02-18 | Frontend + pipeline overhaul |
| `5b7b469` | 2026-02-18 | Chunking strategy optimalizácia (30 timesteps, 10 spatial) |
| (latest) | 2026-02-22 | Batch dask compute, async upsert, HNSW disable/enable |

#### 5.3 Spracovanie klimatických dát

**5.3.1 Detekcia formátu**
Trojvrstvová stratégia:
1. Prípona (`load_raster_auto()`)
2. Magic bytes (`_detect_format_from_magic()`)
3. Engine fallback chain: `netcdf4 → h5netcdf → scipy → auto`

**5.3.2 Raster pipeline**

**Chunking stratégia** (akademicky odôvodnená):

Chunking je kľúčové rozhodnutie pre kvalitu RAG:

| Parameter | Hodnota | Dôvod | Alternatíva | Prečo nie |
|-----------|---------|-------|-------------|-----------|
| Časový chunk | 30 timestepov | ~1 mesiac pre denné dáta, interpretovateľné obdobie | 1 (denný), 365 (ročný) | 1 = príliš veľa chunkú, 365 = strata detailov |
| Priestorový chunk | 10 buniek | ~2.5° pri 0.25° rozlíšení, regionálna presnosť | 1 (pixel), 100 (25°) | 1 = milióny chunkú, 100 = Rím až Štokholm |

Ref: Spatial-RAG (Yu et al., 2025) [34] potvrdzuje dôležitosť priestorového chunking pre geopriestorový RAG.

**Pamäťová optimalizácia** (batched dask compute):

| Problém | Riešenie | Speedup | Ref |
|---------|----------|---------|-----|
| Per-chunk dask.compute() = 0.45s/chunk | Batch 500 slices → 1 dask.compute() | **55x** | Rocklin (2015) [16] |
| 2 GB pre-load cap na 64 GB stroji | Zvýšenie na 6 GB (30% available RAM) | — | — |
| Fallback ak batch zlyhá | Per-chunk compute pre problematické batch | Robustnosť | — |

**5.3.3 Metadátová schéma** (`ClimateChunkMetadata`, `schema.py`)

Normalizovaná schéma s 25+ poliami:

| Kategória | Polia | Príklad |
|-----------|-------|---------|
| Identifikácia | `source_id`, `dataset_name`, `variable`, `long_name` | ERA5, t2m, 2m temperature |
| Priestor | `lat_min`, `lat_max`, `lon_min`, `lon_max`, `location_name` | 48.5, 49.0, 16.5, 17.0, Czech Republic |
| Čas | `time_start`, `time_end` | 2020-01-01, 2020-12-31 |
| Štatistiky | `stats_mean`, `stats_std`, `stats_min`, `stats_max`, `stats_p10`, `stats_median`, `stats_p90`, `stats_range` | 288.4, 5.2, 260.1, 310.5, ... |
| Katalóg | `hazard_type`, `data_type`, `region_country`, `impact_sector`, `access_type`, `catalog_source` | Temperature, Reanalysis, Europe, Agriculture, Open |
| Jednotky | `units` | K, mm, °C |

**`from_chunk_metadata()` factory** normalizuje varianty názvov polí (`lat_min`/`min_lat`, `time_start`/`start_date`).

**`generate_human_readable_text()`** — dynamická generácia textu z metadát (text NIE JE uložený v DB):
- 3 úrovne verbozity: `low` (embedding), `medium` (reranking), `high` (LLM kontext)
- Výpočet počtu dní medzi time_start a time_end
- Unit-aware formátovanie štatistík
- Spracovanie katalógových polí (hazard_type, data_type, ...)
- ~200 riadkov kódu

**5.3.4 Generovanie embeddingov**

Textový embedding namiesto štatistického vektora:
- **Pôvodný prístup** (verzia 1): 8-dim štatistický vektor (mean, std, min, max, P10, median, P90, range)
  - Problém: nedokáže rozlíšiť teplotu od zrážok (obe majú mean/std)
  - Problém: dva nekompatibilné vektorové priestory (Phase 0: 1024-dim text, Phase 1: 8-dim stats)
- **Aktuálny prístup** (verzia 2): 1024-dim text embedding z metadát + štatistík
  - Text: "Variable: temperature, Dataset: ERA5, mean=15.3°C, range=12.1-18.5°C, time=2020-01..2020-12, lat=48.5-49.0, lon=16.5-17.0"
  - Embedding: BAAI/bge-large-en-v1.5 na GPU (3.5s / 2000 textov)
  - Výhoda: sémantické porozumenie, jeden vektorový priestor

**ONNX backend optimalizácia** (`_detect_backend()`, `text_models.py`):

4-úrovňová priorita inferencie:

| Priorita | Backend | Speedup vs CPU | Poznámka |
|----------|---------|----------------|----------|
| 1 | PyTorch + CUDA | 10-50x | Primárny (RTX 5090) |
| 2 | ONNX + CUDAExecutionProvider | 8-40x | Fallback ak PyTorch CUDA zlyhá |
| 3 | ONNX + CPUExecutionProvider | 2-3x | Rýchlejší ako PyTorch CPU |
| 4 | PyTorch CPU | 1x (baseline) | Posledná možnosť |

Poznámka: BGE-M3 nie je kompatibilný s ONNX exportom (non-standard output schema).

Ref: Research_embedding_strategy.md, Spatial-RAG [34], GeoGPT-RAG [36]

#### 5.4 Vektorové úložisko

**Qdrant konfigurácia:**
```python
VectorParams(size=1024, distance=Distance.COSINE)

# Bulk ingestion optimization (Qdrant best practice):
disable_indexing():  indexing_threshold = 100_000_000
enable_indexing():   indexing_threshold = 20_000
```

**Batch upsert:**
- 2000 bodov/batch cez gRPC (binárny protobuf)
- Retry: 3 pokusy s exponenciálnym backoffom (5, 10, 20s)
- Deterministic UUIDs: `uuid5(source_id)` pre reprodukovateľnosť

Ref: Malkov & Yashunin (2020) [28] — HNSW indexing; Qdrant docs [31] — bulk ingestion patterns

#### 5.5 RAG pipeline

**5.5.1 Spatial-aware retrieval**

Implementácia v `rag_endpoint.py`:
```python
# 1. Sémantický search (bi-encoder)
results = db.search(query_vector, limit=40)

# 2. Cross-encoder reranking
results = db.search_and_rerank(query_text, query_vector, limit=10, candidates=40)

# 3. Metadata keyword boosting
for hit in results:
    if query_keyword in hit.payload["variable"]:
        hit.score += 0.05  # max +0.15

# 4. Spatial filtering (ak detegovaná lokácia)
filtered = [h for h in results if _overlaps_bbox(h, lat, lon, margin=2.0)]
```

Ref: Spatial-RAG [34], Filtered ANN [53]

**5.5.2 Generation**

LLM prompt štruktúra:
```
You are a climate data expert. Answer based ONLY on the provided data.
Be precise with numbers. Include uncertainty where relevant.

=== Retrieved Climate Data ===
[1] source=ERA5, var=T2M, score=0.89, mean=288.4K, ...
[2] source=GISTEMP, var=anomaly, score=0.85, mean=1.35°C, ...
...

Question: {user_query}
```

**5.5.3 RAG endpoint a optimalizácie**

| Endpoint | Metóda | Účel | Kľúčový detail |
|----------|--------|------|----------------|
| `/rag/query` | POST | Hlavný RAG endpoint | query, top_k, search_only, conversation_history |
| `/rag/search` | POST | Čistý vektorový search | Bez LLM generácie, pre debugging |
| `/rag/collection-info` | GET | Metadata kolekcie | Lightweight, bez embedder/LLM loading |

**Optimalizácie:**
- **Lazy inicializácia**: Prvý request ~5s (loading models), následné: okamžité
- **Variable list caching**: `_get_variable_list()` scrolluje Qdrant (50 rounds × 500 bodov), 5-min TTL cache
- **Fast path**: `_is_variable_list_question()` regex detekcia → vracia cached zoznam bez embedding/LLM
- **Dynamický token budget**: `_default_max_tokens()` prispôsobuje LLM token limit podľa dĺžky a typu otázky
- **Conversation history**: `RAGRequest.conversation_history: Optional[List[RAGMessage]]` pre multi-turn konverzácie

#### 5.6 Orchestrácia — Dagster

**5.6.1 Registrované joby** (`repository.py`):

| Job | Fázy | Trigger |
|-----|------|---------|
| `batch_catalog_etl_job` | Phase 0 + Phase 1 | Manuálny (API/UI) |
| `catalog_metadata_only_job` | Phase 0 | Automatický (weekly schedule) |
| `catalog_full_etl_job` | Phase 0 → 1 → 2 → 3 (sekvenčné chaining) | Manuálny |
| `single_source_etl_job` | Jednotlivý zdroj | Sensor-triggered |
| `dynamic_source_etl_job` | Legacy: všetky aktívne zdroje | Manuálny |

4 Dagster resources: `ConfigLoaderResource`, `LoggerResource`, `DataPathResource`, `DatabaseResource`.

**5.6.2 Senzory a automatické plánovanie** (`schedules.py`):

| Sensor/Schedule | Interval | Akcia |
|-----------------|----------|-------|
| `source_schedule_sensor` | 60s polling | Kontroluje `source_schedules` tabuľku, spúšťa `single_source_etl_job` pre due zdroje, posúva schedule na ďalšiu occurence |
| `data_freshness_sensor` | Hodinový | Detekcia stale zdrojov (>30 dní od posledného update), triggeruje refresh |
| `weekly_catalog_refresh` | Cron `0 3 * * 0` (nedeľa 3:00) | Spúšťa `catalog_metadata_only_job` (Phase 0) |

**5.6.3 Sekvenčné chaining fáz** (`catalog_jobs.py`):

`catalog_full_etl_job` reťazí fázy s dependency passing:
```python
classify → phase0 → phase1 → phase2 → phase3
```
Každá phase op zaznamenáva processing runs v PostgreSQL cez `SourceStore.record_processing_run()` a `complete_processing_run()`.

#### 5.7 Webové rozhranie

**5.7.1 Backend API**
- 25+ endpointov v `main.py`
- Source CRUD, Catalog browse, RAG query, Schedule management, System info

**5.7.2 Frontend**
- Vue.js 3 SPA s Vue Router
- Dark theme, Tailwind CSS, responsive
- Vite build (100 kB gzipped JS bundle)

#### 5.8 Docker nasadenie

- NVIDIA CUDA 12.4.1 runtime base image
- Python 3.11 + Node.js 20 (multi-stage build)
- GPU passthrough: `deploy.resources.reservations.devices`
- Persistent volumes: `qdrant_storage`, `dagster_postgres`

---

### Kapitola 6 — Testovanie a evaluácia

#### 6.1 Testovacia stratégia

3 úrovne testovania:
1. **Unit testy**: pytest, mockované externé služby
2. **Integračné testy**: Reálny Qdrant + PostgreSQL
3. **Evaluácia RAG**: RAGAS framework s golden test set

#### 6.2 Unit testy

**6.2.1 Testy katalógu** (`test_catalog.py`):
- Čítanie Excel, klasifikácia do fáz, metadata embedding
- Testovanie Phase 0 batch procesu

**6.2.2 Testy embeddingov**:
- Verifikácia dimenzionality (1024-dim)
- Normalizácia vektorov
- Backend auto-detection (CUDA → ONNX → CPU)

**6.2.3 Testy Dagster**:
- Job definícia validácia
- Resource konfigurácia

**6.2.4 Testy web API**:
- Health endpoint
- Source CRUD
- RAG query (s mockovaným LLM)

#### 6.3 Evaluácia RAG pipeline — trojvrstvová metodológia

Implementované v `tests/test_rag_evaluation.py` — 3 nezávislé evaluačné vrstvy:

**6.3.1 Golden test set** (`tests/fixtures/golden_queries.json`):

**52 otázok** v 6 kategóriách:

| Kategória | Počet | Príklad | Obtiažnosť |
|-----------|-------|---------|------------|
| `variable-specific` | 10 | "What is the mean temperature in ERA5?" | easy-hard |
| `spatial` | 10 | "Show precipitation data for Central Europe" | medium-hard |
| `temporal` | 8 | "Temperature trends from 2000 to 2020" | medium |
| `cross-variable` | 8 | "Compare temperature and precipitation" | hard |
| `methodological` | 8 | "What reanalysis products cover Europe?" | medium |
| `dataset-specific` | 8 | "What variables does SPREAD contain?" | easy-medium |

Každý záznam obsahuje: `relevant_chunk_ids` (auto-anotované z reálnych Qdrant dát, 223 195 bodov), `key_numbers`, `notes`, `difficulty`.

Generované skriptom `scripts/annotate_golden_queries.py`.

**6.3.2 Tier 1: Retrieval metriky** (7 testov)

| Metrika | Threshold | Bez rerankingu | S rerankingom |
|---------|-----------|----------------|---------------|
| Hit@5 | ≥ 0.80 | — | — |
| Hit@10 | ≥ 0.90 | — | — |
| MRR@10 | ≥ 0.60 | — | ≥ 0.70 |
| NDCG@10 | ≥ 0.50 | — | ≥ 0.70 |
| Recall@3 | ≥ 0.10 | — | — |
| Recall@5 | ≥ 0.15 | — | — |
| Recall@10 | ≥ 0.30 | — | — |

**Tier 1b: Reranking testy** (2 testy): NDCG@10 reranked ≥ 0.70, MRR@10 reranked ≥ 0.70

**6.3.3 Tier 2: RAGAS evaluácia** (5 testov)

| Metrika | Čo meria | Cieľ | LLM-judged? |
|---------|----------|------|-------------|
| Faithfulness | Odpoveď založená na kontexte (nie halucinovanie) | ≥ 0.85 | Áno |
| Context Precision | Relevantnosť retrieval výsledkov | ≥ 0.70 | Áno |
| Context Recall | Kompletnosť retrieval výsledkov | ≥ 0.75 | Áno |
| Answer Relevancy | Odpoveď adresuje otázku | ≥ 0.80 | Áno |
| **Numerical Coverage** | Kľúčové čísla z golden set v odpovedi | ≥ 0.70 | **Nie (regex)** |

**Numerical Coverage** je vlastná metrika — kontroluje regex matchom, či sa kľúčové čísla z golden annotations objavujú v RAG odpovedi. Dôležité pre klimatických vedcov, kde presnosť čísel je kritická.

Ref: Es et al. (2024) [44], Saad-Falcon et al. (2024) [45]

**6.3.4 Tier 3: Embedding Space Analysis** (3 testy)

| Test | Čo meria | Threshold |
|------|----------|-----------|
| Intra vs. inter-variable similarity | Chunks rovnakej premennej sú bližšie ako rôzne premenné | intra > inter |
| Inter-variable centroid distance | Centroids premenných sú dostatočne vzdialené | > 0.01 |
| Nearest neighbor sanity | 3 sanity queries vracajú sémanticky relevantné výsledky | Manuálna kontrola |

Táto vrstva overuje kvalitu embedding priestoru nezávisle od LLM.

#### 6.4 Porovnanie LLM modelov

Plánované porovnanie generácie pre rovnaké retrieval výsledky:

| Model | Latencia | Faithfulness | Numerical Accuracy | Cena |
|-------|----------|--------------|-------------------|------|
| GPT-4o-mini | ~2s | — | — | $0.15/1M |
| Llama 3.1 70B | ~5s | — | — | Lokálny |
| Claude Sonnet | ~3s | — | — | $3/1M |

#### 6.5 Výkonnostné testovanie

**Embedding pipeline benchmark (pred vs. po optimalizácii):**

| Dataset | Chunky | Pred (per-chunk dask) | Po (batch dask) | Speedup |
|---------|--------|-----------------------|-----------------|---------|
| SPREAD | ~140K | ~16+ hodín | ~18 minút | **55x** |
| E-OBS | ~33K | ~4 hodiny | ~15 minút | **16x** |
| CRU TS | ~5.7K | ~2.3 min | ~1 min | **2x** |
| SPI-GPCC | 2.078 | ~10 min | ~6s | **100x** |

**Breakdown per batch (2000 chunks, SPREAD):**

| Fáza | Pred optimalizáciou | Po optimalizácii |
|------|---------------------|------------------|
| Chunk iteration | ~895s (per-chunk dask) | ~8s (batched dask) |
| GPU embedding | ~3.5s | ~3.5s (bez zmeny) |
| Qdrant upsert | ~1.5s | ~1.0s (HNSW disabled) |
| **Celkovo** | **~900s / 2000 chunks** | **~15s / 2000 chunks** |

---

### Kapitola 7 — Diskusia

#### 7.1 Zhrnutie dosiahnutých výsledkov

1. ✅ Multi-formátový ETL pipeline (NetCDF, GRIB, GeoTIFF, CSV, ZIP, TAR, GZ)
2. ✅ GPU-akcelerované embeddingy (BAAI/bge-large-en-v1.5, RTX 5090)
3. ✅ Spatial-aware RAG retrieval
4. ✅ Webové rozhranie (Vue.js + FastAPI)
5. ✅ Produkčné nasadenie (Docker, Caddy, auto-HTTPS)
6. ✅ 5-fázový katalógový systém (233 záznamov, 69 datasetov)
7. ✅ 55x zrýchlenie embedding pipeline

#### 7.2 Porovnanie s existujúcimi riešeniami

| Systém | Formáty | RAG | Spatial filtering | Vektorová DB | Open-source |
|--------|---------|-----|-------------------|--------------|-------------|
| **Náš systém** | 6+ (NetCDF, GRIB, ...) | ✅ Pokročilý (2-stage) | ✅ Bounding box | Qdrant | ✅ |
| ChatClimate [37] | Len text (IPCC) | ✅ Základný | ❌ | FAISS | ❌ |
| GeoGPT-RAG [36] | GIS formáty | ✅ Custom GeoEmbedding | ✅ | Vlastný | ❌ |
| Spatial-RAG [34] | Geo-JSON | ✅ Hybridný | ✅ Prepracovaný | Vlastný | ❌ |
| ClimatePub4KG [38] | Len text | ❌ (KG-based) | ❌ | Neo4j | ✅ |
| CAIRNS [39] | Len text | ✅ Citation-aware | ❌ | — | ❌ |

**Naša jedinečnosť:**
- Jediný systém spracovávajúci multi-formátové klimatické DÁTA (nie len texty)
- Generator-based streaming: nikdy nenahráva celý raster do RAM
- 5-fázový prístup k heterogénnym zdrojom (od open-access po manuálne)
- Plne open-source, self-hosted, GPU-akcelerovaný

#### 7.3 Evolúcia embedding stratégie

| Verzia | Prístup | Problémy | Akademická motivácia |
|--------|---------|----------|----------------------|
| V1 (Oct 2025) | 8-dim štatistický vektor | Nemožno rozlíšiť premenné | — |
| V2 (Jan 2026) | 1024-dim text embedding (BGE-M3) | Dva vektorové priestory | Reimers & Gurevych [22] |
| V3 (Feb 2026) | 1024-dim text embedding (BGE-large) + reranking | Pomalý ingestion | Xiao et al. [21], Nogueira [23] |
| V3.1 (Feb 2026) | + Batch dask, async upsert, HNSW disable | — | Rocklin [16], Malkov [28] |

#### 7.4 Limitácie

1. **Jazykové obmedzenie**: Embeddingy sú anglické (BGE-large-en); multijazyčný model (BGE-M3) je pomalší
2. **Phase 2-4 neúplné**: Registračné portály a API adaptéry nie sú plne implementované
3. **Evaluácia na malom golden sete**: 50-100 otázok nemusí pokryť všetky scenáre
4. **LLM závislosť na externom API**: OpenRouter = single point of failure
5. **Numerická presnosť**: LLM môže zaokrúhliť alebo zameniť čísla z kontextu

#### 7.5 Možnosti rozšírenia a budúca práca

1. **Multimodálne embeddingy**: Kombinácia textových + priestorových embeddingov (ref: AlphaEarth [54])
2. **Knowledge Graph integrácia**: ClimatePub4KG prístup pre vzťahy medzi datasetmi [38]
3. **Fine-tuning**: RAFT prístup (Zhang et al., 2024) — domain-specific fine-tuning [55]
4. **Streaming RAG**: Real-time aktualizácia vektorovej databázy pri nových dátach
5. **Federovaný prístup**: Priame dotazovanie nad remote ESGF/CDS bez lokálneho stiahnutia

---

## 2. Literárna rešerš — kompletné akademické zdroje

### Foundational RAG
- [29] Lewis et al. (2020): "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks", NeurIPS 2020
- [30] Gao et al. (2024): "Retrieval-Augmented Generation for Large Language Models: A Survey", arXiv:2312.10997
- [32] Shi et al. (2024): "REPLUG: Retrieval-Augmented Black-Box Language Models", NAACL 2024
- Izacard & Grave (2021): "Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering", EACL 2021
- Borgeaud et al. (2022): "Improving Language Models by Retrieving from Trillions of Tokens", ICML 2022

### Dense Retrieval & Embeddings
- [21] Xiao et al. (2024): "C-Pack: Packaged Resources to Advance General Chinese Embedding", arXiv:2309.07597
- [22] Reimers & Gurevych (2019): "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks", EMNLP 2019
- [25] Chen et al. (2024): "BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity", arXiv:2402.03216
- [26] Wang et al. (2024): "Text Embeddings by Weakly-Supervised Contrastive Pre-training (E5)", ACL 2024
- Karpukhin et al. (2020): "Dense Passage Retrieval for Open-Domain QA (DPR)", EMNLP 2020
- Muennighoff et al. (2023): "MTEB: Massive Text Embedding Benchmark", EACL 2023

### Cross-Encoder Reranking
- [23] Nogueira & Cho (2019): "Passage Re-ranking with BERT", arXiv:1901.04085
- [24] Glass et al. (2022): "Re2G: Retrieve, Rerank, Generate", NAACL 2022

### Vektorové databázy
- [28] Malkov & Yashunin (2020): "Efficient and Robust Approximate Nearest Neighbor using Hierarchical Navigable Small World Graphs", IEEE TPAMI
- [53] Amanbayev, A., Tsan, B., Dang, T. & Rusu, F. (2025): "Filtered Approximate Nearest Neighbor Search in Vector Databases", arXiv:2602.11443

### Klimatické a geopriestorové RAG
- [34] Yu et al. (2025): "Spatial-RAG: Spatial Retrieval Augmented Generation", arXiv:2502.18470
- [35] Yu, R., et al. (2025): "RAG for Geoscience: What We Expect, Gaps and Opportunities", arXiv:2508.11246
- [36] Huang et al. (2025): "GeoGPT-RAG", arXiv:2509.09686
- [37] Vaghefi et al. (2023): "ChatClimate: Grounding Conversational AI in Climate Science", arXiv:2304.05510
- [38] Adamu et al. (2025): "ClimatePub4KG: A Climate Knowledge Graph", arXiv:2509.10087
- [39] Kong, L., Joshi, A. & Karimi, S. (2025): "CAIRNS: Balancing Readability and Scientific Accuracy in Climate Adaptation Question Answering", arXiv:2512.02251
- [40] Nguyen et al. (2024): "Responsible RAG for Climate Decision-Making", arXiv:2410.23902

### Klimatická AI
- [1] Rolnick et al. (2022): "Tackling Climate Change with Machine Learning", ACM Computing Surveys
- [2] Nguyen et al. (2023): "ClimateBERT: A Pretrained Language Model for Climate-Related Text"
- [3] Thulke et al. (2024): "ClimateGPT: Towards AI Synthesizing Interdisciplinary Research on Climate Change"
- Lam et al. (2023): "GraphCast: Learning Skillful Medium-Range Global Weather Forecasting", Science

### Evaluácia RAG
- [44] Shahul Es et al. (2024): "RAGAS: Automated Evaluation of Retrieval Augmented Generation", EACL 2024
- [45] Saad-Falcon et al. (2024): "ARES: An Automated Evaluation Framework for Retrieval-Augmented Generation", NAACL 2024
- [41] Manning et al. (2008): "Introduction to Information Retrieval", Cambridge University Press
- [43] Järvelin & Kekäläinen (2002): "Cumulated Gain-Based Evaluation of IR Techniques", ACM TOIS
- Salemi & Zamani (2024): "eRAG: Enhanced Retrieval Augmented Generation", SIGIR 2024

### Ďalšie referencie
- [33] Anthropic (2024): "Contextual Retrieval" — blog post o metadata-as-prefix
- [15] Hoyer & Hamman (2017): "xarray: N-D labeled arrays and datasets in Python", JORS
- [16] Rocklin (2015): "Dask: Parallel Computation with Blocked algorithms and Task Scheduling"
- [49] Ramírez (2021): "FastAPI" — framework documentation
- [51] Merkel (2014): "Docker: Lightweight Linux Containers", Linux Journal
- [52] Hevner et al. (2004): "Design Science in Information Systems Research", MIS Quarterly
- [54] Google DeepMind (2025): "AlphaEarth Foundations", arXiv:2507.22291
- [55] Zhang et al. (2024): "RAFT: Adapting Language Model to Domain Specific RAG"

---

## 3. Implementačné rozhodnutia s odôvodnením

> Pre každé kľúčové rozhodnutie: **Čo**, **Prečo**, **Alternatíva**, **Akademický zdroj**

### R1: Embedding model — BGE-large-en-v1.5 (1024-dim)
- **Čo**: Sentence-BERT architektúra s instruction-augmented queryingom
- **Prečo**: #1 na MTEB retrieval benchmarku, 1024-dim dostatočné pre sémantický priestor
- **Alternatívy**: BGE-M3 (multilingválny, ale 568M parametrov = pomalší), E5-large-v2 (nižší score), MiniLM (384-dim = nedostatočné)
- **Zdroj**: Xiao et al. [21], Muennighoff et al. (MTEB) [MTEB]

### R2: Vektorová databáza — Qdrant s HNSW
- **Čo**: Open-source vektorová DB s HNSW indexom a payload filteringom
- **Prečo**: Najlepší payload filtering (pre spatial/temporal), gRPC transport, on-disk storage
- **Alternatívy**: Pinecone (proprietárny), Milvus (zložitejšia inštalácia), ChromaDB (len prototypy)
- **Zdroj**: Malkov & Yashunin [28]

### R3: Chunking stratégia — 30 timestepov × 10 priestorových buniek
- **Čo**: Fixný chunk = ~1 mesiac × ~2.5° región
- **Prečo**: Granulárne RAG odpovede na mesačnej/regionálnej úrovni; Spatial-RAG potvrdzuje dôležitosť priestorovej granularity
- **Alternatívy**: Denný (príliš veľa chunkú), Ročný (strata detailov), Adaptívny (nekonzistentný)
- **Zdroj**: Spatial-RAG [34], naše merania (140K chunkú pre SPREAD pri tomto nastavení)

### R4: Dvoj-fázový retrieval — bi-encoder + cross-encoder
- **Čo**: Bi-encoder retrieves 40 kandidátov, cross-encoder reranking na top 10
- **Prečo**: Bi-encoder = rýchly recall, cross-encoder = presné rerankovanie; kombinácia dosahuje lepšie výsledky ako samotný bi-encoder
- **Alternatívy**: Len bi-encoder (nižšia presnosť), len cross-encoder (príliš pomalý pre celú kolekciu)
- **Zdroj**: Nogueira & Cho [23], Glass et al. [24]

### R5: Batch dask compute — 500 slices/batch
- **Čo**: Namiesto individuálneho `dask.compute()` pre každý chunk, batch-compute 500 naraz
- **Prečo**: Amortizuje disk I/O overhead; 55x speedup na SPREAD datasete
- **Alternatívy**: Per-chunk compute (0.45s/chunk = bottleneck), celý array pre-load (6GB+ = OOM risk)
- **Zdroj**: Dask dokumentácia, Rocklin [16]

### R6: HNSW disable počas bulk ingestion
- **Čo**: Nastaviť `indexing_threshold=100M` pred bulk loadom, `20K` po
- **Prečo**: HNSW graf sa inkrementálne prestavuje pri každom upserte; vypnutie + jedno finálne zostavenie je 2-5x rýchlejšie
- **Alternatívy**: Nechať inkrementálne (pomalšie), iný index (IVF = menej presný)
- **Zdroj**: Qdrant best practices [31], Malkov & Yashunin [28]

### R7: Async embed+upsert pipeline
- **Čo**: GPU embedding v hlavnom vlákne, Qdrant upsert v background ThreadPoolExecutor
- **Prečo**: GPU idle počas network I/O, network idle počas GPU compute → overlap
- **Alternatívy**: Synchronný (jednoduchší, ale GPU idle ~1s/batch), asyncio (komplikovanejšie)
- **Zdroj**: Vlastná optimalizácia, inšpirovaná producer-consumer pattern

### R8: 5-fázový spracovací model
- **Čo**: Phase 0 (metadata) → 1 (direct) → 2 (auth) → 3 (API) → 4 (manual)
- **Prečo**: Okamžité RAG povedomie (Phase 0) + postupné obohacovanie o reálne dáta
- **Alternatívy**: Všetko naraz (nemožné — rôzna prístupnosť), len metadata (nedostatočné pre presné odpovede)
- **Zdroj**: Vlastný návrh, inšpirovaný ETL best practices [19]

### R9: PostgreSQL pre BatchProgress (nie JSON)
- **Čo**: Stav spracovania v PostgreSQL tabuľke `catalog_progress`
- **Prečo**: Transactional semantics, prežije reštart kontajnera, bulk operácie (1 query namiesto 233)
- **Alternatívy**: JSON súbor (nestabilný pri crash), Redis (overkill), shelve (nie transactional)
- **Zdroj**: Vlastné skúsenosti po strate progresu pri JSON-based riešení

### R10: Spatial-aware retrieval s metadata boosting
- **Čo**: Bounding box filtering (±2° = ~200 km) + keyword score boost (+0.05/match)
- **Prečo**: Čisto sémantický search vracia priestorovo irelevantné výsledky; spatial filtering dramaticky zlepšuje presnosť
- **Alternatívy**: Len sémantický (nepresný pre lokácie), spatial-first (môže vynechať relevantné globálne datasety)
- **Zdroj**: Spatial-RAG [34], Filtered ANN [53]

### R11: Two-stage LLM prompting (Data Selection → Answer)
- **Čo**: LLM sa volá dvakrát — Stage 1 selektuje relevantné premenné/lokácie/časy, Stage 2 generuje odpoveď s cieleným per-variable searchom
- **Prečo**: Priamy RAG (query → search → answer) vracia príliš generické výsledky; LLM-driven data selection dramaticky zvyšuje relevantnosť
- **Alternatívy**: Single-stage RAG (menej presný), hardcoded filtre (neflexibilné), agentic RAG s nástrojmi (príliš zložité)
- **Zdroj**: Inšpirované query decomposition (Shi et al. [32]), agentic RAG patterns (Gao et al. [30])

### R12: Dynamická generácia textu z metadát (nie uloženie v DB)
- **Čo**: `generate_human_readable_text()` dynamicky generuje text z metadát pri každom použití; text sa neukladá v Qdrant
- **Prečo**: DB ostáva čistá a filtrovateľná (len štruktúrované metadáta), text sa generuje v rôznej verbozite podľa kontextu (embedding vs. reranking vs. LLM)
- **Alternatívy**: Uložiť text v payloade (väčší storage, fixná verbozita), generovať len raz (strata flexibility)
- **Zdroj**: Anthropic Contextual Retrieval [33] — metadata-as-prefix prístup

### R13: Trojvrstvová RAG evaluácia s vlastnou Numerical Coverage metrikou
- **Čo**: Tier 1 (retrieval metriky), Tier 2 (RAGAS + Numerical Coverage), Tier 3 (embedding space analysis)
- **Prečo**: Štandardné RAG benchmarky neriešia numerickú presnosť pre vedecké dáta; vlastná metrika overuje, či LLM správne reprodukuje čísla z kontextu
- **Alternatívy**: Len RAGAS (chýba numerická presnosť), len retrieval metriky (neoveruje end-to-end)
- **Zdroj**: RAGAS [44], ARES [45], vlastný návrh pre numerickú presnosť

---

## 4. Evolúcia projektu (git história)

### Fázy vývoja

| Obdobie | Commits | Hlavné zmeny | Kľúčové commity |
|---------|---------|--------------|------------------|
| **Oct 2025** | ~20 | Prvá verzia: základný ETL, Qdrant, RAG | `e37ab66`, `c32eb6e` |
| **Nov 2025** | ~40 | Formátová detekcia, CSV loader, bugfixy | `5daac43` (zip), `fb62422` (format detection) |
| **Dec 2025** | ~50 | Refactoring, Dagster integrácia, Docker | `610ceb3` (cleanup), `23d744c` (Dagster) |
| **Jan 2026** | ~20 | Embedding stratégia zmena (stats→text) | `9397d7e` (README update) |
| **Feb 2026 (1. pol.)** | ~15 | Katalógový systém, 5 fáz | `e375c3e` (batch module), `5b7b469` (chunking opt) |
| **Feb 2026 (2. pol.)** | ~5 | Performance optimalizácia (55x speedup) | Batch dask, async upsert, HNSW |

### Kľúčové architektonické zmeny

**1. Embedding stratégia (V1 → V3.1)**
```
V1 (Oct): 8-dim stats vector → PROBLÉM: nemožno rozlíšiť premenné
V2 (Jan): 1024-dim BGE-M3 text → PROBLÉM: dva vektorové priestory
V3 (Feb): 1024-dim BGE-large + reranking → PROBLÉM: pomalý ingestion
V3.1 (Feb): + batch dask + async upsert + HNSW disable → 55x speedup
```

**2. Batch processing (jednoduchý → produkčný)**
```
V1: Synchronný, per-chunk, bez retry → crash = strata progresu
V2: JSON progress, retry logic → JSON korrupcia pri crash
V3: PostgreSQL progress, bulk ops, memory/disk guards, dedup → produkčný
```

**3. Chunking (100 priestorových → 10)**
```
V1: 100 buniek/chunk (25°) → Rím až Štokholm v jednom chunku
V2: 10 buniek/chunk (2.5°) → regionálna presnosť
Motivácia: Spatial-RAG paper [34]
```

---

## 5. Porovnávacie tabuľky

### Porovnanie embedding stratégií

| Aspekt | V1 (Stats) | V2 (BGE-M3 Text) | V3 (BGE-large Text) |
|--------|------------|-------------------|---------------------|
| Dimenzie | 8 | 1024 | 1024 |
| Sémantika | Žiadna | Multijazyčná | Anglická, lepší retrieval |
| Model veľkosť | 0 (numpy) | 568M | 335M |
| Inference GPU | Nie | ~5s / 2000 textov | ~3.5s / 2000 textov |
| Rozlíšenie premenných | ❌ | ✅ | ✅ |
| Vektorový priestor | Nekompatibilný s Phase 0 | Kompatibilný | Kompatibilný |
| MTEB Retrieval | N/A | 67.19 | 63.55 |

### Porovnanie vektorových databáz

| Feature | Qdrant | Milvus | ChromaDB | Pinecone |
|---------|--------|--------|----------|----------|
| HNSW index | ✅ | ✅ | ✅ | Proprietárny |
| Payload filtering | ✅ Pokročilý | ✅ | Základný | ✅ |
| gRPC | ✅ | ✅ | ❌ | ❌ |
| On-disk storage | ✅ | ✅ | ✅ | Cloud-only |
| Self-hosted | ✅ | ✅ | ✅ | ❌ |
| Bulk ingestion opt. | ✅ (indexing threshold) | ✅ | ❌ | N/A |

### Porovnanie RAG prístupov pre klimatické dáta

| Aspekt | Náš systém | ChatClimate | GeoGPT-RAG | Spatial-RAG |
|--------|------------|-------------|-------------|-------------|
| Vstupné formáty | 6+ (NetCDF, GRIB, ...) | Len text | GIS | Geo-JSON |
| Embedding | BGE-large 1024d | — | Custom GeoEmbedding | Vlastný |
| Reranking | BGE-reranker-v2-m3 | ❌ | GeoReranker | ❌ |
| Spatial filtering | Bounding box | ❌ | ✅ | ✅ Hybridný |
| Streaming | ✅ Generator-based | ❌ | ❌ | ❌ |
| Scalability | 140K+ chunks tested | Malý corpus | — | — |
| Open-source | ✅ | ❌ | ❌ | ❌ |

---

## 6. Výkonnostné merania

### Ingestion pipeline (pred vs. po optimalizácii)

| Metrika | Pred (Feb 18) | Po (Feb 22) | Speedup | Príčina |
|---------|---------------|-------------|---------|---------|
| SPREAD 2000 chunks | ~900s | ~15s | **55x** | Batch dask compute |
| Chunk iteration | ~895s | ~8s | **112x** | 500-slice batch vs per-chunk |
| GPU embedding | ~3.5s | ~3.5s | 1x | Bez zmeny |
| Qdrant upsert | ~1.5s | ~1.0s | 1.5x | HNSW disabled |
| SPREAD total (140K) | ~16+ hodín | ~18 minút | **53x** | Kumulatívny efekt |

### Phase 0 processing

| Metrika | Hodnota |
|---------|---------|
| Entries | 233 |
| Čas | <30s (vrátane embedding) |
| GPU embedding | <200ms pre 233 textov |
| DB operácie | 3 queries (get completed, mark started, mark completed) |

### Qdrant kolekcia

| Metrika | Hodnota |
|---------|---------|
| Bodov (pred SPREAD) | ~42,000 |
| Bodov (po SPREAD) | ~180,000+ |
| Vektory | 1024-dim, COSINE |
| Index | HNSW (m=16) |
| RAM usage | ~2 GB |

---

## 7. Kompletný zoznam referencií

[1] Rolnick, D., Donti, P.L., Kaack, L.H., et al. (2022). Tackling Climate Change with Machine Learning. *ACM Computing Surveys*, 55(2), 1-96.

[2] Nguyen, T., et al. (2023). ClimateBERT: A Pretrained Language Model for Climate-Related Text. *arXiv:2110.12010*.

[3] Thulke, D., et al. (2024). ClimateGPT: Towards AI Synthesizing Interdisciplinary Research on Climate Change. *arXiv:2401.09646*.

[4] Hersbach, H., et al. (2020). The ERA5 global reanalysis. *Quarterly Journal of the Royal Meteorological Society*, 146(730), 1999-2049.

[5] Eyring, V., et al. (2016). Overview of the Coupled Model Intercomparison Project Phase 6 (CMIP6). *Geoscientific Model Development*, 9(5), 1937-1958.

[6] Cornes, R.C., et al. (2018). An Ensemble Version of the E-OBS Temperature and Precipitation Data Sets. *Journal of Geophysical Research: Atmospheres*, 123(17), 9391-9409.

[7] Harris, I., et al. (2020). Version 4 of the CRU TS monthly high-resolution gridded multivariate climate dataset. *Scientific Data*, 7(1), 1-18.

[8] Funk, C., et al. (2015). The climate hazards infrared precipitation with stations—a new environmental record for monitoring extremes. *Scientific Data*, 2(1), 1-21.

[9] Fick, S.E. & Hijmans, R.J. (2017). WorldClim 2: new 1-km spatial resolution climate surfaces for global land areas. *International Journal of Climatology*, 37(12), 4302-4315.

[10] GISTEMP Team (2024). GISS Surface Temperature Analysis (GISTEMP), version 4. NASA Goddard Institute for Space Studies.

[11] Serrano-Notivoli, R., et al. (2017). SPREAD: a high-resolution daily gridded precipitation dataset for Spain. *Earth System Science Data*, 9(2), 721-738.

[12] Rew, R. & Davis, G. (1990). NetCDF: an interface for scientific data access. *IEEE Computer Graphics and Applications*, 10(4), 76-82.

[13] WMO (2003). Guide to the WMO Table Driven Code Form Used for the Representation and Exchange of Regularly Spaced Data in Binary Form: FM 92 GRIB Edition 2.

[14] CF Conventions Committee (2023). CF Conventions and Metadata. *cfconventions.org*.

[15] Hoyer, S. & Hamman, J. (2017). xarray: N-D labeled arrays and datasets in Python. *Journal of Open Research Software*, 5(1).

[16] Rocklin, M. (2015). Dask: Parallel Computation with Blocked algorithms and Task Scheduling. *Proceedings of the 14th Python in Science Conference*.

[17] Gillies, S., et al. (2013). Rasterio: geospatial raster I/O for Python programmers. *GitHub repository*.

[18] McKinney, W. (2010). Data Structures for Statistical Computing in Python. *Proceedings of the 9th Python in Science Conference*.

[19] Vassiliadis, P., et al. (2002). A Survey on Extract-Transform-Load Technology. *International Journal of Data Warehousing and Mining*, 5(3).

[20] Dagster Labs (2024). Dagster: The Data Orchestrator. *dagster.io*.

[21] Xiao, S., et al. (2024). C-Pack: Packaged Resources to Advance General Chinese Embedding. *SIGIR 2024*. arXiv:2309.07597.

[22] Reimers, N. & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. *EMNLP 2019*.

[23] Nogueira, R. & Cho, K. (2019). Passage Re-ranking with BERT. *arXiv:1901.04085*.

[24] Glass, M., et al. (2022). Re2G: Retrieve, Rerank, Generate. *NAACL 2022*.

[25] Chen, J., et al. (2024). BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation. *arXiv:2402.03216*.

[26] Wang, L., et al. (2024). Text Embeddings by Weakly-Supervised Contrastive Pre-training. *ACL 2024*.

[27] Wang, L., et al. (2024). Multilingual E5 Text Embeddings: A Technical Report. *arXiv:2402.05672*.

[28] Malkov, Y.A. & Yashunin, D.A. (2020). Efficient and Robust Approximate Nearest Neighbor using Hierarchical Navigable Small World Graphs. *IEEE TPAMI*, 42(4), 824-836.

[29] Lewis, P., et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. *NeurIPS 2020*.

[30] Gao, Y., et al. (2024). Retrieval-Augmented Generation for Large Language Models: A Survey. *arXiv:2312.10997*.

[31] Qdrant Documentation (2024). Bulk Upload Optimization. *qdrant.tech/documentation*.

[32] Shi, W., et al. (2024). REPLUG: Retrieval-Augmented Black-Box Language Models. *NAACL 2024*.

[33] Anthropic (2024). Contextual Retrieval. *anthropic.com/news/contextual-retrieval*.

[34] Yu, Z., et al. (2025). Spatial-RAG: Spatial Retrieval Augmented Generation. *arXiv:2502.18470*.

[35] Yu, R., et al. (2025). RAG for Geoscience: What We Expect, Gaps and Opportunities. *arXiv:2508.11246*.

[36] Huang, L., et al. (2025). GeoGPT-RAG: A Retrieval-Augmented Generation Framework for Geoscience. *arXiv:2509.09686*.

[37] Vaghefi, S.A., et al. (2023). ChatClimate: Grounding Conversational AI in Climate Science. *arXiv:2304.05510*.

[38] Adamu, H., et al. (2025). ClimatePub4KG: A Climate Publications Knowledge Graph. *arXiv:2509.10087*.

[39] Kong, L., Joshi, A. & Karimi, S. (2025). CAIRNS: Balancing Readability and Scientific Accuracy in Climate Adaptation Question Answering. *arXiv:2512.02251*.

[40] Nguyen, T., et al. (2024). Towards Responsible RAG for Climate Decision-Making. *arXiv:2410.23902*.

[41] Manning, C.D., Raghavan, P. & Schütze, H. (2008). *Introduction to Information Retrieval*. Cambridge University Press.

[42] Voorhees, E.M. (1999). The TREC-8 Question Answering Track Report. *TREC 1999*.

[43] Järvelin, K. & Kekäläinen, J. (2002). Cumulated Gain-Based Evaluation of IR Techniques. *ACM Transactions on Information Systems*, 20(4), 422-446.

[44] Shahul Es, et al. (2024). RAGAS: Automated Evaluation of Retrieval Augmented Generation. *EACL 2024*. arXiv:2309.15217.

[45] Saad-Falcon, J., et al. (2024). ARES: An Automated Evaluation Framework for RAG Systems. *NAACL 2024*. arXiv:2311.09476.

[46] TruEra (2024). TruLens: Evaluation and Tracking for LLM Experiments. *trulens.org*.

[47] Confident AI (2024). DeepEval: The Open-Source LLM Evaluation Framework. *deepeval.com*.

[48] Bouchacourt, D., et al. (2024). ARAGOG: Advanced RAG Output Grading. *arXiv:2404.01037*.

[49] Ramírez, S. (2021). FastAPI: Modern, Fast (High-Performance) Web Framework for Building APIs. *fastapi.tiangolo.com*.

[50] You, E. (2014). Vue.js: The Progressive JavaScript Framework. *vuejs.org*.

[51] Merkel, D. (2014). Docker: Lightweight Linux Containers for Consistent Development and Deployment. *Linux Journal*, 2014(239).

[52] Hevner, A.R., et al. (2004). Design Science in Information Systems Research. *MIS Quarterly*, 28(1), 75-105.

[53] Amanbayev, A., Tsan, B., Dang, T. & Rusu, F. (2025). Filtered Approximate Nearest Neighbor Search in Vector Databases: System Design and Performance Analysis. *arXiv:2602.11443*.

[54] Google DeepMind (2025). AlphaEarth Foundations: Multi-Modal AI for Earth Observation. *arXiv:2507.22291*.

[55] Zhang, T., et al. (2024). RAFT: Adapting Language Model to Domain Specific RAG. *arXiv:2403.10131*.

---

## Poznámky pre autora

### Zoznam toho, o čom musíš napísať (mapovanie na kapitoly):

**Kap. 1 (Úvod)**: Motivácia (klimatické dáta sú fragmentované), definícia problému (RAG nad heterogénnymi dátami), ciele (5-fázový ETL + spatial-aware RAG)

**Kap. 2 (Literárna rešerš)**: VŠETKY referencie [1]-[55] treba spracovať. Kľúčové: Lewis [29] (RAG), Xiao [21] (BGE), Malkov [28] (HNSW), Es [44] (RAGAS), Yu [34] (Spatial-RAG)

**Kap. 3 (Ciele a metodika)**: Design Science Research [52], technologický stack tabuľka, vývojové prostredie (Docker, RTX 5090)

**Kap. 4 (Návrh)**: Architektúra diagram, 5-fázový model, dátová vrstva (PostgreSQL + Qdrant), ETL pipeline návrh, RAG pipeline návrh

**Kap. 5 (Implementácia)**: Štruktúra projektu, každý modul detailne s kódom, chunking stratégia s odôvodnením, embedding pipeline, batch optimalizácie

**Kap. 6 (Testovanie)**: Golden test set, retrieval metriky (Hit@K, MRR, NDCG), RAGAS metriky, výkonnostné testy (55x speedup tabuľky)

**Kap. 7 (Diskusia)**: Porovnanie s ChatClimate/GeoGPT-RAG/Spatial-RAG, evolúcia embedding stratégie (V1→V3.1), limitácie, budúca práca

**Kap. 8 (Záver)**: Sumarizácia dosiahnutých cieľov, prínos práce
