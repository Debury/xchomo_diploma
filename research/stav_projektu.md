# Stav projektu — Climate Data RAG Pipeline

*Aktualizováno: 17. února 2026*

---

## 1. Přehled projektu

Diplomová práce implementuje RAG (Retrieval-Augmented Generation) systém nad klimatickými daty. Systém umožňuje uživatelům klást dotazy v přirozeném jazyce nad heterogenními klimatickými datasety a získávat přesné, podložené odpovědi s odkazem na zdrojová data.

### Architektura

```
Caddy (80/443, auto-TLS)
  → FastAPI (8000) — REST API + Vue.js frontend
       → Qdrant (6333) — vektorová databáze (cosine similarity)
       → Ollama / Groq / OpenRouter — LLM pro generování odpovědí
  → Dagit (3000) — Dagster ETL orchestrace
       → PostgreSQL (5432) — Dagster metadata
```

**Nasazení:** Docker Compose (7 služeb), doména `climaterag.online`, Digital Ocean droplet.

---

## 2. Co bylo implementováno

### 2.1 ETL Pipeline (zpracování dat)

- **Katalogové zpracování:** Zpracování D1.1.xlsx Excel katalogu (233 záznamů, 69 unikátních datasetů) v 5 fázích
- **Fáze 0 — Metadata-only embedding:** Všech 233 záznamů okamžitě dostupných pro RAG (bez stahování dat)
- **Fáze 1 — Přímé HTTP stahování + rastrové zpracování:**
  - CMIP6 (teplota, srážky, tlak): 95s zpracování, 3 124 chunků
  - CRU TS (globální pozorovaná data): 2.3 min, 5 718 chunků
  - E-OBS (evropská pozorování): ~32s/500-chunk dávka, celkem ~33 000 chunků
- **Streaming batch zpracování:** Crash recovery, retry logika, memory guard, timeout ochrana
- **Chunking strategie:** Prostorovo-temporální granularita (time=1, lat=100, lon=100) pro zachování přesnosti dat

### 2.2 Embedding a vektorová databáze

- **Model:** BAAI/bge-m3 (1024-dim, multilingvální, SOTA na MTEB benchmarku)
- **Optimalizace:** Batch GPU embedding (10-20x zrychlení oproti sekvenčnímu), numpy pre-load
- **Qdrant:** Cosine similarity search, payload indexy pro filtrování

### 2.3 RAG Pipeline

- FastAPI endpoint `POST /rag/query` — vektorové vyhledávání + LLM generování
- Graceful LLM fallback — systém vrátí surové výsledky pokud není LLM dostupný
- Podpora více LLM providerů: Ollama (lokální), Groq, OpenRouter

### 2.4 Web rozhraní

- **Vue.js admin dashboard** s Tailwind CSS (dark theme)
  - Dashboard — přehled zdrojů, Qdrant statistiky, poslední aktivita
  - Katalog — prohlížení Excel katalogu, spouštění zpracování, stavové badges
  - ETL Monitor — real-time průběh, log viewer
  - Plánování — správa Dagster schedules
  - Nastavení — systémové informace, konfigurace

### 2.5 Infrastruktura

- Docker Compose s 7 službami
- Caddy reverse proxy s automatickým TLS
- Dagster ETL orchestrace
- Persistentní logování (`logs/catalog_pipeline.log`)

---

## 3. Výkonnostní výsledky

| Dataset | Čas zpracování | Počet chunků | Velikost |
|---------|----------------|--------------|----------|
| CMIP6 (tas/pr/psl) | 95s | 3 124 | ~50MB |
| CRU TS | 2.3 min | 5 718 | ~120MB |
| E-OBS (tg/tn/tx/rr) | ~32s/batch | ~33 000 | ~270MB |
| Metadata (Phase 0) | <30s | 233 | — |

**Celkem:** ~42 000+ chunků v Qdrant vektorové databázi.

---

## 4. Technická rozhodnutí a zdůvodnění

### Proč BGE-M3?
- SOTA výkon na MTEB benchmarku (retrieval, classification, clustering)
- Multilingvální podpora (čeština, angličtina, slovenština) — důležité pro dotazy v češtině nad anglickými daty
- 1024-dim vektory — dobrý kompromis mezi kvalitou a rychlostí

### Proč streaming zpracování?
- Velké NetCDF soubory (E-OBS: 270MB) by způsobily OOM při načtení celého souboru
- Generator-based chunking — nikdy nenačítá celý soubor do RAM
- Batch embedding — 10-20x rychlejší než sekvenční zpracování

### Proč 5-fázový přístup?
- Klimatická data mají heterogenní přístupové metody (volné HTTP, registrace, API portály, manuální)
- Phase 0 poskytuje okamžitou RAG awareness ze samotných metadat
- Postupné zpracování umožňuje průběžné testování a validaci

### Proč Qdrant?
- Nativní podpora cosine similarity s HNSW indexem
- Payload filtry pro metadata-based dotazy
- Dobrá integrace s Python ekosystémem
- Horizontální škálovatelnost (sharding) pro budoucí růst

---

## 5. Co zbývá (TODO)

### Vysoká priorita
- [ ] **Fáze 2-4 zpracování:** Datasety vyžadující registraci, API portály (CDS, ESGF), manuální zdroje
- [ ] **RAG evaluační suite:** RAGAS metriky, golden test set, embedding space analýza
- [ ] **Kvalita LLM odpovědí:** Fine-tuning promptů, chain-of-thought, numerická přesnost

### Střední priorita
- [ ] **Diplomová práce:** Metodologická kapitola, evaluační výsledky, architektonické diagramy
- [ ] **Rozšířené chunking strategie:** Adaptivní velikost chunků dle typu dat
- [ ] **Query rozšíření:** HyDE (Hypothetical Document Embeddings), query decomposition

### Nízká priorita
- [ ] **Produkční hardening:** Monitoring, alerting, automatické zálohy
- [ ] **CI/CD pipeline:** Automatické testy, deployment
- [ ] **Uživatelské rozhraní:** Vylepšení RAG dotazovacího rozhraní, vizualizace výsledků

---

## 6. Git historie

- **185 commitů** na hlavní větvi
- **Prosinec 2025:** Počáteční vývoj, multi-agent architektura, Docker setup
- **Únor 2026:** Hlavní push — katalogové batch zpracování, Phase 1 pipeline, crash recovery, výkonnostní optimalizace
- **Klíčové větve:** main, pipeline, pipeline_v2, feature/climate-rag-enhancements
