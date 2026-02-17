# Poznámky k prezentaci — schůzka s vedoucím

*Připraveno: 17. února 2026*

---

## 1. Co bylo uděláno

### Kompletní ETL pipeline
- **Data acquisition → Transformation → Embedding → Vector DB → RAG**
- Plně funkční end-to-end pipeline od surových klimatických dat po odpovědi v přirozeném jazyce

### Katalogové zpracování (D1.1.xlsx)
- 233 záznamů z Excel katalogu, 69 unikátních datasetů
- Klasifikace do 5 fází podle typu přístupu k datům
- **Fáze 0:** Metadata-only embedding — všech 233 záznamů okamžitě dostupných pro RAG (bez stahování)
- **Fáze 1:** Přímé HTTP stahování + rastrové zpracování — CMIP6, CRU, E-OBS datasety úspěšně zpracovány

### Streaming batch zpracování
- Crash recovery — automatický retry po selhání
- Memory guard — ochrana proti OOM při zpracování velkých souborů
- Timeout ochrana — prevence zamrznutí na pomalých zdrojích
- Persistentní logování — `logs/catalog_pipeline.log`

### Výkonnostní optimalizace
- Batch GPU embedding: **10-20x zrychlení** oproti sekvenčnímu zpracování
- Numpy pre-load pro efektivní práci s rastrovými daty
- Generator-based streaming — nikdy nenačítá celý soubor do RAM

### Embedding model
- **BAAI/bge-m3** (1024-dim, multilingvální, SOTA na MTEB)
- Cosine similarity search v Qdrant vektorové databázi

### Web rozhraní a API
- **FastAPI REST API** — CRUD, RAG dotazy, katalogové operace, systémové info
- **Vue.js admin dashboard** — Katalog browser, ETL monitor, plánování, nastavení
- **Dagster ETL orchestrace** — Dagit UI na portu 3000

### Deployment
- **Docker Compose** — 7 služeb (FastAPI, Qdrant, Dagster, PostgreSQL, Caddy, Dagit, Jupyter)
- **Caddy** reverse proxy s automatickým TLS
- Nasazeno na **climaterag.online** (Digital Ocean)

---

## 2. Git historie

| Období | Aktivita |
|--------|----------|
| Prosinec 2025 | Počáteční vývoj, multi-agent architektura, Docker setup |
| Leden 2026 | Rastrový pipeline, embedding integrace, základní RAG |
| 17. února 2026 | Hlavní push — katalogové batch zpracování, Phase 1 pipeline, crash recovery, výkonnostní optimalizace |

- **185 commitů** celkem na hlavní větvi
- Klíčové větve: `main`, `pipeline`, `pipeline_v2`, `feature/climate-rag-enhancements`

---

## 3. Proč tento přístup

### RAG nad klimatickými daty
- Umožňuje dotazy v přirozeném jazyce nad heterogenními datasety
- Uživatel nemusí znát formát dat, proměnné ani strukturu — stačí se zeptat
- Odpovědi jsou podložené konkrétními daty (ne hallucinations)

### Chunking strategie (time=1, lat=100, lon=100)
- Zachovává prostorovo-temporální přesnost dat
- Každý chunk nese konkrétní hodnoty pro konkrétní místo a čas
- Umožňuje přesné odpovědi na dotazy typu "jaká byla teplota v Praze v červenci 2023"

### BGE-M3 model
- **Multilingvální** — podpora češtiny, angličtiny, slovenštiny
- **SOTA retrieval performance** na MTEB benchmarku
- 1024-dim vektory — dobrý kompromis kvalita/rychlost

### Streaming zpracování
- Velké NetCDF soubory (E-OBS: 270MB, 33K chunků) by způsobily OOM
- Generator-based přístup — konstantní paměťová náročnost
- Batch embedding — efektivní využití GPU

### 5-fázový přístup
- Realita klimatických dat: smíšené přístupové metody
- Phase 0 poskytuje okamžitou hodnotu (RAG awareness z metadat)
- Postupné zpracování = postupná validace

---

## 4. Co zbývá

### Bezprostřední TODO
- **Fáze 2-4:** Datasety vyžadující registraci (Phase 2), API portály CDS/ESGF (Phase 3), manuální zdroje (Phase 4)
- **RAG evaluační suite:** RAGAS metriky, golden test set, embedding space analýza
- **LLM kvalita:** Tuning promptů, numerická přesnost odpovědí

### Diplomová práce
- Metodologická kapitola — popis pipeline, chunking, embedding
- Evaluační výsledky — RAGAS skóre, retrieval metriky, porovnání s baseline
- Architektonické diagramy — system overview, data flow, deployment

### Produkční hardening
- Monitoring a alerting
- Automatické zálohy Qdrant kolekce
- CI/CD pipeline

---

## 5. Technické detaily pro diskuzi

### Výkonnostní tabulka

| Dataset | Čas zpracování | Chunků | Velikost souboru |
|---------|----------------|--------|-----------------|
| CMIP6 (tas/pr/psl) | 95s | 3 124 | ~50MB |
| CRU TS | 2.3 min | 5 718 | ~120MB |
| E-OBS (tg/tn/tx/rr) | ~32s/batch | ~33 000 | ~270MB |
| Metadata (Phase 0) | <30s | 233 | — |

**Celkem: ~42 000+ chunků** ve vektorové databázi.

### Architektura

```
Internet → Caddy (TLS) → FastAPI (8000)
                              ↓
                         Qdrant (6333) ← Dagster ETL ← klimatická data (NetCDF, GeoTIFF)
                              ↓
                    LLM (Ollama/Groq/OpenRouter)
                              ↓
                     Vue.js Admin Dashboard
```

### Porovnání embedding modelů

| Model | Dimenze | Multilingvální | MTEB Score | Poznámka |
|-------|---------|----------------|------------|----------|
| **BAAI/bge-m3** | 1024 | Ano (100+ jazyků) | SOTA | Zvolený model |
| bge-large-en-v1.5 | 1024 | Ne (EN only) | Vysoký | Předchozí verze |
| e5-large-v2 | 1024 | Ne | Vysoký | Alternativa |
| multilingual-e5-large | 1024 | Ano | Střední | Slabší retrieval |

### Klíčové metriky pro evaluaci (plánované)

| Metrika | Typ | Cíl | Popis |
|---------|-----|-----|-------|
| Hit@5 | Retrieval | ≥ 0.80 | Alespoň 1 relevantní chunk v top-5 |
| MRR@10 | Retrieval | ≥ 0.60 | Průměrná reciproká pozice prvního relevantního |
| NDCG@10 | Retrieval | ≥ 0.50 | Kvalita rankingu |
| Faithfulness | RAGAS | ≥ 0.85 | Odpovědi podložené kontextem |
| Context Recall | RAGAS | ≥ 0.75 | Pokrytí relevantních chunků |
| Numerical Coverage | Custom | ≥ 0.90 | Zachování číselných hodnot |

---

## 6. Otázky pro vedoucího

- Preference pro formát evaluačních výsledků v diplomové práci?
- Doporučené baseline pro porovnání (BM25, TF-IDF, jiný embedding model)?
- Rozsah Phase 2-4 zpracování pro účely diplomové práce — všech 233 záznamů nebo vybraná podmnožina?
- Časový plán pro odevzdání — deadline pro kompletní evaluaci?
