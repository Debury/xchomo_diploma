# Testing Guide - Climate Embeddings Project

Komplexný návod na testovanie všetkých funkcionalít na externom serveri.

## 🚀 Rýchly štart

### 1. Spustenie služieb

```bash
# Spustiť všetky Docker služby
make docker-compose-up

# Alebo manuálne
docker-compose up -d

# Skontrolovať stav
docker-compose ps
```

### 2. Verifikácia služieb

```bash
# Skontrolovať zdravie všetkých služieb
make verify-services

# Alebo jednotlivo
make check-qdrant    # http://localhost:6333
make check-ollama    # http://localhost:11434
make api-health      # http://localhost:8000
```

## 📋 Testovanie formátov dát

### Automatický test všetkých formátov

```bash
# Spustí kompletný test suite pre všetky formáty
make test-formats
```

Testuje:
- ✅ **NetCDF** (.nc, .nc4) - klimatické modely
- ✅ **GeoTIFF** (.tif, .tiff) - rastrové mapy
- ✅ **CSV** (.csv) - tabulkové dáta
- ✅ **GRIB** (.grib, .grb2) - meteorologické dáta
- ✅ **HDF5** (.h5, .hdf5) - vedecké dáta
- ✅ **ASCII Grid** (.asc) - grid dáta
- ✅ **Zarr** (.zarr) - chunked arrays
- ✅ **ZIP** (.zip) - archívy s viacerými súbormi

### Manuálne testovanie jednotlivých formátov

#### NetCDF
```bash
docker-compose exec web-api python << 'EOF'
from climate_embeddings.loaders import load_raster_auto

result = load_raster_auto("data/external/era5_temperature.nc")
print(f"✓ Loaded {len(result.embeddings)} embeddings")
print(f"  Shape: {result.embeddings.shape}")
print(f"  Variables: {result.metadata.get('variables', [])}")
EOF
```

#### GeoTIFF
```bash
docker-compose exec web-api python << 'EOF'
from climate_embeddings.loaders import load_raster_auto

result = load_raster_auto("data/external/temperature_map.tif")
print(f"✓ Loaded {len(result.embeddings)} embeddings")
print(f"  Bounds: {result.metadata.get('bounds')}")
EOF
```

#### CSV
```bash
docker-compose exec web-api python << 'EOF'
from climate_embeddings.loaders import load_raster_auto

result = load_raster_auto("data/external/station_data.csv")
print(f"✓ Loaded {len(result.embeddings)} embeddings")
EOF
```

#### ZIP archív
```bash
docker-compose exec web-api python << 'EOF'
from climate_embeddings.loaders.raster_pipeline import load_from_zip

results = load_from_zip("data/external/climate_bundle.zip")
print(f"✓ Loaded {len(results)} files from ZIP")
for r in results:
    print(f"  - {r.source_file}: {len(r.embeddings)} embeddings")
EOF
```

## 🧪 Unit testy

### Všetky testy
```bash
make test
```

### Špecifické test suites
```bash
make test-raster       # Raster loading (NetCDF, GeoTIFF, CSV)
make test-rag          # RAG komponenty (embeddings, index, pipeline)
make test-embeddings   # Qdrant integrácia
make test-dagster      # Dagster jobs
make test-api          # Web API endpoints
```

### Test coverage
```bash
make test-coverage     # Generuje HTML report do htmlcov/
```

## 🔍 Testovanie embeddings

### Text embeddings

```bash
# BGE model (1024-dim)
docker-compose exec web-api python << 'EOF'
from climate_embeddings.embeddings import get_text_embedder

embedder = get_text_embedder("bge-large")
embedding = embedder.encode("temperature data from ERA5")
print(f"✓ BGE embedding shape: {embedding.shape}")  # (1024,)
EOF

# GTE model (1024-dim)
docker-compose exec web-api python << 'EOF'
from climate_embeddings.embeddings import get_text_embedder

embedder = get_text_embedder("gte-large")
embedding = embedder.encode("precipitation trends")
print(f"✓ GTE embedding shape: {embedding.shape}")  # (1024,)
EOF

# MiniLM model (384-dim, rýchly)
docker-compose exec web-api python << 'EOF'
from climate_embeddings.embeddings import get_text_embedder

embedder = get_text_embedder("minilm")
embedding = embedder.encode("wind speed data")
print(f"✓ MiniLM embedding shape: {embedding.shape}")  # (384,)
EOF
```

### Vector index

```bash
docker-compose exec web-api python << 'EOF'
import numpy as np
from climate_embeddings.index import VectorIndex

# Vytvoriť index
index = VectorIndex(dimension=1024, metric="cosine")

# Pridať vektory
vectors = np.random.randn(100, 1024).astype(np.float32)
metadata = [{"id": i, "type": "test"} for i in range(100)]
index.add_batch(vectors, metadata)

# Vyhľadávanie
query = np.random.randn(1024).astype(np.float32)
results = index.search(query, k=5)

print(f"✓ Added {len(vectors)} vectors")
print(f"✓ Found {len(results)} nearest neighbors")
for r in results:
    print(f"  - Score: {r.score:.4f}, Metadata: {r.metadata}")
EOF
```

## 🤖 RAG Pipeline Testing

### Kompletný RAG workflow

```bash
docker-compose exec web-api python << 'EOF'
import numpy as np
from climate_embeddings.rag import RAGPipeline
from climate_embeddings.index import VectorIndex

# 1. Vytvoriť index s dátami
index = VectorIndex(dimension=1024, metric="cosine")

# Pridať nejaké vektory s metadátami
vectors = np.random.randn(10, 1024).astype(np.float32)
metadata = [
    {"text": "Temperature in Europe increased by 1.5°C since 2000"},
    {"text": "Precipitation patterns changed in Mediterranean region"},
    {"text": "Arctic sea ice extent decreased by 40%"},
    {"text": "CO2 concentrations reached 420 ppm in 2024"},
    {"text": "Heat waves became more frequent in summer"},
    {"text": "Drought conditions persisted in Central Europe"},
    {"text": "Sea level rose by 3mm per year globally"},
    {"text": "Extreme weather events increased in frequency"},
    {"text": "Glaciers in Alps retreated significantly"},
    {"text": "Ocean temperatures reached record highs"}
]
index.add_batch(vectors, metadata)

# 2. Vytvoriť RAG pipeline
rag = RAGPipeline(
    index=index,
    embedder_name="bge-large",
    llm_model="llama3.2:1b",
    llm_base_url="http://ollama:11434"
)

# 3. Retrieve relevantné dokumenty
results = rag.retrieve("What are the temperature trends?", k=3)
print("✓ Retrieved documents:")
for r in results:
    print(f"  - {r.metadata['text']} (score: {r.score:.4f})")

# 4. RAG query s LLM generovaním
print("\n✓ Asking RAG system...")
answer = rag.ask("What are the main climate changes observed?", k=5)
print(f"Answer: {answer}")
EOF
```

## 💾 Qdrant Integration Testing

### Basic Qdrant operations

```bash
# Vytvoriť kolekciu a uložiť embeddings
docker-compose exec web-api python << 'EOF'
import numpy as np
from src.embeddings.database import VectorDatabase
from src.embeddings.generator import EmbeddingGenerator

# Generovať text embeddings
generator = EmbeddingGenerator()
texts = [
    "Temperature data from ERA5 reanalysis",
    "Precipitation measurements from weather stations",
    "Wind speed data from climate models",
    "Sea level pressure observations"
]
embeddings = generator.generate_embeddings(texts)

# Uložiť do Qdrant
db = VectorDatabase(collection_name="climate_data")
metadata = [{"text": t, "type": "climate", "idx": i} for i, t in enumerate(texts)]
db.add_embeddings(embeddings, metadata)

print(f"✓ Stored {len(embeddings)} embeddings in Qdrant")
print(f"  Collection: {db.collection_name}")
print(f"  Dimension: {embeddings.shape[1]}")
EOF

# Semantic search
docker-compose exec web-api python << 'EOF'
from src.embeddings.search import SemanticSearcher
from src.embeddings.database import VectorDatabase

db = VectorDatabase(collection_name="climate_data")
searcher = SemanticSearcher(database=db)

results = searcher.search("temperature and climate", k=3)
print("✓ Search results:")
for r in results:
    print(f"  - {r['metadata']['text']}")
    print(f"    Similarity: {r['similarity']:.4f}")
EOF
```

## 🌐 API Endpoint Testing

### List sources
```bash
curl http://localhost:8000/sources | jq
```

### List jobs
```bash
curl http://localhost:8000/jobs | jq
```

### Trigger ETL job
```bash
make trigger-etl

# Alebo manuálne
curl -X POST http://localhost:8000/jobs/dynamic_source_etl_job/run \
  -H "Content-Type: application/json" \
  -d '{}'
```

### RAG query endpoint
```bash
curl -X POST http://localhost:8000/rag/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the temperature trends in Europe?",
    "top_k": 5
  }' | jq
```

## ⚙️ Dagster Testing

### Spustenie Dagster UI

```bash
make dagit
# Otvor http://localhost:3000
```

### Verifikácia Dagster workspace

```bash
docker-compose exec dagit python << 'EOF'
from dagster import DagsterInstance
from dagster_project.repository import climate_repository

repo = climate_repository()
print(f"✓ Repository: {repo.name}")
print("\nAvailable jobs:")
for job in repo.get_all_jobs():
    print(f"  - {job.name}")
    
print("\nSchedules:")
for schedule in repo.get_all_schedules():
    print(f"  - {schedule.name}: {schedule.cron_schedule}")
EOF
```

### Spustenie job manuálne

```bash
# Cez API
make trigger-etl
make trigger-embeddings

# Cez CLI (v containeri)
docker-compose exec dagster-daemon dagster job execute \
  -m dagster_project.repository \
  -j dynamic_source_etl_job
```

## 📊 Monitoring & Logs

### Docker logs
```bash
make dagster-logs           # Dagster služby
docker-compose logs -f web-api
docker-compose logs -f qdrant
docker-compose logs -f ollama
```

### Dagster compute logs
```bash
# V Dagster UI: Runs → vyberte run → Compute Logs
# Alebo na serveri:
ls -la .dagster_home/storage/
```

### API logs
```bash
docker-compose logs -f web-api | grep -i error
```

## 🔧 Troubleshooting

### Services not starting
```bash
# Reštartovať služby
docker-compose restart

# Rebuildiť image
make docker-build
docker-compose up -d --build
```

### Import errors
```bash
# Skontrolovať Python path
docker-compose exec web-api python -c "import sys; print('\\n'.join(sys.path))"

# Skontrolovať inštalované balíčky
docker-compose exec web-api pip list | grep -i climate
```

### Qdrant connection issues
```bash
# Skontrolovať Qdrant health
curl http://localhost:6333/health

# Zoznam kolekcií
curl http://localhost:6333/collections | jq
```

### Ollama model issues
```bash
# Skontrolovať dostupné modely
curl http://localhost:11434/api/tags | jq

# Stiahnuť model
docker-compose exec ollama ollama pull llama3.2:1b
```

## ✅ Checklist pre deployment testing

- [ ] Docker služby bežia (`docker-compose ps`)
- [ ] Qdrant zdravý (`make check-qdrant`)
- [ ] Ollama zdravý (`make check-ollama`)
- [ ] API zdravé (`make api-health`)
- [ ] Importy fungujú (`make test-formats` kroky 1-2)
- [ ] NetCDF loading funguje
- [ ] GeoTIFF loading funguje
- [ ] CSV loading funguje
- [ ] ZIP loading funguje
- [ ] Text embeddings fungujú (BGE, GTE)
- [ ] Vector index funguje
- [ ] RAG pipeline funguje
- [ ] Qdrant ukladanie funguje
- [ ] Semantic search funguje
- [ ] API endpoints fungujú
- [ ] Dagster jobs sa dajú spustiť
- [ ] Unit testy prechádzajú (`make test`)

## 📝 Quick Commands Reference

```bash
# Setup
make docker-compose-up      # Spustí všetky služby
make verify-services        # Overí zdravie služieb

# Testing
make test-formats          # Test všetkých formátov
make test-raster           # Test raster loading
make test-rag              # Test RAG komponenty
make test-embeddings       # Test Qdrant
make test-all              # Všetky testy

# Services
make dagit                 # Dagster UI (port 3000)
make api                   # API service (port 8000)
make dagster-logs          # Zobraziť logy

# Triggers
make trigger-etl           # Spustiť ETL job
make trigger-embeddings    # Spustiť embedding job

# Checks
make check-qdrant          # Qdrant status
make check-ollama          # Ollama status
make api-health            # API health
make list-sources          # Zoznam zdrojov

# Cleanup
make docker-compose-down   # Zastaviť služby
make clean                 # Vyčistiť cache
```

## 🎯 Production Deployment Checklist

1. **Environment variables** - skontrolovať `.env`
2. **Data volumes** - namapovať `/data` persistent storage
3. **Qdrant persistence** - volume pre `/qdrant/storage`
4. **Ollama models** - predstiahnuť potrebné modely
5. **Resource limits** - nastaviť v docker-compose.yml
6. **Logging** - konfigurovať log aggregation
7. **Monitoring** - Prometheus/Grafana pre metrics
8. **Backups** - automatický backup Qdrant collections

---

**Poznámka:** Všetky testy predpokladajú že služby bežia cez `docker-compose up -d`.
