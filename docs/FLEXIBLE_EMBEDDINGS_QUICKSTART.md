# Flexible Embeddings - Quick Start Guide

## Čo je nové?

Embedding systém bol kompletne prepracovaný:
- ❌ **ODSTRÁNENÉ:** Hardcoded šablóny a premenné
- ✅ **PRIDANÉ:** Automatická detekcia akýchkoľvek dát
- ✅ **PRIDANÉ:** Sample values pre RAG
- ✅ **PRIDANÉ:** Tri úrovne verbosity

## Rýchly štart

### 1. Spracovanie datasetu

```python
from src.embeddings import EmbeddingPipeline

# Inicializácia pipeline
pipeline = EmbeddingPipeline()

# Spracovanie AKÉHOKOĽVEK datasetu
result = pipeline.process_dataset("data/processed/your_data.nc")

print(f"Vytvorených embeddings: {result['num_embeddings']}")
```

### 2. Vyhľadávanie (RAG-ready)

```python
from src.embeddings import SemanticSearcher

# Inicializácia vyhľadávača
searcher = SemanticSearcher()

# Vyhľadávanie v prirodzenom jazyku
results = searcher.search(
    "teplota okolo 15 stupňov",
    k=5
)

# Výsledky obsahujú:
for result in results:
    print(result['metadata']['variable'])      # Názov premennej
    print(result['metadata']['stat_mean'])     # Priemer
    print(result['metadata']['sample_values']) # Sample hodnoty!
    print(result['document'])                  # Plný text
```

### 3. Vlastná extrakcia metadata

```python
from src.embeddings import MetadataExtractor
import xarray as xr

# Načítanie datasetu
ds = xr.open_dataset("data.nc")

# Extrakcia metadata
extractor = MetadataExtractor()
metadata_list = extractor.extract_from_dataset(
    data=ds,
    file_path="data.nc",
    dataset_id="my_dataset"
)

# Metadata obsahuje VŠETKO:
meta = metadata_list[0]
print(meta['variable'])           # Názov premennej
print(meta['stat_mean'])          # Štatistiky
print(meta['sample_values'])      # Sample hodnoty
print(meta['spatial_extent'])     # Priestorové info
print(meta['temporal_extent'])    # Časové info
```

### 4. Generovanie textu

```python
from src.embeddings import TextGenerator

# Rôzne úrovne verbosity
for verbosity in ['low', 'medium', 'high']:
    text_gen = TextGenerator(config={'verbosity': verbosity})
    text = text_gen.generate_document(metadata)
    print(f"\n{verbosity.upper()}:\n{text}")
```

## Príklad výstupu

### Low verbosity:
```
Dataset 'era5' contains variable 't2m' (2m temperature) measured in K | 
Statistics: mean=281.26K, range=[221.00, 317.40]K | 
Coordinates: latitude range [15.00, 75.00]°, longitude range [200.00, 330.00]° | 
Sample values (K): [241.20, 280.60, 292.79, ..., 293.40, 295.69]
```

### Medium verbosity (odporúčané pre RAG):
```
Dataset 'era5' contains variable 't2m' (2m temperature) measured in K | 
Statistics: mean=281.26K, range=[221.00, 317.40]K, std=16.32K, 3869000 data points | 
Coordinates: latitude range [15.00, 75.00]°, longitude range [200.00, 330.00]°, 
time period from 2013-01-01 to 2014-12-31, 6h frequency, dimensions: time=2920, lat=25, lon=53 | 
Sample values (K): [241.20, 280.60, 292.79, ..., 293.40, 295.69]
```

### High verbosity:
```
Dataset 'era5' contains variable 't2m' (2m temperature) measured in K | 
Statistics: count=3869000, max=317.40K, mean=281.26K, median=285.20K, min=221.00K, std=16.32K | 
Coordinates: latitude range [15.00, 75.00]°, longitude range [200.00, 330.00]°, 
time period from 2013-01-01 to 2014-12-31, 6h frequency, dimensions: time=2920, lat=25, lon=53 | 
Sample values (K): [241.20, 280.60, 292.79, ..., 293.40, 295.69] | 
Additional info: Variable attributes: precision=2, GRIB_id=11, GRIB_name=TMP | 
title: ERA5 Reanalysis | references: https://...
```

## Konfigurácia

Upravte `config/pipeline_config.yaml`:

```yaml
embeddings:
  statistics:
    - "mean"
    - "min"
    - "max"
    - "std"
    - "median"
    # Pridajte ľubovoľné ďalšie...
  
  text_generation:
    include_sample_values: true   # Dôležité pre RAG!
    include_statistics: true
    include_coordinates: true
    include_attributes: true
    verbosity: "medium"           # low/medium/high
```

## RAG Integration

```python
from src.embeddings import SemanticSearcher

# Setup
searcher = SemanticSearcher()

# Vyhľadávanie pre RAG
results = searcher.search("čo je priemerná teplota?", k=3)

# Pripravte kontext pre LLM
context = ""
for r in results:
    meta = r['metadata']
    context += f"""
    Premenná: {meta['variable']}
    Priemer: {meta['stat_mean']}
    Rozsah: [{meta['stat_min']}, {meta['stat_max']}]
    Vzorky: {meta['sample_values']}
    
    {r['document']}
    ---
    """

# Pošlite do LLM
# response = llm.generate(f"Context: {context}\n\nQuestion: {query}")
```

## Testovanie

```bash
# Spustite test
python scripts/test_flexible_embeddings.py
```

Test overí:
- ✅ Automatickú detekciu dimenzií
- ✅ Extrakciu všetkých premenných
- ✅ Generovanie textu vo všetkých verbosity levels
- ✅ Vytvorenie embeddings
- ✅ Semantic search s RAG

## Podporované formáty

- ✅ NetCDF (.nc)
- ✅ Parquet (.parquet)
- ✅ CSV (.csv)

Systém automaticky detekuje:
- 📍 Latitude/Longitude (lat, lon, latitude, longitude, y, x)
- ⏰ Time (time, valid_time, date, datetime)
- 📏 Levels (level, pressure, height, vertical)
- 📊 Všetky numerické premenné

## Rozdiel oproti starej verzii

| Stará verzia | Nová verzia |
|-------------|-------------|
| Hardcoded template | Dynamické generovanie |
| Špecifické premenné (t2m, tp) | Akékoľvek premenné |
| Bez sample values | ✅ Sample values pre RAG |
| Jedna verbosity | ✅ 3 úrovne verbosity |
| Fixné dimenzie | ✅ Auto-detekcia dimenzií |

## FAQ

**Q: Funguje to s mojimi dátami?**  
A: Áno! Ak máš NetCDF, CSV alebo Parquet s číselnými dátami, funguje to.

**Q: Musím meniť konfiguráciu pre nový dataset?**  
A: Nie! Systém automaticky deteguje všetko.

**Q: Ako pridám nové štatistiky?**  
A: Pridaj ich do `config/pipeline_config.yaml` pod `embeddings.statistics`.

**Q: Prečo sú tam sample values?**  
A: Pre RAG! LLM model vidí skutočné hodnoty a môže lepšie odpovedať na otázky.

**Q: Ktorú verbosity použiť?**  
A: `medium` - najlepší pomer informácie/dĺžka pre embeddings.

## Ďalšie príklady

### Batch processing:
```python
pipeline = EmbeddingPipeline()

# Spracuj celý adresár
result = pipeline.process_directory(
    "data/processed",
    pattern="*.nc",
    recursive=True
)

print(f"Spracovaných súborov: {result['num_files']}")
print(f"Vytvorených embeddings: {result['num_embeddings']}")
```

### Custom metadata config:
```python
config = {
    'statistics': ['mean', 'min', 'max', 'percentile_95'],
    'verbosity': 'high'
}

extractor = MetadataExtractor(config=config)
text_gen = TextGenerator(config=config)
```

## Záver

Systém je **production-ready** a funguje s akýmikoľvek meteorologickými dátami!

Neváhaj a testuj na svojich datasetoch! 🚀
