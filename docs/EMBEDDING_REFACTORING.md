# Embedding System Refactoring - Flexible Approach

## Prehľad zmien

Embedding systém bol kompletne refaktorovaný, aby odstránil všetky hardcoded hodnoty a šablóny. Nový systém je plne flexibilný a dokáže spracovať akýkoľvek meteorologický dataset bez úpravy kódu.

## Hlavné zmeny

### 1. Nový modul: `metadata_extractor.py`

**Trieda:** `MetadataExtractor`

**Funkcie:**
- Automaticky deteguje všetky dimenzie (lat, lon, time, level, atď.)
- Extrahuje všetky premenné bez ohľadu na názov
- Vypočíta štatistiky (mean, min, max, std, median, percentiles)
- Uloží vzorky hodnôt pre RAG
- Extrahuje všetky atribúty (units, long_name, standard_name, atď.)
- Podporuje NetCDF, CSV, Parquet formáty

**Príklad použitia:**
```python
from src.embeddings import MetadataExtractor

extractor = MetadataExtractor()
metadata_list = extractor.extract_from_dataset(
    data=xr.open_dataset("data.nc"),
    file_path="data.nc",
    dataset_id="my_dataset"
)
```

### 2. Nový modul: `text_generator.py`

**Trieda:** `TextGenerator`

**Funkcie:**
- Dynamické generovanie textu bez hardcoded šablón
- Tri úrovne verbosity (low, medium, high)
- Zahrnutie sample values pre RAG kontext
- Flexibilné formátovanie pre akýkoľvek typ dát

**Príklad použitia:**
```python
from src.embeddings import TextGenerator

# Medium verbosity - odporúčané pre RAG
text_gen = TextGenerator(config={'verbosity': 'medium'})
texts = text_gen.generate_batch(metadata_list)
```

**Výstup (medium verbosity):**
```
Dataset 'era5_2023' contains variable 't2m' (2 meter temperature) measured in K | 
Statistics: mean=281.26K, range=[221.00, 317.40]K, std=16.32K, 3869000 data points | 
Coordinates: latitude range [15.00, 75.00]°, longitude range [200.00, 330.00]°, 
time period from 2013-01-01 00:00:00 to 2014-12-31 18:00:00, 6h frequency, 
dimensions: time=2920, lat=25, lon=53 | 
Sample values (K): [241.20, 280.60, 292.79, ..., 293.40, 295.69]
```

### 3. Refaktorovaný `pipeline.py`

**Zmeny:**
- Používa `MetadataExtractor` namiesto hardcoded extraction
- Používa `TextGenerator` namiesto hardcoded templates
- Nová metóda `_prepare_db_metadata()` pre ChromaDB kompatibilitu
- Odstránené všetky hardcoded názvy premenných a dimenzií

### 4. Aktualizovaná konfigurácia

**`config/pipeline_config.yaml`:**

```yaml
embeddings:
  # Statistics to compute (flexible - no hardcoded variables)
  statistics:
    - "mean"
    - "min"
    - "max"
    - "std"
    - "count"
    - "median"
    - "percentile_25"
    - "percentile_75"
  
  # Text generation for embeddings (RAG-optimized)
  text_generation:
    include_sample_values: true  # Crucial for RAG
    include_statistics: true
    include_coordinates: true
    include_attributes: true
    verbosity: "medium"  # low, medium, high
```

## Výhody novej implementácie

### 1. Flexibilita
- ✅ Funguje s akýmkoľvek meteorologickým datasetom
- ✅ Nepotrebuje úpravu kódu pre nové premenné
- ✅ Automatická detekcia dimenzií a súradníc
- ✅ Podporuje rôzne formáty (NetCDF, CSV, Parquet)

### 2. RAG optimalizácia
- ✅ Zahrnutie sample values v embeddings
- ✅ Kompletné štatistiky dostupné pre retrieval
- ✅ Bohaté metadata v ChromaDB
- ✅ Kontextovo bohaté dokumenty

### 3. Údržba
- ✅ Žiadne hardcoded šablóny
- ✅ Konfigurovateľné cez YAML
- ✅ Jednoduchá rozšíriteľnosť
- ✅ Čitateľnejší kód

## Migrácia zo starej verzie

### Pred refaktoringom:
```python
# Hardcoded template v config
template: "Dataset {id}: variable {variable} daily mean={mean:.2f}{unit}"

# Hardcoded extrakcia
lat_coord = self._find_coord(ds, ['latitude', 'lat', 'y'])
```

### Po refaktoringu:
```python
# Automatická extrakcia
extractor = MetadataExtractor()
metadata = extractor.extract_from_dataset(data, file_path, dataset_id)

# Dynamické generovanie textu
text_gen = TextGenerator(config={'verbosity': 'medium'})
text = text_gen.generate_document(metadata)
```

## Príklady použitia

### Základné použitie:
```python
from src.embeddings import EmbeddingPipeline

pipeline = EmbeddingPipeline()

# Funguje s akýmkoľvek NC súborom
result = pipeline.process_dataset("path/to/any_weather_data.nc")
```

### Semantic search s RAG:
```python
from src.embeddings import SemanticSearcher

searcher = SemanticSearcher()

# Vyhľadávanie
results = searcher.search("temperature data around 15 degrees", k=5)

for result in results:
    print(f"Variable: {result['metadata']['variable']}")
    print(f"Mean: {result['metadata']['stat_mean']}")
    print(f"Samples: {result['metadata']['sample_values']}")
    print(f"Document: {result['document']}")
```

## Testovanie

Spustite test script:
```bash
python scripts/test_flexible_embeddings.py
```

Test overí:
1. ✅ Metadata extraction z NetCDF
2. ✅ Dynamické generovanie textu (low/medium/high verbosity)
3. ✅ Celý embedding pipeline
4. ✅ RAG-friendly semantic search

## Štruktúra metadata

### Extrahované pre každú premennú:
```python
{
    'id': 'dataset_variable',
    'dataset_id': 'dataset',
    'variable': 'variable_name',
    'long_name': 'Descriptive name',
    'standard_name': 'CF standard name',
    'unit': 'K',
    'dimensions': ['time', 'lat', 'lon'],
    'shape': [2920, 25, 53],
    
    # Statistics
    'stat_mean': 281.26,
    'stat_min': 221.0,
    'stat_max': 317.4,
    'stat_std': 16.32,
    'stat_count': 3869000,
    'stat_median': 285.2,
    
    # Coordinates
    'spatial_extent': {
        'lat_min': 15.0,
        'lat_max': 75.0,
        'lon_min': 200.0,
        'lon_max': 330.0
    },
    'temporal_extent': {
        'start_date': '2013-01-01 00:00:00',
        'end_date': '2014-12-31 18:00:00',
        'frequency': '6h'
    },
    
    # Sample values (pre RAG)
    'sample_values': [241.2, 280.6, 292.79, 285.7, 287.79],
    
    # Všetky atribúty
    'variable_attributes': {...},
    'global_attributes': {...}
}
```

## Kompatibilita s RAG

Nový systém je optimalizovaný pre RAG:

1. **Sample values** - RAG model vidí skutočné hodnoty
2. **Bohaté štatistiky** - Kontext pre generovanie odpovedí
3. **Kompletné metadata** - Všetky informácie o premennej
4. **Flexibilný text** - Optimalizovaný pre embedding similarity

## Ďalšie kroky

1. ✅ Metadata extraction - hotovo
2. ✅ Text generation - hotovo
3. ✅ Pipeline refactoring - hotovo
4. ✅ Config update - hotovo
5. ✅ RAG testing - hotovo

## Záver

Embedding systém je teraz:
- 🎯 Flexibilný - funguje s akýmikoľvek dátami
- 🚀 RAG-ready - zahrnuté sample values
- 🔧 Konfigurovateľný - žiadne hardcoded hodnoty
- 📈 Škálovateľný - pripravený na produkciu

Systém je pripravený na integráciu s RAG modelmi a môže spracovať akýkoľvek meteorologický dataset bez úprav kódu.
