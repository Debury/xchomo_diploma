# 🎯 Roadmap - Kompletné riešenie diplomovej práce

## 📋 Analýza aktuálneho stavu

### ✅ Čo už funguje (Implementované)

1. **ETL Pipeline**
   - ✅ Dagster orchestration
   - ✅ Dynamické spracovanie zdrojov
   - ✅ Multi-format podpora (NetCDF, GRIB, HDF5, GeoTIFF, CSV, Zarr)
   - ✅ Memory-safe chunking (žiadne OOM chyby)
   - ✅ Auto-detekcia formátov

2. **Webové rozhranie**
   - ✅ FastAPI REST API
   - ✅ Frontend UI pre správu zdrojov
   - ✅ CRUD operácie pre zdroje
   - ✅ ETL trigger cez API

3. **Embeddings & Vector DB**
   - ✅ Qdrant vector database
   - ✅ BAAI/bge-large-en-v1.5 embeddings (1024-dim)
   - ✅ Text generation pre RAG
   - ✅ Semantic search

4. **RAG Pipeline**
   - ✅ Ollama LLM integrácia
   - ✅ Context retrieval
   - ✅ Answer generation

5. **Source Management**
   - ✅ Dynamické pridávanie zdrojov
   - ✅ Format auto-detection
   - ✅ Status tracking

### ⚠️ Čo treba doplniť/zdokonaliť

1. **Špecifické integrácie pre klimatické datasety**
   - ⏳ ERA5 (CDS API integrácia)
   - ⏳ CMIP6 (THREDDS/OPeNDAP)
   - ⏳ EURO-CORDEX
   - ⏳ E-OBS
   - ⏳ CRU
   - ⏳ NCEP-NCAR2

2. **Pokročilé spracovanie**
   - ⏳ Regridding (normalizácia priestorového rozlíšenia)
   - ⏳ Temporal alignment
   - ⏳ Multi-variable handling
   - ⏳ Metadata extraction a normalizácia

3. **Kvalita a testovanie**
   - ⏳ Rozšírené testy pre všetky formáty
   - ⏳ Integration tests s reálnymi dátami
   - ⏳ Performance benchmarking
   - ⏳ Error handling a recovery

4. **Dokumentácia**
   - ⏳ Technická dokumentácia
   - ⏳ User guide
   - ⏳ API dokumentácia
   - ⏳ Architektúrny diagram

5. **Optimalizácia**
   - ⏳ Parallel processing
   - ⏳ Caching stratégie
   - ⏳ Batch operations
   - ⏳ Resource management

---

## 🗺️ Plán implementácie (Fázy)

### **FÁZA 1: Rozšírenie podpory pre špecifické datasety** (2-3 týždne)

#### 1.1 ERA5 Integrácia
- [ ] CDS API client wrapper
- [ ] Automatické stiahnutie ERA5 dát
- [ ] Metadata extraction z ERA5
- [ ] Unit conversion (Kelvin → Celsius)
- [ ] Temporal/spatial subsetting

**Súbory:**
- `src/data_acquisition/era5_client.py` (nový)
- `src/data_acquisition/__init__.py`
- `dagster_project/ops/era5_ops.py` (nový)

#### 1.2 CMIP6 Integrácia
- [ ] THREDDS/OPeNDAP client
- [ ] CMIP6 metadata parser
- [ ] Multi-model handling
- [ ] Scenario extraction (SSP, RCP)

**Súbory:**
- `src/data_acquisition/cmip6_client.py` (nový)
- `dagster_project/ops/cmip6_ops.py` (nový)

#### 1.3 E-OBS & CRU Integrácia
- [ ] E-OBS downloader (gridded observations)
- [ ] CRU TS downloader
- [ ] Station data handling
- [ ] Quality flags processing

**Súbory:**
- `src/data_acquisition/eobs_client.py` (nový)
- `src/data_acquisition/cru_client.py` (nový)

#### 1.4 EURO-CORDEX Integrácia
- [ ] CORDEX data access
- [ ] Regional model handling
- [ ] Downscaling metadata

**Súbory:**
- `src/data_acquisition/cordex_client.py` (nový)

---

### **FÁZA 2: Pokročilé spracovanie a normalizácia** (2-3 týždne)

#### 2.1 Regridding & Spatial Normalization
- [ ] CDO wrapper alebo Python implementácia
- [ ] Bilinear interpolation
- [ ] Conservative remapping
- [ ] Target grid selection (configurable)

**Súbory:**
- `src/data_transformation/regridding.py` (nový)
- `src/data_transformation/spatial_ops.py` (nový)

#### 2.2 Temporal Alignment
- [ ] Time series alignment
- [ ] Frequency conversion (hourly → daily → monthly)
- [ ] Missing data handling
- [ ] Temporal aggregation strategies

**Súbory:**
- `src/data_transformation/temporal_ops.py` (nový)

#### 2.3 Metadata Normalization
- [ ] CF conventions compliance
- [ ] Variable name mapping
- [ ] Unit standardization
- [ ] Coordinate system normalization

**Súbory:**
- `src/data_transformation/metadata_normalizer.py` (nový)

#### 2.4 Multi-Variable Processing
- [ ] Variable selection strategies
- [ ] Cross-variable relationships
- [ ] Derived variables (e.g., wind speed from u/v)
- [ ] Variable grouping

**Súbory:**
- `src/data_transformation/variable_ops.py` (nový)

---

### **FÁZA 3: Rozšírené testovanie a validácia** (1-2 týždne)

#### 3.1 Format Testing Suite
- [ ] Test pre každý podporovaný formát
- [ ] Test s reálnymi dátami z rôznych zdrojov
- [ ] Edge cases (malé/veľké súbory, chybný formát)
- [ ] Memory leak testing

**Súbory:**
- `tests/test_formats_comprehensive.py` (nový)
- `tests/test_real_data_sources.py` (nový)
- `tests/test_memory_safety.py` (nový)

#### 3.2 Integration Tests
- [ ] End-to-end ETL testy
- [ ] RAG pipeline testy
- [ ] API endpoint testy
- [ ] Dagster job testy

**Súbory:**
- `tests/test_integration_etl.py` (nový)
- `tests/test_integration_rag.py` (nový)

#### 3.3 Data Quality Tests
- [ ] Metadata validation
- [ ] Data range checks
- [ ] Missing data detection
- [ ] Consistency checks

**Súbory:**
- `tests/test_data_quality.py` (nový)
- `src/data_transformation/quality_checks.py` (nový)

---

### **FÁZA 4: Optimalizácia a škálovateľnosť** (1-2 týždne)

#### 4.1 Parallel Processing
- [ ] Multi-source parallel processing
- [ ] Chunk-level parallelism
- [ ] Dask integration pre veľké datasets
- [ ] Resource pooling

**Súbory:**
- `src/utils/parallel_processing.py` (nový)
- `dagster_project/ops/parallel_ops.py` (nový)

#### 4.2 Caching & Performance
- [ ] Download caching
- [ ] Embedding cache
- [ ] Metadata cache
- [ ] Query result caching

**Súbory:**
- `src/utils/cache.py` (nový)

#### 4.3 Resource Management
- [ ] Memory monitoring
- [ ] CPU usage optimization
- [ ] Disk space management
- [ ] Cleanup strategies

**Súbory:**
- `src/utils/resource_manager.py` (nový)

---

### **FÁZA 5: Dokumentácia a finálne úpravy** (1 týždeň)

#### 5.1 Technická dokumentácia
- [ ] Architektúrny diagram
- [ ] API dokumentácia
- [ ] Configuration guide
- [ ] Deployment guide

**Súbory:**
- `docs/ARCHITECTURE.md` (nový)
- `docs/API.md` (nový)
- `docs/DEPLOYMENT.md` (nový)

#### 5.2 User Guide
- [ ] Getting started guide
- [ ] Source management guide
- [ ] RAG query examples
- [ ] Troubleshooting

**Súbory:**
- `docs/USER_GUIDE.md` (nový)
- `docs/TROUBLESHOOTING.md` (nový)

#### 5.3 Code Quality
- [ ] Code review
- [ ] Linting fixes
- [ ] Type hints completion
- [ ] Docstring updates

---

## 🎯 Prioritizácia (Čo urobiť najprv)

### **VYSOKÁ PRIORITA** (Pre funkčné riešenie)

1. ✅ **ERA5 integrácia** - najdôležitejší dataset
2. ✅ **Regridding** - kľúčové pre normalizáciu
3. ✅ **Rozšírené testy** - validácia riešenia
4. ✅ **Dokumentácia** - pre diplomovú prácu

### **STREDNÁ PRIORITA** (Pre kompletnosť)

5. CMIP6 integrácia
6. E-OBS/CRU integrácia
7. Temporal alignment
8. Metadata normalization

### **NÍZKA PRIORITA** (Nice to have)

9. EURO-CORDEX
10. NCEP-NCAR2
11. Advanced caching
12. Performance optimization

---

## 📊 Metriky úspechu

### Funkčnosť
- [ ] Podpora pre minimálne 3 hlavné datasety (ERA5, CMIP6, E-OBS)
- [ ] Úspešné spracovanie aspoň 5 rôznych formátov
- [ ] RAG pipeline funguje s reálnymi dátami
- [ ] Web UI umožňuje pridávanie a správu zdrojov

### Kvalita
- [ ] Test coverage > 70%
- [ ] Žiadne memory leaks
- [ ] Error handling pre všetky edge cases
- [ ] Dokumentácia kompletná

### Výkon
- [ ] Spracovanie 1GB dát bez OOM
- [ ] RAG query < 5 sekúnd
- [ ] ETL job < 30 minút pre typický dataset

---

## 🛠️ Konkrétne kroky pre najbližšie 2 týždne

### Týždeň 1: ERA5 Integrácia

**Deň 1-2: ERA5 Client**
```python
# src/data_acquisition/era5_client.py
class ERA5Client:
    def __init__(self, api_key, api_url):
        self.client = cdsapi.Client(url=api_url, key=api_key)
    
    def download(self, request_params):
        # Download ERA5 data
        pass
    
    def extract_metadata(self, file_path):
        # Extract ERA5-specific metadata
        pass
```

**Deň 3-4: ERA5 Dagster Op**
```python
# dagster_project/ops/era5_ops.py
@op
def download_era5_data(context, era5_client, request_params):
    # Download and process ERA5
    pass
```

**Deň 5: Testy a integrácia**
- Unit testy pre ERA5 client
- Integration test s malým ERA5 datasetom
- Dokumentácia

### Týždeň 2: Regridding & Normalization

**Deň 1-3: Regridding Implementation**
```python
# src/data_transformation/regridding.py
def regrid_to_target(
    source_data: xr.Dataset,
    target_grid: dict,
    method: str = "bilinear"
) -> xr.Dataset:
    # Implement regridding
    pass
```

**Deň 4-5: Integration & Testing**
- Integrácia do ETL pipeline
- Testy s rôznymi gridmi
- Dokumentácia

---

## 📝 Poznámky pre implementáciu

### Best Practices
1. **Memory Safety**: Vždy používať chunking pre veľké súbory
2. **Error Handling**: Graceful degradation, nie crash
3. **Logging**: Detailné logy pre debugging
4. **Configuration**: Všetko cez config súbory, nie hardcoded
5. **Testing**: Test pre každú novú funkcionalitu

### Technológie
- **CDO**: Pre regridding (ak potrebné, wrapper)
- **xarray**: Pre NetCDF/GRIB handling
- **dask**: Pre parallel processing
- **cfgrib**: Pre GRIB files
- **rasterio**: Pre GeoTIFF

### Dátové zdroje pre testovanie
- ERA5: Malý subset (1 mesiac, malá oblasť)
- CMIP6: Sample dataset z ESGF
- E-OBS: Test dataset
- CSV: NASA GISTEMP (už funguje)

---

## 🎓 Pre diplomovú prácu

### Čo zdôrazniť v práci:

1. **Heterogenita zdrojov** - Ako systém rieši rôzne formáty
2. **Memory-safe processing** - Chunking stratégie
3. **Normalizácia** - Regridding, temporal alignment
4. **Embeddings** - Prečo a ako pre LLM
5. **Orchestration** - Dagster pre ETL
6. **RAG Pipeline** - Integrácia s LLM

### Výsledky a evaluácia:

1. **Testovanie na reálnych dátach**
   - ERA5: 1 rok, Európa
   - CMIP6: 1 model, 1 scenario
   - E-OBS: 1 rok

2. **Performance metríky**
   - Processing time
   - Memory usage
   - Embedding quality

3. **Porovnanie**
   - S existujúcimi nástrojmi (CDO, GDAL)
   - Výhody/nevýhody

---

## ✅ Checklist pre dokončenie

### Funkčnosť
- [ ] ERA5 integrácia funguje
- [ ] CMIP6 integrácia funguje
- [ ] E-OBS integrácia funguje
- [ ] Regridding funguje
- [ ] Temporal alignment funguje
- [ ] RAG pipeline funguje s reálnymi dátami

### Kvalita
- [ ] Testy prešli (>70% coverage)
- [ ] Žiadne kritické bugy
- [ ] Dokumentácia kompletná
- [ ] Code review hotový

### Dokumentácia
- [ ] README aktualizovaný
- [ ] API dokumentácia
- [ ] User guide
- [ ] Technická dokumentácia

---

**Status**: 🟡 V procese  
**Next Milestone**: ERA5 integrácia + Regridding  
**Target Completion**: 4-6 týždňov

