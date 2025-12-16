# Demo Setup - Slovakia Summer 2023 Data

## Prehľad

Tento demo dataset obsahuje klimatické dáta pre Slovensko v lete 2023 (jún-august).

## Dataset Informácie

- **Zdroj**: `slovakia_summer_2023`
- **Formát**: CSV
- **Obdobie**: 2023-06-01 až 2023-08-31 (92 dní)
- **Stanice**: 3 (Bratislava, Košice, Žilina)
- **Premenné**: 
  - TMAX, TMIN, TAVG (teplota)
  - PRCP (zrážky)
  - SNOW (sneh)
  - hurs (relatívna vlhkosť)
  - AWND, WSF2, WSF5 (vietor)

## Kroky na spustenie demo

### 1. Vytvorenie source

```bash
# Spusti script na vytvorenie source
python create_demo_source.py
```

Alebo manuálne cez API:

```bash
curl -X POST http://localhost:8000/sources \
  -H "Content-Type: application/json" \
  -d '{
    "source_id": "slovakia_summer_2023",
    "url": "file:///path/to/xchomo_diploma/data/raw/slovakia_summer_2023_sample.csv",
    "format": "csv",
    "description": "Sample climate data for Slovakia - Summer 2023",
    "tags": ["slovakia", "summer", "2023", "demo"],
    "variables": ["TMAX", "TMIN", "TAVG", "PRCP", "SNOW", "hurs", "AWND", "WSF2", "WSF5"],
    "is_active": true
  }'
```

### 2. Spracovanie dát (Dagster)

1. Otvor Dagster UI: http://localhost:3000
2. Nájdi job: `dynamic_source_etl_job`
3. Spusti s konfiguráciou:
   ```json
   {
     "ops": {
       "process_source": {
         "config": {
           "source_id": "slovakia_summer_2023"
         }
       }
     }
   }
   ```

### 3. Testovanie cez Chat

Po spracovaní môžeš testovať s týmito otázkami:

**Slovensky:**
- "Ukáž mi štatistiky teploty pre Slovensko v lete 2023"
- "Aké boli priemerné teploty v Bratislave v júli 2023?"
- "Porovnaj teplotu medzi Bratislavou a Košicami"
- "Aké boli maximálne a minimálne teploty v auguste?"

**English:**
- "What is the average temperature in Slovakia in summer 2023?"
- "Show me temperature statistics for Slovakia in summer 2023"
- "Compare temperature between Bratislava and Kosice"
- "What was the temperature range in August 2023?"

## Očakávané výsledky

Po spracovaní by mal systém vedieť odpovedať na:
- Štatistiky teploty (priemer, min, max)
- Porovnania medzi stanicami
- Filtrovanie podľa času (jún, júl, august)
- Filtrovanie podľa regiónu (Bratislava, Košice, Žilina)
- Analýza zrážok a vlhkosti

## Štruktúra dát

CSV obsahuje:
- `date`: dátum (YYYY-MM-DD)
- `station_id`: ID stanice (SK001, SK002, SK003)
- `station_name`: názov stanice
- `latitude`, `longitude`: geografické súradnice
- `TMAX`, `TMIN`, `TAVG`: teploty (°C)
- `PRCP`: zrážky (mm)
- `SNOW`: sneh (cm)
- `hurs`: relatívna vlhkosť (%)
- `AWND`, `WSF2`, `WSF5`: rýchlosti vetra (m/s)

## Poznámky

- Dataset obsahuje 92 riadkov (92 dní × 3 stanice = 276 záznamov)
- Dáta sú syntetické, ale realistické
- Vhodné pre demo a testovanie RAG funkcionality

