# Universal Dataset Support Examples

## Overview

The system is designed to handle **ANY climate/environmental dataset** from **ANY location** in the world. The normalized schema automatically adapts to different data types, formats, and geographic regions.

## Example 1: Air Pollution Data from Russia

### Input Data
- **Dataset**: Russian Air Quality Monitoring Network
- **Variable**: PM2.5 (particulate matter)
- **Location**: Moscow, Russia (55.75°N, 37.61°E)
- **Time**: 2024-01-15
- **Format**: CSV or NetCDF

### Normalized Payload in Vector DB

```json
{
  "dataset_name": "Russian Air Quality Network",
  "source_id": "RU_AQ_Moscow",
  "variable": "PM25",
  "long_name": "PM2.5 particulate matter concentration",
  "standard_name": "mass_concentration_of_pm2p5_ambient_aerosol_particles_in_air",
  "unit": "μg/m³",
  "time_start": "2024-01-15",
  "time_end": "2024-01-15",
  "temporal_frequency": "hourly",
  "latitude_min": 55.75,
  "latitude_max": 55.75,
  "longitude_min": 37.61,
  "longitude_max": 37.61,
  "stats_mean": 45.2,
  "stats_std": 12.5,
  "stats_min": 28.0,
  "stats_max": 78.5,
  "stats_median": 42.0,
  "stats_p90": 65.0,
  "file_path": "datasets/russia_air_quality/moscow_2024_01.csv",
  "chunk_index": 1
}
```

### Generated Human-Readable Text (for embedding)

```
Climate variable: PM2.5 particulate matter concentration (PM25) | Dataset: Russian Air Quality Network | Time: 2024-01-15 | Geographic coordinates: Latitude: 55.75°N Longitude: 37.61°E | Mean: 45.2 μg/m³ | Value range: 28.0 to 78.5 μg/m³
```

### RAG Query Examples

**Query 1**: "What was the air pollution level in Moscow in January 2024?"
- ✅ System finds: `RU_AQ_Moscow` dataset, `PM25` variable, location `55.75°N, 37.61°E`
- ✅ LLM answers: "The PM2.5 concentration in Moscow on January 15, 2024 averaged 45.2 μg/m³, with a range of 28.0 to 78.5 μg/m³."

**Query 2**: "Compare air pollution in Moscow and New York"
- ✅ System retrieves chunks from both locations
- ✅ LLM compares: Moscow (45.2 μg/m³) vs New York (if available)

**Query 3**: "Show me all air pollution data from Russia"
- ✅ Filter: `dataset_name` contains "Russia" OR `longitude_min` between 20-180 (Russia's longitude range)
- ✅ Returns all Russian air quality chunks

## Example 2: Precipitation Data from Brazil

### Input Data
- **Dataset**: Brazilian Weather Service
- **Variable**: Total Precipitation
- **Location**: São Paulo, Brazil (23.55°S, 46.64°W)
- **Time**: 2024-03-01 to 2024-03-31
- **Format**: NetCDF

### Normalized Payload

```json
{
  "dataset_name": "Brazilian Weather Service",
  "source_id": "BR_Weather_SP",
  "variable": "precipitation",
  "long_name": "Total precipitation",
  "standard_name": "precipitation_amount",
  "unit": "mm",
  "time_start": "2024-03-01",
  "time_end": "2024-03-31",
  "temporal_frequency": "daily",
  "latitude_min": -23.55,
  "latitude_max": -23.55,
  "longitude_min": -46.64,
  "longitude_max": -46.64,
  "stats_mean": 125.5,
  "stats_std": 45.2,
  "stats_min": 0.0,
  "stats_max": 280.0,
  "stats_median": 110.0
}
```

### RAG Query

**Query**: "How much did it rain in São Paulo in March 2024?"
- ✅ System finds: Brazilian dataset, precipitation variable, São Paulo coordinates
- ✅ LLM answers: "Total precipitation in São Paulo in March 2024 averaged 125.5 mm per day, with a maximum daily rainfall of 280.0 mm."

## Example 3: Temperature Data from Antarctica

### Input Data
- **Dataset**: Antarctic Research Station
- **Variable**: Air Temperature
- **Location**: McMurdo Station, Antarctica (77.84°S, 166.67°E)
- **Time**: 2024-07-01 to 2024-07-31
- **Format**: CSV

### Normalized Payload

```json
{
  "dataset_name": "Antarctic Research Station",
  "source_id": "ANT_McMurdo",
  "variable": "TEMP",
  "long_name": "Air temperature",
  "unit": "°C",
  "time_start": "2024-07-01",
  "time_end": "2024-07-31",
  "temporal_frequency": "daily",
  "latitude_min": -77.84,
  "latitude_max": -77.84,
  "longitude_min": 166.67,
  "longitude_max": 166.67,
  "stats_mean": -25.3,
  "stats_std": 8.5,
  "stats_min": -42.0,
  "stats_max": -10.5
}
```

### RAG Query

**Query**: "What was the temperature in Antarctica in July 2024?"
- ✅ System finds: Antarctic dataset, temperature variable, McMurdo coordinates
- ✅ LLM answers: "Air temperature at McMurdo Station, Antarctica in July 2024 averaged -25.3°C, with a range from -42.0°C to -10.5°C."

## Key Features That Make This Universal

### 1. **Flexible Variable Names**
- ✅ Any variable name works: `PM25`, `PM2.5`, `pm25`, `particulate_matter_2.5`
- ✅ System uses `long_name` and `standard_name` for understanding

### 2. **Global Coordinate Support**
- ✅ Northern hemisphere: positive latitude (55.75°N)
- ✅ Southern hemisphere: negative latitude (-23.55°S = São Paulo)
- ✅ Eastern hemisphere: positive longitude (37.61°E = Moscow)
- ✅ Western hemisphere: negative longitude (-46.64°W = São Paulo)
- ✅ All longitudes: -180° to +180° or 0° to 360°

### 3. **Any Unit System**
- ✅ Metric: `μg/m³`, `mm`, `°C`, `m/s`
- ✅ Imperial: `°F`, `inches`, `mph`
- ✅ Custom: any unit string is preserved

### 4. **Any Time Format**
- ✅ ISO: `2024-01-15`
- ✅ Timestamp: `2024-01-15T12:00:00Z`
- ✅ Custom: any time string is preserved

### 5. **Any Dataset Type**
- ✅ Air pollution: PM2.5, PM10, NO2, O3, CO
- ✅ Weather: temperature, precipitation, wind, pressure
- ✅ Climate: sea level, ice extent, ocean temperature
- ✅ Environmental: soil moisture, vegetation index, fire risk

## Filtering Examples

### Filter by Geographic Region

```python
# All data from Russia (longitude 20°E to 180°E, latitude 41°N to 82°N)
filter_dict = {
    "longitude_min": {"$gte": 20.0},
    "longitude_max": {"$lte": 180.0},
    "latitude_min": {"$gte": 41.0},
    "latitude_max": {"$lte": 82.0}
}

# All data from South America (latitude -56°S to 12°N, longitude -82°W to -35°W)
filter_dict = {
    "latitude_min": {"$gte": -56.0},
    "latitude_max": {"$lte": 12.0},
    "longitude_min": {"$gte": -82.0},
    "longitude_max": {"$lte": -35.0}
}
```

### Filter by Variable Type

```python
# All air pollution data
filter_dict = {
    "variable": {"$in": ["PM25", "PM10", "NO2", "O3", "CO"]}
}

# All temperature data
filter_dict = {
    "$or": [
        {"variable": {"$regex": "temp", "$options": "i"}},
        {"long_name": {"$regex": "temperature", "$options": "i"}}
    ]
}
```

### Filter by Dataset

```python
# All Russian datasets
filter_dict = {
    "dataset_name": {"$regex": "Russia|Russian|RU_", "$options": "i"}
}

# All air quality datasets
filter_dict = {
    "dataset_name": {"$regex": "air.*quality|AQ|pollution", "$options": "i"}
}
```

## RAG Query Examples for Mixed Datasets

**Query**: "What environmental data do we have from Russia?"
- ✅ Retrieves: Air pollution, temperature, precipitation, etc. from Russia
- ✅ LLM summarizes: "We have air quality data (PM2.5) from Moscow, temperature data from St. Petersburg, and precipitation data from Siberia."

**Query**: "Compare air pollution levels in Moscow and Beijing"
- ✅ Retrieves: PM2.5 data from both cities
- ✅ LLM compares: "Moscow averaged 45.2 μg/m³, while Beijing averaged 68.5 μg/m³ (if available)."

**Query**: "Show me all data from January 2024"
- ✅ Filter: `time_start >= "2024-01-01" AND time_end <= "2024-01-31"`
- ✅ Returns: All datasets from January 2024 (temperature, precipitation, air pollution, etc.)

## Conclusion

✅ **The system is 100% universal** - it works with:
- Any geographic location (Russia, Brazil, Antarctica, anywhere)
- Any variable type (temperature, precipitation, air pollution, etc.)
- Any data format (CSV, NetCDF, GeoTIFF, GRIB, etc.)
- Any unit system (metric, imperial, custom)
- Any time period (historical, current, future projections)

The normalized schema ensures consistent storage and retrieval, while the dynamic text generation and LLM prompt ensure accurate understanding of any variable type.


