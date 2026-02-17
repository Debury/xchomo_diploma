# Data Quality Samples — What's Actually in Qdrant

*Generated: 17 February 2026*

**Collection:** `climate_data` | **Points:** 131,992 | **Vectors:** 1024-dim (BGE-M3, cosine) | **Status:** green

---

## 1. How It Works

Each chunk in Qdrant consists of:
1. **Vector** — 1024-dimensional float array from BAAI/bge-m3 (unit-normalized, cosine similarity)
2. **Payload** — structured metadata (dataset, variable, coordinates, statistics, etc.)
3. **Text** — reconstructed at query time from payload by `schema.py` for LLM context

The text is NOT stored in Qdrant — it's built dynamically from the payload fields into a pipe-delimited string like:
```
Climate variable: mean temperature, air_temperature (tg) | Variable code: tg | Meaning: mean temperature |
Standard name: air_temperature | Unit: Celsius | Dataset: E-OBS | Time: 2011-09-19 |
Geographic coordinates: Latitude: -40.3750° to -15.6250°N |
Statistics: Mean: 5.49 Celsius | Std dev: 2.53 Celsius | Range: [-2.07, 9.86] Celsius |
Median: 6.01 Celsius | Hazard: Mean surface temperature | Location: Europe
```

This text representation is what gets embedded into the 1024-dim vector.

---

## 2. Vector Properties

```
Dimensions:  1024
Norm:        1.0000 (unit-normalized)
Value range: [-0.1956, 0.2706] (typical)
First 10:    [-0.0506, -0.0011, -0.0589, -0.0361, -0.0221, -0.0266, -0.0320, 0.0179, 0.0028, 0.0140]
```

---

## 3. Sample: CRU TS (Observed Temperature)

**Source:** `catalog_CRU_2` | **Dataset:** CRU TS4.09 Mean Temperature

### Chunk with full variable metadata (tmp)
```json
{
  "dataset_name": "CRU",
  "source_id": "catalog_CRU_2",
  "variable": "tmp",
  "long_name": "near-surface temperature",
  "unit": "degrees Celsius",
  "time_start": "2023-10-16T00:00:00",
  "latitude_min": -39.75,
  "latitude_max": 9.75,
  "longitude_min": 70.25,
  "longitude_max": 119.75,
  "stats_mean": 25.91,
  "stats_std": 3.37,
  "stats_min": 12.60,
  "stats_max": 30.40,
  "stats_p10": 20.40,
  "stats_median": 27.20,
  "stats_p90": 28.60,
  "stats_range": 17.80,
  "dataset_title": "CRU TS4.09 Mean Temperature",
  "dataset_institution": "British Atmospheric Data Centre, RAL, UK",
  "hazard_type": "Mean surface temperature",
  "location_name": "Global"
}
```

**Quality notes:**
- Has `long_name`, `unit`, full statistics — GOOD
- Covers South/Southeast Asia region (lat 70-120°E), Oct 2023
- Mean 25.91°C makes sense for tropical region in October

### Chunk with anomaly variable (maea)
```json
{
  "dataset_name": "CRU",
  "source_id": "catalog_CRU_2",
  "variable": "maea",
  "time_start": "2024-12-16T00:00:00",
  "latitude_min": -29.75,
  "latitude_max": 19.75,
  "stats_mean": -0.18,
  "stats_std": 0.37,
  "stats_min": -0.95,
  "stats_max": 0.79,
  "hazard_type": "Mean surface temperature",
  "location_name": "Global"
}
```

**Quality notes:**
- Variable `maea` — no `long_name` or `unit` — ISSUE: embedding text will just say "Climate variable: maea" with no semantic meaning
- The values look like temperature anomalies (range -0.95 to +0.79)
- Missing: longitude fields in some CRU chunks

---

## 4. Sample: CMIP6 (Climate Model Output)

**Source:** `catalog_CMIP6_1` | **Dataset:** HadGEM3-GC31-LL (MOHC), historical experiment

```json
{
  "dataset_name": "CMIP6",
  "source_id": "catalog_CMIP6_1",
  "variable": "tas",
  "long_name": "Near-Surface Air Temperature",
  "standard_name": "air_temperature",
  "unit": "K",
  "time_start": "1956-10-16 00:00:00",
  "latitude_min": 35.625,
  "latitude_max": 89.375,
  "stats_mean": 269.27,
  "stats_std": 15.49,
  "stats_min": 234.25,
  "stats_max": 295.67,
  "dataset_experiment": "all-forcing simulation of the recent past",
  "dataset_experiment_id": "historical",
  "dataset_frequency": "mon",
  "dataset_grid": "Native N96 grid; 192 x 144 longitude/latitude",
  "dataset_nominal_resolution": "250 km",
  "dataset_institution": "Met Office Hadley Centre",
  "dataset_source_id": "HadGEM3-GC31-LL",
  "dataset_variant_label": "r1i1p1f3",
  "hazard_type": "Mean surface temperature",
  "impact_sector": "Health, Energy, Agriculture",
  "location_name": "Global"
}
```

**Quality notes:**
- EXCELLENT metadata — `long_name`, `standard_name`, `unit`, full CMIP6 attributes
- Unit is Kelvin (K) — mean 269.27K = -3.88°C, makes sense for Northern Hemisphere October
- Rich dataset-level attributes (experiment, grid, resolution, institution, variant)
- Very verbose payload (~40 fields) — lots of CMIP6 global attributes stored

---

## 5. Sample: E-OBS (European Observations)

**Source:** `catalog_E-OBS_3` | **Dataset:** E-OBS v32.0e

```json
{
  "dataset_name": "E-OBS",
  "source_id": "catalog_E-OBS_3",
  "variable": "tg",
  "long_name": "mean temperature",
  "standard_name": "air_temperature",
  "unit": "Celsius",
  "time_start": "2016-03-15T00:00:00",
  "latitude_min": 9.625,
  "latitude_max": 34.375,
  "stats_mean": 3.57,
  "stats_std": 4.88,
  "stats_min": -7.25,
  "stats_max": 22.44,
  "stats_p10": -0.54,
  "stats_median": 2.31,
  "stats_p90": 8.98,
  "dataset_E-OBS_version": "32.0e",
  "hazard_type": "Mean surface temperature",
  "location_name": "Europe"
}
```

**Quality notes:**
- Good metadata — `long_name`, `standard_name`, `unit` all present
- Mean 3.57°C for March 2016 in Europe — plausible
- Has percentile statistics (p10, p90) — useful for distribution analysis

---

## 6. Sample: GISTEMP (Metadata-only Phase 0 + CSV data)

### Phase 0 metadata entry
```json
{
  "source_id": "catalog_GISTEMP_7",
  "dataset_name": "GISTEMP",
  "variable": "catalog_mean_surface_temperature",
  "is_metadata_only": true,
  "catalog_row_index": 7,
  "hazard_type": "Mean surface temperature",
  "data_type": "Gridded observations",
  "spatial_coverage": "Global",
  "spatial_resolution": "2 degrees",
  "temporal_coverage_text": "1880 - Present",
  "temporal_frequency": "Monthly",
  "access_type": "Open",
  "link": "https://data.giss.nasa.gov/gistemp/",
  "location_name": "Global"
}
```

### Phase 1 actual data (CSV)
```json
{
  "dataset_name": "GISTEMP",
  "source_id": "catalog_GISTEMP_7",
  "variable": "Land-Ocean: Global Means",
  "stats_mean": 0.095,
  "stats_std": 0.400,
  "stats_min": -0.530,
  "stats_max": 1.410,
  "stats_p10": -0.295,
  "stats_median": -0.015,
  "stats_p90": 0.730,
  "file_path": "tmpaiqzh4yk.csv",
  "chunk_index": 0,
  "row_count": 148,
  "hazard_type": "Mean surface temperature",
  "location_name": "Global"
}
```

**Quality notes:**
- Phase 0 entry has descriptive catalog metadata — good for "what datasets exist?" queries
- Phase 1 CSV data: 148 rows of global mean anomalies (1880-present), range [-0.53, +1.41]°C — matches known GISTEMP data
- No `unit` field on the CSV chunk — ISSUE: embedding won't mention °C

---

## 7. RAG Query Test

**Query:** "What is the mean temperature in Central Europe?"

**Result:** The LLM correctly identified that the retrieved chunks were NOT from Central Europe (lat -40° to -15°, Southern Hemisphere) and said so transparently. It listed the actual values from the retrieved E-OBS chunks and noted that European-focused sources exist but weren't in the top-3 results.

**Retrieval scores:** 0.573, 0.572, 0.572 (cosine similarity) — relatively low, all from E-OBS but wrong spatial region.

**Assessment:**
- LLM honesty: GOOD — didn't hallucinate Central Europe data
- Retrieval quality: POOR for this query — latitude filtering not working well, the text says "Location: Europe" but the actual lat/lon is Southern Hemisphere
- Root cause: The `location_name: "Europe"` field is misleading — it's the catalog-level location, but the actual chunk coordinates are not in Europe

---

## 8. Identified Issues

| Issue | Severity | Dataset | Description |
|-------|----------|---------|-------------|
| Missing `long_name`/`unit` on anomaly variables | Medium | CRU | `maea` variable has no semantic description → weak embedding |
| Missing `unit` on CSV chunks | Medium | GISTEMP | CSV data doesn't carry unit metadata |
| Misleading `location_name` | High | E-OBS | Catalog says "Europe" but chunks can have Southern Hemisphere coords |
| Missing longitude on some chunks | Low | CRU | Some CRU chunks lack `longitude_min/max` fields |
| No `text` field stored | Info | All | Text is reconstructed at query time — can't inspect what was actually embedded |
| CMIP6 unit in Kelvin | Low | CMIP6 | Not an error, but mixing K and °C across datasets may confuse retrieval |
