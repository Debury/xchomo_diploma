"""Process six open climate & air-quality data sources and validate embeddings/RAG."""

from __future__ import annotations

import os
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple

import pandas as pd
import requests
import xarray as xr
import numpy as np
from rich.console import Console
from rich.table import Table

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.embeddings import EmbeddingGenerator, EmbeddingPipeline, SemanticSearcher

console = Console()

# Force the vector database into in-memory mode so we do not require Qdrant running locally
os.environ.setdefault("VECTOR_DB_MEMORY_ONLY", "true")

DOWNLOAD_DIR = Path("data/external/open_sources")
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class DataSource:
    """Simple container describing an open dataset."""

    id: str
    category: str
    description: str
    fetcher: str
    kwargs: Dict[str, str]


# ----------------------------------------------------------------------------
# Helper utilities
# ----------------------------------------------------------------------------

def dataset_from_series(
    times: Iterable,
    values: Iterable,
    variable_name: str,
    units: str,
    attrs: Dict[str, str],
) -> xr.Dataset:
    """Convert a 1-D time series into an xarray dataset saved as NetCDF."""

    series = pd.to_numeric(pd.Series(list(values)), errors="coerce")
    timestamps = pd.to_datetime(list(times))
    mask = series.notna()
    if int(mask.sum()) == 0:
        raise RuntimeError(f"No finite values found for variable '{variable_name}'")

    cleaned_values = series[mask].astype(float).to_numpy()
    cleaned_times = pd.DatetimeIndex(timestamps)[mask.to_numpy()]
    ds = xr.Dataset({variable_name: ("time", cleaned_values)})
    ds = ds.assign_coords(time=("time", cleaned_times))
    ds[variable_name].attrs.update({"units": units, **attrs})
    return ds


def save_dataset(ds: xr.Dataset, file_name: str) -> Path:
    target = DOWNLOAD_DIR / file_name
    ds.to_netcdf(target)
    return target


# ----------------------------------------------------------------------------
# Fetcher implementations
# ----------------------------------------------------------------------------

def fetch_gistemp(_: DataSource) -> Path:
    url = "https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv"
    df = pd.read_csv(url, skiprows=1, header=None)
    df.columns = df.iloc[0]
    df = df.iloc[1:]
    df = df.rename(columns={"Year": "year", "J-D": "annual_anomaly"})
    df = df[["year", "annual_anomaly"]]
    df = df[df["annual_anomaly"] != "***"]
    df["annual_anomaly"] = pd.to_numeric(df["annual_anomaly"], errors="coerce")
    ds = dataset_from_series(
        times=pd.to_datetime(df["year"].astype(int), format="%Y"),
        values=df["annual_anomaly"],
        variable_name="temperature_anomaly",
        units="°C",
        attrs={
            "long_name": "NASA GISTEMP global land-ocean temperature anomaly (12-month mean)",
            "source": "GISTEMP v4",
        },
    )
    return save_dataset(ds, "gistemp_global.nc")


def fetch_noaa_cag(_: DataSource) -> Path:
    url = (
        "https://www.ncdc.noaa.gov/cag/global/time-series/globe/land_ocean/12/12/1880-2024.json"
        "?base_prd=true&begbaseyear=1901&endbaseyear=2000"
    )
    payload = requests.get(url, timeout=60).json()
    data_points = payload.get("data", {})
    if not data_points:
        raise RuntimeError("NOAA Climate at a Glance API returned no data")

    rows: List[Tuple[pd.Timestamp, float]] = []
    for key, value in data_points.items():
        # keys are YYYYMM strings
        if len(key) == 4:
            ts = pd.to_datetime(key, format="%Y")
        elif len(key) == 6:
            ts = pd.to_datetime(key, format="%Y%m")
        elif len(key) == 8:
            ts = pd.to_datetime(key, format="%Y%m%d")
        else:
            raise ValueError(f"Unexpected NOAA CAG timestamp key: {key}")
        val = value.get("anomaly") if isinstance(value, dict) else value
        rows.append((ts, float(val)))
    rows.sort()
    times, values = zip(*rows)
    ds = dataset_from_series(
        times=times,
        values=values,
        variable_name="land_ocean_anomaly",
        units="°C",
        attrs={
            "long_name": "NOAA Climate at a Glance global land+ocean anomaly",
            "source": "NOAA CAG",
        },
    )
    return save_dataset(ds, "noaa_cag_global.nc")


def fetch_openmeteo_temp(source: DataSource) -> Path:
    params = {
        "latitude": source.kwargs["latitude"],
        "longitude": source.kwargs["longitude"],
        "start_date": source.kwargs.get("start_date", "2024-01-01"),
        "end_date": source.kwargs.get("end_date", "2024-06-30"),
        "daily": "temperature_2m_mean,temperature_2m_max,temperature_2m_min",
        "timezone": "UTC",
    }
    response = requests.get("https://archive-api.open-meteo.com/v1/archive", params=params, timeout=60).json()
    daily = response.get("daily")
    if not daily:
        raise RuntimeError("Open-Meteo archive API returned no daily block")

    times = pd.to_datetime(daily["time"])
    values = daily["temperature_2m_mean"]
    ds = dataset_from_series(
        times=times,
        values=values,
        variable_name="temperature_mean_2m",
        units="°C",
        attrs={
            "long_name": "Open-Meteo archive mean 2m temperature",
            "source": "Open-Meteo Archive",
        },
    )
    # add max/min as auxiliary variables
    ds["temperature_max_2m"] = ("time", pd.Series(daily["temperature_2m_max"]).astype(float).values)
    ds["temperature_min_2m"] = ("time", pd.Series(daily["temperature_2m_min"]).astype(float).values)
    return save_dataset(ds, f"{source.id}.nc")


def fetch_openmeteo_air(source: DataSource) -> Path:
    params = {
        "latitude": source.kwargs["latitude"],
        "longitude": source.kwargs["longitude"],
        "start_date": source.kwargs.get("start_date", "2024-01-01"),
        "end_date": source.kwargs.get("end_date", "2024-01-10"),
        "hourly": source.kwargs["pollutants"],
        "timezone": "UTC",
    }
    response = requests.get("https://air-quality-api.open-meteo.com/v1/air-quality", params=params, timeout=60).json()
    hourly = response.get("hourly")
    if not hourly:
        raise RuntimeError("Open-Meteo air-quality API returned no hourly block")

    times = pd.to_datetime(hourly["time"])
    pollutant = source.kwargs["pollutants"].split(",")[0]
    values = hourly[pollutant]
    ds = dataset_from_series(
        times=times,
        values=values,
        variable_name=pollutant,
        units=source.kwargs.get("units", "µg/m³"),
        attrs={
            "long_name": f"Open-Meteo air-quality {pollutant}",
            "source": "Open-Meteo Air Quality",
        },
    )
    return save_dataset(ds, f"{source.id}.nc")


FETCHERS: Dict[str, Callable[[DataSource], Path]] = {
    "gistemp": fetch_gistemp,
    "noaa_cag": fetch_noaa_cag,
    "openmeteo_temp": fetch_openmeteo_temp,
    "openmeteo_air": fetch_openmeteo_air,
}


DATA_SOURCES: List[DataSource] = [
    DataSource(
        id="gistemp_global",
        category="temperature",
        description="NASA GISTEMP global land-ocean mean temperature anomalies (D1 > GISTEMP)",
        fetcher="gistemp",
        kwargs={}
    ),
    DataSource(
        id="noaa_cag_global",
        category="temperature",
        description="NOAA Climate at a Glance global Land+Ocean anomalies",
        fetcher="noaa_cag",
        kwargs={}
    ),
    DataSource(
        id="openmeteo_bratislava_temp",
        category="temperature",
        description="Open-Meteo archive daily 2m temperature for Bratislava",
        fetcher="openmeteo_temp",
        kwargs={"latitude": 48.1486, "longitude": 17.1077, "start_date": "2024-01-01", "end_date": "2024-03-31"},
    ),
    DataSource(
        id="openmeteo_bratislava_pm25",
        category="air_quality",
        description="Open-Meteo air-quality hourly PM2.5 for Bratislava",
        fetcher="openmeteo_air",
        kwargs={"latitude": 48.1486, "longitude": 17.1077, "pollutants": "pm2_5"},
    ),
    DataSource(
        id="openmeteo_vienna_pm10",
        category="air_quality",
        description="Open-Meteo air-quality hourly PM10 for Vienna",
        fetcher="openmeteo_air",
        kwargs={"latitude": 48.2082, "longitude": 16.3738, "pollutants": "pm10"},
    ),
    DataSource(
        id="openmeteo_prague_o3",
        category="air_quality",
        description="Open-Meteo air-quality hourly ozone for Prague",
        fetcher="openmeteo_air",
        kwargs={"latitude": 50.0755, "longitude": 14.4378, "pollutants": "ozone", "units": "µg/m³"},
    ),
]


# ----------------------------------------------------------------------------
# Main orchestration
# ----------------------------------------------------------------------------

def ingest_sources() -> Tuple[EmbeddingPipeline, List[Dict[str, object]]]:
    console.rule("Downloading and processing open datasets")

    generator = EmbeddingGenerator(model_name=os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-base"))
    pipeline = EmbeddingPipeline(generator=generator)
    ingestion_stats: List[Dict[str, object]] = []

    for source in DATA_SOURCES:
        console.print(f"[bold]→ Processing {source.id}[/bold] :: {source.description}")
        fetcher = FETCHERS[source.fetcher]
        netcdf_path = fetcher(source)
        stats = pipeline.process_dataset(str(netcdf_path), source_id=source.id)
        stats.update({"source_id": source.id, "category": source.category, "description": source.description})
        if stats.get("num_embeddings", 0) == 0:
            raise RuntimeError(f"No embeddings created for source {source.id}")
        ingestion_stats.append(stats)
        console.print(
            f"   Stored {stats['num_embeddings']} embeddings covering variables: {', '.join(stats['variables'])}"
        )

    return pipeline, ingestion_stats


def summarize_ingestion(stats: List[Dict[str, object]]) -> None:
    table = Table(title="Ingestion Summary")
    table.add_column("Source ID", style="cyan")
    table.add_column("Category", style="magenta")
    table.add_column("Embeddings", justify="right")
    table.add_column("Variables")

    total_embeddings = 0
    for row in stats:
        table.add_row(
            row["source_id"],
            row["category"],
            str(row["num_embeddings"]),
            ", ".join(row["variables"]),
        )
        total_embeddings += row["num_embeddings"]

    console.print(table)
    console.print(f"[green]Total embeddings stored:[/green] {total_embeddings}")


def run_semantic_checks(pipeline: EmbeddingPipeline) -> List[Dict[str, object]]:
    console.rule("Semantic spot checks")
    searcher = SemanticSearcher(database=pipeline.database, generator=pipeline.generator)
    queries = [
        "global temperature anomaly 2023",
        "daily mean temperature in Bratislava",
        "pm2.5 pollution levels in Bratislava",
        "Vienna pm10 concentration",
        "ozone levels Prague winter",
    ]

    hits: List[Dict[str, object]] = []
    for query in queries:
        results = searcher.search(query, k=3)
        if not results:
            raise RuntimeError(f"No semantic hits for query '{query}'")
        top = results[0]
        console.print(f"[bold cyan]{query}[/bold cyan] → best match {top['id']} (sim={top['similarity']:.3f})")
        hits.append({"query": query, "best_id": top["id"], "similarity": float(top["similarity"])})

    return hits


def analyze_embeddings(pipeline: EmbeddingPipeline, semantic_hits: List[Dict[str, object]]) -> None:
    console.rule("Embedding quality overview")
    snapshot = pipeline.database.get(include=["embeddings", "metadatas", "documents", "ids"])
    ids = snapshot.get("ids") or []
    if not ids:
        console.print("[yellow]No embeddings found in current database snapshot.[/yellow]")
        return

    metadatas = snapshot.get("metadatas") or []
    embeddings = snapshot.get("embeddings") or []
    total = len(ids)
    console.print(f"[green]Embeddings stored:[/green] {total}")

    source_counts = Counter((meta or {}).get("source_id", "unknown") for meta in metadatas)
    variable_counts = Counter((meta or {}).get("variable", "unknown") for meta in metadatas)

    table = Table(title="Embeddings by Source", show_edge=True)
    table.add_column("Source", style="cyan")
    table.add_column("Count", style="green", justify="right")
    for source, count in source_counts.items():
        table.add_row(source, str(count))
    console.print(table)

    table_vars = Table(title="Variables Covered", show_edge=True)
    table_vars.add_column("Variable", style="magenta")
    table_vars.add_column("Count", style="green", justify="right")
    for variable, count in variable_counts.items():
        table_vars.add_row(variable, str(count))
    console.print(table_vars)

    if embeddings:
        emb_array = np.asarray(embeddings, dtype=np.float32)
        norms = np.linalg.norm(emb_array, axis=1)
        console.print(
            f"Vector norm mean {norms.mean():.3f} ± {norms.std():.3f} (min {norms.min():.3f}, max {norms.max():.3f})"
        )
        if emb_array.shape[0] > 1:
            sims = emb_array @ emb_array.T
            denom = np.linalg.norm(emb_array, axis=1, keepdims=True)
            denom = denom @ denom.T
            cosine = np.divide(sims, np.clip(denom, 1e-9, None))
            upper = cosine[np.triu_indices_from(cosine, k=1)]
            console.print(
                f"Pairwise cosine similarity mean {upper.mean():.3f} ± {upper.std():.3f}"
            )

    required_fields = ("source_id", "variable", "timestamp", "text")
    completeness = sum(
        all(meta.get(field) for field in required_fields)
        for meta in metadatas
        if isinstance(meta, dict)
    )
    completeness_pct = (completeness / total) * 100
    console.print(f"Metadata completeness: {completeness_pct:.1f}% ({completeness}/{total})")

    if semantic_hits:
        avg_sim = np.mean([hit["similarity"] for hit in semantic_hits])
        console.print(f"Average top-1 semantic similarity: {avg_sim:.3f}")
        if avg_sim < 0.8:
            console.print(
                "[yellow]Consider a larger embedding model (e.g., bge-large-en) for harder queries.[/yellow]"
            )
        else:
            console.print(
                "[green]Current embedding model provides strong semantic alignment for these queries.[/green]"
            )


def synthesize_answer(hits: List[Dict[str, object]]) -> Tuple[str, List[str]]:
    if not hits:
        return "No supporting context available.", []

    statements: List[str] = []
    references: List[str] = []
    for hit in hits:
        meta = hit.get("metadata") or {}
        variable = meta.get("variable", "variable")
        source = meta.get("source_id", "source")
        unit = meta.get("unit", "")
        mean = meta.get("stat_mean")
        latest = (meta.get("temporal_extent") or {}).get("end")
        if mean is not None and isinstance(mean, (int, float)):
            statements.append(
                f"{variable} from {source} averages {mean:.2f}{unit} (latest sample {latest or 'n/a'})."
            )
        else:
            statements.append(f"{variable} from {source} covers {meta.get('long_name', 'the dataset')}.")
        references.append(f"{source}:{variable}")
    return " ".join(statements), references


def run_rag_demo(pipeline: EmbeddingPipeline) -> None:
    console.rule("RAG demonstration")
    searcher = SemanticSearcher(database=pipeline.database, generator=pipeline.generator)
    questions = [
        "How are recent global land-ocean temperature anomalies evolving?",
        "What do Bratislava air-quality measurements show for PM2.5?",
        "Summarize particulate trends for Vienna and ozone readings for Prague.",
    ]

    for question in questions:
        hits = searcher.search(question, k=3)
        answer, references = synthesize_answer(hits)
        console.print(f"[bold]{question}[/bold]")
        console.print(f"{answer}")
        console.print(f"References: {', '.join(references) if references else 'n/a'}\n")


def main() -> None:
    pipeline, stats = ingest_sources()
    summarize_ingestion(stats)
    semantic_hits = run_semantic_checks(pipeline)
    analyze_embeddings(pipeline, semantic_hits)
    run_rag_demo(pipeline)
    console.print("\n[bold green]✅ Multi-source embedding test completed successfully[/bold green]")


if __name__ == "__main__":
    main()
