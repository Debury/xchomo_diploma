#!/usr/bin/env python3
"""
Standalone RAG evaluation script.

Sends all claims and questions to the RAG endpoint, captures full LLM responses,
and writes detailed results to a timestamped markdown report.

Usage:
    python tests/run_rag_evaluation.py
    python tests/run_rag_evaluation.py --url http://159.65.207.173:8000
    python tests/run_rag_evaluation.py --top-k 10 --no-reranker
    python tests/run_rag_evaluation.py --output docs/my_report.md
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

# ──────────────────────────────────────────────────────────────────────────────
# TEST DATA
# ──────────────────────────────────────────────────────────────────────────────

CLAIMS = [
    {
        "id": "C1",
        "claim": "Were 2023 and 2024 the warmest years on record, with 2024 breaching the 1.5°C threshold?",
        "validation": (
            "True. The Copernicus Climate Change Service (C3S), which operates the ERA5 dataset, "
            "officially confirmed that 2024 was the warmest year on record. ERA5 data shows the "
            "global average temperature for 2024 was 1.60°C above the pre-industrial (1850-1900) level, "
            "making it the first calendar year to exceed the 1.5°C limit set out in the Paris Agreement."
        ),
        "expected_datasets": ["ERA5", "C3S", "Copernicus"],
        "expected_keywords": ["temperature", "warm", "global", "surface"],
    },
    {
        "id": "C2",
        "claim": "Did European heatwaves in 2022 and 2023 exceed 40°C and feature extreme heat stress?",
        "validation": (
            "True. The European State of the Climate (ESOTC) reports for 2022 and 2023, which heavily "
            "utilize the E-OBS and ERA5 datasets, confirm this. The 2022 report notes that western "
            "Europe experienced temperatures roughly 10°C higher than typical summer maximums "
            "(surpassing 40°C in the UK for the first time). The 2023 report confirms record-breaking "
            "numbers of days with 'extreme heat stress' across Southern Europe."
        ),
        "expected_datasets": ["E-OBS", "ERA5"],
        "expected_keywords": ["temperature", "heat", "Europe"],
    },
    {
        "id": "C3",
        "claim": "Does IMERG show a trend of increasingly intense heavy rainfall events over the last decade?",
        "validation": (
            "True. NASA's GPM-IMERG scientific publications and recent validation studies confirm that "
            "IMERG's high-resolution data successfully captures the global shift toward more intense, "
            "short-duration extreme precipitation events, driven by a warmer atmosphere holding more moisture."
        ),
        "expected_datasets": ["IMERG", "GPM"],
        "expected_keywords": ["precipitation", "rainfall", "intense"],
    },
    {
        "id": "C4",
        "claim": "Were megadroughts exacerbated by extreme potential evapotranspiration, visible in SPEIbase?",
        "validation": (
            "True. The SPEI specifically factors in temperature and potential evapotranspiration (PET), "
            "unlike simpler indices like the SPI. Scientific consensus using SPEIbase confirms that "
            "recent multi-year droughts (like those in the Mediterranean and Horn of Africa) were "
            "classified as 'hot droughts,' driven largely by extreme PET."
        ),
        "expected_datasets": ["SPEI", "SPEIbase"],
        "expected_keywords": ["drought", "evapotranspiration", "SPEI", "precipitation"],
    },
    {
        "id": "C5",
        "claim": "Does GRACE data show accelerating ice sheet loss and global sea-level rise exceeding 4 mm/yr?",
        "validation": (
            "True. NASA's GRACE and GRACE-FO missions are the gold standard for measuring ice sheet mass "
            "loss via satellite gravimetry. The WMO State of the Global Climate 2024 report confirmed "
            "that the rate of global mean sea-level rise from 2014 to 2023 more than doubled compared to "
            "the first decade of satellite records, reaching approximately 4.77 mm per year."
        ),
        "expected_datasets": ["GRACE", "GRACE-FO"],
        "expected_keywords": ["ice", "sea level", "mass", "water", "gravimetry"],
    },
    {
        "id": "C6",
        "claim": "Did Mediterranean marine heatwaves reach 4°C to 5°C above average between 2022 and 2025?",
        "validation": (
            "True. Copernicus Marine Service and ESOTC data confirm exceptional marine heatwaves. "
            "SST anomalies in the Mediterranean and the North Atlantic frequently spiked to 4-5°C "
            "above the 1991-2020 climatological average, leading to severe ecological impacts."
        ),
        "expected_datasets": ["Copernicus Marine", "SST", "ERA5"],
        "expected_keywords": ["sea surface temperature", "SST", "marine", "Mediterranean", "temperature"],
    },
    {
        "id": "C7",
        "claim": "Has atmospheric CO2 consistently surpassed 420 ppm in the 2020s with growth rate exceeding 2.5 ppm per year?",
        "validation": (
            "True. CAMS and NOAA's Global Monitoring Laboratory confirm that the global surface average "
            "for CO2 permanently crossed the 420 ppm threshold in 2023/2024. The growth rate has "
            "consistently hovered between 2.5 and 3.0 ppm per year during the last decade."
        ),
        "expected_datasets": ["CAMS", "CO2", "Copernicus Atmosphere"],
        "expected_keywords": ["co2", "carbon", "aerosol", "atmosphere"],
    },
    {
        "id": "C8",
        "claim": "Do CAMS and MERRA-2 track significant Saharan dust intrusion anomalies into Europe?",
        "validation": (
            "True. Both CAMS (Copernicus) and MERRA-2 (NASA's aerosol reanalysis) are routinely cited "
            "in meteorological reports detailing massive, anomalous Saharan dust plumes that have "
            "repeatedly blanketed parts of Western and Southern Europe, severely degrading air quality."
        ),
        "expected_datasets": ["CAMS", "MERRA"],
        "expected_keywords": ["dust", "aerosol", "Sahara", "air quality"],
    },
]

QUESTIONS = [
    {
        "id": "Q1",
        "question": (
            "How do the anomalies in the SPEIbase drought index and ERA5 temperature data "
            "correlate with the Fire radiative power (Copernicus) and CAMS air pollution "
            "emissions during the 2023 Northern Hemisphere summer, particularly in Canada?"
        ),
        "valid_answer": (
            "In 2023, extreme negative anomalies in the SPEIbase (indicating severe, prolonged drought) "
            "and record-high surface temperatures in ERA5 over North America created unprecedented fuel "
            "aridity. This directly correlated with extreme Fire Radiative Power (FRP) recorded by "
            "Copernicus satellite monitoring. Consequently, CAMS tracked massive plumes of particulate "
            "matter and carbon emissions from these fires, severely degrading air quality across North "
            "America and tracking smoke plumes reaching all the way to Europe."
        ),
        "validation": (
            "True. CAMS and WMO State of the Global Climate 2023 confirm this exact sequence. "
            "CAMS reported that the 2023 Canadian wildfires generated a record-breaking 480 megatonnes "
            "of carbon emissions, directly fueled by the extreme heat and drought conditions."
        ),
        "expected_datasets": ["SPEI", "ERA5", "CAMS", "fire"],
        "expected_keywords": ["drought", "temperature", "fire", "emission", "Canada", "atmosphere"],
    },
    {
        "id": "Q2",
        "question": (
            "By analyzing IMERG precipitation rates alongside ERA5 Land soil moisture and "
            "JPL GRACE terrestrial water storage anomalies, what characterized the physical "
            "progression of the catastrophic 2022 Pakistan floods?"
        ),
        "valid_answer": (
            "IMERG satellite data captured relentless, heavily anomalous monsoon rainfall over several "
            "consecutive weeks in the summer of 2022. Because ERA5 Land reanalysis showed that regional "
            "soils were already at maximum saturation from earlier rains, the excess precipitation could "
            "not infiltrate the ground, resulting in massive surface runoff. Concurrently, JPL GRACE "
            "satellite gravimetry detected a historic, massive positive anomaly in terrestrial water "
            "storage, physically quantifying the immense volume of floodwater that accumulated across "
            "the Indus River basin."
        ),
        "validation": (
            "True. NASA Earth Observatory and peer-reviewed hydrological studies utilized this exact "
            "combination of data. NASA highlighted IMERG data showing rainfall anomalies exceeding "
            "400% of the average, while GRACE-FO data mapped the massive increase in groundwater "
            "and surface water mass that submerged nearly one-third of the country."
        ),
        "expected_datasets": ["IMERG", "ERA5", "GRACE"],
        "expected_keywords": ["precipitation", "flood", "water", "soil", "Pakistan"],
    },
]


# ──────────────────────────────────────────────────────────────────────────────
# EVALUATION HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def query_rag(base_url: str, question: str, top_k: int = 10,
              use_reranker: bool = True, timeout: int = 300) -> Dict[str, Any]:
    """Send a question to the RAG endpoint."""
    resp = requests.post(
        f"{base_url}/rag/query",
        json={
            "question": question,
            "top_k": top_k,
            "use_llm": True,
            "use_reranker": use_reranker,
            "timeout": 180,
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()


def score_keywords(text: str, keywords: List[str]) -> Dict[str, Any]:
    """Check keyword presence in text."""
    import unicodedata
    text_lower = text.lower()
    text_normalized = unicodedata.normalize("NFKD", text_lower)
    found = [kw for kw in keywords if kw.lower() in text_lower or kw.lower() in text_normalized]
    missing = [kw for kw in keywords if kw.lower() not in text_lower and kw.lower() not in text_normalized]
    return {
        "found": found,
        "missing": missing,
        "hit_rate": len(found) / len(keywords) if keywords else 0,
    }


def score_datasets(chunks: List[Dict], expected: List[str]) -> Dict[str, Any]:
    """Check if expected datasets appear in retrieved chunks.

    Uses alias groups so that e.g. ERA5 satisfies 'C3S' or 'Copernicus'.
    """
    # Alias groups — any member satisfies any other member
    _ALIAS_GROUPS = [
        {"era5", "c3s", "copernicus", "copernicus climate", "cerra", "era5-heat", "era5 land"},
        {"cams", "copernicus atmosphere"},
        {"copernicus marine", "cmems", "sst_med", "hadlsst", "sst"},
        {"grace", "grace-fo", "jpl grace", "jpl_grace"},
        {"imerg", "gpm", "gpm-imerg"},
        {"merra-2", "merra2", "merra", "merra2_aerosol"},
        {"e-obs", "eobs", "cy-obs"},
        {"spei", "speibase", "spei-gd", "spi"},
    ]

    all_text = " ".join(
        (c.get("text", "") + " " + json.dumps(c.get("metadata", {})))
        for c in chunks
    ).lower()

    found = []
    missing = []
    for ds in expected:
        ds_low = ds.lower()
        if ds_low in all_text:
            found.append(ds)
            continue
        # Check aliases: if any alias from the same group is present, count as found
        matched = False
        for group in _ALIAS_GROUPS:
            if ds_low in group:
                if any(alias in all_text for alias in group):
                    matched = True
                    break
        if matched:
            found.append(ds)
        else:
            missing.append(ds)
    return {
        "found": found,
        "missing": missing,
        "hit_rate": len(found) / len(expected) if expected else 0,
    }


def format_chunks_md(chunks: List[Dict], max_chunks: int = 5) -> List[str]:
    """Format retrieved chunks as markdown with actual data used."""
    lines = []
    for i, c in enumerate(chunks[:max_chunks]):
        score = c.get("score", 0)
        meta = c.get("metadata", {})
        ds = meta.get("dataset_name", meta.get("source_id", "?"))
        var = meta.get("variable", meta.get("hazard_type", "?"))
        text = c.get("text", "")
        # Truncate text to keep report readable
        text_short = text[:200].replace("|", "/").replace("\n", " ")
        if len(text) > 200:
            text_short += "..."
        lines.append(f"| {i+1} | {score:.4f} | {ds} | {var} | {text_short} |")
    return lines


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def run_evaluation(args) -> str:
    """Run the full evaluation and return markdown report."""
    base_url = args.url.rstrip("/")
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Check health
    try:
        h = requests.get(f"{base_url}/health", timeout=10)
        h.raise_for_status()
        health = h.json()
    except Exception as e:
        print(f"ERROR: API not reachable at {base_url}: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"API healthy at {base_url}. Running evaluation...")

    md = []
    md.append(f"# RAG Evaluation Report")
    md.append(f"")
    md.append(f"**Date:** {ts}")
    md.append(f"**API:** `{base_url}`")
    md.append(f"**top_k:** {args.top_k} | **reranker:** {not args.no_reranker}")
    md.append(f"")
    md.append(f"---")
    md.append(f"")

    all_results = []

    # ── CLAIMS ────────────────────────────────────────────────────────────
    md.append(f"## Claims (C1–C8)")
    md.append(f"")

    for claim in CLAIMS:
        cid = claim["id"]
        question = claim["claim"]
        print(f"  [{cid}] {question[:70]}...", end=" ", flush=True)

        t0 = time.time()
        try:
            result = query_rag(base_url, question, args.top_k, not args.no_reranker)
            elapsed = time.time() - t0
            error = None
        except Exception as e:
            elapsed = time.time() - t0
            result = {"answer": "", "chunks": [], "references": [], "llm_used": False}
            error = str(e)

        answer = result.get("answer", "")
        chunks = result.get("chunks", [])
        refs = result.get("references", [])
        llm_used = result.get("llm_used", False)
        reranker_used = result.get("reranker_used", False)
        search_ms = result.get("search_time_ms", 0)
        llm_ms = result.get("llm_time_ms", 0)

        ds_score = score_datasets(chunks, claim["expected_datasets"])
        kw_chunks = score_keywords(
            " ".join(c.get("text", "") + " " + json.dumps(c.get("metadata", {})) for c in chunks),
            claim["expected_keywords"],
        )
        kw_answer = score_keywords(answer, claim["expected_keywords"])

        status = "OK" if error is None else "ERROR"
        print(f"{status} ({elapsed:.1f}s, ds={ds_score['hit_rate']:.0%}, kw={kw_answer['hit_rate']:.0%})")

        all_results.append({
            "id": cid,
            "ds_hit": ds_score["hit_rate"],
            "kw_chunk_hit": kw_chunks["hit_rate"],
            "kw_answer_hit": kw_answer["hit_rate"],
            "llm_used": llm_used,
            "search_ms": search_ms,
            "llm_ms": llm_ms or 0,
            "elapsed": elapsed,
            "n_chunks": len(chunks),
        })

        # Write detailed MD
        md.append(f"### {cid}: {question}")
        md.append(f"")
        if error:
            md.append(f"> **ERROR:** `{error}`")
            md.append(f"")
            continue

        md.append(f"**Expected validation:** {claim['validation']}")
        md.append(f"")
        md.append(f"**LLM Answer** (llm={llm_used}, reranker={reranker_used}, "
                   f"search={search_ms:.0f}ms, llm={llm_ms:.0f}ms):")
        md.append(f"")
        md.append(f"> {answer}")
        md.append(f"")

        # Retrieved chunks table
        md.append(f"**Retrieved chunks ({len(chunks)}):**")
        md.append(f"")
        md.append(f"| # | Score | Dataset | Variable | Context Data |")
        md.append(f"|---|-------|---------|----------|--------------|")
        md.extend(format_chunks_md(chunks, max_chunks=10))
        md.append(f"")

        # References
        if refs:
            md.append(f"**References:** {', '.join(refs[:8])}")
            md.append(f"")

        # Scores
        md.append(f"**Scores:**")
        md.append(f"- Dataset relevance: {ds_score['hit_rate']:.0%} "
                   f"(found: {ds_score['found']}, missing: {ds_score['missing']})")
        md.append(f"- Keyword in chunks: {kw_chunks['hit_rate']:.0%} "
                   f"(found: {kw_chunks['found']}, missing: {kw_chunks['missing']})")
        md.append(f"- Keyword in answer: {kw_answer['hit_rate']:.0%} "
                   f"(found: {kw_answer['found']}, missing: {kw_answer['missing']})")
        md.append(f"")
        md.append(f"---")
        md.append(f"")

    # ── QUESTIONS ─────────────────────────────────────────────────────────
    md.append(f"## Complex Questions (Q1–Q2)")
    md.append(f"")

    for q in QUESTIONS:
        qid = q["id"]
        question = q["question"]
        print(f"  [{qid}] {question[:70]}...", end=" ", flush=True)

        t0 = time.time()
        try:
            result = query_rag(base_url, question, args.top_k, not args.no_reranker)
            elapsed = time.time() - t0
            error = None
        except Exception as e:
            elapsed = time.time() - t0
            result = {"answer": "", "chunks": [], "references": [], "llm_used": False}
            error = str(e)

        answer = result.get("answer", "")
        chunks = result.get("chunks", [])
        refs = result.get("references", [])
        llm_used = result.get("llm_used", False)
        reranker_used = result.get("reranker_used", False)
        search_ms = result.get("search_time_ms", 0)
        llm_ms = result.get("llm_time_ms", 0)

        ds_score = score_datasets(chunks, q["expected_datasets"])
        kw_chunks = score_keywords(
            " ".join(c.get("text", "") + " " + json.dumps(c.get("metadata", {})) for c in chunks),
            q["expected_keywords"],
        )
        kw_answer = score_keywords(answer, q["expected_keywords"])

        status = "OK" if error is None else "ERROR"
        print(f"{status} ({elapsed:.1f}s, ds={ds_score['hit_rate']:.0%}, kw={kw_answer['hit_rate']:.0%})")

        all_results.append({
            "id": qid,
            "ds_hit": ds_score["hit_rate"],
            "kw_chunk_hit": kw_chunks["hit_rate"],
            "kw_answer_hit": kw_answer["hit_rate"],
            "llm_used": llm_used,
            "search_ms": search_ms,
            "llm_ms": llm_ms or 0,
            "elapsed": elapsed,
            "n_chunks": len(chunks),
        })

        md.append(f"### {qid}: {question}")
        md.append(f"")
        if error:
            md.append(f"> **ERROR:** `{error}`")
            md.append(f"")
            continue

        md.append(f"**Expected answer:** {q['valid_answer']}")
        md.append(f"")
        md.append(f"**Validation:** {q['validation']}")
        md.append(f"")
        md.append(f"**LLM Answer** (llm={llm_used}, reranker={reranker_used}, "
                   f"search={search_ms:.0f}ms, llm={llm_ms:.0f}ms):")
        md.append(f"")
        md.append(f"> {answer}")
        md.append(f"")

        md.append(f"**Retrieved chunks ({len(chunks)}):**")
        md.append(f"")
        md.append(f"| # | Score | Dataset | Variable | Context Data |")
        md.append(f"|---|-------|---------|----------|--------------|")
        md.extend(format_chunks_md(chunks, max_chunks=10))
        md.append(f"")

        if refs:
            md.append(f"**References:** {', '.join(refs[:8])}")
            md.append(f"")

        md.append(f"**Scores:**")
        md.append(f"- Dataset relevance: {ds_score['hit_rate']:.0%} "
                   f"(found: {ds_score['found']}, missing: {ds_score['missing']})")
        md.append(f"- Keyword in chunks: {kw_chunks['hit_rate']:.0%} "
                   f"(found: {kw_chunks['found']}, missing: {kw_chunks['missing']})")
        md.append(f"- Keyword in answer: {kw_answer['hit_rate']:.0%} "
                   f"(found: {kw_answer['found']}, missing: {kw_answer['missing']})")
        md.append(f"")
        md.append(f"---")
        md.append(f"")

    # ── SUMMARY TABLE ─────────────────────────────────────────────────────
    md.append(f"## Summary")
    md.append(f"")
    md.append(f"| ID | Dataset% | KW Chunks% | KW Answer% | LLM | Chunks | Search ms | LLM ms | Total s |")
    md.append(f"|----|----------|------------|------------|-----|--------|-----------|--------|---------|")

    for r in all_results:
        llm_str = "Yes" if r["llm_used"] else "No"
        md.append(
            f"| {r['id']} | {r['ds_hit']:.0%} | {r['kw_chunk_hit']:.0%} | "
            f"{r['kw_answer_hit']:.0%} | {llm_str} | {r['n_chunks']} | "
            f"{r['search_ms']:.0f} | {r['llm_ms']:.0f} | {r['elapsed']:.1f} |"
        )

    md.append(f"")

    # Averages
    n = len(all_results)
    if n > 0:
        avg_ds = sum(r["ds_hit"] for r in all_results) / n
        avg_kw_c = sum(r["kw_chunk_hit"] for r in all_results) / n
        avg_kw_a = sum(r["kw_answer_hit"] for r in all_results) / n
        avg_search = sum(r["search_ms"] for r in all_results) / n
        avg_llm = sum(r["llm_ms"] for r in all_results) / n
        avg_elapsed = sum(r["elapsed"] for r in all_results) / n
        llm_count = sum(1 for r in all_results if r["llm_used"])

        md.append(f"**Averages:**")
        md.append(f"- Dataset relevance: **{avg_ds:.0%}**")
        md.append(f"- Keyword in chunks: **{avg_kw_c:.0%}**")
        md.append(f"- Keyword in answer: **{avg_kw_a:.0%}**")
        md.append(f"- LLM used: {llm_count}/{n}")
        md.append(f"- Avg search time: {avg_search:.0f} ms")
        md.append(f"- Avg LLM time: {avg_llm:.0f} ms")
        md.append(f"- Avg total time: {avg_elapsed:.1f} s")

        # Overall weighted score
        overall = avg_ds * 0.35 + avg_kw_c * 0.30 + avg_kw_a * 0.35
        md.append(f"- **Overall score: {overall:.0%}**")
    md.append(f"")

    return "\n".join(md)


def main():
    parser = argparse.ArgumentParser(description="RAG Evaluation — detailed report with full LLM answers")
    parser.add_argument("--url", default="http://localhost:8000", help="Base API URL")
    parser.add_argument("--top-k", type=int, default=10, help="Number of chunks to retrieve")
    parser.add_argument("--no-reranker", action="store_true", help="Disable cross-encoder reranker")
    parser.add_argument("--output", default=None, help="Output markdown path (default: docs/rag_eval_<timestamp>.md)")
    args = parser.parse_args()

    report = run_evaluation(args)

    # Determine output path
    if args.output:
        out_path = Path(args.output)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = Path("docs") / f"rag_eval_{ts}.md"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    print(f"\nReport written to: {out_path}")


if __name__ == "__main__":
    main()
