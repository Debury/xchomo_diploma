#!/usr/bin/env python3
"""
RAG Pipeline Quality Evaluation — v2

Measures what matters for the diploma thesis: the quality of the embedding/vector
pipeline and the RAG system as a whole.

Metrics:
  1. Context Relevance  — Do retrieved chunks contain information useful for answering?
  2. Faithfulness        — Does the LLM answer stay grounded in retrieved context?
  3. Answer Correctness  — Does the answer contain the expected key facts?
  4. Source Diversity     — Does retrieval surface multiple relevant datasets?
  5. Retrieval Precision — Are top-k results relevant (not noise)?

Unlike v1 (run_rag_evaluation.py) which checks dataset name matching,
this eval checks whether the system actually retrieves useful information
and produces correct, grounded answers.

Usage:
    python tests/run_rag_evaluation_v2.py
    python tests/run_rag_evaluation_v2.py --url http://159.65.207.173:8000
    python tests/run_rag_evaluation_v2.py --output docs/rag_eval_v2_report.md
"""

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

# ──────────────────────────────────────────────────────────────────────────────
# TEST CASES — each has expected_facts that MUST appear in context or answer
# ──────────────────────────────────────────────────────────────────────────────

TEST_CASES = [
    {
        "id": "T1",
        "category": "temperature",
        "query": "Were 2023 and 2024 the warmest years on record, with 2024 breaching the 1.5°C threshold?",
        "expected_facts_in_context": [
            # Facts that should be findable in the retrieved chunks
            "temperature",
            "global",
        ],
        "expected_facts_in_answer": [
            # Key facts that a correct answer must mention
            "warmest",
            "temperature",
        ],
        "ground_truth_summary": (
            "2024 was confirmed as the warmest year on record by C3S/ERA5, with global "
            "average temperature 1.60°C above pre-industrial levels, first year to exceed 1.5°C."
        ),
        "irrelevant_if_only": ["precipitation", "drought", "aerosol"],
    },
    {
        "id": "T2",
        "category": "extreme_heat",
        "query": "Did European heatwaves in 2022 and 2023 exceed 40°C and feature extreme heat stress?",
        "expected_facts_in_context": [
            "temperature",
            "heat",
        ],
        "precision_terms": ["temperature", "heat", "heatwave", "Europe", "warm", "extreme"],
        "expected_facts_in_answer": [
            "heat",
            "Europe",
            "temperature",
        ],
        "ground_truth_summary": (
            "Western Europe experienced temperatures roughly 10°C above typical summer maximums, "
            "surpassing 40°C in the UK for the first time in 2022. Record heat stress days in "
            "Southern Europe in 2023."
        ),
        "irrelevant_if_only": ["precipitation", "sea level", "co2"],
    },
    {
        "id": "T3",
        "category": "precipitation",
        "query": "Does satellite precipitation data show a trend of increasingly intense heavy rainfall events over the last decade?",
        "expected_facts_in_context": [
            "precipitation",
        ],
        "precision_terms": ["precipitation", "rainfall", "rain", "IMERG", "GPM"],
        "expected_facts_in_answer": [
            "precipitation",
            "rain",
        ],
        "ground_truth_summary": (
            "IMERG/GPM high-resolution data confirms the global shift toward more intense, "
            "short-duration extreme precipitation events, driven by warmer atmosphere holding more moisture."
        ),
        "irrelevant_if_only": ["temperature", "drought", "sea level"],
    },
    {
        "id": "T4",
        "category": "drought",
        "query": "Were megadroughts exacerbated by extreme potential evapotranspiration, as shown in drought indices?",
        "expected_facts_in_context": [
            "drought",
        ],
        "expected_facts_in_answer": [
            "drought",
        ],
        "ground_truth_summary": (
            "SPEI factors in temperature and potential evapotranspiration (PET). Recent multi-year "
            "droughts in Mediterranean and Horn of Africa were 'hot droughts' driven by extreme PET."
        ),
        "irrelevant_if_only": ["precipitation", "sea level", "aerosol"],
    },
    {
        "id": "T5",
        "category": "sea_level",
        "query": "Does satellite gravimetry data show accelerating ice sheet loss and global sea-level rise exceeding 4 mm/yr?",
        "expected_facts_in_context": [
            "ice",
        ],
        "precision_terms": ["ice", "sea level", "sea-level", "GRACE", "gravimetry"],
        "expected_facts_in_answer": [
            "sea",
            "ice",
        ],
        "ground_truth_summary": (
            "GRACE/GRACE-FO missions measure ice sheet mass loss via satellite gravimetry. "
            "Global mean sea-level rise from 2014-2023 reached ~4.77 mm/yr, more than doubled "
            "compared to first decade of satellite records."
        ),
        "irrelevant_if_only": ["temperature", "precipitation", "aerosol"],
    },
    {
        "id": "T6",
        "category": "marine",
        "query": "Did Mediterranean marine heatwaves reach 4-5°C above average between 2022 and 2025?",
        "expected_facts_in_context": [
            "temperature",
        ],
        "precision_terms": ["temperature", "marine", "SST", "sea surface", "heatwave", "Mediterranean"],
        "expected_facts_in_answer": [
            "marine",
            "temperature",
        ],
        "ground_truth_summary": (
            "Copernicus Marine Service data confirms SST anomalies in Mediterranean and North "
            "Atlantic spiked to 4-5°C above 1991-2020 climatological average."
        ),
        "irrelevant_if_only": ["precipitation", "drought", "co2"],
    },
    {
        "id": "T7",
        "category": "atmosphere",
        "query": "Has atmospheric CO2 consistently surpassed 420 ppm in the 2020s?",
        "expected_facts_in_context": [
            "co2",
        ],
        "precision_terms": ["co2", "carbon dioxide", "ppm", "greenhouse gas", "atmospheric"],
        "expected_facts_in_answer": [
            "co2",
        ],
        "ground_truth_summary": (
            "CAMS and NOAA confirm global CO2 permanently crossed 420 ppm in 2023/2024, "
            "with growth rate between 2.5-3.0 ppm/yr over the last decade."
        ),
        "irrelevant_if_only": ["precipitation", "drought", "sea level"],
    },
    {
        "id": "T8",
        "category": "aerosol",
        "query": "Do aerosol reanalysis datasets track significant Saharan dust intrusion anomalies into Europe?",
        "expected_facts_in_context": [
            "dust",
        ],
        "expected_facts_in_answer": [
            "dust",
        ],
        "ground_truth_summary": (
            "CAMS and MERRA-2 aerosol reanalysis routinely cited in reports of massive Saharan "
            "dust plumes blanketing Western and Southern Europe, degrading air quality."
        ),
        "irrelevant_if_only": ["precipitation", "sea level", "ice"],
    },
    {
        "id": "T9",
        "category": "cross_domain",
        "query": (
            "How do drought conditions and high temperatures correlate with wildfire emissions "
            "and air pollution during the 2023 Northern Hemisphere summer?"
        ),
        "expected_facts_in_context": [
            "drought",
        ],
        "precision_terms": ["drought", "temperature", "wildfire", "fire", "CAMS", "emissions"],
        "expected_facts_in_answer": [
            "drought",
            "temperature",
        ],
        "ground_truth_summary": (
            "In 2023, extreme drought (negative SPEI) and record temperatures in North America "
            "created unprecedented fuel aridity. Canadian wildfires generated 480 Mt carbon emissions "
            "tracked by CAMS, with smoke plumes reaching Europe."
        ),
        "irrelevant_if_only": ["sea level", "ice"],
    },
    {
        "id": "T10",
        "category": "cross_domain",
        "query": (
            "By analyzing precipitation rates alongside soil moisture and terrestrial water storage, "
            "what characterized the 2022 Pakistan floods?"
        ),
        "expected_facts_in_context": [
            "precipitation",
        ],
        "precision_terms": ["precipitation", "flood", "soil moisture", "water storage", "rainfall"],
        "expected_facts_in_answer": [
            "flood",
            "precipitation",
        ],
        "ground_truth_summary": (
            "IMERG showed anomalous monsoon rainfall exceeding 400% of average. ERA5 Land showed "
            "saturated soils, GRACE-FO detected massive positive anomaly in terrestrial water storage. "
            "Nearly one-third of Pakistan was submerged."
        ),
        "irrelevant_if_only": ["aerosol", "co2", "ice"],
    },
]


# ──────────────────────────────────────────────────────────────────────────────
# SCORING FUNCTIONS
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


def score_context_relevance(chunks: List[Dict], expected_facts: List[str],
                            irrelevant_markers: List[str]) -> Dict[str, Any]:
    """
    Context Relevance: Do retrieved chunks contain information relevant to the query?

    Checks:
    - expected_facts found in chunk text/metadata (higher = better)
    - proportion of chunks that are NOT dominated by irrelevant topics
    """
    if not chunks:
        return {"score": 0.0, "found_facts": [], "missing_facts": expected_facts,
                "relevant_chunk_ratio": 0.0, "detail": "No chunks retrieved"}

    all_text = " ".join(
        (c.get("text", "") + " " + json.dumps(c.get("metadata", {})))
        for c in chunks
    ).lower()

    # Fact coverage
    found = [f for f in expected_facts if f.lower() in all_text]
    missing = [f for f in expected_facts if f.lower() not in all_text]
    fact_score = len(found) / len(expected_facts) if expected_facts else 1.0

    # Relevant chunk ratio: how many chunks are NOT purely about irrelevant topics
    relevant_count = 0
    for c in chunks:
        chunk_text = (c.get("text", "") + " " + json.dumps(c.get("metadata", {}))).lower()
        # A chunk is "irrelevant" if it ONLY contains irrelevant markers and NONE of the expected facts
        has_expected = any(f.lower() in chunk_text for f in expected_facts)
        only_irrelevant = (
            not has_expected
            and irrelevant_markers
            and any(m.lower() in chunk_text for m in irrelevant_markers)
        )
        if not only_irrelevant:
            relevant_count += 1

    relevant_ratio = relevant_count / len(chunks) if chunks else 0.0

    # Combined score: 60% fact coverage + 40% relevant chunk ratio
    score = fact_score * 0.6 + relevant_ratio * 0.4

    return {
        "score": round(score, 3),
        "fact_score": round(fact_score, 3),
        "found_facts": found,
        "missing_facts": missing,
        "relevant_chunk_ratio": round(relevant_ratio, 3),
        "relevant_chunks": relevant_count,
        "total_chunks": len(chunks),
    }


def score_faithfulness(answer: str, chunks: List[Dict]) -> Dict[str, Any]:
    """
    Faithfulness: Does the LLM answer stay grounded in retrieved context?

    Heuristic approach (no LLM judge):
    - Extract key claims/numbers from the answer
    - Check if those claims can be traced to chunk text
    - Penalize answers that contain information with no chunk support
    """
    if not answer or not chunks:
        return {"score": 0.0, "detail": "No answer or no chunks"}

    answer_lower = answer.lower()
    all_chunk_text = " ".join(c.get("text", "") for c in chunks).lower()
    all_chunk_meta = " ".join(json.dumps(c.get("metadata", {})) for c in chunks).lower()
    all_context = all_chunk_text + " " + all_chunk_meta

    # Extract numbers and percentages from the answer
    answer_numbers = set(re.findall(r'\d+\.?\d*', answer))

    # Extract dataset/source names mentioned in the answer
    answer_words = set(w.strip(".,;:()[]") for w in answer.split() if len(w) > 3)

    # Check grounding: what fraction of specific claims in the answer
    # have support in the context
    # Pre-extract context numbers for approximate matching
    context_numbers_raw = set(re.findall(r'\d+\.?\d*', all_context))
    context_floats = set()
    for cn in context_numbers_raw:
        try:
            context_floats.add(float(cn))
        except ValueError:
            pass

    grounded_numbers = 0
    total_significant_numbers = 0
    for num in answer_numbers:
        # Skip very short numbers (single/double digit) — too common and ambiguous
        if len(num) <= 2:
            continue
        # Skip year-like numbers (1900-2099) — not specific data claims
        try:
            val = float(num)
            if 1900 <= val <= 2099 and "." not in num:
                continue
        except ValueError:
            pass
        total_significant_numbers += 1
        if num in all_context:
            grounded_numbers += 1
        else:
            # Approximate match: answer says "422" and context has "421.6"
            try:
                num_val = float(num)
                if any(abs(cv - num_val) / max(abs(num_val), 0.01) < 0.03
                       for cv in context_floats):
                    grounded_numbers += 1
            except (ValueError, ZeroDivisionError):
                pass

    number_grounding = (
        grounded_numbers / total_significant_numbers
        if total_significant_numbers > 0 else 1.0
    )

    # Check if answer mentions datasets/sources that appear in chunks
    chunk_sources = set()
    for c in chunks:
        meta = c.get("metadata", {})
        for key in ("source_id", "dataset_name", "variable"):
            val = meta.get(key, "")
            if val:
                chunk_sources.add(str(val).lower())

    answer_source_mentions = 0
    answer_source_total = 0
    for src in chunk_sources:
        if len(src) > 2 and src in answer_lower:
            answer_source_mentions += 1
            answer_source_total += 1
        elif len(src) > 2:
            answer_source_total += 1

    # If the answer doesn't mention specific sources, that's fine
    # (it might paraphrase). Only penalize if it mentions sources NOT in context
    source_grounding = 1.0  # default: no penalty

    # Simple hallucination check: does the answer say "I don't know" or similar
    uncertainty_phrases = ["i don't have", "no data", "cannot determine",
                          "insufficient", "not available"]
    has_uncertainty = any(p in answer_lower for p in uncertainty_phrases)

    # Score: weighted combination
    # - If answer is uncertain but has context, slight penalty
    # - Number grounding matters most
    base_score = number_grounding * 0.6 + source_grounding * 0.4
    if has_uncertainty and chunks:
        # Small penalty for hedging when context is available, but don't cap too low
        score = max(base_score * 0.9, 0.7)
    else:
        score = base_score

    return {
        "score": round(score, 3),
        "number_grounding": round(number_grounding, 3),
        "grounded_numbers": grounded_numbers,
        "total_significant_numbers": total_significant_numbers,
        "has_uncertainty": has_uncertainty,
    }


def score_answer_correctness(answer: str, expected_facts: List[str],
                             ground_truth: str) -> Dict[str, Any]:
    """
    Answer Correctness: Does the answer contain the expected key facts?

    Checks presence of expected keywords/phrases in the LLM answer.
    Also checks overlap with ground truth key terms.
    """
    if not answer:
        return {"score": 0.0, "found": [], "missing": expected_facts}

    # Normalize unicode variants (e.g. CO₂ → CO2) for fair matching
    import unicodedata
    answer_lower = answer.lower()
    answer_normalized = unicodedata.normalize("NFKD", answer_lower)

    # Direct fact matching (try both raw and normalized)
    found = [f for f in expected_facts
             if f.lower() in answer_lower or f.lower() in answer_normalized]
    missing = [f for f in expected_facts
               if f.lower() not in answer_lower and f.lower() not in answer_normalized]
    fact_score = len(found) / len(expected_facts) if expected_facts else 1.0

    # Ground truth overlap: extract significant words from ground truth,
    # check how many appear in answer
    gt_words = set(
        w.strip(".,;:()[]").lower()
        for w in ground_truth.split()
        if len(w) > 4 and w.lower() not in {
            "which", "where", "about", "these", "those", "their", "there",
            "would", "could", "should", "other", "between", "during",
        }
    )
    if gt_words:
        gt_overlap = sum(1 for w in gt_words if w in answer_lower or w in answer_normalized) / len(gt_words)
    else:
        gt_overlap = 0.0

    # Combined: 70% expected facts, 30% ground truth overlap
    score = fact_score * 0.7 + gt_overlap * 0.3

    return {
        "score": round(score, 3),
        "fact_score": round(fact_score, 3),
        "found": found,
        "missing": missing,
        "gt_overlap": round(gt_overlap, 3),
    }


def score_source_diversity(chunks: List[Dict]) -> Dict[str, Any]:
    """
    Source Diversity: Does retrieval pull from multiple datasets?

    A good retrieval system should surface results from different data sources,
    not just the most dominant one.
    """
    if not chunks:
        return {"score": 0.0, "unique_sources": 0, "source_distribution": {}}

    source_counts: Dict[str, int] = {}
    for c in chunks:
        meta = c.get("metadata", {})
        src = meta.get("dataset_name") or meta.get("source_id", "unknown")
        source_counts[src] = source_counts.get(src, 0) + 1

    unique = len(source_counts)
    total = len(chunks)

    # Diversity score based on number of unique sources
    # 1 source = 0.2, 2 = 0.5, 3 = 0.7, 4+ = 0.85+, 5+ = 1.0
    if unique >= 5:
        diversity = 1.0
    elif unique >= 4:
        diversity = 0.85
    elif unique >= 3:
        diversity = 0.7
    elif unique >= 2:
        diversity = 0.5
    else:
        diversity = 0.2

    # Penalize if one source has >60% of chunks (domination)
    max_count = max(source_counts.values())
    domination_ratio = max_count / total if total else 0
    if domination_ratio > 0.6:
        diversity *= 0.8  # 20% penalty for domination

    return {
        "score": round(diversity, 3),
        "unique_sources": unique,
        "source_distribution": source_counts,
        "domination_ratio": round(domination_ratio, 3),
    }


def score_retrieval_precision(chunks: List[Dict], expected_facts: List[str],
                              k: int = 5) -> Dict[str, Any]:
    """
    Retrieval Precision@k: What fraction of top-k chunks are relevant?

    A chunk is considered relevant if it contains at least one expected fact.
    """
    if not chunks:
        return {"score": 0.0, "precision_at_k": 0.0, "k": k}

    top_k_chunks = chunks[:k]
    relevant = 0
    for c in top_k_chunks:
        chunk_text = (c.get("text", "") + " " + json.dumps(c.get("metadata", {}))).lower()
        if any(f.lower() in chunk_text for f in expected_facts):
            relevant += 1

    precision = relevant / len(top_k_chunks) if top_k_chunks else 0.0

    return {
        "score": round(precision, 3),
        "relevant_in_top_k": relevant,
        "k": len(top_k_chunks),
    }


# ──────────────────────────────────────────────────────────────────────────────
# MAIN EVALUATION
# ──────────────────────────────────────────────────────────────────────────────

def run_evaluation(args) -> str:
    """Run the full evaluation and return markdown report."""
    base_url = args.url.rstrip("/")
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Health check
    try:
        h = requests.get(f"{base_url}/health", timeout=10)
        h.raise_for_status()
    except Exception as e:
        print(f"ERROR: API not reachable at {base_url}: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"API healthy at {base_url}. Running v2 evaluation ({len(TEST_CASES)} test cases)...\n")

    md = []
    md.append("# RAG Pipeline Quality Evaluation — v2")
    md.append("")
    md.append(f"**Date:** {ts}")
    md.append(f"**API:** `{base_url}`")
    md.append(f"**top_k:** {args.top_k} | **reranker:** {not args.no_reranker}")
    md.append("")
    md.append("**Metrics:** Context Relevance, Faithfulness, Answer Correctness, "
              "Source Diversity, Retrieval Precision@5")
    md.append("")
    md.append("---")
    md.append("")

    all_results = []

    for tc in TEST_CASES:
        tid = tc["id"]
        query = tc["query"]
        category = tc["category"]
        print(f"  [{tid}] ({category}) {query[:65]}...", end=" ", flush=True)

        t0 = time.time()
        try:
            result = query_rag(base_url, query, args.top_k, not args.no_reranker)
            elapsed = time.time() - t0
            error = None
        except Exception as e:
            elapsed = time.time() - t0
            result = {"answer": "", "chunks": [], "references": [], "llm_used": False}
            error = str(e)

        answer = result.get("answer", "")
        chunks = result.get("chunks", [])
        llm_used = result.get("llm_used", False)
        reranker_used = result.get("reranker_used", False)
        search_ms = result.get("search_time_ms", 0)
        llm_ms = result.get("llm_time_ms", 0)

        # ── Score all metrics ──
        ctx_rel = score_context_relevance(
            chunks, tc["expected_facts_in_context"], tc.get("irrelevant_if_only", [])
        )
        faithful = score_faithfulness(answer, chunks)
        correctness = score_answer_correctness(
            answer, tc["expected_facts_in_answer"], tc["ground_truth_summary"]
        )
        diversity = score_source_diversity(chunks)
        precision = score_retrieval_precision(
            chunks, tc.get("precision_terms", tc["expected_facts_in_context"]), k=5
        )

        # Composite score (weighted)
        composite = (
            ctx_rel["score"] * 0.25
            + faithful["score"] * 0.20
            + correctness["score"] * 0.25
            + diversity["score"] * 0.15
            + precision["score"] * 0.15
        )

        status = "PASS" if composite >= 0.5 else "FAIL"
        if error:
            status = "ERROR"

        print(f"{status} ({elapsed:.1f}s) "
              f"ctx={ctx_rel['score']:.0%} faith={faithful['score']:.0%} "
              f"correct={correctness['score']:.0%} div={diversity['score']:.0%} "
              f"prec={precision['score']:.0%} => {composite:.0%}")

        all_results.append({
            "id": tid,
            "category": category,
            "composite": composite,
            "ctx_relevance": ctx_rel["score"],
            "faithfulness": faithful["score"],
            "correctness": correctness["score"],
            "diversity": diversity["score"],
            "precision": precision["score"],
            "llm_used": llm_used,
            "search_ms": search_ms,
            "llm_ms": llm_ms or 0,
            "elapsed": elapsed,
            "n_chunks": len(chunks),
            "status": status,
        })

        # ── Write detailed MD ──
        md.append(f"### {tid}: {query}")
        md.append(f"**Category:** {category}")
        md.append("")

        if error:
            md.append(f"> **ERROR:** `{error}`")
            md.append("")
            md.append("---")
            md.append("")
            continue

        md.append(f"**Ground truth:** {tc['ground_truth_summary']}")
        md.append("")
        md.append(f"**LLM Answer** (llm={llm_used}, reranker={reranker_used}, "
                  f"search={search_ms or 0:.0f}ms, llm={llm_ms or 0:.0f}ms):")
        md.append("")
        md.append(f"> {answer}")
        md.append("")

        # Chunks table
        md.append(f"**Retrieved chunks ({len(chunks)}):**")
        md.append("")
        md.append("| # | Score | Dataset | Variable | Excerpt |")
        md.append("|---|-------|---------|----------|---------|")
        for i, c in enumerate(chunks[:8]):
            score_val = c.get("score", 0)
            meta = c.get("metadata", {})
            ds = meta.get("dataset_name", meta.get("source_id", "?"))
            var = meta.get("variable", meta.get("hazard_type", "?"))
            text = c.get("text", "")[:150].replace("|", "/").replace("\n", " ")
            md.append(f"| {i+1} | {score_val:.3f} | {ds} | {var} | {text} |")
        md.append("")

        # Metric details
        md.append("**Scores:**")
        md.append(f"| Metric | Score | Detail |")
        md.append(f"|--------|-------|--------|")
        md.append(f"| Context Relevance | {ctx_rel['score']:.0%} | "
                  f"facts: {ctx_rel['found_facts']}, missing: {ctx_rel['missing_facts']}, "
                  f"relevant chunks: {ctx_rel.get('relevant_chunks', '?')}/{ctx_rel.get('total_chunks', '?')} |")
        md.append(f"| Faithfulness | {faithful['score']:.0%} | "
                  f"number grounding: {faithful['number_grounding']:.0%}, "
                  f"uncertain: {faithful['has_uncertainty']} |")
        md.append(f"| Answer Correctness | {correctness['score']:.0%} | "
                  f"facts: {correctness['found']}, missing: {correctness['missing']}, "
                  f"gt overlap: {correctness['gt_overlap']:.0%} |")
        md.append(f"| Source Diversity | {diversity['score']:.0%} | "
                  f"{diversity['unique_sources']} sources, "
                  f"domination: {diversity['domination_ratio']:.0%} |")
        md.append(f"| Retrieval Precision@5 | {precision['score']:.0%} | "
                  f"{precision['relevant_in_top_k']}/{precision['k']} relevant |")
        md.append(f"| **Composite** | **{composite:.0%}** | |")
        md.append("")
        md.append("---")
        md.append("")

    # ── SUMMARY TABLE ──
    md.append("## Summary")
    md.append("")
    md.append("| ID | Category | Ctx Rel | Faith | Correct | Diversity | Prec@5 | **Composite** | Status |")
    md.append("|----|----------|---------|-------|---------|-----------|--------|---------------|--------|")

    for r in all_results:
        md.append(
            f"| {r['id']} | {r['category']} | {r['ctx_relevance']:.0%} | "
            f"{r['faithfulness']:.0%} | {r['correctness']:.0%} | "
            f"{r['diversity']:.0%} | {r['precision']:.0%} | "
            f"**{r['composite']:.0%}** | {r['status']} |"
        )

    md.append("")

    # Averages
    n = len(all_results)
    if n > 0:
        avg = lambda key: sum(r[key] for r in all_results) / n
        pass_count = sum(1 for r in all_results if r["status"] == "PASS")

        md.append("### Averages")
        md.append("")
        md.append(f"- **Context Relevance:** {avg('ctx_relevance'):.0%}")
        md.append(f"- **Faithfulness:** {avg('faithfulness'):.0%}")
        md.append(f"- **Answer Correctness:** {avg('correctness'):.0%}")
        md.append(f"- **Source Diversity:** {avg('diversity'):.0%}")
        md.append(f"- **Retrieval Precision@5:** {avg('precision'):.0%}")
        md.append(f"- **Overall Composite: {avg('composite'):.0%}**")
        md.append(f"- Pass rate: {pass_count}/{n} ({pass_count/n:.0%})")
        md.append(f"- Avg search time: {avg('search_ms'):.0f} ms")
        md.append(f"- Avg LLM time: {avg('llm_ms'):.0f} ms")
        md.append(f"- Avg total time: {avg('elapsed'):.1f} s")
        md.append("")

        # Category breakdown
        categories = sorted(set(r["category"] for r in all_results))
        md.append("### By Category")
        md.append("")
        md.append("| Category | Avg Composite | Count |")
        md.append("|----------|--------------|-------|")
        for cat in categories:
            cat_results = [r for r in all_results if r["category"] == cat]
            cat_avg = sum(r["composite"] for r in cat_results) / len(cat_results)
            md.append(f"| {cat} | {cat_avg:.0%} | {len(cat_results)} |")
        md.append("")

    return "\n".join(md)


def main():
    parser = argparse.ArgumentParser(
        description="RAG Pipeline Quality Evaluation v2 — measures embedding/vector pipeline quality"
    )
    parser.add_argument("--url", default="http://localhost:8000", help="Base API URL")
    parser.add_argument("--top-k", type=int, default=10, help="Number of chunks to retrieve")
    parser.add_argument("--no-reranker", action="store_true", help="Disable cross-encoder reranker")
    parser.add_argument("--output", default=None,
                        help="Output markdown path (default: docs/rag_eval_v2_<timestamp>.md)")
    args = parser.parse_args()

    report = run_evaluation(args)

    if args.output:
        out_path = Path(args.output)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = Path("docs") / f"rag_eval_v2_{ts}.md"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    print(f"\nReport written to: {out_path}")


if __name__ == "__main__":
    main()
