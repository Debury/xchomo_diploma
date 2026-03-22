"""
Automated RAG evaluation test suite.

Tests the RAG system against validated climate claims and questions.
Evaluates both retrieval quality (dataset relevance) and answer quality (when LLM is available).

Usage:
    pytest tests/test_rag_claims.py -v
    pytest tests/test_rag_claims.py -v --tb=short  # compact output
    pytest tests/test_rag_claims.py -v -s          # with print output for full report
"""

import json
import time
import requests
import pytest

RAG_URL = "http://localhost:8000/rag/query"
TOP_K = 10

# ---------------------------------------------------------------------------
# Test data: claims, questions, expected datasets, expected keywords
# ---------------------------------------------------------------------------

CLAIMS = [
    {
        "id": "C1",
        "claim": "Were 2023 and 2024 the warmest years on record, with 2024 breaching the 1.5°C threshold?",
        "expected_datasets": ["ERA5", "C3S", "Copernicus"],
        "expected_keywords": ["temperature", "warm", "global", "surface"],
        "ground_truth": True,
    },
    {
        "id": "C2",
        "claim": "Did European heatwaves in 2022 and 2023 exceed 40°C and feature extreme heat stress?",
        "expected_datasets": ["E-OBS", "ERA5"],
        "expected_keywords": ["temperature", "heat", "Europe"],
        "ground_truth": True,
    },
    {
        "id": "C3",
        "claim": "Does IMERG show a trend of increasingly intense heavy rainfall events over the last decade?",
        "expected_datasets": ["IMERG", "GPM"],
        "expected_keywords": ["precipitation", "rainfall", "intense"],
        "ground_truth": True,
    },
    {
        "id": "C4",
        "claim": "Were megadroughts exacerbated by extreme potential evapotranspiration, visible in SPEIbase?",
        "expected_datasets": ["SPEI", "SPEIbase"],
        "expected_keywords": ["drought", "evapotranspiration", "SPEI", "precipitation"],
        "ground_truth": True,
    },
    {
        "id": "C5",
        "claim": "Does GRACE data show accelerating ice sheet loss and global sea-level rise exceeding 4 mm/yr?",
        "expected_datasets": ["GRACE", "GRACE-FO"],
        "expected_keywords": ["ice", "sea level", "mass", "water", "gravimetry"],
        "ground_truth": True,
    },
    {
        "id": "C6",
        "claim": "Did Mediterranean marine heatwaves reach 4°C to 5°C above average between 2022 and 2025?",
        "expected_datasets": ["Copernicus Marine", "SST", "ERA5"],
        "expected_keywords": ["sea surface temperature", "SST", "marine", "Mediterranean", "temperature"],
        "ground_truth": True,
    },
    {
        "id": "C7",
        "claim": "Has atmospheric CO2 consistently surpassed 420 ppm in the 2020s with growth rate exceeding 2.5 ppm per year?",
        "expected_datasets": ["CAMS", "CO2", "Copernicus Atmosphere"],
        "expected_keywords": ["co2", "carbon", "atmosphere", "ppm"],
        "ground_truth": True,
    },
    {
        "id": "C8",
        "claim": "Do CAMS and MERRA-2 track significant Saharan dust intrusion anomalies into Europe?",
        "expected_datasets": ["CAMS", "MERRA"],
        "expected_keywords": ["dust", "aerosol", "Sahara", "air quality"],
        "ground_truth": True,
    },
]

QUESTIONS = [
    {
        "id": "Q1",
        "question": (
            "How do the anomalies in the SPEIbase drought index and ERA5 temperature data "
            "correlate with Fire radiative power and CAMS air pollution emissions during the "
            "2023 Northern Hemisphere summer, particularly in Canada?"
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
        "expected_datasets": ["IMERG", "ERA5", "GRACE"],
        "expected_keywords": ["precipitation", "flood", "water", "soil", "Pakistan"],
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def query_rag(question: str) -> dict:
    """Send a question to the RAG endpoint and return the JSON response."""
    resp = requests.post(
        RAG_URL,
        json={"question": question, "top_k": TOP_K},
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()


def check_dataset_relevance(chunks: list[dict], expected_datasets: list[str]) -> dict:
    """
    Check if any of the expected datasets appear in the retrieved chunks.
    Returns dict with match info.
    """
    all_text = " ".join(
        (c.get("text", "") + " " + json.dumps(c.get("metadata", {})))
        for c in chunks
    ).lower()

    found = []
    missing = []
    for ds in expected_datasets:
        if ds.lower() in all_text:
            found.append(ds)
        else:
            missing.append(ds)

    return {
        "found": found,
        "missing": missing,
        "hit_rate": len(found) / len(expected_datasets) if expected_datasets else 0,
    }


def check_keyword_presence(chunks: list[dict], expected_keywords: list[str]) -> dict:
    """
    Check if expected keywords appear in retrieved chunks (text + metadata).
    """
    # Search in both text and full metadata for better coverage
    all_text = " ".join(
        c.get("text", "") + " " + json.dumps(c.get("metadata", {}))
        for c in chunks
    ).lower()

    found = []
    missing = []
    for kw in expected_keywords:
        if kw.lower() in all_text:
            found.append(kw)
        else:
            missing.append(kw)

    return {
        "found": found,
        "missing": missing,
        "hit_rate": len(found) / len(expected_keywords) if expected_keywords else 0,
    }


def check_answer_quality(answer: str, expected_keywords: list[str]) -> dict:
    """
    Check if the LLM answer mentions expected keywords.
    """
    if not answer or "LLM error" in answer or "Showing search results instead" in answer:
        return {"llm_available": False, "hit_rate": 0, "found": [], "missing": expected_keywords}

    answer_lower = answer.lower()
    found = [kw for kw in expected_keywords if kw.lower() in answer_lower]
    missing = [kw for kw in expected_keywords if kw.lower() not in answer_lower]

    return {
        "llm_available": True,
        "hit_rate": len(found) / len(expected_keywords) if expected_keywords else 0,
        "found": found,
        "missing": missing,
    }


# ---------------------------------------------------------------------------
# Score aggregation
# ---------------------------------------------------------------------------

class ScoreTracker:
    """Collects results across all tests for a final summary."""

    def __init__(self):
        self.results = []

    def add(self, test_id: str, dataset_score: float, keyword_score: float,
            answer_score: float | None, avg_retrieval_score: float,
            llm_available: bool, search_time_ms: float):
        self.results.append({
            "id": test_id,
            "dataset_relevance": dataset_score,
            "keyword_relevance": keyword_score,
            "answer_quality": answer_score,
            "avg_retrieval_score": avg_retrieval_score,
            "llm_available": llm_available,
            "search_time_ms": search_time_ms,
        })

    def summary(self) -> str:
        if not self.results:
            return "No results collected."

        n = len(self.results)
        avg_dataset = sum(r["dataset_relevance"] for r in self.results) / n
        avg_keyword = sum(r["keyword_relevance"] for r in self.results) / n
        avg_retrieval = sum(r["avg_retrieval_score"] for r in self.results) / n
        avg_search = sum(r["search_time_ms"] for r in self.results) / n

        llm_results = [r for r in self.results if r["llm_available"]]
        avg_answer = (
            sum(r["answer_quality"] for r in llm_results) / len(llm_results)
            if llm_results else None
        )

        lines = [
            "",
            "=" * 70,
            "RAG EVALUATION SUMMARY",
            "=" * 70,
            f"  Total test cases:        {n}",
            f"  Dataset relevance:       {avg_dataset * 100:.1f}%",
            f"  Keyword relevance:       {avg_keyword * 100:.1f}%",
            f"  Avg retrieval score:     {avg_retrieval:.3f}",
            f"  Avg search time:         {avg_search:.0f} ms",
        ]

        if avg_answer is not None:
            lines.append(f"  Answer quality:          {avg_answer * 100:.1f}% ({len(llm_results)}/{n} with LLM)")
        else:
            lines.append(f"  Answer quality:          N/A (LLM unavailable)")

        # Overall score: weighted combination
        if avg_answer is not None:
            overall = avg_dataset * 0.35 + avg_keyword * 0.30 + avg_answer * 0.35
        else:
            overall = avg_dataset * 0.55 + avg_keyword * 0.45

        lines.append(f"  ─────────────────────────────────")
        lines.append(f"  OVERALL SCORE:           {overall * 100:.1f}%")
        lines.append("=" * 70)

        # Per-test breakdown
        lines.append("")
        lines.append(f"  {'ID':<5} {'Dataset%':>9} {'Keyword%':>9} {'Answer%':>9} {'Retrieval':>10} {'Time':>8}")
        lines.append(f"  {'─' * 5} {'─' * 9} {'─' * 9} {'─' * 9} {'─' * 10} {'─' * 8}")
        for r in self.results:
            ans = f"{r['answer_quality'] * 100:.0f}%" if r["llm_available"] else "N/A"
            lines.append(
                f"  {r['id']:<5} {r['dataset_relevance'] * 100:>8.0f}% "
                f"{r['keyword_relevance'] * 100:>8.0f}% {ans:>9} "
                f"{r['avg_retrieval_score']:>10.3f} {r['search_time_ms']:>7.0f}ms"
            )

        lines.append("")
        return "\n".join(lines)


tracker = ScoreTracker()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session", autouse=True)
def check_api_health():
    """Ensure the API is running before tests."""
    try:
        resp = requests.get("http://localhost:8000/health", timeout=10)
        resp.raise_for_status()
    except Exception as e:
        pytest.skip(f"API not available: {e}")


@pytest.fixture(scope="session", autouse=True)
def print_summary(request):
    """Print the evaluation summary after all tests complete."""
    yield
    print(tracker.summary())


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("claim", CLAIMS, ids=[c["id"] for c in CLAIMS])
def test_claim_retrieval(claim):
    """Test that retrieval returns relevant datasets for each claim."""
    result = query_rag(claim["claim"])
    chunks = result.get("chunks", [])

    assert len(chunks) > 0, f"No chunks retrieved for claim {claim['id']}"

    # Evaluate
    ds_check = check_dataset_relevance(chunks, claim["expected_datasets"])
    kw_check = check_keyword_presence(chunks, claim["expected_keywords"])
    ans_check = check_answer_quality(result.get("answer", ""), claim["expected_keywords"])

    scores = [c.get("score", 0) for c in chunks if c.get("score")]
    avg_score = sum(scores) / len(scores) if scores else 0

    tracker.add(
        test_id=claim["id"],
        dataset_score=ds_check["hit_rate"],
        keyword_score=kw_check["hit_rate"],
        answer_score=ans_check["hit_rate"] if ans_check.get("llm_available") else None,
        avg_retrieval_score=avg_score,
        llm_available=ans_check.get("llm_available", False),
        search_time_ms=result.get("search_time_ms", 0),
    )

    # Report details on failure
    detail = (
        f"\n  Claim {claim['id']}: {claim['claim'][:80]}...\n"
        f"  Datasets found: {ds_check['found']} | missing: {ds_check['missing']}\n"
        f"  Keywords found: {kw_check['found']} | missing: {kw_check['missing']}\n"
        f"  Avg retrieval score: {avg_score:.3f}\n"
        f"  Top sources: {result.get('references', [])[:5]}"
    )

    # Soft assertion: warn but don't fail if retrieval score is low
    if ds_check["hit_rate"] == 0:
        print(f"\n  WARNING: No expected datasets found for {claim['id']}{detail}")

    # At minimum, we expect some keyword overlap in retrieved chunks
    assert kw_check["hit_rate"] > 0 or ds_check["hit_rate"] > 0, (
        f"Neither datasets nor keywords found in retrieval results.{detail}"
    )


@pytest.mark.parametrize("question", QUESTIONS, ids=[q["id"] for q in QUESTIONS])
def test_question_retrieval(question):
    """Test that retrieval returns relevant datasets for complex questions."""
    result = query_rag(question["question"])
    chunks = result.get("chunks", [])

    assert len(chunks) > 0, f"No chunks retrieved for question {question['id']}"

    ds_check = check_dataset_relevance(chunks, question["expected_datasets"])
    kw_check = check_keyword_presence(chunks, question["expected_keywords"])
    ans_check = check_answer_quality(result.get("answer", ""), question["expected_keywords"])

    scores = [c.get("score", 0) for c in chunks if c.get("score")]
    avg_score = sum(scores) / len(scores) if scores else 0

    tracker.add(
        test_id=question["id"],
        dataset_score=ds_check["hit_rate"],
        keyword_score=kw_check["hit_rate"],
        answer_score=ans_check["hit_rate"] if ans_check.get("llm_available") else None,
        avg_retrieval_score=avg_score,
        llm_available=ans_check.get("llm_available", False),
        search_time_ms=result.get("search_time_ms", 0),
    )

    detail = (
        f"\n  Question {question['id']}: {question['question'][:80]}...\n"
        f"  Datasets found: {ds_check['found']} | missing: {ds_check['missing']}\n"
        f"  Keywords found: {kw_check['found']} | missing: {kw_check['missing']}\n"
        f"  Avg retrieval score: {avg_score:.3f}\n"
        f"  Top sources: {result.get('references', [])[:5]}"
    )

    if ds_check["hit_rate"] == 0:
        print(f"\n  WARNING: No expected datasets found for {question['id']}{detail}")

    assert kw_check["hit_rate"] > 0 or ds_check["hit_rate"] > 0, (
        f"Neither datasets nor keywords found in retrieval results.{detail}"
    )
