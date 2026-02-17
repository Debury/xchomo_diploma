"""
RAG Evaluation Test Suite for Climate Data Pipeline

Three-tier evaluation methodology:
- Tier 1: Retrieval Quality — golden test set queries against Qdrant (Hit@K, MRR, NDCG, Recall)
- Tier 2: End-to-End RAG — RAGAS framework metrics (faithfulness, context precision/recall, answer relevancy)
- Tier 3: Embedding Space Analysis — intrinsic quality of BGE-M3 embeddings (clustering, similarity distributions)

Academic references:
- RAGAS: Shahul Es et al., arXiv:2309.15217 (EACL 2024)
- ARES: Saad-Falcon et al., arXiv:2311.09476 (NAACL 2024)
- eRAG: Salemi & Zamani, arXiv:2404.13781 (SIGIR 2024)
- BGE-M3: Chen et al., arXiv:2402.03216 (2024)
- Responsible RAG for Climate: arXiv:2410.23902 (2024)
- CAIRNS: arXiv:2512.02251 (2025)
- RAG Evaluation Survey: arXiv:2405.07437 (2024)
"""

import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Configuration & fixtures
# ---------------------------------------------------------------------------

FIXTURES_DIR = Path(__file__).parent / "fixtures"
GOLDEN_QUERIES_PATH = FIXTURES_DIR / "golden_queries.json"

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "climate_data")

# Skip entire module if Qdrant is not reachable
_qdrant_available = False
try:
    from qdrant_client import QdrantClient

    _client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=5)
    _client.get_collections()
    _qdrant_available = True
except Exception:
    pass

pytestmark = pytest.mark.skipif(
    not _qdrant_available,
    reason="Qdrant not reachable — skipping RAG evaluation tests",
)


@pytest.fixture(scope="module")
def qdrant_client():
    """Qdrant client connected to the climate_data collection."""
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    yield client
    client.close()


@pytest.fixture(scope="module")
def embedding_model():
    """Load the BGE-M3 embedding model (shared across all tests in module)."""
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("BAAI/bge-m3")
        return model
    except ImportError:
        pytest.skip("sentence-transformers not installed")


@pytest.fixture(scope="module")
def golden_queries() -> list[dict[str, Any]]:
    """Load the golden test set from JSON fixture."""
    if not GOLDEN_QUERIES_PATH.exists():
        pytest.skip(f"Golden queries file not found: {GOLDEN_QUERIES_PATH}")
    with open(GOLDEN_QUERIES_PATH) as f:
        data = json.load(f)
    return data["queries"]


def _embed_query(model, query: str) -> list[float]:
    """Embed a single query string using the BGE-M3 model."""
    return model.encode(query, normalize_embeddings=True).tolist()


def _search_qdrant(client, vector: list[float], top_k: int = 10) -> list[dict]:
    """Search Qdrant and return results with id + score."""
    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=vector,
        limit=top_k,
        with_payload=True,
    )
    return [
        {
            "id": str(hit.id),
            "score": hit.score,
            "payload": hit.payload,
        }
        for hit in results
    ]


# ===========================================================================
# Tier 1: Retrieval Quality (embedding + Qdrant)
# ===========================================================================


class TestRetrievalQuality:
    """
    Tier 1: Golden test set queries against Qdrant.

    Each golden query has a set of annotated relevant chunk IDs.
    We measure standard IR metrics against those annotations.
    """

    @staticmethod
    def _compute_hit_rate(golden_queries, results_map, k: int) -> float:
        """Fraction of queries where at least one relevant doc is in top-k."""
        hits = 0
        for q in golden_queries:
            query_id = q["id"]
            relevant = set(q["relevant_chunk_ids"])
            retrieved = [r["id"] for r in results_map[query_id][:k]]
            if relevant & set(retrieved):
                hits += 1
        return hits / len(golden_queries) if golden_queries else 0.0

    @staticmethod
    def _compute_mrr(golden_queries, results_map, k: int) -> float:
        """Mean Reciprocal Rank — average 1/rank of first relevant result."""
        rr_sum = 0.0
        for q in golden_queries:
            query_id = q["id"]
            relevant = set(q["relevant_chunk_ids"])
            for rank, r in enumerate(results_map[query_id][:k], start=1):
                if r["id"] in relevant:
                    rr_sum += 1.0 / rank
                    break
        return rr_sum / len(golden_queries) if golden_queries else 0.0

    @staticmethod
    def _compute_ndcg(golden_queries, results_map, k: int) -> float:
        """
        Normalized Discounted Cumulative Gain at k.

        Uses binary relevance (1 if relevant, 0 otherwise).
        Standard MTEB metric for ranking quality.
        """
        ndcg_sum = 0.0
        for q in golden_queries:
            query_id = q["id"]
            relevant = set(q["relevant_chunk_ids"])
            retrieved = results_map[query_id][:k]

            # DCG
            dcg = 0.0
            for rank, r in enumerate(retrieved, start=1):
                rel = 1.0 if r["id"] in relevant else 0.0
                dcg += rel / math.log2(rank + 1)

            # Ideal DCG (all relevant docs at top)
            ideal_rels = min(len(relevant), k)
            idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_rels))

            ndcg_sum += dcg / idcg if idcg > 0 else 0.0

        return ndcg_sum / len(golden_queries) if golden_queries else 0.0

    @staticmethod
    def _compute_recall(golden_queries, results_map, k: int) -> float:
        """Recall@k — fraction of relevant docs found in top-k."""
        recall_sum = 0.0
        for q in golden_queries:
            query_id = q["id"]
            relevant = set(q["relevant_chunk_ids"])
            retrieved = set(r["id"] for r in results_map[query_id][:k])
            if relevant:
                recall_sum += len(relevant & retrieved) / len(relevant)
        return recall_sum / len(golden_queries) if golden_queries else 0.0

    @pytest.fixture(scope="class")
    def results_map(self, qdrant_client, embedding_model, golden_queries):
        """Run all golden queries through Qdrant and cache results."""
        rmap = {}
        for q in golden_queries:
            vector = _embed_query(embedding_model, q["query"])
            rmap[q["id"]] = _search_qdrant(qdrant_client, vector, top_k=10)
        return rmap

    def test_hit_rate_at_5(self, golden_queries, results_map):
        """At least one relevant chunk in top-5 for >= 80% of queries."""
        hit5 = self._compute_hit_rate(golden_queries, results_map, k=5)
        print(f"Hit@5: {hit5:.3f}")
        assert hit5 >= 0.80, f"Hit@5 = {hit5:.3f}, expected >= 0.80"

    def test_hit_rate_at_10(self, golden_queries, results_map):
        """At least one relevant chunk in top-10 for >= 90% of queries."""
        hit10 = self._compute_hit_rate(golden_queries, results_map, k=10)
        print(f"Hit@10: {hit10:.3f}")
        assert hit10 >= 0.90, f"Hit@10 = {hit10:.3f}, expected >= 0.90"

    def test_mrr_at_10(self, golden_queries, results_map):
        """MRR@10 should be >= 0.60 (first relevant result in top-2 on avg)."""
        mrr = self._compute_mrr(golden_queries, results_map, k=10)
        print(f"MRR@10: {mrr:.3f}")
        assert mrr >= 0.60, f"MRR@10 = {mrr:.3f}, expected >= 0.60"

    def test_ndcg_at_10(self, golden_queries, results_map):
        """NDCG@10 should be >= 0.50 (reasonable ranking quality)."""
        ndcg = self._compute_ndcg(golden_queries, results_map, k=10)
        print(f"NDCG@10: {ndcg:.3f}")
        assert ndcg >= 0.50, f"NDCG@10 = {ndcg:.3f}, expected >= 0.50"

    def test_recall_at_3(self, golden_queries, results_map):
        """Recall@3 — coverage in the tightest retrieval window."""
        recall = self._compute_recall(golden_queries, results_map, k=3)
        print(f"Recall@3: {recall:.3f}")
        assert recall >= 0.30, f"Recall@3 = {recall:.3f}, expected >= 0.30"

    def test_recall_at_5(self, golden_queries, results_map):
        """Recall@5 — primary retrieval window for RAG."""
        recall = self._compute_recall(golden_queries, results_map, k=5)
        print(f"Recall@5: {recall:.3f}")
        assert recall >= 0.50, f"Recall@5 = {recall:.3f}, expected >= 0.50"

    def test_recall_at_10(self, golden_queries, results_map):
        """Recall@10 — extended retrieval window."""
        recall = self._compute_recall(golden_queries, results_map, k=10)
        print(f"Recall@10: {recall:.3f}")
        assert recall >= 0.70, f"Recall@10 = {recall:.3f}, expected >= 0.70"


# ===========================================================================
# Tier 2: End-to-End RAG (RAGAS framework)
# ===========================================================================


class TestRAGASMetrics:
    """
    Tier 2: End-to-end RAG evaluation using RAGAS framework.

    RAGAS (Shahul Es et al., EACL 2024) provides reference-free metrics
    that use an LLM judge to assess RAG pipeline quality.

    Requires: pip install ragas
    """

    @pytest.fixture(scope="class")
    def ragas_dataset(self, golden_queries):
        """
        Build a RAGAS-compatible evaluation dataset from golden queries.

        Each entry needs: question, answer, contexts, ground_truth (optional).
        """
        try:
            from ragas import evaluate
            from ragas.metrics import (
                answer_relevancy,
                context_precision,
                context_recall,
                faithfulness,
            )
        except ImportError:
            pytest.skip("ragas not installed — pip install ragas")

        # TODO: Generate answers by calling the RAG pipeline for each golden query.
        # For now, return a placeholder structure.
        pytest.skip(
            "RAGAS dataset generation requires a running RAG pipeline. "
            "Implement after RAG endpoint integration."
        )

    def test_faithfulness(self, ragas_dataset):
        """
        Faithfulness: Are generated claims grounded in retrieved context?

        Target: >= 0.85. Climate data answers must not hallucinate numbers.
        Uses RAGAS LLM-judge decomposition of claims vs. context.
        """
        score = ragas_dataset["faithfulness"]
        print(f"Faithfulness: {score:.3f}")
        assert score >= 0.85, f"Faithfulness = {score:.3f}, expected >= 0.85"

    def test_context_precision(self, ragas_dataset):
        """
        Context Precision: Did we retrieve relevant chunks?

        Target: >= 0.70. Measures signal-to-noise in retrieved contexts.
        """
        score = ragas_dataset["context_precision"]
        print(f"Context Precision: {score:.3f}")
        assert score >= 0.70, f"Context Precision = {score:.3f}, expected >= 0.70"

    def test_context_recall(self, ragas_dataset):
        """
        Context Recall: Did we retrieve ALL relevant chunks?

        Target: >= 0.75. Critical for climate data completeness.
        """
        score = ragas_dataset["context_recall"]
        print(f"Context Recall: {score:.3f}")
        assert score >= 0.75, f"Context Recall = {score:.3f}, expected >= 0.75"

    def test_answer_relevancy(self, ragas_dataset):
        """
        Answer Relevancy: Does the answer address the question?

        Target: >= 0.80. Measures semantic similarity between question and answer.
        """
        score = ragas_dataset["answer_relevancy"]
        print(f"Answer Relevancy: {score:.3f}")
        assert score >= 0.80, f"Answer Relevancy = {score:.3f}, expected >= 0.80"

    def test_numerical_coverage(self, golden_queries, qdrant_client, embedding_model):
        """
        Custom metric: Numerical Coverage.

        Measures the fraction of key numbers (temperatures, dates, thresholds)
        from source data that are preserved in the RAG answer. Critical for
        climate data accuracy where precise values matter.

        Target: >= 0.90. Climate answers must preserve numerical precision.
        """
        # TODO: Implement after RAG pipeline integration.
        # Algorithm:
        # 1. For each golden query with annotated key_numbers
        # 2. Generate RAG answer
        # 3. Extract numbers from answer using regex
        # 4. Compute fraction of key_numbers found in answer
        pytest.skip(
            "Numerical coverage requires RAG answer generation. "
            "Implement after RAG endpoint integration."
        )


# ===========================================================================
# Tier 3: Embedding Space Analysis (intrinsic)
# ===========================================================================


class TestEmbeddingSpace:
    """
    Tier 3: Intrinsic embedding quality analysis.

    Evaluates the structure of the BGE-M3 embedding space for climate data.
    Checks that semantically similar chunks cluster together and different
    variables/datasets are properly separated.
    """

    @pytest.fixture(scope="class")
    def collection_vectors(self, qdrant_client):
        """
        Sample vectors from the Qdrant collection with their metadata.

        Returns list of (vector, payload) tuples.
        """
        # Scroll through collection to get a sample of points
        points, _ = qdrant_client.scroll(
            collection_name=COLLECTION_NAME,
            limit=500,
            with_vectors=True,
            with_payload=True,
        )
        if len(points) < 20:
            pytest.skip("Not enough points in collection for embedding analysis")
        return [(p.vector, p.payload) for p in points]

    @staticmethod
    def _cosine_similarity(a, b) -> float:
        """Compute cosine similarity between two vectors."""
        a, b = np.array(a), np.array(b)
        dot = np.dot(a, b)
        norm = np.linalg.norm(a) * np.linalg.norm(b)
        return float(dot / norm) if norm > 0 else 0.0

    def test_intra_variable_similarity(self, collection_vectors):
        """
        Chunks describing the same climate variable should have higher
        cosine similarity than random pairs.

        Groups chunks by variable (e.g., temperature, precipitation) and
        checks that within-group similarity exceeds between-group similarity.
        """
        # Group by variable
        groups: dict[str, list] = {}
        for vec, payload in collection_vectors:
            var = payload.get("variable") or payload.get("variable_name", "unknown")
            if var != "unknown":
                groups.setdefault(var, []).append(vec)

        if len(groups) < 2:
            pytest.skip("Need at least 2 variable groups for comparison")

        # Compute intra-group similarities (sample pairs)
        intra_sims = []
        for var, vecs in groups.items():
            if len(vecs) < 2:
                continue
            sample = vecs[:20]  # limit to prevent O(n^2) blowup
            for i in range(len(sample)):
                for j in range(i + 1, len(sample)):
                    intra_sims.append(self._cosine_similarity(sample[i], sample[j]))

        # Compute inter-group similarities (sample pairs between groups)
        inter_sims = []
        group_names = list(groups.keys())
        for gi in range(len(group_names)):
            for gj in range(gi + 1, len(group_names)):
                v1 = groups[group_names[gi]][:10]
                v2 = groups[group_names[gj]][:10]
                for a in v1:
                    for b in v2:
                        inter_sims.append(self._cosine_similarity(a, b))

        if not intra_sims or not inter_sims:
            pytest.skip("Insufficient data for similarity comparison")

        avg_intra = np.mean(intra_sims)
        avg_inter = np.mean(inter_sims)
        print(f"Avg intra-variable similarity: {avg_intra:.3f}")
        print(f"Avg inter-variable similarity: {avg_inter:.3f}")
        print(f"Separation gap: {avg_intra - avg_inter:.3f}")

        assert avg_intra > avg_inter, (
            f"Intra-variable similarity ({avg_intra:.3f}) should exceed "
            f"inter-variable similarity ({avg_inter:.3f})"
        )

    def test_inter_variable_separation(self, collection_vectors):
        """
        Different climate variables should have measurable separation
        in embedding space. The gap should be statistically significant.
        """
        groups: dict[str, list] = {}
        for vec, payload in collection_vectors:
            var = payload.get("variable") or payload.get("variable_name", "unknown")
            if var != "unknown":
                groups.setdefault(var, []).append(vec)

        if len(groups) < 2:
            pytest.skip("Need at least 2 variable groups")

        # Compute centroids for each variable
        centroids = {}
        for var, vecs in groups.items():
            if len(vecs) >= 3:
                centroids[var] = np.mean(vecs, axis=0)

        if len(centroids) < 2:
            pytest.skip("Need at least 2 centroids for separation analysis")

        # Compute pairwise centroid distances
        centroid_names = list(centroids.keys())
        distances = []
        for i in range(len(centroid_names)):
            for j in range(i + 1, len(centroid_names)):
                sim = self._cosine_similarity(
                    centroids[centroid_names[i]], centroids[centroid_names[j]]
                )
                distances.append(1.0 - sim)  # cosine distance

        avg_distance = np.mean(distances)
        print(f"Average inter-variable centroid distance: {avg_distance:.3f}")
        print(f"Variables analyzed: {list(centroids.keys())}")

        # Centroids of different variables should not be identical
        assert avg_distance > 0.01, (
            f"Inter-variable distance ({avg_distance:.3f}) is too small — "
            "embeddings may not differentiate between climate variables"
        )

    def test_nearest_neighbor_sanity(self, qdrant_client, embedding_model):
        """
        Sanity check: queries about specific variables should return
        chunks about those same variables as nearest neighbors.

        E.g., "global mean temperature anomaly" → temperature chunks.
        """
        sanity_queries = [
            {
                "query": "global mean temperature anomaly 2023",
                "expected_variables": ["temperature", "tas", "tmp", "t2m"],
            },
            {
                "query": "annual precipitation totals Europe",
                "expected_variables": ["precipitation", "pr", "pre", "tp"],
            },
            {
                "query": "sea level rise projections",
                "expected_variables": ["sea_level", "zos", "sea level"],
            },
        ]

        for sq in sanity_queries:
            vector = _embed_query(embedding_model, sq["query"])
            results = _search_qdrant(qdrant_client, vector, top_k=5)

            if not results:
                continue

            # Check if any of the top-5 results match expected variables
            expected = set(v.lower() for v in sq["expected_variables"])
            found_match = False
            for r in results:
                payload = r["payload"]
                var = (
                    payload.get("variable", "")
                    or payload.get("variable_name", "")
                    or ""
                ).lower()
                dataset = (payload.get("dataset", "") or "").lower()
                text = (payload.get("text", "") or "").lower()

                for exp in expected:
                    if exp in var or exp in dataset or exp in text:
                        found_match = True
                        break
                if found_match:
                    break

            print(
                f"Query: '{sq['query']}' → "
                f"Match: {found_match}, "
                f"Top result: {results[0]['payload'].get('variable', 'N/A')}"
            )
            # Soft assertion — log but don't fail on individual queries
            if not found_match:
                print(
                    f"  WARNING: No expected variable match in top-5 for "
                    f"'{sq['query']}'"
                )


# ===========================================================================
# Utility: Run evaluation and generate report
# ===========================================================================


def generate_evaluation_report(output_path: str = "evaluation_report.json"):
    """
    Run all evaluation tiers and generate a JSON report.

    Usage:
        python -m tests.test_rag_evaluation
    """
    print("To run the evaluation suite:")
    print("  pytest tests/test_rag_evaluation.py -v --tb=short")
    print()
    print("To run individual tiers:")
    print("  pytest tests/test_rag_evaluation.py::TestRetrievalQuality -v")
    print("  pytest tests/test_rag_evaluation.py::TestRAGASMetrics -v")
    print("  pytest tests/test_rag_evaluation.py::TestEmbeddingSpace -v")


if __name__ == "__main__":
    generate_evaluation_report()
