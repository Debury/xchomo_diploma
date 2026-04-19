# `docs/` — Index

This directory holds research notes, evaluation runs, and iteration history.
Only one evaluation file is **canonical** and should be cited from the thesis
and README; everything else is kept for traceability.

## Canonical

| File | Purpose |
|------|---------|
| [`rag_eval_v2_clean_final.md`](./rag_eval_v2_clean_final.md) | **Final RAG evaluation** — top_k=10, reranker=off, Claude Sonnet 4.6. Composite **89 %**, Faithfulness 99 %, Context Relevance 98 %, pass rate 10/10. Cite this one. |

## Research / design notes

| File | Purpose |
|------|---------|
| [`diploma_research_document.md`](./diploma_research_document.md) | Literature-review style document used while writing the thesis. |
| [`optimization_research.md`](./optimization_research.md) | Performance and caching research notes. |
| [`rag_pipeline_improvements.md`](./rag_pipeline_improvements.md) | Proposed / implemented RAG pipeline improvements. |
| [`rag_quality_research.md`](./rag_quality_research.md) | Quality-focused research notes (reranking, prompting, grounding). |
| [`catalog_audit.md`](./catalog_audit.md) | D1.1.xlsx coverage audit — which entries ingest automatically, which require manual steps. |

## Iteration history (superseded)

Kept for audit trail only — **do not cite** from the thesis. Each represents an
intermediate eval run against a specific configuration that we later changed.

- `rag_eval_20260331_145710.md`
- `rag_eval_v1_reindex.md`
- `rag_eval_v2_20260331_145949.md`
- `rag_eval_v2_all_fixes_final.md`
- `rag_eval_v2_data_boost.md`
- `rag_eval_v2_deterministic.md`
- `rag_eval_v2_final.md`
- `rag_eval_v2_no_diversity.md`
- `rag_eval_v2_no_reranker.md`
- `rag_eval_v2_reindex.md`
- `rag_eval_v2_sonnet_multiagent.md`
- `rag_eval_v2_with_retry.md`

## Subdirectories

- [`superpowers/`](./superpowers/) — experimental / auxiliary tooling notes.
