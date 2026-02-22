# RAG Retrieval Quality — Academic References

Research references supporting the retrieval improvements implemented in this project.

## Cross-Encoder Re-ranking

Two-stage retrieve-then-rerank pipeline: bi-encoder for fast candidate retrieval,
cross-encoder for precise relevance scoring.

- **Nogueira & Cho (2019)** — "Passage Re-ranking with BERT"
  arXiv:1901.04085. Foundational work on using BERT as a cross-encoder for
  passage re-ranking, demonstrating significant gains over BM25.

- **Xiao et al. (2023)** — "C-Pack: Packaged Resources To Advance General
  Chinese Embedding" arXiv:2309.07597. Introduces BGE embedding and reranker
  model family (BAAI/bge-reranker-v2-m3 used in this project).

- **Glass et al. (2022)** — "Re2G: Retrieve, Rerank, Generate"
  NAACL 2022. Shows that adding a reranking stage between retrieval and
  generation improves downstream QA accuracy.

## Climate-Domain RAG

- **arXiv:2509.10087** — "Querying Climate Knowledge". Semantic retrieval
  applied to scientific climate literature discovery.

- **arXiv:2510.05336** — "WeatherArchive-Bench". RAG benchmark for weather
  and climate data retrieval tasks.

- **arXiv:2512.02251** — "CAIRNS". Climate adaptation information retrieval
  and QA accuracy evaluation.

- **arXiv:2410.23902** — "Responsible RAG for Climate Decision Making".
  Guidelines for building trustworthy RAG systems in the climate domain.

## Evaluation Methodology

- **Shahul Es et al. (2024)** — "RAGAS: Automated Evaluation of Retrieval
  Augmented Generation" arXiv:2309.15217, EACL 2024. Reference-free metrics
  (faithfulness, context precision/recall, answer relevancy) used in Tier 2.

- **arXiv:2405.07437** — "RAG Evaluation Survey (2024)". Comprehensive survey
  of RAG evaluation approaches and metrics taxonomy.

## Hybrid / Advanced Retrieval

- **Chen et al. (2024)** — "BGE M3-Embedding: Multi-Lingual, Multi-Functionality,
  Multi-Granularity Text Embeddings" arXiv:2402.03216. Dense + sparse + colbert
  hybrid retrieval; motivates the BGE model family choice.

- **arXiv:2411.13154** — "DMQR-RAG: Diverse Multi-Query Rewriting for RAG".
  Query expansion strategies that complement cross-encoder reranking.

## Implementation in This Project

| Component | Model / Method | Reference |
|-----------|---------------|-----------|
| Bi-encoder | BAAI/bge-large-en-v1.5 (1024-dim) | Xiao et al. 2023 |
| Cross-encoder reranker | BAAI/bge-reranker-v2-m3 | Xiao et al. 2023 |
| Metadata boosting | Keyword match on variable/dataset fields | Domain heuristic |
| Tier 1 metrics | Hit@K, MRR, NDCG, Recall | Standard IR (Manning et al.) |
| Tier 2 metrics | RAGAS (faithfulness, precision, recall, relevancy) | Shahul Es et al. 2024 |
| Tier 3 metrics | Intra/inter-variable cosine similarity | Embedding analysis |
