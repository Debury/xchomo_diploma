# RAG Evaluation Methods and Academic Research

A comprehensive reference for evaluating a climate data RAG (Retrieval-Augmented Generation) system, compiled for diploma thesis purposes. Covers evaluation metrics, academic papers, evaluation frameworks, and a recommended evaluation strategy.

---

## Part A: Evaluation Methods

### A.1 Retrieval Metrics

Retrieval quality is assessed using standard information retrieval (IR) metrics. These measure how well the retrieval component returns relevant documents before any generation occurs.

#### Precision@k

The fraction of the top-k retrieved documents that are relevant.

```
Precision@k = |{relevant documents in top-k}| / k
```

- Simple and interpretable.
- Does not account for ranking order within the top-k.
- Useful when the user inspects a fixed number of results.

#### Recall@k

The fraction of all relevant documents that appear in the top-k retrieved results.

```
Recall@k = |{relevant documents in top-k}| / |{all relevant documents}|
```

- Critical for RAG systems: if the retriever misses a relevant chunk, the generator cannot use it.
- In practice, Recall@5 or Recall@10 is the most important retrieval metric for RAG, because the generator only sees the top-k passages.

#### Mean Reciprocal Rank (MRR)

The average of the reciprocal ranks of the first relevant document across a set of queries.

```
MRR = (1/|Q|) * SUM(1 / rank_i)
```

where `rank_i` is the position of the first relevant result for query `i`.

- Focuses on how quickly the user (or generator) encounters the first relevant result.
- Ranges from 0 to 1; higher is better.
- Well-suited when there is typically one correct answer.

#### Normalized Discounted Cumulative Gain (nDCG@k)

Measures ranking quality with graded relevance (not just binary relevant/irrelevant).

```
DCG@k = SUM(rel_i / log2(i + 1))    for i = 1..k
nDCG@k = DCG@k / IDCG@k
```

where `IDCG@k` is the DCG of the ideal (perfect) ranking.

- Accounts for position: relevant documents ranked higher contribute more.
- Supports graded relevance labels (e.g., 0=irrelevant, 1=partially relevant, 2=highly relevant).
- The standard metric for evaluating ranking systems in IR research.

#### Mean Average Precision (MAP)

The mean of Average Precision (AP) across all queries, where AP is the average of precision values at each relevant document position.

```
AP = (1/|R|) * SUM(Precision@k * rel(k))    for k = 1..N
MAP = (1/|Q|) * SUM(AP_i)
```

- Combines precision and recall into a single metric.
- Rewards systems that rank all relevant documents highly.
- Standard metric in TREC-style IR evaluations.

#### Hit Rate (Hit@k)

Binary indicator of whether at least one relevant document appears in the top-k results.

```
Hit@k = 1 if any relevant document is in top-k, else 0
```

- The simplest retrieval metric; useful as a sanity check.
- For RAG: "Did the retriever find anything useful at all?"

---

### A.2 Embedding Quality Evaluation Methods

#### Intrinsic Evaluation

Intrinsic evaluation measures the quality of embeddings directly, without reference to a downstream task.

| Method | Description | Pros | Cons |
|--------|-------------|------|------|
| **Cosine similarity correlation** | Compare cosine similarity of embedding pairs against human-rated similarity scores (e.g., STS Benchmark). | Direct measure of semantic quality. | May not correlate with downstream performance. |
| **Analogy tasks** | Test relational patterns (e.g., "king - man + woman = queen"). | Tests fine-grained semantic structure. | Less relevant for document-level retrieval. |
| **Nearest-neighbor inspection** | Manually inspect whether nearest neighbors in embedding space are semantically related. | Qualitative insight into embedding behavior. | Not scalable; subjective. |
| **Clustering quality** | Cluster embeddings and measure coherence (silhouette score, adjusted Rand index). | Tests whether embeddings group related content. | Requires predefined clusters/labels. |
| **MTEB Benchmark** | Comprehensive benchmark across 8 tasks, 58 datasets, 112 languages. | Standardized, widely adopted. | General-purpose; may not reflect domain-specific quality. |

#### Extrinsic Evaluation

Extrinsic evaluation measures embedding quality through downstream task performance.

| Method | Description | Applicability |
|--------|-------------|---------------|
| **Retrieval accuracy** | Use embeddings in a retrieval pipeline and measure Recall@k, nDCG@k. | Directly relevant for RAG. |
| **Classification accuracy** | Train a classifier on embeddings, measure F1/accuracy. | Tests discriminative power. |
| **RAG end-to-end quality** | Measure final answer quality when embeddings are used for retrieval. | The most relevant evaluation for a RAG system. |

**Recommendation for climate RAG:** Use extrinsic evaluation (retrieval Recall@k) as the primary embedding quality metric, supplemented by intrinsic nearest-neighbor inspection for qualitative analysis.

---

### A.3 End-to-End RAG Pipeline Metrics

These metrics evaluate the full pipeline: query -> retrieval -> generation -> answer.

#### Faithfulness (Groundedness)

Measures whether the generated answer is factually consistent with the retrieved context. A faithful answer makes no claims that cannot be traced back to the retrieved documents.

- Operationalized by decomposing the answer into atomic claims and verifying each against the context.
- Score = (number of claims supported by context) / (total number of claims).
- Critical for preventing hallucination in domain-specific systems.

#### Answer Relevancy

Measures whether the generated answer actually addresses the user's question.

- Computed by generating hypothetical questions from the answer and measuring their semantic similarity to the original question.
- Low relevancy indicates the answer is off-topic or includes unnecessary information.

#### Context Relevancy (Context Precision)

Measures how much of the retrieved context is actually relevant to answering the question.

- Penalizes retrieval of irrelevant passages that may distract the generator.
- Computed as the fraction of retrieved sentences/passages that are pertinent.

#### Context Recall

Measures whether the retrieved context contains all the information needed to answer the question.

- Requires ground-truth answers for comparison.
- Computed by checking whether each claim in the ground-truth answer can be attributed to the retrieved context.

#### Answer Correctness

Measures the factual accuracy of the answer against a ground-truth reference answer.

- Combines semantic similarity (embedding-based) with factual overlap (F1 on claims).
- Requires a human-annotated reference answer dataset.

#### Answer Similarity

Measures the semantic similarity between the generated answer and a reference answer using embedding cosine similarity.

- Captures meaning equivalence even when phrasing differs.
- Useful as a complement to exact-match or F1 metrics.

---

### A.4 RAGAS Framework

**RAGAS** (Retrieval Augmented Generation Assessment) is the most widely adopted open-source framework for RAG evaluation, published at EACL 2024 by Es et al.

#### Core Design Principles

1. **Reference-free evaluation:** Does not require human-annotated ground truth for most metrics. Uses LLMs as judges to assess quality.
2. **Component-wise evaluation:** Separately evaluates retrieval and generation, enabling targeted optimization.
3. **Automated scoring:** All metrics return values in [0, 1], higher is better.

#### RAGAS Metrics Suite

| Metric | Inputs Required | What It Measures |
|--------|----------------|-----------------|
| **Faithfulness** | Question, Context, Answer | Factual consistency of answer with context |
| **Answer Relevancy** | Question, Answer | Whether the answer addresses the question |
| **Context Precision** | Question, Context, Ground Truth | Proportion of relevant items ranked high in context |
| **Context Recall** | Context, Ground Truth | Whether context contains all needed information |
| **Context Relevancy** | Question, Context | How focused/relevant the retrieved context is |
| **Answer Correctness** | Answer, Ground Truth | Factual accuracy against reference answer |
| **Answer Similarity** | Answer, Ground Truth | Semantic similarity to reference answer |

#### RAGAS Score (Harmonic Mean)

The overall RAGAS score is the harmonic mean of the core metrics:

```
RAGAS Score = harmonic_mean(Faithfulness, Answer Relevancy, Context Precision, Context Recall)
```

#### Strengths and Limitations

**Strengths:**
- No manual annotation needed for core metrics (faithfulness, relevancy).
- Easy integration with LangChain, LlamaIndex, Haystack.
- Widely cited and academically validated.
- Open-source (Apache 2.0).

**Limitations:**
- LLM-as-judge introduces its own biases and costs.
- Metrics correlate with but do not replace human judgment.
- Context Recall does require ground-truth answers.
- Results depend on the judge LLM quality (GPT-4 recommended).

---

### A.5 Other Evaluation Frameworks

#### ARES (Stanford, 2024)

**Automated RAG Evaluation System.** Published at NAACL 2024 by Saad-Falcon, Khattab, Potts, and Zaharia.

- Uses synthetic training data to fine-tune lightweight LM judges.
- Applies Prediction-Powered Inference (PPI) with a small set of human annotations.
- Evaluates context relevance, answer faithfulness, and answer relevance.
- Effective across domain shifts with only a few hundred human annotations.
- More rigorous statistical guarantees than RAGAS through PPI.

**When to use:** When you need confidence intervals on evaluation scores and have some (but limited) human annotations.

#### TruLens (Snowflake/TruEra)

- Open-source library for evaluating and tracing LLM applications.
- Built-in feedback functions: Groundedness, Context Relevance, Answer Relevance (the "RAG Triad").
- OpenTelemetry-based tracing to capture each pipeline step.
- Good for runtime monitoring and debugging.

**When to use:** For continuous monitoring of a deployed RAG system; less suited for thesis-style batch evaluation.

#### DeepEval (Confident AI)

- Open-source Python evaluation framework with unit-testing paradigm.
- 14+ pre-built metrics including hallucination, relevance, toxicity, bias.
- Synthetic data generation for test set creation.
- CI/CD integration for continuous evaluation.
- Supports custom metrics.

**When to use:** When you want to integrate RAG evaluation into a testing pipeline with broad metric coverage.

#### ARAGOG (2024)

- Advanced RAG Output Grading framework.
- Empirically compares RAG techniques: HyDE, reranking, MMR, multi-query, sentence window retrieval.
- Found that HyDE and LLM reranking enhance retrieval precision most significantly.
- Provides a methodology for comparative RAG technique evaluation.

**When to use:** When comparing multiple RAG configurations head-to-head.

#### Framework Comparison

| Feature | RAGAS | ARES | TruLens | DeepEval |
|---------|-------|------|---------|----------|
| Reference-free | Yes (core metrics) | Partially (needs some annotations) | Yes | Yes |
| Statistical guarantees | No | Yes (PPI) | No | No |
| Synthetic data gen | No | Yes | No | Yes |
| Runtime monitoring | Limited | No | Yes | Yes |
| CI/CD integration | Limited | No | No | Yes |
| Academic publication | EACL 2024 | NAACL 2024 | No | No |
| Open source | Yes | Yes | Yes | Yes |
| Custom metrics | Yes | Limited | Yes | Yes |

---

## Part B: Relevant Papers with Citations

### B.1 Foundational RAG Papers

**1. Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Kuttler, H., Lewis, M., Yih, W., Rocktaschel, T., Riedel, S., and Kiela, D. (2020).** "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks." *NeurIPS 2020.* [arXiv:2005.11401](https://arxiv.org/abs/2005.11401)
> The foundational RAG paper. Introduces the RAG model combining a parametric seq2seq model with a non-parametric dense vector index of Wikipedia, accessed via a neural retriever. Sets state-of-the-art on three open-domain QA tasks.

**2. Izacard, G. and Grave, E. (2021).** "Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering." *EACL 2021, pp. 874-880.* [arXiv:2007.01282](https://arxiv.org/abs/2007.01282)
> Introduces Fusion-in-Decoder (FiD), which independently encodes multiple retrieved passages and fuses them in the decoder. State-of-the-art on Natural Questions and TriviaQA. Shows performance improves with more retrieved passages.

**3. Borgeaud, S., Mensch, A., Hoffmann, J., Cai, T., Rutherford, E., Millican, K., van den Driessche, G., Lespiau, J.-B., Damoc, B., Clark, A., et al. (2022).** "Improving Language Models by Retrieving from Trillions of Tokens." *ICML 2022.* [arXiv:2112.04426](https://arxiv.org/abs/2112.04426)
> Introduces RETRO (Retrieval Enhanced Transformers). A 7.5B parameter RETRO model outperforms 175B GPT-3 and 280B Gopher on multiple benchmarks. Demonstrates that retrieval can substitute for massive parameterization.

**4. Shi, W., Min, S., Yasunaga, M., Seo, M., James, R., Lewis, M., Zettlemoyer, L., and Yih, W. (2024).** "REPLUG: Retrieval-Augmented Black-Box Language Models." *NAACL 2024.* [arXiv:2301.12652](https://arxiv.org/abs/2301.12652)
> Treats the LLM as a black box and augments it with a tunable retrieval model. Simply prepends retrieved documents to the input. Improves GPT-3 language modeling by 6.3% and Codex on MMLU by 5.1%. Directly relevant to architectures using API-based LLMs (like Groq/OpenRouter).

**5. Asai, A., Wu, Z., Wang, Y., Sil, A., and Hajishirzi, H. (2024).** "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection." *ICLR 2024 (Oral, top 1%).* [arXiv:2310.11511](https://arxiv.org/abs/2310.11511)
> Introduces reflection tokens that allow the model to decide when to retrieve and to self-assess generation quality. Outperforms standard RAG across open-domain QA, reasoning, and fact verification tasks at 7B-13B scale.

**6. Gao, Y., Xiong, Y., Gao, X., Jia, K., Pan, J., Bi, Y., Dai, Y., Sun, J., et al. (2024).** "Retrieval-Augmented Generation for Large Language Models: A Survey." *arXiv:2312.10997.*
> The most comprehensive RAG survey to date. Covers Naive RAG, Advanced RAG, and Modular RAG paradigms. Reviews retrieval, generation, and augmentation techniques. Includes evaluation frameworks and benchmarks.

---

### B.2 Dense Retrieval and Embedding Papers

**7. Devlin, J., Chang, M.-W., Lee, K., and Toutanova, K. (2019).** "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding." *NAACL 2019, pp. 4171-4186.* [arXiv:1810.04805](https://arxiv.org/abs/1810.04805)
> The foundation for all modern dense retrieval and embedding models. Introduces bidirectional pre-training that enables rich contextual representations. BERT embeddings underpin DPR, Sentence-BERT, BGE, and most retrieval models used in RAG.

**8. Karpukhin, V., Oguz, B., Min, S., Lewis, P., Wu, L., Edunov, S., Chen, D., and Yih, W. (2020).** "Dense Passage Retrieval for Open-Domain Question Answering." *EMNLP 2020.* [arXiv:2004.04906](https://arxiv.org/abs/2004.04906)
> Introduces DPR, the dual-encoder framework for dense passage retrieval. Outperforms BM25 by 9-19% on top-20 retrieval accuracy. The blueprint for all subsequent dense retrieval systems including those used in modern RAG.

**9. Reimers, N. and Gurevych, I. (2019).** "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks." *EMNLP-IJCNLP 2019, pp. 3982-3992.* [arXiv:1908.10084](https://arxiv.org/abs/1908.10084)
> Introduces SBERT, which adapts BERT with siamese networks for efficient sentence embedding. Reduces semantic similarity search from 65 hours to 5 seconds for 10K sentences. The architectural foundation for the sentence-transformers library used by BGE models.

**10. Khattab, O. and Zaharia, M. (2020).** "ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT." *SIGIR 2020.* [arXiv:2004.12832](https://arxiv.org/abs/2004.12832)
> Introduces late interaction retrieval: independently encodes queries and documents, then uses MaxSim for fine-grained matching. Two orders-of-magnitude faster than full cross-attention models. ColBERTv2 later improves efficiency further.

**11. Xiao, S., Liu, Z., Zhang, P., and Muennighoff, N. (2023).** "C-Pack: Packaged Resources To Advance General Chinese Embedding." *arXiv:2309.07597.*
> Introduces the BGE (BAAI General Embedding) model family, including the bge-large-en-v1.5 model used in this thesis. Trained with RetroMAE pre-training and contrastive learning. Ranked 1st on MTEB and C-MTEB benchmarks upon release.

**12. Wang, L., Yang, N., Huang, X., Jiao, B., Yang, L., Jiang, D., Majumder, R., and Wei, F. (2024).** "Text Embeddings by Weakly-Supervised Contrastive Pre-training." *arXiv:2212.03533.*
> Introduces E5, a competing embedding family. First model to outperform BM25 on BEIR without labeled data. When fine-tuned, achieves best MTEB results with 40x fewer parameters than alternatives. Useful as a baseline comparison for BGE.

**13. Muennighoff, N., Tazi, N., Magne, L., and Reimers, N. (2023).** "MTEB: Massive Text Embedding Benchmark." *EACL 2023.* [arXiv:2210.07316](https://arxiv.org/abs/2210.07316)
> Introduces the standard benchmark for text embeddings: 8 tasks, 58 datasets, 112 languages. Finds no single embedding method dominates all tasks. Essential for contextualizing BGE-large-en-v1.5 performance claims in the thesis.

**14. Thakur, N., Reimers, N., Ruckle, A., Srivastava, A., and Gurevych, I. (2021).** "BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models." *NeurIPS 2021 Datasets and Benchmarks Track.* [arXiv:2104.08663](https://arxiv.org/abs/2104.08663)
> Introduces BEIR, 18 diverse retrieval datasets for zero-shot evaluation. Shows BM25 is a surprisingly strong baseline, and dense retrievers have room for improvement on out-of-distribution data. Important for discussing generalization of the embedding model.

---

### B.3 RAG Evaluation Papers

**15. Es, S., James, J., Espinosa Anke, L., and Schockaert, S. (2024).** "RAGAs: Automated Evaluation of Retrieval Augmented Generation." *EACL 2024 System Demonstrations, pp. 150-158.* [arXiv:2309.15217](https://arxiv.org/abs/2309.15217)
> Introduces the RAGAS framework with four core metrics: faithfulness, answer relevancy, context precision, and context recall. Reference-free evaluation using LLMs as judges. The most cited RAG evaluation paper. Open-source at github.com/explodinggradients/ragas.

**16. Saad-Falcon, J., Khattab, O., Potts, C., and Zaharia, M. (2024).** "ARES: An Automated Evaluation Framework for Retrieval-Augmented Generation Systems." *NAACL 2024, pp. 338-354.* [arXiv:2311.09476](https://arxiv.org/abs/2311.09476)
> Fine-tunes lightweight LM judges on synthetic data and uses Prediction-Powered Inference (PPI) for statistical guarantees. Accurate with only a few hundred human annotations. More rigorous than RAGAS for formal evaluation.

**17. Eibich, M. and Nagpal, C. (2024).** "ARAGOG: Advanced RAG Output Grading." *arXiv:2404.01037.*
> Empirically compares RAG techniques (HyDE, reranking, MMR, sentence window retrieval). Finds HyDE and LLM reranking most effective. Provides a methodology for comparative evaluation applicable to thesis experiments.

**18. Niu, S., Liu, Y., Wang, J., and Song, H. (2024).** "RAGTruth: A Hallucination Corpus for Developing Trustworthy Retrieval-Augmented Language Models." *ACL 2024.* [arXiv:2401.00396](https://arxiv.org/abs/2401.00396)
> Nearly 18,000 manually annotated RAG responses with word-level hallucination labels. Benchmarks hallucination frequencies across LLMs. Demonstrates that fine-tuned small LMs can match GPT-4 for hallucination detection.

**19. Chen, J., Lin, H., Han, X., and Sun, L. (2024).** "Benchmarking Large Language Models in Retrieval-Augmented Generation." *AAAI 2024, pp. 17754-17762.* [arXiv:2309.01431](https://arxiv.org/abs/2309.01431)
> Introduces the RGB benchmark evaluating four RAG abilities: noise robustness, negative rejection, information integration, and counterfactual robustness. Reveals significant weaknesses in negative rejection and counterfactual handling. Relevant for understanding LLM limitations in the climate RAG pipeline.

**20. Yang, Z., Liu, S., Sun, M., et al. (2024).** "Evaluation of Retrieval-Augmented Generation: A Survey." *arXiv:2405.07437.*
> Comprehensive survey of RAG evaluation methods and benchmarks. Covers retrieval, generation, and end-to-end evaluation. Provides a taxonomy of evaluation approaches useful for thesis methodology discussion.

**21. Fan, Y., et al. (2025).** "Retrieval Augmented Generation Evaluation in the Era of Large Language Models: A Comprehensive Survey." *arXiv:2504.14891.*
> The most recent survey on RAG evaluation. Reviews traditional and emerging evaluation approaches for performance, factual accuracy, safety, and efficiency. Catalogs RAG-specific datasets and frameworks.

---

### B.4 Climate/Geospatial AI Papers

**22. Rolnick, D., Donti, P. L., Kaack, L. H., et al. (2022).** "Tackling Climate Change with Machine Learning." *ACM Computing Surveys, Vol. 55, No. 2, Article 42 (96 pages).* [arXiv:1906.05433](https://arxiv.org/abs/1906.05433)
> The definitive survey on ML for climate change, co-authored by Yoshua Bengio and Demis Hassabis. Identifies high-impact problems from smart grids to disaster management. Provides the broader context for why climate data RAG systems matter.

**23. Vaghefi, S., Wang, Q., Muccione, V., Ni, J., Kraus, M., Bingler, J., Schimanski, T., Colesanti Senni, C., Webersinke, N., Huggel, C., and Leippold, M. (2023).** "ChatClimate: Grounding Conversational AI in Climate Science." *Communications Earth & Environment, Nature.* [arXiv:2304.05510](https://arxiv.org/abs/2304.05510)
> Enhances GPT-4 with IPCC AR6 reports for climate Q&A. The most directly comparable system to this thesis. Demonstrates that RAG with authoritative climate sources improves answer accuracy over vanilla LLMs. Published in a Nature journal.

**24. Webersinke, N., Kraus, M., Bingler, J., and Leippold, M. (2022).** "ClimateBERT: A Pretrained Language Model for Climate-Related Text." *AAAI 2022 Fall Symposium.* [arXiv:2110.12010](https://arxiv.org/abs/2110.12010)
> Pre-trains a transformer on 2M paragraphs of climate text. Achieves 48% improvement on masked language modeling and 3-36% error reduction on downstream climate tasks. Demonstrates the value of domain-specific pre-training for climate NLP.

**25. Thulke, D., et al. (2024).** "ClimateGPT: Towards AI Synthesizing Interdisciplinary Research on Climate Change." *arXiv:2401.09646.*
> An LLM designed for interdisciplinary climate change research across environmental science, economics, and social science. Uses RAG to ground responses in scientific literature. Another comparable system for thesis positioning.

**26. Lam, R., Sanchez-Gonzalez, A., Willson, M., et al. (2023).** "Learning Skillful Medium-Range Global Weather Forecasting." *Science, Vol. 382, Issue 6677, pp. 1416-1421.* [arXiv:2212.12794](https://arxiv.org/abs/2212.12794)
> Google DeepMind's GraphCast outperforms operational weather forecasting on 90% of verification targets. Demonstrates the potential of ML for climate/weather data processing. Context for why AI-assisted climate data access matters.

**27. Xiao, A., et al. (2024).** "Foundation Models for Remote Sensing and Earth Observation: A Survey." *arXiv:2410.16602.*
> Comprehensive survey of foundation models for remote sensing. Covers vision transformers, self-supervised learning, and multimodal approaches for earth observation. Relevant to understanding how AI is transforming geospatial data processing.

---

### B.5 Domain-Specific RAG Applications

**28. Zhang, T., Patil, S. G., Jain, N., Shen, S., Zaharia, M., Stoica, I., and Gonzalez, J. E. (2024).** "RAFT: Adapting Language Model to Domain Specific RAG." *arXiv:2403.10131.*
> Introduces Retrieval-Augmented Fine-Tuning, which trains models to ignore distractor documents and cite verbatim from relevant sources. Outperforms standard RAG in specialized domains. Directly relevant to improving climate domain RAG accuracy.

**29. Xiong, G., Jin, Q., Lu, Z., and Zhang, A. (2024).** "Benchmarking Retrieval-Augmented Generation for Medicine." *ACL 2024 Findings.* [Available at ACL Anthology](https://aclanthology.org/2024.findings-acl.372/)
> Benchmarks RAG for medical QA using knowledge graph augmentation. Tests Llama-2, GPT-3.5, and GPT-4. Demonstrates domain-specific challenges parallel to climate data RAG: specialized terminology, factual precision requirements, citation needs.

**30. Jiang, Z., et al. (2024).** "LongRAG: Enhancing Retrieval-Augmented Generation with Long-context LLMs." *arXiv:2406.15319.*
> Groups related documents into 4K-token retrieval units instead of traditional 100-word paragraphs. Achieves 62.7% EM on Natural Questions without training. Relevant to the climate RAG system's chunking strategy for large climate datasets.

---

### B.6 Chunking and Indexing Strategies

**31. Anthropic. (2024).** "Contextual Retrieval." *Anthropic Research Blog.*
> Introduces contextual chunking: prepending chunk-specific explanatory context before embedding. Combined with BM25 hybrid search, reduces retrieval failure by 67%. Directly applicable to improving the climate RAG system's chunking pipeline.

**32. Kamradt, G. (2024).** "Reconstructing Context: Evaluating Advanced Chunking Strategies for Retrieval-Augmented Generation." *arXiv:2504.19754.*
> Compares fixed-size, recursive, semantic, and late chunking strategies. Finds that contextual retrieval preserves semantic coherence but requires more compute, while late chunking offers better efficiency at the cost of relevance.

**33. Sarmah, B., et al. (2024).** "Comparative Evaluation of Advanced Chunking for Retrieval-Augmented Generation in Large Language Models for Clinical Decision Support." *Bioengineering, 12(11), 1194.*
> Adaptive chunking achieves 87% accuracy vs. 50% baseline in clinical RAG. Proposition and semantic strategies improve over naive chunking but less than adaptive approaches. Demonstrates the importance of chunking strategy choice.

**34. Santhanam, K., Khattab, O., Saad-Falcon, J., Potts, C., and Zaharia, M. (2022).** "ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction." *NAACL 2022.* [arXiv:2112.01488](https://arxiv.org/abs/2112.01488)
> Improves ColBERT with residual compression, reducing storage by 6-10x. Combines denoised supervision with cross-encoder knowledge distillation. Relevant as an alternative retrieval architecture to dense bi-encoder approaches.

---

## Part C: Recommended Evaluation Strategy

### C.1 System Context

The climate data RAG system under evaluation has the following characteristics:

- **Embedding model:** BAAI/bge-large-en-v1.5 (1024-dimensional, cosine distance)
- **Vector database:** Qdrant (HNSW indexing, cosine metric)
- **Corpus:** 233 catalog entries from an Excel catalog, generating ~1000+ Qdrant points through chunking
- **LLM backends:** OpenRouter, Groq, and Ollama (multiple models, API-based)
- **Query types:** Natural language questions about climate datasets (availability, variables, spatial/temporal coverage, access methods)

### C.2 Metrics to Implement

#### Tier 1: Must-Have (Core Evaluation)

| Metric | Component | Implementation | Priority |
|--------|-----------|---------------|----------|
| **Recall@5** | Retrieval | Count relevant chunks in top-5 results | Highest |
| **Recall@10** | Retrieval | Count relevant chunks in top-10 results | Highest |
| **MRR** | Retrieval | Reciprocal rank of first relevant chunk | High |
| **Faithfulness** | Generation | RAGAS faithfulness metric (LLM-as-judge) | Highest |
| **Answer Relevancy** | Generation | RAGAS answer relevancy metric | High |

These five metrics cover both retrieval and generation quality and can be computed with modest annotation effort.

#### Tier 2: Should-Have (Comprehensive Evaluation)

| Metric | Component | Implementation | Priority |
|--------|-----------|---------------|----------|
| **nDCG@5** | Retrieval | With graded relevance labels (0/1/2) | Medium |
| **Context Precision** | Pipeline | RAGAS context precision | Medium |
| **Answer Correctness** | Pipeline | RAGAS answer correctness vs. gold answers | Medium |
| **Precision@5** | Retrieval | Fraction of relevant in top-5 | Medium |

#### Tier 3: Nice-to-Have (Extended Analysis)

| Metric | Component | Purpose |
|--------|-----------|---------|
| **Hit@1, Hit@3** | Retrieval | Sanity check for retrieval |
| **Context Recall** | Pipeline | RAGAS context recall with ground truth |
| **Latency** | System | End-to-end query response time |
| **LLM comparison** | Generation | Same queries across Ollama/Groq/OpenRouter |

### C.3 Test Set Creation

#### Size Recommendation

- **Minimum:** 50 query-answer pairs (sufficient for thesis-level evaluation)
- **Recommended:** 80-100 query-answer pairs (enables statistical significance testing)
- **Ideal:** 150+ pairs (enables train/test splits for metric calibration)

#### Query Categories

Design queries to cover the system's intended use cases:

| Category | Count | Example Query |
|----------|-------|---------------|
| **Dataset discovery** | 15-20 | "What climate datasets cover precipitation in Central Europe?" |
| **Variable/parameter lookup** | 15-20 | "Which datasets include sea surface temperature at daily resolution?" |
| **Spatial coverage** | 10-15 | "Are there gridded datasets available for the Czech Republic?" |
| **Temporal coverage** | 10-15 | "What datasets provide historical climate data before 1950?" |
| **Access/download info** | 10-15 | "How can I download ERA5 reanalysis data?" |
| **Comparison queries** | 5-10 | "What is the difference between CORDEX and E-OBS datasets?" |
| **Negative/out-of-scope** | 5-10 | "What is the current stock price of Apple?" |

#### Annotation Guidelines

For each query, annotate:

1. **Query text** (natural language question)
2. **Expected answer** (1-3 sentence reference answer written by a domain expert)
3. **Relevant chunks** (list of Qdrant point IDs or catalog entry row indices that contain the answer)
4. **Relevance grade** for each retrieved chunk:
   - 0 = irrelevant
   - 1 = partially relevant (contains related but insufficient information)
   - 2 = highly relevant (directly answers or substantially contributes to the answer)
5. **Difficulty rating** (easy / medium / hard)

#### Annotation Process

1. Generate candidate queries by reviewing the Excel catalog entries.
2. For each query, run the current system and examine the top-10 retrieved chunks.
3. Label each chunk's relevance (0/1/2).
4. Write a reference answer based on the catalog data (not the system's generated answer).
5. Have a second annotator verify 20% of the annotations for inter-annotator agreement.

### C.4 Baseline Comparisons

Report the following baselines to contextualize results:

| Baseline | Description | Purpose |
|----------|-------------|---------|
| **BM25 (lexical)** | Elasticsearch/Whoosh keyword search on the same corpus | Shows the value of semantic embeddings over keyword matching |
| **Random retrieval** | Randomly sample k chunks from the corpus | Lower bound on retrieval performance |
| **No-RAG LLM** | Ask the LLM the same questions without any retrieved context | Shows the value of retrieval augmentation |
| **Metadata-only embedding** | Use only catalog metadata (no raster/file content) | Shows the value of ingesting actual data files |
| **Alternative embedding model** | E5-large or all-MiniLM-L6-v2 on the same corpus | Validates the choice of BGE-large-en-v1.5 |

### C.5 Experiment Protocol

#### Step 1: Retrieval Evaluation (Without LLM)

1. Run all test queries against Qdrant.
2. Retrieve top-10 chunks per query.
3. Compute Recall@5, Recall@10, MRR, nDCG@5, Precision@5.
4. Compare against BM25 and random baselines.
5. Analyze failure cases (queries with Recall@10 = 0).

#### Step 2: Generation Evaluation (With LLM)

1. For each test query, run the full RAG pipeline.
2. Compute RAGAS Faithfulness and Answer Relevancy using an LLM judge.
3. Compute Answer Correctness against reference answers.
4. Compare across LLM backends (Ollama local vs. Groq vs. OpenRouter).
5. Analyze hallucination patterns and failure modes.

#### Step 3: Ablation Studies

1. **Chunking strategy:** Compare different chunk sizes (256, 512, 1024 tokens).
2. **Top-k selection:** Compare k=3, k=5, k=10 for retrieval.
3. **With/without metadata enrichment:** Test the value of location and temporal metadata.
4. **Phase comparison:** Evaluate retrieval quality for Phase 0 (metadata-only) vs. Phase 1 (full data) entries.

### C.6 What to Report in the Thesis

#### Results Tables

1. **Retrieval metrics table:** Recall@5, Recall@10, MRR, nDCG@5, Precision@5 for each configuration.
2. **Generation metrics table:** Faithfulness, Answer Relevancy, Answer Correctness for each LLM backend.
3. **Baseline comparison table:** Side-by-side comparison with all baselines.
4. **Ablation results table:** Impact of chunking, top-k, metadata, and processing phase.

#### Visualizations

1. **Recall@k curve:** Plot Recall at k=1,3,5,10,20 to show retrieval saturation.
2. **Per-category performance:** Bar chart of metrics broken down by query category.
3. **Faithfulness vs. Answer Correctness scatter:** Shows the relationship between grounding and accuracy.
4. **Failure case analysis:** Table of the worst-performing queries with root cause analysis.

#### Discussion Points

1. **Embedding model suitability:** Is BGE-large-en-v1.5 (a general-purpose model) adequate for climate-specific terminology, or would domain-specific fine-tuning (a la ClimateBERT) improve results?
2. **Metadata vs. data embeddings:** How much does ingesting actual raster data improve retrieval over catalog metadata alone?
3. **LLM faithfulness:** Do different LLM backends (local Ollama vs. cloud APIs) exhibit different hallucination rates?
4. **Chunking impact:** How sensitive is retrieval quality to chunk size and overlap settings?
5. **Scalability analysis:** How do metrics change as the corpus grows from 233 entries to potentially thousands?
6. **Limitations:** Clearly state the test set size, annotation biases, and LLM-as-judge limitations.

#### Statistical Reporting

- Report mean and standard deviation across query categories.
- Use bootstrap confidence intervals (95%) for overall metrics.
- For baseline comparisons, use paired t-tests or Wilcoxon signed-rank tests.
- Report inter-annotator agreement (Cohen's kappa) for the annotation subset.

---

## Appendix: Quick-Start Implementation Guide

### Installing RAGAS for Evaluation

```bash
pip install ragas
```

### Minimal RAGAS Evaluation Script

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from datasets import Dataset

# Prepare evaluation data
eval_data = {
    "question": ["What datasets cover precipitation in Europe?"],
    "answer": ["ERA5 and E-OBS both provide precipitation data covering Europe..."],
    "contexts": [["ERA5 is a reanalysis dataset from ECMWF covering 1979-present...",
                   "E-OBS provides gridded observational data for Europe..."]],
    "ground_truth": ["ERA5 reanalysis and E-OBS observational gridded dataset both cover precipitation for Europe."]
}

dataset = Dataset.from_dict(eval_data)

results = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
)

print(results)
```

### Computing Retrieval Metrics Manually

```python
import numpy as np

def recall_at_k(relevant_ids: set, retrieved_ids: list, k: int) -> float:
    """Recall@k: fraction of relevant documents in top-k."""
    if not relevant_ids:
        return 0.0
    retrieved_at_k = set(retrieved_ids[:k])
    return len(relevant_ids & retrieved_at_k) / len(relevant_ids)

def mrr(relevant_ids: set, retrieved_ids: list) -> float:
    """Mean Reciprocal Rank for a single query."""
    for i, doc_id in enumerate(retrieved_ids):
        if doc_id in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0

def ndcg_at_k(relevance_scores: list, k: int) -> float:
    """nDCG@k with pre-computed relevance scores for retrieved docs."""
    scores = relevance_scores[:k]
    dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(scores))
    ideal = sorted(relevance_scores, reverse=True)[:k]
    idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0
```

---

## References Summary

| # | Short Citation | Year | Venue | Topic |
|---|---------------|------|-------|-------|
| 1 | Lewis et al. | 2020 | NeurIPS | Original RAG model |
| 2 | Izacard & Grave | 2021 | EACL | Fusion-in-Decoder |
| 3 | Borgeaud et al. | 2022 | ICML | RETRO |
| 4 | Shi et al. | 2024 | NAACL | REPLUG |
| 5 | Asai et al. | 2024 | ICLR | Self-RAG |
| 6 | Gao et al. | 2024 | arXiv | RAG survey |
| 7 | Devlin et al. | 2019 | NAACL | BERT |
| 8 | Karpukhin et al. | 2020 | EMNLP | DPR |
| 9 | Reimers & Gurevych | 2019 | EMNLP | Sentence-BERT |
| 10 | Khattab & Zaharia | 2020 | SIGIR | ColBERT |
| 11 | Xiao et al. | 2023 | arXiv | BGE embeddings |
| 12 | Wang et al. | 2024 | arXiv | E5 embeddings |
| 13 | Muennighoff et al. | 2023 | EACL | MTEB benchmark |
| 14 | Thakur et al. | 2021 | NeurIPS | BEIR benchmark |
| 15 | Es et al. | 2024 | EACL | RAGAS framework |
| 16 | Saad-Falcon et al. | 2024 | NAACL | ARES framework |
| 17 | Eibich & Nagpal | 2024 | arXiv | ARAGOG |
| 18 | Niu et al. | 2024 | ACL | RAGTruth |
| 19 | Chen et al. | 2024 | AAAI | RGB benchmark |
| 20 | Yang et al. | 2024 | arXiv | RAG evaluation survey |
| 21 | Fan et al. | 2025 | arXiv | RAG evaluation survey (latest) |
| 22 | Rolnick et al. | 2022 | ACM CS | ML for climate change |
| 23 | Vaghefi et al. | 2023 | Nature CE&E | ChatClimate |
| 24 | Webersinke et al. | 2022 | AAAI FS | ClimateBERT |
| 25 | Thulke et al. | 2024 | arXiv | ClimateGPT |
| 26 | Lam et al. | 2023 | Science | GraphCast weather ML |
| 27 | Xiao et al. | 2024 | arXiv | RS foundation models survey |
| 28 | Zhang et al. | 2024 | arXiv | RAFT domain-specific RAG |
| 29 | Xiong et al. | 2024 | ACL Findings | RAG for medicine |
| 30 | Jiang et al. | 2024 | arXiv | LongRAG |
| 31 | Anthropic | 2024 | Blog | Contextual retrieval |
| 32 | Kamradt et al. | 2024 | arXiv | Chunking strategies |
| 33 | Sarmah et al. | 2024 | Bioengineering | Adaptive chunking |
| 34 | Santhanam et al. | 2022 | NAACL | ColBERTv2 |
| 35 | Salemi & Zamani | 2024 | SIGIR | eRAG: per-document downstream eval |
| 36 | Chen et al. | 2024 | arXiv | BGE-M3 (BAAI/bge-m3) model |
| 37 | — | 2024 | arXiv:2410.23902 | Responsible RAG for Climate |
| 38 | — | 2025 | arXiv:2512.02251 | CAIRNS: Climate adaptation QA (7-dim rubric) |
| 39 | — | 2024 | arXiv:2405.07437 | RAG Evaluation Survey (comprehensive) |

---

## Part D: Implementation Plan for Climate RAG Evaluation

*Added: February 2026*

This section describes the concrete evaluation methodology being implemented for the Climate Data RAG pipeline, building on the metrics and frameworks surveyed in Parts A–C.

### D.1 Three-Tier Evaluation Architecture

The evaluation is structured in three tiers of increasing complexity:

**Tier 1: Retrieval Quality (embedding + Qdrant)**

Measures how well the retrieval component returns relevant documents before any generation occurs. Uses a golden test set (`tests/fixtures/golden_queries.json`) with 50-100 climate questions and annotated relevant chunk IDs.

Metrics:
- **Hit@5, Hit@10** — baseline pass/fail: does the top-K contain at least one relevant chunk?
- **MRR@10** — Mean Reciprocal Rank: how quickly does the first relevant result appear?
- **NDCG@10** — Normalized Discounted Cumulative Gain: overall ranking quality (standard MTEB metric)
- **Recall@K** at K=3, 5, 10 — coverage: what fraction of relevant chunks are retrieved?

Implementation: Direct Qdrant client queries with BGE-M3 embeddings, compared against golden set annotations.

**Tier 2: End-to-End RAG (RAGAS framework)**

Uses the RAGAS framework (Es et al., EACL 2024, arXiv:2309.15217) for reference-free LLM-judge evaluation:
- **Faithfulness** — are generated claims grounded in retrieved context? (Target: ≥ 0.85)
- **Context Precision** — did we retrieve relevant chunks? (Target: ≥ 0.70)
- **Context Recall** — did we retrieve ALL relevant chunks? (Target: ≥ 0.75)
- **Answer Relevancy** — does the answer address the question? (Target: ≥ 0.80)

Custom metric:
- **Numerical Coverage** — fraction of key numbers (temperatures, dates, thresholds) from source data preserved in the answer. Critical for climate data accuracy. (Target: ≥ 0.90)

**Tier 3: Embedding Space Analysis (intrinsic)**

Evaluates the structure of the BGE-M3 embedding space:
- Cosine similarity distribution within vs. between climate variables
- t-SNE/UMAP visualization of embedding clusters by dataset/variable
- Nearest-neighbor sanity checks (temperature chunks should cluster together)

### D.2 Golden Test Set Construction

The golden test set is stored in `tests/fixtures/golden_queries.json` with the following schema:

```json
{
  "queries": [
    {
      "id": "q001",
      "query": "What is the global mean temperature anomaly...",
      "category": "variable-specific | spatial | temporal | cross-variable | methodological",
      "dataset": "CMIP6 | CRU | E-OBS | null",
      "relevant_chunk_ids": ["chunk_id_1", "chunk_id_2"],
      "key_numbers": ["1.45", "2023"],
      "difficulty": "easy | medium | hard"
    }
  ]
}
```

**Construction guidelines:**
1. Cover all 6 query categories (variable-specific, dataset-specific, spatial, temporal, cross-variable, methodological)
2. Include queries at easy/medium/hard difficulty levels
3. Annotate 2-5 relevant chunk IDs per query by manual inspection of Qdrant contents
4. For numerical coverage metric, annotate key numbers that should appear in answers
5. Target 50-100 queries for statistical significance (currently 10 seed queries provided)

### D.3 RAGAS Integration

```python
from ragas import evaluate
from ragas.metrics import faithfulness, context_precision, context_recall, answer_relevancy
from datasets import Dataset

# Build evaluation dataset from golden queries + RAG pipeline
eval_data = {
    "question": [q["query"] for q in golden_queries],
    "answer": [rag_pipeline.query(q["query"]) for q in golden_queries],
    "contexts": [rag_pipeline.retrieve(q["query"]) for q in golden_queries],
    "ground_truth": [q.get("ground_truth", "") for q in golden_queries],
}
dataset = Dataset.from_dict(eval_data)

result = evaluate(dataset, metrics=[faithfulness, context_precision, context_recall, answer_relevancy])
```

### D.4 Climate-Specific Evaluation Considerations

Standard RAG evaluation metrics are necessary but not sufficient for climate data. Additional considerations:

1. **Numerical precision:** Climate answers must preserve exact values (temperatures to 0.1°C, dates, thresholds). The custom Numerical Coverage metric addresses this.

2. **Unit consistency:** Answers should use correct units (°C, mm, hPa) and not mix different unit systems.

3. **Temporal specificity:** The system must distinguish between historical observations and future projections, and between different time periods.

4. **Spatial accuracy:** Retrieved chunks should match the geographic region of the query. A query about "Czech Republic" should not return global averages.

5. **Source attribution:** Answers should be traceable to specific datasets and variables, enabling verification.

These considerations align with the CAIRNS framework (arXiv:2512.02251) which proposes a 7-dimension evaluation rubric for climate adaptation QA, and with the responsible RAG principles outlined in arXiv:2410.23902.

### D.5 Additional Academic References

| Paper | Venue | Key Contribution | Relevance |
|-------|-------|-----------------|-----------|
| eRAG (Salemi & Zamani, arXiv:2404.13781) | SIGIR 2024 | Per-document downstream evaluation | Alternative to aggregate metrics |
| M3-Embedding (Chen et al., arXiv:2402.03216) | 2024 | BGE-M3 model paper | Our embedding model |
| Responsible RAG for Climate (arXiv:2410.23902) | 2024 | Climate-domain RAG evaluation | Domain-specific eval considerations |
| CAIRNS (arXiv:2512.02251) | 2025 | Climate adaptation QA with 7-dim rubric | Gold standard for climate QA eval |
| RAG Evaluation Survey (arXiv:2405.07437) | 2024 | Comprehensive eval methods survey | Taxonomy of evaluation approaches |

### D.6 Implementation Status

| Component | Status | File |
|-----------|--------|------|
| Golden test set (seed) | ✅ Done (10 queries) | `tests/fixtures/golden_queries.json` |
| Tier 1: Retrieval metrics | ✅ Done (skeleton) | `tests/test_rag_evaluation.py` |
| Tier 2: RAGAS integration | ⬜ Pending (needs RAG endpoint) | `tests/test_rag_evaluation.py` |
| Tier 3: Embedding analysis | ✅ Done (skeleton) | `tests/test_rag_evaluation.py` |
| Golden test set expansion (50-100 queries) | ⬜ Pending | `tests/fixtures/golden_queries.json` |
| Numerical coverage metric | ⬜ Pending (needs RAG endpoint) | `tests/test_rag_evaluation.py` |
| t-SNE/UMAP visualization | ⬜ Pending | — |
