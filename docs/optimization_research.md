# Raster Pipeline Performance Optimization — Academic Research & Rationale

## Context

The catalog batch pipeline (Phase 1) processes climate datasets from the D1.1.xlsx catalog
by downloading NetCDF/GeoTIFF files, chunking them spatially and temporally, generating
embeddings for each chunk, and upserting them into Qdrant vector database. Fine-grained
spatial chunks (lat=10, lon=10 grid points) are intentionally preserved so data scientists
can query specific small regions and aggregate later via RAG.

For large datasets like SLOCLIM (~84,000 chunks), the pipeline took 4+ hours. Profiling
showed 86% CPU utilization with 4.1 GB RAM, with embedding generation on CPU as the
dominant bottleneck.

## Bottleneck Analysis

| Stage                        | Time (84K chunks) | CPU   | Notes                                |
|------------------------------|-------------------|-------|--------------------------------------|
| HTTP Download (streaming)    | 1–17 min          | Low   | Network-bound, properly streamed     |
| NetCDF Pre-load (xarray)     | 5–30 sec          | Burst | Pre-loaded if var < 2 GB             |
| Chunk iteration + numpy stats| ~1–2 min          | Med   | 8 stats per chunk, small arrays      |
| Text generation (84K)        | ~30–60 sec        | Low   | String formatting                    |
| **Embedding (84K, CPU)**     | **1.5–4.5 hours** | **86%** | **bge-m3 (568M params) on CPU**    |

> **Update**: Switched from `BAAI/bge-m3` (568M params) to `BAAI/bge-large-en-v1.5` (335M params).
> Both produce 1024-dim vectors with COSINE distance, so no Qdrant collection changes needed.
> bge-large-en-v1.5 is compatible with ONNX Runtime (bge-m3 was not, due to KeyError on
> `last_hidden_state`), enabling the 2-3x CPU speedup from optimization #1 below.
| Qdrant upsert (168 batches)  | ~1–5 min          | Low   | 2000-point gRPC batches (v1.17.0)    |

## Optimizations Implemented

### 1. ONNX Runtime Backend for Embedding Model

**What**: Replaced default PyTorch inference with ONNX Runtime backend for sentence-transformers.

**Why**: ONNX Runtime applies graph-level optimizations (operator fusion, constant folding,
memory planning) that PyTorch eager mode cannot. On CPU, this yields 2–3x speedup for
transformer inference without any accuracy loss.

**Academic references**:
- Sentence Transformers v5.1.0 release notes — native ONNX/OpenVINO backend support:
  https://github.com/huggingface/sentence-transformers/releases/tag/v5.1.0
- Sentence Transformers efficiency documentation:
  https://sbert.net/docs/sentence_transformer/usage/efficiency.html
- Philipp Schmid, "Accelerate Sentence Transformers with Hugging Face Optimum":
  https://www.philschmid.de/optimize-sentence-transformers
- ONNX Runtime performance benchmarks:
  https://onnxruntime.ai/docs/performance/benchmarks.html

**Expected speedup**: 2–3x for embedding generation (the dominant bottleneck).

**Implementation**: `text_models.py` — auto-detects ONNX Runtime availability and uses
`SentenceTransformer(model, backend="onnx")`. Falls back to PyTorch if unavailable.

### 2. gRPC Transport for Qdrant

**What**: Switched Qdrant client from REST (HTTP/JSON) to gRPC (HTTP/2 + Protocol Buffers).

**Why**: gRPC uses binary serialization (protobuf) instead of JSON, and HTTP/2 multiplexing
instead of HTTP/1.1 request-response. For batch vector upserts with 1024-dimensional
float vectors, binary serialization eliminates JSON encoding/decoding overhead (~8 MB
of floats per batch). HTTP/2 multiplexing allows pipelining multiple requests.

**Academic references**:
- Qdrant large-scale ingestion guide (recommends gRPC for bulk operations):
  https://qdrant.tech/course/essentials/day-4/large-scale-ingestion/
- Qdrant performance optimization documentation:
  https://qdrant.tech/documentation/guides/optimize/
- gRPC vs REST performance comparison for ML serving:
  https://grpc.io/docs/what-is-grpc/introduction/

**Expected speedup**: 1.5–2x for upsert throughput.

**Implementation**: `database.py` — `prefer_grpc=True` with explicit `grpc_port=6334`.

**Note**: gRPC required upgrading both Qdrant server (v1.11.0 → v1.17.0) and qdrant-client
(1.11.x → 1.17.x) to resolve a protobuf mismatch (`PayloadIncludeSelector got list` error)
that occurred when client and server versions were out of sync.

### 3. Increased Upsert Batch Size (500 → 2000)

**What**: Increased Qdrant upsert batch from 500 to 2000 points per call.

**Why**: Each upsert call has fixed overhead (network round-trip, server-side transaction
setup). For 1024-dim float32 vectors, 2000 points ≈ 8 MB per batch — well within
Qdrant's recommended limits. Qdrant documentation recommends batch sizes of
1,000–10,000 for datasets in the 100K–1M range.

**Academic references**:
- Qdrant indexing optimization for bulk uploads:
  https://qdrant.tech/articles/indexing-optimization/
- Qdrant batch upload best practices:
  https://qdrant.tech/documentation/concepts/points/#upload-points

**Expected speedup**: 1.3–1.5x (4x fewer network round-trips).

**Implementation**: `database.py` — `BATCH_SIZE = 2000`.

### 4. Pipelined Upserts (Background Thread)

**What**: Qdrant upserts now run in a background thread while the next embedding batch
is being computed on CPU.

**Why**: The pipeline was strictly sequential: embed → upsert → embed → upsert.
Since embedding is CPU-bound and upserting is I/O-bound (network), these can overlap.
While the CPU computes embeddings for batch N+1, the network sends batch N to Qdrant.
This is a classic producer-consumer pipeline optimization.

**Academic references**:
- Pipeline parallelism pattern in data-intensive systems:
  Kleppmann, M. (2017). "Designing Data-Intensive Applications", O'Reilly. Chapter 10.
- Overlap of computation and communication in ML pipelines:
  https://arxiv.org/abs/1806.03377 (PipeDream: Generalized Pipeline Parallelism for DNN Training)
- Python concurrent.futures for I/O overlap:
  https://docs.python.org/3/library/concurrent.futures.html

**Expected speedup**: Eliminates I/O wait time during embedding computation (~5–15%
of total time saved, depending on upsert latency).

**Implementation**: `batch_orchestrator.py` — `_flush_batch()` submits upsert to
`ThreadPoolExecutor`, next `_flush_batch()` waits for previous completion before
starting new embedding.

## Further Optimization Opportunities (Not Yet Implemented)

### h5netcdf Engine for NetCDF Reading
- h5netcdf measured up to 3.8x faster than netcdf4 for certain workloads.
- Reference: https://github.com/pydata/xarray/discussions/9968

### Disable HNSW During Bulk Ingestion
- Setting `m=0` in HNSW config during initial load avoids incremental index updates.
- Re-enable after bulk load completes. Expected 2–3x for initial loads of >10K points.
- Reference: https://qdrant.tech/articles/indexing-optimization/

### Zarr/Kerchunk for Repeated Dataset Access
- Converting NetCDF to Zarr or generating Kerchunk reference files enables parallel
  chunk-level reads without full file download.
- CarbonPlan demonstrated Kerchunk reference generation in 3 min vs 20 min direct download.
- Reference: https://carbonplan.org/blog/kerchunk-climate-data
- Reference: arXiv 2207.09503 — "A Comparison of HDF5, Zarr, and netCDF4"

### INT8 Quantized ONNX Model
- Further 1.5–2x speedup on top of ONNX by using quantized model weights.
- Reference: https://sbert.net/docs/sentence_transformer/usage/efficiency.html

## Cumulative Expected Impact

Applying optimizations 1–4: estimated **3–4x overall speedup** for Phase 1 pipeline.
SLOCLIM processing time: ~4 hours → ~1–1.5 hours.
