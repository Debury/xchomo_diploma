from typing import List, Optional
import logging
import os

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError("pip install sentence-transformers")

logger = logging.getLogger(__name__)
_CACHE = {}


def _detect_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            logger.info(f"CUDA available: {name}")
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _detect_backend(model_name: str = "", device: str = "") -> str:
    """Pick the fastest available backend for embeddings.

    Priority:
    1. PyTorch + CUDA  — fastest (10-50x vs CPU), uses GPU natively
    2. ONNX + CUDAExecutionProvider — fast, but rarely installed
    3. ONNX + CPUExecutionProvider — 2-3x faster than PyTorch on CPU
    4. PyTorch CPU — fallback

    Note: bge-m3 is NOT compatible with optimum ONNX export (KeyError on
    'last_hidden_state').  Only enable ONNX for models known to work.
    """
    # bge-m3 uses a non-standard output schema that optimum cannot handle
    if "bge-m3" in model_name:
        logger.info(f"ONNX backend disabled for {model_name} (incompatible output schema)")
        return "torch"

    # If CUDA is available, PyTorch on GPU beats ONNX on CPU
    if device == "cuda":
        try:
            import onnxruntime as ort
            providers = ort.get_available_providers()
            if "CUDAExecutionProvider" in providers:
                logger.info("ONNX Runtime with CUDA provider — using 'onnx' backend on GPU")
                return "onnx"
        except ImportError:
            pass
        logger.info("CUDA available — using PyTorch backend on GPU (faster than ONNX on CPU)")
        return "torch"

    # CPU-only: ONNX is 2-3x faster than PyTorch
    try:
        import onnxruntime  # noqa: F401
        logger.info("ONNX Runtime available — using 'onnx' backend for CPU embeddings")
        return "onnx"
    except ImportError:
        logger.info("ONNX Runtime not installed — using default PyTorch backend")
        return "torch"


class TextEmbedder:
    def __init__(self, model_name="BAAI/bge-large-en-v1.5", device: Optional[str] = None):
        self.name = model_name
        self.device = device or os.getenv("EMBEDDING_DEVICE") or _detect_device()
        backend = os.getenv("EMBEDDING_BACKEND") or _detect_backend(model_name, self.device)
        cache_key = f"{model_name}@{self.device}@{backend}"
        if cache_key in _CACHE:
            self.model = _CACHE[cache_key]
        else:
            logger.info(f"Loading {model_name} on {self.device} (backend={backend})...")
            if backend == "onnx":
                try:
                    self.model = SentenceTransformer(
                        model_name, device=self.device, backend="onnx"
                    )
                except Exception as e:
                    logger.warning(f"ONNX backend failed ({e}), falling back to PyTorch")
                    self.model = SentenceTransformer(model_name, device=self.device)
            else:
                self.model = SentenceTransformer(model_name, device=self.device)
            _CACHE[cache_key] = self.model

        self.instruction = "Represent this sentence for searching relevant passages: " if "bge" in model_name else ""
        logger.info(f"TextEmbedder ready: {model_name} on {self.device}")

    def embed_queries(self, queries: List[str], batch_size: int = 128):
        inputs = [self.instruction + q for q in queries]
        return self.model.encode(inputs, normalize_embeddings=True, convert_to_numpy=True, batch_size=batch_size)

    def embed_documents(self, texts: List[str], batch_size: int = 512):
        return self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True, batch_size=batch_size)


class Reranker:
    """Cross-encoder reranker using BAAI/bge-reranker-v2-m3.

    Re-scores (query, passage) pairs for more accurate ranking than
    bi-encoder cosine similarity alone.

    Reference: Xiao et al., "C-Pack: Packaged Resources To Advance
    General Chinese Embedding", arXiv:2309.07597
    """

    DEFAULT_MODEL = "BAAI/bge-reranker-v2-m3"

    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        self.name = model_name or self.DEFAULT_MODEL
        self.device = device or os.getenv("RERANKER_DEVICE") or _detect_device()
        cache_key = f"reranker:{self.name}@{self.device}"

        if cache_key in _CACHE:
            self.model = _CACHE[cache_key]
        else:
            logger.info(f"Loading cross-encoder {self.name} on {self.device}...")
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(self.name, device=self.device)
            _CACHE[cache_key] = self.model
        logger.info(f"Reranker ready: {self.name} on {self.device}")

    def rerank(
        self,
        query: str,
        passages: List[str],
        top_k: Optional[int] = None,
    ) -> List[dict]:
        """Score (query, passage) pairs and return sorted indices with scores.

        Args:
            query: The search query.
            passages: List of passage texts to rerank.
            top_k: Return only the top-k results. None returns all.

        Returns:
            List of {"index": int, "score": float} sorted by score descending.
        """
        if not passages:
            return []

        pairs = [[query, p] for p in passages]
        scores = self.model.predict(pairs)

        ranked = sorted(
            [{"index": i, "score": float(s)} for i, s in enumerate(scores)],
            key=lambda x: x["score"],
            reverse=True,
        )

        if top_k is not None:
            ranked = ranked[:top_k]
        return ranked