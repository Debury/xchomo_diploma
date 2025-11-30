"""
RAG (Retrieval-Augmented Generation) pipeline for climate Q&A.

Orchestrates:
1. Query encoding with text embedding model
2. Vector search with metadata filtering
3. Context building from retrieved chunks
4. LLM answer generation

Example:
    >>> from src.embeddings.vector_index import VectorIndex
    >>> from src.embeddings.text_embeddings import get_text_embedder
    >>> from src.llm.ollama_client import OllamaClient
    >>> from src.llm.rag_pipeline import RAGPipeline
    >>> 
    >>> # Load index and models
    >>> index = VectorIndex.load("climate_index.pkl")
    >>> text_embedder = get_text_embedder("bge-large")
    >>> llm = OllamaClient()
    >>> 
    >>> # Create pipeline
    >>> rag = RAGPipeline(index=index, text_embedder=text_embedder, llm_client=llm)
    >>> 
    >>> # Ask question
    >>> answer = rag.ask("What is the global temperature trend since 2000?")
    >>> print(answer)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from ..index.vector_index import VectorIndex, SearchResult

# LLM client import - compatible with existing src/llm structure
try:
    from src.llm.ollama_client import OllamaClient
except ImportError:
    # Fallback for when used standalone
    OllamaClient = None

logger = logging.getLogger(__name__)


@dataclass
class RAGContext:
    """Context retrieved for a query."""
    
    query: str
    results: List[SearchResult]
    formatted_context: str
    metadata_summary: Dict[str, Any]


class RAGPipeline:
    """
    Complete RAG pipeline for climate data Q&A.
    
    Components:
    - Text embedder for query encoding
    - Vector index for similarity search
    - LLM client for answer generation
    """
    
    def __init__(
        self,
        index: VectorIndex,
        text_embedder: Callable[[str], np.ndarray],
        llm_client: OllamaClient,
        top_k: int = 5,
        temperature: float = 0.7,
    ):
        """
        Initialize RAG pipeline.
        
        Args:
            index: Vector index with climate embeddings
            text_embedder: Function to encode queries to vectors
            llm_client: LLM client for answer generation
            top_k: Number of chunks to retrieve
            temperature: LLM sampling temperature
        """
        self.index = index
        self.text_embedder = text_embedder
        self.llm_client = llm_client
        self.top_k = top_k
        self.temperature = temperature
    
    def retrieve(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None,
    ) -> RAGContext:
        """
        Retrieve relevant chunks for a query.
        
        Args:
            query: User question
            filters: Optional metadata filters
            top_k: Override default top_k
            
        Returns:
            RAGContext with retrieved results and formatted context
        """
        k = top_k or self.top_k
        
        # Encode query
        logger.info(f"Encoding query: {query}")
        query_embedding = self.text_embedder(query)
        
        # Search vector index
        logger.info(f"Searching for top-{k} similar chunks...")
        results = self.index.search(
            query_embedding=query_embedding,
            k=k,
            filters=filters,
        )
        
        if not results:
            logger.warning("No results found for query")
            return RAGContext(
                query=query,
                results=[],
                formatted_context="No relevant data found.",
                metadata_summary={},
            )
        
        # Format context for LLM
        formatted_context = self._format_context(results)
        metadata_summary = self._summarize_metadata(results)
        
        logger.info(f"Retrieved {len(results)} chunks with avg score {np.mean([r.score for r in results]):.3f}")
        
        return RAGContext(
            query=query,
            results=results,
            formatted_context=formatted_context,
            metadata_summary=metadata_summary,
        )
    
    def _format_context(self, results: List[SearchResult]) -> str:
        """Format retrieved results into LLM-friendly context."""
        lines = ["RETRIEVED CLIMATE DATA CONTEXT:\n"]
        
        for i, result in enumerate(results, 1):
            meta = result.metadata
            
            # Build context entry
            entry = f"\n[Chunk {i}] (Similarity: {result.score:.3f})"
            
            if "source_id" in meta:
                entry += f"\nSource: {meta['source_id']}"
            
            if "time_range" in meta:
                entry += f"\nTime range: {meta['time_range']}"
            
            if "spatial_bounds" in meta:
                bounds = meta["spatial_bounds"]
                entry += f"\nSpatial bounds: {bounds}"
            
            if "variables" in meta:
                entry += f"\nVariables: {', '.join(meta['variables'])}"
            
            # Add statistics if available
            if "statistics" in meta:
                stats = meta["statistics"]
                entry += "\nStatistics:"
                for var, var_stats in stats.items():
                    if isinstance(var_stats, dict):
                        entry += f"\n  {var}: mean={var_stats.get('mean', 'N/A'):.2f}, "
                        entry += f"std={var_stats.get('std', 'N/A'):.2f}, "
                        entry += f"min={var_stats.get('min', 'N/A'):.2f}, "
                        entry += f"max={var_stats.get('max', 'N/A'):.2f}"
            
            lines.append(entry)
        
        return "\n".join(lines)
    
    def _summarize_metadata(self, results: List[SearchResult]) -> Dict[str, Any]:
        """Extract summary statistics from retrieved metadata."""
        if not results:
            return {}
        
        # Collect sources
        sources = set()
        time_ranges = []
        variables = set()
        
        for result in results:
            meta = result.metadata
            if "source_id" in meta:
                sources.add(meta["source_id"])
            if "time_range" in meta:
                time_ranges.append(meta["time_range"])
            if "variables" in meta:
                if isinstance(meta["variables"], list):
                    variables.update(meta["variables"])
                else:
                    variables.add(meta["variables"])
        
        return {
            "num_chunks": len(results),
            "sources": list(sources),
            "time_ranges": time_ranges,
            "variables": list(variables),
            "avg_similarity": float(np.mean([r.score for r in results])),
        }
    
    def ask(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Ask a question and get LLM-generated answer based on retrieved context.
        
        Args:
            query: User question
            filters: Optional metadata filters
            top_k: Override default top_k
            temperature: Override default temperature
            
        Returns:
            Generated answer string
        """
        # Retrieve context
        context = self.retrieve(query, filters=filters, top_k=top_k)
        
        if not context.results:
            return "I couldn't find any relevant climate data to answer your question. Please try rephrasing or check if data for this topic is available."
        
        # Generate answer using LLM
        temp = temperature if temperature is not None else self.temperature
        
        try:
            answer = self.llm_client.generate_rag_answer(
                query=query,
                context_hits=[{
                    "metadata": r.metadata,
                    "score": r.score,
                } for r in context.results],
                temperature=temp,
            )
            
            logger.info(f"Generated answer ({len(answer)} chars)")
            return answer
            
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            # Fallback to simple context summary
            return self._fallback_answer(context)
    
    def _fallback_answer(self, context: RAGContext) -> str:
        """Generate simple fallback answer when LLM is unavailable."""
        summary = context.metadata_summary
        
        answer = f"Based on {summary['num_chunks']} data chunk(s):\n\n"
        answer += f"Sources: {', '.join(summary['sources'])}\n"
        answer += f"Variables: {', '.join(summary['variables'])}\n"
        answer += f"Average relevance: {summary['avg_similarity']:.2%}\n\n"
        answer += "Context:\n" + context.formatted_context[:500]
        
        return answer


def build_index_from_embeddings(
    embeddings_path: str,
    dim: int,
    metric: str = "cosine",
) -> VectorIndex:
    """
    Build VectorIndex from saved embeddings file.
    
    Args:
        embeddings_path: Path to JSONL file with embeddings
        dim: Embedding dimension
        metric: Similarity metric
        
    Returns:
        VectorIndex with loaded embeddings
    """
    import json
    from pathlib import Path
    
    index = VectorIndex(dim=dim, metric=metric)
    
    with open(embeddings_path, "r") as f:
        for line in f:
            record = json.loads(line)
            embedding = np.array(record["vector"], dtype=np.float32)
            metadata = record.get("metadata", {})
            index.add(embedding, metadata)
    
    logger.info(f"Built index with {len(index)} embeddings from {embeddings_path}")
    return index


# Example usage
if __name__ == "__main__":
    # This would be used like:
    # from src.embeddings.vector_index import VectorIndex
    # from src.embeddings.text_embeddings import get_text_embedder
    # from src.llm.ollama_client import OllamaClient
    
    print("RAG Pipeline module - see docstring for usage example")
