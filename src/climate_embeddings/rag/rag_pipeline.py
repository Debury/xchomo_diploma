"""
RAG Pipeline orchestration.
"""
from typing import Callable, List, Dict, Any, Optional
import numpy as np
import logging

# Import from sibling modules
from ..index.vector_index import VectorIndex

logger = logging.getLogger(__name__)

class RAGPipeline:
    def __init__(
        self,
        index: VectorIndex,
        text_embedder: Any, # Can be TextEmbedder class or callable
        llm_client: Any = None,
        top_k: int = 5,
        temperature: float = 0.7,
    ):
        self.index = index
        self.text_embedder = text_embedder
        self.llm_client = llm_client
        self.top_k = top_k
        self.temperature = temperature

    def ask(self, query: str) -> str:
        """
        Process a query through the RAG pipeline.
        
        Args:
            query: User question about climate data
            
        Returns:
            Answer string with context from retrieved data
        """
        # 1. Embed query
        if hasattr(self.text_embedder, 'embed_queries'):
            query_vec = self.text_embedder.embed_queries([query])[0]
        else:
            query_vec = self.text_embedder(query)

        # 2. Retrieve similar chunks
        results = self.index.search(query_vec, k=self.top_k)
        
        if not results:
            return "No relevant data found."

        # 3. Build structured context
        context_chunks = []
        for idx, res in enumerate(results, 1):
            # Handle both dict (from Qdrant) and object (from local Index) styles
            if isinstance(res, dict):
                meta = res.get('metadata', {})
                score = res.get('score', 0.0)
                stats = res.get('stats', [])
            else:
                meta = getattr(res, 'metadata', {})
                score = getattr(res, 'score', 0.0)
                stats = getattr(res, 'stats', [])
            
            # Extract key information
            source_id = meta.get('source_id', 'unknown')
            variable = meta.get('variable', 'unknown')
            text_content = meta.get('text_content', '')
            
            # Build context entry
            context_entry = {
                'rank': idx,
                'similarity': score,
                'source_id': source_id,
                'variable': variable,
                'text': text_content if text_content else self._format_metadata_summary(meta, stats),
                'metadata': meta
            }
            context_chunks.append(context_entry)

        # 4. Format context for LLM
        context_str = self._format_context_for_llm(context_chunks)

        # 5. Generate answer
        if self.llm_client:
            try:
                if hasattr(self.llm_client, 'generate_rag_answer'):
                    # Use the LLM client's RAG method
                    return self.llm_client.generate_rag_answer(
                        query=query,
                        context_hits=context_chunks,
                        temperature=self.temperature
                    )
                else:
                    # Fallback: simple prompt
                    prompt = f"Context:\n{context_str}\n\nQuestion: {query}\nAnswer:"
                    return f"LLM Answer placeholder for: {query}"
            except Exception as e:
                logger.error(f"LLM generation error: {e}")
                return f"LLM Error: {e}"
        
        # Return formatted context if no LLM
        return f"Retrieved {len(context_chunks)} relevant data chunks:\n\n{context_str}"
    
    def _format_metadata_summary(self, meta: Dict[str, Any], stats: List[float]) -> str:
        """Format a summary from metadata and statistics."""
        parts = []
        
        variable = meta.get('variable', 'unknown')
        parts.append(f"Variable: {variable}")
        
        if 'time_start' in meta:
            parts.append(f"Time: {meta['time_start']}")
        
        if 'lat_min' in meta and 'lat_max' in meta:
            parts.append(f"Latitude: {meta['lat_min']:.2f}° to {meta['lat_max']:.2f}°")
        
        if stats and len(stats) >= 4:
            parts.append(f"Mean: {stats[0]:.2f}, Max: {stats[3]:.2f}")
        
        return " | ".join(parts)
    
    def _format_context_for_llm(self, context_chunks: List[Dict[str, Any]]) -> str:
        """Format retrieved context chunks for LLM consumption."""
        lines = []
        
        for chunk in context_chunks:
            lines.append(f"[Chunk {chunk['rank']}] Similarity: {chunk['similarity']:.3f}")
            lines.append(f"Source: {chunk['source_id']} | Variable: {chunk['variable']}")
            lines.append(f"Data: {chunk['text']}")
            lines.append("")  # Empty line between chunks
        
        return "\n".join(lines)