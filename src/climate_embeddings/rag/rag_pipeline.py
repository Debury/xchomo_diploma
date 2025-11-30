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
        # 1. Embed
        if hasattr(self.text_embedder, 'embed_queries'):
            query_vec = self.text_embedder.embed_queries([query])[0]
        else:
            query_vec = self.text_embedder(query)

        # 2. Retrieve
        results = self.index.search(query_vec, k=self.top_k)
        
        if not results:
            return "No relevant data found."

        # 3. Context
        context_lines = []
        for res in results:
            # Handle both dict (from Qdrant) and object (from local Index) styles if needed
            meta = res['metadata'] if isinstance(res, dict) else res.metadata
            stats = res['stats'] if isinstance(res, dict) else getattr(res, 'stats', [])
            
            desc = f"Source: {meta.get('source_id', 'unknown')} | Var: {meta.get('variable', '?')}"
            if len(stats) > 0:
                desc += f" | Avg: {stats[0]:.2f} | Max: {stats[3]:.2f}"
            context_lines.append(desc)

        context_str = "\n".join(context_lines)

        # 4. Generate
        if self.llm_client:
            try:
                # Basic synchronous generation call
                prompt = f"Context:\n{context_str}\n\nQuestion: {query}\nAnswer:"
                # Adapt to your specific LLM client signature
                if hasattr(self.llm_client, 'generate_rag_answer'):
                    # This path is usually async in your code, handling sync here for CLI
                    return f"[Async Client] Context found: {len(results)} items."
                return f"LLM Answer placeholder for: {query}"
            except Exception as e:
                return f"LLM Error: {e}"
        
        return f"Retrieved Context:\n{context_str}"