"""
Optimized RAG Endpoint - Best Practices
Fast, reliable RAG with proper error handling and timeouts
"""
import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from fastapi import HTTPException
import asyncio

logger = logging.getLogger(__name__)

# ====================================================================================
# MODELS
# ====================================================================================

class RAGRequest(BaseModel):
    question: str
    top_k: int = 5
    use_llm: bool = True
    temperature: float = 0.7
    source_id: Optional[str] = None
    variable: Optional[str] = None
    timeout: int = 30  # LLM timeout in seconds

class RAGResponse(BaseModel):
    question: str
    answer: str
    chunks: List[Dict[str, Any]]
    references: List[str]
    llm_used: bool
    search_time_ms: float
    llm_time_ms: Optional[float] = None

# ====================================================================================
# RAG PIPELINE
# ====================================================================================

async def rag_query(request: RAGRequest) -> RAGResponse:
    """
    Optimized RAG pipeline with proper timeout handling.
    """
    import time
    from src.climate_embeddings.embeddings.text_models import TextEmbedder
    from src.embeddings.database import VectorDatabase
    from src.llm.ollama_client import OllamaClient
    from src.utils.config_loader import ConfigLoader
    
    try:
        # Load config
        config_loader = ConfigLoader("config/pipeline_config.yaml")
        config = config_loader.load()
        
        # Initialize components
        db = VectorDatabase(config=config)
        embedder = TextEmbedder()
        
        # 1. VECTOR SEARCH (fast - should take <500ms)
        search_start = time.time()
        
        # Embed query
        query_vec = embedder.embed_query(request.question)
        
        # Build filter
        filter_dict = {}
        if request.source_id:
            filter_dict["source_id"] = request.source_id
        if request.variable:
            filter_dict["variable"] = request.variable
        
        # Search Qdrant
        results = db.search(
            query_vector=query_vec.tolist(),
            limit=request.top_k,
            filter_dict=filter_dict if filter_dict else None
        )
        
        search_time = (time.time() - search_start) * 1000  # Convert to ms
        
        # 2. FORMAT CONTEXT
        chunks = []
        references = set()
        context_text = ""
        
        for i, hit in enumerate(results, 1):
            # Extract payload
            if hasattr(hit, 'payload'):
                meta = hit.payload
                score = hit.score
            else:
                meta = hit.get('payload', {})
                score = hit.get('score', 0.0)
            
            # Build context entry
            source_id = meta.get('source_id', 'unknown')
            variable = meta.get('variable', 'unknown')
            
            # Generate human-readable text from metadata
            from src.climate_embeddings.schema import generate_human_readable_text
            text = generate_human_readable_text(meta)
            
            chunks.append({
                "rank": i,
                "score": round(score, 3),
                "source_id": source_id,
                "variable": variable,
                "text": text,
                "metadata": meta
            })
            
            references.add(f"{source_id}:{variable}")
            context_text += f"\n[{i}] {text}\n"
        
        # 3. LLM GENERATION (with timeout)
        llm_used = False
        llm_time = None
        answer = ""
        
        if request.use_llm and chunks:
            try:
                llm_start = time.time()
                
                # Try Ollama with timeout
                client = OllamaClient()
                
                # Async timeout wrapper
                answer = await asyncio.wait_for(
                    asyncio.to_thread(
                        client.generate_rag_answer,
                        query=request.question,
                        context_hits=chunks,
                        temperature=request.temperature
                    ),
                    timeout=request.timeout
                )
                
                llm_time = (time.time() - llm_start) * 1000
                llm_used = True
                
            except asyncio.TimeoutError:
                logger.warning(f"LLM timeout after {request.timeout}s")
                answer = f"Found {len(chunks)} relevant results. (LLM timed out - consider reducing context or using faster model)"
            except Exception as e:
                logger.error(f"LLM error: {e}")
                answer = f"Found {len(chunks)} relevant results. (LLM error: {str(e)[:100]})"
        
        # Fallback answer if no LLM
        if not answer:
            if not chunks:
                answer = "No relevant climate data found for your query."
            else:
                answer = f"Found {len(chunks)} relevant climate data chunks:\n{context_text[:500]}"
        
        return RAGResponse(
            question=request.question,
            answer=answer,
            chunks=chunks,
            references=sorted(list(references)),
            llm_used=llm_used,
            search_time_ms=round(search_time, 2),
            llm_time_ms=round(llm_time, 2) if llm_time else None
        )
        
    except Exception as e:
        logger.error(f"RAG pipeline error: {e}")
        raise HTTPException(status_code=500, detail=f"RAG error: {str(e)}")


# ====================================================================================
# SIMPLE SEARCH (no LLM)
# ====================================================================================

async def simple_search(query: str, top_k: int = 5, filters: Optional[Dict] = None) -> List[Dict]:
    """
    Fast vector search without LLM - for debugging and quick queries.
    """
    from src.climate_embeddings.embeddings.text_models import TextEmbedder
    from src.embeddings.database import VectorDatabase
    from src.utils.config_loader import ConfigLoader
    
    try:
        config_loader = ConfigLoader("config/pipeline_config.yaml")
        config = config_loader.load()
        
        db = VectorDatabase(config=config)
        embedder = TextEmbedder()
        
        # Search
        query_vec = embedder.embed_query(query)
        results = db.search(
            query_vector=query_vec.tolist(),
            limit=top_k,
            filter_dict=filters
        )
        
        # Format
        chunks = []
        for hit in results:
            if hasattr(hit, 'payload'):
                meta = hit.payload
                score = hit.score
            else:
                meta = hit.get('payload', {})
                score = hit.get('score', 0.0)
            
            chunks.append({
                "score": round(score, 3),
                "source_id": meta.get('source_id'),
                "variable": meta.get('variable'),
                "metadata": meta
            })
        
        return chunks
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")
