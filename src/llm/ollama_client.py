"""
Ollama Client for LLM-powered RAG answers.

Provides async and sync interfaces for calling local Ollama models.
Supports streaming, context injection, and climate-specific prompting.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Generator
import httpx

logger = logging.getLogger(__name__)


class OllamaClient:
    """
    Client for interacting with Ollama LLM server.
    
    Supports:
    - Local and remote Ollama instances
    - Multiple models (llama3, mistral, gemma, etc.)
    - RAG context injection
    - Streaming responses
    - Climate data-specific system prompts
    """
    
    DEFAULT_MODEL = "llama3.2:3b"  # Smaller model for faster responses
    FALLBACK_MODELS = ["mistral", "gemma2:2b", "phi3"]
    
    CLIMATE_SYSTEM_PROMPT = """You are a climate data analyst assistant. Your role is to:
1. Answer questions about climate data using ONLY the provided context
2. Be precise with numbers, dates, and units
3. Acknowledge when data is insufficient to answer fully
4. Explain climate metrics clearly (temperature anomalies, precipitation, etc.)
5. Reference specific data points from the context when available

Important guidelines:
- If the context doesn't contain relevant information, say so clearly
- Never invent or hallucinate data values
- Use proper scientific notation and units
- Cite the source_id when referencing specific datasets"""
    
    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 60.0,
    ):
        """
        Initialize Ollama client.
        
        Args:
            base_url: Ollama server URL. Defaults to OLLAMA_URL env var or http://localhost:11434
            model: Model name to use. Defaults to OLLAMA_MODEL env var or llama3.2:3b
            timeout: Request timeout in seconds
        """
        self.base_url = base_url or os.getenv("OLLAMA_URL", "http://ollama:11434")
        self.model = model or os.getenv("OLLAMA_MODEL", self.DEFAULT_MODEL)
        self.timeout = timeout
        self._available_models: Optional[List[str]] = None
        
        logger.info(f"OllamaClient initialized: {self.base_url}, model={self.model}")
    
    async def check_health(self) -> bool:
        """Check if Ollama server is available."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                return response.status_code == 200
        except Exception as e:
            logger.warning(f"Ollama health check failed: {e}")
            return False
    
    async def list_models(self) -> List[str]:
        """List available models on the Ollama server."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                if response.status_code == 200:
                    data = response.json()
                    self._available_models = [m["name"] for m in data.get("models", [])]
                    return self._available_models
        except Exception as e:
            logger.warning(f"Failed to list Ollama models: {e}")
        return []
    
    async def ensure_model(self) -> str:
        """Ensure a model is available, pulling if necessary."""
        models = await self.list_models()
        
        # Check if current model is available
        if any(self.model in m for m in models):
            return self.model
        
        # Try to pull the model
        logger.info(f"Model {self.model} not found, attempting to pull...")
        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                response = await client.post(
                    f"{self.base_url}/api/pull",
                    json={"name": self.model},
                )
                if response.status_code == 200:
                    logger.info(f"Successfully pulled model {self.model}")
                    return self.model
        except Exception as e:
            logger.warning(f"Failed to pull model {self.model}: {e}")
        
        # Try fallback models
        for fallback in self.FALLBACK_MODELS:
            if any(fallback in m for m in models):
                logger.info(f"Using fallback model: {fallback}")
                self.model = fallback
                return fallback
        
        raise RuntimeError(f"No LLM model available. Please pull a model with: ollama pull {self.DEFAULT_MODEL}")
    
    def _build_rag_prompt(
        self,
        question: str,
        context_chunks: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Build a RAG prompt with context injection.
        
        Args:
            question: User's question
            context_chunks: List of retrieved context chunks with text and metadata
            system_prompt: Optional custom system prompt
            
        Returns:
            Formatted prompt string
        """
        context_parts = []
        
        for i, chunk in enumerate(context_chunks, 1):
            metadata = chunk.get("metadata", {})
            source_id = metadata.get("source_id", "unknown")
            variable = metadata.get("variable", "data")
            text = chunk.get("text", "") or chunk.get("document", "")
            
            # Include key statistics if available
            stats_parts = []
            for stat_key in ["stat_mean", "stat_min", "stat_max", "stat_std"]:
                if stat_key in metadata:
                    stat_name = stat_key.replace("stat_", "")
                    stats_parts.append(f"{stat_name}={metadata[stat_key]:.4f}")
            
            unit = metadata.get("unit", "")
            temporal = metadata.get("temporal_extent", {})
            time_range = f"{temporal.get('start', 'N/A')} to {temporal.get('end', 'N/A')}" if temporal else "N/A"
            
            context_str = f"""[Source {i}: {source_id}]
Variable: {variable}
Time Range: {time_range}
Unit: {unit}
Statistics: {', '.join(stats_parts) if stats_parts else 'N/A'}
Content: {text}
"""
            context_parts.append(context_str)
        
        context_block = "\n---\n".join(context_parts)
        
        prompt = f"""Based on the following climate data context, answer the user's question.

=== CONTEXT ===
{context_block}
===============

Question: {question}

Provide a clear, accurate answer based only on the context above. If the context doesn't contain enough information, say so."""
        
        return prompt
    
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> str:
        """
        Generate a response from the LLM.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text response
        """
        system = system_prompt or self.CLIMATE_SYSTEM_PROMPT
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            }
        }
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                )
                
                if response.status_code != 200:
                    logger.error(f"Ollama generate failed: {response.status_code} - {response.text}")
                    raise RuntimeError(f"Ollama API error: {response.status_code}")
                
                data = response.json()
                return data.get("response", "").strip()
                
        except httpx.TimeoutException:
            logger.error("Ollama request timed out")
            raise RuntimeError("LLM request timed out. The model may be loading.")
        except Exception as e:
            logger.error(f"Ollama generate error: {e}")
            raise
    
    async def generate_rag_answer(
        self,
        question: str,
        context_chunks: List[Dict[str, Any]],
        temperature: float = 0.3,
    ) -> str:
        """
        Generate a RAG answer using retrieved context.
        
        Args:
            question: User's question
            context_chunks: Retrieved context chunks from vector search
            temperature: Sampling temperature
            
        Returns:
            LLM-generated answer
        """
        if not context_chunks:
            return "No relevant context found in the database to answer this question."
        
        prompt = self._build_rag_prompt(question, context_chunks)
        
        try:
            answer = await self.generate(
                prompt=prompt,
                temperature=temperature,
            )
            return answer
        except Exception as e:
            logger.error(f"RAG answer generation failed: {e}")
            # Fallback to simple summarization
            return self._fallback_summarize(context_chunks)
    
    def _fallback_summarize(self, chunks: List[Dict[str, Any]]) -> str:
        """Fallback summarization when LLM is unavailable."""
        if not chunks:
            return "No data available."
        
        sentences = []
        for chunk in chunks:
            metadata = chunk.get("metadata", {})
            variable = metadata.get("variable", "variable")
            source_id = metadata.get("source_id", "source")
            stat_mean = metadata.get("stat_mean")
            unit = metadata.get("unit", "")
            
            if isinstance(stat_mean, (int, float)):
                sentences.append(f"{variable} from {source_id} averages {stat_mean:.2f}{unit}.")
            else:
                sentences.append(f"Data found for {variable} from {source_id}.")
        
        return " ".join(sentences) + " (Note: LLM unavailable, showing basic summary)"
    
    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
    ) -> Generator[str, None, None]:
        """
        Generate a streaming response from the LLM.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            
        Yields:
            Generated text chunks
        """
        system = system_prompt or self.CLIMATE_SYSTEM_PROMPT
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system,
            "stream": True,
            "options": {
                "temperature": temperature,
            }
        }
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/api/generate",
                json=payload,
            ) as response:
                async for line in response.aiter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            if "response" in data:
                                yield data["response"]
                            if data.get("done"):
                                break
                        except json.JSONDecodeError:
                            continue


# Synchronous wrapper for non-async contexts
class OllamaClientSync:
    """Synchronous wrapper for OllamaClient."""
    
    def __init__(self, **kwargs):
        self._async_client = OllamaClient(**kwargs)
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate response synchronously."""
        import asyncio
        return asyncio.run(self._async_client.generate(prompt, **kwargs))
    
    def generate_rag_answer(self, question: str, context_chunks: List[Dict[str, Any]], **kwargs) -> str:
        """Generate RAG answer synchronously."""
        import asyncio
        return asyncio.run(self._async_client.generate_rag_answer(question, context_chunks, **kwargs))
