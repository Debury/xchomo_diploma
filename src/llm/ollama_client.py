import os
import requests
from typing import List, Dict, Any

class OllamaClient:
    def __init__(self):
        self.base_url = os.getenv("OLLAMA_URL", "http://ollama:11434")
        self.model = os.getenv("OLLAMA_MODEL", "llama3.2:3b")

    def generate_rag_answer(
        self,
        query: str,
        context_hits: List[Dict[str, Any]],
        temperature: float = 0.7
    ) -> str:
        """
        Generate an answer using RAG context.
        
        Args:
            query: User question
            context_hits: List of context dictionaries with 'metadata' and 'score'
            temperature: LLM temperature for generation
            
        Returns:
            Generated answer string
        """
        # Format context with better structure
        context_lines = []
        for idx, hit in enumerate(context_hits, 1):
            meta = hit.get('metadata', {}) if isinstance(hit, dict) else getattr(hit, 'metadata', {})
            score = hit.get('score', 0.0) if isinstance(hit, dict) else getattr(hit, 'score', 0.0)
            
            # Get text content or generate from metadata
            text_content = meta.get('text_content', '')
            if not text_content:
                # Fallback: build from metadata
                parts = []
                if 'source_id' in meta:
                    parts.append(f"Dataset: {meta['source_id']}")
                if 'variable' in meta:
                    parts.append(f"Variable: {meta['variable']}")
                if 'time_start' in meta:
                    parts.append(f"Time: {meta['time_start']}")
                if 'stat_mean' in meta:
                    parts.append(f"Mean: {meta['stat_mean']:.2f}")
                text_content = " | ".join(parts) if parts else str(meta)
            
            context_lines.append(f"[Source {idx}] (Similarity: {score:.3f})\n{text_content}")
        
        context_str = "\n\n".join(context_lines)
        
        # Create a more structured prompt
        system_prompt = """You are a climate data assistant. Answer questions about climate data using the provided context. 
Be precise, cite specific values when available, and mention the data sources and time periods you're referencing."""
        
        prompt = f"""{system_prompt}

Context from climate data:
{context_str}

Question: {query}

Answer based on the context above:"""
        
        try:
            resp = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": temperature}
                },
                timeout=120  # Allow time for model loading
            )
            resp.raise_for_status()
            result = resp.json()
            return result.get("response", "No response generated.")
        except requests.exceptions.Timeout:
            return "LLM Error: Request timed out. The model may still be loading."
        except requests.exceptions.RequestException as e:
            return f"LLM Error: {str(e)}"
        except Exception as e:
            return f"LLM Error: {str(e)}"

    def check_health(self) -> bool:
        """Check if Ollama service is available."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return resp.status_code == 200
        except:
            return False