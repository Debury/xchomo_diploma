import os
import requests
from typing import List, Dict, Any

class OllamaClient:
    def __init__(self):
        self.base_url = os.getenv("OLLAMA_URL", "http://ollama:11434")
        # Recommended models (in order of quality for RAG with climate data):
        # - granite4:3b (NEWEST - IBM 2025, very fast, good for factual data, low RAM ~2-4GB) ⭐ SELECTED
        # - gemma3:12b (NEWEST - Google 2025, BEST for structured/scientific data, excellent accuracy, low hallucinations, ~8-12GB RAM)
        # - gemma3:4b (NEWEST - Google 2025, good balance of quality/speed, ~4-6GB RAM)
        # - granite4:8b (NEWEST - IBM 2025, excellent for factual data, Mamba+Transformer architecture, ~6-8GB RAM)
        # - gemma2:9b (older but still excellent for structured data)
        # - qwen2.5:7b (very good for scientific data, multilingual)
        # - llama3.1:8b (good balance, but less accurate than Gemma for structured data)
        # - llama3.2:3b (fastest but least accurate, prone to hallucinations)
        self.model = os.getenv("OLLAMA_MODEL", "granite4:3b")

    def generate_rag_answer(
        self,
        query: str,
        context_hits: List[Dict[str, Any]],
        temperature: float = 0.3  # Lower temperature for more factual, less creative answers
    ) -> str:
        """
        Generate an answer using RAG context with enhanced prompt to reduce hallucinations.
        
        Args:
            query: User question
            context_hits: List of context dictionaries with 'metadata' and 'score'
            temperature: LLM temperature for generation (default 0.3 for factual accuracy)
            
        Returns:
            Generated answer string
        """
        # Format context with better structure and variable type information
        context_lines = []
        variable_types = {}  # Track variable types to help LLM understand
        
        for idx, hit in enumerate(context_hits, 1):
            meta = hit.get('metadata', {}) if isinstance(hit, dict) else getattr(hit, 'metadata', {})
            score = hit.get('score', 0.0) if isinstance(hit, dict) else getattr(hit, 'score', 0.0)
            
            variable = meta.get('variable', 'unknown')
            
            # Classify variable types to help LLM understand
            var_lower = variable.lower()
            var_type = "unknown"
            if any(x in var_lower for x in ['tmax', 'tmin', 'temp', 'temperature']):
                var_type = "temperature (°F or °C)"
            elif any(x in var_lower for x in ['prcp', 'precip', 'rain']):
                var_type = "precipitation (inches or mm)"
            elif any(x in var_lower for x in ['snow']):
                var_type = "snowfall (inches or mm)"
            elif any(x in var_lower for x in ['dtd', 'dx', 'dt']):  # DT32, DX32, etc.
                var_type = "count of days (not a temperature value!)"
            elif any(x in var_lower for x in ['dd', 'hdd', 'cdd']):  # HTDD, CLDD
                var_type = "degree days (heating/cooling index)"
            elif any(x in var_lower for x in ['wind', 'wdf', 'wsf']):
                var_type = "wind (speed in mph/m/s or direction in degrees)"
            
            variable_types[variable] = var_type
            
            # Get text content or generate from metadata
            text_content = meta.get('text_content', '')
            if not text_content:
                # Build structured metadata
                parts = []
                if 'source_id' in meta:
                    parts.append(f"Source: {meta['source_id']}")
                if variable != 'unknown':
                    parts.append(f"Variable: {variable} ({var_type})")
                if 'time_start' in meta:
                    parts.append(f"Period: {meta['time_start']}")
                if 'stat_mean' in meta:
                    parts.append(f"Mean: {meta['stat_mean']:.2f}")
                if 'stat_min' in meta and 'stat_max' in meta:
                    parts.append(f"Range: {meta['stat_min']:.2f} to {meta['stat_max']:.2f}")
                text_content = " | ".join(parts) if parts else str(meta)
            
            context_lines.append(f"[Context {idx}] (Relevance: {score:.1%})\n{text_content}")
        
        context_str = "\n\n".join(context_lines)
        
        # Enhanced system prompt with clear instructions
        system_prompt = """You are an expert climate data analyst. Your task is to answer questions about climate data using ONLY the provided context.

CRITICAL RULES:
1. ONLY use information from the provided context. Do NOT make up or infer values not explicitly stated.
2. Distinguish between different variable types:
   - TEMPERATURE variables (TMAX, TMIN, EMXT, EMNT): These are actual temperature values in °F or °C
   - COUNT variables (DT32, DX32, DX70, etc.): These are COUNTS of days, NOT temperature values
   - DEGREE DAYS (HTDD, CLDD): These are heating/cooling indices, NOT temperatures
   - PRECIPITATION (PRCP, EMXP): These are precipitation amounts in inches or mm
3. When asked about "temperature range", use TMAX and TMIN values, NOT count variables like DT32
4. Always cite the source and time period when providing values
5. If the context doesn't contain the answer, say "The provided context does not contain this information"
6. Be precise with units and values - do not round unless necessary
7. If multiple sources conflict, mention this and cite all relevant sources

Answer format:
- Start with a direct answer
- Cite specific values with units
- Mention source and time period
- Explain any important distinctions (e.g., count vs. temperature)"""
        
        prompt = f"""{system_prompt}

=== CONTEXT DATA ===
{context_str}

=== USER QUESTION ===
{query}

=== YOUR ANSWER ===
Provide a precise, factual answer based ONLY on the context above. If the context doesn't contain the answer, state that clearly."""
        
        try:
            resp = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "top_p": 0.9,  # Nucleus sampling for better quality
                        "top_k": 40,  # Limit vocabulary for more focused answers
                        "num_predict": 500,  # Limit response length
                        "repeat_penalty": 1.1,  # Reduce repetition
                        "stop": ["\n\n\n", "=== CONTEXT", "=== USER"]  # Stop sequences
                    },
                    "system": system_prompt  # Use system prompt for better instruction following
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