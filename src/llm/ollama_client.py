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
        # Format context with better structure and metadata information
        context_lines = []
        
        for idx, hit in enumerate(context_hits, 1):
            meta = hit.get('metadata', {}) if isinstance(hit, dict) else getattr(hit, 'metadata', {})
            score = hit.get('score', 0.0) if isinstance(hit, dict) else getattr(hit, 'score', 0.0)
            
            variable = meta.get('variable', 'unknown')
            
            # Dynamically extract unit and metadata information
            unit = meta.get('unit', meta.get('units', ''))
            unit_str = str(unit).strip() if unit else ''
            long_name = meta.get('long_name', '')
            standard_name = meta.get('standard_name', '')
            
            # Build variable description from available metadata (no hardcoded patterns)
            var_info_parts = []
            if variable != 'unknown':
                var_info_parts.append(f"Variable: {variable}")
            if long_name:
                var_info_parts.append(f"Description: {long_name}")
            if standard_name:
                var_info_parts.append(f"Standard name: {standard_name}")
            if unit_str:
                var_info_parts.append(f"Unit: {unit_str}")
            
            # Generate human-readable text dynamically from structured metadata
            # (text_content is no longer stored in DB - we generate it from metadata)
            from src.climate_embeddings.schema import generate_human_readable_text
            text_content = generate_human_readable_text(meta)
            
            # If text_content was somehow stored (backward compatibility), use it
            if not text_content and meta.get('text_content'):
                text_content = meta.get('text_content')
            
            if not text_content:
                # Build structured metadata dynamically from available fields
                parts = []
                if 'source_id' in meta:
                    parts.append(f"Source: {meta['source_id']}")
                
                # Add variable information from metadata
                if var_info_parts:
                    parts.append(" | ".join(var_info_parts))
                
                # CRITICAL: Identify variable type from description/name to help LLM understand
                var_type_hint = ""
                long_name_lower = long_name.lower() if long_name else ""
                var_lower = variable.lower() if variable else ""
                
                # Detect variable type from description/name
                if "maximum" in long_name_lower or "max" in long_name_lower or "highest" in long_name_lower:
                    var_type_hint = " (MAXIMUM TEMPERATURE variable)"
                elif "minimum" in long_name_lower or "min" in long_name_lower or "lowest" in long_name_lower:
                    var_type_hint = " (MINIMUM TEMPERATURE variable)"
                elif "count" in long_name_lower or "days" in long_name_lower or "number" in long_name_lower:
                    var_type_hint = " (COUNT variable - NOT a temperature measurement)"
                elif "temp" in long_name_lower or "temperature" in long_name_lower:
                    var_type_hint = " (TEMPERATURE variable)"
                
                # Also check variable name patterns
                if not var_type_hint:
                    if "max" in var_lower or "tmax" in var_lower:
                        var_type_hint = " (MAXIMUM TEMPERATURE variable)"
                    elif "min" in var_lower or "tmin" in var_lower:
                        var_type_hint = " (MINIMUM TEMPERATURE variable)"
                    elif "dt" in var_lower or "dx" in var_lower:
                        var_type_hint = " (COUNT variable - NOT a temperature measurement)"
                
                if var_type_hint:
                    parts.append(f"Variable type:{var_type_hint}")
                
                if 'time_start' in meta:
                    parts.append(f"Period: {meta['time_start']}")
                
                # CRITICAL: Spatial coordinates - clearly separate from temperature values
                # NEVER confuse longitude/latitude with temperature!
                spatial_info = []
                if 'lat_min' in meta and 'lat_max' in meta:
                    lat_min = meta['lat_min']
                    lat_max = meta['lat_max']
                    if lat_min == lat_max:
                        spatial_info.append(f"Location: {lat_min:.2f}°N")
                    else:
                        spatial_info.append(f"Latitude: {lat_min:.2f}° to {lat_max:.2f}°N")
                if 'lon_min' in meta and 'lon_max' in meta:
                    lon_min = meta['lon_min']
                    lon_max = meta['lon_max']
                    if lon_min == lon_max:
                        spatial_info.append(f"{lon_min:.2f}°E")
                    else:
                        spatial_info.append(f"Longitude: {lon_min:.2f}° to {lon_max:.2f}°E")
                if spatial_info:
                    parts.append(f"Geographic coordinates: {' '.join(spatial_info)} (NOT temperature values!)")
                
                if 'stat_mean' in meta:
                    mean_val = meta['stat_mean']
                    formatted_mean = f"Mean value: {mean_val:.2f}{' ' + unit_str if unit_str else ''}"
                    # Convert to Celsius if needed
                    if 'F' in unit_str.upper() or 'FAHRENHEIT' in unit_str.upper():
                        mean_c = (mean_val - 32) * 5/9
                        formatted_mean += f" ({mean_c:.2f}°C)"
                    elif 'K' in unit_str.upper() or 'KELVIN' in unit_str.upper():
                        mean_c = mean_val - 273.15
                        formatted_mean += f" ({mean_c:.2f}°C)"
                    parts.append(formatted_mean)
                
                # CRITICAL: Clarify that this is the VALUE RANGE of this variable, NOT the temperature range
                if 'stat_min' in meta and 'stat_max' in meta:
                    min_val = meta['stat_min']
                    max_val = meta['stat_max']
                    formatted_range = f"Value range (min to max for THIS variable): {min_val:.2f} to {max_val:.2f}{' ' + unit_str if unit_str else ''}"
                    # Convert to Celsius if needed
                    if 'F' in unit_str.upper() or 'FAHRENHEIT' in unit_str.upper():
                        min_c = (min_val - 32) * 5/9
                        max_c = (max_val - 32) * 5/9
                        formatted_range += f" ({min_c:.2f} to {max_c:.2f}°C)"
                    elif 'K' in unit_str.upper() or 'KELVIN' in unit_str.upper():
                        min_c = min_val - 273.15
                        max_c = max_val - 273.15
                        formatted_range += f" ({min_c:.2f} to {max_c:.2f}°C)"
                    parts.append(formatted_range)
                
                text_content = " | ".join(parts) if parts else str(meta)
            
            context_lines.append(f"[Context {idx}] (Relevance: {score:.1%})\n{text_content}")
        
        # Collect all time periods and temperature variables from context
        time_periods = set()
        temp_vars_found = []
        
        for hit in context_hits:
            meta = hit.get('metadata', {}) if isinstance(hit, dict) else getattr(hit, 'metadata', {})
            
            # Collect time periods
            time_start = meta.get('time_start', '')
            if time_start:
                time_periods.add(str(time_start))
            time_end = meta.get('time_end', '')
            if time_end and time_end != time_start:
                time_periods.add(str(time_end))
            
            # Collect temperature variables
            variable = meta.get('variable', '')
            long_name = meta.get('long_name', '').lower()
            var_lower = variable.lower()
            
            # Check if it's a temperature variable
            is_temp = False
            is_max = False
            is_min = False
            
            if 'temp' in long_name or 'temp' in var_lower:
                is_temp = True
                if 'max' in long_name or 'max' in var_lower or 'tmax' in var_lower:
                    is_max = True
                elif 'min' in long_name or 'min' in var_lower or 'tmin' in var_lower:
                    is_min = True
            
            if is_temp:
                if is_max:
                    temp_vars_found.append(f"MAXIMUM temperature: {variable}")
                elif is_min:
                    temp_vars_found.append(f"MINIMUM temperature: {variable}")
                else:
                    temp_vars_found.append(f"Temperature: {variable}")
        
        # Build context string with summaries at the BEGINNING (critical for LLM to see them first)
        summary_lines = []
        
        # Add summary of available time periods FIRST (CRITICAL - prevents hallucination)
        if time_periods:
            sorted_periods = sorted(time_periods)
            summary_lines.append(f"⚠️ CRITICAL: Available time periods in context (ONLY use these - DO NOT invent others):")
            summary_lines.append("\n".join(f"  - {tp}" for tp in sorted_periods))
            summary_lines.append("")  # Empty line for separation
        
        # Add summary of available temperature variables
        if temp_vars_found:
            summary_lines.append(f"Available temperature variables in context:")
            summary_lines.append("\n".join(f"  - {v}" for v in set(temp_vars_found)))
            if not any('MAXIMUM' in v for v in temp_vars_found):
                summary_lines.append("  ⚠️ WARNING: No MAXIMUM temperature variable found - cannot calculate full temperature range!")
            if not any('MINIMUM' in v for v in temp_vars_found):
                summary_lines.append("  ⚠️ WARNING: No MINIMUM temperature variable found - cannot calculate full temperature range!")
            summary_lines.append("")  # Empty line for separation
        
        # Combine: summaries first, then context chunks
        if summary_lines:
            context_str = "\n".join(summary_lines) + "\n\n" + "\n\n".join(context_lines)
        else:
            context_str = "\n\n".join(context_lines)
        
        # Dynamically detect units from metadata (no hardcoded patterns)
        detected_units = set()
        
        for hit in context_hits:
            meta = hit.get('metadata', {}) if isinstance(hit, dict) else getattr(hit, 'metadata', {})
            unit = meta.get('unit', meta.get('units', ''))
            if unit:
                detected_units.add(str(unit).strip())
        
        # Build unit conversion instructions dynamically based on detected units
        unit_conversion_notes = []
        for unit in detected_units:
            unit_upper = unit.upper()
            if 'F' in unit_upper or 'FAHRENHEIT' in unit_upper:
                unit_conversion_notes.append("- Fahrenheit (°F) to Celsius (°C): °C = (°F - 32) × 5/9")
            elif 'K' in unit_upper and 'KELVIN' in unit_upper or (len(unit) == 1 and unit_upper == 'K'):
                unit_conversion_notes.append("- Kelvin (K) to Celsius (°C): °C = K - 273.15")
            elif 'INCH' in unit_upper or 'IN' in unit_upper:
                unit_conversion_notes.append("- Inches to millimeters: mm = in × 25.4")
        
        # Enhanced system prompt with fully dynamic instructions (no hardcoded variable names)
        system_prompt = f"""You are an expert climate data analyst. Your task is to answer questions about climate data using ONLY the provided context.

CRITICAL RULES:
1. ONLY use information from the provided context. Do NOT make up or infer values not explicitly stated.

1a. TIME PERIODS (CRITICAL - prevents hallucination):
   - ONLY use time periods that are explicitly listed in the "[SUMMARY] Available time periods in context" section
   - DO NOT invent, infer, or assume time periods that are not explicitly stated in the context
   - If the context shows data for "2016-04", DO NOT mention "January" or any other month unless it is explicitly listed in the available time periods
   - If asked about a time range (e.g., "January to April"), check if ALL mentioned periods are in the available time periods list
   - If a time period is missing from the context, explicitly state: "I only have data for [list actual periods], not for [missing period]"
   - Example: If context shows only "2016-04", and question asks about "January to April", answer: "I only have data for April 2016, not for January, February, or March."

2. VARIABLE TYPE IDENTIFICATION (CRITICAL - identify from context metadata):
   To identify variable types, check these fields in order (NO hardcoded variable names):
   a) Description/long_name field: Look for keywords:
      - "days", "count", "number of", "occurrences", "frequency" → COUNT variable (NOT a temperature!)
      - "temperature", "temp", "degrees" (when referring to actual temp measurements) → TEMPERATURE variable
      - "precipitation", "rain", "snow", "rainfall" → PRECIPITATION variable
      - "wind", "speed", "velocity" → WIND variable
   
   b) Variable name patterns (if description is missing): Check if name suggests:
      - Count/occurrence indicators (any pattern suggesting counting) → COUNT variable
      - Maximum/highest temperature indicators → MAXIMUM TEMPERATURE variable
      - Minimum/lowest temperature indicators → MINIMUM TEMPERATURE variable
      - Precipitation indicators → PRECIPITATION variable
   
   c) Unit field: Check the unit:
      - Unit is "days", "count", "occurrences", or missing but values are small integers (0-31) → COUNT variable
      - Unit is "°C", "°F", "K", "Celsius", "Fahrenheit", "Kelvin" → TEMPERATURE variable (if description confirms)
      - Unit is "mm", "inches", "m", "meters" → PRECIPITATION variable
      - Unit is "m/s", "mph", "km/h" → WIND variable
   
   d) Value range and description: If values are small integers (0-31) AND description mentions "days" or "count" → COUNT variable
   
   IMPORTANT: A variable with temperature unit ("°C" or "°F") that has description mentioning "days", "count", or "number of" is STILL a COUNT variable, NOT a temperature measurement!

3. TEMPERATURE RANGE CALCULATION (when asked about "temperature range"):
   CRITICAL: Temperature range requires TWO DIFFERENT variables (identified dynamically from context metadata):
   - Variable 1: Maximum temperature variable (identify from description/name/unit suggesting maximum/highest temperature)
   - Variable 2: Minimum temperature variable (identify from description/name/unit suggesting minimum/lowest temperature)
   
   Calculation steps (apply dynamically to ANY dataset):
   1. Find the maximum temperature variable in context (check variable name, long_name, standard_name, unit)
   2. Find the minimum temperature variable in context (check variable name, long_name, standard_name, unit)
   3. Extract the MAXIMUM value from the maximum temperature variable (use stats_max or max value from context)
   4. Extract the MINIMUM value from the minimum temperature variable (use stats_min or min value from context)
   5. Calculate: Temperature range = (max value from max temp variable) - (min value from min temp variable)
   6. Report: "Temperature ranged from [min]°C to [max]°C, a range of [calculated range]°C"
   
   IMPORTANT: Always report the OVERALL temperature range (min to max across both variables), not individual variable ranges.
   
   ABSOLUTELY FORBIDDEN:
   - ❌ DO NOT use the range (min-max) of a SINGLE variable - that's the range of that variable, NOT the temperature range!
   - ❌ DO NOT use count variables (any variable identified as a count/occurrence) - these are counts, NOT temperatures
   - ❌ DO NOT use degree days or heating/cooling indices - these are indices, NOT temperatures
   - ❌ DO NOT calculate range from one variable's min and max values
   - ❌ DO NOT infer or estimate - only use explicit values from context
   - ❌ DO NOT confuse geographic coordinates (longitude/latitude) with temperature values! Coordinates are locations, NOT temperatures!
   
   CRITICAL: Geographic coordinates (longitude, latitude) are LOCATIONS, NOT temperature values!
   - Longitude values (e.g., -82.54°) are geographic coordinates, NOT temperatures
   - Latitude values (e.g., 35.43°) are geographic coordinates, NOT temperatures
   - NEVER use longitude or latitude values as temperature values!
   
   If context is insufficient:
   - If you see ONLY minimum temperature variable: Say "I can see minimum temperature data, but I need maximum temperature data to calculate the temperature range."
   - If you see ONLY maximum temperature variable: Say "I can see maximum temperature data, but I need minimum temperature data to calculate the temperature range."
   - If you see NEITHER: Say "I need both maximum and minimum temperature variables to calculate the temperature range."
   - NEVER make up values, infer missing data, use incorrect variable types, or confuse coordinates with temperatures!

4. UNIT CONVERSION (use European/SI units):
   - ALWAYS provide temperatures in Celsius (°C) in your answer
   - ALWAYS provide precipitation in millimeters (mm) or meters (m)
   - ALWAYS provide distances in meters (m) or kilometers (km)
   - Convert from other units as needed:
{chr(10).join(unit_conversion_notes) if unit_conversion_notes else "   - Check the unit field in the context for each variable and convert to SI/European units"}

5. Always cite the source and time period when providing values
6. If the context doesn't contain sufficient information, clearly state what is missing (e.g., "I need both maximum and minimum temperature variables to calculate the temperature range")
7. Be precise with units - always show the unit in your answer
8. If multiple sources conflict, mention this and cite all relevant sources

Answer format:
- Start with a direct answer using European/SI units (Celsius, mm, m, etc.)
- Show original values with their units if conversion was needed
- Cite specific values with units
- Mention source and time period
- Explain any important distinctions between variable types (especially count vs. temperature)"""
        
        prompt = f"""{system_prompt}

=== CONTEXT DATA ===
{context_str}

=== USER QUESTION ===
{query}

=== YOUR ANSWER ===
Provide a precise, factual answer based ONLY on the context above. If the context doesn't contain the answer, state that clearly."""
        
        # Try to generate with the configured model
        # Use longer timeout for first request (model may need to load)
        # Subsequent requests will be faster
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
                timeout=300  # Increased to 5 minutes for first load (model may be loading from disk)
            )
            resp.raise_for_status()
            result = resp.json()
            return result.get("response", "No response generated.")
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                # Model not found - try to list available models and suggest fix
                try:
                    models_resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
                    if models_resp.status_code == 200:
                        models = models_resp.json().get("models", [])
                        model_names = [m.get("name", "") for m in models]
                        available = ", ".join(model_names[:5]) if model_names else "none"
                        return f"""LLM Error: Model '{self.model}' not found in Ollama.

Available models: {available}

To install the model, run:
  docker compose exec ollama ollama pull {self.model}

Or use an available model by setting OLLAMA_MODEL environment variable."""
                    else:
                        return f"LLM Error: Model '{self.model}' not found. Please install it with: docker compose exec ollama ollama pull {self.model}"
                except:
                    return f"LLM Error: Model '{self.model}' not found. Please install it with: docker compose exec ollama ollama pull {self.model}"
            else:
                return f"LLM Error: HTTP {e.response.status_code} - {str(e)}"
        except requests.exceptions.Timeout:
            # Check if model is available and suggest solutions
            try:
                models_resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
                if models_resp.status_code == 200:
                    models = models_resp.json().get("models", [])
                    model_names = [m.get("name", "") for m in models]
                    if self.model in model_names:
                        return f"""LLM Error: Request timed out after 5 minutes.

The model '{self.model}' is installed but may be:
1. Still loading into memory (first request takes longer)
2. Too slow for the current hardware
3. Experiencing memory issues

Solutions:
- Wait a few minutes and try again (model may finish loading)
- Try a smaller model: llama3.2:3b or gemma2:2b
- Check Ollama logs: docker compose logs ollama
- Increase server resources (RAM/CPU)"""
                    else:
                        return f"LLM Error: Model '{self.model}' not found. Please install it with: docker compose exec ollama ollama pull {self.model}"
            except:
                pass
            return "LLM Error: Request timed out. The model may still be loading, the server is unavailable, or the model is too slow for the current hardware."
        except requests.exceptions.ConnectionError:
            return f"LLM Error: Cannot connect to Ollama at {self.base_url}. Is the Ollama service running?"
        except requests.exceptions.RequestException as e:
            return f"LLM Error: {str(e)}"
        except Exception as e:
            return f"LLM Error: {str(e)}"

    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 500,
        timeout_s: int = 120,
    ) -> str:
        """
        Simple text generation without RAG context formatting.
        Used for custom prompts with pre-formatted context.
        
        Args:
            prompt: Complete prompt text
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "temperature": temperature,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "num_ctx": 4096
            }
        }
        
        try:
            resp = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=timeout_s
            )
            resp.raise_for_status()
            result = resp.json()
            return result.get("response", "").strip()
        except requests.exceptions.Timeout:
            raise TimeoutError(f"Ollama generation timed out after {timeout_s}s")
        except Exception as e:
            raise Exception(f"Ollama generate error: {e}")
    
    def check_health(self) -> bool:
        """Check if Ollama service is available and model is installed."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if resp.status_code == 200:
                # Also check if the configured model is available
                models = resp.json().get("models", [])
                model_names = [m.get("name", "") for m in models]
                if self.model not in model_names:
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.warning(f"Configured model '{self.model}' not found. Available: {', '.join(model_names[:3])}")
                return True
            return False
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Ollama health check failed: {e}")
            return False
    
    def warm_up_model(self, timeout: int = 60) -> bool:
        """
        Warm up the model by sending a small test request.
        This loads the model into memory, making subsequent requests faster.
        
        Returns:
            True if warm-up successful, False otherwise
        """
        try:
            # Send a minimal prompt to load the model
            resp = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": "test",
                    "stream": False,
                    "options": {
                        "num_predict": 1  # Just generate 1 token to trigger model load
                    }
                },
                timeout=timeout
            )
            return resp.status_code == 200
        except:
            return False
    
    def list_available_models(self) -> list:
        """List all available models in Ollama."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if resp.status_code == 200:
                models = resp.json().get("models", [])
                return [m.get("name", "") for m in models]
            return []
        except:
            return []