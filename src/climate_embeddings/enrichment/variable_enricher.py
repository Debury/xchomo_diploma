"""
Dynamic Variable Enrichment using LLM Agent
Enriches variable metadata with human-readable meanings during processing.
NO HARDCODED MAPPINGS - everything is inferred dynamically.
"""
import logging
from typing import Dict, Any, Optional, List
import os

logger = logging.getLogger(__name__)


def enrich_variable_metadata_batch(
    variables_metadata: List[Dict[str, Any]],
    llm_client: Optional[Any] = None,
    batch_size: int = 10
) -> List[Dict[str, Any]]:
    """
    Enrich variable metadata with LLM-inferred meanings.
    
    Args:
        variables_metadata: List of dicts with variable info (variable name, unit, stats, etc.)
        llm_client: Optional LLM client (if None, will try to create one)
        batch_size: Number of variables to process in one LLM call
    
    Returns:
        List of enriched metadata dicts with added 'long_name' and 'description' fields
    """
    if not variables_metadata:
        return []
    
    # If no LLM client provided, try to create one (optional - graceful degradation)
    if llm_client is None:
        try:
            from src.llm.openrouter_client import OpenRouterClient
            if os.getenv("OPENROUTER_API_KEY"):
                llm_client = OpenRouterClient()
                logger.info("Using LLM for variable enrichment")
            else:
                logger.warning("No LLM available for variable enrichment - using metadata only")
                return variables_metadata
        except Exception as e:
            logger.warning(f"Could not initialize LLM for enrichment: {e} - using metadata only")
            return variables_metadata
    
    enriched = []
    
    # Process in batches to avoid overwhelming LLM
    for i in range(0, len(variables_metadata), batch_size):
        batch = variables_metadata[i:i + batch_size]
        
        try:
            # Build prompt with variable information
            var_info = []
            for var_meta in batch:
                var_name = var_meta.get("variable", "unknown")
                unit = var_meta.get("unit", "")
                stats_mean = var_meta.get("stats_mean")
                stats_min = var_meta.get("stats_min")
                stats_max = var_meta.get("stats_max")
                
                info_parts = [f"Variable: {var_name}"]
                if unit:
                    info_parts.append(f"Unit: {unit}")
                if stats_mean is not None:
                    info_parts.append(f"Mean value: {stats_mean:.2f}")
                if stats_min is not None and stats_max is not None:
                    info_parts.append(f"Range: [{stats_min:.2f}, {stats_max:.2f}]")
                
                var_info.append(" | ".join(info_parts))
            
            prompt = f"""You are analyzing climate data variables. For each variable below, provide:
1. A human-readable long name (e.g., "average temperature" for TAVG)
2. A brief description (1-2 sentences)

Variables to analyze:
{chr(10).join(f"{idx+1}. {info}" for idx, info in enumerate(var_info))}

Respond in this exact format (one line per variable):
VARIABLE: variable_name | LONG_NAME: human readable name | DESCRIPTION: brief description

Example:
VARIABLE: TAVG | LONG_NAME: average temperature | DESCRIPTION: Daily average air temperature calculated from minimum and maximum temperatures."""
            
            # Call LLM
            response = llm_client.generate(
                prompt=prompt,
                temperature=0.1,
                max_tokens=200,
                timeout_s=15
            )
            
            # Parse response
            var_enrichments = {}
            for line in response.strip().split('\n'):
                if 'VARIABLE:' in line and 'LONG_NAME:' in line:
                    try:
                        parts = line.split('|')
                        var_name = None
                        long_name = None
                        description = None
                        
                        for part in parts:
                            part = part.strip()
                            if part.startswith('VARIABLE:'):
                                var_name = part.replace('VARIABLE:', '').strip()
                            elif part.startswith('LONG_NAME:'):
                                long_name = part.replace('LONG_NAME:', '').strip()
                            elif part.startswith('DESCRIPTION:'):
                                description = part.replace('DESCRIPTION:', '').strip()
                        
                        if var_name and long_name:
                            var_enrichments[var_name] = {
                                "long_name": long_name,
                                "description": description
                            }
                    except Exception as e:
                        logger.warning(f"Failed to parse LLM response line: {line}, error: {e}")
            
            # Apply enrichments to batch
            for var_meta in batch:
                var_name = var_meta.get("variable", "unknown")
                if var_name in var_enrichments:
                    enrichment = var_enrichments[var_name]
                    var_meta["long_name"] = enrichment.get("long_name")
                    if enrichment.get("description"):
                        var_meta["variable_description"] = enrichment.get("description")
                    logger.debug(f"Enriched {var_name} with long_name: {enrichment.get('long_name')}")
                
                enriched.append(var_meta)
        
        except Exception as e:
            logger.warning(f"LLM enrichment failed for batch: {e} - using original metadata")
            # If LLM fails, just use original metadata
            enriched.extend(batch)
    
    return enriched

