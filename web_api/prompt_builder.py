"""
Prompt builder for RAG system.
Builds concise, effective prompts for LLM answer generation.
"""

from typing import List, Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def detect_question_type(question: str) -> str:
    """Detect question type for prompt adaptation."""
    q = question.lower()

    if any(p in q for p in ["what variables", "which variables", "list variables",
                             "available variables", "what data", "what fields"]):
        return "variable_list"

    if any(p in q for p in ["compare", "difference", "versus", "vs", "contrast"]):
        return "comparison"

    if any(p in q for p in ["average", "mean", "median", "minimum", "maximum",
                             "statistics", "summary"]):
        return "statistical"

    if any(p in q for p in ["trend", "over time", "historical", "change",
                             "increase", "decrease", "when"]):
        return "temporal"

    return "general"


def build_rag_prompt(
    question: str,
    context_chunks: List[Dict[str, Any]],
    all_variables: Optional[List[str]] = None,
    sources: Optional[List[str]] = None,
    question_type: str = "general",
    selected_variables: Optional[List[str]] = None,
) -> Tuple[str, int]:
    """
    Build a concise RAG prompt from question + retrieved context.

    Returns (prompt_text, max_tokens).
    """
    # Format context chunks into compact summaries
    context_lines = []
    for i, chunk in enumerate(context_chunks[:15], 1):
        meta = chunk.get("metadata", {})
        context_lines.append(_format_chunk(i, meta, chunk.get("score", 0.0)))

    context_text = "\n".join(context_lines) if context_lines else "(no data retrieved)"

    # Variable list questions — special handling
    if question_type == "variable_list" and all_variables:
        var_list = ", ".join(all_variables)
        prompt = f"""You are a climate data assistant. The dataset contains {len(all_variables)} variables:

{var_list}

QUESTION: {question}

List ALL {len(all_variables)} variables above. Do not omit any.

ANSWER:"""
        return prompt, min(600, len(all_variables) * 7)

    # Build optional metadata sections (only if relevant data exists)
    extra_sections = ""
    if all_variables:
        extra_sections += f"\nAvailable variables ({len(all_variables)}): {', '.join(all_variables[:50])}"
        if len(all_variables) > 50:
            extra_sections += f" ... and {len(all_variables) - 50} more"
    if sources:
        extra_sections += f"\nData sources: {', '.join(sources[:20])}"

    # System instruction — always grounded in retrieved context
    base_rules = (
        "You are a climate data assistant. Answer ONLY using the retrieved context below. "
        "Cite chunk numbers like [1], [2]. If the context does not contain enough information "
        "to answer, explain what related data IS available and how it connects to the question. "
        "Never invent data or statistics not present in the context. "
        "Always use the technical terms from the question (e.g. temperature, precipitation, "
        "drought, sea level, CO2, aerosol) in your answer. "
        "Name every dataset from the context that is relevant (e.g. ERA5, IMERG, GRACE, E-OBS, MERRA-2). "
        "End with a 'Relevant datasets:' line listing ALL dataset names from context that apply."
    )

    if question_type == "general":
        system = base_rules
        max_tokens = 1000
    elif question_type == "comparison":
        system = (
            base_rules + " Compare datasets precisely using values from context. "
            "If the comparison cannot be made, explain what data is available."
        )
        max_tokens = 800
    elif question_type == "statistical":
        system = (
            base_rules + " Use exact values (mean, min, max, percentiles) from context. "
            "Be precise with numbers and units."
        )
        max_tokens = 800
    elif question_type == "temporal":
        system = (
            base_rules + " Reference specific dates and time periods from context. "
            "Note any temporal gaps or limitations in the data."
        )
        max_tokens = 800
    else:
        system = base_rules
        max_tokens = 600

    prompt = f"""{system}
{extra_sections}

CONTEXT:
{context_text}

QUESTION: {question}

Provide a concise answer. Start with a 2-3 sentence summary, then add detail if needed.
Reuse the key terms from the question in your answer. Name all relevant datasets.

ANSWER:"""

    return prompt, max_tokens


def _format_chunk(index: int, meta: Dict[str, Any], score: float) -> str:
    """Format a single context chunk as a compact summary line."""
    parts = [f"[{index}]"]

    dataset = meta.get("dataset_name") or meta.get("source_id", "unknown")
    variable = meta.get("variable", "unknown")
    parts.append(f"{dataset} | {variable}")

    # Long name if available
    long_name = meta.get("long_name") or meta.get("standard_name")
    if long_name and long_name != variable:
        parts.append(f"({long_name})")

    # Hazard type for catalog entries
    hazard = meta.get("hazard_type")
    if hazard:
        parts.append(f"hazard={hazard}")

    # Spatial info
    spatial = meta.get("spatial_coverage")
    if spatial:
        parts.append(f"coverage={spatial}")

    # Location
    location = meta.get("location_name") or meta.get("region_country")
    if location:
        parts.append(f"region={location}")

    # Data type
    data_type = meta.get("data_type")
    if data_type:
        parts.append(f"type={data_type}")

    # Station info
    station = meta.get("station_name") or meta.get("station_id")
    if station:
        parts.append(f"station={station}")

    # Time range
    t_start = meta.get("time_start")
    t_end = meta.get("time_end")
    if t_start:
        t_str = str(t_start)[:10]
        if t_end and str(t_end)[:10] != t_str:
            t_str += f"..{str(t_end)[:10]}"
        parts.append(f"time={t_str}")

    # Temporal info for catalog entries
    temporal = meta.get("temporal_coverage_text")
    if temporal:
        parts.append(f"period={temporal}")

    # Statistics
    stats = []
    unit = meta.get("unit") or meta.get("units") or ""
    u = f" {unit}" if unit else ""
    if meta.get("stats_mean") is not None:
        stats.append(f"mean={meta['stats_mean']:.2f}{u}")
    if meta.get("stats_min") is not None and meta.get("stats_max") is not None:
        stats.append(f"range=[{meta['stats_min']:.2f}, {meta['stats_max']:.2f}]{u}")
    if stats:
        parts.append(" | ".join(stats))

    # Access/link for catalog
    access = meta.get("access_type")
    if access:
        parts.append(f"access={access}")

    parts.append(f"score={score:.3f}")
    return " | ".join(parts)
