"""
Prompt builder for RAG system.
Uses XML-structured document format for better Claude grounding.
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
    Build a RAG prompt with XML-structured documents for better grounding.

    Returns (prompt_text, max_tokens).
    """
    # Format context chunks as XML documents
    context_lines = []
    for i, chunk in enumerate(context_chunks[:15], 1):
        meta = chunk.get("metadata", {})
        context_lines.append(_format_chunk_xml(i, meta, chunk.get("score", 0.0)))

    context_text = "\n".join(context_lines) if context_lines else "<documents>\n(no data retrieved)\n</documents>"

    # Variable list questions — special handling
    if question_type == "variable_list" and all_variables:
        var_list = ", ".join(all_variables)
        prompt = f"""You are a climate data assistant. The dataset contains {len(all_variables)} variables:

{var_list}

QUESTION: {question}

List ALL {len(all_variables)} variables above. Do not omit any.

ANSWER:"""
        return prompt, min(600, len(all_variables) * 7)

    # Build optional metadata sections
    extra_sections = ""
    if all_variables:
        extra_sections += f"\nAvailable variables ({len(all_variables)}): {', '.join(all_variables[:50])}"
        if len(all_variables) > 50:
            extra_sections += f" ... and {len(all_variables) - 50} more"
    if sources:
        extra_sections += f"\nData sources: {', '.join(sources[:20])}"

    # Type-specific additions
    type_instruction = ""
    if question_type == "comparison":
        type_instruction = "Compare datasets precisely using values from the documents. "
    elif question_type == "statistical":
        type_instruction = "Use exact values (mean, min, max, percentiles) from the documents. Be precise with numbers and units. "
    elif question_type == "temporal":
        type_instruction = "Reference specific dates and time periods from the documents. Note any temporal gaps. "

    max_tokens = 1200

    # Extract key terms from the question to echo back
    _stopwords = {
        "what", "which", "how", "does", "did", "were", "was", "are", "the",
        "and", "with", "from", "that", "this", "have", "has", "been", "between",
        "during", "into", "over", "last", "show", "data", "using", "alongside",
        "analyzing", "characterized", "consistently", "surpassed", "exceeded",
        "exceeding", "breaching", "feature", "track", "significant",
    }
    key_terms = [
        w.strip(".,;:?!()")
        for w in question.split()
        if len(w.strip(".,;:?!()")) > 3 and w.strip(".,;:?!()").lower() not in _stopwords
    ]
    key_terms_str = ", ".join(dict.fromkeys(t.lower() for t in key_terms))

    prompt = f"""You are a climate data expert assistant for a European audience. Use the retrieved documents below to answer the question.
Cite sources using [doc N] notation. Connect the data to the question — explain what the data shows
and what it means in the context of the question, even if the data is indirect or partial.
Always directly address the question using its key terms. If the data partially answers the question,
explain what IS available and how it relates. Never refuse to answer — always provide analysis.
Use the technical terms from the question in your answer.
When using abbreviations, include the full name (e.g. 'carbon dioxide (CO2)').
IMPORTANT: You MUST use ALL of these exact terms from the question at least once in your answer: {key_terms_str}.
Do NOT replace them with synonyms — use the original words. For example, say "drought" not "dry conditions", say "precipitation" not "rainfall".

UNITS & CONVENTIONS (European audience):
- Temperatures: convert Kelvin to Celsius. Example: `272.82 K` → `-0.3 °C (272.82 K)`.
  Always show the °C value first; the original K can follow in parentheses for reference.
  Conversion: °C = K − 273.15.
- Precipitation: prefer mm, mm/day, mm/month — keep whatever the source uses, don't invent units.
- Keep the document's raw numeric precision (e.g. 272.82), don't round aggressively.
- Spelling: use `temperature`, `Europe`, `metre` (British/European spellings when present in the source are fine).
{type_instruction}{extra_sections}

<documents>
{context_text}
</documents>

<question>{question}</question>

Before answering, extract key facts from each relevant document as direct quotes.
Then synthesize your answer from those quotes.

Format your answer as Markdown, using these exact headings:

## Summary
2–3 sentence direct answer. No bullets here — prose.

## Evidence
- **[Dataset name]** ([doc N]) — one-sentence fact with converted units where applicable.
- Keep each bullet to one line. Start with the dataset/source in bold, then the fact.
- Cite every bullet with [doc N].

## Datasets
- **Dataset A** — one short phrase describing what it contributed.
- **Dataset B** — same pattern.

Do NOT include an extra "Key facts" preamble or the raw document extraction — only the three sections above.

ANSWER:"""

    return prompt, max_tokens


def _format_chunk_xml(index: int, meta: Dict[str, Any], score: float) -> str:
    """Format a single context chunk as XML document for better Claude grounding."""
    dataset = meta.get("dataset_name") or meta.get("source_id", "unknown")
    variable = meta.get("variable", "unknown")
    long_name = meta.get("long_name") or meta.get("standard_name") or ""
    hazard = meta.get("hazard_type") or ""
    spatial = meta.get("spatial_coverage") or ""
    location = meta.get("location_name") or meta.get("region_country") or ""
    data_type = meta.get("data_type") or ""
    station = meta.get("station_name") or meta.get("station_id") or ""

    # Time range
    time_info = ""
    t_start = meta.get("time_start")
    t_end = meta.get("time_end")
    if t_start:
        time_info = str(t_start)[:10]
        if t_end and str(t_end)[:10] != time_info:
            time_info += f" to {str(t_end)[:10]}"
    temporal = meta.get("temporal_coverage_text") or ""

    # Statistics
    stats_parts = []
    unit = meta.get("unit") or meta.get("units") or ""
    u = f" {unit}" if unit else ""
    if meta.get("stats_mean") is not None:
        stats_parts.append(f"mean={meta['stats_mean']:.2f}{u}")
    if meta.get("stats_min") is not None and meta.get("stats_max") is not None:
        stats_parts.append(f"range=[{meta['stats_min']:.2f}, {meta['stats_max']:.2f}]{u}")
    if meta.get("stats_std") is not None:
        stats_parts.append(f"std={meta['stats_std']:.2f}{u}")
    stats_str = ", ".join(stats_parts)

    access = meta.get("access_type") or ""
    impact = meta.get("impact_sector") or ""
    keywords = meta.get("keywords") or ""

    lines = [f'<document index="{index}" score="{score:.3f}">']
    lines.append(f"  <source>{dataset}</source>")
    lines.append(f"  <variable>{variable}</variable>")
    if long_name:
        lines.append(f"  <description>{long_name}</description>")
    if hazard:
        lines.append(f"  <hazard>{hazard}</hazard>")
    if data_type:
        lines.append(f"  <type>{data_type}</type>")
    if spatial or location:
        loc_str = f"{spatial}, {location}" if spatial and location else (spatial or location)
        lines.append(f"  <coverage>{loc_str}</coverage>")
    if station:
        lines.append(f"  <station>{station}</station>")
    if time_info or temporal:
        t_str = time_info or temporal
        lines.append(f"  <period>{t_str}</period>")
    if stats_str:
        lines.append(f"  <statistics>{stats_str}</statistics>")
    if impact:
        lines.append(f"  <sectors>{impact}</sectors>")
    if keywords:
        lines.append(f"  <keywords>{keywords}</keywords>")
    if access:
        lines.append(f"  <access>{access}</access>")
    lines.append("</document>")

    return "\n".join(lines)
