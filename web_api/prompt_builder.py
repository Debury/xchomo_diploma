"""
Dynamic prompt builder for RAG system.
Adapts prompts based on question type, available data, and context.
"""

from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


def build_rag_prompt(
    question: str,
    context_chunks: List[Dict[str, Any]],
    all_variables: Optional[List[str]] = None,
    sources: Optional[List[str]] = None,
    question_type: str = "general"
) -> tuple[str, int]:
    """
    Build dynamic prompt based on question type and available data.
    
    Args:
        question: User's question
        context_chunks: Retrieved context chunks from vector search
        all_variables: All available variables in the dataset (for variable questions)
        sources: List of data sources
        question_type: Type of question (variable_list, comparison, statistical, general)
    
    Returns:
        Tuple of (prompt, max_tokens)
    """
    
    # Build context summary from chunks
    context_lines = []
    seen_vars = set()
    seen_sources = set()
    time_periods = set()  # Collect ALL unique time periods
    
    for i, chunk in enumerate(context_chunks[:15], 1):  # Use top 15 chunks
        meta = chunk.get('metadata', {})
        var = meta.get('variable', 'unknown')
        source = meta.get('source_id', 'unknown')
        score = chunk.get('score', 0.0)
        
        seen_vars.add(var)
        seen_sources.add(source)
        
        # Build compact summary
        stats_parts = []
        if meta.get('stats_mean') is not None:
            stats_parts.append(f"mean={meta['stats_mean']:.2f}")
        if meta.get('stats_min') is not None and meta.get('stats_max') is not None:
            stats_parts.append(f"range=[{meta['stats_min']:.2f}, {meta['stats_max']:.2f}]")
        
        # Extract time range (both start and end if available)
        time_start = meta.get('time_start', '')
        time_end = meta.get('time_end', '')
        
        if time_start:
            time_start_str = str(time_start)[:10]  # Just date part (YYYY-MM-DD)
            time_periods.add(time_start_str)
            
            if time_end and str(time_end)[:10] != time_start_str:
                time_end_str = str(time_end)[:10]
                time_periods.add(time_end_str)
                time_str = f"{time_start_str} to {time_end_str}"
            else:
                time_str = time_start_str
        else:
            time_str = ""
        
        summary = f"[{i}] {var} from {source}"
        if stats_parts:
            summary += f" ({', '.join(stats_parts)})"
        if time_str:
            summary += f" @ {time_str}"
        summary += f" [score: {score:.3f}]"
        
        context_lines.append(summary)
    
    context_text = "\n".join(context_lines)
    
    # Build time periods summary
    time_periods_info = ""
    if time_periods:
        sorted_periods = sorted(time_periods)
        if len(sorted_periods) > 1:
            time_periods_info = f"\n\nAVAILABLE TIME PERIODS IN DATA: {len(sorted_periods)} unique dates from {sorted_periods[0]} to {sorted_periods[-1]}\nAll dates: {', '.join(sorted_periods[:20])}{'...' if len(sorted_periods) > 20 else ''}"
        else:
            time_periods_info = f"\n\nAVAILABLE TIME PERIOD: {sorted_periods[0]}"
    
    # Build sources info
    sources_info = ""
    if sources:
        sources_info = f"\n\nDATA SOURCES ({len(sources)}): {', '.join(sources)}"
    
    # Build variables info - CRITICAL: Always show ALL variables for variable_list questions
    variables_info = ""
    if question_type == "variable_list" and all_variables:
        # For variable list questions, ALWAYS show ALL variables from database
        # Format as a clear numbered list for better LLM comprehension
        var_list = "\n".join([f"{i+1}. {var}" for i, var in enumerate(all_variables)])
        variables_info = f"""

═══════════════════════════════════════════════════════════════════════════════
COMPLETE LIST OF ALL {len(all_variables)} AVAILABLE VARIABLES IN THE DATASET:
═══════════════════════════════════════════════════════════════════════════════
{var_list}
═══════════════════════════════════════════════════════════════════════════════
IMPORTANT: The dataset contains EXACTLY {len(all_variables)} variables. You MUST list ALL of them.
═══════════════════════════════════════════════════════════════════════════════"""
    elif all_variables:
        variables_info = f"\n\nAVAILABLE VARIABLES ({len(all_variables)}): {', '.join(all_variables)}"
    elif seen_vars:
        variables_info = f"\n\nVARIABLES IN CONTEXT (limited sample): {', '.join(sorted(seen_vars))}"
    
    # Build prompt based on question type
    if question_type == "variable_list":
        if all_variables:
            # CRITICAL: For variable list questions, we MUST have all variables
            prompt = f"""You are a climate data assistant. Answer the question using the COMPLETE dataset information provided below.

{variables_info}{sources_info}

CONTEXT (sample data from search - for reference only, DO NOT use this to determine available variables):
{context_text}

QUESTION: {question}

═══════════════════════════════════════════════════════════════════════════════
CRITICAL INSTRUCTIONS - READ CAREFULLY:
═══════════════════════════════════════════════════════════════════════════════

1. The dataset contains EXACTLY {len(all_variables)} variables (listed above in the "COMPLETE LIST" section)
2. You MUST list ALL {len(all_variables)} variables in your answer
3. Do NOT say "only", "just", "one variable", or "single variable" - there are {len(all_variables)} variables
4. Do NOT limit your answer to variables seen in the "CONTEXT" section - that is just a sample
5. Use the complete numbered list from the "COMPLETE LIST" section above
6. Format your answer as: "Available variables ({len(all_variables)}): [list all variables]"
7. If sources are listed, you can mention them, but you MUST still list ALL {len(all_variables)} variables

═══════════════════════════════════════════════════════════════════════════════

ANSWER (you MUST list all {len(all_variables)} variables from the complete list above):"""
            max_tokens = min(600, len(all_variables) * 7)  # More tokens to ensure all variables fit
        else:
            # Fallback if variables not provided
            prompt = f"""You are a climate data assistant. Answer accurately using the information provided.

{variables_info}{sources_info}

CONTEXT (retrieved data samples):
{context_text}

QUESTION: {question}

INSTRUCTIONS:
- List all available variables from the dataset
- Do NOT make up or invent variables
- Use the information provided above

ANSWER:"""
            max_tokens = 200
        
    elif question_type == "comparison":
        prompt = f"""You are a climate data assistant. Compare and analyze the data accurately.

{variables_info}{sources_info}

CONTEXT (retrieved data):
{context_text}

QUESTION: {question}

INSTRUCTIONS:
- Compare the data points accurately
- Reference specific values from the context
- Mention source and variable names
- Be precise with numbers and units
- If comparing variables, mention both

ANSWER:"""
        max_tokens = 250
        
    elif question_type == "statistical":
        prompt = f"""You are a climate data assistant. Provide statistical analysis based on the data.

{variables_info}{sources_info}{time_periods_info}

CONTEXT (retrieved data):
{context_text}

QUESTION: {question}

INSTRUCTIONS:
- Use ALL available time periods from the data, not just one date
- If asked about a time range (e.g., "summer", "June to August"), use data from ALL dates in that range
- Aggregate statistics across ALL time periods when appropriate
- Use exact statistical values from the context (mean, min, max)
- Reference the variable and source
- Be precise with numbers and mention the time range covered
- If the question asks about a specific period but data covers more, mention the full range available

ANSWER:"""
        max_tokens = 300
        
    elif question_type == "temporal":
        prompt = f"""You are a climate data assistant. Answer questions about temporal patterns in the data.

{variables_info}{sources_info}

CONTEXT (retrieved data with timestamps):
{context_text}

QUESTION: {question}

INSTRUCTIONS:
- Reference specific dates/times from the context
- Mention trends or changes over time
- Include variable names and sources
- Be precise with temporal information

ANSWER:"""
        max_tokens = 200
        
    else:  # general
        prompt = f"""You are a climate data assistant. Answer the question accurately using ONLY the provided data.

{variables_info}{sources_info}{time_periods_info}

CONTEXT (retrieved data):
{context_text}

QUESTION: {question}

INSTRUCTIONS:
- Answer based ONLY on the context provided above
- Use ALL available time periods when answering, not just one date
- If asked about a time range, use data from ALL dates in that range
- Reference specific values, variables, sources, and time periods
- Be accurate and precise
- If information is not in the context, say "I don't have that information in the provided data"
- Do NOT make up or invent information
- If the question asks about a period (e.g., "summer", "June to August"), aggregate across ALL dates in that period

ANSWER:"""
        max_tokens = 300
    
    return prompt, max_tokens


def detect_question_type(question: str) -> str:
    """
    Detect the type of question to build appropriate prompt.
    
    Returns:
        Question type: variable_list, comparison, statistical, temporal, general
    """
    q_lower = question.lower()
    
    # Variable list questions
    if any(phrase in q_lower for phrase in [
        "what variables", "which variables", "list variables", "available variables",
        "what data", "which data", "what fields", "what columns"
    ]):
        return "variable_list"
    
    # Comparison questions
    if any(phrase in q_lower for phrase in [
        "compare", "difference", "versus", "vs", "higher", "lower", "more", "less",
        "better", "worse", "contrast"
    ]):
        return "comparison"
    
    # Statistical questions
    if any(phrase in q_lower for phrase in [
        "average", "mean", "median", "minimum", "maximum", "min", "max",
        "range", "statistics", "statistical", "summary", "overview"
    ]):
        return "statistical"
    
    # Temporal questions
    if any(phrase in q_lower for phrase in [
        "when", "time", "date", "year", "month", "day", "trend", "over time",
        "historical", "past", "future", "change", "increase", "decrease"
    ]):
        return "temporal"
    
    return "general"

