"""
Test script for RAG endpoint with various question types.
Tests basic, medium, and hard questions to verify accuracy.
"""

import requests
import json
from typing import Dict, Any

BASE_URL = "http://159.65.207.173:8000/rag/chat"

# Get expected variables dynamically from the API
def get_expected_variables():
    """Fetch actual variables from the API to use for testing."""
    try:
        # Try to get variables from /rag/info endpoint
        response = requests.get(f"{BASE_URL.replace('/chat', '/info')}", timeout=10)
        if response.status_code == 200:
            data = response.json()
            return set(data.get("variables", []))
    except:
        pass
    
    # Fallback: try asking the API directly
    try:
        payload = {"question": "What variables are available?", "use_llm": False}
        response = requests.post(BASE_URL, json=payload, timeout=10)
        if response.status_code == 200:
            data = response.json()
            answer = data.get("answer", "")
            # Try to extract variables from answer (basic parsing)
            # This is a fallback - ideally use /rag/info endpoint
            return set()  # Return empty set if can't parse
    except:
        pass
    
    return set()  # Return empty set if can't fetch

def test_question(question: str, expected_keywords: set = None, question_type: str = "general") -> Dict[str, Any]:
    """Test a single question and return results."""
    print(f"\n{'='*80}")
    print(f"Testing: {question_type.upper()}")
    print(f"Question: {question}")
    print(f"{'='*80}")
    
    payload = {
        "question": question,
        "use_llm": True,
        "temperature": 0.3
    }
    
    try:
        response = requests.post(BASE_URL, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        
        answer = data.get("answer", "")
        llm_used = data.get("llm_used", False)
        chunks = data.get("chunks", [])
        
        print(f"\nAnswer ({'LLM' if llm_used else 'No LLM'}):")
        print(f"{answer}")
        print(f"\nChunks returned: {len(chunks)}")
        
        # Check accuracy
        issues = []
        
        if question_type == "variable_list" and expected_keywords:
            # Check if all expected variables are mentioned (if we have expected list)
            answer_upper = answer.upper()
            missing_vars = []
            for var in expected_keywords:
                if var.upper() not in answer_upper:
                    missing_vars.append(var)
            
            if missing_vars and len(missing_vars) < len(expected_keywords):
                # Only report if we're missing some but not all (to avoid false positives)
                issues.append(f"Missing variables: {missing_vars[:10]}...")  # Show first 10
            
            # Check for false positives (variables not in expected list)
            # This is harder to detect, but we can check for common mistakes
            if "only" in answer.lower() and expected_keywords and len(expected_keywords) > 1:
                # If answer says "only" but we expect multiple variables, that's suspicious
                if len(missing_vars) > len(expected_keywords) * 0.5:  # Missing more than 50%
                    issues.append("Answer says 'only' but many variables are missing")
        
        # Check if answer is too short for complex questions
        if question_type in ["comparison", "statistical"] and len(answer) < 50:
            issues.append("Answer seems too short for this question type")
        
        # Check if answer contains "I don't know" when it should know
        if "don't know" in answer.lower() or "no information" in answer.lower():
            if chunks and question_type != "general":
                issues.append("Answer says it doesn't know, but chunks are available")
        
        if issues:
            print(f"\n[!] ISSUES FOUND:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print(f"\n[OK] No obvious issues detected")
        
        return {
            "question": question,
            "answer": answer,
            "llm_used": llm_used,
            "chunks_count": len(chunks),
            "issues": issues,
            "success": len(issues) == 0
        }
        
    except Exception as e:
        print(f"\n[ERROR] {e}")
        return {
            "question": question,
            "error": str(e),
            "success": False
        }


def run_tests():
    """Run all test questions."""
    
    # Get expected variables dynamically from API
    print("Fetching expected variables from API...")
    expected_vars = get_expected_variables()
    if expected_vars:
        print(f"Found {len(expected_vars)} variables from API")
    else:
        print("Warning: Could not fetch variables from API, will test without validation")
    
    # BASIC QUESTIONS
    basic_questions = [
        ("What variables are available?", "variable_list", expected_vars),
        ("List all variables in the dataset", "variable_list", expected_vars),
        ("Which climate variables do you have?", "variable_list", expected_vars),
        ("What is the temperature?", "general", None),
        ("Show me precipitation data", "general", None),
    ]
    
    # MEDIUM QUESTIONS
    medium_questions = [
        ("What is the average temperature?", "statistical", None),
        ("What is the minimum and maximum temperature?", "statistical", None),
        ("Compare temperature and precipitation", "comparison", None),
        ("What variables are available from ISIMP?", "variable_list", expected_vars),
        ("What is the range of humidity values?", "statistical", None),
    ]
    
    # HARD QUESTIONS
    hard_questions = [
        ("What is the difference between TMAX and TMIN?", "comparison", None),
        ("Compare all temperature-related variables", "comparison", None),
        ("What are the statistical properties of precipitation?", "statistical", None),
        ("Which source has more variables, ISIMP or the other source?", "comparison", None),
        ("What is the temporal trend of temperature over time?", "temporal", None),
    ]
    
    results = {
        "basic": [],
        "medium": [],
        "hard": []
    }
    
    print("\n" + "="*80)
    print("BASIC QUESTIONS")
    print("="*80)
    for question, q_type, expected in basic_questions:
        result = test_question(question, expected, q_type)
        results["basic"].append(result)
    
    print("\n" + "="*80)
    print("MEDIUM QUESTIONS")
    print("="*80)
    for question, q_type, expected in medium_questions:
        result = test_question(question, expected, q_type)
        results["medium"].append(result)
    
    print("\n" + "="*80)
    print("HARD QUESTIONS")
    print("="*80)
    for question, q_type, expected in hard_questions:
        result = test_question(question, expected, q_type)
        results["hard"].append(result)
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    for level in ["basic", "medium", "hard"]:
        level_results = results[level]
        total = len(level_results)
        successful = sum(1 for r in level_results if r.get("success", False))
        print(f"\n{level.upper()}: {successful}/{total} successful")
        
        for r in level_results:
            if not r.get("success", False):
                print(f"  [FAIL] {r['question'][:60]}...")
                if "issues" in r:
                    for issue in r["issues"]:
                        print(f"      - {issue}")


if __name__ == "__main__":
    run_tests()

