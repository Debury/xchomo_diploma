"""
Test script for RAG endpoint with various question types.
Tests basic, medium, and hard questions to verify accuracy.
"""

import requests
import json
from typing import Dict, Any

BASE_URL = "http://159.65.207.173:8000/rag/chat"

# Expected variables (based on user's information)
EXPECTED_VARIABLES = {
    "AWND", "CDSD", "CLDD", "DP01", "DP05", "DP10", "DSND", "DSNW", 
    "DT00", "DT32", "DX32", "DX70", "DX90", "EMNT", "EMSD", "EMSN", 
    "EMXP", "EMXT", "HDSD", "HTDD", "PRCP", "SNOW", "TMAX", "TMIN", 
    "WDF2", "WDF5", "WSF2", "WSF5", "hurs"
}

def test_question(question: str, expected_keywords: list = None, question_type: str = "general") -> Dict[str, Any]:
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
        
        if question_type == "variable_list":
            # Check if all expected variables are mentioned
            answer_upper = answer.upper()
            missing_vars = []
            for var in EXPECTED_VARIABLES:
                if var.upper() not in answer_upper:
                    missing_vars.append(var)
            
            if missing_vars:
                issues.append(f"Missing variables: {missing_vars[:10]}...")  # Show first 10
            
            # Check for false positives (variables not in expected list)
            # This is harder to detect, but we can check for common mistakes
            if "only" in answer.lower() and len(missing_vars) > 5:
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
    
    # BASIC QUESTIONS
    basic_questions = [
        ("What variables are available?", "variable_list", EXPECTED_VARIABLES),
        ("List all variables in the dataset", "variable_list", EXPECTED_VARIABLES),
        ("Which climate variables do you have?", "variable_list", EXPECTED_VARIABLES),
        ("What is the temperature?", "general", None),
        ("Show me precipitation data", "general", None),
    ]
    
    # MEDIUM QUESTIONS
    medium_questions = [
        ("What is the average temperature?", "statistical", None),
        ("What is the minimum and maximum temperature?", "statistical", None),
        ("Compare temperature and precipitation", "comparison", None),
        ("What variables are available from ISIMP?", "variable_list", EXPECTED_VARIABLES),
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

