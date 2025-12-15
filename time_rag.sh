#!/bin/bash
# Time RAG query performance
# Usage: ./time_rag.sh "your question here"

SERVER="http://localhost:8000"
QUESTION="${1:-list variables}"

echo "=== RAG Performance Test ==="
echo "Server: $SERVER"
echo "Question: $QUESTION"
echo ""

# Time the full request
echo "Starting request..."
START=$(date +%s%3N)

RESPONSE=$(curl -s -X POST "$SERVER/rag/chat" \
  -H "Content-Type: application/json" \
  -d "{\"question\": \"$QUESTION\", \"limit\": 3}")

END=$(date +%s%3N)
TOTAL=$((END - START))

echo ""
echo "=== Results ==="
echo "Total request time: ${TOTAL}ms"
echo ""

# Parse JSON for internal timings
SEARCH_MS=$(echo "$RESPONSE" | grep -o '"search_time_ms":[0-9.]*' | cut -d: -f2)
LLM_MS=$(echo "$RESPONSE" | grep -o '"llm_time_ms":[0-9.]*' | cut -d: -f2)
ANSWER=$(echo "$RESPONSE" | grep -o '"answer":"[^"]*"' | cut -d: -f2- | tr -d '"' | head -c 200)

echo "Internal timings:"
echo "  Search time: ${SEARCH_MS:-N/A}ms"
echo "  LLM time: ${LLM_MS:-N/A}ms"
echo ""
echo "Answer: ${ANSWER:-No answer}..."
