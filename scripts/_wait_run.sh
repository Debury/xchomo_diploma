#!/usr/bin/env bash
RUN_ID="$1"
Q='{"query":"query { runOrError(runId: \"'"$RUN_ID"'\") { __typename ... on Run { status } } }"}'
while true; do
  STATUS=$(curl -s -X POST http://localhost:3000/graphql -H 'Content-Type: application/json' -d "$Q" | python -c 'import sys,json; d=json.load(sys.stdin); print(d["data"]["runOrError"]["status"])' 2>/dev/null)
  echo "$(date +%H:%M:%S) $STATUS"
  if [ "$STATUS" = "SUCCESS" ] || [ "$STATUS" = "FAILURE" ] || [ "$STATUS" = "CANCELED" ]; then
    exit 0
  fi
  sleep 5
done
