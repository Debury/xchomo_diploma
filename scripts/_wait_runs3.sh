#!/usr/bin/env bash
while :; do
  all_done=1
  echo "--- $(date +%H:%M:%S) ---"
  while read -r sid rid; do
    [ -z "$rid" ] && continue
    Q='{"query":"query { runOrError(runId: \"'"$rid"'\") { __typename ... on Run { status } } }"}'
    status=$(curl -s -X POST http://localhost:3000/graphql -H 'Content-Type: application/json' -d "$Q" | python -c 'import sys,json; d=json.load(sys.stdin); print(d["data"]["runOrError"]["status"])' 2>/dev/null)
    printf "  %-24s %s\n" "$sid" "$status"
    if [ "$status" != "SUCCESS" ] && [ "$status" != "FAILURE" ] && [ "$status" != "CANCELED" ]; then
      all_done=0
    fi
  done < /tmp/runs3.txt
  if [ $all_done -eq 1 ]; then break; fi
  sleep 20
done
echo "--- all runs terminal ---"
