#!/usr/bin/env bash
# Wait until every run in /tmp/runs_final.txt is terminal. Print a compact
# per-tick table. Tolerates empty status responses.
while :; do
  all_done=1
  echo "--- $(date +%H:%M:%S) ---"
  while read -r sid rid; do
    [ -z "$rid" ] && continue
    Q='{"query":"query { runOrError(runId: \"'"$rid"'\") { __typename ... on Run { status } } }"}'
    status=$(curl -s -X POST http://localhost:3000/graphql -H 'Content-Type: application/json' -d "$Q" | python -c 'import sys,json; d=json.load(sys.stdin); print(d["data"]["runOrError"].get("status",""))' 2>/dev/null)
    status="${status:-UNKNOWN}"
    printf "  %-28s %s\n" "$sid" "$status"
    case "$status" in
      SUCCESS|FAILURE|CANCELED) ;;
      *) all_done=0 ;;
    esac
  done < /tmp/runs_final.txt
  [ $all_done -eq 1 ] && break
  sleep 30
done
echo "--- all runs terminal ---"
