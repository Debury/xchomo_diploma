#!/usr/bin/env bash
# Force-cancel every non-terminal Dagster run with full response readout.
set -eo pipefail

# Pull all non-terminal run IDs
RIDS=$(curl -s -X POST http://localhost:3000/graphql \
  -H 'Content-Type: application/json' \
  -d '{"query":"query { runsOrError(filter: {statuses: [STARTED, STARTING, QUEUED, CANCELING]}) { ... on Runs { results { runId } } } }"}' \
  | python -c 'import sys,json; d=json.load(sys.stdin)["data"]["runsOrError"]["results"]; print(json.dumps([r["runId"] for r in d]))')

echo "Cancelling: $RIDS"

curl -s -X POST http://localhost:3000/graphql \
  -H 'Content-Type: application/json' \
  -d "$(python -c "import json,sys; rids=$RIDS; q='mutation { terminateRuns(runIds: '+json.dumps(rids)+', terminatePolicy: MARK_AS_CANCELED_IMMEDIATELY) { __typename ... on TerminateRunsResult { terminateRunResults { __typename ... on TerminateRunSuccess { run { runId status } } ... on TerminateRunFailure { message } } } ... on PythonError { message } } }'; print(json.dumps({'query': q}))")" \
  | python -m json.tool
