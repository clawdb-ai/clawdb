#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OPENCLAW_PROFILE="${OPENCLAW_PROFILE:-clawdb-test}"
CLAWDB_API_KEY="${CLAWDB_API_KEY:-clawdb-smoke-signing-key}"

python3 -m uvicorn clawdb.api:app --host 127.0.0.1 --port 8080 >/tmp/clawdb_uvicorn.log 2>&1 &
PID=$!
cleanup() {
  kill "$PID" >/dev/null 2>&1 || true
}
trap cleanup EXIT

sleep 2

curl -sS -X POST http://127.0.0.1:8080/v1/memory/messages \
  -H 'content-type: application/json' \
  -d '{"session_id":"smoke","role":"user","content":"integration smoke check for clawdb plugin"}' >/dev/null

(
  export CLAWDB_API_KEY
  cd "${ROOT_DIR}/external/openclaw"
  pnpm exec openclaw --profile "${OPENCLAW_PROFILE}" plugins list --json >/tmp/openclaw_plugins.json
  pnpm exec openclaw --profile "${OPENCLAW_PROFILE}" clawdb-memory status >/tmp/openclaw_clawdb_memory_status.txt
  pnpm exec openclaw --profile "${OPENCLAW_PROFILE}" clawdb-memory search "clawdb plugin" --max-results 3 >/tmp/openclaw_clawdb_memory_search.txt
  pnpm exec openclaw --profile "${OPENCLAW_PROFILE}" clawdb-memory get memory/smoke.md --from 1 --lines 5 >/tmp/openclaw_clawdb_memory_get.txt
)

echo "integration smoke test complete"
echo "- plugin list: /tmp/openclaw_plugins.json"
echo "- status output: /tmp/openclaw_clawdb_memory_status.txt"
echo "- search output: /tmp/openclaw_clawdb_memory_search.txt"
echo "- get output: /tmp/openclaw_clawdb_memory_get.txt"
