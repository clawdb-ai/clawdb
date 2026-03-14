#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OPENCLAW_DIR="${ROOT_DIR}/external/openclaw"
PLUGIN_SRC="${ROOT_DIR}/integration/openclaw/memory-clawdb"
PLUGIN_DST="${OPENCLAW_DIR}/extensions/memory-clawdb"
OPENCLAW_PROFILE="${OPENCLAW_PROFILE:-clawdb-test}"

if [[ ! -d "${OPENCLAW_DIR}" ]]; then
  echo "openclaw clone not found at ${OPENCLAW_DIR}" >&2
  echo "run: git clone --depth 1 https://github.com/openclaw/openclaw.git external/openclaw" >&2
  exit 1
fi

rm -rf "${PLUGIN_DST}"
mkdir -p "${PLUGIN_DST}"
cp -R "${PLUGIN_SRC}"/* "${PLUGIN_DST}"/

echo "installed plugin at ${PLUGIN_DST}"

cat > "${OPENCLAW_DIR}/openclaw.clawdb.config.example.json" <<'JSON'
{
  "plugins": {
    "slots": {
      "memory": "memory-clawdb"
    },
    "allow": ["memory-clawdb"],
    "entries": {
      "memory-core": { "enabled": false },
      "memory-clawdb": {
        "enabled": true,
        "config": {
          "baseUrl": "http://127.0.0.1:8080",
          "apiKey": "replace-with-clawdb-signing-key-or-use-env",
          "requestTimeoutMs": 10000
        }
      }
    }
  }
}
JSON

echo "wrote config template: ${OPENCLAW_DIR}/openclaw.clawdb.config.example.json"

if command -v pnpm >/dev/null 2>&1; then
  (
    cd "${OPENCLAW_DIR}"
    # Register plugin in OpenClaw's profile-managed plugin config.
    pnpm exec openclaw --profile "${OPENCLAW_PROFILE}" plugins install "${PLUGIN_SRC}" --link >/dev/null
    echo "installed into OpenClaw profile: ${OPENCLAW_PROFILE}"
  )
fi
