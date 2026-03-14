#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OPENCLAW_DIR="${ROOT_DIR}/external/openclaw"

if [[ ! -d "${OPENCLAW_DIR}" ]]; then
  mkdir -p "${ROOT_DIR}/external"
  git clone --depth 1 https://github.com/openclaw/openclaw.git "${OPENCLAW_DIR}"
fi

cd "${OPENCLAW_DIR}"
if command -v corepack >/dev/null 2>&1; then
  corepack enable >/dev/null 2>&1 || true
fi
if ! command -v pnpm >/dev/null 2>&1; then
  corepack prepare pnpm@latest --activate
fi
pnpm install

cd "${ROOT_DIR}"
"${ROOT_DIR}/scripts/install_openclaw_integration.sh"

echo "openclaw bootstrap complete"
