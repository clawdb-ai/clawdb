# OpenClaw Compatibility Notes

## Memory slot compatibility

- Plugin id: `memory-clawdb`
- Plugin kind: `memory`
- Slot target: `plugins.slots.memory = memory-clawdb`
- Legacy slot provider: `memory-core` should be disabled in the same profile.

## Routed tools and CLI

- Tool alias: `memory_search` -> `POST /v1/openclaw/memory/search`
- Tool alias: `memory_get` -> `POST /v1/openclaw/memory/get`
- CLI helpers:
  - `openclaw clawdb-memory status`
  - `openclaw clawdb-memory search <query>`
  - `openclaw clawdb-memory get <relPath>`

## Auth/signing and embeddings

- Plugin request signing uses the active clawdb API key (`Authorization: Bearer ...`).
- Plugin forwards OpenClaw model-auth resolved embedding credentials using headers:
  - `x-clawdb-embedding-provider`
  - `x-clawdb-embedding-key`
  - `x-clawdb-embedding-model`
  - optional `x-clawdb-embedding-base-url`
- ClawDB verifies OpenClaw-compatible signatures by default on `/v1/openclaw/memory/*`.

## Recommended profile checks

```bash
pnpm exec openclaw --profile clawdb-test config get plugins.slots.memory
pnpm exec openclaw --profile clawdb-test plugins list --json
pnpm exec openclaw --profile clawdb-test clawdb-memory status
```
