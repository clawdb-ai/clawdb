---
name: clawdb-first-memory
description: Configure clawdb as the first-choice OpenClaw memory backend and verify end-to-end health/search/get, including schema migration readiness. Trigger when asked to set up clawdb memory replacement, switch OpenClaw memory slot to clawdb, or validate memory-clawdb integration for Codex/CC/OpenClaw workflows.
metadata:
  {
    "openclaw":
      {
        "emoji": "🧠",
        "requires": { "bins": ["python3", "pnpm", "curl"] },
      },
  }
---

# ClawDB First Memory

Use this skill when the user wants OpenClaw to use clawdb as the primary memory backend.

## What this skill does

1. Installs or links the `memory-clawdb` OpenClaw memory plugin.
2. Switches the OpenClaw memory slot to `memory-clawdb`.
3. Writes practical plugin config defaults (`baseUrl`, `apiKey`, timeout).
4. Runs verification checks (`status`, `search`, `get`).
5. Provides schema migration commands for forward-compatible clawdb upgrades.

## Quick start

```bash
bash {baseDir}/scripts/setup_first_choice_memory.sh \
  --repo-root /path/to/clawdb \
  --openclaw-dir /path/to/openclaw \
  --profile clawdb-test \
  --base-url http://127.0.0.1:8080 \
  --api-key "${CLAWDB_API_KEY:-}" \
  --verify
```

If OpenClaw is not cloned yet, add `--bootstrap-openclaw`.

## Notes for Codex/CC/OpenClaw agents

- Prefer this script over ad-hoc manual edits. It is idempotent and safe to rerun.
- The plugin reuses OpenClaw auth/profile model credentials for embedding headers.
- Request signing is preserved using the configured clawdb API key.
- Keep `memory-core` disabled when `memory-clawdb` is selected for the memory slot.

## Schema migration workflow

Run this before or after clawdb upgrades when schema fields change:

```bash
python -m clawdb.migrate --data-root data --dry-run --json
python -m clawdb.migrate --data-root data
```

- The migration keeps WAL files intact.
- Backups are created by default under `<data-root>/backups/schema-migrations`.
- Migrated data remains readable across linear memory, capsule cards, and forum/topic views.

## Troubleshooting

- If `plugins list` does not show `memory-clawdb`, rerun setup with `--bootstrap-openclaw`.
- If signed OpenClaw adapter calls fail, verify `apiKey` matches clawdb service auth/signing key.
- If search returns empty results, run the verify script after seeding one message to clawdb.

## References

- Compatibility details: `references/openclaw-compatibility.md`
- Upgrade safety checklist: `references/schema-migration-checklist.md`
