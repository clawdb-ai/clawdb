# Schema Migration Checklist

Use this checklist when upgrading clawdb versions with schema changes.

1. Stop writers or enter read-only mode for migration window.
2. Run a dry run:

```bash
python -m clawdb.migrate --data-root data --dry-run --json
```

3. Review missing-column plan and backup destination.
4. Run migration:

```bash
python -m clawdb.migrate --data-root data
```

5. Validate startup and replay:

```bash
python -m uvicorn clawdb.api:app --host 127.0.0.1 --port 8080
```

6. Validate memory read paths:

```bash
curl -sS http://127.0.0.1:8080/v1/memory/health
curl -sS http://127.0.0.1:8080/v1/memory/present/linear/<session_id>
curl -sS http://127.0.0.1:8080/v1/memory/present/capsules/<session_id>
curl -sS http://127.0.0.1:8080/v1/memory/present/forum/<session_id>
```

7. Verify OpenClaw plugin path still works (`clawdb-memory status/search/get`).
8. Keep backup until a full replay+search parity check passes.

## Safety guarantees

- WAL files are not rewritten by schema migration.
- Migration is idempotent and can be rerun.
- Existing history remains readable through L0/L1/L2 paths after migration.
