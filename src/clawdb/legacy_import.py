from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import pandas as pd

from .config import ClawDBConfig
from .dataframes import DataFrameStore
from .models import MessageIn
from .service import ClawDBService

QMD_SECTION_RE = re.compile(r"^##\s+(?P<ts>\S+)\s+\[(?P<role>[^\]]+)\]\s*$")
QMD_META_RE = re.compile(r"^- (?P<key>[a-zA-Z_]+):\s*(?P<value>.*)$")
QMD_PROJECTION_RE = re.compile(r"^(?P<kind>[^|]+?)\s*\|\s*scope:\s*(?P<scope>.+)$")


@dataclass(frozen=True)
class LegacySourceRecord:
    source_kind: str
    source_path: str
    message: MessageIn


@dataclass
class LegacyImportReport:
    existing_live_messages: int = 0
    clawdb_backup_candidates: int = 0
    qmd_candidates: int = 0
    openviking_candidates: int = 0
    imported: int = 0
    skipped_duplicates: int = 0
    errors: List[str] = field(default_factory=list)


def _default_projection_scope(message: MessageIn) -> str:
    native_channel_id = (message.native_channel_id or "").strip()
    group_id = (message.group_id or "").strip()
    message_thread_id = (message.message_thread_id or "").strip()
    session_id = (message.session_id or "default").strip() or "default"
    if native_channel_id:
        return f"native:{native_channel_id}"
    if group_id:
        return f"group:{group_id}"
    if message_thread_id:
        return f"thread:{message_thread_id}"
    return f"session:{session_id}"


def _message_identity_keys(message: MessageIn) -> List[str]:
    tenant_id = (message.tenant_id or "default").strip() or "default"
    projection_kind = "raw"
    projection_scope = _default_projection_scope(message).strip()
    origin_message_id = (message.origin_message_id or message.message_id or "").strip()
    message_id = (message.message_id or "").strip()
    ts = pd.Timestamp(message.ts).tz_convert("UTC").isoformat()
    content_hash = hashlib.sha256((message.content or "").encode("utf-8")).hexdigest()
    session_id = (message.session_id or "default").strip() or "default"
    role = (message.role or "user").strip()

    keys = [
        f"fingerprint::{tenant_id}::{session_id}::{role}::{ts}::{content_hash}",
    ]
    if message_id:
        keys.append(f"message::{tenant_id}::{message_id}")
    if origin_message_id:
        keys.append(
            f"origin::{tenant_id}::{origin_message_id}::{projection_kind}::{projection_scope}"
        )
    return keys


def _row_to_message(row: pd.Series) -> MessageIn:
    payload = {field_name: row.get(field_name) for field_name in MessageIn.model_fields}
    return MessageIn.model_validate(payload)


def _parse_qmd_rel_path(rel_path: Path) -> Dict[str, Optional[str]]:
    parts = rel_path.with_suffix("").parts
    return {
        "tenant_id": parts[0] if len(parts) >= 1 else "default",
        "channel": parts[1] if len(parts) >= 2 else None,
        "chat_type": parts[2] if len(parts) >= 3 else None,
        "session_id": parts[-1] if parts else "default",
    }


def parse_qmd_memory_file(path: Path, qmd_root: Path) -> List[LegacySourceRecord]:
    rel_path = path.relative_to(qmd_root)
    defaults = _parse_qmd_rel_path(rel_path)
    lines = path.read_text(encoding="utf-8").splitlines()
    records: List[LegacySourceRecord] = []
    idx = 0

    while idx < len(lines):
        line = lines[idx]
        match = QMD_SECTION_RE.match(line.strip())
        if not match:
            idx += 1
            continue
        ts = match.group("ts")
        role = match.group("role").strip().lower()
        idx += 1
        metadata: Dict[str, str] = {}
        while idx < len(lines):
            meta_match = QMD_META_RE.match(lines[idx].strip())
            if not meta_match:
                break
            metadata[meta_match.group("key")] = meta_match.group("value").strip()
            idx += 1
        if idx < len(lines) and not lines[idx].strip():
            idx += 1
        content_lines: List[str] = []
        while idx < len(lines) and not lines[idx].startswith("## "):
            content_lines.append(lines[idx])
            idx += 1
        content = "\n".join(content_lines).strip()
        projection_kind = "raw"
        projection_scope = ""
        raw_projection = metadata.get("projection", "")
        projection_match = QMD_PROJECTION_RE.match(raw_projection)
        if projection_match:
            projection_kind = projection_match.group("kind").strip() or "raw"
            projection_scope = projection_match.group("scope").strip()
        elif raw_projection:
            projection_kind = raw_projection.strip() or "raw"

        payload = {
            "tenant_id": metadata.get("tenant_id") or defaults["tenant_id"] or "default",
            "session_id": metadata.get("session_id") or defaults["session_id"] or "default",
            "role": role if role in {"user", "assistant", "system", "tool"} else "assistant",
            "content": content,
            "origin_message_id": metadata.get("origin_message_id") or metadata.get("message_id"),
            "projection_kind": projection_kind,
            "projection_scope": projection_scope or f"session:{metadata.get('session_id') or defaults['session_id'] or 'default'}",
            "channel": metadata.get("channel") or defaults["channel"],
            "chat_type": metadata.get("chat_type") or defaults["chat_type"],
            "topic_id": metadata.get("topic_id"),
            "message_id": metadata.get("message_id"),
            "ts": ts,
        }
        records.append(
            LegacySourceRecord(
                source_kind="qmd",
                source_path=str(path),
                message=MessageIn.model_validate(payload),
            )
        )

    return records


def _parse_openviking_rel_path(rel_path: Path) -> Dict[str, Optional[str]]:
    parts = rel_path.parts
    channel = parts[0] if len(parts) >= 1 else "openviking"
    tenant_id = parts[1] if len(parts) >= 2 else "default"
    session_id = parts[-2] if len(parts) >= 2 else "default"
    return {
        "tenant_id": tenant_id,
        "channel": channel,
        "session_id": session_id,
    }


def parse_openviking_messages_file(path: Path, openviking_root: Path) -> List[LegacySourceRecord]:
    rel_path = path.relative_to(openviking_root)
    defaults = _parse_openviking_rel_path(rel_path)
    records: List[LegacySourceRecord] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        parts = item.get("parts") or []
        texts = [str(part.get("text") or "") for part in parts if str(part.get("type") or "") == "text"]
        payload = {
            "tenant_id": defaults["tenant_id"] or "default",
            "session_id": defaults["session_id"] or "default",
            "role": str(item.get("role") or "assistant").strip().lower(),
            "content": "\n".join(text for text in texts if text).strip(),
            "origin_message_id": item.get("id"),
            "projection_kind": "raw",
            "projection_scope": f"session:{defaults['session_id'] or 'default'}",
            "channel": defaults["channel"] or "openviking",
            "message_id": item.get("id"),
            "ts": item.get("created_at"),
        }
        role = payload["role"]
        if role not in {"user", "assistant", "system", "tool"}:
            payload["role"] = "assistant"
        records.append(
            LegacySourceRecord(
                source_kind="openviking",
                source_path=str(path),
                message=MessageIn.model_validate(payload),
            )
        )
    return records


def discover_clawdb_backup_roots(data_root: Path) -> List[Path]:
    backup_root = data_root / "backups"
    if not backup_root.exists():
        return []
    roots = {
        messages_dir.parent.parent
        for messages_dir in backup_root.rglob("parquet/messages")
        if messages_dir.is_dir()
    }
    return sorted(root for root in roots if (root / "parquet").exists())


async def load_clawdb_messages_from_root(root: Path) -> List[LegacySourceRecord]:
    parquet_dir = root / "parquet"
    if not parquet_dir.exists():
        return []
    store = DataFrameStore()
    await store.load_parquet(parquet_dir)
    df = store.state.messages_df.reset_index(drop=True)
    records: List[LegacySourceRecord] = []
    for _, row in df.iterrows():
        records.append(
            LegacySourceRecord(
                source_kind="clawdb-backup",
                source_path=str(root),
                message=_row_to_message(row),
            )
        )
    return records


def _iter_qmd_files(qmd_root: Path) -> Iterator[Path]:
    if not qmd_root.exists():
        return iter(())
    return (
        path
        for path in sorted(qmd_root.rglob("*.md"))
        if path.name != "qmd.yml"
    )


def _iter_openviking_files(openviking_root: Path) -> Iterator[Path]:
    if not openviking_root.exists():
        return iter(())
    return iter(sorted(openviking_root.rglob("messages.jsonl")))


async def import_legacy_memory(
    *,
    data_root: Path,
    openviking_root: Path,
    qmd_root: Path,
    dry_run: bool = False,
) -> LegacyImportReport:
    report = LegacyImportReport()
    previous_data_root = os.environ.get("CLAWDB_DATA_ROOT")
    previous_auto_migrate = os.environ.get("CLAWDB_SCHEMA_AUTO_MIGRATE")
    os.environ["CLAWDB_DATA_ROOT"] = str(data_root)
    os.environ["CLAWDB_SCHEMA_AUTO_MIGRATE"] = "false"
    config = ClawDBConfig.from_env()
    service = ClawDBService(config=config)
    await service.startup()
    try:
        live_df = service.df_store.state.messages_df.reset_index(drop=True)
        report.existing_live_messages = int(live_df.shape[0])
        seen_keys = {
            key
            for _, row in live_df.iterrows()
            for key in _message_identity_keys(_row_to_message(row))
        }

        candidates: List[LegacySourceRecord] = []

        for root in discover_clawdb_backup_roots(data_root):
            try:
                loaded = await load_clawdb_messages_from_root(root)
                report.clawdb_backup_candidates += len(loaded)
                candidates.extend(loaded)
            except Exception as exc:
                report.errors.append(f"failed to load clawdb backup {root}: {exc}")

        for path in _iter_qmd_files(qmd_root):
            try:
                loaded = parse_qmd_memory_file(path, qmd_root)
                report.qmd_candidates += len(loaded)
                candidates.extend(loaded)
            except Exception as exc:
                report.errors.append(f"failed to parse qmd file {path}: {exc}")

        for path in _iter_openviking_files(openviking_root):
            try:
                loaded = parse_openviking_messages_file(path, openviking_root)
                report.openviking_candidates += len(loaded)
                candidates.extend(loaded)
            except Exception as exc:
                report.errors.append(f"failed to parse openviking file {path}: {exc}")

        for record in candidates:
            keys = _message_identity_keys(record.message)
            if any(key in seen_keys for key in keys):
                report.skipped_duplicates += 1
                continue
            if not dry_run:
                await service.ingest_message(record.message)
            seen_keys.update(keys)
            report.imported += 1

        if not dry_run:
            await service.flush_now()
    finally:
        await service.shutdown()
        if previous_data_root is None:
            os.environ.pop("CLAWDB_DATA_ROOT", None)
        else:
            os.environ["CLAWDB_DATA_ROOT"] = previous_data_root
        if previous_auto_migrate is None:
            os.environ.pop("CLAWDB_SCHEMA_AUTO_MIGRATE", None)
        else:
            os.environ["CLAWDB_SCHEMA_AUTO_MIGRATE"] = previous_auto_migrate

    return report


async def _run_cli_async(args: argparse.Namespace) -> int:
    data_root = Path(args.data_root).expanduser().resolve()
    openviking_root = Path(args.openviking_root).expanduser().resolve()
    qmd_root = Path(args.qmd_root).expanduser().resolve()
    report = await import_legacy_memory(
        data_root=data_root,
        openviking_root=openviking_root,
        qmd_root=qmd_root,
        dry_run=bool(args.dry_run),
    )
    print(json.dumps(asdict(report), indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m clawdb.legacy_import",
        description="Import legacy clawdb/OpenViking/QMD raw messages into the current clawdb store.",
    )
    parser.add_argument(
        "--data-root",
        default=str(ClawDBConfig.from_env().data_root),
        help="Target clawdb data root",
    )
    parser.add_argument(
        "--openviking-root",
        default="~/.openclaw/openviking-workspace",
        help="OpenViking workspace root",
    )
    parser.add_argument(
        "--qmd-root",
        default="~/.openclaw/qmd-memory",
        help="QMD markdown root",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan sources and report import counts without mutating clawdb data",
    )
    args = parser.parse_args()
    return asyncio.run(_run_cli_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
