from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Set

import pandas as pd

from .config import ClawDBConfig
from .dataframes import DataFrameStore
from .lineage import RAW_PROJECTION_KIND
from .storage_layout import MessageChannelFile


@dataclass(frozen=True)
class ExportSummary:
    message_count: int
    session_count: int
    channel_file_count: int
    openviking_workspace: str
    qmd_root: str
    qmd_config_path: str


def _repo_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _session_overview(session_df: pd.DataFrame) -> str:
    first_ts = pd.to_datetime(session_df["ts"], utc=True).min().isoformat()
    last_ts = pd.to_datetime(session_df["ts"], utc=True).max().isoformat()
    message_count = int(session_df.shape[0])
    channel_values = sorted({str(value or "") for value in session_df["channel"].tolist()})
    return (
        "# ClawDB Import\n\n"
        f"- Imported from clawdb at {_repo_timestamp()}\n"
        f"- Message count: {message_count}\n"
        f"- First message: {first_ts}\n"
        f"- Last message: {last_ts}\n"
        f"- Channels: {', '.join(value or '(blank)' for value in channel_values)}\n"
    )


def _session_abstract(session_df: pd.DataFrame) -> str:
    message_count = int(session_df.shape[0])
    session_id = str(session_df.iloc[0]["session_id"])
    return f"ClawDB import for session {session_id} with {message_count} messages."


def _openviking_message_line(row: pd.Series) -> str:
    original_role = str(row["role"])
    role = original_role if original_role in {"user", "assistant"} else "assistant"
    text = str(row.get("content") or "")
    if original_role not in {"user", "assistant"}:
        text = f"[{original_role}]\n{text}" if text else f"[{original_role}]"
    payload = {
        "id": str(row["message_id"]),
        "role": role,
        "parts": [{"type": "text", "text": text}],
        "created_at": pd.to_datetime(row["ts"], utc=True).isoformat(),
    }
    return json.dumps(payload, ensure_ascii=False)


def _export_view_rows(messages_df: pd.DataFrame) -> pd.DataFrame:
    if messages_df.empty:
        return messages_df.copy()
    out = messages_df.copy().reset_index(drop=True)
    out["tenant_id"] = out["tenant_id"].fillna("default").astype(str)
    out["session_id"] = out["session_id"].fillna("").astype(str)
    out["native_session_id"] = out["native_session_id"].fillna("").astype(str)
    out["origin_message_id"] = out["origin_message_id"].fillna(out["message_id"]).astype(str)
    out["projection_kind"] = out["projection_kind"].fillna("").astype(str)
    out["ts"] = pd.to_datetime(out["ts"], utc=True, errors="coerce")
    out["updated_at"] = pd.to_datetime(out["updated_at"], utc=True, errors="coerce")
    out["canonical_session_id"] = out["native_session_id"]
    blank_session = out["canonical_session_id"].astype(str) == ""
    out.loc[blank_session, "canonical_session_id"] = out.loc[blank_session, "session_id"]
    out.loc[out["canonical_session_id"].astype(str) == "", "canonical_session_id"] = "default"
    out["_is_projection"] = out["projection_kind"].astype(str) != RAW_PROJECTION_KIND
    out = out.sort_values(
        ["tenant_id", "canonical_session_id", "ts", "updated_at", "_is_projection"],
        kind="stable",
    )
    out = out.drop_duplicates(
        subset=["tenant_id", "canonical_session_id", "origin_message_id"],
        keep="last",
    )
    return out.drop(columns=["_is_projection"]).reset_index(drop=True)


async def export_clawdb_memory(
    *,
    data_root: Path,
    openviking_workspace: Path,
    qmd_root: Path,
) -> ExportSummary:
    config = ClawDBConfig.from_env()
    parquet_dir = data_root / "parquet" if data_root else config.parquet_dir

    store = DataFrameStore()
    await store.load_parquet(parquet_dir)
    messages_df = _export_view_rows(store.state.messages_df.reset_index(drop=True))

    openviking_workspace = openviking_workspace.expanduser().resolve()
    qmd_root = qmd_root.expanduser().resolve()
    openviking_workspace.mkdir(parents=True, exist_ok=True)
    qmd_root.mkdir(parents=True, exist_ok=True)

    channel_paths: Set[str] = set()
    if not messages_df.empty:
        for _, row in messages_df.iterrows():
            channel_paths.add(MessageChannelFile.from_record(row).virtual_path)

    for channel_path in sorted(channel_paths):
        rendered, canonical = await store.virtual_memory_file(channel_path)
        rel_path = canonical.replace("memory/", "", 1)
        target = qmd_root / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(rendered, encoding="utf-8")

    qmd_config_path = qmd_root / "qmd.yml"
    qmd_config_path.write_text(
        (
            "collections:\n"
            "  clawdb:\n"
            f"    path: {qmd_root}\n"
            '    pattern: "**/*.md"\n'
        ),
        encoding="utf-8",
    )

    if not messages_df.empty:
        for (tenant_id, session_id), session_df in messages_df.groupby(
            ["tenant_id", "canonical_session_id"],
            sort=True,
        ):
            # OpenViking maps viking://... URIs into {workspace}/viking/{account_id}/...
            target_dir = (
                openviking_workspace
                / "viking"
                / str(tenant_id)
                / "session"
                / "default"
                / str(session_id)
            )
            target_dir.mkdir(parents=True, exist_ok=True)
            ordered = session_df.sort_values("ts", kind="stable")
            lines = [_openviking_message_line(row) for _, row in ordered.iterrows()]
            (target_dir / "messages.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
            (target_dir / ".abstract.md").write_text(_session_abstract(ordered), encoding="utf-8")
            (target_dir / ".overview.md").write_text(_session_overview(ordered), encoding="utf-8")

    return ExportSummary(
        message_count=int(messages_df.shape[0]),
        session_count=int(messages_df[["tenant_id", "canonical_session_id"]].drop_duplicates().shape[0])
        if not messages_df.empty
        else 0,
        channel_file_count=len(channel_paths),
        openviking_workspace=str(openviking_workspace),
        qmd_root=str(qmd_root),
        qmd_config_path=str(qmd_config_path),
    )


async def _run_cli_async(args: argparse.Namespace) -> int:
    data_root = Path(args.data_root).expanduser().resolve()
    openviking_workspace = Path(args.openviking_workspace).expanduser().resolve()
    qmd_root = Path(args.qmd_root).expanduser().resolve()
    summary = await export_clawdb_memory(
        data_root=data_root,
        openviking_workspace=openviking_workspace,
        qmd_root=qmd_root,
    )
    print(json.dumps(summary.__dict__, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Export compacted clawdb memory into OpenViking and QMD layouts.")
    parser.add_argument(
        "--data-root",
        default=str(ClawDBConfig.from_env().data_root),
        help="ClawDB data root (default: CLAWDB_DATA_ROOT or ./data)",
    )
    parser.add_argument(
        "--openviking-workspace",
        default="~/.openclaw/openviking-workspace",
        help="Target OpenViking workspace root",
    )
    parser.add_argument(
        "--qmd-root",
        default="~/.openclaw/qmd-memory",
        help="Target root for exported markdown files and qmd.yml",
    )
    args = parser.parse_args()
    return asyncio.run(_run_cli_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
