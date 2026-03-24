from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Mapping, Optional
from urllib.parse import quote, unquote


BLANK_COMPONENT = "__blank__"
MESSAGE_SCOPE_KINDS = {"native", "group", "thread", "session"}


def encode_component(value: object) -> str:
    text = "" if value is None else str(value)
    if text == "":
        return BLANK_COMPONENT
    return quote(text, safe="")


def decode_component(value: str) -> str:
    if value == BLANK_COMPONENT:
        return ""
    return unquote(value)


def table_has_legacy_partitions(table_dir: Path) -> bool:
    return any(path.is_dir() for path in table_dir.glob("dt=*"))


def table_has_zero_byte_parquet(table_dir: Path) -> bool:
    return any(path.is_file() and path.stat().st_size == 0 for path in table_dir.rglob("*.parquet"))


def table_parquet_files(table_dir: Path) -> List[Path]:
    if not table_dir.exists():
        return []

    files: List[Path] = []
    if table_has_legacy_partitions(table_dir):
        for partition_dir in sorted(path for path in table_dir.glob("dt=*") if path.is_dir()):
            candidates = sorted(
                path
                for path in partition_dir.glob("part-*.parquet")
                if path.is_file() and path.stat().st_size > 0
            )
            if candidates:
                files.append(candidates[-1])
        files.extend(
            sorted(
                path
                for path in table_dir.glob("*.parquet")
                if path.is_file() and path.stat().st_size > 0
            )
        )
    else:
        files.extend(
            sorted(
                path
                for path in table_dir.rglob("*.parquet")
                if path.is_file() and path.stat().st_size > 0
            )
        )

    unique: List[Path] = []
    seen = set()
    for file_path in files:
        resolved = str(file_path.resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(file_path)
    return unique


@dataclass(frozen=True)
class MessageChannelFile:
    tenant_id: str
    channel: str
    chat_type: str
    scope_kind: str
    scope_value: str

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> "MessageChannelFile":
        tenant_id = str(record.get("tenant_id") or "default")
        channel = str(record.get("channel") or "")
        chat_type = str(record.get("chat_type") or "")
        native_channel_id = str(record.get("native_channel_id") or "")
        group_id = str(record.get("group_id") or "")
        message_thread_id = str(record.get("message_thread_id") or "")
        native_session_id = str(record.get("native_session_id") or "")
        session_id = native_session_id or str(record.get("session_id") or "default")

        if native_channel_id:
            scope_kind = "native"
            scope_value = native_channel_id
        elif group_id:
            scope_kind = "group"
            scope_value = group_id
        elif message_thread_id:
            scope_kind = "thread"
            scope_value = message_thread_id
        else:
            scope_kind = "session"
            scope_value = session_id

        return cls(
            tenant_id=tenant_id,
            channel=channel,
            chat_type=chat_type,
            scope_kind=scope_kind,
            scope_value=scope_value,
        )

    @classmethod
    def from_virtual_path(cls, rel_path: str) -> Optional["MessageChannelFile"]:
        normalized = rel_path.replace("\\", "/").lstrip("/")
        if not normalized.startswith("memory/") or not normalized.endswith(".md"):
            return None
        parts = normalized.split("/")
        if len(parts) != 6:
            return None
        _, tenant_id, channel, chat_type, scope_kind, filename = parts
        if scope_kind not in MESSAGE_SCOPE_KINDS:
            return None
        scope_value = filename[:-3]
        return cls(
            tenant_id=decode_component(tenant_id),
            channel=decode_component(channel),
            chat_type=decode_component(chat_type),
            scope_kind=scope_kind,
            scope_value=decode_component(scope_value),
        )

    @property
    def virtual_path(self) -> str:
        return (
            "memory/"
            f"{encode_component(self.tenant_id)}/"
            f"{encode_component(self.channel)}/"
            f"{encode_component(self.chat_type)}/"
            f"{self.scope_kind}/"
            f"{encode_component(self.scope_value)}.md"
        )

    @property
    def storage_relative_path(self) -> Path:
        return (
            Path(f"tenant={encode_component(self.tenant_id)}")
            / f"channel={encode_component(self.channel)}"
            / f"chat={encode_component(self.chat_type)}"
            / f"{self.scope_kind}={encode_component(self.scope_value)}.parquet"
        )
