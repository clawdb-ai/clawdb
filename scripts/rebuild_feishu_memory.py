#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import httpx
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from clawdb.lineage import RAW_PROJECTION_KIND, materialize_message_bundle, normalize_platform
from clawdb.migrate import CURRENT_SCHEMA_VERSION, SCHEMA_VERSION_SLOT
from clawdb.models import MessageIn
from clawdb.service import ClawDBService


APP_ID = os.getenv("FEISHU_APP_ID", "cli_a93d537fe638dcb3")
SECRETS_PATH = Path(os.getenv("FEISHU_SECRETS_PATH", "~/.openclaw/secrets.store.json")).expanduser()
OPENCLAW_ROOT = Path(os.getenv("OPENCLAW_ROOT", "~/.openclaw")).expanduser()
LIVE_ROOT = Path(os.getenv("CLAWDB_LIVE_ROOT", str(OPENCLAW_ROOT / "clawdb-data"))).expanduser()
SYNC_SCRIPT = OPENCLAW_ROOT / "bin" / "sync-clawdb-mirrors.sh"
SUMMARY_DIR = OPENCLAW_ROOT
PROGRESS_PATH = Path(
    os.environ.get("FEISHU_IMPORT_PROGRESS_PATH", "/tmp/feishu_clawdb_rebuild_progress.json")
)
SUMMARY_PATH = Path(
    os.environ.get("FEISHU_IMPORT_SUMMARY_PATH", "/tmp/feishu_clawdb_rebuild_summary.json")
)
REQUEST_TIMEOUT = 30.0
PAGE_SIZE = 50
EMBED_BATCH_SIZE = 64


@dataclass(frozen=True)
class ChatSpec:
    chat_id: str
    name: str


class FeishuClient:
    def __init__(self, app_id: str, app_secret: str) -> None:
        self.app_id = app_id
        self.app_secret = app_secret
        self.client = httpx.Client(timeout=REQUEST_TIMEOUT)
        self.headers = {"Authorization": f"Bearer {self._tenant_token()}"}

    def _tenant_token(self) -> str:
        response = self.client.post(
            "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal",
            json={"app_id": self.app_id, "app_secret": self.app_secret},
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("code") != 0:
            raise RuntimeError(f"Feishu auth failed: {payload}")
        return str(payload["tenant_access_token"])

    def list_chats(self) -> List[ChatSpec]:
        chats: List[ChatSpec] = []
        page_token: Optional[str] = None
        while True:
            params: Dict[str, object] = {"page_size": 100}
            if page_token:
                params["page_token"] = page_token
            response = self.client.get(
                "https://open.feishu.cn/open-apis/im/v1/chats",
                headers=self.headers,
                params=params,
            )
            response.raise_for_status()
            payload = response.json()
            if payload.get("code") != 0:
                raise RuntimeError(f"Feishu chat list failed: {payload}")
            data = payload.get("data", {})
            for item in data.get("items", []):
                chat_id = str(item.get("chat_id") or "").strip()
                if not chat_id:
                    continue
                chats.append(ChatSpec(chat_id=chat_id, name=str(item.get("name") or "").strip()))
            if not data.get("has_more"):
                break
            page_token = str(data.get("page_token") or "").strip()
            if not page_token:
                break
        deduped = {chat.chat_id: chat for chat in chats}
        return [deduped[key] for key in sorted(deduped.keys())]

    def fetch_messages(self, chat: ChatSpec, progress: Dict[str, Any]) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        page_token: Optional[str] = None
        seen_tokens: set[str] = set()
        page_count = 0
        while True:
            params: Dict[str, object] = {
                "container_id_type": "chat",
                "container_id": chat.chat_id,
                "page_size": PAGE_SIZE,
            }
            if page_token:
                if page_token in seen_tokens:
                    raise RuntimeError(f"repeated page token for {chat.chat_id}")
                seen_tokens.add(page_token)
                params["page_token"] = page_token
            response = self.client.get(
                "https://open.feishu.cn/open-apis/im/v1/messages",
                headers=self.headers,
                params=params,
            )
            response.raise_for_status()
            payload = response.json()
            if payload.get("code") != 0:
                raise RuntimeError(f"Feishu message list failed for {chat.chat_id}: {payload}")
            data = payload.get("data", {})
            items = data.get("items", [])
            results.extend(items)
            page_count += 1
            progress["phase"] = "fetch"
            progress["current_chat"] = chat.chat_id
            progress["current_chat_name"] = chat.name
            progress["current_chat_pages"] = page_count
            progress["current_chat_messages_fetched"] = len(results)
            write_json(PROGRESS_PATH, progress)
            if not data.get("has_more"):
                break
            page_token = str(data.get("page_token") or "").strip()
            if not page_token:
                break
            time.sleep(0.05)
        deduped: Dict[str, Dict[str, Any]] = {}
        for item in results:
            message_id = str(item.get("message_id") or "").strip()
            if message_id:
                deduped[message_id] = item
        ordered = list(deduped.values())
        ordered.sort(key=lambda item: (int(item.get("create_time") or 0), str(item.get("message_id") or "")))
        return ordered


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_secret() -> str:
    payload = json.loads(SECRETS_PATH.read_text(encoding="utf-8"))
    secret = str(payload.get("FEISHU_APP_SECRET") or "").strip()
    if not secret:
        raise RuntimeError("FEISHU_APP_SECRET missing")
    return secret


def flatten_strings(value: Any) -> List[str]:
    out: List[str] = []
    if value is None:
        return out
    if isinstance(value, str):
        text = value.strip()
        if text:
            out.append(text)
        return out
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"file_key", "image_key", "media_id", "signature"}:
                continue
            out.extend(flatten_strings(item))
        return out
    if isinstance(value, list):
        for item in value:
            out.extend(flatten_strings(item))
        return out
    text = str(value).strip()
    if text:
        out.append(text)
    return out


def render_content(item: Dict[str, Any]) -> str:
    msg_type = str(item.get("msg_type") or "unknown").strip() or "unknown"
    body = item.get("body") or {}
    raw_content = body.get("content")
    mentions = item.get("mentions") or []
    if isinstance(raw_content, str):
        try:
            parsed = json.loads(raw_content)
        except Exception:
            parsed = raw_content
    else:
        parsed = raw_content
    mention_map = {
        str(mention.get("key") or ""): "@"
        + (str(mention.get("name") or mention.get("id") or "").strip() or "unknown")
        for mention in mentions
        if str(mention.get("key") or "").strip()
    }
    if isinstance(parsed, dict) and isinstance(parsed.get("text"), str):
        text = parsed.get("text") or ""
        for key, repl in mention_map.items():
            text = text.replace(key, repl)
        text = text.strip()
        if text:
            return text
    parts = flatten_strings(parsed)
    rendered = " | ".join(part for part in parts if part)
    for key, repl in mention_map.items():
        rendered = rendered.replace(key, repl)
    rendered = rendered.strip()
    if rendered:
        return rendered
    return f"[{msg_type}]"


def parse_ts_ms(value: Any) -> datetime:
    try:
        return datetime.fromtimestamp(int(value) / 1000.0, tz=timezone.utc)
    except Exception:
        return datetime.now(timezone.utc)


def _first_nonempty(*values: object) -> Optional[str]:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return None


def extract_reply_to_id(item: Dict[str, Any]) -> Optional[str]:
    body = item.get("body") or {}
    return _first_nonempty(
        item.get("reply_to_id"),
        item.get("parent_id"),
        item.get("upper_message_id"),
        body.get("reply_to_id"),
        body.get("parent_id"),
    )


def message_role(item: Dict[str, Any]) -> str:
    msg_type = str(item.get("msg_type") or "").strip().lower()
    sender = item.get("sender") or {}
    sender_type = str(sender.get("sender_type") or "").strip().lower()
    if msg_type == "system":
        return "system"
    if sender_type in {"app", "bot"}:
        return "assistant"
    return "user"


def message_in(chat: ChatSpec, item: Dict[str, Any]) -> MessageIn:
    sender = item.get("sender") or {}
    sender_id = str(sender.get("id") or "").strip() or None
    content = render_content(item)
    create_ts = parse_ts_ms(item.get("create_time"))
    return MessageIn(
        tenant_id="default",
        session_id=chat.chat_id,
        role=message_role(item),
        content=content,
        channel="feishu",
        platform="feishu",
        chat_type="group",
        account_id="default",
        from_id=sender_id,
        sender_id=sender_id,
        sender_name=None,
        group_id=chat.chat_id,
        group_subject=chat.name,
        native_channel_id=chat.chat_id,
        platform_message_id=str(item.get("message_id") or ""),
        origin_message_id=str(item.get("message_id") or ""),
        reply_to_id=extract_reply_to_id(item),
        ts=create_ts,
    )


def summarize_state(service: ClawDBService, fetched_messages: int, source_chats: int, deleted_seen: int) -> Dict[str, Any]:
    state = service.df_store.state
    raw = state.messages_df.copy()
    if not raw.empty:
        raw = raw[raw["projection_kind"].astype(str) == RAW_PROJECTION_KIND].copy()
    topics = state.topics_df.copy()
    capsules = state.capsules_df.copy()

    channel_count = (
        int(raw["native_channel_id"].fillna("").astype(str).replace("", pd.NA).dropna().nunique())
        if not raw.empty
        else 0
    )
    canonical_topic_count = 0
    split_topic_shard_count = 0
    topic_status_counts: Dict[str, int] = {}
    if not topics.empty:
        topics["topic_id"] = topics["topic_id"].fillna("").astype(str)
        topics["canonical_topic_id"] = topics["canonical_topic_id"].fillna(topics["topic_id"]).astype(str)
        topics["status"] = topics["status"].fillna("").astype(str)
        canonical_topic_count = int((topics["topic_id"] == topics["canonical_topic_id"]).sum())
        split_topic_shard_count = int(topics["topic_id"].str.contains("::shard:", regex=False).sum())
        topic_status_counts = {
            str(key): int(value)
            for key, value in topics["status"].value_counts(dropna=False).to_dict().items()
        }

    capsule_count = int(len(capsules))
    open_capsule_count = 0
    sealed_capsule_count = 0
    if not capsules.empty:
        capsules["capsule_state"] = capsules["capsule_state"].fillna("").astype(str)
        open_capsule_count = int((capsules["capsule_state"] == "open").sum())
        sealed_capsule_count = int((capsules["capsule_state"] == "sealed").sum())

    return {
        "source_chats": int(source_chats),
        "source_messages_fetched": int(fetched_messages),
        "source_deleted_messages_seen": int(deleted_seen),
        "raw_messages": int(len(raw)),
        "channels": channel_count,
        "auto_topics_total_rows": int(len(topics)),
        "auto_topics_canonical": canonical_topic_count,
        "auto_topic_split_shards": split_topic_shard_count,
        "capsule_shards": capsule_count,
        "capsule_shards_open": open_capsule_count,
        "capsule_shards_sealed": sealed_capsule_count,
        "topic_status_counts": topic_status_counts,
    }


async def _assign_and_materialize_raw_rows(
    *,
    service: ClawDBService,
    chat: ChatSpec,
    items: Sequence[Dict[str, Any]],
    progress: Dict[str, Any],
    imported_messages: int,
    topic_lookup: Dict[str, Dict[str, str]],
) -> tuple[List[Dict[str, Any]], int]:
    raw_rows: List[Dict[str, Any]] = []
    requests = [message_in(chat, item) for item in items if not bool(item.get("deleted"))]
    for start in range(0, len(requests), EMBED_BATCH_SIZE):
        batch = requests[start : start + EMBED_BATCH_SIZE]
        vectors = await service._embed_topic_texts([req.content for req in batch])
        for batch_idx, (req, topic_vector) in enumerate(zip(batch, vectors), start=1):
            normalized_channel = (req.channel or "").strip().lower() or None
            normalized_chat_type = (req.chat_type or "").strip().lower() or None
            normalized_platform = normalize_platform(req.platform, normalized_channel)
            hydrated_req = req
            reply_reference = str(req.reply_to_id or "").strip()
            if reply_reference and not req.topic_id:
                inherited_topic = topic_lookup.get(reply_reference)
                if inherited_topic:
                    hydrated_req = req.model_copy(
                        update={
                            "topic_id": str(inherited_topic.get("topic_id") or ""),
                            "topic_path": str(inherited_topic.get("topic_path") or ""),
                            "topic_source": "reply",
                            "topic_confidence": 1.0,
                        }
                    )
            topic_assignment = await service._resolve_topic_assignment(
                hydrated_req,
                normalized_channel=normalized_channel,
                normalized_platform=normalized_platform,
                normalized_chat_type=normalized_chat_type,
                topic_vector=topic_vector,
            )
            payload_input = req.model_dump(mode="json")
            payload_input.update(
                {
                    "channel": normalized_channel,
                    "platform": normalized_platform,
                    "chat_type": normalized_chat_type,
                    "topic_id": str(topic_assignment["topic_id"]),
                    "source_topic_id": str(topic_assignment["source_topic_id"]),
                    "topic_path": str(topic_assignment["topic_path"]),
                    "source_topic_path": str(topic_assignment["source_topic_path"]),
                    "topic_source": str(topic_assignment["topic_source"]),
                    "topic_confidence": float(topic_assignment["topic_confidence"]),
                }
            )
            bundle = materialize_message_bundle(payload_input)
            raw_row = dict(bundle["raw_message"])
            raw_rows.append(raw_row)
            imported_messages += 1
            if str(req.content or "").strip():
                scoped_model = topic_assignment["scoped_model"]
                topic_id = str(topic_assignment["topic_id"])
                if list(topic_assignment["topic_vector"]):
                    scoped_model.observe_vector(topic_id, topic_vector)
                else:
                    scoped_model.observe(topic_id, str(req.content))
            raw_origin_id = str(raw_row.get("origin_message_id") or "")
            raw_platform_id = str(raw_row.get("platform_message_id") or "")
            topic_record = {
                "topic_id": str(raw_row.get("source_topic_id") or raw_row.get("topic_id") or ""),
                "topic_path": str(raw_row.get("source_topic_path") or raw_row.get("topic_path") or ""),
            }
            if raw_origin_id:
                topic_lookup[raw_origin_id] = topic_record
            if raw_platform_id:
                topic_lookup[raw_platform_id] = topic_record
            progress["current_chat_messages_ingested"] = start + batch_idx
            progress["total_messages_ingested"] = imported_messages
        write_json(PROGRESS_PATH, progress)
    return raw_rows, imported_messages


async def build_store(temp_root: Path, progress: Dict[str, Any]) -> Dict[str, Any]:
    secret = load_secret()
    client = FeishuClient(APP_ID, secret)
    chats = client.list_chats()
    chat_limit_raw = os.environ.get("FEISHU_IMPORT_CHAT_LIMIT", "").strip()
    if chat_limit_raw:
        chats = chats[: max(0, int(chat_limit_raw))]
    progress["source_chat_count"] = len(chats)
    progress["phase"] = "fetch"
    write_json(PROGRESS_PATH, progress)

    service = ClawDBService()
    fetched_messages = 0
    imported_messages = 0
    deleted_seen = 0
    raw_rows: List[Dict[str, Any]] = []
    topic_lookup: Dict[str, Dict[str, str]] = {}

    for chat_idx, chat in enumerate(chats, start=1):
        progress["chat_index"] = chat_idx
        progress["current_chat"] = chat.chat_id
        progress["current_chat_name"] = chat.name
        progress["current_chat_pages"] = 0
        progress["current_chat_messages_fetched"] = 0
        progress["current_chat_messages_total"] = 0
        progress["current_chat_messages_ingested"] = 0
        write_json(PROGRESS_PATH, progress)

        items = client.fetch_messages(chat, progress)
        fetched_messages += len(items)
        deleted_in_chat = sum(1 for item in items if bool(item.get("deleted")))
        deleted_seen += deleted_in_chat
        progress["current_chat_messages_total"] = len(items)
        progress["total_messages_fetched"] = fetched_messages
        progress["phase"] = "ingest"
        write_json(PROGRESS_PATH, progress)

        chat_rows, imported_messages = await _assign_and_materialize_raw_rows(
            service=service,
            chat=chat,
            items=items,
            progress=progress,
            imported_messages=imported_messages,
            topic_lookup=topic_lookup,
        )
        raw_rows.extend(chat_rows)
        progress["phase"] = "fetch"
        write_json(PROGRESS_PATH, progress)

    progress["phase"] = "materialize"
    write_json(PROGRESS_PATH, progress)
    if raw_rows:
        service.df_store.state.messages_df = pd.DataFrame(raw_rows)
    rebuild = await service.df_store.rebuild_storage_from_authoritative_raw(
        vector_dim=service.config.topic_gep_dim
    )
    await service._rebuild_topic_state_from_store(rebuild_materialized_topics=False)
    await service.flush_now()
    await service.metadata.save_checkpoint(CURRENT_SCHEMA_VERSION, slot=SCHEMA_VERSION_SLOT)

    summary = summarize_state(
        service,
        fetched_messages=fetched_messages,
        source_chats=len(chats),
        deleted_seen=deleted_seen,
    )
    summary.update(
        {
            "temp_root": str(temp_root),
            "imported_messages": int(imported_messages),
            "semantic_jobs_processed": 0,
            "rebuild_topic_count": int(rebuild.topic_count),
            "rebuild_capsule_count": int(rebuild.capsule_count),
            "rebuild_session_rollup_count": int(rebuild.session_rollup_count),
            "rebuild_projection_message_count": int(rebuild.projection_message_count),
        }
    )
    return summary


def swap_live_root(temp_root: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    backup_root = LIVE_ROOT.with_name(f"{LIVE_ROOT.name}.pre-feishu-import-{timestamp}")
    subprocess.run(["systemctl", "--user", "stop", "clawdb.service"], check=True)
    try:
        if LIVE_ROOT.exists():
            LIVE_ROOT.rename(backup_root)
        temp_root.rename(LIVE_ROOT)
        subprocess.run(["systemctl", "--user", "start", "clawdb.service"], check=True)
    except Exception:
        if LIVE_ROOT.exists() and not any(LIVE_ROOT.iterdir()):
            shutil.rmtree(LIVE_ROOT, ignore_errors=True)
        if backup_root.exists() and not LIVE_ROOT.exists():
            backup_root.rename(LIVE_ROOT)
        subprocess.run(["systemctl", "--user", "start", "clawdb.service"], check=False)
        raise
    return backup_root


def post_sync() -> None:
    if SYNC_SCRIPT.exists():
        subprocess.run([str(SYNC_SCRIPT)], check=True)


def main() -> None:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    temp_root = OPENCLAW_ROOT / f"clawdb-data.rebuild-feishu-{timestamp}"
    progress: Dict[str, Any] = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "phase": "init",
        "temp_root": str(temp_root),
        "live_root": str(LIVE_ROOT),
        "source_chat_count": 0,
        "chat_index": 0,
        "current_chat": "",
        "current_chat_name": "",
        "current_chat_pages": 0,
        "current_chat_messages_fetched": 0,
        "current_chat_messages_total": 0,
        "current_chat_messages_ingested": 0,
        "total_messages_fetched": 0,
        "total_messages_ingested": 0,
    }
    write_json(PROGRESS_PATH, progress)
    temp_root.mkdir(parents=True, exist_ok=False)
    os.environ["CLAWDB_DATA_ROOT"] = str(temp_root)
    os.environ.setdefault("CLAWDB_OPENCLAW_REQUIRE_SIGNATURE", "false")
    os.environ.setdefault("CLAWDB_QUEUE_BACKEND", "memory")
    os.environ.setdefault("CLAWDB_INGEST_BACKPRESSURE_LAG_THRESHOLD", "1000000")
    os.environ.setdefault("CLAWDB_SEMANTIC_PIPELINE_MODE", "async")
    os.environ.setdefault("CLAWDB_TOPIC_EMBEDDING_PROVIDER", "openai")
    os.environ.setdefault("CLAWDB_TOPIC_EMBEDDING_BASE_URL", "http://127.0.0.1:11440/v1")
    os.environ.setdefault("CLAWDB_TOPIC_EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B")
    os.environ.setdefault("CLAWDB_TOPIC_EMBEDDING_API_KEY", "local-topic-embedder")
    summary = asyncio.run(build_store(temp_root, progress))
    skip_swap = os.environ.get("FEISHU_IMPORT_SKIP_SWAP", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if skip_swap:
        summary["backup_root"] = ""
    else:
        progress["phase"] = "swap"
        write_json(PROGRESS_PATH, progress)
        backup_root = swap_live_root(temp_root)
        progress["phase"] = "sync"
        write_json(PROGRESS_PATH, progress)
        post_sync()
        summary["backup_root"] = str(backup_root)
    summary["completed_at"] = datetime.now(timezone.utc).isoformat()
    target_summary = SUMMARY_DIR / f"clawdb-feishu-import-summary-{timestamp}.json"
    write_json(target_summary, summary)
    write_json(SUMMARY_PATH, summary)
    progress["phase"] = "done"
    progress["summary_path"] = str(target_summary)
    write_json(PROGRESS_PATH, progress)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
