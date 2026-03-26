#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import httpx
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from clawdb.dataframes import DataFrameStore
from clawdb.lineage import RAW_PROJECTION_KIND
from clawdb.models import MessageIn


APP_ID = os.getenv("FEISHU_APP_ID", "cli_a93d537fe638dcb3")
SECRETS_PATH = Path(os.getenv("FEISHU_SECRETS_PATH", "~/.openclaw/secrets.store.json")).expanduser()
CLAWDB_BASE_URL = os.getenv("CLAWDB_BASE_URL", "http://127.0.0.1:8080").rstrip("/")
CLAWDB_LIVE_ROOT = Path(os.getenv("CLAWDB_LIVE_ROOT", "~/.openclaw/clawdb-data")).expanduser()
STATE_PATH = Path(
    os.getenv("FEISHU_LIVE_STATE_PATH", "~/.openclaw/feishu/clawdb-live-state.json")
).expanduser()
REQUEST_TIMEOUT = float(os.getenv("FEISHU_LIVE_TIMEOUT_SECONDS", "30"))
CLAWDB_REQUEST_TIMEOUT = float(os.getenv("CLAWDB_LIVE_TIMEOUT_SECONDS", "180"))
POLL_SECONDS = max(5.0, float(os.getenv("FEISHU_LIVE_POLL_SECONDS", "30")))
CHAT_REFRESH_SECONDS = max(60.0, float(os.getenv("FEISHU_LIVE_CHAT_REFRESH_SECONDS", "300")))
PAGE_SIZE = max(1, min(50, int(os.getenv("FEISHU_LIVE_PAGE_SIZE", "50"))))
PRESENCE_CHECK_WAIT_SECONDS = max(0.0, float(os.getenv("FEISHU_LIVE_PRESENCE_CHECK_WAIT_SECONDS", "12")))


@dataclass(frozen=True)
class ChatSpec:
    chat_id: str
    name: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f"{path.suffix}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def load_secret() -> str:
    payload = json.loads(SECRETS_PATH.read_text(encoding="utf-8"))
    secret = str(payload.get("FEISHU_APP_SECRET") or "").strip()
    if not secret:
        raise RuntimeError("FEISHU_APP_SECRET missing")
    return secret


def log_event(event: str, **payload: object) -> None:
    print(json.dumps({"event": event, **payload}, ensure_ascii=False), flush=True)


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
        idempotency_key=f"feishu-live:{str(item.get('message_id') or '').strip()}",
        ts=create_ts,
    )


def _message_sort_key(item: Dict[str, Any]) -> Tuple[int, str]:
    return (int(item.get("create_time") or 0), str(item.get("message_id") or ""))


def _state_for_messages(items: Sequence[Dict[str, Any]]) -> Dict[str, object]:
    if not items:
        return {"last_create_time_ms": 0, "last_message_ids": []}
    ordered = sorted(items, key=_message_sort_key)
    last_ms = int(ordered[-1].get("create_time") or 0)
    message_ids = sorted(
        {
            str(item.get("message_id") or "")
            for item in ordered
            if int(item.get("create_time") or 0) == last_ms and str(item.get("message_id") or "")
        }
    )
    return {"last_create_time_ms": last_ms, "last_message_ids": message_ids}


def merge_state_entry(previous: Dict[str, object], items: Sequence[Dict[str, Any]]) -> Dict[str, object]:
    if not items:
        return {
            "last_create_time_ms": int(previous.get("last_create_time_ms") or 0),
            "last_message_ids": sorted(
                {str(item) for item in previous.get("last_message_ids") or [] if str(item)}
            ),
        }
    updated = _state_for_messages(items)
    previous_ms = int(previous.get("last_create_time_ms") or 0)
    updated_ms = int(updated.get("last_create_time_ms") or 0)
    if updated_ms == previous_ms:
        updated["last_message_ids"] = sorted(
            {
                *{str(item) for item in previous.get("last_message_ids") or [] if str(item)},
                *{str(item) for item in updated.get("last_message_ids") or [] if str(item)},
            }
        )
    return updated


def derive_initial_state() -> Dict[str, Any]:
    if not CLAWDB_LIVE_ROOT.exists():
        return {"version": 1, "updated_at": utc_now(), "chats": {}}

    async def _load() -> Dict[str, Any]:
        store = DataFrameStore()
        await store.load_parquet(CLAWDB_LIVE_ROOT / "parquet")
        df = store.state.messages_df.copy()
        if df.empty:
            return {"version": 1, "updated_at": utc_now(), "chats": {}}
        raw = df[df["projection_kind"].astype(str) == RAW_PROJECTION_KIND].copy()
        if raw.empty:
            return {"version": 1, "updated_at": utc_now(), "chats": {}}
        raw["platform"] = raw["platform"].fillna("").astype(str)
        raw["group_id"] = raw["group_id"].fillna("").astype(str)
        raw["platform_message_id"] = raw["platform_message_id"].fillna("").astype(str)
        raw["ts"] = pd.to_datetime(raw["ts"], utc=True, errors="coerce")
        raw = raw[(raw["platform"] == "feishu") & (raw["group_id"] != "") & raw["ts"].notna()].copy()
        chats: Dict[str, Dict[str, object]] = {}
        for chat_id, group in raw.groupby("group_id", sort=True):
            latest_ts = group["ts"].max()
            latest_ms = int(latest_ts.timestamp() * 1000) if pd.notna(latest_ts) else 0
            latest_ids = sorted(
                {
                    str(item)
                    for item in group[group["ts"] == latest_ts]["platform_message_id"].astype(str).tolist()
                    if str(item)
                }
            )
            chats[str(chat_id)] = {
                "last_create_time_ms": latest_ms,
                "last_message_ids": latest_ids,
            }
        return {"version": 1, "updated_at": utc_now(), "chats": chats}

    return asyncio.run(_load())


def merge_bootstrap_state(existing: Dict[str, Any], derived: Dict[str, Any]) -> Dict[str, Any]:
    merged = {
        "version": 1,
        "updated_at": utc_now(),
        "chats": {},
    }
    existing_chats = existing.get("chats", {}) if isinstance(existing, dict) else {}
    derived_chats = derived.get("chats", {}) if isinstance(derived, dict) else {}
    for chat_id in sorted({*existing_chats.keys(), *derived_chats.keys()}):
        existing_entry = existing_chats.get(chat_id, {})
        derived_entry = derived_chats.get(chat_id, {})
        existing_ms = int(existing_entry.get("last_create_time_ms") or 0)
        derived_ms = int(derived_entry.get("last_create_time_ms") or 0)
        if derived_ms > existing_ms:
            merged["chats"][chat_id] = {
                "last_create_time_ms": derived_ms,
                "last_message_ids": sorted(
                    {str(item) for item in derived_entry.get("last_message_ids") or [] if str(item)}
                ),
            }
            continue
        if existing_ms > derived_ms:
            merged["chats"][chat_id] = {
                "last_create_time_ms": existing_ms,
                "last_message_ids": sorted(
                    {str(item) for item in existing_entry.get("last_message_ids") or [] if str(item)}
                ),
            }
            continue
        merged["chats"][chat_id] = {
            "last_create_time_ms": existing_ms,
            "last_message_ids": sorted(
                {
                    *{str(item) for item in existing_entry.get("last_message_ids") or [] if str(item)},
                    *{str(item) for item in derived_entry.get("last_message_ids") or [] if str(item)},
                }
            ),
        }
    return merged


def load_state() -> Dict[str, Any]:
    existing: Dict[str, Any] = {}
    if STATE_PATH.exists():
        try:
            payload = json.loads(STATE_PATH.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                existing = payload
        except Exception:
            pass
    state = merge_bootstrap_state(existing, derive_initial_state())
    write_json(STATE_PATH, state)
    return state


class FeishuClient:
    def __init__(self, app_id: str, app_secret: str) -> None:
        self.app_id = app_id
        self.app_secret = app_secret
        self.client = httpx.Client(timeout=REQUEST_TIMEOUT)
        self._token = ""
        self._token_deadline = 0.0

    def _refresh_token(self) -> None:
        response = self.client.post(
            "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal",
            json={"app_id": self.app_id, "app_secret": self.app_secret},
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("code") != 0:
            raise RuntimeError(f"Feishu auth failed: {payload}")
        self._token = str(payload["tenant_access_token"])
        expires_in = max(60, int(payload.get("expire") or 3600))
        self._token_deadline = time.time() + max(30, expires_in - 60)

    def _headers(self) -> Dict[str, str]:
        if not self._token or time.time() >= self._token_deadline:
            self._refresh_token()
        return {"Authorization": f"Bearer {self._token}"}

    def _get(self, path: str, *, params: Dict[str, object]) -> Dict[str, Any]:
        response = self.client.get(
            f"https://open.feishu.cn{path}",
            headers=self._headers(),
            params=params,
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("code") != 0:
            raise RuntimeError(f"Feishu GET {path} failed: {payload}")
        return payload

    def list_chats(self) -> List[ChatSpec]:
        chats: List[ChatSpec] = []
        page_token: Optional[str] = None
        while True:
            params: Dict[str, object] = {"page_size": 100}
            if page_token:
                params["page_token"] = page_token
            payload = self._get("/open-apis/im/v1/chats", params=params)
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

    def fetch_new_messages(self, chat: ChatSpec, state_entry: Dict[str, object]) -> List[Dict[str, Any]]:
        last_ms = int(state_entry.get("last_create_time_ms") or 0)
        last_ids = {str(item) for item in state_entry.get("last_message_ids") or [] if str(item)}
        out: List[Dict[str, Any]] = []
        page_token: Optional[str] = None
        stop = False
        while not stop:
            params: Dict[str, object] = {
                "container_id_type": "chat",
                "container_id": chat.chat_id,
                "page_size": PAGE_SIZE,
                "sort_type": "ByCreateTimeDesc",
            }
            if page_token:
                params["page_token"] = page_token
            payload = self._get("/open-apis/im/v1/messages", params=params)
            data = payload.get("data", {})
            items = data.get("items", [])
            if not items:
                break
            for item in items:
                create_ms = int(item.get("create_time") or 0)
                message_id = str(item.get("message_id") or "")
                if create_ms < last_ms:
                    stop = True
                    break
                if create_ms == last_ms and message_id in last_ids:
                    continue
                out.append(item)
            if stop or not data.get("has_more"):
                break
            page_token = str(data.get("page_token") or "").strip()
            if not page_token:
                break
        out.sort(key=_message_sort_key)
        deduped: Dict[str, Dict[str, Any]] = {}
        for item in out:
            message_id = str(item.get("message_id") or "").strip()
            if message_id:
                deduped[message_id] = item
        return list(sorted(deduped.values(), key=_message_sort_key))


class ClawDBClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(timeout=CLAWDB_REQUEST_TIMEOUT)

    def ingest_message(self, req: MessageIn) -> Dict[str, Any]:
        response = self.client.post(
            f"{self.base_url}/v1/memory/messages",
            json=req.model_dump(mode="json"),
        )
        response.raise_for_status()
        return response.json()

    def delete_message(self, *, message_id: str) -> Dict[str, Any]:
        response = self.client.post(
            f"{self.base_url}/v1/memory/messages/delete",
            json={
                "tenant_id": "default",
                "platform": "feishu",
                "account_id": "default",
                "platform_message_id": str(message_id),
                "ts": utc_now(),
            },
        )
        response.raise_for_status()
        return response.json()


def message_exists(message_id: str) -> bool:
    time.sleep(PRESENCE_CHECK_WAIT_SECONDS)

    async def _check() -> bool:
        store = DataFrameStore()
        await store.load_parquet(CLAWDB_LIVE_ROOT / "parquet")
        df = store.state.messages_df.copy()
        if df.empty:
            return False
        raw = df[df["projection_kind"].astype(str) == RAW_PROJECTION_KIND].copy()
        if raw.empty:
            return False
        raw["origin_message_id"] = raw["origin_message_id"].fillna(raw["message_id"]).astype(str)
        raw["platform_message_id"] = raw["platform_message_id"].fillna("").astype(str)
        target = str(message_id or "").strip()
        if not target:
            return False
        return bool(
            (
                (raw["origin_message_id"].astype(str) == target)
                | (raw["platform_message_id"].astype(str) == target)
            ).any()
        )

    return asyncio.run(_check())


def sync_once(
    *,
    feishu: FeishuClient,
    clawdb: ClawDBClient,
    state: Dict[str, Any],
    chats: Sequence[ChatSpec],
) -> Dict[str, int]:
    stats = {"chats": 0, "messages_seen": 0, "messages_ingested": 0, "messages_deleted": 0}
    chat_state = state.setdefault("chats", {})
    for chat in chats:
        stats["chats"] += 1
        entry = chat_state.setdefault(chat.chat_id, {"last_create_time_ms": 0, "last_message_ids": []})
        new_items = feishu.fetch_new_messages(chat, entry)
        if not new_items:
            continue
        stats["messages_seen"] += len(new_items)
        applied_items: List[Dict[str, Any]] = []
        for item in new_items:
            message_id = str(item.get("message_id") or "").strip()
            if not message_id:
                continue
            if bool(item.get("deleted")):
                try:
                    clawdb.delete_message(message_id=message_id)
                    stats["messages_deleted"] += 1
                    applied_items.append(item)
                    chat_state[chat.chat_id] = merge_state_entry(entry, applied_items)
                    state["updated_at"] = utc_now()
                    write_json(STATE_PATH, state)
                except Exception as exc:
                    log_event("feishu.live.delete_failed", chat_id=chat.chat_id, message_id=message_id, error=str(exc))
                    break
                continue
            req = message_in(chat, item)
            try:
                clawdb.ingest_message(req)
                stats["messages_ingested"] += 1
                applied_items.append(item)
                chat_state[chat.chat_id] = merge_state_entry(entry, applied_items)
                state["updated_at"] = utc_now()
                write_json(STATE_PATH, state)
            except Exception as exc:
                if message_exists(message_id):
                    log_event(
                        "feishu.live.ingest_confirmed_after_error",
                        chat_id=chat.chat_id,
                        message_id=message_id,
                        error=str(exc),
                    )
                    stats["messages_ingested"] += 1
                    applied_items.append(item)
                    chat_state[chat.chat_id] = merge_state_entry(entry, applied_items)
                    state["updated_at"] = utc_now()
                    write_json(STATE_PATH, state)
                    continue
                log_event("feishu.live.ingest_failed", chat_id=chat.chat_id, message_id=message_id, error=str(exc))
                break
        chat_state[chat.chat_id] = merge_state_entry(entry, applied_items)
    state["updated_at"] = utc_now()
    write_json(STATE_PATH, state)
    return stats


def main() -> None:
    secret = load_secret()
    feishu = FeishuClient(APP_ID, secret)
    clawdb = ClawDBClient(CLAWDB_BASE_URL)
    state = load_state()
    chats: List[ChatSpec] = []
    next_chat_refresh = 0.0
    while True:
        try:
            now = time.time()
            if now >= next_chat_refresh or not chats:
                chats = feishu.list_chats()
                next_chat_refresh = now + CHAT_REFRESH_SECONDS
                log_event("feishu.live.chat_refresh", chat_count=len(chats))
            stats = sync_once(feishu=feishu, clawdb=clawdb, state=state, chats=chats)
            log_event("feishu.live.sync", **stats, state_path=str(STATE_PATH))
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            log_event("feishu.live.error", error=str(exc))
            time.sleep(min(60.0, POLL_SECONDS))
            continue
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
