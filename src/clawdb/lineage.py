from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Mapping, Optional

from .embeddings import deterministic_embedding_ref


RAW_PROJECTION_KIND = "raw_global"
PRIVATE_DM_PROJECTION_KIND = "private_dm"
GROUP_PUBLIC_PROJECTION_KIND = "group_public"
DM_MIRROR_PUBLIC_PROJECTION_KIND = "dm_mirror_public"

MESSAGE_STATE_ACTIVE = "active"
MESSAGE_STATE_EDITED = "edited"
MESSAGE_STATE_DELETED = "deleted"

PLATFORM_IDENTITY_COLUMNS = [
    "account_key",
    "from_user_key",
    "to_user_key",
    "sender_user_key",
    "group_chat_key",
]

_PLATFORM_ALIASES = {
    "lark": "feishu",
}


@dataclass(frozen=True)
class ProjectionSpec:
    kind: str
    scope: str
    session_id: str
    visibility: str
    native_session_id: str = ""


def normalize_platform(platform: Optional[str], channel: Optional[str] = None) -> str:
    raw = (platform or channel or "").strip().lower()
    if not raw:
        return "generic"
    return _PLATFORM_ALIASES.get(raw, raw)


def normalize_identity(platform: str, value: Optional[str], expected_kind: str) -> str:
    raw = (value or "").strip()
    if not raw:
        return ""
    canonical_prefix = f"{platform}_{expected_kind}:"
    if raw.startswith(canonical_prefix):
        return raw
    if platform == "feishu":
        if expected_kind == "user":
            if raw.startswith("feishu_chat:") or raw.startswith("oc_"):
                raise ValueError(f"Feishu user identity cannot use chat prefix: {raw}")
            return f"feishu_user:{raw}"
        if expected_kind == "chat":
            if raw.startswith("feishu_user:") or raw.startswith("ou_"):
                raise ValueError(f"Feishu chat identity cannot use user prefix: {raw}")
            return f"feishu_chat:{raw}"
        if expected_kind == "account":
            if raw.startswith("feishu_user:") or raw.startswith("feishu_chat:"):
                raise ValueError(f"Feishu account identity cannot use user/chat prefix: {raw}")
            return f"feishu_account:{raw}"
    return f"{platform}_{expected_kind}:{raw}"


def normalize_platform_identities(payload: Mapping[str, object]) -> Dict[str, str]:
    platform = normalize_platform(
        str(payload.get("platform") or "") or None,
        str(payload.get("channel") or "") or None,
    )
    account_key = normalize_identity(
        platform,
        str(payload.get("account_key") or payload.get("account_id") or ""),
        "account",
    ) or f"{platform}_account:_"
    return {
        "platform": platform,
        "account_key": account_key,
        "from_user_key": normalize_identity(
            platform,
            str(payload.get("from_user_key") or payload.get("from_id") or ""),
            "user",
        ),
        "to_user_key": normalize_identity(
            platform,
            str(payload.get("to_user_key") or payload.get("to_id") or ""),
            "user",
        ),
        "sender_user_key": normalize_identity(
            platform,
            str(payload.get("sender_user_key") or payload.get("sender_id") or ""),
            "user",
        ),
        "group_chat_key": normalize_identity(
            platform,
            str(payload.get("group_chat_key") or payload.get("group_id") or ""),
            "chat",
        ),
    }


def canonical_origin_message_id(payload: Mapping[str, object]) -> str:
    explicit = str(payload.get("origin_message_id") or "").strip()
    if explicit:
        return explicit
    identities = normalize_platform_identities(payload)
    platform = identities["platform"]
    account_key = str(identities["account_key"] or "").strip() or f"{platform}_account:_"
    platform_message_id = str(payload.get("platform_message_id") or "").strip()
    if platform_message_id:
        source = f"{platform}::{account_key}::{platform_message_id}"
        digest = hashlib.sha256(source.encode("utf-8")).hexdigest()[:32]
        return f"orig_{digest}"
    return str(payload.get("message_id") or "").strip()


def build_projection_specs(payload: Mapping[str, object]) -> List[ProjectionSpec]:
    identities = normalize_platform_identities(payload)
    platform = identities["platform"]
    account_key = str(identities["account_key"] or "").strip() or f"{platform}_account:_"
    chat_type = str(payload.get("chat_type") or "").strip().lower()
    incoming_session_id = str(payload.get("session_id") or payload.get("native_session_id") or "").strip()
    role = str(payload.get("role") or "").strip().lower()

    def _preferred_user_id() -> str:
        if role == "assistant":
            candidates = [
                payload.get("to_user_key"),
                payload.get("from_user_key"),
                payload.get("sender_user_key"),
                payload.get("to_id"),
                payload.get("from_id"),
                payload.get("sender_id"),
            ]
        else:
            candidates = [
                payload.get("sender_user_key"),
                payload.get("from_user_key"),
                payload.get("to_user_key"),
                payload.get("sender_id"),
                payload.get("from_id"),
                payload.get("to_id"),
            ]
        for candidate in candidates:
            normalized = normalize_identity(platform, str(candidate or ""), "user")
            if normalized:
                return normalized
        return ""

    specs: List[ProjectionSpec] = []
    if chat_type == "group":
        group_key = str(identities["group_chat_key"] or "").strip()
        if not group_key:
            fallback_group = str(payload.get("native_channel_id") or incoming_session_id or "_").strip()
            group_key = normalize_identity(platform, fallback_group, "chat")
        group_scope = f"group:{account_key}:{group_key}"
        specs.append(
            ProjectionSpec(
                kind=GROUP_PUBLIC_PROJECTION_KIND,
                scope=group_scope,
                session_id=group_scope,
                visibility="public",
                native_session_id=incoming_session_id,
            )
        )
        actor_key = _preferred_user_id()
        if actor_key:
            dm_scope = f"dm:{account_key}:{actor_key}"
            specs.append(
                ProjectionSpec(
                    kind=DM_MIRROR_PUBLIC_PROJECTION_KIND,
                    scope=dm_scope,
                    session_id=dm_scope,
                    visibility="public",
                )
            )
    else:
        user_key = _preferred_user_id()
        if not user_key:
            fallback_user = incoming_session_id or str(payload.get("to_id") or payload.get("from_id") or "_")
            user_key = normalize_identity(platform, fallback_user, "user")
        dm_scope = f"dm:{account_key}:{user_key}"
        specs.append(
            ProjectionSpec(
                kind=PRIVATE_DM_PROJECTION_KIND,
                scope=dm_scope,
                session_id=dm_scope,
                visibility="private",
                native_session_id=incoming_session_id,
            )
        )
    deduped: Dict[tuple[str, str], ProjectionSpec] = {}
    for spec in specs:
        deduped[(spec.kind, spec.scope)] = spec
    return list(deduped.values())


def projection_message_id(origin_message_id: str, kind: str, scope: str) -> str:
    digest = hashlib.sha256(f"{kind}::{scope}".encode("utf-8")).hexdigest()[:16]
    return f"{origin_message_id}::proj::{digest}"


def materialize_projection_rows(raw_message: Mapping[str, object]) -> List[Dict[str, object]]:
    source = dict(raw_message)
    origin_message_id = str(source.get("origin_message_id") or source.get("message_id") or "").strip()
    source["origin_message_id"] = origin_message_id
    identities = normalize_platform_identities(source)
    source["platform"] = identities["platform"]
    for column in PLATFORM_IDENTITY_COLUMNS:
        source[column] = identities[column]
    source["native_session_id"] = str(source.get("native_session_id") or "")
    projections: List[Dict[str, object]] = []
    for spec in build_projection_specs(source):
        projections.append(
            {
                **source,
                "message_id": projection_message_id(origin_message_id, spec.kind, spec.scope),
                "session_id": spec.session_id,
                "capsule_level": "L1",
                "projection_kind": spec.kind,
                "projection_scope": spec.scope,
                "visibility": spec.visibility,
                "native_session_id": spec.native_session_id,
            }
        )
    return projections


def materialize_message_bundle(payload: Mapping[str, object]) -> Dict[str, object]:
    identities = normalize_platform_identities(payload)
    platform = identities["platform"]
    origin_message_id = canonical_origin_message_id(payload)
    ts_value = payload.get("ts")
    if isinstance(ts_value, datetime):
        ts = ts_value
    elif ts_value:
        ts = datetime.fromisoformat(str(ts_value).replace("Z", "+00:00"))
    else:
        ts = datetime.now(timezone.utc)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    ts_iso = ts.isoformat()
    content = str(payload.get("content") or "")
    base = {
        "origin_message_id": origin_message_id,
        "tenant_id": str(payload.get("tenant_id") or "default"),
        "role": str(payload.get("role") or "user"),
        "content": content,
        "ts": ts_iso,
        "channel": str(payload.get("channel") or ""),
        "chat_type": str(payload.get("chat_type") or ""),
        "account_id": str(payload.get("account_id") or ""),
        "from_id": str(payload.get("from_id") or ""),
        "to_id": str(payload.get("to_id") or ""),
        "sender_id": str(payload.get("sender_id") or ""),
        "sender_name": str(payload.get("sender_name") or ""),
        "sender_username": str(payload.get("sender_username") or ""),
        "sender_e164": str(payload.get("sender_e164") or ""),
        "group_id": str(payload.get("group_id") or ""),
        "group_subject": str(payload.get("group_subject") or ""),
        "group_channel": str(payload.get("group_channel") or ""),
        "group_space": str(payload.get("group_space") or ""),
        "native_channel_id": str(payload.get("native_channel_id") or ""),
        "message_thread_id": str(payload.get("message_thread_id") or ""),
        "thread_parent_id": str(payload.get("thread_parent_id") or ""),
        "reply_to_id": str(payload.get("reply_to_id") or ""),
        "topic_id": str(payload.get("topic_id") or "default"),
        "source_topic_id": str(payload.get("source_topic_id") or payload.get("topic_id") or "default"),
        "topic_parent_id": str(payload.get("topic_parent_id") or ""),
        "topic_path": str(payload.get("topic_path") or payload.get("topic_id") or "default"),
        "source_topic_path": str(
            payload.get("source_topic_path")
            or payload.get("topic_path")
            or payload.get("topic_id")
            or "default"
        ),
        "topic_confidence": payload.get("topic_confidence"),
        "topic_source": str(payload.get("topic_source") or ""),
        "embedding_ref": str(payload.get("embedding_ref") or deterministic_embedding_ref("raw_message", content)),
        "capsule_level": "L0",
        "idempotency_key": str(payload.get("idempotency_key") or ""),
        "visibility": "raw",
        "platform": platform,
        "platform_message_id": str(payload.get("platform_message_id") or ""),
        "native_session_id": "",
        "message_state": MESSAGE_STATE_ACTIVE,
        "updated_at": ts_iso,
        "deleted_at": None,
        **{column: identities[column] for column in PLATFORM_IDENTITY_COLUMNS},
    }
    raw_row = {
        **base,
        "message_id": origin_message_id,
        "session_id": "",
        "projection_kind": RAW_PROJECTION_KIND,
        "projection_scope": "global",
        "native_session_id": str(payload.get("session_id") or ""),
    }
    projections = materialize_projection_rows(raw_row)
    return {
        "tenant_id": str(payload.get("tenant_id") or "default"),
        "session_id": str(payload.get("session_id") or ""),
        "request_message_id": str(payload.get("message_id") or origin_message_id),
        "origin_message_id": origin_message_id,
        "idempotency_key": str(payload.get("idempotency_key") or ""),
        "raw_message": raw_row,
        "projections": projections,
    }
