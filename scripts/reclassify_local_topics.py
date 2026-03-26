#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from clawdb.dataframes import MESSAGES_COLUMNS, DataFrameStore, authoritative_raw_messages
from clawdb.lineage import MESSAGE_STATE_DELETED, RAW_PROJECTION_KIND, normalize_platform
from clawdb.migrate import CURRENT_SCHEMA_VERSION, SCHEMA_VERSION_SLOT
from clawdb.models import MessageIn
from clawdb.service import ClawDBService


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def _topic_lookup_record(row: Dict[str, object]) -> Dict[str, str]:
    return {
        "topic_id": str(row.get("source_topic_id") or row.get("topic_id") or ""),
        "topic_path": str(row.get("source_topic_path") or row.get("topic_path") or ""),
    }


def _summarize(service: ClawDBService, temp_root: Path) -> Dict[str, object]:
    state = service.df_store.state
    messages = state.messages_df.copy()
    raw = messages[messages["projection_kind"].astype(str) == RAW_PROJECTION_KIND].copy()
    topics = state.topics_df.copy()
    groups = (
        int(raw["group_id"].fillna("").astype(str).replace("", pd.NA).dropna().nunique())
        if not raw.empty
        else 0
    )
    active_topics = 0
    active_topics_by_scope: Dict[str, int] = {}
    if not topics.empty:
        topics["status"] = topics["status"].fillna("").astype(str)
        topics["canonical_topic_id"] = topics["canonical_topic_id"].fillna(topics["topic_id"]).astype(str)
        topics["topic_path"] = topics["topic_path"].fillna("").astype(str)
        active = topics[topics["status"] != "compacted"].copy()
        active_topics = int(active["canonical_topic_id"].nunique())
        if not active.empty:
            active["_scope"] = active["topic_path"].astype(str).map(
                lambda value: value.split("/")[2] if len(value.split("/")) >= 3 else ""
            )
            active_topics_by_scope = {
                str(key): int(value)
                for key, value in (
                    active.groupby("_scope")["canonical_topic_id"]
                    .nunique()
                    .sort_values(ascending=False)
                    .head(20)
                    .to_dict()
                    .items()
                )
            }
    return {
        "temp_root": str(temp_root),
        "raw_messages": int(len(raw)),
        "groups": groups,
        "active_topics": active_topics,
        "topic_gep_min_dot_product": service.config.topic_gep_min_dot_product,
        "active_topics_by_scope_top20": active_topics_by_scope,
    }


async def build_temp_store(
    *,
    live_root: Path,
    temp_root: Path,
    batch_size: int,
) -> Dict[str, object]:
    old_store = DataFrameStore()
    await old_store.load_parquet(live_root / "parquet")
    raw = authoritative_raw_messages(old_store.state.messages_df)
    raw = raw.sort_values(["ts", "updated_at", "origin_message_id", "message_id"], kind="stable").reset_index(
        drop=True
    )

    for key, value in {
        "CLAWDB_DATA_ROOT": str(temp_root),
        "CLAWDB_OPENCLAW_REQUIRE_SIGNATURE": "false",
        "CLAWDB_QUEUE_BACKEND": "memory",
        "CLAWDB_INGEST_BACKPRESSURE_LAG_THRESHOLD": "1000000",
        "CLAWDB_SEMANTIC_PIPELINE_MODE": "async",
        "CLAWDB_TOPIC_EMBEDDING_PROVIDER": "openai",
        "CLAWDB_TOPIC_EMBEDDING_BASE_URL": "http://127.0.0.1:11440/v1",
        "CLAWDB_TOPIC_EMBEDDING_MODEL": "Qwen/Qwen3-Embedding-0.6B",
        "CLAWDB_TOPIC_EMBEDDING_API_KEY": "local-topic-embedder",
    }.items():
        os.environ[key] = value

    service = ClawDBService()
    rebuilt_rows = []
    topic_lookup: Dict[str, Dict[str, str]] = {}

    total = int(len(raw))
    for start in range(0, total, batch_size):
        batch = raw.iloc[start : start + batch_size].copy().reset_index(drop=True)
        vectors = await service._embed_topic_texts(batch["content"].fillna("").astype(str).tolist())
        for idx, (_, row) in enumerate(batch.iterrows()):
            row_dict = row.to_dict()
            message_state = str(row.get("message_state") or "active")
            normalized_channel = str(row.get("channel") or "").strip().lower() or None
            normalized_chat_type = str(row.get("chat_type") or "").strip().lower() or None
            normalized_platform = normalize_platform(str(row.get("platform") or "") or None, normalized_channel)
            session_id = str(row.get("native_session_id") or row.get("session_id") or "")
            req = MessageIn(
                tenant_id=str(row.get("tenant_id") or "default"),
                session_id=session_id,
                role=str(row.get("role") or "user"),
                content=str(row.get("content") or ""),
                channel=normalized_channel,
                platform=normalized_platform,
                chat_type=normalized_chat_type or None,
                account_id=str(row.get("account_id") or "") or None,
                account_key=str(row.get("account_key") or "") or None,
                from_id=str(row.get("from_id") or "") or None,
                from_user_key=str(row.get("from_user_key") or "") or None,
                to_id=str(row.get("to_id") or "") or None,
                to_user_key=str(row.get("to_user_key") or "") or None,
                sender_id=str(row.get("sender_id") or "") or None,
                sender_user_key=str(row.get("sender_user_key") or "") or None,
                sender_name=str(row.get("sender_name") or "") or None,
                sender_username=str(row.get("sender_username") or "") or None,
                sender_e164=str(row.get("sender_e164") or "") or None,
                group_id=str(row.get("group_id") or "") or None,
                group_chat_key=str(row.get("group_chat_key") or "") or None,
                group_subject=str(row.get("group_subject") or "") or None,
                group_channel=str(row.get("group_channel") or "") or None,
                group_space=str(row.get("group_space") or "") or None,
                native_channel_id=str(row.get("native_channel_id") or "") or None,
                platform_message_id=str(row.get("platform_message_id") or "") or None,
                origin_message_id=str(row.get("origin_message_id") or row.get("message_id") or ""),
                projection_target_user_key=str(row.get("projection_target_user_key") or "") or None,
                message_thread_id=str(row.get("message_thread_id") or "") or None,
                thread_parent_id=str(row.get("thread_parent_id") or "") or None,
                reply_to_id=str(row.get("reply_to_id") or "") or None,
                capsule_level=str(row.get("capsule_level") or "L0"),
                idempotency_key=str(row.get("idempotency_key") or "") or None,
                message_id=str(row.get("message_id") or row.get("origin_message_id") or ""),
                ts=pd.to_datetime(row.get("ts"), utc=True, errors="coerce").to_pydatetime(),
            )
            hydrated_req = req
            reply_reference = str(req.reply_to_id or "").strip()
            if reply_reference:
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

            if message_state != MESSAGE_STATE_DELETED:
                topic_assignment = await service._resolve_topic_assignment(
                    hydrated_req,
                    normalized_channel=normalized_channel,
                    normalized_platform=normalized_platform,
                    normalized_chat_type=normalized_chat_type,
                    topic_vector=vectors[idx],
                )
                row_dict["channel"] = normalized_channel or ""
                row_dict["platform"] = normalized_platform or ""
                row_dict["chat_type"] = normalized_chat_type or ""
                row_dict["topic_id"] = str(topic_assignment["topic_id"])
                row_dict["source_topic_id"] = str(topic_assignment["source_topic_id"])
                row_dict["topic_path"] = str(topic_assignment["topic_path"])
                row_dict["source_topic_path"] = str(topic_assignment["source_topic_path"])
                row_dict["topic_source"] = str(topic_assignment["topic_source"])
                row_dict["topic_confidence"] = float(topic_assignment["topic_confidence"])
                if str(req.content or "").strip():
                    scoped_model = topic_assignment["scoped_model"]
                    topic_id = str(topic_assignment["topic_id"])
                    topic_vector = list(topic_assignment["topic_vector"])
                    if topic_vector:
                        scoped_model.observe_vector(topic_id, topic_vector)
                    else:
                        scoped_model.observe(topic_id, str(req.content))
            else:
                row_dict["source_topic_id"] = str(row.get("source_topic_id") or row.get("topic_id") or "default")
                row_dict["source_topic_path"] = str(
                    row.get("source_topic_path") or row.get("topic_path") or row_dict["source_topic_id"]
                )

            topic_record = _topic_lookup_record(row_dict)
            origin_id = str(row_dict.get("origin_message_id") or row_dict.get("message_id") or "")
            platform_message_id = str(row_dict.get("platform_message_id") or "")
            if origin_id:
                topic_lookup[origin_id] = topic_record
            if platform_message_id:
                topic_lookup[platform_message_id] = topic_record
            rebuilt_rows.append(row_dict)

        if (start // batch_size) % 50 == 0:
            print(json.dumps({"progress": min(total, start + len(batch)), "total": total}), flush=True)

    service.df_store.state.messages_df = pd.DataFrame(rebuilt_rows, columns=MESSAGES_COLUMNS)
    service.df_store.state.sessions_df = old_store.state.sessions_df.copy()
    rebuild = await service.df_store.rebuild_storage_from_authoritative_raw(vector_dim=service.config.topic_gep_dim)
    await service._rebuild_topic_state_from_store(rebuild_materialized_topics=False)
    await service.flush_now()
    await service.metadata.save_checkpoint(CURRENT_SCHEMA_VERSION, slot=SCHEMA_VERSION_SLOT)
    summary = _summarize(service, temp_root)
    summary.update(
        {
            "rebuild_topic_count": int(rebuild.topic_count),
            "rebuild_capsule_count": int(rebuild.capsule_count),
            "rebuild_session_rollup_count": int(rebuild.session_rollup_count),
            "rebuild_projection_message_count": int(rebuild.projection_message_count),
        }
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reclassify local ClawDB topics from authoritative raw messages")
    parser.add_argument(
        "--live-root",
        default=str(Path("~/.openclaw/clawdb-data").expanduser()),
        help="Existing live ClawDB data root",
    )
    parser.add_argument(
        "--temp-root",
        default="",
        help="Optional target temp root. Defaults to a timestamped sibling of the live root.",
    )
    parser.add_argument(
        "--summary-path",
        default="",
        help="Optional summary json path. Defaults to a timestamped file beside the live root.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    live_root = Path(args.live_root).expanduser().resolve()
    if not live_root.exists():
        raise SystemExit(f"live root does not exist: {live_root}")
    timestamp = _timestamp()
    temp_root = (
        Path(args.temp_root).expanduser().resolve()
        if str(args.temp_root).strip()
        else live_root.with_name(f"{live_root.name}.reclassify-topics-{timestamp}")
    )
    summary_path = (
        Path(args.summary_path).expanduser().resolve()
        if str(args.summary_path).strip()
        else live_root.parent / f"clawdb-topic-reclassify-summary-{timestamp}.json"
    )
    summary = asyncio.run(build_temp_store(live_root=live_root, temp_root=temp_root, batch_size=max(1, args.batch_size)))
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
