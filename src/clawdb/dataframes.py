from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .models import SearchResult, WalRecord


MESSAGES_COLUMNS = [
    "message_id",
    "tenant_id",
    "session_id",
    "role",
    "content",
    "ts",
    "topic_id",
    "embedding_ref",
    "capsule_level",
    "idempotency_key",
]

CAPSULES_COLUMNS = [
    "capsule_id",
    "tenant_id",
    "session_id",
    "topic_id",
    "summary",
    "level",
    "score",
    "updated_at",
]

CACHE_INDEX_COLUMNS = [
    "key",
    "tenant_id",
    "session_id",
    "query_type",
    "capsule_level",
    "entity_type",
    "entity_id",
    "last_access",
    "hit_count",
    "miss_count",
]


@dataclass
class DataFramesState:
    messages_df: pd.DataFrame
    capsules_df: pd.DataFrame
    cache_index_df: pd.DataFrame


class DataFrameStore:
    def __init__(self) -> None:
        self._state = DataFramesState(
            messages_df=pd.DataFrame(columns=MESSAGES_COLUMNS),
            capsules_df=pd.DataFrame(columns=CAPSULES_COLUMNS),
            cache_index_df=pd.DataFrame(columns=CACHE_INDEX_COLUMNS),
        )
        self._state.messages_df = self._state.messages_df.astype(
            {
                "message_id": "string",
                "tenant_id": "string",
                "session_id": "string",
                "role": "string",
                "content": "string",
                "topic_id": "string",
                "embedding_ref": "string",
                "capsule_level": "string",
                "idempotency_key": "string",
            }
        )
        self._lock = asyncio.Lock()

    @property
    def state(self) -> DataFramesState:
        return self._state

    async def add_message(self, payload: Dict[str, object]) -> None:
        async with self._lock:
            row = {
                "message_id": str(payload["message_id"]),
                "tenant_id": str(payload.get("tenant_id") or "default"),
                "session_id": str(payload["session_id"]),
                "role": str(payload["role"]),
                "content": str(payload["content"]),
                "ts": pd.to_datetime(payload["ts"], utc=True),
                "topic_id": str(payload.get("topic_id") or "default"),
                "embedding_ref": str(payload.get("embedding_ref") or ""),
                "capsule_level": str(payload.get("capsule_level") or "L0"),
                "idempotency_key": str(payload.get("idempotency_key") or ""),
            }
            row_df = pd.DataFrame([row], columns=MESSAGES_COLUMNS)
            if self._state.messages_df.empty:
                self._state.messages_df = row_df
            else:
                self._state.messages_df = pd.concat(
                    [self._state.messages_df, row_df],
                    ignore_index=True,
                )

    async def refresh_capsules(self, tenant_id: str, session_id: str) -> int:
        async with self._lock:
            subset = self._state.messages_df[
                (self._state.messages_df["tenant_id"].astype(str) == tenant_id)
                & (self._state.messages_df["session_id"].astype(str) == session_id)
            ]
            if subset.empty:
                return 0
            summary = " ".join(subset["content"].astype(str).tail(20).tolist())[:2000]
            capsule_id = f"caps-{tenant_id}-{session_id}"
            row = {
                "capsule_id": capsule_id,
                "tenant_id": tenant_id,
                "session_id": session_id,
                "topic_id": "default",
                "summary": summary,
                "level": "L1",
                "score": 1.0,
                "updated_at": pd.Timestamp.utcnow(),
            }
            self._state.capsules_df = self._state.capsules_df[
                self._state.capsules_df["capsule_id"].astype(str) != capsule_id
            ]
            self._state.capsules_df = pd.concat(
                [self._state.capsules_df, pd.DataFrame([row], columns=CAPSULES_COLUMNS)],
                ignore_index=True,
            )
            return int(self._state.capsules_df.shape[0])

    async def apply_wal_record(self, record: WalRecord) -> None:
        if record.event_type == "message_upsert":
            await self.add_message(record.payload)
        elif record.event_type == "capsule_refresh":
            await self.refresh_capsules(
                str(record.payload.get("tenant_id") or "default"),
                str(record.payload["session_id"]),
            )

    async def record_cache_lookup(
        self,
        *,
        key: str,
        tenant_id: str,
        session_id: str,
        query_type: str,
        capsule_level: str,
        hit: bool,
    ) -> None:
        async with self._lock:
            cache_df = self._state.cache_index_df
            now = pd.Timestamp.utcnow()
            if cache_df.empty:
                row = {
                    "key": key,
                    "tenant_id": tenant_id,
                    "session_id": session_id,
                    "query_type": query_type,
                    "capsule_level": capsule_level,
                    "entity_type": "search",
                    "entity_id": key,
                    "last_access": now,
                    "hit_count": 1 if hit else 0,
                    "miss_count": 0 if hit else 1,
                }
                self._state.cache_index_df = pd.DataFrame([row], columns=CACHE_INDEX_COLUMNS)
                return

            mask = (
                (cache_df["key"].astype(str) == key)
                & (cache_df["tenant_id"].astype(str) == tenant_id)
                & (cache_df["session_id"].astype(str) == session_id)
                & (cache_df["query_type"].astype(str) == query_type)
                & (cache_df["capsule_level"].astype(str) == capsule_level)
            )
            matched = cache_df[mask]
            if matched.empty:
                row = {
                    "key": key,
                    "tenant_id": tenant_id,
                    "session_id": session_id,
                    "query_type": query_type,
                    "capsule_level": capsule_level,
                    "entity_type": "search",
                    "entity_id": key,
                    "last_access": now,
                    "hit_count": 1 if hit else 0,
                    "miss_count": 0 if hit else 1,
                }
                self._state.cache_index_df = pd.concat(
                    [cache_df, pd.DataFrame([row], columns=CACHE_INDEX_COLUMNS)],
                    ignore_index=True,
                )
                return

            idx = matched.index[0]
            def _safe_int(value: object) -> int:
                if value is None or pd.isna(value):
                    return 0
                return int(value)
            self._state.cache_index_df.at[idx, "last_access"] = now
            if hit:
                self._state.cache_index_df.at[idx, "hit_count"] = (
                    _safe_int(self._state.cache_index_df.at[idx, "hit_count"]) + 1
                )
            else:
                self._state.cache_index_df.at[idx, "miss_count"] = (
                    _safe_int(self._state.cache_index_df.at[idx, "miss_count"]) + 1
                )

    def _token_score(self, text: str, query_tokens: List[str]) -> float:
        if not text or not query_tokens:
            return 0.0
        lower = text.lower()
        overlap = 0
        for token in query_tokens:
            if token in lower:
                overlap += 1
        return overlap / max(len(query_tokens), 1)

    def _semantic_score(self, text: str, query: str) -> float:
        text_set = set(text.lower().split())
        query_set = set(query.lower().split())
        union = text_set | query_set
        if not union:
            return 0.0
        return len(text_set & query_set) / len(union)

    async def hybrid_search(
        self,
        query: str,
        tenant_id: str,
        session_id: Optional[str],
        max_results: int,
        min_score: float,
    ) -> List[SearchResult]:
        query_clean = query.strip()
        if not query_clean:
            return []
        query_tokens = [t for t in query_clean.lower().split() if t]
        async with self._lock:
            df = self._state.messages_df
            df = df[df["tenant_id"].astype(str) == tenant_id]
            if session_id:
                df = df[df["session_id"].astype(str) == session_id]
            if df.empty:
                return []
            scored: List[Tuple[float, SearchResult]] = []
            grouped: Dict[str, int] = {}
            for _, row in df.sort_values("ts", kind="stable").iterrows():
                sid = str(row["session_id"])
                tid = str(row.get("tenant_id") or "default")
                grouped_key = f"{tid}:{sid}"
                grouped[grouped_key] = grouped.get(grouped_key, 0) + 1
                content = str(row["content"])
                bm25 = self._token_score(content, query_tokens)
                semantic = self._semantic_score(content, query_clean)
                score = (0.6 * bm25) + (0.4 * semantic)
                if score < min_score:
                    continue
                line_no = grouped[grouped_key]
                path = f"memory/{sid}.md" if tid == "default" else f"memory/{tid}/{sid}.md"
                result = SearchResult(
                    path=path,
                    start_line=line_no,
                    end_line=line_no,
                    score=round(float(score), 6),
                    score_lexical=round(float(bm25), 6),
                    score_semantic=round(float(semantic), 6),
                    snippet=content[:700],
                    source="memory",
                    source_tier=str(row.get("capsule_level") or "L0"),
                    citation=f"message:{row['message_id']}",
                )
                scored.append((score, result))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [item[1] for item in scored[: max(1, max_results)]]

    async def virtual_memory_file(self, rel_path: str) -> Tuple[str, str]:
        normalized = rel_path.replace("\\", "/").lstrip("/")
        if not normalized.startswith("memory/") or not normalized.endswith(".md"):
            raise FileNotFoundError(f"unsupported memory path: {rel_path}")
        parts = normalized.split("/")
        tenant_id = "default"
        if len(parts) == 2:
            session_id = parts[-1].replace(".md", "")
        elif len(parts) == 3:
            tenant_id = parts[1]
            session_id = parts[2].replace(".md", "")
        else:
            raise FileNotFoundError(f"unsupported memory path: {rel_path}")
        async with self._lock:
            df = self._state.messages_df[
                (self._state.messages_df["tenant_id"].astype(str) == tenant_id)
                & (self._state.messages_df["session_id"].astype(str) == session_id)
            ].sort_values("ts", kind="stable")
            if df.empty:
                raise FileNotFoundError(f"no session memory found for tenant={tenant_id} session={session_id}")
            lines = []
            for _, row in df.iterrows():
                role = str(row["role"])
                content = str(row["content"])
                lines.append(f"- [{role}] {content}")
        canonical = f"memory/{session_id}.md" if tenant_id == "default" else f"memory/{tenant_id}/{session_id}.md"
        return "\n".join(lines), canonical

    async def save_parquet(self, parquet_dir: Path) -> None:
        async with self._lock:
            state = self._state
            await asyncio.to_thread(self._save_parquet_sync, parquet_dir, state)

    def _save_parquet_sync(self, parquet_dir: Path, state: DataFramesState) -> None:
        parquet_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")

        def _write_partitioned(df: pd.DataFrame, name: str) -> None:
            if df.empty:
                target = parquet_dir / name / "dt=empty"
                target.mkdir(parents=True, exist_ok=True)
                (target / f"part-{timestamp}.parquet").touch(exist_ok=True)
                return
            write_df = df.copy()
            if "ts" in write_df.columns:
                write_df["dt"] = pd.to_datetime(write_df["ts"], utc=True).dt.strftime("%Y-%m-%d")
            elif "updated_at" in write_df.columns:
                write_df["dt"] = pd.to_datetime(write_df["updated_at"], utc=True).dt.strftime("%Y-%m-%d")
            else:
                write_df["dt"] = datetime.utcnow().strftime("%Y-%m-%d")
            for dt, part in write_df.groupby("dt"):
                target = parquet_dir / name / f"dt={dt}"
                target.mkdir(parents=True, exist_ok=True)
                part.drop(columns=["dt"]).to_parquet(target / f"part-{timestamp}.parquet", index=False)

        _write_partitioned(state.messages_df, "messages")
        _write_partitioned(state.capsules_df, "capsules")
        _write_partitioned(state.cache_index_df, "cache_index")

    async def load_parquet(self, parquet_dir: Path) -> None:
        async with self._lock:
            await asyncio.to_thread(self._load_parquet_sync, parquet_dir)

    def _load_parquet_sync(self, parquet_dir: Path) -> None:
        def _read_all(name: str, columns: List[str]) -> pd.DataFrame:
            base = parquet_dir / name
            if not base.exists():
                return pd.DataFrame(columns=columns)
            files = sorted(base.glob("dt=*/part-*.parquet"))
            if not files:
                return pd.DataFrame(columns=columns)
            parts = []
            for file in files:
                try:
                    part = pd.read_parquet(file)
                    for col in columns:
                        if col not in part.columns:
                            part[col] = None
                    parts.append(part[columns])
                except Exception:
                    continue
            if not parts:
                return pd.DataFrame(columns=columns)
            return pd.concat(parts, ignore_index=True)

        self._state = DataFramesState(
            messages_df=_read_all("messages", MESSAGES_COLUMNS),
            capsules_df=_read_all("capsules", CAPSULES_COLUMNS),
            cache_index_df=_read_all("cache_index", CACHE_INDEX_COLUMNS),
        )
