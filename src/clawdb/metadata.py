from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd


@dataclass(frozen=True)
class CheckpointRecord:
    last_seq: int
    updated_at: datetime


class DataFrameMetadataStore:
    def __init__(self, parquet_path: Path) -> None:
        self._parquet_path = parquet_path
        self._init_lock = asyncio.Lock()
        self._initialized = False

    async def initialize(self) -> None:
        async with self._init_lock:
            if self._initialized:
                return
            self._parquet_path.parent.mkdir(parents=True, exist_ok=True)
            await asyncio.to_thread(self._initialize_sync)
            self._initialized = True

    def _initialize_sync(self) -> None:
        if self._parquet_path.exists():
            return
        df = pd.DataFrame(columns=["slot", "last_seq", "updated_at"])
        df.to_parquet(self._parquet_path, index=False)

    async def load_checkpoint(self, slot: str = "default") -> Optional[CheckpointRecord]:
        await self.initialize()
        return await asyncio.to_thread(self._load_checkpoint_sync, slot)

    def _read_frame_sync(self) -> pd.DataFrame:
        if not self._parquet_path.exists():
            return pd.DataFrame(columns=["slot", "last_seq", "updated_at"])
        try:
            df = pd.read_parquet(self._parquet_path)
        except Exception:
            return pd.DataFrame(columns=["slot", "last_seq", "updated_at"])
        for col in ["slot", "last_seq", "updated_at"]:
            if col not in df.columns:
                df[col] = None
        return df[["slot", "last_seq", "updated_at"]]

    def _load_checkpoint_sync(self, slot: str) -> Optional[CheckpointRecord]:
        df = self._read_frame_sync()
        if df.empty:
            return None
        subset = df[df["slot"].astype(str) == slot]
        if subset.empty:
            return None
        subset = subset.sort_values("updated_at", kind="stable")
        row = subset.iloc[-1]
        updated = pd.to_datetime(row["updated_at"], utc=True).to_pydatetime()
        if updated.tzinfo is None:
            updated = updated.replace(tzinfo=timezone.utc)
        return CheckpointRecord(
            last_seq=int(row["last_seq"]),
            updated_at=updated,
        )

    async def save_checkpoint(self, last_seq: int, slot: str = "default") -> None:
        await self.initialize()
        await asyncio.to_thread(self._save_checkpoint_sync, int(last_seq), slot)

    def _save_checkpoint_sync(self, last_seq: int, slot: str) -> None:
        df = self._read_frame_sync()
        df = df[df["slot"].astype(str) != slot]
        row = pd.DataFrame(
            [
                {
                    "slot": slot,
                    "last_seq": int(last_seq),
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }
            ]
        )
        updated = pd.concat([df, row], ignore_index=True)
        tmp = self._parquet_path.with_suffix(self._parquet_path.suffix + ".tmp")
        updated.to_parquet(tmp, index=False)
        tmp.replace(self._parquet_path)
