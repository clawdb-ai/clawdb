from __future__ import annotations

import asyncio
import json
import os
import time
import zlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List

from .models import WalRecord


class WalManager:
    def __init__(
        self,
        wal_dir: Path,
        sync_policy: str = "always",
        sync_interval_ms: int = 25,
    ) -> None:
        self._wal_dir = wal_dir
        self._wal_dir.mkdir(parents=True, exist_ok=True)
        self._wal_path = self._wal_dir / "wal-00000001.log"
        self._seq = 0
        self._append_lock = asyncio.Lock()
        self._sync_policy = sync_policy
        self._sync_interval_ms = sync_interval_ms
        self._last_sync_monotonic = 0.0
        self._bootstrap_seq()

    def _bootstrap_seq(self) -> None:
        if not self._wal_path.exists():
            self._seq = 0
            return
        last_seq = 0
        with self._wal_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    last_seq = max(last_seq, int(data.get("seq", 0)))
                except json.JSONDecodeError:
                    continue
        self._seq = last_seq

    def _record_checksum(self, seq: int, ts: str, event_type: str, payload: Dict[str, Any]) -> int:
        base = json.dumps(
            {
                "seq": seq,
                "ts": ts,
                "event_type": event_type,
                "payload": payload,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        return zlib.crc32(base)

    async def append(self, event_type: str, payload: Dict[str, Any]) -> WalRecord:
        async with self._append_lock:
            self._seq += 1
            ts = datetime.now(timezone.utc)
            ts_text = ts.isoformat()
            checksum = self._record_checksum(self._seq, ts_text, event_type, payload)
            record = WalRecord(
                seq=self._seq,
                ts=ts,
                event_type=event_type,
                payload=payload,
                checksum=checksum,
            )
            await asyncio.to_thread(self._append_sync, record)
            return record

    def _append_sync(self, record: WalRecord) -> None:
        with self._wal_path.open("a", encoding="utf-8") as f:
            f.write(record.model_dump_json())
            f.write("\n")
            f.flush()
            if self._sync_policy == "always":
                os.fsync(f.fileno())
                self._last_sync_monotonic = time.monotonic()
            elif self._sync_policy == "interval":
                now = time.monotonic()
                elapsed_ms = (now - self._last_sync_monotonic) * 1000.0
                if self._last_sync_monotonic == 0.0 or elapsed_ms >= self._sync_interval_ms:
                    os.fsync(f.fileno())
                    self._last_sync_monotonic = now
            else:
                # Unknown policy falls back to strongest mode to avoid durability regressions.
                os.fsync(f.fileno())
                self._last_sync_monotonic = time.monotonic()

    def replay(self, from_seq_exclusive: int = 0) -> Iterator[WalRecord]:
        if not self._wal_path.exists():
            return iter([])

        def _iter() -> Iterator[WalRecord]:
            with self._wal_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    seq = int(data["seq"])
                    if seq <= from_seq_exclusive:
                        continue
                    record = WalRecord.model_validate(data)
                    expected = self._record_checksum(
                        record.seq,
                        record.ts.isoformat(),
                        record.event_type,
                        record.payload,
                    )
                    if expected != record.checksum:
                        raise RuntimeError(f"WAL checksum mismatch at seq={record.seq}")
                    yield record

        return _iter()

    @property
    def last_seq(self) -> int:
        return self._seq

    @property
    def wal_path(self) -> Path:
        return self._wal_path
