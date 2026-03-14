from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import IntEnum
from time import monotonic
from typing import AsyncIterator, Dict, List, Tuple


class LockRank(IntEnum):
    SESSION = 10
    TOPIC = 20
    CAPSULE = 30
    INDEX = 40


class DeadlockRiskError(RuntimeError):
    pass


@dataclass
class LockState:
    acquired_at: float
    owner_task_id: int
    rank: LockRank


@dataclass
class WaitState:
    task_id: int
    key: str
    rank: LockRank
    started_at: float


class DeadlockSafeLockManager:
    def __init__(self, lock_timeout_seconds: float = 1.5, watchdog_seconds: float = 10.0):
        self._locks: Dict[str, asyncio.Lock] = {}
        self._held: Dict[int, List[Tuple[str, LockRank]]] = {}
        self._states: Dict[str, LockState] = {}
        self._waiting: Dict[int, WaitState] = {}
        self._lock_timeout = lock_timeout_seconds
        self._watchdog_seconds = watchdog_seconds

    def _task_id(self) -> int:
        task = asyncio.current_task()
        if task is None:
            raise RuntimeError("Lock acquisition requires an asyncio task")
        return id(task)

    def _get_lock(self, key: str) -> asyncio.Lock:
        if key not in self._locks:
            self._locks[key] = asyncio.Lock()
        return self._locks[key]

    @asynccontextmanager
    async def acquire(self, key: str, rank: LockRank) -> AsyncIterator[None]:
        task_id = self._task_id()
        held = self._held.setdefault(task_id, [])
        if held and rank < held[-1][1]:
            raise DeadlockRiskError(
                f"Lock order violation: attempted {rank.name} after {held[-1][1].name}"
            )
        lock = self._get_lock(key)
        self._waiting[task_id] = WaitState(
            task_id=task_id,
            key=key,
            rank=rank,
            started_at=monotonic(),
        )
        try:
            await asyncio.wait_for(lock.acquire(), timeout=self._lock_timeout)
        except TimeoutError as exc:
            owner = self._states.get(key)
            owner_text = str(owner.owner_task_id) if owner is not None else "none"
            raise DeadlockRiskError(
                f"Lock acquisition timeout key={key} rank={rank.name} owner={owner_text}"
            ) from exc
        finally:
            self._waiting.pop(task_id, None)
        held.append((key, rank))
        self._states[key] = LockState(acquired_at=monotonic(), owner_task_id=task_id, rank=rank)
        try:
            yield
        finally:
            if lock.locked():
                lock.release()
            self._states.pop(key, None)
            entries = self._held.get(task_id, [])
            if entries:
                entries.pop()
            if not entries:
                self._held.pop(task_id, None)

    async def watchdog_once(self) -> List[str]:
        now = monotonic()
        alerts: List[str] = []
        for key, state in self._states.items():
            held_seconds = now - state.acquired_at
            if held_seconds >= self._watchdog_seconds:
                alerts.append(
                    f"lock_watchdog: key={key} rank={state.rank.name} owner={state.owner_task_id} held={held_seconds:.2f}s"
                )
        for task_id, wait in self._waiting.items():
            wait_seconds = now - wait.started_at
            if wait_seconds < self._lock_timeout:
                continue
            owner = self._states.get(wait.key)
            owner_text = str(owner.owner_task_id) if owner else "none"
            alerts.append(
                f"lock_wait_watchdog: key={wait.key} rank={wait.rank.name} waiter={task_id} owner={owner_text} waited={wait_seconds:.2f}s"
            )
            if owner and owner.owner_task_id in self._waiting:
                owner_wait = self._waiting[owner.owner_task_id]
                owner_target = self._states.get(owner_wait.key)
                if owner_target and owner_target.owner_task_id == task_id:
                    alerts.append(
                        f"lock_deadlock_cycle: waiter={task_id} waits_for={owner.owner_task_id} and reverse_wait_detected"
                    )
        return alerts
