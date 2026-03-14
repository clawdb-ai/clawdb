from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Dict
from uuid import uuid4


@dataclass
class QueueEvent:
    event_id: str
    wal_seq: int
    event_type: str
    tenant_id: str
    session_id: str
    created_at: datetime
    payload: Dict[str, Any]


class AsyncMessageQueue:
    async def publish(self, event: QueueEvent) -> None:
        raise NotImplementedError

    async def consume(self) -> AsyncIterator[QueueEvent]:
        raise NotImplementedError

    async def lag(self) -> int:
        raise NotImplementedError

    async def close(self) -> None:
        return None


class InMemoryMessageQueue(AsyncMessageQueue):
    def __init__(self) -> None:
        self._queue: asyncio.Queue[QueueEvent] = asyncio.Queue()

    async def publish(self, event: QueueEvent) -> None:
        await self._queue.put(event)

    async def consume(self) -> AsyncIterator[QueueEvent]:
        while True:
            event = await self._queue.get()
            yield event

    async def lag(self) -> int:
        return self._queue.qsize()


class KafkaMessageQueue(AsyncMessageQueue):
    def __init__(
        self,
        topic: str,
        bootstrap_servers: str,
        group_id: str = "clawdb-memory",
    ) -> None:
        self.topic = topic
        self.bootstrap_servers = bootstrap_servers
        self.group_id = group_id
        self._producer = None
        self._consumer = None

    async def _ensure_clients(self) -> None:
        if self._producer is not None and self._consumer is not None:
            return
        try:
            from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
        except ImportError as exc:
            raise RuntimeError(
                "Kafka backend selected but aiokafka is not installed"
            ) from exc
        if self._producer is None:
            self._producer = AIOKafkaProducer(bootstrap_servers=self.bootstrap_servers)
            await self._producer.start()
        if self._consumer is None:
            self._consumer = AIOKafkaConsumer(
                self.topic,
                bootstrap_servers=self.bootstrap_servers,
                group_id=self.group_id,
                enable_auto_commit=True,
                value_deserializer=lambda b: b.decode("utf-8"),
            )
            await self._consumer.start()

    async def publish(self, event: QueueEvent) -> None:
        import json

        await self._ensure_clients()
        payload = {
            "event_id": event.event_id,
            "wal_seq": event.wal_seq,
            "event_type": event.event_type,
            "tenant_id": event.tenant_id,
            "session_id": event.session_id,
            "created_at": event.created_at.isoformat(),
            "payload": event.payload,
        }
        assert self._producer is not None
        await self._producer.send_and_wait(self.topic, json.dumps(payload).encode("utf-8"))

    async def consume(self) -> AsyncIterator[QueueEvent]:
        import json

        await self._ensure_clients()
        assert self._consumer is not None
        async for msg in self._consumer:
            data = json.loads(msg.value)
            yield QueueEvent(
                event_id=str(data["event_id"]),
                wal_seq=int(data["wal_seq"]),
                event_type=str(data["event_type"]),
                tenant_id=str(data.get("tenant_id", "default")),
                session_id=str(data.get("session_id", "default")),
                created_at=datetime.fromisoformat(data["created_at"]),
                payload=dict(data.get("payload", {})),
            )

    async def lag(self) -> int:
        # A precise lag implementation depends on broker offsets. Keep this conservative.
        return 0

    async def close(self) -> None:
        if self._consumer is not None:
            await self._consumer.stop()
            self._consumer = None
        if self._producer is not None:
            await self._producer.stop()
            self._producer = None


class ZeroMQMessageQueue(AsyncMessageQueue):
    def __init__(self, endpoint: str = "inproc://clawdb-memory-events") -> None:
        self.endpoint = endpoint
        self._started = False
        self._context = None
        self._push = None
        self._pull = None
        self._lag = 0
        self._lag_lock = asyncio.Lock()
        self._start_lock = asyncio.Lock()

    async def _ensure_started(self) -> None:
        if self._started:
            return
        async with self._start_lock:
            if self._started:
                return
            try:
                import zmq
                import zmq.asyncio
            except ImportError as exc:
                raise RuntimeError("ZeroMQ backend selected but pyzmq is not installed") from exc
            self._context = zmq.asyncio.Context.instance()
            self._pull = self._context.socket(zmq.PULL)
            self._pull.bind(self.endpoint)
            self._push = self._context.socket(zmq.PUSH)
            self._push.connect(self.endpoint)
            self._started = True

    async def publish(self, event: QueueEvent) -> None:
        import json

        await self._ensure_started()
        assert self._push is not None
        payload = {
            "event_id": event.event_id,
            "wal_seq": event.wal_seq,
            "event_type": event.event_type,
            "tenant_id": event.tenant_id,
            "session_id": event.session_id,
            "created_at": event.created_at.isoformat(),
            "payload": event.payload,
        }
        await self._push.send_string(json.dumps(payload))
        async with self._lag_lock:
            self._lag += 1

    async def consume(self) -> AsyncIterator[QueueEvent]:
        import json

        await self._ensure_started()
        assert self._pull is not None
        while True:
            raw = await self._pull.recv_string()
            data = json.loads(raw)
            async with self._lag_lock:
                self._lag = max(0, self._lag - 1)
            yield QueueEvent(
                event_id=str(data["event_id"]),
                wal_seq=int(data["wal_seq"]),
                event_type=str(data["event_type"]),
                tenant_id=str(data.get("tenant_id", "default")),
                session_id=str(data.get("session_id", "default")),
                created_at=datetime.fromisoformat(data["created_at"]),
                payload=dict(data.get("payload", {})),
            )

    async def lag(self) -> int:
        async with self._lag_lock:
            return self._lag

    async def close(self) -> None:
        async with self._start_lock:
            if self._push is not None:
                self._push.close(linger=0)
                self._push = None
            if self._pull is not None:
                self._pull.close(linger=0)
                self._pull = None
            self._started = False


def build_event(wal_seq: int, event_type: str, payload: Dict[str, Any]) -> QueueEvent:
    return QueueEvent(
        event_id=str(uuid4()),
        wal_seq=wal_seq,
        event_type=event_type,
        tenant_id=str(payload.get("tenant_id", "default")),
        session_id=str(payload.get("session_id", "default")),
        created_at=datetime.now(timezone.utc),
        payload=payload,
    )


def create_queue(backend: str, topic: str, zeromq_endpoint: str) -> AsyncMessageQueue:
    if backend == "kafka":
        servers = "localhost:9092"
        return KafkaMessageQueue(topic=topic, bootstrap_servers=servers)
    if backend == "zeromq":
        return ZeroMQMessageQueue(endpoint=zeromq_endpoint)
    return InMemoryMessageQueue()
