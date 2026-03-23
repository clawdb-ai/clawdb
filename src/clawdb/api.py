from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from .auth import build_request_auth, verify_openclaw_signature_or_raise
from .models import (
    CapsuleRefreshRequest,
    MessageIn,
    MessageEditRequest,
    MessageDeleteRequest,
    SessionForkRequest,
    SessionSnapshotRequest,
    SessionSpawnRequest,
    OpenClawMemoryReadRequest,
    OpenClawMemorySearchRequest,
    SearchRequest,
)
from .service import ClawDBService


service = ClawDBService()


@asynccontextmanager
async def lifespan(_: FastAPI):
    await service.startup()
    try:
        yield
    finally:
        await service.shutdown()


app = FastAPI(title="ClawDB Memory Service", version="0.1.0", lifespan=lifespan)


@app.post("/v1/memory/messages")
async def create_message(req: MessageIn):
    try:
        return await service.ingest_message(req)
    except ClawDBService.BackpressureRejectedError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/v1/memory/messages/edit")
async def edit_message(req: MessageEditRequest):
    try:
        return await service.edit_message(req)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/v1/memory/messages/delete")
async def delete_message(req: MessageDeleteRequest):
    try:
        return await service.delete_message(req)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/v1/memory/search")
async def search_memory(req: SearchRequest):
    return await service.search(req)


@app.post("/v1/memory/capsules/refresh")
async def refresh_capsules(req: CapsuleRefreshRequest):
    return await service.refresh_capsules(req)


@app.get("/v1/memory/sessions/{session_id}")
async def session_memory(session_id: str):
    # Expose session summary through virtual memory file shape.
    try:
        return await service.openclaw_memory_get(f"memory/{session_id}.md", 1, 5000)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/v1/memory/present/linear/{session_id}")
async def present_linear(session_id: str, tenant_id: str = "default"):
    try:
        return await service.present_linear_im(tenant_id=tenant_id, session_id=session_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/v1/memory/present/capsules/{session_id}")
async def present_capsules(session_id: str, tenant_id: str = "default"):
    return await service.present_capsule_cards(tenant_id=tenant_id, session_id=session_id)


@app.get("/v1/memory/present/forum/{session_id}")
async def present_forum(session_id: str, tenant_id: str = "default"):
    return await service.present_forum_style(tenant_id=tenant_id, session_id=session_id)


@app.post("/v1/memory/sessions/snapshot")
async def create_session_snapshot(req: SessionSnapshotRequest):
    return await service.create_snapshot(req)


@app.get("/v1/memory/sessions/{session_id}/snapshots")
async def list_session_snapshots(session_id: str, tenant_id: str = "default"):
    return await service.list_session_snapshots(tenant_id=tenant_id, session_id=session_id)


@app.post("/v1/memory/sessions/fork")
async def fork_session(req: SessionForkRequest):
    return await service.fork_session(req)


@app.post("/v1/memory/sessions/spawn")
async def spawn_session(req: SessionSpawnRequest):
    return await service.spawn_session(req)


@app.get("/v1/memory/index/status")
async def index_status():
    return await service.index_status()


@app.post("/v1/memory/index/rebuild")
async def index_rebuild():
    return await service.rebuild_indexes()


@app.get("/v1/memory/health")
async def memory_health():
    return await service.health()


@app.get("/v1/memory/metrics/cache-hit")
async def cache_hit_report():
    return await service.cache_hit_report()


@app.get("/metrics")
async def prometheus_metrics():
    payload = generate_latest()
    return PlainTextResponse(payload.decode("utf-8"), media_type=CONTENT_TYPE_LATEST)


@app.post("/v1/openclaw/memory/search")
async def openclaw_memory_search(req: OpenClawMemorySearchRequest, request: Request):
    auth = build_request_auth(
        authorization_header=request.headers.get("authorization"),
        signature=request.headers.get("x-openclaw-signature"),
        signature_ts=request.headers.get("x-openclaw-signature-ts"),
        embedding_provider=request.headers.get("x-clawdb-embedding-provider"),
        embedding_key=request.headers.get("x-clawdb-embedding-key"),
        embedding_model=request.headers.get("x-clawdb-embedding-model"),
        embedding_base_url=request.headers.get("x-clawdb-embedding-base-url"),
        embedding_auth_source=request.headers.get("x-clawdb-embedding-auth-source"),
    )
    verify_openclaw_signature_or_raise(
        auth,
        path=request.url.path,
        require_signature=service.config.openclaw_require_signature,
    )
    return await service.openclaw_memory_search(req, embedding_ctx=auth.embedding)


@app.post("/v1/openclaw/memory/get")
async def openclaw_memory_get(req: OpenClawMemoryReadRequest, request: Request):
    auth = build_request_auth(
        authorization_header=request.headers.get("authorization"),
        signature=request.headers.get("x-openclaw-signature"),
        signature_ts=request.headers.get("x-openclaw-signature-ts"),
        embedding_provider=request.headers.get("x-clawdb-embedding-provider"),
        embedding_key=request.headers.get("x-clawdb-embedding-key"),
        embedding_model=request.headers.get("x-clawdb-embedding-model"),
        embedding_base_url=request.headers.get("x-clawdb-embedding-base-url"),
        embedding_auth_source=request.headers.get("x-clawdb-embedding-auth-source"),
    )
    verify_openclaw_signature_or_raise(
        auth,
        path=request.url.path,
        require_signature=service.config.openclaw_require_signature,
    )
    try:
        return await service.openclaw_memory_get(
            rel_path=req.relPath,
            from_line=req.fromLine or 1,
            lines=req.lines or 200,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
