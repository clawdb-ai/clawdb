from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import List, Optional

import httpx


DETERMINISTIC_EMBEDDING_PROVIDER = "deterministic"
DETERMINISTIC_EMBEDDING_MODEL = "hashed-token-v1"


@dataclass(frozen=True)
class EmbeddingAuthContext:
    provider: str
    api_key: str
    model: Optional[str] = None
    base_url: Optional[str] = None
    auth_source: Optional[str] = None


def normalize_embedding_text(value: object) -> str:
    return str(value or "").replace("\r\n", "\n").replace("\r", "\n")


def embedding_source_hash(value: object) -> str:
    normalized = normalize_embedding_text(value)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def deterministic_embedding_ref(entity_type: str, value: object) -> str:
    source_hash = embedding_source_hash(value)
    digest = hashlib.sha256(
        (
            f"{DETERMINISTIC_EMBEDDING_PROVIDER}:"
            f"{DETERMINISTIC_EMBEDDING_MODEL}:"
            f"{str(entity_type or '').strip()}:"
            f"{source_hash}"
        ).encode("utf-8")
    ).hexdigest()
    return f"embed:{str(entity_type or '').strip() or 'entity'}:{digest[:32]}"


def embedding_backend_signature(ctx: EmbeddingAuthContext) -> str:
    provider = str(ctx.provider or "").strip().lower() or "unknown"
    model = str(ctx.model or "").strip() or "_default_"
    base_url = str(ctx.base_url or "").strip().rstrip("/")
    auth_source = str(ctx.auth_source or "").strip()
    return f"{provider}:{model}:{base_url}:{auth_source}"


class EmbeddingRouter:
    async def embed_texts(
        self,
        ctx: EmbeddingAuthContext,
        texts: List[str],
    ) -> List[List[float]]:
        provider = ctx.provider.strip().lower()
        if provider in {"openai", "kimi-coding", "kimi", "moonshot"}:
            return await self._embed_openai(ctx, texts)
        if provider == "voyage":
            return await self._embed_voyage(ctx, texts)
        if provider == "mistral":
            return await self._embed_mistral(ctx, texts)
        raise RuntimeError(f"unsupported embedding provider: {ctx.provider}")

    async def _embed_openai(self, ctx: EmbeddingAuthContext, texts: List[str]) -> List[List[float]]:
        provider = ctx.provider.strip().lower()
        if provider in {"kimi-coding", "kimi", "moonshot"}:
            default_base = "https://api.kimi.com/coding"
            default_model = "k2p5"
        else:
            default_base = "https://api.openai.com/v1"
            default_model = "text-embedding-3-small"
        url_base = (ctx.base_url or default_base).rstrip("/")
        url = f"{url_base}/embeddings"
        payload = {
            "model": ctx.model or default_model,
            "input": texts,
        }
        headers = {
            "content-type": "application/json",
            "authorization": f"Bearer {ctx.api_key}",
        }
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.post(url, headers=headers, json=payload)
        if response.status_code >= 400:
            raise RuntimeError(f"openai embeddings failed: {response.status_code} {response.text[:200]}")
        data = response.json().get("data", [])
        return [list(item.get("embedding", [])) for item in data]

    async def _embed_voyage(self, ctx: EmbeddingAuthContext, texts: List[str]) -> List[List[float]]:
        url_base = (ctx.base_url or "https://api.voyageai.com/v1").rstrip("/")
        url = f"{url_base}/embeddings"
        payload = {
            "model": ctx.model or "voyage-3.5-lite",
            "input": texts,
        }
        headers = {
            "content-type": "application/json",
            "authorization": f"Bearer {ctx.api_key}",
        }
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.post(url, headers=headers, json=payload)
        if response.status_code >= 400:
            raise RuntimeError(f"voyage embeddings failed: {response.status_code} {response.text[:200]}")
        data = response.json().get("data", [])
        return [list(item.get("embedding", [])) for item in data]

    async def _embed_mistral(self, ctx: EmbeddingAuthContext, texts: List[str]) -> List[List[float]]:
        url_base = (ctx.base_url or "https://api.mistral.ai/v1").rstrip("/")
        url = f"{url_base}/embeddings"
        payload = {
            "model": ctx.model or "mistral-embed",
            "input": texts,
        }
        headers = {
            "content-type": "application/json",
            "authorization": f"Bearer {ctx.api_key}",
        }
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.post(url, headers=headers, json=payload)
        if response.status_code >= 400:
            raise RuntimeError(f"mistral embeddings failed: {response.status_code} {response.text[:200]}")
        data = response.json().get("data", [])
        return [list(item.get("embedding", [])) for item in data]
