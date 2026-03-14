from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import httpx


@dataclass(frozen=True)
class EmbeddingAuthContext:
    provider: str
    api_key: str
    model: Optional[str] = None
    base_url: Optional[str] = None
    auth_source: Optional[str] = None


class EmbeddingRouter:
    async def embed_texts(
        self,
        ctx: EmbeddingAuthContext,
        texts: List[str],
    ) -> List[List[float]]:
        provider = ctx.provider.strip().lower()
        if provider == "openai":
            return await self._embed_openai(ctx, texts)
        if provider == "voyage":
            return await self._embed_voyage(ctx, texts)
        if provider == "mistral":
            return await self._embed_mistral(ctx, texts)
        raise RuntimeError(f"unsupported embedding provider: {ctx.provider}")

    async def _embed_openai(self, ctx: EmbeddingAuthContext, texts: List[str]) -> List[List[float]]:
        url_base = (ctx.base_url or "https://api.openai.com/v1").rstrip("/")
        url = f"{url_base}/embeddings"
        payload = {
            "model": ctx.model or "text-embedding-3-small",
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
