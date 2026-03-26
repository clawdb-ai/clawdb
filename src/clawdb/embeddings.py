from __future__ import annotations

import csv
import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import httpx


DETERMINISTIC_EMBEDDING_PROVIDER = "deterministic"
DETERMINISTIC_EMBEDDING_MODEL = "hashed-token-v1"
API_KEY_PATTERN = re.compile(r"\bsk-[A-Za-z0-9][A-Za-z0-9_-]+\b")


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


def _candidate_embedding_key_files() -> List[Path]:
    explicit = os.getenv("CLAWDB_EMBEDDING_KEY_FILE", "").strip()
    candidates: List[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    candidates.extend(
        [
            Path("~/kimi_keys.csv").expanduser(),
            Path("~/kimi_keys.txt").expanduser(),
        ]
    )
    deduped: List[Path] = []
    seen = set()
    for candidate in candidates:
        normalized = candidate.resolve(strict=False)
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(candidate)
    return deduped


def _extract_first_api_key(text: str) -> Optional[str]:
    match = API_KEY_PATTERN.search(text or "")
    return match.group(0) if match else None


def _read_api_key_from_csv(path: Path) -> Optional[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        direct = _extract_first_api_key(sample)
        if direct:
            return direct
        try:
            reader = csv.DictReader(handle)
            for row in reader:
                if not row:
                    continue
                for value in row.values():
                    if value is None:
                        continue
                    extracted = _extract_first_api_key(str(value))
                    if extracted:
                        return extracted
        except csv.Error:
            handle.seek(0)
        handle.seek(0)
        reader2 = csv.reader(handle)
        for row in reader2:
            for value in row:
                extracted = _extract_first_api_key(str(value))
                if extracted:
                    return extracted
    return None


def _read_api_key_from_file(path: Path) -> Optional[str]:
    try:
        if path.suffix.lower() == ".csv":
            return _read_api_key_from_csv(path)
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                extracted = _extract_first_api_key(line)
                if extracted:
                    return extracted
    except OSError:
        return None
    return None


def _infer_provider(api_key: str, source_path: Optional[Path]) -> str:
    lowered_key = api_key.strip().lower()
    path_text = str(source_path or "").lower()
    if lowered_key.startswith("sk-kimi-") or "kimi" in path_text:
        return "kimi-coding"
    return "openai"


def resolve_fallback_embedding_context() -> Optional[EmbeddingAuthContext]:
    provider = os.getenv("CLAWDB_EMBEDDING_PROVIDER", "").strip().lower()
    api_key = os.getenv("CLAWDB_EMBEDDING_API_KEY", "").strip()
    model = os.getenv("CLAWDB_EMBEDDING_MODEL", "").strip() or None
    base_url = os.getenv("CLAWDB_EMBEDDING_BASE_URL", "").strip() or None
    auth_source: Optional[str] = None
    source_path: Optional[Path] = None

    if api_key:
        auth_source = "env:CLAWDB_EMBEDDING_API_KEY"
    else:
        for candidate in _candidate_embedding_key_files():
            if not candidate.exists():
                continue
            resolved = _read_api_key_from_file(candidate)
            if resolved:
                api_key = resolved
                source_path = candidate
                auth_source = f"file:{candidate.expanduser()}"
                break

    if not api_key:
        return None

    if not provider:
        provider = _infer_provider(api_key, source_path)

    if provider in {"kimi-coding", "kimi", "moonshot"}:
        if not model:
            model = "k2p5"
        if not base_url:
            base_url = "https://api.kimi.com/coding"
    elif provider == "openai":
        if not model:
            model = "text-embedding-3-small"
        if not base_url:
            base_url = "https://api.openai.com/v1"
    elif provider == "voyage":
        if not model:
            model = "voyage-3.5-lite"
        if not base_url:
            base_url = "https://api.voyageai.com/v1"
    elif provider == "mistral":
        if not model:
            model = "mistral-embed"
        if not base_url:
            base_url = "https://api.mistral.ai/v1"

    return EmbeddingAuthContext(
        provider=provider,
        api_key=api_key,
        model=model,
        base_url=base_url,
        auth_source=auth_source,
    )


def resolve_topic_embedding_context() -> Optional[EmbeddingAuthContext]:
    provider = os.getenv("CLAWDB_TOPIC_EMBEDDING_PROVIDER", "").strip().lower()
    api_key = os.getenv("CLAWDB_TOPIC_EMBEDDING_API_KEY", "").strip()
    model = os.getenv("CLAWDB_TOPIC_EMBEDDING_MODEL", "").strip() or None
    base_url = os.getenv("CLAWDB_TOPIC_EMBEDDING_BASE_URL", "").strip() or None
    auth_source: Optional[str] = None

    key_file = os.getenv("CLAWDB_TOPIC_EMBEDDING_KEY_FILE", "").strip()
    if not api_key and key_file:
        candidate = Path(key_file).expanduser()
        if candidate.exists():
            resolved = _read_api_key_from_file(candidate)
            if resolved:
                api_key = resolved
                auth_source = f"file:{candidate.expanduser()}"

    if not provider and any([api_key, model, base_url]):
        provider = "openai"
    if not provider:
        provider = "openai"

    if not api_key and base_url:
        api_key = "local-topic-embedder"
        auth_source = "implicit:local-base-url"
    elif api_key and auth_source is None:
        auth_source = "env:CLAWDB_TOPIC_EMBEDDING_API_KEY"

    if provider in {"openai", "kimi-coding", "kimi", "moonshot"}:
        if not model:
            model = "Qwen/Qwen3-Embedding-0.6B"
        if not base_url:
            base_url = "http://127.0.0.1:11440/v1"
        if not api_key:
            api_key = "local-topic-embedder"
            if auth_source is None:
                auth_source = "implicit:default-local-topic-embedder"

    if not api_key:
        return None

    return EmbeddingAuthContext(
        provider=provider,
        api_key=api_key,
        model=model,
        base_url=base_url,
        auth_source=auth_source,
    )


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
