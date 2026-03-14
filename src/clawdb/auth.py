from __future__ import annotations

import hashlib
import hmac
import time
from dataclasses import dataclass
from typing import Optional

from fastapi import HTTPException

from .embeddings import EmbeddingAuthContext


@dataclass(frozen=True)
class OpenClawRequestAuth:
    bearer_token: Optional[str]
    signature: Optional[str]
    signature_ts: Optional[int]
    embedding: Optional[EmbeddingAuthContext]


def parse_bearer(authorization_header: Optional[str]) -> Optional[str]:
    if not authorization_header:
        return None
    parts = authorization_header.strip().split(" ", 1)
    if len(parts) != 2:
        return None
    if parts[0].lower() != "bearer":
        return None
    token = parts[1].strip()
    return token or None


def build_request_auth(
    authorization_header: Optional[str],
    signature: Optional[str],
    signature_ts: Optional[str],
    embedding_provider: Optional[str],
    embedding_key: Optional[str],
    embedding_model: Optional[str],
    embedding_base_url: Optional[str],
    embedding_auth_source: Optional[str],
) -> OpenClawRequestAuth:
    ts_int: Optional[int] = None
    if signature_ts:
        try:
            ts_int = int(signature_ts)
        except ValueError:
            ts_int = None

    embedding = None
    if embedding_provider and embedding_key:
        embedding = EmbeddingAuthContext(
            provider=embedding_provider,
            api_key=embedding_key,
            model=embedding_model,
            base_url=embedding_base_url,
            auth_source=embedding_auth_source,
        )

    return OpenClawRequestAuth(
        bearer_token=parse_bearer(authorization_header),
        signature=signature,
        signature_ts=ts_int,
        embedding=embedding,
    )


def verify_openclaw_signature_or_raise(
    auth: OpenClawRequestAuth,
    path: str,
    require_signature: bool = True,
    max_skew_seconds: int = 300,
) -> None:
    if not auth.signature and not auth.signature_ts:
        if require_signature:
            raise HTTPException(status_code=401, detail="missing signature headers")
        return

    signing_key = auth.bearer_token or (auth.embedding.api_key if auth.embedding else None)
    if not signing_key:
        raise HTTPException(status_code=401, detail="signature provided without available signing key")
    if not auth.signature or auth.signature_ts is None:
        raise HTTPException(status_code=401, detail="incomplete signature headers")

    now = int(time.time())
    if abs(now - auth.signature_ts) > max_skew_seconds:
        raise HTTPException(status_code=401, detail="signature timestamp skew exceeded")

    payload = f"{path}\n{auth.signature_ts}".encode("utf-8")
    expected = hmac.new(signing_key.encode("utf-8"), payload, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, auth.signature):
        raise HTTPException(status_code=401, detail="invalid signature")
