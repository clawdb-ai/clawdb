from __future__ import annotations

import os
import time
from typing import List, Sequence, Union

import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModel, AutoTokenizer


MODEL_ID = os.getenv("QWEN3_TOPIC_EMBEDDER_MODEL", "Qwen/Qwen3-Embedding-0.6B")
DEVICE = os.getenv("QWEN3_TOPIC_EMBEDDER_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float16 if DEVICE.startswith("cuda") else torch.float32


def _last_token_pool(
    last_hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_indexes = torch.arange(last_hidden_states.shape[0], device=last_hidden_states.device)
    return last_hidden_states[batch_indexes, sequence_lengths]


def _normalize_inputs(value: Union[str, Sequence[str]]) -> List[str]:
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


class EmbeddingRequest(BaseModel):
    input: Union[str, List[str]]
    model: str | None = None
    encoding_format: str | None = None


class EmbeddingDatum(BaseModel):
    object: str = "embedding"
    index: int
    embedding: List[float]


class EmbeddingUsage(BaseModel):
    prompt_tokens: int
    total_tokens: int


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[EmbeddingDatum]
    model: str
    usage: EmbeddingUsage


tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, padding_side="left", trust_remote_code=True)
model = AutoModel.from_pretrained(
    MODEL_ID,
    torch_dtype=DTYPE,
    trust_remote_code=True,
).to(DEVICE)
model.eval()

app = FastAPI(title="clawdb-qwen3-topic-embedder")
started_at = time.time()


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "model": MODEL_ID,
        "device": DEVICE,
        "uptime_seconds": int(max(0.0, time.time() - started_at)),
    }


@app.post("/v1/embeddings", response_model=EmbeddingResponse)
def embeddings(req: EmbeddingRequest) -> EmbeddingResponse:
    requested_model = str(req.model or MODEL_ID).strip()
    if requested_model and requested_model != MODEL_ID:
        raise HTTPException(status_code=400, detail=f"unsupported model: {requested_model}")

    inputs = _normalize_inputs(req.input)
    if not inputs:
        raise HTTPException(status_code=400, detail="input must not be empty")

    batch = tokenizer(
        inputs,
        padding=True,
        truncation=True,
        max_length=int(os.getenv("QWEN3_TOPIC_EMBEDDER_MAX_LENGTH", "512")),
        return_tensors="pt",
    )
    batch = {key: value.to(DEVICE) for key, value in batch.items()}

    with torch.no_grad():
        outputs = model(**batch)
        pooled = _last_token_pool(outputs.last_hidden_state, batch["attention_mask"])
        normalized = F.normalize(pooled, p=2, dim=1)

    vectors = normalized.detach().cpu().tolist()
    token_count = int(batch["attention_mask"].sum().item())
    return EmbeddingResponse(
        data=[EmbeddingDatum(index=idx, embedding=[float(item) for item in vector]) for idx, vector in enumerate(vectors)],
        model=MODEL_ID,
        usage=EmbeddingUsage(prompt_tokens=token_count, total_tokens=token_count),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "qwen3_topic_embedder:app",
        host=os.getenv("QWEN3_TOPIC_EMBEDDER_HOST", "127.0.0.1"),
        port=int(os.getenv("QWEN3_TOPIC_EMBEDDER_PORT", "11440")),
        reload=False,
    )
