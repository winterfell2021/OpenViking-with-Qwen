"""Embedding service using Qwen3-Embedding-0.6B (ModelScope).

Exposes two endpoints:
  POST /embed              — native format
  POST /v1/embeddings      — OpenAI-compatible format (for openviking openai provider)
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import List, Union

import torch
import torch.nn.functional as F
from fastapi import FastAPI
from pydantic import BaseModel
from torch import Tensor

MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
MODEL_PATH = "/app/models/Qwen/Qwen3-Embedding-0.6B"

model = None
tokenizer = None
dim = None


def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths
        ]


def _encode(texts: list[str]) -> list[list[float]]:
    encoded = tokenizer(texts, padding=True, truncation=True, max_length=8192, return_tensors="pt")
    encoded.to(model.device)
    with torch.no_grad():
        output = model(**encoded)
    embeddings = last_token_pool(output.last_hidden_state, encoded["attention_mask"])
    embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings.tolist()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer, dim
    from modelscope import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, padding_side="left")
    model = AutoModel.from_pretrained(MODEL_PATH)
    model.eval()
    dim = model.config.hidden_size
    yield


app = FastAPI(lifespan=lifespan)


# ── Native endpoint ────────────────────────────────────────────────────────────

class EmbedRequest(BaseModel):
    input: Union[str, list[str]]
    instruction: str = ""


@app.post("/embed")
def embed(req: EmbedRequest):
    texts = [req.input] if isinstance(req.input, str) else req.input
    if req.instruction:
        texts = [f"Instruct: {req.instruction}\nQuery:{t}" for t in texts]
    return {"embeddings": _encode(texts), "model": MODEL_ID, "dim": dim}


# ── OpenAI-compatible endpoint ─────────────────────────────────────────────────

class OAIEmbedRequest(BaseModel):
    input: Union[str, List[str]]
    model: str = MODEL_ID
    encoding_format: str = "float"
    dimensions: int = None


@app.post("/v1/embeddings")
def oai_embeddings(req: OAIEmbedRequest):
    texts = [req.input] if isinstance(req.input, str) else req.input
    vecs = _encode(texts)
    return {
        "object": "list",
        "model": req.model,
        "data": [
            {"object": "embedding", "index": i, "embedding": v}
            for i, v in enumerate(vecs)
        ],
        "usage": {
            "prompt_tokens": sum(len(t.split()) for t in texts),
            "total_tokens": sum(len(t.split()) for t in texts),
        },
    }


# ── Health ─────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID, "dim": dim}
