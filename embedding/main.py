"""Embedding service using Qwen3-Embedding-0.6B (ModelScope)."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Union

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


class EmbedRequest(BaseModel):
    input: Union[str, list[str]]
    instruction: str = ""


@app.post("/embed")
def embed(req: EmbedRequest):
    texts = [req.input] if isinstance(req.input, str) else req.input
    if req.instruction:
        texts = [f"Instruct: {req.instruction}\nQuery:{t}" for t in texts]

    encoded = tokenizer(texts, padding=True, truncation=True, max_length=8192, return_tensors="pt")
    encoded.to(model.device)
    with torch.no_grad():
        output = model(**encoded)
    embeddings = last_token_pool(output.last_hidden_state, encoded["attention_mask"])
    embeddings = F.normalize(embeddings, p=2, dim=1)
    return {
        "embeddings": embeddings.tolist(),
        "model": MODEL_ID,
        "dim": dim,
    }


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID, "dim": dim}
