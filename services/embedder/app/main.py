from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer
import onnxruntime as ort
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "bge-m3"
MODEL_PATH = MODEL_DIR / "model.onnx"
SPECIAL_TOKENS = {0, 1, 2}


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))

    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 0
    sess_options.inter_op_num_threads = 1
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    app.session = ort.InferenceSession(
        str(MODEL_PATH), sess_options=sess_options, providers=["CPUExecutionProvider"]
    )

    yield


app = FastAPI(lifespan=lifespan)


class EmbedRequest(BaseModel):
    prompt: str
    return_dense: bool = True
    return_sparse: bool = False
    return_colbert: bool = False


class EmbedResponse(BaseModel):
    dense: list | None
    sparse: dict | None
    colbert: list | None


def build_sparse_dict(input_ids: np.ndarray, token_weights: np.ndarray) -> dict:
    sparse = {}
    for token_id, weight in zip(input_ids[0], token_weights[0]):
        tid = int(token_id)
        if tid in SPECIAL_TOKENS:
            continue
        weight = float(weight)
        if weight > 0:
            sparse[tid] = max(sparse.get(tid, 0), weight)
    return sparse


@app.post("/embed", response_model=EmbedResponse)
def embed(req: EmbedRequest):
    if not req.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt is empty")
    if not any([req.return_dense, req.return_sparse, req.return_colbert]):
        raise HTTPException(
            status_code=400, detail="At least one output must be requested"
        )

    tokenizer = app.tokenizer
    session = app.session

    encoded = tokenizer(
        req.prompt,
        return_tensors="np",
        padding=True,
        truncation=True,
        max_length=8192,
    )

    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    onnx_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}

    dense, sparse_weights, colbert = session.run(None, onnx_inputs)
    sparse_dict = build_sparse_dict(input_ids, sparse_weights)

    return {
        "dense": dense[0].tolist() if req.return_dense else None,
        "sparse": sparse_dict if req.return_sparse else None,
        "colbert": colbert[0].tolist() if req.return_colbert else None,
    }
