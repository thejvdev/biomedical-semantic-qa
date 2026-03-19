from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from tokenizers import Tokenizer
import onnxruntime as ort
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "bge-m3"
MODEL_PATH = MODEL_DIR / "model.onnx"
TOKENIZER_PATH = MODEL_DIR / "tokenizer.json"
SPECIAL_TOKENS = {0, 1, 2}


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    app.tokenizer.enable_padding(pad_id=1, pad_token="<pad>")
    app.tokenizer.enable_truncation(max_length=8192)

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
    texts: list[str]
    return_dense: bool = True
    return_sparse: bool = False
    return_colbert: bool = False


class EmbedResponse(BaseModel):
    dense: list[list[float]] | None
    sparse: list[dict[int, float]] | None
    colbert: list[list[list[float]]] | None


def build_sparse_dicts(input_ids: np.ndarray, token_weights: np.ndarray) -> list:
    results = []
    for i in range(input_ids.shape[0]):
        sparse = {}
        for token_id, weight in zip(input_ids[i], token_weights[i]):
            tid = int(token_id)
            if tid in SPECIAL_TOKENS:
                continue
            weight = float(weight)
            if weight > 0:
                sparse[tid] = max(sparse.get(tid, 0), weight)
        results.append(sparse)
    return results


@app.post("/embed", response_model=EmbedResponse)
def embed(req: EmbedRequest):
    if not req.texts:
        raise HTTPException(status_code=400, detail="Texts list is empty")

    if not any([req.return_dense, req.return_sparse, req.return_colbert]):
        raise HTTPException(
            status_code=400, detail="At least one output must be requested"
        )

    encodings = app.tokenizer.encode_batch(req.texts)

    input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
    attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
    onnx_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}

    dense, sparse_weights, colbert = app.session.run(None, onnx_inputs)

    return {
        "dense": dense.tolist() if req.return_dense else None,
        "sparse": (
            build_sparse_dicts(input_ids, sparse_weights) if req.return_sparse else None
        ),
        "colbert": colbert.tolist() if req.return_colbert else None,
    }
