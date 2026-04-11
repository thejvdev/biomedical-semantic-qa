from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
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
    app.state.tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    app.state.tokenizer.enable_padding(pad_id=1, pad_token="<pad>")
    app.state.tokenizer.enable_truncation(max_length=8192)

    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 0
    sess_options.inter_op_num_threads = 1
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    providers = [
        ("CUDAExecutionProvider", {
            "device_id": 0,
        }),
        "CPUExecutionProvider",
    ]

    app.state.session = ort.InferenceSession(
        str(MODEL_PATH), sess_options=sess_options, providers=providers
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
def embed(req: Request, body: EmbedRequest):
    if not body.texts:
        raise HTTPException(status_code=400, detail="Texts list is empty")

    if not any([body.return_dense, body.return_sparse, body.return_colbert]):
        raise HTTPException(
            status_code=400, detail="At least one output must be requested"
        )

    tokenizer = req.app.state.tokenizer
    session = req.app.state.session

    encodings = tokenizer.encode_batch(body.texts)

    input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
    attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
    onnx_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}

    dense, sparse_weights, colbert = session.run(None, onnx_inputs)

    return {
        "dense": dense.tolist() if body.return_dense else None,
        "sparse": (
            build_sparse_dicts(input_ids, sparse_weights)
            if body.return_sparse
            else None
        ),
        "colbert": colbert.tolist() if body.return_colbert else None,
    }
