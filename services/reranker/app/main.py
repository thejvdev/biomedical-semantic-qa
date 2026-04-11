from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from tokenizers import Tokenizer
import onnxruntime as ort
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "bge-reranker-v2-m3"
MODEL_PATH = MODEL_DIR / "model.onnx"
TOKENIZER_PATH = MODEL_DIR / "tokenizer.json"


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    app.state.tokenizer.enable_padding(pad_id=1, pad_token="<pad>")
    app.state.tokenizer.enable_truncation(max_length=512)

    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 0
    sess_options.inter_op_num_threads = 1
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    providers = [
        (
            "CUDAExecutionProvider",
            {
                "device_id": 0,
            },
        ),
        "CPUExecutionProvider",  # fallback
    ]

    app.state.session = ort.InferenceSession(
        str(MODEL_PATH),
        sess_options=sess_options,
        providers=providers,
    )

    print("Available providers:", ort.get_available_providers())
    print("Session providers:", app.state.session.get_providers())

    yield


app = FastAPI(lifespan=lifespan)


class RerankRequest(BaseModel):
    query: str
    texts: list[str]
    return_proba: bool = True


class RerankResponse(BaseModel):
    scores: list[float]


@app.post("/rerank", response_model=RerankResponse)
def rerank(req: Request, body: RerankRequest):
    if not body.query.strip():
        raise HTTPException(status_code=400, detail="Query is empty")

    if not body.texts:
        raise HTTPException(status_code=400, detail="Texts list is empty")

    tokenizer = req.app.state.tokenizer
    session = req.app.state.session

    pairs = [(body.query, text) for text in body.texts]
    encodings = tokenizer.encode_batch(pairs)

    input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
    attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)

    onnx_inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }

    outputs = session.run(None, onnx_inputs)
    scores = outputs[0].flatten()

    if body.return_proba:
        scores = 1 / (1 + np.exp(-scores))

    return {"scores": scores.tolist()}
