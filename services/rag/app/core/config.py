import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parents[2]
RESULTS_DIR = BASE_DIR / "results"
LOG_DIR = BASE_DIR / "logs"

QDRANT_URL = os.getenv("QDRANT_URL")
DENSE_COLLECTION = "dense_collection"
HYBRID_COLLECTION = "hybrid_collection"
VECTOR_SIZE = 1024

EMBED_URL = os.getenv("EMBED_URL")
RERANK_URL = os.getenv("RERANK_URL")
