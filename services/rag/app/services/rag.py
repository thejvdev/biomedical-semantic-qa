import asyncio
from pathlib import Path
from typing import Literal
import httpx
from qdrant_client import AsyncQdrantClient
from app.core.config import RESULTS_DIR
from app.core.utils import save_json
from app.crud.qdrant import (
    upsert_data,
    similarity_search,
    hybrid_search,
)
from app.services.ingestion import parse_document, flatten_article
from app.services.embedder import embed
from app.services.reranker import rerank


async def ingest_document(
    file_path: str | Path,
    collection_name: str,
    *,
    qdrant: AsyncQdrantClient,
    embedder: httpx.AsyncClient,
    with_sparse: bool = False,
    batch_size: int = 16,
    log: bool = False,
    log_dir: Path = RESULTS_DIR,
):
    chunks = await asyncio.to_thread(parse_document, file_path)
    if log:
        save_json(log_dir / "ingestion/1_chunks.json", chunks)

    texts = [flatten_article(chunk) for chunk in chunks]
    vectors = await embed(
        embedder, texts=texts, return_sparse=with_sparse, batch_size=batch_size
    )

    if log:
        save_json(log_dir / "ingestion/2_vectors.json", vectors)

    dense_vectors = vectors["dense"]
    sparse_vectors = vectors["sparse"] if with_sparse else None

    await upsert_data(
        qdrant,
        collection_name=collection_name,
        dense_vectors=dense_vectors,
        sparse_vectors=sparse_vectors,
        metadatas=chunks,
        upsert_batch_size=3,
        log=log,
    )


async def query_documents(
    query: str,
    collection_name: str,
    *,
    qdrant: AsyncQdrantClient,
    embedder: httpx.AsyncClient,
    reranker: httpx.AsyncClient,
    with_sparse: bool = False,
    fusion_alg: Literal["rrf", "dbsf"] = "rrf",
    dense_threshold: float = 0.5,
    sparse_threshold: float = 0.3,
    search_top_k: int = 100,
    rerank_top_k: int = 10,
    return_proba: bool = True,
    batch_size: int = 16,
    log: bool = False,
    log_dir: Path = RESULTS_DIR,
) -> list[dict]:
    if not query.strip():
        return []

    query_vectors = await embed(embedder, [query], return_sparse=with_sparse)
    if log:
        save_json(log_dir / "search/1_query_vector.json", query_vectors)

    dense_query = query_vectors["dense"][0]
    sparse_query = query_vectors["sparse"][0] if with_sparse else None

    if with_sparse:
        scored_points = await hybrid_search(
            qdrant,
            collection_name=collection_name,
            dense_query=dense_query,
            sparse_query=sparse_query,
            fusion_alg=fusion_alg,
            dense_threshold=dense_threshold,
            sparse_threshold=sparse_threshold,
            top_k=search_top_k,
        )
    else:
        scored_points = await similarity_search(
            qdrant,
            collection_name=collection_name,
            query_vector=dense_query,
            score_threshold=dense_threshold,
            top_k=search_top_k,
        )

    if log:
        save_json(log_dir / "search/2_similarity.json", scored_points)

    candidates = [flatten_article(point["payload"]) for point in scored_points]
    reranked = await rerank(
        reranker,
        query=query,
        candidates=candidates,
        return_proba=return_proba,
        batch_size=batch_size,
    )

    reranked_chunks = [
        {"article": point["payload"], "score": score}
        for point, score in zip(scored_points, reranked)
    ]
    reranked_chunks = sorted(reranked_chunks, key=lambda x: x["score"], reverse=True)[
        :rerank_top_k
    ]

    if log:
        save_json(log_dir / "search/3_reranked.json", reranked_chunks)

    return reranked_chunks
