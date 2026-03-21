import uuid
from typing import Literal
from qdrant_client import AsyncQdrantClient, models


async def create_collection(
    qdrant: AsyncQdrantClient,
    collection_name: str,
    vector_size: int,
    with_sparse: bool = False,
    log: bool = True,
):
    if await qdrant.collection_exists(collection_name):
        if log:
            print(f"Collection '{collection_name}' already exists.")
        return

    vectors_config = models.VectorParams(
        size=vector_size, distance=models.Distance.COSINE
    )

    sparse_config = None
    if with_sparse:
        sparse_config = {
            "sparse": models.SparseVectorParams(
                index=models.SparseIndexParams(full_scan_threshold=1000)
            )
        }

    await qdrant.create_collection(
        collection_name=collection_name,
        vectors_config=vectors_config,
        sparse_vectors_config=sparse_config,
    )

    if log:
        print(f"Collection '{collection_name}' created successfully.")


async def delete_collection(
    qdrant: AsyncQdrantClient, collection_name: str, log: bool = True
):
    if await qdrant.collection_exists(collection_name):
        await qdrant.delete_collection(collection_name)
        if log:
            print(f"Collection '{collection_name}' deleted successfully.")


async def upsert_data(
    qdrant: AsyncQdrantClient,
    collection_name: str,
    dense_vectors: list[list[float]],
    metadatas: list[dict],
    sparse_vectors: list[dict[int, float]] = None,
    log: bool = True,
):
    sparse_iter = (
        sparse_vectors if sparse_vectors is not None else [None] * len(dense_vectors)
    )

    points = []
    for dense, sparse, metadata in zip(dense_vectors, sparse_iter, metadatas):
        vectors = {"": dense}

        if sparse is not None:
            vectors["sparse"] = models.SparseVector(
                indices=list(sparse.keys()),
                values=list(sparse.values()),
            )

        points.append(
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector=vectors,
                payload=metadata,
            )
        )

    await qdrant.upsert(collection_name=collection_name, points=points)

    if log:
        print(f"Successfully upserted {len(points)} points to '{collection_name}'.")


async def similarity_search(
    qdrant: AsyncQdrantClient,
    collection_name: str,
    query_vector: list[float],
    score_threshold: float = 0.5,
    top_k: int = 100,
) -> list[dict]:
    response = await qdrant.query_points(
        collection_name=collection_name,
        query=query_vector,
        score_threshold=score_threshold,
        limit=top_k,
    )
    return [point.model_dump() for point in response.points]


async def hybrid_search(
    qdrant: AsyncQdrantClient,
    collection_name: str,
    dense_query: list[float],
    sparse_query: dict[int, float],
    fusion_alg: Literal["rrf", "dbsf"] = "rrf",
    dense_threshold: float = 0.5,
    sparse_threshold: float = 0.3,
    top_k: int = 100,
) -> list[dict]:
    sparse_vector = models.SparseVector(
        indices=list(sparse_query.keys()), values=list(sparse_query.values())
    )

    fusion = models.Fusion.RRF if fusion_alg == "rrf" else models.Fusion.DBSF

    response = await qdrant.query_points(
        collection_name=collection_name,
        prefetch=[
            models.Prefetch(
                query=dense_query,
                using="",
                score_threshold=dense_threshold,
            ),
            models.Prefetch(
                query=sparse_vector,
                using="sparse",
                score_threshold=sparse_threshold,
            ),
        ],
        query=models.FusionQuery(fusion=fusion),
        limit=top_k,
    )

    return [point.model_dump() for point in response.points]
