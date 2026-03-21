import asyncio
import httpx
from qdrant_client import AsyncQdrantClient
from app.core.config import (
    QDRANT_URL,
    DENSE_COLLECTION,
    HYBRID_COLLECTION,
    VECTOR_SIZE,
    DOCLING_URL,
    EMBED_URL,
    RERANK_URL,
    BASE_DIR,
    RESULTS_DIR,
)
from app.crud.qdrant import create_collection, delete_collection
from app.services.rag import ingest_document, query_documents


async def main():
    qdrant = AsyncQdrantClient(url=QDRANT_URL)

    await delete_collection(qdrant, DENSE_COLLECTION)
    await delete_collection(qdrant, HYBRID_COLLECTION)

    await create_collection(
        qdrant,
        collection_name=DENSE_COLLECTION,
        vector_size=VECTOR_SIZE,
        with_sparse=True,
    )
    await create_collection(
        qdrant,
        collection_name=HYBRID_COLLECTION,
        vector_size=VECTOR_SIZE,
        with_sparse=True,
    )

    docling = httpx.AsyncClient(base_url=DOCLING_URL, timeout=600.0)
    embedder = httpx.AsyncClient(base_url=EMBED_URL, timeout=30.0)
    reranker = httpx.AsyncClient(base_url=RERANK_URL, timeout=60.0)

    file_path = BASE_DIR / "examples/1706.03762v7.pdf"
    query = (
        "Transformer attention mechanism computational complexity "
        "time and memory requirements scaling with number of tokens"
    )

    try:
        await ingest_document(
            file_path,
            DENSE_COLLECTION,
            qdrant=qdrant,
            docling=docling,
            embedder=embedder,
            batch_size=10,
            log=True,
            log_dir=RESULTS_DIR / "dense",
        )

        await query_documents(
            query,
            DENSE_COLLECTION,
            qdrant=qdrant,
            embedder=embedder,
            reranker=reranker,
            search_top_k=20,
            rerank_top_k=5,
            batch_size=10,
            log=True,
            log_dir=RESULTS_DIR / "dense",
        )

        await ingest_document(
            file_path,
            HYBRID_COLLECTION,
            qdrant=qdrant,
            docling=docling,
            embedder=embedder,
            with_sparse=True,
            batch_size=10,
            log=True,
            log_dir=RESULTS_DIR / "hybrid",
        )

        await query_documents(
            query,
            HYBRID_COLLECTION,
            qdrant=qdrant,
            embedder=embedder,
            reranker=reranker,
            with_sparse=True,
            fusion_alg="dbsf",
            search_top_k=20,
            rerank_top_k=5,
            batch_size=10,
            log=True,
            log_dir=RESULTS_DIR / "hybrid",
        )

    finally:
        await qdrant.close()
        await docling.aclose()
        await embedder.aclose()
        await reranker.aclose()


if __name__ == "__main__":
    asyncio.run(main())
