import asyncio
from pathlib import Path

import httpx
from qdrant_client import AsyncQdrantClient
from tqdm import tqdm

from app.core.config import (
    QDRANT_URL,
    DENSE_COLLECTION,
    HYBRID_COLLECTION,
    VECTOR_SIZE,
    EMBED_URL,
    RERANK_URL,
    BASE_DIR,
    RESULTS_DIR,
    LOG_DIR,
)
from app.core.utils import fetch_filepaths, save_json
from app.crud.qdrant import create_collection, delete_collection
from app.services.rag import ingest_document, query_documents

import json
import os


async def process_dataset(
    input_dir: str | Path,
    collection_name: str,
    *,
    qdrant: AsyncQdrantClient,
    embedder: httpx.AsyncClient,
    retries: int = 5,
    log_dir: Path = LOG_DIR,
):
    xml_files = fetch_filepaths(input_dir)
    error_files = []

    for file_path in tqdm(xml_files, desc="Ingesting dataset"):
        success = False

        for i in range(retries):
            try:
                await ingest_document(
                    file_path,
                    collection_name,
                    qdrant=qdrant,
                    embedder=embedder,
                    with_sparse=True,
                    batch_size=16,
                )
                success = True
                break
            except Exception as e:
                print("type:", type(e))
                print("args:", e.args)
                print("repr:", repr(e))
                print("Exception was raised")
                if i < retries - 1:
                    await asyncio.sleep(2**i)

        if not success:
            print("Not success")
            error_files.append(str(file_path))

    if error_files:
        save_json(log_dir / "error_files.json", error_files)


async def test1():
    qdrant = AsyncQdrantClient(url=QDRANT_URL)

    await delete_collection(qdrant, DENSE_COLLECTION)
    await delete_collection(qdrant, HYBRID_COLLECTION)

    await create_collection(
        qdrant,
        collection_name=DENSE_COLLECTION,
        vector_size=VECTOR_SIZE,
    )
    await create_collection(
        qdrant,
        collection_name=HYBRID_COLLECTION,
        vector_size=VECTOR_SIZE,
        with_sparse=True,
    )

    embedder = httpx.AsyncClient(base_url=EMBED_URL, timeout=30.0)
    reranker = httpx.AsyncClient(base_url=RERANK_URL, timeout=60.0)

    file_path = BASE_DIR / "examples/pubmed26n0001.xml"

    # Ground Truth: pmid 1
    query = "Were animal models, specifically Pseudomonas or Haplorhini, used in the development of formate assays?"

    try:
        await ingest_document(
            file_path,
            DENSE_COLLECTION,
            qdrant=qdrant,
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
        await embedder.aclose()
        await reranker.aclose()


async def test2():
    qdrant = AsyncQdrantClient(url=QDRANT_URL)

    await delete_collection(qdrant, HYBRID_COLLECTION)
    await create_collection(
        qdrant,
        collection_name=HYBRID_COLLECTION,
        vector_size=VECTOR_SIZE,
        with_sparse=True,
    )

    embedder = httpx.AsyncClient(base_url=EMBED_URL, timeout=30.0)

    await process_dataset(
        BASE_DIR / "examples", HYBRID_COLLECTION, qdrant=qdrant, embedder=embedder
    )

    await qdrant.close()
    await embedder.aclose()



def reformat_chunks(chunks: list, query: dict)->dict:
    result: dict = {
        'documents': [],
        'snippets': [],
        **query
    }
    for chunk in chunks:
        result['documents'].append(
            f"http://www.ncbi.nlm.nih.gov/pubmed/{chunk['article']['pmid']}"
        )
        result['snippets'].append({
            "beginSection": "abstract",
            "endSection": "abstract",     
            "text": chunk['article']['abstract'],
            "document": f"http://www.ncbi.nlm.nih.gov/pubmed/{chunk['article']['pmid']}",
            "offsetInBeginSection": 0,
            "offsetInEndSection": len(chunk['article']['abstract']) 
        })

    return result
        
    



async def process_json_from_bioasq(json_path: Path):
    with open(json_path) as f:
        query_data = json.load(f)

    qdrant_url: str = os.environ.get("QDRANT_URL", "http://localhost:6333")
    embed_url: str = os.environ.get("EMBED_URL", "http://localhost:8001")
    reranker_url: str = os.environ.get("EMBED_URL", "http://localhost:8002")
    vector_size: int = 1024 

    qdrant = AsyncQdrantClient(url=qdrant_url)

    embedder = httpx.AsyncClient(base_url=embed_url, timeout=30.0)
    reranker = httpx.AsyncClient(base_url=reranker_url, timeout=60.0)

    searched_results: dict[str, list] = {
        "questions": []
    }

    for query in tqdm(query_data['questions'], desc="Processing queries ..."):

        completed = False
        while not completed:
            try:
                chunks = await query_documents(
                    query['body'],
                    "test-collection",
                    qdrant=qdrant,
                    embedder=embedder,
                    reranker=reranker,
                    with_sparse=True,
                    fusion_alg="dbsf",
                    search_top_k=20,
                    rerank_top_k=5,
                    batch_size=10,
                    log_dir=RESULTS_DIR / "hybrid",
                )

                # filter chunks by tresh
                chunks = list(filter(
                    lambda x: x['score'] >= 0.7,
                    chunks
                ))

                searched_results['questions'].append(
                    reformat_chunks(chunks, query)
                )
                completed = True
            except Exception:
                continue



    await qdrant.close()
    await embedder.aclose()
    await reranker.aclose()

    with open(json_path.parent / "processed_query_for_rag.json", 'w') as f:
        json.dump(searched_results, f, ensure_ascii=False, indent=2)





async def main():
    # await test1()
    #await test2()
    await process_json_from_bioasq(
        json_path=Path(__file__).parents[1] / "query_for_rag.json"
    )


if __name__ == "__main__":
    asyncio.run(main())
