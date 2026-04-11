import os
import argparse
import asyncio
import httpx
from pathlib import Path
from qdrant_client import AsyncQdrantClient


from app.crud.qdrant import create_collection, delete_collection
from .main import process_dataset





if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Create Vector Collection",
        description="Create vector collection with qdrant from provided path that contain xml from pubmed."
    )

    parser.add_argument(
        "--xml_folder", 
        type=Path, 
        default=Path("../../pubmed_dataset/xml")
    )
    parser.add_argument(
        "--collection_name",
        type=str,
        default="test-collection"
    )

    args = parser.parse_args()

    # --- other varialbes setup ---
    qdrant_url: str = os.environ.get("QDRANT_URL", "http://localhost:6333")
    embed_url: str = os.environ.get("EMBED_URL", "http://localhost:8001")
    vector_size: int = 1024 
    # -----------------------------


    qdrant = AsyncQdrantClient(url=qdrant_url)

    asyncio.run(delete_collection(qdrant, args.collection_name))
    asyncio.run(create_collection(
        qdrant,
        collection_name=args.collection_name,
        vector_size=vector_size,
        with_sparse=True,
    ))

    embedder = httpx.AsyncClient(base_url=embed_url, timeout=30.0)

    asyncio.run(process_dataset(
        args.xml_folder, args.collection_name, qdrant=qdrant, embedder=embedder
    ))

    asyncio.run(qdrant.close())
    asyncio.run(embedder.aclose())

