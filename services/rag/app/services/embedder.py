import httpx


async def embed(
    embedder: httpx.AsyncClient,
    texts: list[str],
    return_sparse: bool = False,
    batch_size: int = 16,
) -> dict[str, list] | None:
    if not texts:
        return None

    result = {"dense": [], "sparse": []}

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        payload = {
            "texts": batch,
            "return_dense": True,
            "return_sparse": return_sparse,
        }

        response = await embedder.post("/embed", json=payload)
        response.raise_for_status()

        data = response.json()

        result["dense"].extend(data.get("dense", []))
        if return_sparse:
            result["sparse"].extend(data.get("sparse", []))

    return result
