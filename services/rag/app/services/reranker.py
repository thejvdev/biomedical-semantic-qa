import httpx


async def rerank(
    reranker: httpx.AsyncClient,
    query: str,
    candidates: list[str],
    return_proba: bool = True,
    batch_size: int = 16,
) -> list[float] | None:
    if not candidates:
        return None

    result = []

    for i in range(0, len(candidates), batch_size):
        batch = candidates[i : i + batch_size]
        payload = {
            "query": query,
            "texts": batch,
            "return_proba": return_proba,
        }

        response = await reranker.post("/rerank", json=payload)
        response.raise_for_status()

        data = response.json()
        result.extend(data.get("scores", []))

    return result
