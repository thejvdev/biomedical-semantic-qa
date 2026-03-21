from pathlib import Path
import httpx


async def chunk_document(
    docling: httpx.AsyncClient,
    file_path: Path,
) -> list[dict]:
    with open(file_path, "rb") as f:
        response = await docling.post(
            "/chunk",
            files={"file": (file_path.name, f, "application/pdf")},
        )

    response.raise_for_status()

    data = response.json()
    return data.get("chunks", [])
