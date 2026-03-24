# RAG Service

Not a real service, just a Python module that contains the core logic of the RAG system.

## File Structure

- [`app/`](app/) - project core:
  - [`core/`](app/core/) - core utilities and configuration:
    - [`config.py`](app/core/config.py) - configuration and constants
    - [`utils.py`](app/core/utils.py) - shared utilities
  - [`crud/`](app/crud/) - data access layer:
    - [`qdrant.py`](app/crud/qdrant.py) - interaction logic with the Qdrant
  - [`services/`](app/services/) - project services:
    - [`embedder.py`](app/services/embedder.py) - wrapper around the embedder client
    - [`ingestion.py`](app/services/ingestion.py) - file preparation
    - [`rag.py`](app/services/rag.py) - main module that integrates other services
    - [`reranker.py`](app/services/reranker.py) - wrapper around the reranker client
  - [`main.py`](app/main.py) - main entry point

### `main.py`

- `process_dataset`:
  - prepares the entire dataset from the specified folder path
  - you can try increasing the `batch_size`

- `test1`:
  - example of interacting with the prepared API, from collection creation to searching for relevant chunks
  - includes examples for both dense search and hybrid search

- `test2`:
  - example of using `process_dataset` to prepare the entire dataset

> [!NOTE]
>
> The `logs/` and `results/` folders will be at the same level as the `app/` folder.
