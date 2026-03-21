import asyncio
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, UploadFile, File
import tempfile
from pydantic import BaseModel
from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer

BASE_DIR = Path(__file__).resolve().parents[1]
TOKENIZER_PATH = BASE_DIR / "tokenizer.json"


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.converter = DocumentConverter()
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    hf_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
    app.state.chunker = HybridChunker(
        tokenizer=HuggingFaceTokenizer(tokenizer=hf_tokenizer, max_tokens=800)
    )
    yield


app = FastAPI(lifespan=lifespan)


class ChunkResponse(BaseModel):
    chunks: list[dict]


@app.post("/chunk", response_model=ChunkResponse)
async def chunk_document(req: Request, file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=Path(file.filename).suffix
    ) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        converter = req.app.state.converter
        chunker = req.app.state.chunker

        result = await asyncio.to_thread(converter.convert, tmp_path)
        doc = result.document

        table_map = {table.self_ref: table for table in doc.tables}
        chunks = []
        seen_tables = set()

        for chunk in chunker.chunk(doc):
            contains_table = False
            is_duplicate = False
            pages = set()

            for doc_item in chunk.meta.doc_items:
                for prov in doc_item.prov:
                    pages.add(prov.page_no)

                if not contains_table and doc_item.label == "table":
                    ref = doc_item.self_ref

                    if ref in seen_tables:
                        is_duplicate = True
                        break

                    table = table_map.get(ref)
                    if table:
                        seen_tables.add(ref)
                        chunk.text = table.export_to_markdown(doc)
                        contains_table = True

            if is_duplicate:
                continue

            chunk_dict = {
                "content": chunk.text,
                "metadata": {
                    "headings": chunk.meta.headings,
                    "contains_table": contains_table,
                    "pages": sorted(list(pages)),
                },
            }

            chunks.append(chunk_dict)

        return {"chunks": chunks}

    finally:
        Path(tmp_path).unlink(missing_ok=True)
