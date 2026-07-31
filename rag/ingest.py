"""
Ingests PDFs into a local vector store.

Pipeline: PDF -> per-page text extraction -> overlapping character chunks ->
local sentence-transformer embeddings -> Chroma persistent collection.

Everything here runs locally. No book content is sent anywhere at this
stage (the only network call in the whole project that sends text to an
LLM is in rag/query.py, and it only sends the small retrieved snippets
needed to answer your specific question, not the books themselves).
"""

from __future__ import annotations

import sys
from pathlib import Path

import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import CHROMA_DIR, CHUNK_OVERLAP, CHUNK_SIZE, EMBEDDING_MODEL


def extract_pages(pdf_path: Path) -> list[tuple[int, str]]:
    """Returns [(page_number, text), ...] for a PDF, skipping empty pages."""
    reader = PdfReader(str(pdf_path))
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = text.strip()
        if text:
            pages.append((i + 1, text))
    return pages


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Simple overlapping character-window chunker.

    Good enough for retrieval purposes; splits on paragraph boundaries where
    possible so chunks don't cut sentences in half as often.
    """
    if len(text) <= size:
        return [text]

    n = len(text)
    chunks = []
    start = 0
    while start < n:
        end = min(start + size, n)
        window = text[start:end]
        reached_end = end >= n

        # try to break on a paragraph or sentence boundary near the end,
        # but only if we're not already at the end of the text
        if not reached_end:
            for sep in ("\n\n", ". ", "\n"):
                idx = window.rfind(sep)
                if idx > size * 0.5:  # don't shrink the chunk too aggressively
                    window = window[: idx + len(sep)]
                    break

        stripped = window.strip()
        if stripped:
            chunks.append(stripped)

        if reached_end:
            break

        # advance by window length minus overlap; guard against the
        # degenerate case where a shrunk window is smaller than overlap,
        # which would otherwise crawl forward 1 char at a time
        step = len(window) - overlap
        start += step if step > overlap * 0.25 else size - overlap

    return chunks


def ingest_pdf(pdf_path: Path, collection, embedder) -> int:
    """Ingests a single PDF into the given Chroma collection. Returns chunk count."""
    book_name = pdf_path.stem
    pages = extract_pages(pdf_path)

    all_chunks, metadatas, ids = [], [], []
    for page_num, page_text in pages:
        for j, chunk in enumerate(chunk_text(page_text)):
            all_chunks.append(chunk)
            metadatas.append({"book": book_name, "page": page_num})
            ids.append(f"{book_name}::p{page_num}::c{j}")

    if not all_chunks:
        return 0

    # embed in batches to keep memory reasonable on CPU-only machines
    batch_size = 64
    for i in tqdm(range(0, len(all_chunks), batch_size), desc=f"embedding {book_name}"):
        batch = all_chunks[i : i + batch_size]
        embeddings = embedder.encode(batch, show_progress_bar=False).tolist()
        collection.upsert(
            documents=batch,
            embeddings=embeddings,
            metadatas=metadatas[i : i + batch_size],
            ids=ids[i : i + batch_size],
        )

    return len(all_chunks)


def ingest_directory(pdf_dir: Path) -> None:
    pdf_dir = Path(pdf_dir)
    pdf_paths = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_paths:
        print(f"No PDFs found in {pdf_dir}")
        return

    print(f"Loading embedding model ({EMBEDDING_MODEL})...")
    embedder = SentenceTransformer(EMBEDDING_MODEL)

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection("quant_books")

    total = 0
    for pdf_path in pdf_paths:
        print(f"\nIngesting {pdf_path.name}")
        count = ingest_pdf(pdf_path, collection, embedder)
        print(f"  -> {count} chunks")
        total += count

    print(f"\nDone. {total} chunks from {len(pdf_paths)} books stored in {CHROMA_DIR}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest PDFs into the RAG vector store")
    parser.add_argument("--pdf-dir", required=True, type=Path)
    args = parser.parse_args()
    ingest_directory(args.pdf_dir)
