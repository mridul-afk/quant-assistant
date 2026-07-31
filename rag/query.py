"""
Answers a question grounded in the ingested book library.

Retrieves the most relevant chunks from Chroma, then asks Claude to answer
using only that retrieved context, citing book + page for every claim.
"""

from __future__ import annotations

import sys
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import CHROMA_DIR, EMBEDDING_MODEL, RETRIEVAL_TOP_K
from rag.llm_backend import get_backend

SYSTEM_PROMPT = """You are a quantitative finance research assistant. You \
answer using ONLY the retrieved book excerpts provided in the user message \
-- do not use outside knowledge to fill gaps. If the excerpts don't contain \
enough to answer, say so plainly rather than guessing.

For every substantive claim, cite the source in the form (Book, p.N) using \
the book/page metadata attached to each excerpt. Prefer precise, technical \
answers over hedged, vague ones -- this is a technical audience. If the \
excerpts contain a derivation or formula, reproduce the reasoning in your \
own words rather than quoting verbatim.

For questions about strategy, trading decisions, or risk, structure your \
answer as Thesis / Evidence / Risks / Position-sizing note, and state your \
assumptions explicitly. Never give a bare buy/sell recommendation without \
risk framing -- if asked "should I buy X," redirect toward running it \
through this project's actual backtest and signal tools (strategy/signals.py, \
strategy/backtest.py) instead of giving general investing advice."""


def retrieve(question: str, collection, embedder, k: int = RETRIEVAL_TOP_K):
    query_embedding = embedder.encode([question]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=k)
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    return list(zip(docs, metas))


def build_context_block(retrieved) -> str:
    parts = []
    for i, (doc, meta) in enumerate(retrieved, 1):
        parts.append(f"[Excerpt {i} -- {meta['book']}, p.{meta['page']}]\n{doc}")
    return "\n\n".join(parts)


def ask(question: str) -> str:
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    client_db = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        collection = client_db.get_collection("quant_books")
    except ValueError:
        raise RuntimeError(
            "No ingested books found. Run: python cli.py ingest-books --pdf-dir <path>"
        )

    retrieved = retrieve(question, collection, embedder)
    if not retrieved:
        return "No relevant material found in your book library for this question."

    context = build_context_block(retrieved)

    backend = get_backend()
    answer = backend.generate(
        system=SYSTEM_PROMPT,
        user=f"Retrieved excerpts:\n\n{context}\n\nQuestion: {question}",
    )

    sources = sorted({f"{m['book']}, p.{m['page']}" for _, m in retrieved})
    answer += "\n\n---\nSources retrieved: " + "; ".join(sources)
    return answer


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ask a question grounded in your book library")
    parser.add_argument("question", type=str)
    args = parser.parse_args()
    print(ask(args.question))
