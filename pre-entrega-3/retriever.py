"""Retrieval layer: open the vector store and search it."""
import logging
import time
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document

from config import TOP_K
from exceptions import RetrievalError, VectorStoreError
from ingest import STORE_PATH, get_vectorstore

logger = logging.getLogger(__name__)

RetrievedChunks = list[tuple[Document, float]]


def load_vectorstore(store_path: Path = STORE_PATH) -> Chroma:
    """Open the persistent vector store and fail early if it has not been populated.

    Raises:
        VectorStoreError: the store cannot be opened or contains no chunks.
    """
    store = get_vectorstore(store_path)
    try:
        is_empty = not store.get()["ids"]
    except Exception as exc:
        raise VectorStoreError(f"cannot read the vector store: {exc}") from exc
    if is_empty:
        raise VectorStoreError("the vector store is empty, run 'python ingest.py' first")
    return store


async def retrieve(store: Chroma, query: str, top_k: int = TOP_K) -> RetrievedChunks:
    """Return the top_k chunks closest to the query, with their cosine distance.

    The query is embedded by the same embedding model used for indexing (store holds it),
    with the "query: " prefix applied automatically. Lower distance means more similar.
    The search runs in a worker thread so it does not block the event loop.

    Raises:
        RetrievalError: the search failed.
    """
    started = time.perf_counter()
    try:
        results = await store.asimilarity_search_with_score(query, k=top_k)
    except Exception as exc:  # chromadb and the embedding model raise unrelated types
        raise RetrievalError(f"similarity search failed: {exc}") from exc
    logger.info("Retrieved %d chunks in %.2fs", len(results), time.perf_counter() - started)
    for doc, distance in results:
        logger.info(
            "  distance=%.3f source=%s chunk=%d",
            distance, doc.metadata["source"], doc.metadata["chunk_index"],
        )
    return results


def format_context(results: RetrievedChunks) -> str:
    """Join the retrieved chunks into one context string, labeling each with its source.

    The labels let the model cite the right file in its references.
    """
    return "\n\n".join(
        f"[source: {doc.metadata['source']} | chunk {doc.metadata['chunk_index']}]\n{doc.page_content}"
        for doc, _ in results
    )
