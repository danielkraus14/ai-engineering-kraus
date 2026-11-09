"""Robustness tests: persistence, delete, input validation, and real forced failures.

Nothing is mocked. Errors are provoked for real (an empty folder, an invalid API key, a model
that does not exist) and each test checks that the failure is caught and reported with the
project's own exception, chained to the original one.

Needs GOOGLE_API_KEY (the generation tests call Gemini). Run with: python test_robustness.py
"""
import asyncio
import logging
import os
import shutil
import sys
import tempfile
from collections.abc import Awaitable, Callable
from pathlib import Path

from dotenv import load_dotenv

from chain import build_rag_chain
from config import FALLBACK_MODEL, MAX_QUERY_CHARS
from exceptions import (
    DocumentLoadError,
    GenerationError,
    InvalidQueryError,
    VectorStoreError,
)
from ingest import DATA_DIR, count_chunks, ingest_documents, load_documents
from rag import generate_answer, get_rag_response, validate_query
from retriever import load_vectorstore, retrieve
from schemas import RAGAnswer

load_dotenv()
logger = logging.getLogger("test_robustness")

TEST_QUESTION = "Cuanto es el deposito en garantia de un alquiler?"
INVALID_API_KEY = "invalid-key-for-testing"


async def test_ingestion_persistence_and_delete() -> None:
    """Ingest into a temporary store: idempotent re-run, and count drops after a delete."""
    with tempfile.TemporaryDirectory() as tmp:
        data_dir, store_path = Path(tmp) / "data", Path(tmp) / "store"
        shutil.copytree(DATA_DIR, data_dir)

        store = ingest_documents(data_dir, store_path)
        full_count = count_chunks(store)
        assert full_count > 0, "nothing was indexed"

        store = ingest_documents(data_dir, store_path)
        assert count_chunks(store) == full_count, "re-ingesting changed the count (not idempotent)"

        (data_dir / "faq_barrios.md").unlink()
        store = ingest_documents(data_dir, store_path)
        reduced_count = count_chunks(store)
        assert reduced_count < full_count, "count did not drop after deleting a document"
        sources = {m["source"] for m in store.get()["metadatas"]}
        assert "faq_barrios.md" not in sources, "stale chunks of the deleted file remain"
        logger.info("count before delete=%d, after delete=%d", full_count, reduced_count)


async def test_metadata_traceability() -> None:
    """Every stored chunk carries source and chunk_index so answers can be traced."""
    store = load_vectorstore()
    for metadata in store.get()["metadatas"]:
        assert metadata.get("source"), "chunk without source"
        assert isinstance(metadata.get("chunk_index"), int), "chunk without chunk_index"


async def test_missing_data_dir() -> None:
    try:
        load_documents(Path(tempfile.gettempdir()) / "does-not-exist-rag")
    except DocumentLoadError:
        return
    raise AssertionError("DocumentLoadError was not raised")


async def test_empty_data_dir() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        try:
            load_documents(Path(tmp))
        except DocumentLoadError:
            return
    raise AssertionError("DocumentLoadError was not raised")


async def test_unreadable_document() -> None:
    """A file that is not valid UTF-8 raises DocumentLoadError chained to the decode error."""
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "broken.md").write_bytes(b"\xff\xfe\x00 not utf-8 \x80")
        try:
            load_documents(Path(tmp))
        except DocumentLoadError as exc:
            assert isinstance(exc.__cause__, UnicodeDecodeError), "exception is not chained"
            return
    raise AssertionError("DocumentLoadError was not raised")


async def test_empty_vectorstore() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        try:
            load_vectorstore(Path(tmp) / "empty-store")
        except VectorStoreError:
            return
    raise AssertionError("VectorStoreError was not raised")


async def test_invalid_queries() -> None:
    for bad in ["", "   ", "x" * (MAX_QUERY_CHARS + 1), None]:
        try:
            await get_rag_response(bad)  # type: ignore[arg-type]
        except InvalidQueryError:
            continue
        raise AssertionError(f"InvalidQueryError was not raised for {bad!r}"[:80])
    assert validate_query("  hola  ") == "hola", "query was not stripped"


async def test_invalid_api_key() -> None:
    """A real call with a wrong key fails on both models and becomes a chained GenerationError."""
    documents = await retrieve(load_vectorstore(), TEST_QUESTION)
    original_key = os.environ.get("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = INVALID_API_KEY
    try:
        chain = build_rag_chain()
        try:
            await generate_answer(chain, documents, TEST_QUESTION)
        except GenerationError as exc:
            assert exc.__cause__ is not None, "exception is not chained"
            return
        raise AssertionError("GenerationError was not raised")
    finally:
        if original_key is None:
            os.environ.pop("GOOGLE_API_KEY", None)
        else:
            os.environ["GOOGLE_API_KEY"] = original_key


async def test_fallback_model_takes_over() -> None:
    """The primary model does not exist, so the call must still succeed through the fallback."""
    documents = await retrieve(load_vectorstore(), TEST_QUESTION)
    chain = build_rag_chain(primary_model="gemini-model-that-does-not-exist", fallback_model=FALLBACK_MODEL)
    answer = await generate_answer(chain, documents, TEST_QUESTION)
    assert isinstance(answer, RAGAnswer), "fallback did not return a RAGAnswer"
    assert answer.found_in_context, "fallback answered but did not find the deposit information"


TESTS: list[Callable[[], Awaitable[None]]] = [
    test_ingestion_persistence_and_delete,
    test_metadata_traceability,
    test_missing_data_dir,
    test_empty_data_dir,
    test_unreadable_document,
    test_empty_vectorstore,
    test_invalid_queries,
    test_invalid_api_key,
    test_fallback_model_takes_over,
]


async def main() -> int:
    if not os.environ.get("GOOGLE_API_KEY"):
        print("GOOGLE_API_KEY is not set: copy .env.example to .env and fill it in.")
        return 2
    failed: list[str] = []
    for test in TESTS:
        try:
            await test()
            print(f"PASS {test.__name__}")
        except AssertionError as exc:
            failed.append(test.__name__)
            print(f"FAIL {test.__name__}: {exc}")
        except Exception as exc:  # an unexpected exception type is also a failure
            failed.append(test.__name__)
            print(f"FAIL {test.__name__}: unexpected {type(exc).__name__}: {exc}")
    print(f"\n{len(TESTS) - len(failed)}/{len(TESTS)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for noisy in ("httpx", "httpcore", "huggingface_hub", "sentence_transformers", "google_genai", "ingest", "retriever"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    sys.exit(asyncio.run(main()))
