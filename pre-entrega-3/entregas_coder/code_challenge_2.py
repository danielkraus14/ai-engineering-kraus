import json
import os
import tempfile
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.utils import embedding_functions

DEFAULT_N_RESULTS = 3


class VectorMemoryError(Exception):
    """Raised when a ChromaDB operation fails (disk, corrupted index, bad query)."""


class VectorMemoryManager:
    """Persistent vector memory backed by ChromaDB with the four CRUD operations.

    Documents are embedded with Chroma's default embedding function and stored on disk,
    so the collection survives process restarts. Writes use upsert so that re-ingesting
    a document with an existing id updates it instead of raising a duplicate-id error.

    Error policy: invalid arguments raise ValueError (a caller bug, nothing was attempted);
    any failure inside ChromaDB is wrapped in VectorMemoryError with a clear message and
    the original exception chained, so callers handle one storage error type.
    """

    def __init__(self, persist_path: str, collection_name: str) -> None:
        try:
            # PersistentClient writes the index to disk at persist_path.
            self.client = chromadb.PersistentClient(path=persist_path)

            # The embedding function is set explicitly (instead of relying on the implicit default)
            # so that the model used for indexing and querying is always the same one.
            self.embedding_function = embedding_functions.DefaultEmbeddingFunction()
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                embedding_function=self.embedding_function,
            )
        except Exception as exc:
            raise VectorMemoryError(
                f"could not open collection {collection_name!r} at {persist_path!r}: {exc}") from exc

    def upsert_documents(self, ids: List[str], documents: List[str],
                         metadatas: List[Dict[str, Any]]) -> None:
        """Insert new documents or update existing ones (matched by id).

        Unlike add(), upsert does not fail on duplicated ids, which makes ingestion
        idempotent: running the same load twice leaves the collection unchanged.
        """
        if not (len(ids) == len(documents) == len(metadatas)):
            raise ValueError("ids, documents and metadatas must have the same length")
        try:
            self.collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
        except Exception as exc:
            raise VectorMemoryError(f"upsert of {len(ids)} document(s) failed: {exc}") from exc

    def semantic_search(self, query_text: str, n_results: int = DEFAULT_N_RESULTS,
                        where: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Return the documents closest to query_text.

        The result is Chroma's query dict; each field is a list with one entry per query,
        so the matches for our single query are at index 0. Distances are L2 by default:
        smaller means more similar. `where` optionally filters by metadata.
        """
        try:
            return self.collection.query(
                query_texts=[query_text],
                n_results=min(n_results, max(self.collection.count(), 1)),
                where=where,
            )
        except Exception as exc:
            raise VectorMemoryError(f"query {query_text!r} failed: {exc}") from exc

    def delete_by_id(self, ids: List[str]) -> None:
        """Delete documents by id. Unknown ids are ignored by Chroma (no error)."""
        if not ids:
            raise ValueError("ids must not be empty")
        try:
            self.collection.delete(ids=ids)
        except Exception as exc:
            raise VectorMemoryError(f"delete of {len(ids)} id(s) failed: {exc}") from exc


def fmt_meta(metadata: Dict[str, Any]) -> str:
    """Metadata with sorted keys: Chroma does not guarantee key order, and logs must be comparable."""
    return json.dumps(metadata, sort_keys=True)


def print_results(title: str, results: Dict[str, Any]) -> None:
    """Print a query result as one readable line per match."""
    print(title)
    rows = zip(results["ids"][0], results["distances"][0],
               results["metadatas"][0], results["documents"][0])
    for rank, (doc_id, distance, metadata, document) in enumerate(rows, start=1):
        print(f"  {rank}. id={doc_id} distance={distance:.4f} meta={fmt_meta(metadata)} | {document}")


# Deterministic dataset: three topics, so every search shows a clear expected winner.
IDS = ["doc1", "doc2", "doc3", "doc4", "doc5", "doc6"]
DOCUMENTS = [
    "Docker packages an application into a container that runs the same everywhere.",
    "Kubernetes orchestrates containers and scales them across a cluster.",
    "Python is a programming language popular for data science and automation.",
    "Pandas offers data frames to clean and analyze tabular data in Python.",
    "Pasta carbonara is made with eggs, cheese, pepper and cured pork cheek.",
    "A good tomato sauce simmers slowly with garlic, basil and olive oil.",
]
# `source` adds traceability: in a RAG it tells where each chunk came from.
METADATAS = [
    {"category": "devops", "source": "devops_handbook.md"},
    {"category": "devops", "source": "devops_handbook.md"},
    {"category": "programming", "source": "python_notes.md"},
    {"category": "programming", "source": "python_notes.md"},
    {"category": "cooking", "source": "recipes.md"},
    {"category": "cooking", "source": "recipes.md"},
]


def run_demo() -> None:
    # A temporary directory keeps the demo reproducible: every run starts from an empty database.
    with tempfile.TemporaryDirectory() as db_path:
        manager = VectorMemoryManager(db_path, "test_collection")

        manager.upsert_documents(IDS, DOCUMENTS, METADATAS)
        print(f"Documents after first upsert: {manager.collection.count()}")

        # Idempotency: same ids again must not fail nor duplicate anything.
        manager.upsert_documents(IDS, DOCUMENTS, METADATAS)
        print(f"Documents after repeating the same upsert: {manager.collection.count()}")

        # Upsert on an existing id updates the stored document and its metadata.
        manager.upsert_documents(["doc1"], ["Docker images are built from a Dockerfile."],
                                 [{"category": "devops", "source": "devops_handbook.md", "revised": True}])
        updated = manager.collection.get(ids=["doc1"])
        print(f"After updating doc1: count={manager.collection.count()} "
              f"document={updated['documents'][0]!r} meta={fmt_meta(updated['metadatas'][0])}")

        # Semantic queries: no shared keywords with the expected document.
        queries = {
            "How do I run my app in isolated boxes?": "devops",
            "analyzing spreadsheets with code": "programming",
            "what should I cook for dinner tonight?": "cooking",
        }
        for query, expected in queries.items():
            results = manager.semantic_search(query, n_results=2)
            print_results(f"Query: {query!r} (expected category: {expected})", results)
            top_category = results["metadatas"][0][0]["category"]
            assert top_category == expected, f"expected {expected}, got {top_category}"

        # Metadata filter: restrict the search to one category.
        filtered = manager.semantic_search("containers", n_results=3, where={"category": "cooking"})
        print_results("Query: 'containers' restricted to category=cooking", filtered)
        assert all(m["category"] == "cooking" for m in filtered["metadatas"][0])

        # Persistence: a new manager on the same path sees the data written by the first one.
        reopened = VectorMemoryManager(db_path, "test_collection")
        print(f"Documents seen by a new manager on the same path: {reopened.collection.count()}")
        assert reopened.collection.count() == len(IDS)

        # Delete: the count decreases and the document no longer shows up in searches.
        cooking_query = "what should I cook for dinner tonight?"
        top_before = manager.semantic_search(cooking_query, n_results=1)["ids"][0][0]
        manager.delete_by_id([top_before])
        print(f"Deleted {top_before}: count={manager.collection.count()}")
        assert manager.collection.count() == len(IDS) - 1
        after = manager.semantic_search(cooking_query, n_results=len(IDS))
        print_results(f"Query after delete: {cooking_query!r}", after)
        assert top_before not in after["ids"][0]
        assert manager.collection.get(ids=[top_before])["ids"] == []

        # Deleting an id that does not exist is a no-op in Chroma.
        manager.delete_by_id(["does-not-exist"])
        print(f"Deleted unknown id: count unchanged at {manager.collection.count()}")

        run_error_handling_demo(manager, db_path)


def run_error_handling_demo(manager: VectorMemoryManager, db_path: str) -> None:
    """Force real failures and show that each one is reported clearly instead of crashing."""
    count_before = manager.collection.count()

    # Caller bug: lists of different length are rejected before touching the database.
    try:
        manager.upsert_documents(["x1", "x2"], ["only one document"], [{"category": "test"}])
    except ValueError as exc:
        print(f"Caught ValueError (mismatched lengths): {exc}")
    else:
        raise AssertionError("mismatched lengths should have raised ValueError")

    try:
        manager.delete_by_id([])
    except ValueError as exc:
        print(f"Caught ValueError (empty ids): {exc}")

    # Storage failure: a metadata value Chroma cannot store (a nested dict).
    try:
        manager.upsert_documents(["x3"], ["bad metadata"], [{"nested": {"a": 1}}])
    except VectorMemoryError as exc:
        print(f"Caught VectorMemoryError (invalid metadata): {exc}")
    else:
        raise AssertionError("invalid metadata should have raised VectorMemoryError")

    # Invalid filter on a query.
    try:
        manager.semantic_search("anything", where={"category": {"$bogus": "x"}})
    except VectorMemoryError as exc:
        print(f"Caught VectorMemoryError (invalid where filter): {exc}")
    else:
        raise AssertionError("invalid where filter should have raised VectorMemoryError")

    # Invalid path: a regular file cannot be used as the database directory.
    file_path = os.path.join(db_path, "not_a_directory.txt")
    with open(file_path, "w", encoding="utf-8") as handle:
        handle.write("plain file")
    try:
        VectorMemoryManager(file_path, "broken")
    except VectorMemoryError as exc:
        # The temp directory name is random; mask it so the log is identical between runs.
        print(f"Caught VectorMemoryError (invalid path): {str(exc).replace(db_path, '<db_path>')}")
    else:
        raise AssertionError("invalid path should have raised VectorMemoryError")

    # None of the failed operations changed the stored data.
    assert manager.collection.count() == count_before
    print(f"Collection intact after all failures: count={manager.collection.count()}")


if __name__ == "__main__":
    run_demo()
