"""Project exceptions. Each layer raises its own type so callers can tell failures apart.

Low-level errors (OSError, chromadb errors, API errors) are always re-raised with
"raise ... from exc", which keeps the original traceback as __cause__.
"""


class RAGError(Exception):
    """Base class of every error raised by this project."""


class DocumentLoadError(RAGError):
    """The documents in data/ could not be found, read or decoded."""


class EmbeddingError(RAGError):
    """The embedding model or its tokenizer could not be loaded."""


class VectorStoreError(RAGError):
    """A ChromaDB operation failed, or the store is missing or empty."""


class RetrievalError(RAGError):
    """The similarity search failed."""


class GenerationError(RAGError):
    """The LLM call failed after all retries and the fallback model."""


class InvalidQueryError(RAGError, ValueError):
    """The user question is empty or too long. Also a ValueError for convenience."""
