"""Single factory for the embedding model, used by both ingestion and retrieval."""
from langchain_huggingface import HuggingFaceEmbeddings

from config import DOCUMENT_PREFIX, EMBEDDING_MODEL_NAME, QUERY_PREFIX


def get_embeddings() -> HuggingFaceEmbeddings:
    """Build the e5 embedding model with the right prefix for each side.

    Having one factory guarantees that indexing and querying use the same model and the same
    settings, which is the number one cause of random-looking retrieval results.

    embed_documents() uses encode_kwargs and embed_query() uses query_encode_kwargs, so the
    "passage: " and "query: " prefixes are applied automatically, each on its own side.
    Vectors are normalized so that cosine similarity is consistent in both directions.
    """
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        encode_kwargs={"normalize_embeddings": True, "prompt": DOCUMENT_PREFIX},
        query_encode_kwargs={"normalize_embeddings": True, "prompt": QUERY_PREFIX},
    )
