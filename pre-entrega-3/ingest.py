"""Ingestion module: load documents from data/, chunk them and persist them in ChromaDB."""
import hashlib
import logging
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer

from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    SENTENCE_SEPARATORS,
    VECTORSTORE_DIR,
)
from embeddings import get_embeddings
from exceptions import DocumentLoadError, EmbeddingError, VectorStoreError

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"
SUPPORTED_EXTENSIONS = {".md", ".txt"}
STORE_PATH = Path(__file__).parent / VECTORSTORE_DIR


def load_documents(data_dir: Path = DATA_DIR) -> list[Document]:
    """Read every supported file in data_dir and return one Document per file.

    The file name is stored in metadata["source"] so that the final answer can cite
    which document each piece of information came from.

    Raises:
        DocumentLoadError: the folder does not exist, has no supported files, or a file
            cannot be read or is not valid UTF-8.
    """
    if not data_dir.is_dir():
        raise DocumentLoadError(f"data directory not found: {data_dir}")

    # sorted() makes the order deterministic, so every run processes files the same way.
    paths = sorted(p for p in data_dir.iterdir() if p.suffix in SUPPORTED_EXTENSIONS)
    if not paths:
        raise DocumentLoadError(f"no {sorted(SUPPORTED_EXTENSIONS)} files found in {data_dir}")

    documents: list[Document] = []
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise DocumentLoadError(f"cannot read {path.name}: {exc}") from exc
        if not text.strip():
            raise DocumentLoadError(f"{path.name} is empty")
        documents.append(Document(page_content=text, metadata={"source": path.name}))
    logger.info("Loaded %d documents from %s", len(documents), data_dir.name)
    return documents


def split_documents(documents: list[Document]) -> list[Document]:
    """Split documents into overlapping chunks sized in tokens of the embedding model.

    The length is measured with the embedding model's own tokenizer (not tiktoken) because
    that tokenizer decides what the model truncates. Chunks are built from whole sentences,
    so they never end mid-sentence and the overlap repeats complete sentences.

    keep_separator="end" is required with regex separators: with False, LangChain joins the
    pieces using the literal pattern text and corrupts the chunks.

    Raises:
        EmbeddingError: the tokenizer of the embedding model cannot be loaded.
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
    except OSError as exc:
        raise EmbeddingError(f"cannot load tokenizer {EMBEDDING_MODEL_NAME}: {exc}") from exc

    splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=SENTENCE_SEPARATORS,
        is_separator_regex=True,
        keep_separator="end",
    )
    chunks = splitter.split_documents(documents)

    # Number the chunks inside each source file. Together with the source name this gives
    # every chunk a stable identity, useful for citations and for idempotent indexing.
    counters: dict[str, int] = {}
    for chunk in chunks:
        source = chunk.metadata["source"]
        chunk.metadata["chunk_index"] = counters.get(source, 0)
        counters[source] = chunk.metadata["chunk_index"] + 1
    return chunks


def make_chunk_id(chunk: Document) -> str:
    """Build a deterministic id from the source, the position and a hash of the text.

    Same chunk gives the same id on every run, so re-ingesting never duplicates it. The text
    hash makes the id change when the content changes, so edited documents are detected.
    """
    digest = hashlib.sha256(chunk.page_content.encode("utf-8")).hexdigest()[:8]
    return f"{chunk.metadata['source']}-{chunk.metadata['chunk_index']:03d}-{digest}"


def get_vectorstore(store_path: Path = STORE_PATH, collection_name: str = COLLECTION_NAME) -> Chroma:
    """Open (or create) the persistent ChromaDB collection.

    The collection uses cosine distance, the metric e5 was trained for. Chroma defaults to
    squared L2, which would rank results differently.

    Raises:
        VectorStoreError: ChromaDB cannot open or create the collection.
        EmbeddingError: the embedding model cannot be loaded.
    """
    embeddings = get_embeddings()
    try:
        return Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=str(store_path),
            collection_configuration={"hnsw": {"space": "cosine"}},
        )
    except Exception as exc:  # chromadb raises several unrelated types
        raise VectorStoreError(f"cannot open the vector store at {store_path}: {exc}") from exc


def ingest_documents(data_dir: Path = DATA_DIR, store_path: Path = STORE_PATH) -> Chroma:
    """Synchronize the vector store with the documents in data_dir and return it.

    Instead of checking whether the folder exists (which cannot detect an interrupted run),
    the stored ids are compared with the ids the current documents should produce. Only the
    missing chunks are embedded, and chunks that no longer exist are removed (the delete
    operation of the store). When nothing changed, no embedding is computed.

    Raises:
        DocumentLoadError, EmbeddingError: see load_documents and split_documents.
        VectorStoreError: a ChromaDB read, write or delete failed.
    """
    chunks = split_documents(load_documents(data_dir))
    chunk_ids = [make_chunk_id(chunk) for chunk in chunks]
    store = get_vectorstore(store_path)

    try:
        existing_ids = set(store.get()["ids"])
        expected_ids = set(chunk_ids)

        stale_ids = existing_ids - expected_ids
        if stale_ids:
            store.delete(ids=sorted(stale_ids))
            logger.info("Removed %d stale chunks", len(stale_ids))

        missing = [(c, i) for c, i in zip(chunks, chunk_ids) if i not in existing_ids]
        if not missing:
            logger.info("Vector store is up to date (%d chunks), nothing to index", len(expected_ids))
            return store

        store.add_documents(
            documents=[chunk for chunk, _ in missing],
            ids=[chunk_id for _, chunk_id in missing],
        )
    except Exception as exc:  # chromadb raises several unrelated types
        raise VectorStoreError(f"vector store operation failed: {exc}") from exc
    logger.info("Indexed %d new chunks, collection now has %d", len(missing), len(expected_ids))
    return store


def count_chunks(store: Chroma) -> int:
    """Number of chunks currently stored."""
    try:
        return len(store.get()["ids"])
    except Exception as exc:
        raise VectorStoreError(f"cannot count chunks: {exc}") from exc


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    # Third-party libraries log every HTTP request at INFO level and bury our own messages.
    for noisy in ("httpx", "huggingface_hub", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    ingest_documents()
