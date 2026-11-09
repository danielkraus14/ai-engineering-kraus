"""Public entry point of the RAG system: get_rag_response(query)."""
import asyncio
import logging
import time
from functools import cache

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.runnables import Runnable

from chain import build_rag_chain
from config import MAX_QUERY_CHARS
from exceptions import GenerationError, InvalidQueryError
from retriever import RetrievedChunks, load_vectorstore, retrieve
from schemas import RAGAnswer

load_dotenv()
logger = logging.getLogger(__name__)


@cache
def _get_store() -> Chroma:
    """Open the vector store once per process (loading the embedding model is slow)."""
    return load_vectorstore()


@cache
def _get_chain() -> Runnable:
    """Build the chain once per process."""
    return build_rag_chain()


def validate_query(query: str) -> str:
    """Return the query without surrounding spaces, or raise InvalidQueryError."""
    if not isinstance(query, str) or not query.strip():
        raise InvalidQueryError("the question must be a non-empty string")
    cleaned = query.strip()
    if len(cleaned) > MAX_QUERY_CHARS:
        raise InvalidQueryError(f"the question has {len(cleaned)} characters, the limit is {MAX_QUERY_CHARS}")
    return cleaned


def find_ungrounded_quotes(answer: RAGAnswer, documents: RetrievedChunks) -> list[str]:
    """Return the quotes that do not appear literally in the retrieved chunks.

    A quote that is not in the context was invented or altered by the model, so this is a
    cheap check that the references are real. Whitespace is normalized before comparing.
    """
    normalize = lambda text: " ".join(text.split())
    context = normalize(" ".join(doc.page_content for doc, _ in documents))
    return [ref.quote for ref in answer.references if normalize(ref.quote) not in context]


async def generate_answer(chain: Runnable, documents: RetrievedChunks, query: str) -> RAGAnswer:
    """Run the generation chain and turn any failure into a GenerationError.

    Raises:
        GenerationError: the primary model failed after its retries and the fallback model
            failed too (or returned output that does not match the schema).
    """
    try:
        return await chain.ainvoke({"documents": documents, "question": query})
    except Exception as exc:  # API, network and parsing errors have unrelated types
        raise GenerationError(f"the LLM call failed after retries and fallback: {exc}") from exc


async def get_rag_response(query: str) -> RAGAnswer:
    """Answer a question using only the documents stored in the vector store.

    Steps: similarity search in ChromaDB, build the prompt with the retrieved chunks, call
    the LLM asynchronously and parse the reply into a RAGAnswer.

    Raises:
        InvalidQueryError: empty or too long question.
        VectorStoreError: the store is missing or empty.
        RetrievalError: the similarity search failed.
        GenerationError: the LLM call failed even with retries and fallback.
    """
    query = validate_query(query)
    started = time.perf_counter()
    logger.info("Question: %s", query)
    documents = await retrieve(_get_store(), query)
    answer = await generate_answer(_get_chain(), documents, query)

    ungrounded = find_ungrounded_quotes(answer, documents)
    if ungrounded:
        logger.warning("Quotes not found in the retrieved context: %s", ungrounded)
    logger.info(
        "Answer ready in %.2fs (found_in_context=%s, references=%d)",
        time.perf_counter() - started, answer.found_in_context, len(answer.references),
    )
    return answer


async def _demo() -> None:
    for query in [
        "Cuanto es el deposito en garantia de un alquiler y se puede pagar en cuotas?",
        "Se pueden pagar los honorarios o el alquiler con criptomonedas?",
    ]:
        answer = await get_rag_response(query)
        print(answer.model_dump_json(indent=2))
        print()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for noisy in ("httpx", "httpcore", "huggingface_hub", "sentence_transformers", "google_genai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    asyncio.run(_demo())
