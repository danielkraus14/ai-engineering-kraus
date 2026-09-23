import asyncio

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from schemas import TechnicalExtraction
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pipeline_extraccion")

load_dotenv()

PRIMARY_MODEL = "gemini-3.6-flash"
FALLBACK_MODEL = "gemini-3.6-flash-lite"
MAX_RETRY_ATTEMPTS = 3

SYSTEM_PROMPT = "Analiza el texto y extrae las tecnologías, nivel de criticidad y resumen técnico."


def build_structured_llm(model_name: str) -> Runnable:
    """Wrap a chat model so its output is validated against TechnicalExtraction.

    include_raw=True keeps the raw AIMessage alongside the parsed result, so
    callers can inspect response_metadata (e.g. finish_reason) to detect a
    response truncated by the token limit before trusting the parsed object.
    """
    llm = ChatGoogleGenerativeAI(model=model_name)
    return llm.with_structured_output(TechnicalExtraction, include_raw=True)


def build_resilient_chain() -> Runnable:
    """Compose prompt | model with retry and a fallback model for resilience.

    Retry handles transient network failures on the same model; the fallback
    covers the case where the primary model keeps failing after all retries
    (e.g. an outage), by switching to a different model still constrained
    to the same structured output schema.
    """
    primary = build_structured_llm(PRIMARY_MODEL).with_retry(
        stop_after_attempt=MAX_RETRY_ATTEMPTS, wait_exponential_jitter=True
    )
    fallback = build_structured_llm(FALLBACK_MODEL).with_retry(
        stop_after_attempt=MAX_RETRY_ATTEMPTS, wait_exponential_jitter=True
    )
    resilient_llm = primary.with_fallbacks([fallback])

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
    ])
    return prompt | resilient_llm


async def process_text(text: str, provider: str = "gemini") -> TechnicalExtraction | None:
    chain = build_resilient_chain()
    logger.info(f"[{provider}] Procesando texto ({len(text)} caracteres)...")
    try:
        response = await chain.ainvoke({"input": text})

        finish_reason = response["raw"].response_metadata.get("finish_reason")
        if finish_reason not in (None, "STOP"):
            logger.error(f"[{provider}] Respuesta incompleta (finish_reason={finish_reason}), se descarta")
            return None

        if response["parsing_error"] is not None:
            logger.error(f"[{provider}] Error de parseo: {response['parsing_error']}")
            return None

        result = response["parsed"]
        logger.info(f"[{provider}] Extracción validada: {result.model_dump()}")
        return result
    except Exception as e:
        logger.error(f"[{provider}] Falló tras reintentos: {e}")
        return None


if __name__ == "__main__":
    sample_text = "LangGraph es una extensión de LangChain para agentes cíclicos."
    asyncio.run(process_text(sample_text))
