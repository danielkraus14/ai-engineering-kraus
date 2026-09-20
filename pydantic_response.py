import asyncio
from typing import List

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

load_dotenv()

PRIMARY_MODEL = "gemini-3.6-flash"
FALLBACK_MODEL = "gemini-3.6-flash-lite"
MAX_RETRY_ATTEMPTS = 3

SYSTEM_PROMPT = "Analiza el texto y extrae las entidades."


class EntityExtraction(BaseModel):
    topic: str
    entities: List[str]
    sentiment_score: float = Field(ge=0, le=1)


def build_structured_llm(model_name: str) -> Runnable:
    """Wrap a chat model so its output is validated against EntityExtraction."""
    llm = ChatGoogleGenerativeAI(model=model_name)
    return llm.with_structured_output(EntityExtraction)


def build_resilient_chain() -> Runnable:
    """Compose prompt | model with retry and a fallback model for resilience.

    Retry handles transient network failures on the same model; the fallback
    covers the case where the primary model keeps failing after all retries
    (e.g. an outage), by switching to a different model still constrained
    to the same structured output schema.
    """
    primary = build_structured_llm(PRIMARY_MODEL).with_retry(
        stop_after_attempt=MAX_RETRY_ATTEMPTS
    )
    fallback = build_structured_llm(FALLBACK_MODEL).with_retry(
        stop_after_attempt=MAX_RETRY_ATTEMPTS
    )
    resilient_llm = primary.with_fallbacks([fallback])

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
    ])
    return prompt | resilient_llm


async def run_validated_chain(text: str) -> EntityExtraction | None:
    chain = build_resilient_chain()
    try:
        result = await chain.ainvoke({"input": text})
        print(result)
        return result
    except Exception as e:
        print(f"Error: {e}")
        return None


if __name__ == "__main__":
    sample_text = "LangGraph es una extensión de LangChain para agentes cíclicos."
    asyncio.run(run_validated_chain(sample_text))
