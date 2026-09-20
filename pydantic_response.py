import asyncio
from typing import List

from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

load_dotenv()

PRIMARY_MODEL = "gemini-3.6-flash"
FALLBACK_MODEL = "gemini-3.6-flash-lite"
MAX_RETRY_ATTEMPTS = 3

SYSTEM_PROMPT = "Analiza el texto y extrae las entidades."


class EntityExtraction(BaseModel):
    """Structured extraction of the main topic, named entities, and sentiment of a text."""

    topic: str = Field(description="Short label (2-5 words) naming the main subject of the text")
    entities: List[str] = Field(
        description="Named entities (people, organizations, products, technologies) mentioned in the text"
    )
    sentiment_score: float = Field(
        ge=0, le=1, description="Overall sentiment of the text, from 0 (very negative) to 1 (very positive)"
    )

    @field_validator("entities")
    @classmethod
    def entities_not_empty(cls, value: List[str]) -> List[str]:
        if not value:
            raise ValueError("entities must contain at least one item")
        return value


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


async def run_validated_chain(text: str) -> EntityExtraction | None:
    chain = build_resilient_chain()
    try:
        result = await chain.ainvoke({"input": text})
        print(result.model_dump_json(indent=2))
        return result
    except Exception as e:
        print(f"Error: {e}")
        return None


if __name__ == "__main__":
    sample_text = "LangGraph es una extensión de LangChain para agentes cíclicos."
    asyncio.run(run_validated_chain(sample_text))
