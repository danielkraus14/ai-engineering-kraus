"""LCEL chain: prompt | llm | PydanticOutputParser, with retry and fallback model."""
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI

from config import FALLBACK_MODEL, MAX_RETRY_ATTEMPTS, PRIMARY_MODEL
from prompts import HUMAN_PROMPT, NO_ANSWER_TEXT, SYSTEM_PROMPT
from retriever import format_context
from schemas import RAGAnswer


def build_generation_chain(model_name: str, parser: PydanticOutputParser) -> Runnable:
    """Build prompt | llm | parser for one model.

    The parser is part of the chain on purpose: the retry and fallback wrappers applied later
    then also cover malformed or invalid output, not just network errors.
    """
    prompt = ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT), ("human", HUMAN_PROMPT)]
    ).partial(
        format_instructions=parser.get_format_instructions(),
        no_answer=NO_ANSWER_TEXT,
    )
    # No temperature is set: Gemini 3.x models ignore it (fixed sampling defaults), so passing
    # one would only suggest a determinism we do not have. Repeatability is checked by running
    # the same questions several times instead.
    llm = ChatGoogleGenerativeAI(model=model_name)
    return prompt | llm | parser


def build_rag_chain(
    primary_model: str = PRIMARY_MODEL, fallback_model: str = FALLBACK_MODEL
) -> Runnable:
    """Build the full chain: format retrieved documents, generate, parse into RAGAnswer.

    Input: {"documents": [(Document, distance), ...], "question": str}.
    Resilience uses both mechanisms because they cover different failures: retry handles
    transient problems of one model (timeouts, rate limits, an invalid JSON that a second
    attempt usually fixes), and the fallback model takes over if the primary stays down.
    The model names are parameters so tests can force the fallback with a real API call.
    """
    parser = PydanticOutputParser(pydantic_object=RAGAnswer)
    retry_options = {"stop_after_attempt": MAX_RETRY_ATTEMPTS, "wait_exponential_jitter": True}
    primary = build_generation_chain(primary_model, parser).with_retry(**retry_options)
    fallback = build_generation_chain(fallback_model, parser).with_retry(**retry_options)

    return (
        RunnablePassthrough.assign(
            context=RunnableLambda(lambda inputs: format_context(inputs["documents"]))
        )
        | primary.with_fallbacks([fallback])
    )
