"""End-to-end tests of the RAG system: questions answered in the documents and trap questions.

Each question is asked several times because the LLM does not guarantee identical output,
so a single passing run proves little. Checks avoid free-text matching where possible and
use the structured fields of the answer instead.

Run with: python test_rag.py
"""
import asyncio
import logging
import sys
import time
import unicodedata
from dataclasses import dataclass, field

from ingest import load_documents
from rag import get_rag_response
from schemas import RAGAnswer

RUNS_PER_QUESTION = 3
NO_ANSWER_MARKER = "no lo se"

logger = logging.getLogger("test_rag")


@dataclass
class TestCase:
    """One test question and what a correct answer must satisfy."""

    case_id: str
    question: str
    expects_answer: bool
    # Each inner list is a requirement; the answer must contain at least one variant of each.
    required_facts: list[list[str]] = field(default_factory=list)
    expected_source: str | None = None


TEST_CASES = [
    TestCase(
        "real-1", "Cuanto es el deposito en garantia de un alquiler y se puede pagar en cuotas?",
        True, [["un mes"], ["tres cuotas", "3 cuotas"]], "alquileres.md",
    ),
    TestCase(
        "real-2", "Cuantos dormitorios tiene el departamento DV-101 y cuanto son las expensas?",
        True, [["dos dormitorios", "2 dormitorios"], ["62.000", "62000"]], "propiedades.md",
    ),
    TestCase(
        "real-3", "Que porcentaje de honorarios paga el comprador y en que momento se pagan?",
        True, [["3 por ciento", "3%", "3 %"], ["boleto"]], "compraventa.md",
    ),
    TestCase("trap-1", "Se pueden pagar los honorarios o el alquiler con criptomonedas?", False),
    TestCase("trap-2", "La inmobiliaria ofrece servicio de mudanza para los inquilinos nuevos?", False),
]


def normalize(text: str) -> str:
    """Lowercase, remove accents and collapse whitespace, to compare texts robustly."""
    decomposed = unicodedata.normalize("NFD", text.lower())
    without_accents = "".join(c for c in decomposed if unicodedata.category(c) != "Mn")
    return " ".join(without_accents.split())


def build_corpus() -> str:
    """All the indexed text, normalized, used to verify that quotes are real."""
    return normalize(" ".join(doc.page_content for doc in load_documents()))


def check_answer(case: TestCase, answer: RAGAnswer, corpus: str) -> list[str]:
    """Return the list of failed checks (empty means the answer is correct)."""
    failures: list[str] = []
    text = normalize(answer.answer)

    if case.expects_answer:
        if not answer.found_in_context:
            failures.append("found_in_context is false but the answer is in the documents")
        for variants in case.required_facts:
            if not any(normalize(v) in text for v in variants):
                failures.append(f"answer is missing a required fact (any of {variants})")
        if case.expected_source and case.expected_source not in {r.source for r in answer.references}:
            failures.append(f"references do not cite {case.expected_source}")
    else:
        if answer.found_in_context:
            failures.append("found_in_context is true for a question not covered by the documents")
        if answer.references:
            failures.append("a refusal must not include references")
        if NO_ANSWER_MARKER not in text:
            failures.append("answer text does not say 'No lo sé'")

    # Independent of the rest: every quote must exist literally in the indexed documents.
    for ref in answer.references:
        if normalize(ref.quote) not in corpus:
            failures.append(f"invented quote not found in documents: {ref.quote[:60]!r}")
    return failures


async def run_case(case: TestCase, corpus: str) -> list[bool]:
    """Ask one question RUNS_PER_QUESTION times and return the pass/fail of each run."""
    results: list[bool] = []
    for run in range(1, RUNS_PER_QUESTION + 1):
        started = time.perf_counter()
        logger.info("[%s run %d/%d] start", case.case_id, run, RUNS_PER_QUESTION)
        answer = await get_rag_response(case.question)
        failures = check_answer(case, answer, corpus)
        status = "PASS" if not failures else "FAIL"
        logger.info(
            "[%s run %d/%d] %s in %.1fs | found_in_context=%s references=%d",
            case.case_id, run, RUNS_PER_QUESTION, status, time.perf_counter() - started,
            answer.found_in_context, len(answer.references),
        )
        logger.info("[%s run %d/%d] answer: %s", case.case_id, run, RUNS_PER_QUESTION, answer.answer[:160].replace("\n", " "))
        for failure in failures:
            logger.error("[%s run %d/%d] %s", case.case_id, run, RUNS_PER_QUESTION, failure)
        results.append(not failures)
    return results


async def main() -> int:
    corpus = build_corpus()
    summary: dict[str, list[bool]] = {}
    for case in TEST_CASES:
        summary[case.case_id] = await run_case(case, corpus)

    print("\nSummary")
    print(f"{'case':8} {'type':6} {'passed':>8}")
    for case in TEST_CASES:
        passed = sum(summary[case.case_id])
        kind = "real" if case.expects_answer else "trap"
        print(f"{case.case_id:8} {kind:6} {passed}/{RUNS_PER_QUESTION:>6}")
    all_passed = all(all(runs) for runs in summary.values())
    print("\nResult:", "ALL CHECKS PASSED" if all_passed else "SOME CHECKS FAILED")
    return 0 if all_passed else 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    # Keep only the rag.py messages about retrieval out of the way: tests log their own result.
    logging.getLogger("rag").setLevel(logging.WARNING)
    for noisy in ("httpx", "httpcore", "huggingface_hub", "sentence_transformers", "google_genai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    sys.exit(asyncio.run(main()))
