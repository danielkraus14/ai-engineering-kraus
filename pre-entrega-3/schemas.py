"""Pydantic models for the structured RAG answer."""
from typing import Self

from pydantic import BaseModel, Field, model_validator


class SourceReference(BaseModel):
    """One piece of evidence used to build the answer."""

    source: str = Field(description="File name of the document the evidence comes from, e.g. alquileres.md")
    quote: str = Field(
        description="Short fragment copied literally from the context that supports the answer"
    )


class RAGAnswer(BaseModel):
    """Structured answer of the RAG system.

    found_in_context is a separate boolean (instead of relying on the answer text saying
    "No lo sé") so that tests and callers can check the refusal without parsing free text.
    """

    answer: str = Field(
        description="Answer in Spanish based only on the context, or 'No lo sé' if the context lacks the information"
    )
    found_in_context: bool = Field(
        description="True only if the context contains the information needed to answer the question"
    )
    references: list[SourceReference] = Field(
        default_factory=list,
        description="Evidence supporting the answer. Must be empty when found_in_context is false",
    )

    @model_validator(mode="after")
    def check_references_match_answer(self) -> Self:
        """An answer must cite its evidence, and a refusal must not cite anything.

        This turns a hallucination pattern (claiming an answer without any support) into a
        validation error that can trigger a retry, instead of passing silently.
        """
        if self.found_in_context and not self.references:
            raise ValueError("found_in_context is true but no references were given")
        if not self.found_in_context and self.references:
            raise ValueError("found_in_context is false but references were given")
        return self
