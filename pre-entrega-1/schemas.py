"""Pydantic models shared by every LLM client.

Typed models (instead of nested dicts) give us validation at the boundary:
a bad temperature or an unknown role fails fast, before any network call.
"""

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Provider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class ChatMessage(BaseModel):
    """A single message in the conversation."""

    model_config = ConfigDict(frozen=True)

    role: Role
    content: str = Field(min_length=1, description="Message text, cannot be empty")

    @field_validator("content")
    @classmethod
    def content_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("content cannot be blank")
        return value


class ModelConfig(BaseModel):
    """Generation parameters, validated before reaching the provider."""

    provider: Provider
    model: str = Field(min_length=1)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=512, gt=0, le=32_000)
    timeout_s: float = Field(default=30.0, gt=0, description="Per-request timeout")
    max_retries: int = Field(default=3, ge=0, le=10, description="Retries on transient errors")
    backoff_base_s: float = Field(default=1.0, gt=0, description="Base delay for exponential backoff")


class ErrorType(str, Enum):
    RATE_LIMIT = "rate_limit"
    NETWORK = "network"
    TIMEOUT = "timeout"
    AUTHENTICATION = "authentication"
    SERVER = "server"
    BAD_REQUEST = "bad_request"
    UNKNOWN = "unknown"


# Transient errors worth retrying; the rest will fail the same way every time.
RETRYABLE_ERRORS = frozenset(
    {ErrorType.RATE_LIMIT, ErrorType.NETWORK, ErrorType.TIMEOUT, ErrorType.SERVER}
)


class LLMErrorInfo(BaseModel):
    """Controlled error returned to the caller instead of raising."""

    type: ErrorType
    message: str
    provider: Provider
    attempts: int = Field(ge=1)

    @property
    def retryable(self) -> bool:
        return self.type in RETRYABLE_ERRORS


class TokenUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0


class ModelResponse(BaseModel):
    """Unified response for both providers. success=False means `error` is set."""

    provider: Provider
    model: str
    content: str = ""
    usage: TokenUsage | None = None
    latency_ms: float = 0.0
    error: LLMErrorInfo | None = None

    @property
    def success(self) -> bool:
        return self.error is None


class StreamChunk(BaseModel):
    """One streamed fragment. The final chunk has done=True and carries usage or error."""

    provider: Provider
    text: str = ""
    done: bool = False
    usage: TokenUsage | None = None
    error: LLMErrorInfo | None = None
