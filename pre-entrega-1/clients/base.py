"""Abstract async client with retry, backoff and controlled errors.

Subclasses only implement the provider-specific calls (`_complete`,
`_stream_text`, `_classify_error`). The resilience policy lives here once,
so both providers behave the same way when something fails.
"""

import asyncio
import logging
import random
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Sequence
from typing import Any, ClassVar

from pydantic import SecretStr

from schemas import (
    ChatMessage,
    ErrorType,
    LLMErrorInfo,
    ModelConfig,
    ModelResponse,
    Provider,
    StreamChunk,
    TokenUsage,
)

# Jitter spreads retries of concurrent requests so they do not hit the API in sync.
JITTER_RATIO = 0.1


def classify_sdk_error(sdk: Any, exc: Exception) -> ErrorType:
    """Map an SDK exception to our ErrorType.

    `openai` and `anthropic` expose the same exception names, so one function
    serves both. Order matters: APITimeoutError subclasses APIConnectionError.
    """
    if isinstance(exc, (sdk.APITimeoutError, asyncio.TimeoutError)):
        return ErrorType.TIMEOUT
    if isinstance(exc, sdk.APIConnectionError):
        return ErrorType.NETWORK
    if isinstance(exc, sdk.RateLimitError):
        return ErrorType.RATE_LIMIT
    if isinstance(exc, (sdk.AuthenticationError, sdk.PermissionDeniedError)):
        return ErrorType.AUTHENTICATION
    if isinstance(exc, sdk.APIStatusError):
        # 5xx (and Anthropic's 529 "overloaded") are transient on the provider side.
        return ErrorType.SERVER if exc.status_code >= 500 else ErrorType.BAD_REQUEST
    return ErrorType.UNKNOWN


class BaseLLMClient(ABC):
    """Common interface: `generate()` for a full answer, `stream()` for fragments."""

    provider: ClassVar[Provider]

    def __init__(
        self,
        config: ModelConfig,
        api_key: SecretStr,
        base_url: str | None = None,
        sdk_client: Any | None = None,
    ) -> None:
        if config.provider != self.provider:
            raise ValueError(f"{type(self).__name__} cannot use a {config.provider.value} config")
        self.config = config
        self.logger = logging.getLogger(f"llm.{self.provider.value}")
        # sdk_client lets tests inject a fake SDK; production builds the real async one.
        self._client = sdk_client or self._build_sdk_client(api_key, base_url)

    # ---- provider-specific hooks -------------------------------------------------

    @abstractmethod
    def _build_sdk_client(self, api_key: SecretStr, base_url: str | None) -> Any:
        """Return the official async SDK client (AsyncOpenAI / AsyncAnthropic)."""

    @abstractmethod
    async def _complete(self, messages: Sequence[ChatMessage]) -> tuple[str, TokenUsage | None]:
        """One non-streaming request. Raises SDK exceptions on failure."""

    @abstractmethod
    def _stream_text(self, messages: Sequence[ChatMessage]) -> AsyncIterator[str | TokenUsage]:
        """Yield text fragments as they arrive, then optionally a final TokenUsage."""

    @abstractmethod
    def _classify_error(self, exc: Exception) -> ErrorType:
        """Map a provider exception to an ErrorType."""

    # ---- public API ----------------------------------------------------------------

    async def generate(self, messages: Sequence[ChatMessage]) -> ModelResponse:
        """Full (non-streaming) answer. Never raises on API errors: check `response.success`."""
        started = time.perf_counter()
        attempt = 0
        while True:
            attempt += 1
            self.logger.info("generate attempt=%d model=%s start", attempt, self.config.model)
            try:
                text, usage = await self._complete(messages)
            except Exception as exc:
                error = self._to_error_info(exc, attempt)
                if not await self._should_retry(error, attempt):
                    return ModelResponse(
                        provider=self.provider,
                        model=self.config.model,
                        latency_ms=_elapsed_ms(started),
                        error=error,
                    )
                continue

            latency = _elapsed_ms(started)
            self.logger.info("generate attempt=%d end ok latency_ms=%.0f", attempt, latency)
            return ModelResponse(
                provider=self.provider,
                model=self.config.model,
                content=text,
                usage=usage,
                latency_ms=latency,
            )

    async def stream(self, messages: Sequence[ChatMessage]) -> AsyncIterator[StreamChunk]:
        """Yield StreamChunk fragments as they arrive; the last one has done=True.

        Retries only happen before the first fragment: once text reached the
        caller, retrying would duplicate it, so a mid-stream failure ends the
        stream with a controlled error instead.
        """
        attempt = 0
        while True:
            attempt += 1
            emitted = False
            usage: TokenUsage | None = None
            self.logger.info("stream attempt=%d model=%s start", attempt, self.config.model)
            try:
                async for piece in self._stream_text(messages):
                    if isinstance(piece, TokenUsage):
                        usage = piece
                        continue
                    if not emitted:
                        self.logger.info("stream attempt=%d first token received", attempt)
                    emitted = True
                    yield StreamChunk(provider=self.provider, text=piece)
            except Exception as exc:
                error = self._to_error_info(exc, attempt)
                if emitted:
                    self.logger.error("stream failed mid-response, not retrying to avoid duplicate text")
                    yield StreamChunk(provider=self.provider, done=True, error=error)
                    return
                if not await self._should_retry(error, attempt):
                    yield StreamChunk(provider=self.provider, done=True, error=error)
                    return
                continue

            self.logger.info("stream attempt=%d end ok", attempt)
            yield StreamChunk(provider=self.provider, done=True, usage=usage)
            return

    async def close(self) -> None:
        await self._client.close()

    # ---- internals -----------------------------------------------------------------

    def _to_error_info(self, exc: Exception, attempt: int) -> LLMErrorInfo:
        error_type = self._classify_error(exc)
        if error_type is ErrorType.UNKNOWN:
            # Unexpected errors are still contained, but keep the traceback for debugging.
            self.logger.exception("unexpected error on attempt=%d", attempt)
        return LLMErrorInfo(
            type=error_type,
            message=f"{type(exc).__name__}: {exc}",
            provider=self.provider,
            attempts=attempt,
        )

    async def _should_retry(self, error: LLMErrorInfo, attempt: int) -> bool:
        """Log the failure and sleep with exponential backoff if a retry makes sense."""
        if not error.retryable:
            self.logger.error("attempt=%d failed with %s (not retryable)", attempt, error.type.value)
            return False
        if attempt > self.config.max_retries:
            self.logger.error(
                "attempt=%d failed with %s, retries exhausted (max_retries=%d)",
                attempt,
                error.type.value,
                self.config.max_retries,
            )
            return False
        delay = self.config.backoff_base_s * 2 ** (attempt - 1)
        delay += random.uniform(0, delay * JITTER_RATIO)
        self.logger.warning(
            "attempt=%d failed with %s, retrying in %.2fs", attempt, error.type.value, delay
        )
        await asyncio.sleep(delay)
        return True


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000
