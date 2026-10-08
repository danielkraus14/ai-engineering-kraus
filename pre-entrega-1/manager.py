"""AsyncLLMManager: picks the provider from configuration and adds a fallback.

Two layers of resilience:
- Retry (inside each client): handles transient errors of one provider
  (rate limit, network, timeout, 5xx) with exponential backoff.
- Fallback (here): if the primary provider still fails, for any reason,
  including an invalid API key, the same request goes to the other provider.
"""

import logging
import os
from collections.abc import AsyncIterator, Sequence

from pydantic import SecretStr

from clients import AnthropicClient, BaseLLMClient, OpenAIClient
from schemas import ChatMessage, ModelConfig, ModelResponse, Provider, Role, StreamChunk

logger = logging.getLogger("llm.manager")

CLIENT_REGISTRY: dict[Provider, type[BaseLLMClient]] = {
    Provider.OPENAI: OpenAIClient,
    Provider.ANTHROPIC: AnthropicClient,
}

API_KEY_ENV = {
    Provider.OPENAI: "OPENAI_API_KEY",
    Provider.ANTHROPIC: "ANTHROPIC_API_KEY",
}

DEFAULT_MODELS = {
    Provider.OPENAI: "gpt-4.1-mini",
    Provider.ANTHROPIC: "claude-haiku-4-5",
}

MODEL_ENV = {
    Provider.OPENAI: "OPENAI_MODEL",
    Provider.ANTHROPIC: "ANTHROPIC_MODEL",
}


class ConfigurationError(Exception):
    """Raised at startup when the configuration cannot produce a client."""


Prompt = str | Sequence[ChatMessage]


def build_client(config: ModelConfig, api_key: str | None, base_url: str | None = None) -> BaseLLMClient:
    """Factory: instantiate the client class registered for config.provider."""
    if not api_key:
        raise ConfigurationError(
            f"Missing API key for {config.provider.value}: set {API_KEY_ENV[config.provider]} in .env"
        )
    client_cls = CLIENT_REGISTRY[config.provider]
    return client_cls(config, api_key=SecretStr(api_key), base_url=base_url)


def config_from_env(provider: Provider) -> ModelConfig:
    """Build a validated ModelConfig for `provider` from environment variables."""
    return ModelConfig(
        provider=provider,
        model=os.getenv(MODEL_ENV[provider]) or DEFAULT_MODELS[provider],
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
        max_tokens=int(os.getenv("LLM_MAX_TOKENS", "512")),
        max_retries=int(os.getenv("LLM_MAX_RETRIES", "3")),
    )


class AsyncLLMManager:
    """Single entry point to talk to whichever provider is configured."""

    def __init__(self, primary: BaseLLMClient, fallback: BaseLLMClient | None = None) -> None:
        self.primary = primary
        self.fallback = fallback

    @classmethod
    def from_env(cls) -> "AsyncLLMManager":
        """Load LLM_PROVIDER (and optional LLM_FALLBACK_PROVIDER) from the environment."""
        primary_provider = _parse_provider("LLM_PROVIDER", os.getenv("LLM_PROVIDER", "openai"))
        primary = build_client(
            config_from_env(primary_provider), os.getenv(API_KEY_ENV[primary_provider])
        )

        fallback = None
        fallback_name = os.getenv("LLM_FALLBACK_PROVIDER", "")
        if fallback_name:
            fallback_provider = _parse_provider("LLM_FALLBACK_PROVIDER", fallback_name)
            if fallback_provider is primary_provider:
                raise ConfigurationError("LLM_FALLBACK_PROVIDER must differ from LLM_PROVIDER")
            try:
                fallback = build_client(
                    config_from_env(fallback_provider), os.getenv(API_KEY_ENV[fallback_provider])
                )
            except ConfigurationError as exc:
                # A missing fallback key should not block the primary provider.
                logger.warning("fallback disabled: %s", exc)

        logger.info(
            "manager ready primary=%s fallback=%s",
            primary.provider.value,
            fallback.provider.value if fallback else "none",
        )
        return cls(primary, fallback)

    async def generate(self, prompt: Prompt, system: str | None = None) -> ModelResponse:
        messages = _to_messages(prompt, system)
        response = await self.primary.generate(messages)
        if response.success or self.fallback is None:
            return response

        logger.warning(
            "primary %s failed (%s), switching to fallback %s",
            self.primary.provider.value,
            response.error.type.value,
            self.fallback.provider.value,
        )
        return await self.fallback.generate(messages)

    async def stream(self, prompt: Prompt, system: str | None = None) -> AsyncIterator[StreamChunk]:
        """Stream from the primary; fall back only if it failed before emitting any text."""
        messages = _to_messages(prompt, system)
        emitted = False
        async for chunk in self.primary.stream(messages):
            emitted = emitted or bool(chunk.text)
            failed_before_text = chunk.done and chunk.error and not emitted
            if failed_before_text and self.fallback is not None:
                logger.warning(
                    "primary %s stream failed (%s), switching to fallback %s",
                    self.primary.provider.value,
                    chunk.error.type.value,
                    self.fallback.provider.value,
                )
                async for fallback_chunk in self.fallback.stream(messages):
                    yield fallback_chunk
                return
            yield chunk

    async def close(self) -> None:
        await self.primary.close()
        if self.fallback:
            await self.fallback.close()


def _parse_provider(env_name: str, value: str) -> Provider:
    try:
        return Provider(value.strip().lower())
    except ValueError as exc:
        valid = ", ".join(p.value for p in Provider)
        raise ConfigurationError(f"{env_name}={value!r} is invalid, use one of: {valid}") from exc


def _to_messages(prompt: Prompt, system: str | None) -> list[ChatMessage]:
    messages = [ChatMessage(role=Role.USER, content=prompt)] if isinstance(prompt, str) else list(prompt)
    if system:
        messages.insert(0, ChatMessage(role=Role.SYSTEM, content=system))
    return messages
