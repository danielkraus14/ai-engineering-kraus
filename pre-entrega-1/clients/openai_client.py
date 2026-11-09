from collections.abc import AsyncIterator, Sequence

import openai
from openai import AsyncOpenAI
from pydantic import SecretStr

from clients.base import BaseLLMClient, classify_sdk_error
from schemas import ChatMessage, ErrorType, Provider, TokenUsage


class OpenAIClient(BaseLLMClient):
    provider = Provider.OPENAI

    def _build_sdk_client(self, api_key: SecretStr, base_url: str | None) -> AsyncOpenAI:
        # max_retries=0: the SDK would otherwise retry silently on its own,
        # hiding attempts from our logs and doubling the backoff policy.
        return AsyncOpenAI(
            api_key=api_key.get_secret_value(),
            base_url=base_url,
            timeout=self.config.timeout_s,
            max_retries=0,
        )

    def _to_openai_messages(self, messages: Sequence[ChatMessage]) -> list[dict[str, str]]:
        return [{"role": m.role.value, "content": m.content} for m in messages]

    async def _complete(self, messages: Sequence[ChatMessage]) -> tuple[str, TokenUsage | None]:
        response = await self._client.chat.completions.create(
            model=self.config.model,
            messages=self._to_openai_messages(messages),
            temperature=self.config.temperature,
            max_completion_tokens=self.config.max_tokens,
        )
        usage = None
        if response.usage:
            usage = TokenUsage(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
            )
        return response.choices[0].message.content or "", usage

    async def _stream_text(self, messages: Sequence[ChatMessage]) -> AsyncIterator[str | TokenUsage]:
        stream = await self._client.chat.completions.create(
            model=self.config.model,
            messages=self._to_openai_messages(messages),
            temperature=self.config.temperature,
            max_completion_tokens=self.config.max_tokens,
            stream=True,
            # Usage only arrives in a final extra chunk when explicitly requested.
            stream_options={"include_usage": True},
        )
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
            if chunk.usage:
                yield TokenUsage(
                    input_tokens=chunk.usage.prompt_tokens,
                    output_tokens=chunk.usage.completion_tokens,
                )

    def _classify_error(self, exc: Exception) -> ErrorType:
        return classify_sdk_error(openai, exc)
