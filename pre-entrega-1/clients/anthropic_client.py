from collections.abc import AsyncIterator, Sequence
from typing import Any

import anthropic
from anthropic import AsyncAnthropic
from pydantic import SecretStr

from clients.base import BaseLLMClient, classify_sdk_error
from schemas import ChatMessage, ErrorType, Provider, Role, TokenUsage


class AnthropicClient(BaseLLMClient):

    provider = Provider.ANTHROPIC

    def _build_sdk_client(self, api_key: SecretStr, base_url: str | None) -> AsyncAnthropic:
        # max_retries=0 for the same reason as in OpenAIClient: retries are ours.
        return AsyncAnthropic(
            api_key=api_key.get_secret_value(),
            base_url=base_url,
            timeout=self.config.timeout_s,
            max_retries=0,
        )

    def _request_kwargs(self, messages: Sequence[ChatMessage]) -> dict[str, Any]:
        # Anthropic takes the system prompt as a top-level field, not as a message.
        system = "\n\n".join(m.content for m in messages if m.role is Role.SYSTEM)
        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "messages": [
                {"role": m.role.value, "content": m.content}
                for m in messages
                if m.role is not Role.SYSTEM
            ],
        }
        if system:
            kwargs["system"] = system
        return kwargs

    async def _complete(self, messages: Sequence[ChatMessage]) -> tuple[str, TokenUsage | None]:
        response = await self._client.messages.create(**self._request_kwargs(messages))
        text = "".join(block.text for block in response.content if block.type == "text")
        usage = TokenUsage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )
        return text, usage

    async def _stream_text(self, messages: Sequence[ChatMessage]) -> AsyncIterator[str | TokenUsage]:
        async with self._client.messages.stream(**self._request_kwargs(messages)) as stream:
            async for text in stream.text_stream:
                yield text
            final = await stream.get_final_message()
        yield TokenUsage(
            input_tokens=final.usage.input_tokens,
            output_tokens=final.usage.output_tokens,
        )

    def _classify_error(self, exc: Exception) -> ErrorType:
        return classify_sdk_error(anthropic, exc)
