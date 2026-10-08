"""Deterministic tests with fake SDK clients (no network, no API keys).

They cover what cannot be forced reliably against the real APIs:
rate limiting, 5xx errors, mid-stream failures and the fallback path.

Run with:  python -m unittest discover -s tests -v
"""

import asyncio
import time
import unittest
from types import SimpleNamespace
from typing import Any

import anthropic
import httpx2
import openai
from pydantic import SecretStr, ValidationError

from clients import AnthropicClient, OpenAIClient
from manager import AsyncLLMManager
from schemas import ChatMessage, ErrorType, ModelConfig, Provider, Role

REQUEST = httpx2.Request("POST", "https://api.example.test")
KEY = SecretStr("test-key")
QUESTION = [ChatMessage(role=Role.USER, content="¿Qué es la entropía?")]


def status_error(sdk: Any, cls_name: str, status: int) -> Exception:
    cls = getattr(sdk, cls_name)
    return cls(f"HTTP {status}", response=httpx2.Response(status, request=REQUEST), body=None)


def fast_config(provider: Provider, **overrides: Any) -> ModelConfig:
    values = {"provider": provider, "model": "test-model", "backoff_base_s": 0.01, "max_retries": 3}
    return ModelConfig(**(values | overrides))


# ---- fake SDKs -----------------------------------------------------------------


class FakeOpenAIStream:
    def __init__(self, pieces: list[Any]) -> None:
        self.pieces = pieces

    async def __aiter__(self):
        for piece in self.pieces:
            if isinstance(piece, Exception):
                raise piece
            yield SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content=piece))], usage=None
            )
        yield SimpleNamespace(
            choices=[], usage=SimpleNamespace(prompt_tokens=10, completion_tokens=len(self.pieces))
        )


class FakeOpenAISDK:
    """Each call to create() consumes the next scripted outcome."""

    def __init__(self, script: list[Any], delay_s: float = 0.0) -> None:
        self.script = list(script)
        self.calls: list[dict[str, Any]] = []
        self.delay_s = delay_s
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        await asyncio.sleep(self.delay_s)
        outcome = self.script.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        if kwargs.get("stream"):
            return FakeOpenAIStream(outcome)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=outcome))],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
        )

    async def close(self) -> None:
        pass


class FakeAnthropicStream:
    def __init__(self, pieces: list[str]) -> None:
        self.pieces = pieces

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None

    @property
    async def text_stream(self):
        for piece in self.pieces:
            yield piece

    async def get_final_message(self) -> Any:
        return SimpleNamespace(usage=SimpleNamespace(input_tokens=7, output_tokens=len(self.pieces)))


class FakeAnthropicSDK:
    def __init__(self, script: list[Any]) -> None:
        self.script = list(script)
        self.calls: list[dict[str, Any]] = []
        self.messages = SimpleNamespace(create=self._create, stream=self._stream)

    async def _create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        outcome = self.script.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=outcome)],
            usage=SimpleNamespace(input_tokens=7, output_tokens=3),
        )

    def _stream(self, **kwargs: Any) -> FakeAnthropicStream:
        self.calls.append(kwargs)
        return FakeAnthropicStream(self.script.pop(0))

    async def close(self) -> None:
        pass


def openai_client(script: list[Any], **config: Any) -> tuple[OpenAIClient, FakeOpenAISDK]:
    sdk = FakeOpenAISDK(script)
    return OpenAIClient(fast_config(Provider.OPENAI, **config), api_key=KEY, sdk_client=sdk), sdk


async def collect(stream) -> list:
    return [chunk async for chunk in stream]


# ---- tests ---------------------------------------------------------------------


class TestSchemas(unittest.TestCase):
    def test_temperature_out_of_range_is_rejected(self) -> None:
        for value in (-0.1, 2.1):
            with self.assertRaises(ValidationError):
                ModelConfig(provider=Provider.OPENAI, model="m", temperature=value)

    def test_temperature_bounds_are_accepted(self) -> None:
        for value in (0, 2):
            self.assertEqual(ModelConfig(provider=Provider.OPENAI, model="m", temperature=value).temperature, value)

    def test_max_tokens_must_be_positive(self) -> None:
        with self.assertRaises(ValidationError):
            ModelConfig(provider=Provider.OPENAI, model="m", max_tokens=0)

    def test_invalid_role_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            ChatMessage(role="tool", content="hi")


class TestGenerate(unittest.IsolatedAsyncioTestCase):
    async def test_rate_limit_is_retried_then_succeeds(self) -> None:
        rate_limit = status_error(openai, "RateLimitError", 429)
        client, sdk = openai_client([rate_limit, rate_limit, "La entropía mide el desorden."])
        response = await client.generate(QUESTION)
        self.assertTrue(response.success)
        self.assertEqual(response.content, "La entropía mide el desorden.")
        self.assertEqual(len(sdk.calls), 3)

    async def test_invalid_api_key_is_not_retried(self) -> None:
        client, sdk = openai_client([status_error(openai, "AuthenticationError", 401)])
        response = await client.generate(QUESTION)
        self.assertFalse(response.success)
        self.assertEqual(response.error.type, ErrorType.AUTHENTICATION)
        self.assertEqual(response.error.attempts, 1)
        self.assertEqual(len(sdk.calls), 1)

    async def test_network_error_returns_controlled_error_after_retries(self) -> None:
        network = openai.APIConnectionError(request=REQUEST)
        client, sdk = openai_client([network] * 3, max_retries=2)
        response = await client.generate(QUESTION)
        self.assertEqual(response.error.type, ErrorType.NETWORK)
        self.assertEqual(response.error.attempts, 3)
        self.assertEqual(len(sdk.calls), 3)

    async def test_timeout_is_classified_before_network(self) -> None:
        client, _ = openai_client([openai.APITimeoutError(request=REQUEST)], max_retries=0)
        response = await client.generate(QUESTION)
        self.assertEqual(response.error.type, ErrorType.TIMEOUT)

    async def test_server_error_is_retryable(self) -> None:
        client, sdk = openai_client([status_error(openai, "InternalServerError", 503), "ok"])
        response = await client.generate(QUESTION)
        self.assertTrue(response.success)
        self.assertEqual(len(sdk.calls), 2)

    async def test_parameters_reach_the_sdk(self) -> None:
        client, sdk = openai_client(["ok"], temperature=0.2, max_tokens=99)
        await client.generate(QUESTION)
        self.assertEqual(sdk.calls[0]["temperature"], 0.2)
        self.assertEqual(sdk.calls[0]["max_completion_tokens"], 99)

    async def test_calls_do_not_block_the_event_loop(self) -> None:
        clients = [
            OpenAIClient(fast_config(Provider.OPENAI), api_key=KEY, sdk_client=FakeOpenAISDK(["ok"], delay_s=0.2))
            for _ in range(3)
        ]
        started = time.perf_counter()
        responses = await asyncio.gather(*(c.generate(QUESTION) for c in clients))
        elapsed = time.perf_counter() - started
        self.assertTrue(all(r.success for r in responses))
        # Three 0.2 s calls in sequence would take 0.6 s.
        self.assertLess(elapsed, 0.4)


class TestStream(unittest.IsolatedAsyncioTestCase):
    async def test_stream_yields_fragments_then_done_with_usage(self) -> None:
        client, sdk = openai_client([["La ", "entropía ", "mide..."]])
        chunks = await collect(client.stream(QUESTION))
        self.assertEqual([c.text for c in chunks[:-1]], ["La ", "entropía ", "mide..."])
        self.assertTrue(chunks[-1].done)
        self.assertIsNone(chunks[-1].error)
        self.assertEqual(chunks[-1].usage.output_tokens, 3)
        self.assertTrue(sdk.calls[0]["stream"])

    async def test_stream_retries_if_failure_happens_before_first_token(self) -> None:
        client, sdk = openai_client([status_error(openai, "RateLimitError", 429), ["Hola"]])
        chunks = await collect(client.stream(QUESTION))
        self.assertEqual(chunks[0].text, "Hola")
        self.assertIsNone(chunks[-1].error)
        self.assertEqual(len(sdk.calls), 2)

    async def test_stream_does_not_retry_after_text_was_emitted(self) -> None:
        client, sdk = openai_client([["Hola ", openai.APIConnectionError(request=REQUEST)]])
        chunks = await collect(client.stream(QUESTION))
        self.assertEqual(chunks[0].text, "Hola ")
        self.assertTrue(chunks[-1].done)
        self.assertEqual(chunks[-1].error.type, ErrorType.NETWORK)
        self.assertEqual(len(sdk.calls), 1)

    async def test_anthropic_stream_and_system_prompt(self) -> None:
        sdk = FakeAnthropicSDK([["Hola", " mundo"]])
        client = AnthropicClient(fast_config(Provider.ANTHROPIC), api_key=KEY, sdk_client=sdk)
        messages = [ChatMessage(role=Role.SYSTEM, content="Be brief."), *QUESTION]
        chunks = await collect(client.stream(messages))
        self.assertEqual("".join(c.text for c in chunks), "Hola mundo")
        self.assertEqual(chunks[-1].usage.input_tokens, 7)
        # Anthropic expects the system prompt as a top-level field, not as a message.
        self.assertEqual(sdk.calls[0]["system"], "Be brief.")
        self.assertEqual([m["role"] for m in sdk.calls[0]["messages"]], ["user"])

    async def test_anthropic_errors_are_classified(self) -> None:
        sdk = FakeAnthropicSDK([status_error(anthropic, "RateLimitError", 429), "ok"])
        client = AnthropicClient(fast_config(Provider.ANTHROPIC), api_key=KEY, sdk_client=sdk)
        response = await client.generate(QUESTION)
        self.assertTrue(response.success)
        self.assertEqual(len(sdk.calls), 2)


class TestManagerFallback(unittest.IsolatedAsyncioTestCase):
    def build(self, primary_script: list[Any], fallback_script: list[Any]) -> AsyncLLMManager:
        primary, _ = openai_client(primary_script)
        fallback = AnthropicClient(
            fast_config(Provider.ANTHROPIC), api_key=KEY, sdk_client=FakeAnthropicSDK(fallback_script)
        )
        return AsyncLLMManager(primary, fallback)

    async def test_generate_falls_back_when_primary_fails(self) -> None:
        manager = self.build([status_error(openai, "AuthenticationError", 401)], ["desde Anthropic"])
        response = await manager.generate("¿Qué es la entropía?")
        self.assertTrue(response.success)
        self.assertEqual(response.provider, Provider.ANTHROPIC)

    async def test_stream_falls_back_when_primary_fails_before_text(self) -> None:
        manager = self.build([status_error(openai, "AuthenticationError", 401)], [["desde ", "Anthropic"]])
        chunks = await collect(manager.stream("¿Qué es la entropía?"))
        self.assertEqual("".join(c.text for c in chunks), "desde Anthropic")
        self.assertTrue(all(c.provider is Provider.ANTHROPIC for c in chunks))
        self.assertIsNone(chunks[-1].error)

    async def test_no_fallback_returns_controlled_error(self) -> None:
        primary, _ = openai_client([status_error(openai, "AuthenticationError", 401)])
        response = await AsyncLLMManager(primary).generate("hola")
        self.assertFalse(response.success)
        self.assertEqual(response.error.provider, Provider.OPENAI)


if __name__ == "__main__":
    unittest.main()
