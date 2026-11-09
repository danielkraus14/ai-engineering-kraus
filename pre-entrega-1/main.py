"""Validation script for the Unified Async LLM Client.

Sections:
  1. Pydantic validation rejects invalid model parameters.
  2. Normal (non-streaming) answer from the provider set in LLM_PROVIDER.
  3. Streaming answer, printed fragment by fragment as it arrives.
  4. Controlled errors, run concurrently: invalid API keys for both providers
     and an unreachable host. Works without valid keys and never crashes.
"""

import asyncio
import logging
import sys
import time

from dotenv import load_dotenv
from pydantic import SecretStr, ValidationError

from clients import AnthropicClient, BaseLLMClient, OpenAIClient
from manager import DEFAULT_MODELS, AsyncLLMManager, ConfigurationError
from schemas import ChatMessage, ModelConfig, ModelResponse, Provider, Role

QUESTION = "¿Qué es la entropía?"
SYSTEM_PROMPT = "Respond in Spanish, in at most three sentences."
UNREACHABLE_URL = "http://127.0.0.1:9/v1"  # port 9 (discard): connection refused right away
INVALID_KEY = SecretStr("sk-invalid-key-for-demo")


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s.%(msecs)03d | %(levelname)-7s | %(name)-13s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    # SDK HTTP logs are noisy and duplicate ours.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpx2").setLevel(logging.WARNING)


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}", flush=True)


def print_response(response: ModelResponse) -> None:
    if response.success:
        print(f"[{response.provider.value}/{response.model}] {response.latency_ms:.0f} ms, usage={response.usage}")
        print(response.content)
    else:
        print(f"Controlled error -> {response.error.model_dump_json()}")


def demo_validation() -> None:
    section("1. Pydantic validation of model parameters")
    for bad in (
        {"provider": "openai", "model": "gpt-4.1-mini", "temperature": 2.5},
        {"provider": "anthropic", "model": "claude-haiku-4-5", "max_tokens": 0},
        {"provider": "gemini", "model": "x"},
    ):
        try:
            ModelConfig(**bad)
        except ValidationError as exc:
            first = exc.errors()[0]
            print(f"Rejected {bad} -> {first['loc'][0]}: {first['msg']}")
    try:
        ChatMessage(role=Role.USER, content="   ")
    except ValidationError as exc:
        print(f"Rejected blank ChatMessage -> {exc.errors()[0]['msg']}")


async def demo_generate(manager: AsyncLLMManager) -> None:
    section(f"2. Normal mode: {QUESTION}")
    response = await manager.generate(QUESTION, system=SYSTEM_PROMPT)
    print_response(response)


async def demo_stream(manager: AsyncLLMManager) -> None:
    section(f"3. Streaming mode: {QUESTION}")
    started = time.perf_counter()
    first_token_ms: float | None = None
    fragments = 0
    async for chunk in manager.stream(QUESTION, system=SYSTEM_PROMPT):
        if chunk.text:
            if first_token_ms is None:
                first_token_ms = (time.perf_counter() - started) * 1000
            fragments += 1
            print(chunk.text, end="", flush=True)
        if chunk.done:
            print()
            if chunk.error:
                print(f"Controlled error -> {chunk.error.model_dump_json()}")
            else:
                total_ms = (time.perf_counter() - started) * 1000
                print(
                    f"[{chunk.provider.value}] fragments={fragments} "
                    f"first_token_ms={first_token_ms:.0f} total_ms={total_ms:.0f} usage={chunk.usage}"
                )


async def run_case(name: str, client: BaseLLMClient) -> None:
    logger = logging.getLogger("demo")
    logger.info("case=%s start", name)
    response = await client.generate([ChatMessage(role=Role.USER, content=QUESTION)])
    logger.info("case=%s end success=%s", name, response.success)
    print(f"case={name} -> {response.error.model_dump_json() if response.error else 'ok'}")
    await client.close()


async def demo_errors() -> None:
    section("4. Controlled errors, run concurrently (none of them crashes the program)")
    fast_retries = {"max_retries": 2, "backoff_base_s": 0.5, "timeout_s": 5}
    cases = {
        "openai_invalid_key": OpenAIClient(
            ModelConfig(provider=Provider.OPENAI, model=DEFAULT_MODELS[Provider.OPENAI], **fast_retries),
            api_key=INVALID_KEY,
        ),
        "anthropic_invalid_key": AnthropicClient(
            ModelConfig(provider=Provider.ANTHROPIC, model=DEFAULT_MODELS[Provider.ANTHROPIC], **fast_retries),
            api_key=INVALID_KEY,
        ),
        "openai_network_down": OpenAIClient(
            ModelConfig(provider=Provider.OPENAI, model=DEFAULT_MODELS[Provider.OPENAI], **fast_retries),
            api_key=INVALID_KEY,
            base_url=UNREACHABLE_URL,
        ),
    }
    started = time.perf_counter()
    await asyncio.gather(*(run_case(name, client) for name, client in cases.items()))
    print(f"All error cases finished in {time.perf_counter() - started:.2f}s (concurrent, not sequential)")


async def main() -> None:
    load_dotenv()
    setup_logging()

    demo_validation()

    try:
        manager = AsyncLLMManager.from_env()
    except ConfigurationError as exc:
        section("2-3. Skipped: provider not configured")
        print(f"Configuration error: {exc}")
    else:
        try:
            await demo_generate(manager)
            await demo_stream(manager)
        finally:
            await manager.close()

    await demo_errors()


if __name__ == "__main__":
    asyncio.run(main())
