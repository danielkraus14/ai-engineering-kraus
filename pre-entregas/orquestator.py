import asyncio
import time
import random

# Shared reference clock so every t= is relative to program start
PROGRAM_START = time.monotonic()


def elapsed():
    return time.monotonic() - PROGRAM_START


async def call_model(semaphore, call_id, model_name, duration):
    """Simulates an I/O-bound call to a model (network wait), not CPU-bound.

    Being I/O-bound, the time is spent waiting on an external response, not
    computing: that's why asyncio.sleep (non-blocking) can hand control back
    to the event loop and let other tasks progress while this one "waits".
    """
    print(f"[{call_id}] {model_name:<12} queued  at t={elapsed():6.2f}s (waiting on semaphore)")
    async with semaphore:
        start = elapsed()
        print(f"[{call_id}] {model_name:<12} START   at t={start:6.2f}s")
        await asyncio.sleep(duration)
        end = elapsed()
        print(f"[{call_id}] {model_name:<12} END     at t={end:6.2f}s (delay {end - start:.2f}s)")
        return f"[{call_id}] simulated answer for {model_name}"


async def gpt_4_call(semaphore, call_id):
    return await call_model(semaphore, call_id, "gpt_4", 1.5)


async def claude_3_call(semaphore, call_id):
    return await call_model(semaphore, call_id, "claude_3", 2)


async def local_llama_call(semaphore, call_id):
    return await call_model(semaphore, call_id, "local_llama", 3)


async def main():
    semaphore = asyncio.Semaphore(2)
    models = [gpt_4_call, claude_3_call, local_llama_call]

    # Deterministic coverage: one call per model first, guaranteeing every
    # run demonstrates all 3 cases even though the rest is randomized.
    tasks = [model(semaphore, i) for i, model in enumerate(models)]
    # Fill up to 10 calls with random models to stress the semaphore.
    tasks += [
        random.choice(models)(semaphore, i)
        for i in range(len(models), 10)
    ]

    print(f"--- Firing {len(tasks)} calls (semaphore=2, timeout=2s) ---")
    try:
        async with asyncio.timeout(2):
            results = await asyncio.gather(*tasks)
            print(f"\nCompleted without timeout. Results: {results}")
    except TimeoutError:
        print(f"\nError: one or more calls exceeded the 2s limit (t={elapsed():.2f}s)")


asyncio.run(main())