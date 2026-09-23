# ai-engineering-kraus

Exercises for the AI Engineering course.

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in the API keys you need for the exercises you plan to run:

```bash
cp .env.example .env
```

## Exercises

### `orquestator.py` — Concurrent model orchestrator
Simulates concurrent calls to gpt_4, claude_3, and local_llama using `asyncio.gather`, `asyncio.Semaphore(2)` (max concurrency), and `asyncio.timeout(2)`.

```bash
python orquestator.py
```

### `client_abstraction.py` — LLM client factory
Factory Pattern abstraction over OpenAI, Anthropic, and Gemini SDKs, with Pydantic config validation and unified error handling via `LLMError`. Requires the corresponding API key(s) in `.env` for the provider set in `main()`.

```bash
python client_abstraction.py
```

### `gemini_client.py` — LCEL async chain
Declarative LangChain Expression Language chain (`ChatPromptTemplate | ChatGoogleGenerativeAI | StrOutputParser`) run asynchronously via `chain.ainvoke(...)`. Requires `GOOGLE_API_KEY` in `.env`.

```bash
python gemini_client.py
```

### `pydantic_response.py` — Structured output with retry
LCEL chain that forces the model's output to validate against a Pydantic schema (`EntityExtraction`) via `.with_structured_output()`, wrapped with `.with_retry(stop_after_attempt=3)` for resilience against transient failures. Requires `GOOGLE_API_KEY` in `.env`.

```bash
python pydantic_response.py
```

### `schemas.py` + `chain.py` — Technical entity extraction pipeline
Validated pipeline that extracts structured technical information (technologies mentioned, criticality level, technical summary) from a raw paragraph of text (e.g. an architecture description or error log).

- `schemas.py` defines `TechnicalExtraction`, a Pydantic model with `tecnologias: List[str]` (must be non-empty, enforced via `@field_validator`), `nivel_de_criticidad: CriticalityLevel` (an enum restricted to `baja`/`media`/`alta`), and `resumen_tecnico: str`.
- `chain.py` builds an LCEL chain (`prompt | model.with_structured_output(TechnicalExtraction)`) wrapped with `.with_retry(stop_after_attempt=3, wait_exponential_jitter=True)` and a `.with_fallbacks([...])` secondary model for resilience against malformed or incomplete JSON. The async `process_text(text: str)` function runs the chain via `.ainvoke()` and logs the start, successful validation, and any failure after retries are exhausted. Requires `GOOGLE_API_KEY` in `.env`.

```bash
python chain.py
```

Example output:

```json
{
  "tecnologias": ["FastAPI", "Redis", "PostgreSQL"],
  "nivel_de_criticidad": "alta",
  "resumen_tecnico": "API con caché en Redis y persistencia en PostgreSQL; cuello de botella en conexiones concurrentes."
}
```
