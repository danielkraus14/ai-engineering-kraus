# Pre-entrega 1: Unified Async LLM Client

Cliente asíncrono unificado para **OpenAI** y **Anthropic** en Python 3.12. Ofrece una interfaz común con dos modos, respuesta completa (`generate`) y streaming (`stream`). Valida la entrada con Pydantic, reintenta con backoff exponencial y, cuando una llamada falla, devuelve un error controlado en lugar de cortar el programa.

## Estructura

```
pre-entrega-1/
├── schemas.py                 # Pydantic: ChatMessage, ModelConfig, ModelResponse, StreamChunk, LLMErrorInfo
├── clients/
│   ├── base.py                # BaseLLMClient (ABC): generate(), stream(), retry + backoff, clasificación de errores
│   ├── openai_client.py       # OpenAIClient    -> AsyncOpenAI
│   └── anthropic_client.py    # AnthropicClient -> AsyncAnthropic
├── manager.py                 # AsyncLLMManager: elige proveedor por LLM_PROVIDER + fallback opcional
├── main.py                    # Script de validación ("¿Qué es la entropía?" normal + streaming + errores)
├── tests/test_clients.py      # Tests deterministas con SDK simulado (sin red ni API keys)
├── requirements.txt
└── .env.example
```

## Instalación

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # y completar las API keys
```

## Variables de entorno

| Variable | Requerida | Descripción |
|---|---|---|
| `LLM_PROVIDER` | No (default `openai`) | Proveedor principal: `openai` o `anthropic` |
| `OPENAI_API_KEY` | Si el proveedor es `openai` | API key de OpenAI |
| `ANTHROPIC_API_KEY` | Si el proveedor es `anthropic` | API key de Anthropic |
| `LLM_FALLBACK_PROVIDER` | No | Proveedor de respaldo si el principal falla. Si falta su key, el fallback se desactiva con un warning |
| `OPENAI_MODEL` | No (default `gpt-4.1-mini`) | Modelo de OpenAI |
| `ANTHROPIC_MODEL` | No (default `claude-haiku-4-5`) | Modelo de Anthropic |
| `LLM_TEMPERATURE` | No (default `0.7`) | Validado entre 0 y 2 |
| `LLM_MAX_TOKENS` | No (default `512`) | Validado `> 0` |
| `LLM_MAX_RETRIES` | No (default `3`) | Reintentos ante errores transitorios |

## Ejecución

### Script de validación

```bash
python main.py
```

Corre cuatro secciones:

1. **Validación Pydantic**: rechaza `temperature=2.5`, `max_tokens=0`, un proveedor inexistente y un mensaje vacío.
2. **Modo normal**: `manager.generate("¿Qué es la entropía?")` con el proveedor de `LLM_PROVIDER`.
3. **Modo streaming**: `manager.stream(...)` imprime cada fragmento apenas llega y al final muestra el tiempo hasta el primer token, el tiempo total y el uso de tokens.
4. **Errores controlados en paralelo**: API key inválida en OpenAI y en Anthropic, más un host inaccesible. Esta sección **no necesita keys válidas**: muestra que ningún error rompe el programa y que el error de red se reintenta con backoff.

Si falta la key del proveedor principal, las secciones 2 y 3 se saltean con un mensaje claro y el resto del script corre igual.

Para probar con el otro proveedor sin tocar `.env`:

```bash
LLM_PROVIDER=anthropic python main.py
```

### Tests

```bash
python -m unittest discover -s tests -v
```

Son 19 tests que reemplazan el SDK por uno simulado. Así cubren de forma determinista los casos que no se pueden forzar contra la API real: rate limit (429), errores 5xx, timeout, falla a mitad del stream, fallback entre proveedores y que llamadas concurrentes no se bloqueen entre sí.

## Uso desde código

```python
from manager import AsyncLLMManager

manager = AsyncLLMManager.from_env()

response = await manager.generate("¿Qué es la entropía?")
if response.success:
    print(response.content)
else:
    print(response.error.type, response.error.message)

async for chunk in manager.stream("¿Qué es la entropía?"):
    print(chunk.text, end="", flush=True)
    if chunk.done and chunk.error:
        print("\nError:", chunk.error.type)
```

## Decisiones de diseño

- **Interfaz común (Template Method).** `BaseLLMClient` concentra la política de reintentos y el armado de respuestas. Cada proveedor implementa solo tres hooks: `_complete`, `_stream_text` y `_classify_error`. Para sumar un proveedor alcanza con crear una subclase y registrarla en `CLIENT_REGISTRY`.
- **Errores como datos, no como excepciones.** `generate()` siempre devuelve un `ModelResponse` y, si algo falla, `response.error` trae un `LLMErrorInfo` con `type`, `message`, `provider` y `attempts`. En streaming, el último `StreamChunk` (`done=True`) trae `usage` o `error`.
- **Qué se reintenta y qué no.** Se reintentan rate limit, red, timeout y 5xx, porque son transitorios. Una API key inválida o un request mal formado van a fallar igual cada vez, así que se devuelven enseguida.
- **Backoff exponencial con jitter.** La espera es `backoff_base_s * 2^(intento-1)` más hasta un 10% aleatorio, para que varias llamadas concurrentes no reintenten todas al mismo tiempo.
- **Reintentos del SDK desactivados (`max_retries=0`).** Los SDKs oficiales reintentan por su cuenta en silencio. Al desactivarlo, cada intento queda visible en los logs y la política de reintentos vive en un solo lugar.
- **Streaming: solo se reintenta antes del primer token.** Si ya se mandó texto al usuario y la conexión se corta, reintentar duplicaría el comienzo de la respuesta. En ese caso el stream termina con un error controlado.
- **Retry y fallback se complementan.** El retry cubre fallas transitorias de un proveedor. El fallback (`LLM_FALLBACK_PROVIDER`) cubre las que el retry no resuelve, como una key inválida, una cuota agotada o una caída prolongada.
- **Anthropic y `temperature`.** El SDK actual de Anthropic (1.x) ya no acepta parámetros de sampling en `messages.create`. Por eso `ModelConfig.temperature` se valida para ambos proveedores pero solo se envía a OpenAI. Además, Anthropic recibe el system prompt como campo `system` y no como un mensaje más; `AnthropicClient` hace esa conversión.

## Conceptos: por qué async

Una llamada a un LLM es **I/O-bound**: casi todo el tiempo se va en esperar la respuesta de la red mientras el modelo genera, y casi nada en usar la CPU local. `asyncio` aprovecha esas esperas: mientras una corrutina está suspendida en un `await`, el event loop atiende a las demás. Por eso la sección 4 de `main.py` resuelve tres casos en el tiempo del más lento y no en la suma de los tres.

El error clásico es usar el cliente **síncrono** (`OpenAI`, `Anthropic`) dentro de una función `async`. La llamada bloquea el hilo del event loop y frena a todas las demás corrutinas hasta que el modelo responde. Este proyecto usa solo `AsyncOpenAI` y `AsyncAnthropic`, y siempre con `await`. El test `test_calls_do_not_block_the_event_loop` lo verifica: tres llamadas de 0,2 s en paralelo tienen que terminar en menos de 0,4 s.

El **streaming** no acorta el tiempo total de generación, pero sí el tiempo hasta que el usuario ve el primer token (*time to first token*). `stream()` es un generador asíncrono: hace `yield` de cada fragmento dentro de un `async for` que recorre el stream del SDK.
