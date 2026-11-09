# Embeddings: semantic relevance vs keyword overlap

Evaluates `gemini-embedding-001` on one technical concept ("Despliegue de microservicios"):
5 paraphrases with different vocabulary, 2 keyword traps (one irrelevant, one with the opposite meaning),
compared against a TF-IDF keyword baseline. Deliverable: `analisis_embeddings.pdf`.

## Setup

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # set GOOGLE_API_KEY
```

## Run

```bash
python cosine_example.py   # scikit-learn cosine similarity, no API needed
python evaluate.py         # 3 runs, leave-one-out, invariant checks -> results.json (needs API key)
python build_pdf.py        # builds analisis_embeddings.pdf from results.json
```

## Files

- `sentences.py` - dataset (concept, relevant sentences, traps)
- `similarity.py` - Gemini embeddings with exponential-backoff retry, cosine similarity, TF-IDF baseline
- `semantic_search.py` - minimal semantic search (flowchart implementation)
- `evaluate.py` - repeated runs with timestamped logs and PASS/FAIL checks
- `build_pdf.py` - PDF generator

## Exercise 2: token-based document chunking

`code_challenge_1.py` defines `DocumentProcessor`: regex cleaning (keeps paragraph breaks) followed by
`RecursiveCharacterTextSplitter` whose `length_function` counts tokens with tiktoken (`cl100k_base`), so
`chunk_size` and `chunk_overlap` are measured in tokens, not characters. Running it prints tokens before/after
cleaning, tokens per chunk (asserted <= chunk size) and the overlap shared by consecutive chunks. No API key needed.

```bash
python code_challenge_1.py
```

`sample_text.txt` is a longer realistic Spanish text with duplicate spaces and blank-line runs, handy for manual tests.

## Exercise 3: persistent vector memory with ChromaDB

`code_challenge_2.py` defines `VectorMemoryManager`: a `PersistentClient`, a collection with an explicit default
embedding function, idempotent `upsert_documents` (no duplicate-id errors), `semantic_search` (with optional metadata
filter) and `delete_by_id`. Storage failures are wrapped in `VectorMemoryError`; invalid arguments raise `ValueError`. The demo uses a temporary directory and checks upsert idempotency, in-place updates, semantic
queries with no shared keywords, metadata filtering, persistence across two manager instances, deletion
(count drops, document leaves the search results) and forced errors (mismatched lengths, invalid metadata,
invalid filter, invalid path). No API key needed;
the first run downloads the ~80 MB ONNX embedding model (all-MiniLM-L6-v2).

```bash
python code_challenge_2.py
```
