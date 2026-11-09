"""Shared settings. Ingestion and querying must import the same values from here."""

# The same embedding model must index the documents and embed the questions.
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-base"

# e5 was trained with these prefixes: "passage: " on documents, "query: " on questions.
DOCUMENT_PREFIX = "passage: "
QUERY_PREFIX = "query: "

# Number of chunks passed to the LLM. 3-5 is the recommended range: enough margin for the
# answer chunk to be included (it ranked 1st or 2nd in our tests) without flooding the prompt
# (about 4 x 460 tokens), which would risk the "lost in the middle" effect.
TOP_K = 4

# Generation models. The fallback is a lighter model from the same family, used only if the
# primary keeps failing after its retries (outage, rate limit, repeated malformed output).
PRIMARY_MODEL = "gemini-3.6-flash"
FALLBACK_MODEL = "gemini-3.5-flash-lite"
MAX_RETRY_ATTEMPTS = 3

# Longest question accepted. A real question is a sentence; a huge input is a mistake or abuse
# and would waste embedding time and LLM tokens.
MAX_QUERY_CHARS = 1000

# Where ChromaDB persists the index on disk (ignored by git).
VECTORSTORE_DIR = "vectorstore"
COLLECTION_NAME = "inmobiliaria"

# Chunk size is measured in tokens of the embedding model's own tokenizer.
# The model reads at most 512 tokens per text. A document chunk also needs room for the
# "passage: " prefix (2 tokens) and the two special tokens (2 tokens), so 500 + 4 = 504 fits.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Split on sentence ends first. With the default separators (paragraph breaks first), whole
# paragraphs longer than CHUNK_OVERLAP tokens can never be repeated, so only 4 of 18 chunk
# boundaries had real overlap. Sentence-level units make the overlap apply to 18 of 18.
# The pattern is a regex (a lookbehind keeps the punctuation attached to its sentence).
SENTENCE_SEPARATORS = [r"(?<=[.!?])\s+", " ", ""]
