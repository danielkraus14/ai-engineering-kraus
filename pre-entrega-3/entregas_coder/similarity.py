"""Similarity helpers: cosine similarity, embeddings client and keyword baseline."""

import logging
import os
import time
from typing import List

import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

EMBEDDING_MODEL = "gemini-embedding-001"
MAX_ATTEMPTS = 4
BASE_DELAY_SECONDS = 2.0

logger = logging.getLogger(__name__)


def cosine_matrix(vectors: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity (n x n) between the rows of `vectors`."""
    return cosine_similarity(vectors)


def cosine_between(query: np.ndarray, documents: np.ndarray) -> np.ndarray:
    """Cosine similarity of one query vector against many document vectors."""
    return cosine_similarity(query.reshape(1, -1), documents)[0]


def embed_texts(texts: List[str]) -> np.ndarray:
    """Embed texts with Gemini, retrying with exponential backoff on API errors.

    Retrying matters here because the exercise is evaluated by repeated runs;
    one transient 429/503 must not invalidate a whole comparison.
    """
    load_dotenv()
    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    config = types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY")
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = client.models.embed_content(
                model=EMBEDDING_MODEL, contents=texts, config=config
            )
            return np.array([e.values for e in response.embeddings], dtype=float)
        except Exception as exc:  # network and quota errors share one policy
            if attempt == MAX_ATTEMPTS:
                raise
            delay = BASE_DELAY_SECONDS * 2 ** (attempt - 1)
            logger.warning("Embedding attempt %d failed (%s); retrying in %.1fs",
                           attempt, exc, delay)
            time.sleep(delay)
    raise RuntimeError("unreachable")


def keyword_similarity(query: str, documents: List[str]) -> np.ndarray:
    """Lexical baseline: cosine over TF-IDF vectors (counts shared words only)."""
    vectorizer = TfidfVectorizer(lowercase=True)
    matrix = vectorizer.fit_transform([query] + documents)
    return cosine_similarity(matrix[0], matrix[1:])[0]
