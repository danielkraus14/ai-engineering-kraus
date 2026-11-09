"""Minimal semantic search following the flowchart in the deliverable."""

from typing import Dict, List, Tuple

import numpy as np

from similarity import cosine_between, embed_texts


def search(query: str, corpus: Dict[str, str], top_k: int = 3) -> List[Tuple[str, float]]:
    """Return the top_k (doc_id, score) pairs most similar to the query.

    Steps: embed query + documents -> cosine similarity -> sort -> top_k.
    In production the document vectors are computed once and stored in a
    vector index; here they are embedded together to keep the demo small.
    """
    doc_ids = list(corpus.keys())
    vectors = embed_texts([query] + [corpus[i] for i in doc_ids])
    scores = cosine_between(vectors[0], vectors[1:])
    order = np.argsort(scores)[::-1][:top_k]
    return [(doc_ids[i], float(scores[i])) for i in order]
