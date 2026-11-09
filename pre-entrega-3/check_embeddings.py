"""Throwaway check: prove that the e5 prefixes are really applied by HuggingFaceEmbeddings.

Idea: if the "prompt" kwarg works, then embedding "text" with prompt="passage: " must give
the same vector as embedding the string "passage: text" by hand with no prompt at all.
"""
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings

MODEL_NAME = "intfloat/multilingual-e5-base"
SAMPLE = "El departamento tiene dos dormitorios, balcon y cochera cubierta."

# Normalized vectors have length 1, so the dot product equals the cosine similarity.
NORMALIZE = {"normalize_embeddings": True}


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors."""
    a_arr, b_arr = np.array(a), np.array(b)
    return float(a_arr @ b_arr / (np.linalg.norm(a_arr) * np.linalg.norm(b_arr)))


# Baseline: no prefix handling at all. We add prefixes manually when we need them.
plain = HuggingFaceEmbeddings(model_name=MODEL_NAME, encode_kwargs=NORMALIZE)

# Configured: documents get "passage: ", queries get "query: ", both applied automatically.
configured = HuggingFaceEmbeddings(
    model_name=MODEL_NAME,
    encode_kwargs={**NORMALIZE, "prompt": "passage: "},
    query_encode_kwargs={**NORMALIZE, "prompt": "query: "},
)

doc_configured = configured.embed_documents([SAMPLE])[0]
doc_manual = plain.embed_documents(["passage: " + SAMPLE])[0]
doc_no_prefix = plain.embed_documents([SAMPLE])[0]

query_configured = configured.embed_query(SAMPLE)
query_manual = plain.embed_query("query: " + SAMPLE)

print(f"model: {MODEL_NAME}")
print(f"document, configured vs manual 'passage: ' : {cosine(doc_configured, doc_manual):.6f}  (expect ~1.0)")
print(f"document, configured vs no prefix          : {cosine(doc_configured, doc_no_prefix):.6f}  (expect < 1.0)")
print(f"query,    configured vs manual 'query: '   : {cosine(query_configured, query_manual):.6f}  (expect ~1.0)")
print(f"same text as document vs as query          : {cosine(doc_configured, query_configured):.6f}  (expect < 1.0)")
