"""Minimal cosine similarity example with scikit-learn (deterministic, no API)."""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Two numeric vectors (e.g. embeddings with 4 dimensions for readability).
a = np.array([[0.2, 0.8, 0.1, 0.5]])
b = np.array([[0.3, 0.7, 0.0, 0.6]])

# scikit-learn expects 2D arrays: (n_samples, n_features).
sklearn_score = cosine_similarity(a, b)[0][0]

# Same value computed by hand: dot(a, b) / (||a|| * ||b||).
manual_score = float(np.dot(a[0], b[0]) / (np.linalg.norm(a) * np.linalg.norm(b)))

print(f"sklearn cosine similarity: {sklearn_score:.6f}")
print(f"manual cosine similarity:  {manual_score:.6f}")
assert np.isclose(sklearn_score, manual_score)

# Edge cases that explain how to read the score.
print("identical direction:", cosine_similarity([[1, 2]], [[2, 4]])[0][0])   # 1.0
print("orthogonal:         ", cosine_similarity([[1, 0]], [[0, 1]])[0][0])   # 0.0
print("opposite direction: ", cosine_similarity([[1, 2]], [[-1, -2]])[0][0]) # -1.0
