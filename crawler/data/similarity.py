import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        raise ValueError(f"Vector shape mismatch: {a.shape} vs {b.shape}")

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(np.dot(a, b) / (norm_a * norm_b))


def cosine_similarity_batch(query: np.ndarray, corpus: np.ndarray) -> np.ndarray:
    if corpus.ndim == 1:
        corpus = corpus.reshape(1, -1)

    if query.ndim == 1:
        query = query.reshape(1, -1)

    norm_query = np.linalg.norm(query)
    norm_corpus = np.linalg.norm(corpus, axis=1)

    valid_mask = (norm_query > 0) & (norm_corpus > 0)

    similarities = np.zeros(corpus.shape[0])
    similarities[valid_mask] = (
        corpus[valid_mask] @ query[0]
    ) / (norm_corpus[valid_mask] * norm_query)

    return similarities
