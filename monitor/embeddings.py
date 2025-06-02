"""
embeddings.py — Embed query batches and compute distribution statistics.

Uses all-MiniLM-L6-v2 (384-dim) — same model as Projects 01, 05, 07.
Model is loaded once and reused across batches.
"""

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_queries(queries: list[str]) -> np.ndarray:
    """Returns (N, 384) float32 embedding matrix."""
    return get_model().encode(queries, convert_to_numpy=True, show_progress_bar=False)


def compute_baseline(embeddings: np.ndarray) -> dict:
    """Capture baseline distribution stats. Save this; measure everything against it."""
    return {
        "mean": embeddings.mean(axis=0).tolist(),
        "std": embeddings.std(axis=0).tolist(),
        "centroid": embeddings.mean(axis=0).tolist(),
        "n_samples": int(len(embeddings)),
        "embedding_model": MODEL_NAME,
    }


def centroid_drift(baseline_centroid: np.ndarray, current_embeddings: np.ndarray) -> float:
    """
    Cosine distance between baseline centroid and current batch centroid.
    0 = identical; 1 = orthogonal (maximum drift).
    This is a leading indicator — it moves before quality scores drop.
    """
    current_centroid = current_embeddings.mean(axis=0)
    cos_sim = np.dot(baseline_centroid, current_centroid) / (
        np.linalg.norm(baseline_centroid) * np.linalg.norm(current_centroid) + 1e-9
    )
    return float(1.0 - cos_sim)
