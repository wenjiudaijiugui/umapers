from __future__ import annotations

from typing import Any

import numpy as np


def _load_trustworthiness() -> Any:
    try:
        from sklearn.manifold import trustworthiness
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError(
            "umapers.diagnostics.trustworthiness_report requires scikit-learn"
        ) from exc
    return trustworthiness


def _as_2d_f32(x: Any, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array, got ndim={arr.ndim}")
    return arr


def _embedding_array(model_or_embedding: Any) -> np.ndarray:
    embedding = getattr(model_or_embedding, "embedding_", model_or_embedding)
    return _as_2d_f32(embedding, "embedding")


def trustworthiness_report(
    data: Any,
    model_or_embedding: Any,
    *,
    n_neighbors: int = 15,
) -> dict[str, float | int]:
    """Return a compact nearest-neighbor preservation report."""
    if n_neighbors < 1:
        raise ValueError("n_neighbors must be >= 1")

    x = _as_2d_f32(data, "data")
    emb = _embedding_array(model_or_embedding)
    if x.shape[0] != emb.shape[0]:
        raise ValueError("data and embedding row counts must match")

    trustworthiness = _load_trustworthiness()
    score = float(trustworthiness(x, emb, n_neighbors=n_neighbors))
    return {
        "trustworthiness": score,
        "n_samples": int(x.shape[0]),
        "n_features": int(x.shape[1]),
        "n_components": int(emb.shape[1]),
        "n_neighbors": int(n_neighbors),
    }
