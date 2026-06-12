from __future__ import annotations

from typing import Any

import numpy as np


def _load_pyplot() -> Any:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError("umapers.plot requires matplotlib; install `umapers[plot]`") from exc
    return plt


def _embedding_array(model_or_embedding: Any) -> np.ndarray:
    embedding = getattr(model_or_embedding, "embedding_", model_or_embedding)
    arr = np.asarray(embedding, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"embedding must be a 2D array, got ndim={arr.ndim}")
    if arr.shape[1] < 2:
        raise ValueError("embedding must have at least two columns")
    return arr


def points(
    model_or_embedding: Any,
    *,
    labels: Any | None = None,
    values: Any | None = None,
    ax: Any | None = None,
    s: float = 6.0,
    alpha: float = 0.85,
    cmap: str | None = None,
    **scatter_kwargs: Any,
) -> Any:
    """Scatter a 2D embedding and return the matplotlib axes."""
    if labels is not None and values is not None:
        raise ValueError("labels and values are mutually exclusive")

    emb = _embedding_array(model_or_embedding)
    plt = _load_pyplot()
    if ax is None:
        _, ax = plt.subplots()

    color = values if values is not None else labels
    if color is not None:
        color = np.asarray(color)
        if color.shape[0] != emb.shape[0]:
            raise ValueError("labels/values length must match embedding rows")

    ax.scatter(
        emb[:, 0],
        emb[:, 1],
        c=color,
        s=s,
        alpha=alpha,
        cmap=cmap,
        **scatter_kwargs,
    )
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    return ax
