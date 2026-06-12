from __future__ import annotations

from typing import Any


def points(
    model_or_embedding: Any,
    *,
    labels: Any | None = ...,
    values: Any | None = ...,
    ax: Any | None = ...,
    s: float = ...,
    alpha: float = ...,
    cmap: str | None = ...,
    **scatter_kwargs: Any,
) -> Any:
    """Scatter a 2D embedding and return the matplotlib axes."""
