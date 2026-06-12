from __future__ import annotations

from typing import Any


def trustworthiness_report(
    data: Any,
    model_or_embedding: Any,
    *,
    n_neighbors: int = ...,
) -> dict[str, float | int]:
    """Return a compact nearest-neighbor preservation report."""
