#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from umapers import Umap


def make_clustered_data(n_samples: int, n_features: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_clusters = 4
    centers = rng.normal(0.0, 2.0, size=(n_clusters, n_features)).astype(np.float32)
    labels = rng.integers(0, n_clusters, size=n_samples)
    noise = rng.normal(0.0, 0.45, size=(n_samples, n_features)).astype(np.float32)
    data = centers[labels] + noise
    data -= data.mean(axis=0, keepdims=True)
    data /= data.std(axis=0, keepdims=True) + 1e-6
    return data.astype(np.float32)


def jsonable_report(report: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in report.items():
        if isinstance(value, np.ndarray):
            out[key] = value.astype(float).tolist()
        elif isinstance(value, np.generic):
            out[key] = value.item()
        else:
            out[key] = value
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate an exact-vs-approximate kNN recall report for umapers."
    )
    parser.add_argument("--n-samples", type=int, default=512)
    parser.add_argument("--n-features", type=int, default=16)
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--metric", default="euclidean")
    parser.add_argument("--approx-candidates", type=int, default=50)
    parser.add_argument("--approx-iters", type=int, default=14)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("benchmarks/ann_recall_report.json"))
    args = parser.parse_args()

    data = make_clustered_data(args.n_samples, args.n_features, args.seed)
    model = Umap(
        n_neighbors=args.n_neighbors,
        metric=args.metric,
        ann_mode="approximate",
        approx_knn_candidates=args.approx_candidates,
        approx_knn_iters=args.approx_iters,
        random_seed=args.seed,
        init="random",
    )
    report = jsonable_report(model.knn_diagnostics(data))
    report["dataset"] = {
        "kind": "clustered_gaussian",
        "seed": args.seed,
        "n_samples": args.n_samples,
        "n_features": args.n_features,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
