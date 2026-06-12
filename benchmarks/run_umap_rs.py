#!/usr/bin/env python3
import argparse

import numpy as np

try:
    from umapers import Umap
except Exception as exc:  # pragma: no cover - environment wiring
    raise SystemExit(
        "failed to import umapers; run `maturin develop --manifest-path umap_rs/Cargo.toml` first"
    ) from exc


def parse_bool(value: str) -> bool:
    v = value.strip().lower()
    if v in {"1", "true", "yes", "y"}:
        return True
    if v in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid bool value: {value}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run umapers on CSV data")
    p.add_argument("--input", required=True, help="input CSV path")
    p.add_argument("--output", required=True, help="output embedding CSV path")
    p.add_argument("--n-neighbors", type=int, default=15)
    p.add_argument("--n-components", type=int, default=2)
    p.add_argument("--n-epochs", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--init", choices=["random", "spectral"], default="random")
    p.add_argument("--metric", choices=["euclidean", "manhattan", "cosine"], default="euclidean")
    p.add_argument("--use-approximate-knn", type=parse_bool, default=True)
    p.add_argument("--approx-knn-candidates", type=int, default=50)
    p.add_argument("--approx-knn-iters", type=int, default=14)
    p.add_argument("--approx-knn-threshold", type=int, default=4096)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    x = np.loadtxt(args.input, delimiter=",", dtype=np.float32)
    if x.ndim == 1:
        x = x.reshape(1, -1)

    model = Umap(
        n_neighbors=args.n_neighbors,
        n_components=args.n_components,
        metric=args.metric,
        n_epochs=args.n_epochs,
        learning_rate=1.0,
        init=args.init,
        min_dist=0.1,
        spread=1.0,
        set_op_mix_ratio=1.0,
        local_connectivity=1.0,
        repulsion_strength=1.0,
        negative_sample_rate=5,
        random_seed=args.seed,
        use_approximate_knn=args.use_approximate_knn,
        approx_knn_candidates=args.approx_knn_candidates,
        approx_knn_iters=args.approx_knn_iters,
        approx_knn_threshold=args.approx_knn_threshold,
    )

    embedding = model.fit_transform(x)
    np.savetxt(args.output, embedding, delimiter=",", fmt="%.8f")


if __name__ == "__main__":
    main()
