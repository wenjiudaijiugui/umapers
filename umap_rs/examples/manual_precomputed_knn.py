from __future__ import annotations

import numpy as np

from umapers import Umap


def make_dataset(n_samples: int = 96, n_features: int = 8, seed: int = 99) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n_samples, n_features)).astype(np.float32)
    x[:, 0] += np.linspace(-2.0, 2.0, n_samples, dtype=np.float32)
    x -= x.mean(axis=0, keepdims=True)
    x /= x.std(axis=0, keepdims=True) + 1e-6
    return x.astype(np.float32)


def exact_knn_graph(x: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    diffs = x[:, None, :] - x[None, :, :]
    squared = np.sum(diffs * diffs, axis=2, dtype=np.float32)
    np.fill_diagonal(squared, np.inf)

    knn_indices = np.argsort(squared, axis=1)[:, :k].astype(np.int64)
    knn_dists = np.take_along_axis(np.sqrt(squared, dtype=np.float32), knn_indices, axis=1)
    return knn_indices, knn_dists.astype(np.float32)


def main() -> None:
    np.set_printoptions(precision=4, suppress=True)

    x = make_dataset()
    k = 10
    knn_indices, knn_dists = exact_knn_graph(x, k)

    model = Umap(
        n_neighbors=k,
        n_components=2,
        n_epochs=50,
        metric="euclidean",
        init="random",
        random_seed=53,
        use_approximate_knn=False,
    )
    out = np.empty((x.shape[0], 2), dtype=np.float32)
    emb = model.fit_transform_with_knn(
        x,
        knn_indices,
        knn_dists,
        knn_metric="euclidean",
        out=out,
    )

    assert emb is out
    assert emb.shape == (x.shape[0], 2)
    assert emb.dtype == np.float32
    assert np.all(np.isfinite(emb))

    print("data shape:", x.shape, "dtype:", x.dtype)
    print("knn_indices shape:", knn_indices.shape, "dtype:", knn_indices.dtype)
    print("knn_dists shape:", knn_dists.shape, "dtype:", knn_dists.dtype)
    print("embedding shape:", emb.shape, "dtype:", emb.dtype)
    print("first row indices:", knn_indices[0])
    print("first row dists:", knn_dists[0])
    print("first row embedding:", emb[0])
    print("manual_precomputed_knn.py checks passed")


if __name__ == "__main__":
    main()
