from __future__ import annotations

import numpy as np

from umapers import Umap


def make_dataset(n_samples: int = 160, n_features: int = 10, seed: int = 123) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(n_samples, n_features)).astype(np.float32)
    trend = np.linspace(-1.0, 1.0, n_features, dtype=np.float32)
    x = base + trend
    x -= x.mean(axis=0, keepdims=True)
    x /= x.std(axis=0, keepdims=True) + 1e-6
    return x.astype(np.float32)


def main() -> None:
    np.set_printoptions(precision=4, suppress=True)

    x = make_dataset()
    model = Umap(
        n_neighbors=10,
        n_components=2,
        n_epochs=60,
        metric="euclidean",
        init="random",
        random_seed=17,
        use_approximate_knn=False,
    )
    model.fit(x)

    query = x[:20]
    transformed_out = np.empty((query.shape[0], 2), dtype=np.float32)
    transformed = model.transform(query, out=transformed_out)
    assert transformed is transformed_out
    assert transformed.shape == (query.shape[0], 2)
    assert transformed.dtype == np.float32
    assert np.all(np.isfinite(transformed))

    reconstructed_out = np.empty((query.shape[0], x.shape[1]), dtype=np.float32)
    reconstructed = model.inverse_transform(transformed, out=reconstructed_out)
    assert reconstructed is reconstructed_out
    assert reconstructed.shape == query.shape
    assert reconstructed.dtype == np.float32
    assert np.all(np.isfinite(reconstructed))

    mae = float(np.mean(np.abs(query - reconstructed)))
    print("query shape:", query.shape, "dtype:", query.dtype)
    print("transform(out=...) shape:", transformed.shape, "dtype:", transformed.dtype)
    print("inverse_transform(out=...) shape:", reconstructed.shape, "dtype:", reconstructed.dtype)
    print("mean absolute reconstruction error:", mae)
    print("manual_transform_inverse.py checks passed")


if __name__ == "__main__":
    main()
