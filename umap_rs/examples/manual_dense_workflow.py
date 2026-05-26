from __future__ import annotations

import numpy as np

from umapers import Umap, fit_transform


def make_dataset(n_samples: int = 180, n_features: int = 12, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centers = np.stack(
        [
            np.linspace(-2.0, 2.0, n_features, dtype=np.float32),
            np.linspace(1.5, -1.5, n_features, dtype=np.float32),
            np.zeros(n_features, dtype=np.float32),
        ]
    )
    labels = rng.integers(0, len(centers), size=n_samples)
    noise = rng.normal(loc=0.0, scale=0.35, size=(n_samples, n_features)).astype(np.float32)
    x = centers[labels] + noise
    x -= x.mean(axis=0, keepdims=True)
    x /= x.std(axis=0, keepdims=True) + 1e-6
    return x.astype(np.float32)


def main() -> None:
    np.set_printoptions(precision=4, suppress=True)

    x = make_dataset()
    kwargs = dict(
        n_neighbors=12,
        n_components=2,
        n_epochs=80,
        metric="euclidean",
        init="random",
        random_seed=7,
        use_approximate_knn=False,
    )

    emb_top = fit_transform(x, **kwargs)
    assert emb_top.shape == (x.shape[0], 2)
    assert emb_top.dtype == np.float32
    assert np.all(np.isfinite(emb_top))

    model = Umap(**kwargs)
    out = np.empty((x.shape[0], 2), dtype=np.float32)
    emb_class = model.fit_transform(x, out=out)
    assert emb_class is out
    assert emb_class.shape == (x.shape[0], 2)
    assert emb_class.dtype == np.float32
    assert np.all(np.isfinite(emb_class))

    print("input shape:", x.shape, "dtype:", x.dtype)
    print("top-level fit_transform shape:", emb_top.shape, "dtype:", emb_top.dtype)
    print("class fit_transform(out=...) returned same buffer:", emb_class is out)
    print("top-level first row:", emb_top[0])
    print("class API first row:", emb_class[0])
    print("mean absolute embedding difference:", float(np.mean(np.abs(emb_top - emb_class))))
    print("manual_dense_workflow.py checks passed")


if __name__ == "__main__":
    main()
