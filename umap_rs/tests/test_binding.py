import inspect
from importlib import resources

import numpy as np

# Ensure we import the installed package, not the repo's top-level
# repo crate directory as a namespace package.
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path = [p for p in sys.path if Path(p or ".").resolve() != _REPO_ROOT]

from umapers import Umap
from umapers import __version__
from umapers import fit_transform
import umapers._api as api
from umapers._umapers import UmapCore


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


def _skip_until_python_package_1() -> None:
    major = int(__version__.split(".", 1)[0])
    if major < 1:
        pytest.skip("1.0.0 docstring/type assets are not shipped yet")


def test_public_api_has_helpful_docstrings() -> None:
    _skip_until_python_package_1()

    docs = {
        "Umap": inspect.getdoc(Umap) or "",
        "Umap.__init__": inspect.getdoc(Umap.__init__) or "",
        "Umap.fit": inspect.getdoc(Umap.fit) or "",
        "Umap.fit_transform": inspect.getdoc(Umap.fit_transform) or "",
        "Umap.fit_transform_with_knn": inspect.getdoc(Umap.fit_transform_with_knn) or "",
        "Umap.transform": inspect.getdoc(Umap.transform) or "",
        "Umap.inverse_transform": inspect.getdoc(Umap.inverse_transform) or "",
        "fit_transform": inspect.getdoc(fit_transform) or "",
    }

    for name, doc in docs.items():
        assert doc.strip(), f"{name} docstring is empty"

    assert "shape" in docs["Umap.fit_transform"].lower()
    assert "dtype" in docs["Umap.fit_transform"].lower()
    assert "out" in docs["Umap.fit_transform"].lower()
    assert "advanced" in docs["Umap.fit_transform_with_knn"].lower()
    assert "knn" in docs["Umap.fit_transform_with_knn"].lower()
    assert "precomputed" in docs["Umap.fit_transform_with_knn"].lower()
    assert "out" in docs["Umap.transform"].lower()
    assert "out" in docs["Umap.inverse_transform"].lower()


def test_top_level_fit_transform_signature_is_inspectable() -> None:
    signature = inspect.signature(fit_transform)

    assert "data" in signature.parameters
    assert "kwargs" in signature.parameters
    assert signature.parameters["kwargs"].kind is inspect.Parameter.VAR_KEYWORD


def test_package_ships_typing_markers_and_stubs() -> None:
    _skip_until_python_package_1()

    package_root = resources.files("umapers")
    expected_files = ("py.typed", "__init__.pyi", "_api.pyi")

    for filename in expected_files:
        resource = package_root / filename
        assert resource.is_file(), f"missing package resource: {filename}"


def test_fit_transform_out_buffer_and_inverse_roundtrip() -> None:
    x = make_dataset()
    model = Umap(
        n_neighbors=12,
        n_components=2,
        n_epochs=80,
        metric="euclidean",
        init="random",
        random_seed=7,
        use_approximate_knn=False,
    )

    out = np.empty((x.shape[0], 2), dtype=np.float32)
    emb = model.fit_transform(x, out=out)

    assert emb is out
    assert emb.dtype == np.float32
    assert emb.shape == (x.shape[0], 2)
    assert np.all(np.isfinite(emb))

    query = x[:24]
    transformed = model.transform(query)
    reconstructed = model.inverse_transform(transformed)

    assert transformed.shape == (query.shape[0], 2)
    assert reconstructed.shape == query.shape
    assert np.all(np.isfinite(transformed))
    assert np.all(np.isfinite(reconstructed))


def test_ann_mode_overrides_legacy_knn_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[dict[str, object]] = []

    class FakeCore:
        def __init__(self, **kwargs: object) -> None:
            captured.append(kwargs)
            self.n_features = None

    monkeypatch.setattr(api, "UmapCore", FakeCore)

    auto = api.Umap(
        ann_mode="auto",
        use_approximate_knn=False,
        approx_knn_threshold=321,
    )
    exact = api.Umap(
        ann_mode="exact",
        use_approximate_knn=True,
        approx_knn_threshold=321,
    )
    approximate = api.Umap(
        ann_mode="approximate",
        use_approximate_knn=False,
        approx_knn_threshold=321,
    )

    assert auto.ann_mode == "auto"
    assert exact.ann_mode == "exact"
    assert approximate.ann_mode == "approximate"
    assert captured[0]["use_approximate_knn"] is False
    assert captured[0]["approx_knn_threshold"] == 321
    assert captured[1]["use_approximate_knn"] is False
    assert captured[1]["approx_knn_threshold"] == 321
    assert captured[2]["use_approximate_knn"] is True
    assert captured[2]["approx_knn_threshold"] == 0


def test_ann_mode_rejects_unknown_value() -> None:
    with pytest.raises(ValueError, match="unsupported ann_mode 'hybrid'"):
        Umap(ann_mode="hybrid")


def test_transform_and_inverse_transform_support_out_buffers() -> None:
    x = make_dataset(n_samples=120, n_features=10, seed=21)
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

    query = x[:18]
    transformed_out = np.empty((query.shape[0], 2), dtype=np.float32)
    transformed = model.transform(query, out=transformed_out)
    assert transformed is transformed_out
    assert np.all(np.isfinite(transformed))

    reconstructed_out = np.empty((query.shape[0], x.shape[1]), dtype=np.float32)
    reconstructed = model.inverse_transform(transformed, out=reconstructed_out)
    assert reconstructed is reconstructed_out
    assert reconstructed.shape == query.shape
    assert np.all(np.isfinite(reconstructed))


def test_inverse_transform_empty_input_preserves_feature_width() -> None:
    x = make_dataset(n_samples=96, n_features=11, seed=25)
    model = Umap(
        n_neighbors=10,
        n_components=2,
        n_epochs=50,
        metric="euclidean",
        init="random",
        random_seed=31,
        use_approximate_knn=False,
    )
    model.fit(x)

    empty_embedded = np.empty((0, 2), dtype=np.float32)
    reconstructed = model.inverse_transform(empty_embedded)
    assert reconstructed.shape == (0, x.shape[1])

    out = np.empty((0, x.shape[1]), dtype=np.float32)
    reconstructed_out = model.inverse_transform(empty_embedded, out=out)
    assert reconstructed_out is out
    assert reconstructed_out.shape == (0, x.shape[1])


def test_precomputed_knn_path_consistency() -> None:
    trustworthiness = pytest.importorskip("sklearn.manifold").trustworthiness
    nearest_neighbors = pytest.importorskip("sklearn.neighbors")
    NearestNeighbors = nearest_neighbors.NearestNeighbors

    x = make_dataset(n_samples=160, n_features=10, seed=123)

    k = 10
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="brute", metric="euclidean", n_jobs=1)
    nbrs.fit(x)
    dists, idx = nbrs.kneighbors(x)
    knn_idx = idx[:, 1 : k + 1].astype(np.int64)
    knn_dist = dists[:, 1 : k + 1].astype(np.float32)

    base_params = dict(
        n_neighbors=k,
        n_components=2,
        n_epochs=60,
        metric="euclidean",
        init="random",
        random_seed=11,
        use_approximate_knn=False,
    )
    model_direct = Umap(**base_params)
    model_knn = Umap(**base_params)

    emb_direct = model_direct.fit_transform(x)
    emb_knn = model_knn.fit_transform_with_knn(x, knn_idx, knn_dist, knn_metric="euclidean")

    assert emb_direct.shape == emb_knn.shape
    assert np.all(np.isfinite(emb_direct))
    assert np.all(np.isfinite(emb_knn))

    trust_direct = float(trustworthiness(x, emb_direct, n_neighbors=k))
    trust_knn = float(trustworthiness(x, emb_knn, n_neighbors=k))
    assert abs(trust_direct - trust_knn) < 0.05


def test_umap_core_precomputed_accepts_noncontiguous_knn_buffers() -> None:
    x = make_dataset(n_samples=64, n_features=6, seed=35)
    k = 8

    knn_idx_base = np.full((x.shape[0], k * 2), -1, dtype=np.int64)
    knn_idx_base[:, ::2] = np.tile(np.arange(k, dtype=np.int64), (x.shape[0], 1))
    knn_idx = knn_idx_base[:, ::2]

    knn_dist_base = np.empty((x.shape[0], k * 2), dtype=np.float32)
    knn_dist_base[:, ::2] = np.tile(np.arange(k, dtype=np.float32), (x.shape[0], 1))
    knn_dist = knn_dist_base[:, ::2]

    assert not knn_idx.flags.c_contiguous
    assert not knn_dist.flags.c_contiguous

    core = UmapCore(
        n_neighbors=k,
        n_components=2,
        n_epochs=20,
        metric="euclidean",
        init="random",
        random_seed=39,
        use_approximate_knn=False,
    )
    emb = core.fit_transform_with_knn(x, knn_idx, knn_dist, knn_metric="euclidean")

    assert emb.shape == (x.shape[0], 2)
    assert emb.dtype == np.float32
    assert np.all(np.isfinite(emb))


def test_precomputed_knn_rejects_non_finite_distances_early() -> None:
    x = make_dataset(n_samples=48, n_features=6, seed=31)
    k = 8
    knn_idx = np.tile(np.arange(k, dtype=np.int64), (x.shape[0], 1))
    knn_dist = np.ones((x.shape[0], k), dtype=np.float32)
    knn_dist[0, 0] = np.nan

    model = Umap(
        n_neighbors=k,
        n_components=2,
        n_epochs=20,
        init="random",
        random_seed=5,
        use_approximate_knn=False,
    )

    with pytest.raises(ValueError, match="knn_dists must contain only finite values"):
        model.fit_transform_with_knn(x, knn_idx, knn_dist, knn_metric="euclidean")


def test_unsupported_metric_and_init_are_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported metric 'chebyshev'"):
        Umap(metric="chebyshev")

    with pytest.raises(ValueError, match="unsupported init 'pca'"):
        Umap(init="pca")


def test_precomputed_knn_rejects_shape_mismatch_early() -> None:
    x = make_dataset(n_samples=48, n_features=6, seed=37)
    k = 8
    knn_idx = np.tile(np.arange(k, dtype=np.int64), (x.shape[0], 1))
    knn_dist = np.ones((x.shape[0], k - 1), dtype=np.float32)

    model = Umap(
        n_neighbors=k,
        n_components=2,
        n_epochs=20,
        init="random",
        random_seed=7,
        use_approximate_knn=False,
    )

    with pytest.raises(ValueError, match="knn_indices and knn_dists must have identical shapes"):
        model.fit_transform_with_knn(x, knn_idx, knn_dist, knn_metric="euclidean")


def test_precomputed_knn_rejects_negative_indices_early() -> None:
    x = make_dataset(n_samples=48, n_features=6, seed=41)
    k = 8
    knn_idx = np.tile(np.arange(k, dtype=np.int64), (x.shape[0], 1))
    knn_idx[0, 0] = -1
    knn_dist = np.ones((x.shape[0], k), dtype=np.float32)

    model = Umap(
        n_neighbors=k,
        n_components=2,
        n_epochs=20,
        init="random",
        random_seed=9,
        use_approximate_knn=False,
    )

    with pytest.raises(ValueError, match="knn indices must be non-negative integers"):
        model.fit_transform_with_knn(x, knn_idx, knn_dist, knn_metric="euclidean")


def test_non_float32_and_non_contiguous_input_is_normalized() -> None:
    x = make_dataset(n_samples=100, n_features=8, seed=5).astype(np.float64)
    x_fortran = np.asfortranarray(x)

    model = Umap(
        n_neighbors=10,
        n_components=2,
        n_epochs=40,
        metric="euclidean",
        init="random",
        random_seed=13,
        use_approximate_knn=False,
    )

    emb = model.fit_transform(x_fortran)
    assert emb.dtype == np.float32
    assert emb.flags.c_contiguous
    assert emb.shape == (x_fortran.shape[0], 2)


def test_fit_transform_rejects_invalid_out_buffers() -> None:
    x = make_dataset(n_samples=96, n_features=8, seed=15)
    model = Umap(
        n_neighbors=10,
        n_components=2,
        n_epochs=40,
        metric="euclidean",
        init="random",
        random_seed=27,
        use_approximate_knn=False,
    )

    with pytest.raises(TypeError, match="out dtype must be float32"):
        bad_dtype = np.empty((x.shape[0], 2), dtype=np.float64)
        model.fit_transform(x, out=bad_dtype)

    with pytest.raises(ValueError, match="out must be C-contiguous"):
        bad_order = np.empty((x.shape[0], 2), dtype=np.float32, order="F")
        model.fit_transform(x, out=bad_order)

    with pytest.raises(ValueError, match="out must be writeable"):
        readonly = np.empty((x.shape[0], 2), dtype=np.float32)
        readonly.setflags(write=False)
        model.fit_transform(x, out=readonly)


def test_transform_and_inverse_transform_require_fit() -> None:
    model = Umap(
        n_neighbors=10,
        n_components=2,
        n_epochs=20,
        init="random",
        random_seed=33,
        use_approximate_knn=False,
    )
    query = np.zeros((8, 6), dtype=np.float32)
    embedded = np.zeros((8, 2), dtype=np.float32)

    with pytest.raises(RuntimeError, match="model is not fitted yet"):
        model.transform(query)

    with pytest.raises(RuntimeError, match="model is not fitted yet"):
        model.inverse_transform(embedded)

    out = np.empty((8, 6), dtype=np.float32)
    with pytest.raises(RuntimeError, match="model must be fit before inverse_transform\\(out=\\.\\.\\.\\)"):
        model.inverse_transform(embedded, out=out)


def test_transform_and_inverse_transform_reject_invalid_out_buffers() -> None:
    x = make_dataset(n_samples=96, n_features=8, seed=18)
    model = Umap(
        n_neighbors=10,
        n_components=2,
        n_epochs=40,
        metric="euclidean",
        init="random",
        random_seed=37,
        use_approximate_knn=False,
    )
    model.fit(x)

    query = x[:12]
    embedded = model.transform(query)

    with pytest.raises(TypeError, match="out dtype must be float32"):
        bad_dtype = np.empty((query.shape[0], 2), dtype=np.float64)
        model.transform(query, out=bad_dtype)

    with pytest.raises(ValueError, match="out must be C-contiguous"):
        bad_order = np.empty((query.shape[0], 2), dtype=np.float32, order="F")
        model.transform(query, out=bad_order)

    with pytest.raises(ValueError, match="out must be writeable"):
        readonly = np.empty((query.shape[0], 2), dtype=np.float32)
        readonly.setflags(write=False)
        model.transform(query, out=readonly)

    with pytest.raises(TypeError, match="out dtype must be float32"):
        bad_dtype = np.empty((query.shape[0], x.shape[1]), dtype=np.float64)
        model.inverse_transform(embedded, out=bad_dtype)

    with pytest.raises(ValueError, match="out must be C-contiguous"):
        bad_order = np.empty((query.shape[0], x.shape[1]), dtype=np.float32, order="F")
        model.inverse_transform(embedded, out=bad_order)

    with pytest.raises(ValueError, match="out must be writeable"):
        readonly = np.empty((query.shape[0], x.shape[1]), dtype=np.float32)
        readonly.setflags(write=False)
        model.inverse_transform(embedded, out=readonly)


def test_zero_column_data_is_rejected_with_value_error() -> None:
    x = np.empty((16, 0), dtype=np.float32)
    model = Umap(
        n_neighbors=5,
        n_components=2,
        n_epochs=20,
        init="random",
        random_seed=3,
        use_approximate_knn=False,
    )

    with pytest.raises(ValueError, match="data must have at least one column"):
        model.fit(x)

    with pytest.raises(ValueError, match="data must have at least one column"):
        model.fit_transform(x)


def test_zero_column_precomputed_knn_is_rejected_with_value_error() -> None:
    x = make_dataset(n_samples=40, n_features=6, seed=9)
    model = Umap(
        n_neighbors=6,
        n_components=2,
        n_epochs=20,
        init="random",
        random_seed=19,
        use_approximate_knn=False,
    )

    knn_idx = np.empty((x.shape[0], 0), dtype=np.int64)
    knn_dist = np.empty((x.shape[0], 0), dtype=np.float32)

    with pytest.raises(ValueError, match="knn columns must be >= n_neighbors"):
        model.fit_transform_with_knn(x, knn_idx, knn_dist, knn_metric="euclidean")


def test_precomputed_knn_rejects_row_count_mismatch_early() -> None:
    x = make_dataset(n_samples=40, n_features=6, seed=43)
    k = 8
    knn_idx = np.tile(np.arange(k, dtype=np.int64), (x.shape[0] - 1, 1))
    knn_dist = np.ones((x.shape[0] - 1, k), dtype=np.float32)

    model = Umap(
        n_neighbors=k,
        n_components=2,
        n_epochs=20,
        init="random",
        random_seed=41,
        use_approximate_knn=False,
    )

    with pytest.raises(ValueError, match="knn row count must match data row count"):
        model.fit_transform_with_knn(x, knn_idx, knn_dist, knn_metric="euclidean")


def test_precomputed_knn_rejects_negative_distances_early() -> None:
    x = make_dataset(n_samples=48, n_features=6, seed=45)
    k = 8
    knn_idx = np.tile(np.arange(k, dtype=np.int64), (x.shape[0], 1))
    knn_dist = np.ones((x.shape[0], k), dtype=np.float32)
    knn_dist[0, 0] = -0.1

    model = Umap(
        n_neighbors=k,
        n_components=2,
        n_epochs=20,
        init="random",
        random_seed=43,
        use_approximate_knn=False,
    )

    with pytest.raises(ValueError, match="knn_dists must be non-negative"):
        model.fit_transform_with_knn(x, knn_idx, knn_dist, knn_metric="euclidean")


def test_precomputed_knn_disable_validation_still_rejects_invalid_distances_in_core() -> None:
    x = make_dataset(n_samples=48, n_features=6, seed=47)
    k = 8
    knn_idx = np.tile(np.arange(k, dtype=np.int64), (x.shape[0], 1))
    knn_dist = np.ones((x.shape[0], k), dtype=np.float32)
    knn_dist[0, 0] = -0.1

    model = Umap(
        n_neighbors=k,
        n_components=2,
        n_epochs=20,
        init="random",
        random_seed=47,
        use_approximate_knn=False,
    )

    with pytest.raises(ValueError, match="precomputed knn distance must be finite and >= 0"):
        model.fit_transform_with_knn(
            x,
            knn_idx,
            knn_dist,
            knn_metric="euclidean",
            validate_precomputed=False,
        )


def test_precomputed_knn_out_buffer_and_metric_variant_work() -> None:
    nearest_neighbors = pytest.importorskip("sklearn.neighbors")
    NearestNeighbors = nearest_neighbors.NearestNeighbors

    x = make_dataset(n_samples=120, n_features=8, seed=49)
    k = 10
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="brute", metric="manhattan", n_jobs=1)
    nbrs.fit(x)
    dists, idx = nbrs.kneighbors(x)
    knn_idx = idx[:, 1 : k + 1].astype(np.int64)
    knn_dist = dists[:, 1 : k + 1].astype(np.float32)

    model = Umap(
        n_neighbors=k,
        n_components=2,
        n_epochs=50,
        metric="manhattan",
        init="random",
        random_seed=53,
        use_approximate_knn=False,
    )
    out = np.empty((x.shape[0], 2), dtype=np.float32)
    emb = model.fit_transform_with_knn(x, knn_idx, knn_dist, knn_metric="manhattan", out=out)

    assert emb is out
    assert emb.shape == (x.shape[0], 2)
    assert np.all(np.isfinite(emb))


def test_precomputed_knn_non_contiguous_arrays_fallback_to_dense_row_path() -> None:
    nearest_neighbors = pytest.importorskip("sklearn.neighbors")
    NearestNeighbors = nearest_neighbors.NearestNeighbors

    x = make_dataset(n_samples=96, n_features=8, seed=50)
    k = 10
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="brute", metric="euclidean", n_jobs=1)
    nbrs.fit(x)
    dists, idx = nbrs.kneighbors(x)
    knn_idx = np.asfortranarray(idx[:, 1 : k + 1].astype(np.int64))
    knn_dist = np.asfortranarray(dists[:, 1 : k + 1].astype(np.float32))

    model = Umap(
        n_neighbors=k,
        n_components=2,
        n_epochs=40,
        init="random",
        random_seed=57,
        use_approximate_knn=False,
    )
    emb = model.fit_transform_with_knn(x, knn_idx, knn_dist, knn_metric="euclidean")

    assert emb.shape == (x.shape[0], 2)
    assert np.all(np.isfinite(emb))


def test_precomputed_knn_rejects_invalid_metric() -> None:
    x = make_dataset(n_samples=40, n_features=6, seed=51)
    k = 8
    knn_idx = np.tile(np.arange(k, dtype=np.int64), (x.shape[0], 1))
    knn_dist = np.ones((x.shape[0], k), dtype=np.float32)

    model = Umap(
        n_neighbors=k,
        n_components=2,
        n_epochs=20,
        init="random",
        random_seed=59,
        use_approximate_knn=False,
    )

    with pytest.raises(ValueError, match="unsupported metric 'chebyshev'"):
        model.fit_transform_with_knn(x, knn_idx, knn_dist, knn_metric="chebyshev")


def test_sparse_csr_fit_transform_and_dense_transform_out_buffer() -> None:
    scipy_sparse = pytest.importorskip("scipy.sparse")

    x = make_dataset(n_samples=90, n_features=14, seed=55)
    x[x < 0.15] = 0.0
    x_csr = scipy_sparse.csr_matrix(x)

    model = Umap(
        n_neighbors=8,
        n_components=2,
        n_epochs=50,
        metric="cosine",
        init="random",
        random_seed=23,
        use_approximate_knn=False,
    )

    emb_out = np.empty((x.shape[0], 2), dtype=np.float32)
    emb = model.fit_transform(x_csr, out=emb_out)
    assert emb is emb_out
    assert emb.shape == (x.shape[0], 2)
    assert np.all(np.isfinite(emb))

    query = x[:12]
    transformed_out = np.empty((query.shape[0], 2), dtype=np.float32)
    transformed = model.transform(query, out=transformed_out)
    assert transformed is transformed_out
    assert np.all(np.isfinite(transformed))


def test_sparse_csr_fit_tracks_feature_count_for_inverse_out_error() -> None:
    scipy_sparse = pytest.importorskip("scipy.sparse")

    x = make_dataset(n_samples=64, n_features=9, seed=71)
    x[x < 0.2] = 0.0
    x_csr = scipy_sparse.csr_matrix(x)

    model = Umap(
        n_neighbors=6,
        n_components=2,
        n_epochs=40,
        metric="manhattan",
        init="random",
        random_seed=29,
        use_approximate_knn=False,
    )
    model.fit(x_csr)

    with pytest.raises(ValueError, match="output buffer shape mismatch"):
        bad_out = np.empty((5, x.shape[1] + 1), dtype=np.float32)
        model.inverse_transform(np.zeros((5, 2), dtype=np.float32), out=bad_out)


def test_sparse_trained_inverse_transform_is_explicitly_unsupported() -> None:
    scipy_sparse = pytest.importorskip("scipy.sparse")

    x = make_dataset(n_samples=64, n_features=9, seed=73)
    x[x < 0.2] = 0.0
    x_csr = scipy_sparse.csr_matrix(x)

    model = Umap(
        n_neighbors=6,
        n_components=2,
        n_epochs=40,
        metric="manhattan",
        init="random",
        random_seed=61,
        use_approximate_knn=False,
    )
    model.fit(x_csr)

    with pytest.raises(ValueError, match="inverse_transform is not supported for sparse-trained models yet"):
        model.inverse_transform(np.zeros((5, 2), dtype=np.float32))


def test_binding_getters_track_feature_count_after_fit() -> None:
    x = make_dataset(n_samples=72, n_features=7, seed=79)
    model = Umap(
        n_neighbors=9,
        n_components=2,
        n_epochs=30,
        init="random",
        random_seed=67,
        use_approximate_knn=False,
    )

    assert model.n_neighbors == 9
    assert model.n_components == 2
    assert model._core.n_features is None

    model.fit(x)
    assert model._core.n_features == x.shape[1]
