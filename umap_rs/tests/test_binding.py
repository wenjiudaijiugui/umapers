import inspect
import importlib
from importlib import resources
import subprocess

import numpy as np

# Ensure we import the installed package, not the repo's top-level
# repo crate directory as a namespace package.
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path = [p for p in sys.path if Path(p or ".").resolve() != _REPO_ROOT]

from umapers import AlignedUmap
from umapers import ParametricUmap
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
        pytest.skip("Python package docstring/type assets require version 1 or newer")


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
        "ParametricUmap": inspect.getdoc(ParametricUmap) or "",
        "AlignedUmap": inspect.getdoc(AlignedUmap) or "",
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
    expected_files = ("py.typed", "__init__.pyi", "_api.pyi", "plot.pyi", "diagnostics.pyi")

    for filename in expected_files:
        resource = package_root / filename
        assert resource.is_file(), f"missing package resource: {filename}"


def test_import_umapers_does_not_import_optional_ecosystem_dependencies() -> None:
    code = (
        "import sys, umapers; "
        "print('matplotlib' in sys.modules); "
        "print('sklearn' in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.splitlines() == ["False", "False"]


def test_plot_points_is_optional_and_returns_axes_when_available() -> None:
    plot = importlib.import_module("umapers.plot")
    emb = np.array([[0.0, 0.1], [1.0, 1.1], [0.2, 1.3]], dtype=np.float32)

    if importlib.util.find_spec("matplotlib") is None:
        with pytest.raises(ImportError, match="umapers.plot requires matplotlib"):
            plot.points(emb)
        return

    axes = plot.points(emb, labels=np.array([0, 1, 0]))
    assert hasattr(axes, "scatter")


def test_trustworthiness_report_uses_optional_sklearn() -> None:
    diagnostics = importlib.import_module("umapers.diagnostics")
    x = make_dataset(n_samples=48, n_features=6, seed=62)
    emb = Umap(
        n_neighbors=6,
        n_components=2,
        n_epochs=20,
        init="random",
        random_seed=47,
        use_approximate_knn=False,
    ).fit_transform(x)

    if importlib.util.find_spec("sklearn") is None:
        with pytest.raises(ImportError, match="requires scikit-learn"):
            diagnostics.trustworthiness_report(x, emb, n_neighbors=5)
        return

    report = diagnostics.trustworthiness_report(x, emb, n_neighbors=5)

    assert report["n_samples"] == x.shape[0]
    assert report["n_features"] == x.shape[1]
    assert report["n_components"] == emb.shape[1]
    assert report["n_neighbors"] == 5
    assert 0.0 <= report["trustworthiness"] <= 1.0


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


def test_default_ann_params_use_quality_tuned_recall_budget() -> None:
    model = Umap()
    assert model.use_approximate_knn is True
    assert model.approx_knn_candidates == 50
    assert model.approx_knn_iters == 14

    params = model.get_params()
    assert params["approx_knn_candidates"] == 50
    assert params["approx_knn_iters"] == 14


def test_profile_fit_transform_reports_stage_timings() -> None:
    x = make_dataset(n_samples=72, n_features=6, seed=83)
    model = Umap(n_neighbors=8, n_components=2, n_epochs=20, random_seed=31, init="random")

    profiled = model.profile_fit_transform(x)
    embedding = profiled["embedding"]
    timings = profiled["timings"]

    assert embedding.shape == (x.shape[0], 2)
    assert np.all(np.isfinite(embedding))
    assert profiled["n_samples"] == x.shape[0]
    assert profiled["n_features"] == x.shape[1]
    assert profiled["n_epochs"] == 20
    assert profiled["n_edges"] > 0
    assert profiled["used_approximate_knn"] is False
    assert timings["total_sec"] > 0.0
    assert timings["knn_sec"] >= 0.0
    assert timings["optimize_sec"] >= 0.0


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


def test_knn_diagnostics_reports_ann_recall_and_sets_attribute() -> None:
    x = make_dataset(n_samples=96, n_features=8, seed=61)
    model = Umap(
        n_neighbors=8,
        n_components=2,
        n_epochs=20,
        metric="euclidean",
        init="random",
        random_seed=45,
        ann_mode="approximate",
        approx_knn_candidates=24,
        approx_knn_iters=7,
    )

    assert not hasattr(model, "knn_diagnostics_")
    diagnostics = model.knn_diagnostics(x)

    assert diagnostics is model.knn_diagnostics_
    assert diagnostics["n_samples"] == x.shape[0]
    assert diagnostics["n_features"] == x.shape[1]
    assert diagnostics["n_neighbors"] == 8
    assert diagnostics["metric"] == "euclidean"
    assert diagnostics["candidate_pool"] >= 8
    assert diagnostics["n_iters"] == 7
    assert diagnostics["mean_recall"] >= diagnostics["worst_decile_recall"]
    assert diagnostics["worst_decile_recall"] + 1e-6 >= diagnostics["min_recall"]
    assert diagnostics["mean_recall"] >= 0.60
    assert diagnostics["per_row_recall"].shape == (x.shape[0],)
    assert diagnostics["per_row_recall"].dtype == np.float32
    assert np.all((diagnostics["per_row_recall"] >= 0.0) & (diagnostics["per_row_recall"] <= 1.0))

    model.set_params(n_neighbors=9)
    assert not hasattr(model, "knn_diagnostics_")


def test_get_params_returns_constructor_arguments() -> None:
    model = Umap(
        n_neighbors=9,
        n_components=3,
        n_epochs=25,
        metric="cosine",
        init="random",
        ann_mode="exact",
        target_metric="categorical",
        target_weight=0.25,
        target_n_neighbors=5,
    )

    params = model.get_params()

    assert set(params) == set(api._PARAM_NAMES)
    assert params["n_neighbors"] == 9
    assert params["n_components"] == 3
    assert params["n_epochs"] == 25
    assert params["metric"] == "cosine"
    assert params["ann_mode"] == "exact"
    assert params["use_approximate_knn"] is False
    assert params["target_metric"] == "categorical"
    assert params["target_weight"] == 0.25
    assert params["target_n_neighbors"] == 5


def test_set_params_rebuilds_core_and_clears_fitted_attributes() -> None:
    x = make_dataset(n_samples=48, n_features=6, seed=80)
    model = Umap(
        n_neighbors=6,
        n_components=2,
        n_epochs=20,
        init="random",
        random_seed=69,
        use_approximate_knn=False,
    )
    model.fit(x)
    original_core = model._core
    assert model.n_features_in_ == x.shape[1]
    assert model.n_samples_fit_ == x.shape[0]
    assert model.embedding_.shape == (x.shape[0], 2)

    returned = model.set_params(n_neighbors=7, target_metric="categorical", target_weight=0.2)

    assert returned is model
    assert model.n_neighbors == 7
    assert model.target_metric == "categorical"
    assert model.target_weight == 0.2
    assert model._core is not original_core
    assert not hasattr(model, "embedding_")
    assert not hasattr(model, "n_features_in_")
    assert not hasattr(model, "n_samples_fit_")


def test_set_params_rejects_unknown_parameter() -> None:
    with pytest.raises(ValueError, match="Invalid parameter 'unknown'"):
        Umap().set_params(unknown=1)


def test_sklearn_clone_and_pipeline_smoke() -> None:
    sklearn_base = pytest.importorskip("sklearn.base")
    sklearn_pipeline = pytest.importorskip("sklearn.pipeline")
    sklearn_preprocessing = pytest.importorskip("sklearn.preprocessing")

    clone = sklearn_base.clone
    Pipeline = sklearn_pipeline.Pipeline
    FunctionTransformer = sklearn_preprocessing.FunctionTransformer

    x = make_dataset(n_samples=60, n_features=7, seed=81)
    model = Umap(
        n_neighbors=6,
        n_components=2,
        n_epochs=20,
        metric="minkowski",
        metric_kwds={"p": 3.0},
        init="random",
        random_seed=77,
        use_approximate_knn=False,
    )

    cloned = clone(model)
    assert isinstance(cloned, Umap)
    assert cloned.get_params() == model.get_params()
    assert not hasattr(cloned, "embedding_")

    pipeline = Pipeline(
        [
            ("umap", model),
            ("identity", FunctionTransformer(validate=False)),
        ]
    )
    emb = pipeline.fit_transform(x)

    assert emb.shape == (x.shape[0], 2)
    assert np.all(np.isfinite(emb))
    assert pipeline.get_params()["umap__n_neighbors"] == 6


def test_target_metric_none_with_y_is_noop() -> None:
    x = make_dataset(n_samples=72, n_features=8, seed=82)
    labels = np.arange(x.shape[0], dtype=np.int64) % 3
    params = dict(
        n_neighbors=8,
        n_components=2,
        n_epochs=30,
        init="random",
        random_seed=71,
        use_approximate_knn=False,
    )

    base = Umap(**params).fit_transform(x)
    with_y = Umap(**params, target_metric=None).fit_transform(x, labels)

    np.testing.assert_array_equal(with_y, base)


def test_categorical_target_weight_zero_matches_unsupervised() -> None:
    x = make_dataset(n_samples=72, n_features=8, seed=83)
    labels = np.arange(x.shape[0], dtype=np.int64) % 3
    params = dict(
        n_neighbors=8,
        n_components=2,
        n_epochs=30,
        init="random",
        random_seed=73,
        use_approximate_knn=False,
    )

    base = Umap(**params).fit_transform(x)
    supervised_zero = Umap(**params, target_metric="categorical", target_weight=0.0).fit_transform(x, labels)

    np.testing.assert_array_equal(supervised_zero, base)


def test_categorical_supervised_fit_transform_changes_embedding_and_supports_out() -> None:
    x = make_dataset(n_samples=84, n_features=8, seed=84)
    labels = np.where(np.arange(x.shape[0]) < x.shape[0] // 2, 0, 1).astype(np.int64)
    labels[::7] = -1
    params = dict(
        n_neighbors=8,
        n_components=2,
        n_epochs=30,
        init="random",
        random_seed=79,
        use_approximate_knn=False,
    )

    base = Umap(**params).fit_transform(x)
    out = np.empty((x.shape[0], 2), dtype=np.float32)
    supervised = Umap(
        **params,
        target_metric="categorical",
        target_weight=0.7,
        target_n_neighbors=4,
    ).fit_transform(x, labels, out=out)

    assert supervised is out
    assert supervised.shape == base.shape
    assert supervised.dtype == np.float32
    assert np.all(np.isfinite(supervised))
    assert not np.array_equal(supervised, base)


def test_categorical_supervised_fit_supports_dense_input() -> None:
    x = make_dataset(n_samples=48, n_features=6, seed=85)
    labels = (np.arange(x.shape[0], dtype=np.int64) % 2).astype(np.int64)

    model = Umap(
        n_neighbors=6,
        n_components=2,
        n_epochs=20,
        init="random",
        random_seed=81,
        use_approximate_knn=False,
        target_metric="categorical",
    )

    assert model.fit(x, labels) is model
    assert model._core.n_features == x.shape[1]


def test_categorical_supervised_rejects_sparse_input() -> None:
    scipy_sparse = pytest.importorskip("scipy.sparse")

    x = make_dataset(n_samples=48, n_features=6, seed=86)
    x[x < 0.1] = 0.0
    labels = np.arange(x.shape[0], dtype=np.int64) % 2
    model = Umap(
        n_neighbors=6,
        n_components=2,
        n_epochs=20,
        init="random",
        random_seed=83,
        use_approximate_knn=False,
        target_metric="categorical",
    )

    with pytest.raises(ValueError, match="categorical supervised UMAP currently supports dense input only"):
        model.fit_transform(scipy_sparse.csr_matrix(x), labels)


def test_categorical_supervised_rejects_label_length_mismatch() -> None:
    x = make_dataset(n_samples=48, n_features=6, seed=87)
    labels = np.zeros(x.shape[0] - 1, dtype=np.int64)
    model = Umap(
        n_neighbors=6,
        n_components=2,
        n_epochs=20,
        init="random",
        random_seed=85,
        use_approximate_knn=False,
        target_metric="categorical",
    )

    with pytest.raises(ValueError, match="target label length"):
        model.fit_transform(x, labels)


def test_categorical_supervised_rejects_invalid_target_parameters() -> None:
    with pytest.raises(ValueError, match="unsupported target_metric 'continuous'"):
        Umap(target_metric="continuous")

    with pytest.raises(ValueError, match="target_weight must be finite and in \\[0, 1\\]"):
        Umap(target_metric="categorical", target_weight=1.1)

    with pytest.raises(ValueError, match="target_n_neighbors must be >= 1"):
        Umap(target_metric="categorical", target_n_neighbors=0)


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
    with pytest.raises(ValueError, match="unsupported metric 'hamming'"):
        Umap(metric="hamming")

    with pytest.raises(ValueError, match="unsupported init 'pca'"):
        Umap(init="pca")


def test_expanded_dense_metrics_fit_transform_and_transform() -> None:
    x = make_dataset(n_samples=72, n_features=8, seed=88)
    query = x[:9]

    metric_specs = [
        ("chebyshev", None),
        ("minkowski", {"p": 3.0}),
        ("correlation", None),
        ("canberra", None),
        ("braycurtis", None),
    ]

    for metric, metric_kwds in metric_specs:
        model = Umap(
            n_neighbors=8,
            n_components=2,
            n_epochs=30,
            metric=metric,
            metric_kwds=metric_kwds,
            init="random",
            random_seed=89,
            use_approximate_knn=False,
        )
        emb = model.fit_transform(x)
        transformed = model.transform(query)

        assert emb.shape == (x.shape[0], 2)
        assert transformed.shape == (query.shape[0], 2)
        assert np.all(np.isfinite(emb))
        assert np.all(np.isfinite(transformed))


def test_expanded_metric_kwds_validation() -> None:
    with pytest.raises(ValueError, match="requires metric_kwds"):
        Umap(metric="minkowski")

    with pytest.raises(ValueError, match="finite and > 0"):
        Umap(metric="minkowski", metric_kwds={"p": 0.0})

    with pytest.raises(ValueError, match="unsupported metric_kwds for minkowski: w"):
        Umap(metric="minkowski", metric_kwds={"p": 2.0, "w": [1.0, 2.0]})

    with pytest.raises(ValueError, match="unsupported metric_kwds for euclidean: p"):
        Umap(metric="euclidean", metric_kwds={"p": 2.0})


def test_output_dens_returns_embedding_and_radii() -> None:
    x = make_dataset(n_samples=64, n_features=8, seed=90)
    model = Umap(
        n_neighbors=8,
        n_components=2,
        n_epochs=30,
        init="random",
        random_seed=95,
        use_approximate_knn=False,
        output_dens=True,
    )

    embedding, radii_original, radii_embedding = model.fit_transform(x)

    assert embedding.shape == (x.shape[0], 2)
    assert radii_original.shape == (x.shape[0],)
    assert radii_embedding.shape == (x.shape[0],)
    assert model.embedding_ is embedding
    assert np.array_equal(model.rad_orig_, radii_original)
    assert np.array_equal(model.rad_emb_, radii_embedding)
    assert np.array_equal(model.radii_original_, radii_original)
    assert np.array_equal(model.radii_embedding_, radii_embedding)
    assert np.all(np.isfinite(radii_original))
    assert np.all(np.isfinite(radii_embedding))


def test_swiss_roll_spectral_quality_tracks_umap_learn() -> None:
    datasets = pytest.importorskip("sklearn.datasets")
    metrics = pytest.importorskip("sklearn.metrics")
    manifold = pytest.importorskip("sklearn.manifold")
    preprocessing = pytest.importorskip("sklearn.preprocessing")
    umap_module = pytest.importorskip("umap")

    x, manifold_position = datasets.make_swiss_roll(
        n_samples=1500,
        noise=0.05,
        random_state=11,
    )
    x = preprocessing.StandardScaler().fit_transform(x).astype(np.float32)
    labels = np.digitize(
        manifold_position,
        np.quantile(manifold_position, np.linspace(0.0, 1.0, 7)[1:-1]),
    )

    kwargs = dict(
        n_neighbors=15,
        n_components=2,
        init="spectral",
        random_seed=0,
        use_approximate_knn=False,
    )
    rust_embedding = Umap(**kwargs).fit_transform(x)
    learn_embedding = umap_module.UMAP(
        n_neighbors=15,
        n_components=2,
        init="spectral",
        random_state=0,
        n_jobs=1,
    ).fit_transform(x)

    rust_silhouette = float(metrics.silhouette_score(rust_embedding, labels))
    learn_silhouette = float(metrics.silhouette_score(learn_embedding, labels))
    rust_trust = float(manifold.trustworthiness(x, rust_embedding, n_neighbors=15))
    learn_trust = float(manifold.trustworthiness(x, learn_embedding, n_neighbors=15))

    assert rust_silhouette >= 0.30
    assert rust_silhouette >= learn_silhouette - 0.08
    assert rust_trust >= learn_trust - 0.002


def test_densmap_lambda_zero_matches_standard_umap() -> None:
    x = make_dataset(n_samples=64, n_features=8, seed=92)
    params = dict(
        n_neighbors=8,
        n_components=2,
        n_epochs=30,
        init="random",
        random_seed=97,
        use_approximate_knn=False,
    )

    standard = Umap(**params).fit_transform(x)
    dens_zero = Umap(**params, densmap=True, dens_lambda=0.0).fit_transform(x)

    np.testing.assert_array_equal(dens_zero, standard)


def test_densmap_rejects_invalid_params_and_out_buffer() -> None:
    x = make_dataset(n_samples=32, n_features=6, seed=94)

    with pytest.raises(ValueError, match="dens_lambda must be finite and >= 0"):
        Umap(dens_lambda=-1.0)

    with pytest.raises(ValueError, match="dens_frac must be finite and in \\[0, 1\\]"):
        Umap(dens_frac=1.5)

    with pytest.raises(ValueError, match="dens_var_shift must be finite and > 0"):
        Umap(dens_var_shift=0.0)

    with pytest.raises(ValueError, match="out cannot be used when output_dens=True"):
        Umap(
            n_neighbors=6,
            n_components=2,
            n_epochs=20,
            init="random",
            random_seed=99,
            use_approximate_knn=False,
            output_dens=True,
        ).fit_transform(x, out=np.empty((x.shape[0], 2), dtype=np.float32))


def test_parametric_umap_fit_transform_and_transform_are_finite() -> None:
    x = make_dataset(n_samples=48, n_features=6, seed=96)
    query = x[:7]
    model = ParametricUmap(
        n_neighbors=6,
        n_components=2,
        n_epochs=20,
        hidden_dim=8,
        train_epochs=6,
        batch_size=16,
        inference_batch_size=16,
        learning_rate=0.01,
        random_seed=105,
    )

    embedding = model.fit_transform(x)
    transformed = model.transform(query)

    assert embedding.dtype == np.float32
    assert embedding.shape == (x.shape[0], 2)
    assert transformed.shape == (query.shape[0], 2)
    assert model.embedding_ is embedding
    assert model.teacher_embedding_.shape == (x.shape[0], 2)
    assert model.n_features_in_ == x.shape[1]
    assert np.all(np.isfinite(embedding))
    assert np.all(np.isfinite(transformed))


def test_parametric_umap_is_deterministic_with_fixed_seed() -> None:
    x = make_dataset(n_samples=40, n_features=5, seed=98)
    params = dict(
        n_neighbors=6,
        n_components=2,
        n_epochs=20,
        hidden_dim=8,
        train_epochs=5,
        batch_size=16,
        inference_batch_size=16,
        random_seed=107,
    )

    emb_a = ParametricUmap(**params).fit_transform(x)
    emb_b = ParametricUmap(**params).fit_transform(x)

    np.testing.assert_array_equal(emb_a, emb_b)


def test_parametric_umap_transform_requires_fit_and_rejects_invalid_params() -> None:
    x = make_dataset(n_samples=32, n_features=5, seed=100)

    with pytest.raises(RuntimeError, match="model is not fitted yet"):
        ParametricUmap(hidden_dim=8, train_epochs=4).transform(x[:3])

    with pytest.raises(ValueError, match="hidden_dim must be >= 1"):
        ParametricUmap(hidden_dim=0).fit_transform(x)

    with pytest.raises(ValueError, match="learning_rate must be finite and > 0"):
        ParametricUmap(learning_rate=0.0).fit_transform(x)


def test_aligned_umap_identity_and_explicit_relations_work() -> None:
    x0 = make_dataset(n_samples=40, n_features=6, seed=102)
    x1 = (x0 + 0.05).astype(np.float32)
    relation = np.column_stack([np.arange(0, 40, 2), np.arange(0, 40, 2)]).astype(np.int64)
    model = AlignedUmap(
        n_neighbors=6,
        n_components=2,
        n_epochs=20,
        init="random",
        random_seed=109,
        alignment_epochs=12,
    )

    identity_embeddings = model.fit_transform_identity([x0, x1])
    explicit_embeddings = model.fit_transform([x0, x1], [relation])

    assert len(identity_embeddings) == 2
    assert len(explicit_embeddings) == 2
    assert identity_embeddings[0].shape == (x0.shape[0], 2)
    assert explicit_embeddings[1].shape == (x1.shape[0], 2)
    assert np.all(np.isfinite(identity_embeddings[0]))
    assert np.all(np.isfinite(explicit_embeddings[1]))
    assert model.embeddings_[0].shape == (x0.shape[0], 2)


def test_aligned_umap_is_deterministic_with_fixed_seed() -> None:
    x0 = make_dataset(n_samples=36, n_features=5, seed=104)
    x1 = (x0 * 1.02).astype(np.float32)
    params = dict(
        n_neighbors=6,
        n_components=2,
        n_epochs=20,
        init="random",
        random_seed=111,
        alignment_epochs=10,
    )

    emb_a = AlignedUmap(**params).fit_transform_identity([x0, x1])
    emb_b = AlignedUmap(**params).fit_transform_identity([x0, x1])

    for lhs, rhs in zip(emb_a, emb_b):
        np.testing.assert_array_equal(lhs, rhs)


def test_aligned_umap_rejects_invalid_identity_and_relation_inputs() -> None:
    x0 = make_dataset(n_samples=32, n_features=5, seed=106)
    x1 = make_dataset(n_samples=30, n_features=5, seed=108)
    model = AlignedUmap(
        n_neighbors=6,
        n_components=2,
        n_epochs=20,
        init="random",
        random_seed=113,
        alignment_epochs=8,
    )

    with pytest.raises(ValueError, match="equal sample counts"):
        model.fit_transform_identity([x0, x1])

    with pytest.raises(ValueError, match="out of range"):
        bad_relation = np.array([[0, 0], [1, x0.shape[0] + 1]], dtype=np.int64)
        model.fit_transform([x0, x0], [bad_relation])

    with pytest.raises(ValueError, match="shape \\(n_pairs, 2\\)"):
        model.fit_transform([x0, x0], [np.zeros((4, 3), dtype=np.int64)])


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

    with pytest.raises(ValueError, match="unsupported metric 'hamming'"):
        model.fit_transform_with_knn(x, knn_idx, knn_dist, knn_metric="hamming")


def test_sparse_csr_expanded_metric_support_and_dense_only_rejection() -> None:
    scipy_sparse = pytest.importorskip("scipy.sparse")

    x = make_dataset(n_samples=64, n_features=12, seed=56)
    x[x < 0.15] = 0.0
    x_csr = scipy_sparse.csr_matrix(x)

    for metric, metric_kwds in [
        ("chebyshev", None),
        ("minkowski", {"p": 3.0}),
        ("correlation", None),
    ]:
        model = Umap(
            n_neighbors=7,
            n_components=2,
            n_epochs=30,
            metric=metric,
            metric_kwds=metric_kwds,
            init="random",
            random_seed=91,
            use_approximate_knn=False,
        )
        emb = model.fit_transform(x_csr)
        transformed = model.transform(x[:8])

        assert emb.shape == (x.shape[0], 2)
        assert transformed.shape == (8, 2)
        assert np.all(np.isfinite(emb))
        assert np.all(np.isfinite(transformed))

    with pytest.raises(ValueError, match="dense input only"):
        Umap(
            n_neighbors=7,
            n_components=2,
            n_epochs=20,
            metric="canberra",
            init="random",
            random_seed=93,
            use_approximate_knn=False,
        ).fit_transform(x_csr)


def test_sparse_csr_rejects_densmap_and_output_dens() -> None:
    scipy_sparse = pytest.importorskip("scipy.sparse")

    x = make_dataset(n_samples=40, n_features=8, seed=57)
    x[x < 0.15] = 0.0
    x_csr = scipy_sparse.csr_matrix(x)

    with pytest.raises(ValueError, match="dense input only"):
        Umap(
            n_neighbors=6,
            n_components=2,
            n_epochs=20,
            init="random",
            random_seed=101,
            use_approximate_knn=False,
            densmap=True,
        ).fit_transform(x_csr)

    with pytest.raises(ValueError, match="dense input only"):
        Umap(
            n_neighbors=6,
            n_components=2,
            n_epochs=20,
            init="random",
            random_seed=103,
            use_approximate_knn=False,
            output_dens=True,
        ).fit(x_csr)


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


def test_sparse_csr_fit_supports_sparse_query_transform() -> None:
    scipy_sparse = pytest.importorskip("scipy.sparse")

    x = make_dataset(n_samples=84, n_features=16, seed=58)
    x[x < 0.2] = 0.0
    x_csr = scipy_sparse.csr_matrix(x)

    model = Umap(
        n_neighbors=7,
        n_components=2,
        n_epochs=40,
        metric="cosine",
        init="random",
        random_seed=37,
        use_approximate_knn=False,
    )
    model.fit_transform(x_csr)

    dense_query = x[:11]
    csr_query = scipy_sparse.csr_matrix(dense_query)
    dense_transformed = model.transform(dense_query)
    sparse_transformed = model.transform(csr_query)

    assert sparse_transformed.shape == dense_transformed.shape
    np.testing.assert_allclose(sparse_transformed, dense_transformed, rtol=1e-6, atol=1e-6)

    out = np.empty((csr_query.shape[0], 2), dtype=np.float32)
    transformed_out = model.transform(csr_query, out=out)
    assert transformed_out is out
    np.testing.assert_allclose(transformed_out, dense_transformed, rtol=1e-6, atol=1e-6)


def test_sparse_csc_and_coo_inputs_normalize_to_csr() -> None:
    scipy_sparse = pytest.importorskip("scipy.sparse")

    x = make_dataset(n_samples=72, n_features=12, seed=59)
    x[x < 0.2] = 0.0
    x_csc = scipy_sparse.csc_matrix(x)

    model = Umap(
        n_neighbors=7,
        n_components=2,
        n_epochs=35,
        metric="euclidean",
        init="random",
        random_seed=41,
        use_approximate_knn=False,
    )
    emb = model.fit_transform(x_csc)

    query = x[:9]
    transformed_dense = model.transform(query)
    transformed_coo = model.transform(scipy_sparse.coo_matrix(query))

    assert emb.shape == (x.shape[0], 2)
    assert transformed_coo.shape == transformed_dense.shape
    assert np.all(np.isfinite(emb))
    np.testing.assert_allclose(transformed_coo, transformed_dense, rtol=1e-6, atol=1e-6)


def test_sparse_query_after_dense_fit_is_explicitly_unsupported() -> None:
    scipy_sparse = pytest.importorskip("scipy.sparse")

    x = make_dataset(n_samples=64, n_features=10, seed=60)
    model = Umap(
        n_neighbors=6,
        n_components=2,
        n_epochs=30,
        init="random",
        random_seed=43,
        use_approximate_knn=False,
    ).fit(x)

    with pytest.raises(ValueError, match="sparse query transform is supported only for sparse-fitted models"):
        model.transform(scipy_sparse.csr_matrix(x[:5]))


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
