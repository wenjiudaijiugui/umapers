from __future__ import annotations

from typing import Any, TypedDict

import numpy as np

from ._umapers import AlignedUmapCore, ParametricUmapCore, UmapCore

class _BaseEstimator:
    pass


class _TransformerMixin:
    pass


_PARAM_NAMES = (
    "n_neighbors",
    "n_components",
    "n_epochs",
    "metric",
    "metric_kwds",
    "learning_rate",
    "min_dist",
    "spread",
    "local_connectivity",
    "set_op_mix_ratio",
    "repulsion_strength",
    "negative_sample_rate",
    "random_seed",
    "init",
    "ann_mode",
    "use_approximate_knn",
    "approx_knn_candidates",
    "approx_knn_iters",
    "approx_knn_threshold",
    "densmap",
    "dens_lambda",
    "dens_frac",
    "dens_var_shift",
    "output_dens",
    "target_metric",
    "target_weight",
    "target_n_neighbors",
)

_PARAMETRIC_PARAM_NAMES = (
    "n_neighbors",
    "n_components",
    "n_epochs",
    "metric",
    "metric_kwds",
    "hidden_dim",
    "train_epochs",
    "batch_size",
    "inference_batch_size",
    "learning_rate",
    "weight_decay",
    "pairwise_loss_weight",
    "pairwise_pairs_per_batch",
    "standardize_input",
    "random_seed",
    "train_mode",
)

_ALIGNED_PARAM_NAMES = (
    "n_neighbors",
    "n_components",
    "n_epochs",
    "metric",
    "metric_kwds",
    "random_seed",
    "init",
    "alignment_regularization",
    "alignment_learning_rate",
    "alignment_epochs",
    "recenter_interval",
)


class UmapKwargs(TypedDict, total=False):
    """Keyword arguments accepted by `Umap(...)` and `fit_transform(..., **kwargs)`."""

    n_neighbors: int
    n_components: int
    n_epochs: int | None
    metric: str
    metric_kwds: dict[str, Any] | None
    learning_rate: float
    min_dist: float
    spread: float
    local_connectivity: float
    set_op_mix_ratio: float
    repulsion_strength: float
    negative_sample_rate: int
    random_seed: int
    init: str
    ann_mode: str
    use_approximate_knn: bool
    approx_knn_candidates: int
    approx_knn_iters: int
    approx_knn_threshold: int
    densmap: bool
    dens_lambda: float
    dens_frac: float
    dens_var_shift: float
    output_dens: bool
    target_metric: str | None
    target_weight: float
    target_n_neighbors: int | None


def _as_f32_matrix(x: Any, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32, order="C")
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array, got ndim={arr.ndim}")
    return arr


def _as_knn_indices(x: Any, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=np.int64, order="C")
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array, got ndim={arr.ndim}")
    return arr


def _as_labels(x: Any, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=np.int64, order="C")
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D label array, got ndim={arr.ndim}")
    return arr


def _as_relation_array(x: Any, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=np.int64, order="C")
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"{name} must have shape (n_pairs, 2)")
    if np.any(arr < 0):
        raise ValueError(f"{name} indices must be non-negative")
    return arr


def _maybe_as_csr_parts(x: Any, name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int] | None:
    fmt = getattr(x, "format", None)
    if fmt == "csr":
        matrix = x
    elif fmt in {"csc", "coo"}:
        tocsr = getattr(x, "tocsr", None)
        if not callable(tocsr):
            raise ValueError(f"{name} sparse input with format={fmt!r} cannot be converted to CSR")
        matrix = tocsr()
        if getattr(matrix, "format", None) != "csr":
            raise ValueError(f"{name}.tocsr() must return a CSR matrix")
    else:
        return None

    if getattr(matrix, "has_canonical_format", True) is False:
        copy = getattr(matrix, "copy", None)
        if callable(copy):
            matrix = copy()
        sum_duplicates = getattr(matrix, "sum_duplicates", None)
        if callable(sum_duplicates):
            sum_duplicates()
        sort_indices = getattr(matrix, "sort_indices", None)
        if callable(sort_indices):
            sort_indices()
    elif getattr(matrix, "has_sorted_indices", True) is False:
        copy = getattr(matrix, "copy", None)
        if callable(copy):
            matrix = copy()
        sort_indices = getattr(matrix, "sort_indices", None)
        if callable(sort_indices):
            sort_indices()

    shape = getattr(matrix, "shape", None)
    if shape is None or len(shape) != 2:
        raise ValueError(f"{name} must be a 2D CSR matrix")

    n_rows, n_cols = int(shape[0]), int(shape[1])
    if n_cols <= 0:
        raise ValueError(f"{name} must have at least one column")

    indptr = np.asarray(matrix.indptr, dtype=np.int64, order="C")
    indices = np.asarray(matrix.indices, dtype=np.int64, order="C")
    data = np.asarray(matrix.data, dtype=np.float32, order="C")
    if indptr.ndim != 1 or indices.ndim != 1 or data.ndim != 1:
        raise ValueError(f"{name} CSR arrays must be 1D")
    if indices.shape[0] != data.shape[0]:
        raise ValueError(f"{name} CSR indices/data length mismatch")
    return indptr, indices, data, n_rows, n_cols


def _as_out_buffer(out: Any, shape: tuple[int, int]) -> np.ndarray:
    if not isinstance(out, np.ndarray):
        raise TypeError("out must be a NumPy ndarray")
    if out.dtype != np.float32:
        raise TypeError(f"out dtype must be float32, got {out.dtype}")
    if out.ndim != 2:
        raise ValueError(f"out must be 2D, got ndim={out.ndim}")
    if not out.flags.c_contiguous:
        raise ValueError("out must be C-contiguous")
    if not out.flags.writeable:
        raise ValueError("out must be writeable")
    if tuple(out.shape) != tuple(shape):
        raise ValueError(f"output buffer shape mismatch: expected {shape}, got {tuple(out.shape)}")
    return out


def _normalize_ann_mode(
    ann_mode: Any,
    use_approximate_knn: bool,
    approx_knn_threshold: int,
) -> tuple[str, bool, int]:
    if isinstance(ann_mode, str) and ann_mode in {"auto", "exact", "approximate"}:
        mode = ann_mode
    else:
        mode = str(ann_mode).lower()
    if mode == "auto":
        return mode, use_approximate_knn, approx_knn_threshold
    if mode == "exact":
        return mode, False, approx_knn_threshold
    if mode == "approximate":
        return mode, True, 0
    raise ValueError(f"unsupported ann_mode '{ann_mode}', expected auto|exact|approximate")


def _normalize_target_metric(target_metric: Any) -> str | None:
    if target_metric is None:
        return None
    if isinstance(target_metric, str) and target_metric == "categorical":
        metric = target_metric
    else:
        metric = str(target_metric).lower()
    if metric in {"none", ""}:
        return None
    if metric == "categorical":
        return metric
    raise ValueError(f"unsupported target_metric '{target_metric}', expected None|categorical")


def _normalize_metric_name(metric: Any) -> str:
    canonical = {
        "euclidean": "euclidean",
        "l2": "euclidean",
        "manhattan": "manhattan",
        "l1": "manhattan",
        "cosine": "cosine",
        "chebyshev": "chebyshev",
        "linfinity": "chebyshev",
        "linf": "chebyshev",
        "minkowski": "minkowski",
        "correlation": "correlation",
        "canberra": "canberra",
        "braycurtis": "braycurtis",
        "bray_curtis": "braycurtis",
    }
    if isinstance(metric, str) and metric in canonical and canonical[metric] == metric:
        return metric
    name = str(metric).lower()
    if name in canonical:
        return canonical[name]
    expected = "|".join(sorted(set(canonical.values())))
    raise ValueError(f"unsupported metric '{metric}', expected {expected}")


def _parse_metric_config(metric: Any, metric_kwds: dict[str, Any] | None) -> tuple[str, dict[str, Any] | None, float | None]:
    metric_name = _normalize_metric_name(metric)
    kwds = {} if metric_kwds is None else metric_kwds
    if not hasattr(kwds, "keys"):
        raise TypeError("metric_kwds must be a mapping or None")

    keys = set(kwds.keys())
    if metric_name == "minkowski":
        if "p" not in keys:
            raise ValueError("metric='minkowski' requires metric_kwds={'p': ...}")
        unsupported = keys - {"p"}
        if unsupported:
            bad = ", ".join(sorted(str(key) for key in unsupported))
            raise ValueError(f"unsupported metric_kwds for minkowski: {bad}")
        p = float(kwds["p"])
        if not np.isfinite(p) or p <= 0.0:
            raise ValueError("metric_kwds['p'] must be finite and > 0 for minkowski")
        return metric_name, metric_kwds, p

    if keys:
        bad = ", ".join(sorted(str(key) for key in keys))
        raise ValueError(f"unsupported metric_kwds for {metric_name}: {bad}")
    return metric_name, metric_kwds, None


def _validate_target_weight(target_weight: float) -> float:
    weight = float(target_weight)
    if not np.isfinite(weight) or weight < 0.0 or weight > 1.0:
        raise ValueError("target_weight must be finite and in [0, 1]")
    return weight


def _validate_target_n_neighbors(target_n_neighbors: int | None) -> int | None:
    if target_n_neighbors is None:
        return None
    n_neighbors = int(target_n_neighbors)
    if n_neighbors < 1:
        raise ValueError("target_n_neighbors must be >= 1 when provided")
    return n_neighbors


def _validate_dens_params(
    dens_lambda: float,
    dens_frac: float,
    dens_var_shift: float,
) -> tuple[float, float, float]:
    dens_lambda = float(dens_lambda)
    dens_frac = float(dens_frac)
    dens_var_shift = float(dens_var_shift)
    if not np.isfinite(dens_lambda) or dens_lambda < 0.0:
        raise ValueError("dens_lambda must be finite and >= 0")
    if not np.isfinite(dens_frac) or dens_frac < 0.0 or dens_frac > 1.0:
        raise ValueError("dens_frac must be finite and in [0, 1]")
    if not np.isfinite(dens_var_shift) or dens_var_shift <= 0.0:
        raise ValueError("dens_var_shift must be finite and > 0")
    return dens_lambda, dens_frac, dens_var_shift


class Umap(_BaseEstimator, _TransformerMixin):
    """High-level Python wrapper around the Rust UMAP core.

    The Python layer is intentionally thin: it normalizes array-like inputs,
    handles optional CSR sparse inputs, and forwards validated data to the
    Rust implementation for fitting and inference.

    Example
    -------
    >>> import numpy as np
    >>> from umapers import Umap
    >>> x = np.random.default_rng(42).normal(size=(100, 8)).astype(np.float32)
    >>> emb = Umap(n_neighbors=15, n_components=2).fit_transform(x)
    """

    def __init__(
        self,
        *,
        n_neighbors: int = 15,
        n_components: int = 2,
        n_epochs: int | None = None,
        metric: str = "euclidean",
        metric_kwds: dict[str, Any] | None = None,
        learning_rate: float = 1.0,
        min_dist: float = 0.1,
        spread: float = 1.0,
        local_connectivity: float = 1.0,
        set_op_mix_ratio: float = 1.0,
        repulsion_strength: float = 1.0,
        negative_sample_rate: int = 5,
        random_seed: int = 42,
        init: str = "spectral",
        ann_mode: str = "auto",
        use_approximate_knn: bool = True,
        approx_knn_candidates: int = 50,
        approx_knn_iters: int = 14,
        approx_knn_threshold: int = 4096,
        densmap: bool = False,
        dens_lambda: float = 2.0,
        dens_frac: float = 0.3,
        dens_var_shift: float = 0.1,
        output_dens: bool = False,
        target_metric: str | None = None,
        target_weight: float = 0.5,
        target_n_neighbors: int | None = None,
    ) -> None:
        """Create a UMAP model.

        Parameters
        ----------
        n_neighbors:
            Number of neighbors used to build the neighborhood graph.
        n_components:
            Output embedding dimension.
        n_epochs:
            Number of optimization epochs. If `None`, the Rust core uses its
            internal default.
        metric:
            Distance metric for dense input and query transforms.
        metric_kwds:
            Optional metric parameters. Currently only
            ``metric="minkowski"`` uses ``{"p": ...}``.
        learning_rate, min_dist, spread, local_connectivity,
        set_op_mix_ratio, repulsion_strength, negative_sample_rate,
        random_seed, init:
            Standard UMAP hyperparameters forwarded to the Rust core.
        ann_mode:
            Python-side shortcut for approximate nearest-neighbor behavior.
            Supported values are `auto`, `exact`, and `approximate`.
        use_approximate_knn:
            Default approximate-kNN behavior when `ann_mode="auto"`.
        approx_knn_candidates, approx_knn_iters, approx_knn_threshold:
            Approximate-kNN tuning parameters forwarded to the Rust core.
        densmap, dens_lambda, dens_frac, dens_var_shift:
            Dense densMAP controls. The density term is active only when
            `densmap=True` and `dens_lambda > 0`.
        output_dens:
            If `True`, `fit_transform` returns
            `(embedding, radii_original, radii_embedding)`.
        target_metric:
            Optional target metric for supervised fitting. The first supported
            value is `categorical`; `None` keeps `y` as a compatibility no-op.
        target_weight:
            Weight used when combining the feature graph with the target graph.
        target_n_neighbors:
            Optional number of same-label target neighbors per sample.

        Examples
        --------
        >>> import numpy as np
        >>> from umapers import Umap
        >>> x = np.random.default_rng(42).normal(size=(200, 16)).astype(np.float32)
        >>> model = Umap(n_neighbors=15, n_components=2, init="random")
        >>> emb = model.fit_transform(x)
        >>> emb.shape
        (200, 2)
        """
        for fitted_attr in (
            "embedding_",
            "n_features_in_",
            "n_samples_fit_",
            "radii_original_",
            "radii_embedding_",
            "rad_orig_",
            "rad_emb_",
            "knn_diagnostics_",
        ):
            if hasattr(self, fitted_attr):
                delattr(self, fitted_attr)

        self.n_neighbors = int(n_neighbors)
        self.n_components = int(n_components)
        self.n_epochs = None if n_epochs is None else int(n_epochs)
        self.metric, self.metric_kwds, self._metric_p = _parse_metric_config(metric, metric_kwds)
        self.learning_rate = float(learning_rate)
        self.min_dist = float(min_dist)
        self.spread = float(spread)
        self.local_connectivity = float(local_connectivity)
        self.set_op_mix_ratio = float(set_op_mix_ratio)
        self.repulsion_strength = float(repulsion_strength)
        self.negative_sample_rate = int(negative_sample_rate)
        self.random_seed = int(random_seed)
        self.init = str(init)
        self.target_metric = _normalize_target_metric(target_metric)
        self.target_weight = _validate_target_weight(target_weight)
        self.target_n_neighbors = _validate_target_n_neighbors(target_n_neighbors)
        ann_mode, use_approximate_knn, approx_knn_threshold = _normalize_ann_mode(
            ann_mode,
            use_approximate_knn,
            approx_knn_threshold,
        )
        self.ann_mode = ann_mode
        self.use_approximate_knn = bool(use_approximate_knn)
        self.approx_knn_candidates = int(approx_knn_candidates)
        self.approx_knn_iters = int(approx_knn_iters)
        self.approx_knn_threshold = int(approx_knn_threshold)
        self.densmap = bool(densmap)
        self.dens_lambda, self.dens_frac, self.dens_var_shift = _validate_dens_params(
            dens_lambda,
            dens_frac,
            dens_var_shift,
        )
        self.output_dens = bool(output_dens)
        self._core = UmapCore(
            n_neighbors=self.n_neighbors,
            n_components=self.n_components,
            n_epochs=self.n_epochs,
            metric=self.metric,
            metric_p=self._metric_p,
            learning_rate=self.learning_rate,
            min_dist=self.min_dist,
            spread=self.spread,
            local_connectivity=self.local_connectivity,
            set_op_mix_ratio=self.set_op_mix_ratio,
            repulsion_strength=self.repulsion_strength,
            negative_sample_rate=self.negative_sample_rate,
            random_seed=self.random_seed,
            init=self.init,
            use_approximate_knn=self.use_approximate_knn,
            approx_knn_candidates=self.approx_knn_candidates,
            approx_knn_iters=self.approx_knn_iters,
            approx_knn_threshold=self.approx_knn_threshold,
            densmap=self.densmap,
            dens_lambda=self.dens_lambda,
            dens_frac=self.dens_frac,
            dens_var_shift=self.dens_var_shift,
            output_dens=self.output_dens,
        )

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Return constructor parameters for sklearn-compatible cloning."""
        return {name: getattr(self, name) for name in _PARAM_NAMES}

    def set_params(self, **params: Any) -> "Umap":
        """Set constructor parameters and rebuild the Rust core."""
        if not params:
            return self

        valid_params = set(_PARAM_NAMES)
        for key in params:
            if key not in valid_params:
                raise ValueError(f"Invalid parameter '{key}' for estimator Umap")

        current = self.get_params(deep=False)
        current.update(params)
        self.__init__(**current)
        return self

    def _sync_fitted_attributes(
        self,
        embedding: np.ndarray,
        *,
        n_features: int,
        n_samples: int,
    ) -> None:
        self.embedding_ = embedding
        self.n_features_in_ = int(n_features)
        self.n_samples_fit_ = int(n_samples)
        self._sync_density_attributes()

    def _sync_fitted_from_core(self, *, n_features: int, n_samples: int) -> None:
        embedding = self._core.embedding
        if embedding is None:
            raise RuntimeError("internal error: fitted core did not expose an embedding")
        self._sync_fitted_attributes(embedding, n_features=n_features, n_samples=n_samples)

    def _sync_density_attributes(self) -> None:
        radii_original = self._core.radii_original
        radii_embedding = self._core.radii_embedding
        if not self.output_dens or radii_original is None or radii_embedding is None:
            for fitted_attr in ("radii_original_", "radii_embedding_", "rad_orig_", "rad_emb_"):
                if hasattr(self, fitted_attr):
                    delattr(self, fitted_attr)
            return
        self.radii_original_ = radii_original
        self.radii_embedding_ = radii_embedding
        self.rad_orig_ = radii_original
        self.rad_emb_ = radii_embedding

    def fit(self, data: Any, y: Any | None = None) -> "Umap":
        """Fit the model on dense or CSR input and return `self`.

        Parameters
        ----------
        data:
            Dense input is converted to a C-contiguous `float32` matrix of
            shape `(n_samples, n_features)`. CSR sparse input is accepted as an
            advanced convenience path and is forwarded to the Rust core.
        y:
            Optional 1D label array. It is used only when
            `target_metric="categorical"`; otherwise it is ignored for
            compatibility with sklearn-style call sites.

        Returns
        -------
        Umap
            The fitted model.
        """
        csr = _maybe_as_csr_parts(data, "data")
        if csr is not None:
            if self.densmap or self.output_dens:
                raise ValueError("densmap and output_dens currently support dense input only")
            if self.target_metric == "categorical" and y is not None:
                raise ValueError("categorical supervised UMAP currently supports dense input only")
            indptr, indices, values, _, n_cols = csr
            self._core.fit_sparse_csr(indptr, indices, values, n_cols)
            self._sync_fitted_from_core(n_features=n_cols, n_samples=indptr.shape[0] - 1)
            return self

        arr = _as_f32_matrix(data, "data")
        if self.target_metric == "categorical" and y is not None:
            labels = _as_labels(y, "y")
            self._core.fit_supervised(
                arr,
                labels,
                self.target_weight,
                self.target_n_neighbors,
            )
            self._sync_fitted_from_core(n_features=arr.shape[1], n_samples=arr.shape[0])
            return self

        self._core.fit(arr)
        self._sync_fitted_from_core(n_features=arr.shape[1], n_samples=arr.shape[0])
        return self

    def fit_transform(
        self,
        data: Any,
        y: Any | None = None,
        *,
        out: np.ndarray | None = None,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Fit the model and return the embedding for the training data.

        Parameters
        ----------
        data:
            Dense input is converted to `float32` dtype and expected to have
            shape `(n_samples, n_features)`. CSR sparse input is supported for
            the current sparse MVP path.
        y:
            Optional 1D label array. It is used only when
            `target_metric="categorical"`; otherwise it is ignored.
        out:
            Optional writable `float32` dtype buffer with shape
            `(n_samples, n_components)`. When provided, the result is written
            in place and the same array is returned.

        Returns
        -------
        numpy.ndarray
            The fitted embedding with shape `(n_samples, n_components)`.
        """
        csr = _maybe_as_csr_parts(data, "data")
        if csr is not None:
            if out is not None and self.output_dens:
                raise ValueError("out cannot be used when output_dens=True")
            if self.densmap or self.output_dens:
                raise ValueError("densmap and output_dens currently support dense input only")
            if self.target_metric == "categorical" and y is not None:
                raise ValueError("categorical supervised UMAP currently supports dense input only")
            indptr, indices, values, n_rows, n_cols = csr
            expected_shape = (n_rows, self.n_components)
            if out is None:
                embedding = self._core.fit_transform_sparse_csr(indptr, indices, values, n_cols)
                self._sync_fitted_attributes(embedding, n_features=n_cols, n_samples=n_rows)
                return embedding
            out_buf = _as_out_buffer(out, expected_shape)
            self._core.fit_transform_sparse_csr_into(indptr, indices, values, n_cols, out_buf)
            self._sync_fitted_attributes(out_buf, n_features=n_cols, n_samples=n_rows)
            return out_buf

        arr = _as_f32_matrix(data, "data")
        expected_shape = (arr.shape[0], self.n_components)
        if out is not None and self.output_dens:
            raise ValueError("out cannot be used when output_dens=True")
        if self.target_metric == "categorical" and y is not None:
            labels = _as_labels(y, "y")
            if out is None:
                embedding = self._core.fit_transform_supervised(
                    arr,
                    labels,
                    self.target_weight,
                    self.target_n_neighbors,
                )
                self._sync_fitted_attributes(
                    embedding,
                    n_features=arr.shape[1],
                    n_samples=arr.shape[0],
                )
                if self.output_dens:
                    return embedding, self.radii_original_, self.radii_embedding_
                return embedding
            out_buf = _as_out_buffer(out, expected_shape)
            self._core.fit_transform_supervised_into(
                arr,
                labels,
                out_buf,
                self.target_weight,
                self.target_n_neighbors,
            )
            self._sync_fitted_attributes(out_buf, n_features=arr.shape[1], n_samples=arr.shape[0])
            return out_buf

        if out is None:
            embedding = self._core.fit_transform(arr)
            self._sync_fitted_attributes(embedding, n_features=arr.shape[1], n_samples=arr.shape[0])
            if self.output_dens:
                return embedding, self.radii_original_, self.radii_embedding_
            return embedding
        out_buf = _as_out_buffer(out, expected_shape)
        self._core.fit_transform_into(arr, out_buf)
        self._sync_fitted_attributes(out_buf, n_features=arr.shape[1], n_samples=arr.shape[0])
        return out_buf

    def profile_fit_transform(self, data: Any) -> dict[str, Any]:
        """Fit dense unsupervised input and return a module timing breakdown.

        This is an instrumentation helper for benchmark analysis. It follows
        the same dense unsupervised Rust path as `fit_transform`, stores the
        fitted model state, and returns the embedding plus per-stage timings.
        """
        if _maybe_as_csr_parts(data, "data") is not None:
            raise ValueError("profile_fit_transform currently supports dense input only")
        if self.target_metric is not None:
            raise ValueError("profile_fit_transform currently supports unsupervised input only")
        if self.densmap or self.output_dens:
            raise ValueError("profile_fit_transform currently supports standard UMAP only")

        arr = _as_f32_matrix(data, "data")
        result = self._core.profile_fit_transform(arr)
        embedding = result["embedding"]
        self._sync_fitted_attributes(embedding, n_features=arr.shape[1], n_samples=arr.shape[0])
        return result

    def fit_transform_with_knn(
        self,
        data: Any,
        knn_indices: Any,
        knn_dists: Any,
        *,
        knn_metric: str = "euclidean",
        validate_precomputed: bool = True,
        out: np.ndarray | None = None,
    ) -> np.ndarray:
        """Fit using a precomputed kNN graph and return the embedding.

        This is a public advanced interface for callers that already have an
        exact or shared kNN graph. It is useful for benchmark parity and for
        integrating external neighbor-search pipelines, but it is not the
        default quickstart path.

        Parameters
        ----------
        data:
            Dense training data with shape `(n_samples, n_features)`. It is
            converted to `float32`.
        knn_indices:
            Precomputed neighbor indices with shape `(n_samples, k)` and
            integer dtype.
        knn_dists:
            Precomputed neighbor distances with shape `(n_samples, k)` and
            `float32`-compatible values.
        knn_metric:
            Metric name for the precomputed graph. It must match the model
            metric.
        validate_precomputed:
            If `True`, the Rust core performs precomputed-kNN validation before
            fitting. The Python binding keeps this path thin and only
            normalizes array dtypes and layouts.
        out:
            Optional writable `float32` buffer with shape
            `(n_samples, n_components)`.

        Returns
        -------
        numpy.ndarray
            The fitted embedding with shape `(n_samples, n_components)`.

        Example
        -------
        >>> import numpy as np
        >>> from sklearn.neighbors import NearestNeighbors
        >>> from umapers import Umap
        >>> x = np.random.default_rng(42).normal(size=(64, 8)).astype(np.float32)
        >>> nbrs = NearestNeighbors(n_neighbors=16, algorithm="brute", metric="euclidean")
        >>> nbrs.fit(x)
        >>> dists, idx = nbrs.kneighbors(x)
        >>> emb = Umap(n_neighbors=15, metric="euclidean").fit_transform_with_knn(
        ...     x,
        ...     idx[:, 1:16].astype(np.int64),
        ...     dists[:, 1:16].astype(np.float32),
        ... )
        """
        arr = _as_f32_matrix(data, "data")
        idx = _as_knn_indices(knn_indices, "knn_indices")
        dist = _as_f32_matrix(knn_dists, "knn_dists")
        expected_shape = (arr.shape[0], self.n_components)
        if out is None:
            return self._core.fit_transform_with_knn(
                arr,
                idx,
                dist,
                knn_metric,
                validate_precomputed,
            )

        out_buf = _as_out_buffer(out, expected_shape)
        self._core.fit_transform_with_knn_into(
            arr,
            idx,
            dist,
            out_buf,
            knn_metric,
            validate_precomputed,
        )
        return out_buf

    def knn_diagnostics(self, data: Any) -> dict[str, Any]:
        """Measure approximate-kNN recall against exact neighbors.

        This is an explicit diagnostic path for ANN tuning and benchmark
        reports. It computes both exact and approximate neighbors for the
        supplied dense data, so it is intentionally not run during normal
        fitting.
        """
        arr = _as_f32_matrix(data, "data")
        diagnostics = self._core.knn_recall_diagnostics(arr)
        self.knn_diagnostics_ = diagnostics
        return diagnostics

    def transform(self, query: Any, *, out: np.ndarray | None = None) -> np.ndarray:
        """Project new samples into the learned embedding space.

        Parameters
        ----------
        query:
            Dense input of shape `(n_samples, n_features)` or CSR/CSC/COO
            sparse input for models fitted with sparse input.
        out:
            Optional writable `float32` buffer with shape
            `(n_samples, n_components)`.

        Returns
        -------
        numpy.ndarray
            The projected embedding.

        Example
        -------
        >>> import numpy as np
        >>> from umapers import Umap
        >>> x = np.random.default_rng(42).normal(size=(100, 8)).astype(np.float32)
        >>> model = Umap(n_neighbors=15, n_components=2).fit(x)
        >>> query_emb = model.transform(x[:10])
        """
        csr = _maybe_as_csr_parts(query, "query")
        if csr is not None:
            indptr, indices, values, n_rows, n_cols = csr
            expected_shape = (n_rows, self.n_components)
            if out is None:
                return self._core.transform_sparse_csr(indptr, indices, values, n_cols)
            out_buf = _as_out_buffer(out, expected_shape)
            self._core.transform_sparse_csr_into(indptr, indices, values, n_cols, out_buf)
            return out_buf

        arr = _as_f32_matrix(query, "query")
        expected_shape = (arr.shape[0], self.n_components)
        if out is None:
            return self._core.transform(arr)
        out_buf = _as_out_buffer(out, expected_shape)
        self._core.transform_into(arr, out_buf)
        return out_buf

    def inverse_transform(self, embedded_query: Any, *, out: np.ndarray | None = None) -> np.ndarray:
        """Map embedded samples back to the original feature space.

        Parameters
        ----------
        embedded_query:
            Dense embedding of shape `(n_samples, n_components)`. It is
            converted to `float32`.
        out:
            Optional writable `float32` buffer with shape
            `(n_samples, n_features)`. The model must already be fit before
            using `out=`.

        Returns
        -------
        numpy.ndarray
            Reconstructed samples in the original feature space.

        Example
        -------
        >>> import numpy as np
        >>> from umapers import Umap
        >>> x = np.random.default_rng(42).normal(size=(100, 8)).astype(np.float32)
        >>> model = Umap(n_neighbors=15, n_components=2).fit(x)
        >>> emb = model.transform(x[:10])
        >>> x_rec = model.inverse_transform(emb)
        """
        arr = _as_f32_matrix(embedded_query, "embedded_query")
        if out is None:
            return self._core.inverse_transform(arr)
        n_features = self._core.n_features
        if n_features is None:
            raise RuntimeError("model must be fit before inverse_transform(out=...)")
        out_buf = _as_out_buffer(out, (arr.shape[0], n_features))
        self._core.inverse_transform_into(arr, out_buf)
        return out_buf


class ParametricUmap(_BaseEstimator, _TransformerMixin):
    """Teacher-distilled parametric UMAP backed by the Rust MLP implementation."""

    def __init__(
        self,
        *,
        n_neighbors: int = 15,
        n_components: int = 2,
        n_epochs: int | None = None,
        metric: str = "euclidean",
        metric_kwds: dict[str, Any] | None = None,
        hidden_dim: int = 64,
        train_epochs: int = 120,
        batch_size: int = 128,
        inference_batch_size: int = 1024,
        learning_rate: float = 0.01,
        weight_decay: float = 1e-4,
        pairwise_loss_weight: float = 0.0,
        pairwise_pairs_per_batch: int = 32,
        standardize_input: bool = True,
        random_seed: int = 42,
        train_mode: str = "optimized",
    ) -> None:
        for fitted_attr in ("embedding_", "teacher_embedding_", "n_features_in_"):
            if hasattr(self, fitted_attr):
                delattr(self, fitted_attr)

        self.n_neighbors = int(n_neighbors)
        self.n_components = int(n_components)
        self.n_epochs = None if n_epochs is None else int(n_epochs)
        self.metric, self.metric_kwds, self._metric_p = _parse_metric_config(metric, metric_kwds)
        self.hidden_dim = int(hidden_dim)
        self.train_epochs = int(train_epochs)
        self.batch_size = int(batch_size)
        self.inference_batch_size = int(inference_batch_size)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.pairwise_loss_weight = float(pairwise_loss_weight)
        self.pairwise_pairs_per_batch = int(pairwise_pairs_per_batch)
        self.standardize_input = bool(standardize_input)
        self.random_seed = int(random_seed)
        self.train_mode = str(train_mode)
        self._core = ParametricUmapCore(
            n_neighbors=self.n_neighbors,
            n_components=self.n_components,
            n_epochs=self.n_epochs,
            metric=self.metric,
            metric_p=self._metric_p,
            hidden_dim=self.hidden_dim,
            train_epochs=self.train_epochs,
            batch_size=self.batch_size,
            inference_batch_size=self.inference_batch_size,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            pairwise_loss_weight=self.pairwise_loss_weight,
            pairwise_pairs_per_batch=self.pairwise_pairs_per_batch,
            standardize_input=self.standardize_input,
            random_seed=self.random_seed,
            train_mode=self.train_mode,
        )

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {name: getattr(self, name) for name in _PARAMETRIC_PARAM_NAMES}

    def set_params(self, **params: Any) -> "ParametricUmap":
        if not params:
            return self
        valid_params = set(_PARAMETRIC_PARAM_NAMES)
        for key in params:
            if key not in valid_params:
                raise ValueError(f"Invalid parameter '{key}' for estimator ParametricUmap")
        current = self.get_params(deep=False)
        current.update(params)
        self.__init__(**current)
        return self

    def _sync_fitted_attributes(self, embedding: np.ndarray, *, n_features: int) -> None:
        self.embedding_ = embedding
        self.n_features_in_ = int(n_features)
        teacher = self._core.teacher_embedding
        if teacher is not None:
            self.teacher_embedding_ = teacher

    def fit(self, data: Any, y: Any | None = None) -> "ParametricUmap":
        arr = _as_f32_matrix(data, "data")
        self._core.fit(arr)
        teacher = self._core.teacher_embedding
        if teacher is None:
            raise RuntimeError("internal error: fitted parametric core did not expose teacher embedding")
        self._sync_fitted_attributes(self._core.transform(arr), n_features=arr.shape[1])
        return self

    def fit_transform(self, data: Any, y: Any | None = None) -> np.ndarray:
        arr = _as_f32_matrix(data, "data")
        embedding = self._core.fit_transform(arr)
        self._sync_fitted_attributes(embedding, n_features=arr.shape[1])
        return embedding

    def transform(self, query: Any) -> np.ndarray:
        arr = _as_f32_matrix(query, "query")
        return self._core.transform(arr)


class AlignedUmap:
    """Dense aligned UMAP with explicit adjacent-slice relation arrays."""

    def __init__(
        self,
        *,
        n_neighbors: int = 15,
        n_components: int = 2,
        n_epochs: int | None = None,
        metric: str = "euclidean",
        metric_kwds: dict[str, Any] | None = None,
        random_seed: int = 42,
        init: str = "spectral",
        alignment_regularization: float = 0.08,
        alignment_learning_rate: float = 0.25,
        alignment_epochs: int | None = None,
        recenter_interval: int = 5,
    ) -> None:
        if hasattr(self, "embeddings_"):
            delattr(self, "embeddings_")
        self.n_neighbors = int(n_neighbors)
        self.n_components = int(n_components)
        self.n_epochs = None if n_epochs is None else int(n_epochs)
        self.metric, self.metric_kwds, self._metric_p = _parse_metric_config(metric, metric_kwds)
        self.random_seed = int(random_seed)
        self.init = str(init)
        self.alignment_regularization = float(alignment_regularization)
        self.alignment_learning_rate = float(alignment_learning_rate)
        self.alignment_epochs = None if alignment_epochs is None else int(alignment_epochs)
        self.recenter_interval = int(recenter_interval)
        self._core = AlignedUmapCore(
            n_neighbors=self.n_neighbors,
            n_components=self.n_components,
            n_epochs=self.n_epochs,
            metric=self.metric,
            metric_p=self._metric_p,
            random_seed=self.random_seed,
            init=self.init,
            alignment_regularization=self.alignment_regularization,
            alignment_learning_rate=self.alignment_learning_rate,
            alignment_epochs=self.alignment_epochs,
            recenter_interval=self.recenter_interval,
        )

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {name: getattr(self, name) for name in _ALIGNED_PARAM_NAMES}

    def set_params(self, **params: Any) -> "AlignedUmap":
        if not params:
            return self
        valid_params = set(_ALIGNED_PARAM_NAMES)
        for key in params:
            if key not in valid_params:
                raise ValueError(f"Invalid parameter '{key}' for estimator AlignedUmap")
        current = self.get_params(deep=False)
        current.update(params)
        self.__init__(**current)
        return self

    def fit_transform_identity(self, datasets: Any) -> list[np.ndarray]:
        arrays = [_as_f32_matrix(dataset, f"datasets[{idx}]") for idx, dataset in enumerate(datasets)]
        embeddings = list(self._core.fit_transform_identity(arrays))
        self.embeddings_ = embeddings
        return embeddings

    def fit_transform(self, datasets: Any, relations: Any) -> list[np.ndarray]:
        arrays = [_as_f32_matrix(dataset, f"datasets[{idx}]") for idx, dataset in enumerate(datasets)]
        relation_arrays = [
            _as_relation_array(relation, f"relations[{idx}]") for idx, relation in enumerate(relations)
        ]
        embeddings = list(self._core.fit_transform(arrays, relation_arrays))
        self.embeddings_ = embeddings
        return embeddings


def fit_transform(data: Any, y: Any | None = None, **kwargs: Any) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Embed a dataset in one call.

    Parameters
    ----------
    data:
        Dense or CSR input accepted by ``Umap.fit_transform``.
    y:
        Optional 1D label array used only when
        ``target_metric="categorical"``.
    **kwargs:
        Hyperparameters forwarded to ``Umap(...)``. Common keys include
        ``n_neighbors``, ``n_components``, ``metric``, ``init``, and
        ``random_seed``.

    Returns
    -------
    numpy.ndarray
        Embedding with shape ``(n_samples, n_components)`` and dtype
        ``float32``.

    Examples
    --------
    >>> import numpy as np
    >>> from umapers import fit_transform
    >>> x = np.random.default_rng(42).normal(size=(200, 16)).astype(np.float32)
    >>> emb = fit_transform(x, n_neighbors=15, n_components=2, init="random")
    >>> emb.shape
    (200, 2)
    """
    model = Umap(**kwargs)
    if y is None and model.target_metric is None and not model.densmap and not model.output_dens:
        csr = _maybe_as_csr_parts(data, "data")
        if csr is not None:
            indptr, indices, values, _, n_cols = csr
            return model._core.fit_transform_sparse_csr_stateless(indptr, indices, values, n_cols)

        arr = _as_f32_matrix(data, "data")
        return model._core.fit_transform_stateless(arr)

    return model.fit_transform(data, y=y)
