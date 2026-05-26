from __future__ import annotations

from typing import Any, TypedDict

import numpy as np

from ._umap_rs import UmapCore


class UmapKwargs(TypedDict, total=False):
    """Keyword arguments accepted by `Umap(...)` and `fit_transform(..., **kwargs)`."""

    n_neighbors: int
    n_components: int
    n_epochs: int | None
    metric: str
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


def _maybe_as_csr_parts(x: Any, name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int] | None:
    if getattr(x, "format", None) != "csr":
        return None

    shape = getattr(x, "shape", None)
    if shape is None or len(shape) != 2:
        raise ValueError(f"{name} must be a 2D CSR matrix")

    n_rows, n_cols = int(shape[0]), int(shape[1])
    if n_cols <= 0:
        raise ValueError(f"{name} must have at least one column")

    indptr = np.asarray(x.indptr, dtype=np.int64, order="C")
    indices = np.asarray(x.indices, dtype=np.int64, order="C")
    data = np.asarray(x.data, dtype=np.float32, order="C")
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
    mode = str(ann_mode).lower()
    if mode == "auto":
        return mode, use_approximate_knn, approx_knn_threshold
    if mode == "exact":
        return mode, False, approx_knn_threshold
    if mode == "approximate":
        return mode, True, 0
    raise ValueError(f"unsupported ann_mode '{ann_mode}', expected auto|exact|approximate")


class Umap:
    """High-level Python wrapper around the Rust UMAP core.

    The Python layer is intentionally thin: it normalizes array-like inputs,
    handles optional CSR sparse inputs, and forwards validated data to the
    Rust implementation for fitting and inference.

    Example
    -------
    >>> import numpy as np
    >>> from umap_rs import Umap
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
        approx_knn_candidates: int = 30,
        approx_knn_iters: int = 10,
        approx_knn_threshold: int = 4096,
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

        Examples
        --------
        >>> import numpy as np
        >>> from umap_rs import Umap
        >>> x = np.random.default_rng(42).normal(size=(200, 16)).astype(np.float32)
        >>> model = Umap(n_neighbors=15, n_components=2, init="random")
        >>> emb = model.fit_transform(x)
        >>> emb.shape
        (200, 2)
        """
        self.n_neighbors = int(n_neighbors)
        self.n_components = int(n_components)
        ann_mode, use_approximate_knn, approx_knn_threshold = _normalize_ann_mode(
            ann_mode,
            use_approximate_knn,
            approx_knn_threshold,
        )
        self.ann_mode = ann_mode
        self._core = UmapCore(
            n_neighbors=n_neighbors,
            n_components=n_components,
            n_epochs=n_epochs,
            metric=metric,
            learning_rate=learning_rate,
            min_dist=min_dist,
            spread=spread,
            local_connectivity=local_connectivity,
            set_op_mix_ratio=set_op_mix_ratio,
            repulsion_strength=repulsion_strength,
            negative_sample_rate=negative_sample_rate,
            random_seed=random_seed,
            init=init,
            use_approximate_knn=use_approximate_knn,
            approx_knn_candidates=approx_knn_candidates,
            approx_knn_iters=approx_knn_iters,
            approx_knn_threshold=approx_knn_threshold,
        )

    def fit(self, data: Any) -> "Umap":
        """Fit the model on dense or CSR input and return `self`.

        Parameters
        ----------
        data:
            Dense input is converted to a C-contiguous `float32` matrix of
            shape `(n_samples, n_features)`. CSR sparse input is accepted as an
            advanced convenience path and is forwarded to the Rust core.

        Returns
        -------
        Umap
            The fitted model.
        """
        csr = _maybe_as_csr_parts(data, "data")
        if csr is not None:
            indptr, indices, values, _, n_cols = csr
            self._core.fit_sparse_csr(indptr, indices, values, n_cols)
            return self

        arr = _as_f32_matrix(data, "data")
        self._core.fit(arr)
        return self

    def fit_transform(self, data: Any, *, out: np.ndarray | None = None) -> np.ndarray:
        """Fit the model and return the embedding for the training data.

        Parameters
        ----------
        data:
            Dense input is converted to `float32` dtype and expected to have
            shape `(n_samples, n_features)`. CSR sparse input is supported for
            the current sparse MVP path.
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
            indptr, indices, values, n_rows, n_cols = csr
            expected_shape = (n_rows, self.n_components)
            if out is None:
                return self._core.fit_transform_sparse_csr(indptr, indices, values, n_cols)
            out_buf = _as_out_buffer(out, expected_shape)
            self._core.fit_transform_sparse_csr_into(indptr, indices, values, n_cols, out_buf)
            return out_buf

        arr = _as_f32_matrix(data, "data")
        expected_shape = (arr.shape[0], self.n_components)
        if out is None:
            return self._core.fit_transform(arr)
        out_buf = _as_out_buffer(out, expected_shape)
        self._core.fit_transform_into(arr, out_buf)
        return out_buf

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
        >>> from umap_rs import Umap
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

    def transform(self, query: Any, *, out: np.ndarray | None = None) -> np.ndarray:
        """Project new dense samples into the learned embedding space.

        Parameters
        ----------
        query:
            Dense input of shape `(n_samples, n_features)`. It is converted to
            a C-contiguous `float32` matrix.
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
        >>> from umap_rs import Umap
        >>> x = np.random.default_rng(42).normal(size=(100, 8)).astype(np.float32)
        >>> model = Umap(n_neighbors=15, n_components=2).fit(x)
        >>> query_emb = model.transform(x[:10])
        """
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
        >>> from umap_rs import Umap
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


def fit_transform(data: Any, **kwargs: Any) -> np.ndarray:
    """Embed a dataset in one call.

    Parameters
    ----------
    data:
        Dense or CSR input accepted by ``Umap.fit_transform``.
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
    >>> from umap_rs import fit_transform
    >>> x = np.random.default_rng(42).normal(size=(200, 16)).astype(np.float32)
    >>> emb = fit_transform(x, n_neighbors=15, n_components=2, init="random")
    >>> emb.shape
    (200, 2)
    """
    model = Umap(**kwargs)
    csr = _maybe_as_csr_parts(data, "data")
    if csr is not None:
        indptr, indices, values, _, n_cols = csr
        return model._core.fit_transform_sparse_csr_stateless(indptr, indices, values, n_cols)

    arr = _as_f32_matrix(data, "data")
    return model._core.fit_transform_stateless(arr)
