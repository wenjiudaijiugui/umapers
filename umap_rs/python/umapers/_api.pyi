from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal, Protocol, SupportsFloat, SupportsIndex, TypedDict, Unpack, overload

import numpy as np
import numpy.typing as npt

Float32Array = npt.NDArray[np.float32]
DensOutput = tuple[Float32Array, Float32Array, Float32Array]


class DenseMatrixLike(Protocol):
    """2D dense array-like input accepted by the Python binding.

    English:
    Any 2D dense object that NumPy can convert into an array.

    中文：
    任何能被 NumPy 转成二维数组的 dense 输入对象。
    """

    @property
    def shape(self) -> tuple[int, ...]: ...

    def __array__(self, dtype: Any = ..., /) -> npt.NDArray[Any]: ...


class CsrMatrixLike(Protocol):
    """Structural type for sparse matrices accepted by the thin binding.

    English:
    Any object that behaves like a SciPy CSR matrix. CSC and COO objects are
    accepted at runtime when they provide ``.tocsr()``.

    中文：
    任何行为上兼容 SciPy CSR 矩阵的对象。运行时也接受带有 ``.tocsr()``
    的 CSC 和 COO 对象。
    """

    format: str
    shape: tuple[int, int]
    indptr: Sequence[SupportsIndex] | DenseMatrixLike
    indices: Sequence[SupportsIndex] | DenseMatrixLike
    data: Sequence[SupportsFloat] | DenseMatrixLike
    def tocsr(self) -> CsrMatrixLike: ...


DenseRows = Sequence[Sequence[SupportsFloat]]
IndexRows = Sequence[Sequence[SupportsIndex]]
MatrixInput = DenseMatrixLike | DenseRows | CsrMatrixLike
LabelInput = Sequence[SupportsIndex] | DenseMatrixLike


class UmapKwargs(TypedDict, total=False):
    """Keyword arguments accepted by `Umap(...)` and `fit_transform(..., **kwargs)`.

    English:
    Annotate parameter dictionaries as `UmapKwargs` when you call
    `Umap(**kwargs)` or `fit_transform(x, **kwargs)`. This lets IDEs expand the
    supported keys and show per-key hover information.

    中文：
    当你使用 `Umap(**kwargs)` 或 `fit_transform(x, **kwargs)` 时，建议把参数
    字典标注为 `UmapKwargs`。这样 IDE 可以展开支持的键，并显示更完整的悬停提示。

    Example / 示例
    ----------------
    >>> import numpy as np
    >>> from umapers import UmapKwargs, fit_transform
    >>> x = np.random.default_rng(42).normal(size=(200, 16)).astype(np.float32)
    >>> kwargs: UmapKwargs = {"n_neighbors": 15, "n_components": 2, "init": "random"}
    >>> emb = fit_transform(x, **kwargs)
    >>> emb.shape
    (200, 2)
    """

    n_neighbors: int
    n_components: int
    n_epochs: int | None
    metric: Literal[
        "euclidean",
        "l2",
        "manhattan",
        "l1",
        "cosine",
        "chebyshev",
        "linfinity",
        "linf",
        "minkowski",
        "correlation",
        "canberra",
        "braycurtis",
        "bray_curtis",
    ]
    metric_kwds: dict[str, Any] | None
    learning_rate: float
    min_dist: float
    spread: float
    local_connectivity: float
    set_op_mix_ratio: float
    repulsion_strength: float
    negative_sample_rate: int
    random_seed: int
    init: Literal["random", "spectral"]
    ann_mode: Literal["auto", "exact", "approximate"]
    use_approximate_knn: bool
    approx_knn_candidates: int
    approx_knn_iters: int
    approx_knn_threshold: int
    densmap: bool
    dens_lambda: float
    dens_frac: float
    dens_var_shift: float
    output_dens: bool
    target_metric: Literal["categorical"] | None
    target_weight: float
    target_n_neighbors: int | None


class Umap:
    """High-level Python wrapper around the Rust UMAP core.

    English:
    The Python layer stays thin: it normalizes array-like inputs and forwards
    compute-heavy work to Rust.

    中文：
    Python 层保持轻量，只负责输入归一化，并把主要计算交给 Rust 核心实现。

    Example / 示例
    ----------------
    >>> import numpy as np
    >>> from umapers import Umap
    >>> x = np.random.default_rng(42).normal(size=(200, 16)).astype(np.float32)
    >>> model = Umap(n_neighbors=15, n_components=2, init="random")
    >>> emb = model.fit_transform(x)
    >>> emb.shape
    (200, 2)
    """

    n_neighbors: int
    n_components: int
    ann_mode: str
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
    embedding_: Float32Array
    n_features_in_: int
    n_samples_fit_: int
    rad_orig_: Float32Array
    rad_emb_: Float32Array
    radii_original_: Float32Array
    radii_embedding_: Float32Array
    knn_diagnostics_: dict[str, Any]

    @overload
    def __init__(
        self,
        *,
        n_neighbors: int = ...,
        n_components: int = ...,
        n_epochs: int | None = ...,
        metric: Literal[
            "euclidean",
            "l2",
            "manhattan",
            "l1",
            "cosine",
            "chebyshev",
            "linfinity",
            "linf",
            "minkowski",
            "correlation",
            "canberra",
            "braycurtis",
            "bray_curtis",
        ] = ...,
        metric_kwds: dict[str, Any] | None = ...,
        learning_rate: float = ...,
        min_dist: float = ...,
        spread: float = ...,
        local_connectivity: float = ...,
        set_op_mix_ratio: float = ...,
        repulsion_strength: float = ...,
        negative_sample_rate: int = ...,
        random_seed: int = ...,
        init: Literal["random", "spectral"] = ...,
        ann_mode: Literal["auto", "exact", "approximate"] = ...,
        use_approximate_knn: bool = ...,
        approx_knn_candidates: int = ...,
        approx_knn_iters: int = ...,
        approx_knn_threshold: int = ...,
        densmap: bool = ...,
        dens_lambda: float = ...,
        dens_frac: float = ...,
        dens_var_shift: float = ...,
        output_dens: bool = ...,
        target_metric: Literal["categorical"] | None = ...,
        target_weight: float = ...,
        target_n_neighbors: int | None = ...,
    ) -> None:
        """Create a UMAP model.

        Parameters
        ----------
        n_neighbors :
            Number of neighbors used to build the neighborhood graph.
        n_components :
            Output embedding dimension.
        n_epochs :
            Number of optimization epochs. If ``None``, the Rust core chooses
            its default.
        metric :
            Distance metric used for fitting and transform operations.
        metric_kwds :
            Optional metric parameters. Currently only ``metric="minkowski"``
            uses ``{"p": ...}``.
        learning_rate, min_dist, spread, local_connectivity, set_op_mix_ratio,
        repulsion_strength, negative_sample_rate, random_seed :
            Standard UMAP hyperparameters forwarded to the Rust core.
        init :
            Initialization strategy. Supported values are ``"random"`` and
            ``"spectral"``.
        ann_mode :
            Nearest-neighbor mode. Supported values are ``"auto"``,
            ``"exact"``, and ``"approximate"``.
        use_approximate_knn, approx_knn_candidates, approx_knn_iters,
        approx_knn_threshold :
            Advanced approximate-kNN controls.
        densmap, dens_lambda, dens_frac, dens_var_shift :
            Dense densMAP controls.
        output_dens :
            If true, ``fit_transform`` returns
            ``(embedding, radii_original, radii_embedding)``.
        target_metric :
            Optional supervised target metric. The first supported value is
            ``"categorical"``; ``None`` keeps ``y`` as a no-op compatibility
            argument.
        target_weight :
            Weight used to combine the feature graph and target graph.
        target_n_neighbors :
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

    @overload
    def __init__(self, **kwargs: Unpack[UmapKwargs]) -> None:
        """Create a UMAP model from keyword parameters.

        Parameters
        ----------
        **kwargs :
            Hyperparameters accepted by ``Umap(...)``. Common keys include
            ``n_neighbors``, ``n_components``, ``metric``, ``init``, and
            ``random_seed``.

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

    def get_params(self, deep: bool = ...) -> dict[str, Any]:
        """Return constructor parameters for sklearn-compatible cloning."""

    def set_params(self, **params: Any) -> Umap:
        """Set constructor parameters and rebuild the Rust core."""

    def fit(self, data: MatrixInput, y: LabelInput | None = ...) -> Umap:
        """Fit the model on dense or sparse input and return `self`.

        English:
        `data` should be a 2D dense array-like object or a CSR/CSC/COO sparse
        matrix. `y` is used only when ``target_metric="categorical"``.

        中文：
        `data` 应为二维 dense array-like 输入，或 CSR/CSC/COO 稀疏矩阵对象。
        `y` 仅在 ``target_metric="categorical"`` 时参与训练。
        """

    def fit_transform(
        self,
        data: MatrixInput,
        y: LabelInput | None = ...,
        *,
        out: Float32Array | None = ...,
    ) -> Float32Array | DensOutput:
        """Fit the model and return the embedding for the training data.

        English:
        `data` should be a 2D dense array-like object or a CSR/CSC/COO sparse
        matrix. The return value is a `float32` array with shape
        `(n_samples, n_components)`. `y` is used only when
        ``target_metric="categorical"``.

        中文：
        `data` 应为二维 dense array-like 输入，或 CSR/CSC/COO 稀疏矩阵对象。
        返回值是形状为 `(n_samples, n_components)` 的 `float32` 数组。
        `y` 仅在 ``target_metric="categorical"`` 时参与训练。

        Example / 示例
        ----------------
        >>> import numpy as np
        >>> from umapers import Umap
        >>> x = np.random.default_rng(42).normal(size=(200, 16)).astype(np.float32)
        >>> emb = Umap(n_neighbors=15, n_components=2, init="random").fit_transform(x)
        >>> emb.dtype
        dtype('float32')
        """

    def profile_fit_transform(self, data: DenseMatrixLike) -> dict[str, Any]:
        """Fit dense unsupervised input and return embedding plus stage timings."""

    def fit_transform_with_knn(
        self,
        data: DenseMatrixLike,
        knn_indices: DenseMatrixLike | IndexRows,
        knn_dists: DenseMatrixLike | DenseRows,
        *,
        knn_metric: Literal[
            "euclidean",
            "l2",
            "manhattan",
            "l1",
            "cosine",
            "chebyshev",
            "linfinity",
            "linf",
            "correlation",
            "canberra",
            "braycurtis",
            "bray_curtis",
        ] = ...,
        validate_precomputed: bool = ...,
        out: Float32Array | None = ...,
    ) -> Float32Array:
        """Fit using a precomputed kNN graph and return the embedding.

        English:
        This is an advanced API for exact/shared kNN reuse, benchmark parity,
        or interop with an external neighbor-search pipeline.

        中文：
        这是高级接口，适合复用精确/shared kNN 图、做公平基准，或接入外部近邻搜索链路。
        """

    def knn_diagnostics(self, data: DenseMatrixLike) -> dict[str, Any]:
        """Measure approximate-kNN recall against exact neighbors.

        English:
        Explicit ANN diagnostic path. Computes exact and approximate neighbors
        for ``data`` and stores the returned report on ``knn_diagnostics_``.

        中文：
        显式 ANN 诊断路径。会为 ``data`` 同时计算 exact 和 approximate
        neighbors，并把返回报告保存到 ``knn_diagnostics_``。
        """

    def transform(self, query: MatrixInput, *, out: Float32Array | None = ...) -> Float32Array:
        """Project new samples into the learned embedding space.

        English:
        Dense query works for dense- and sparse-fitted models. Sparse query
        works for sparse-fitted models without densifying the query matrix. The
        result has shape `(n_samples, n_components)`.

        中文：
        dense query 可用于 dense/sparse 拟合后的模型；sparse query 可用于
        sparse 拟合后的模型，且不会把 query 矩阵整体 densify。返回结果形状为
        `(n_samples, n_components)`。
        """

    def inverse_transform(
        self,
        embedded_query: DenseMatrixLike,
        *,
        out: Float32Array | None = ...,
    ) -> Float32Array:
        """Map embedded samples back to the original feature space.

        English:
        The result has shape `(n_samples, n_features)`.

        中文：
        返回结果形状为 `(n_samples, n_features)`。
        """


class ParametricUmap:
    """Teacher-distilled parametric UMAP backed by the Rust MLP implementation."""

    n_neighbors: int
    n_components: int
    n_epochs: int | None
    metric: str
    metric_kwds: dict[str, Any] | None
    hidden_dim: int
    train_epochs: int
    batch_size: int
    inference_batch_size: int
    learning_rate: float
    weight_decay: float
    pairwise_loss_weight: float
    pairwise_pairs_per_batch: int
    standardize_input: bool
    random_seed: int
    train_mode: str
    embedding_: Float32Array
    teacher_embedding_: Float32Array
    n_features_in_: int

    def __init__(
        self,
        *,
        n_neighbors: int = ...,
        n_components: int = ...,
        n_epochs: int | None = ...,
        metric: str = ...,
        metric_kwds: dict[str, Any] | None = ...,
        hidden_dim: int = ...,
        train_epochs: int = ...,
        batch_size: int = ...,
        inference_batch_size: int = ...,
        learning_rate: float = ...,
        weight_decay: float = ...,
        pairwise_loss_weight: float = ...,
        pairwise_pairs_per_batch: int = ...,
        standardize_input: bool = ...,
        random_seed: int = ...,
        train_mode: Literal["optimized", "naive"] = ...,
    ) -> None: ...

    def get_params(self, deep: bool = ...) -> dict[str, Any]: ...

    def set_params(self, **params: Any) -> ParametricUmap: ...

    def fit(self, data: MatrixInput, y: LabelInput | None = ...) -> ParametricUmap: ...

    def fit_transform(self, data: MatrixInput, y: LabelInput | None = ...) -> Float32Array: ...

    def transform(self, query: DenseMatrixLike) -> Float32Array: ...


class AlignedUmap:
    """Dense aligned UMAP with explicit adjacent-slice relation arrays."""

    n_neighbors: int
    n_components: int
    n_epochs: int | None
    metric: str
    metric_kwds: dict[str, Any] | None
    random_seed: int
    init: str
    alignment_regularization: float
    alignment_learning_rate: float
    alignment_epochs: int | None
    recenter_interval: int
    embeddings_: list[Float32Array]

    def __init__(
        self,
        *,
        n_neighbors: int = ...,
        n_components: int = ...,
        n_epochs: int | None = ...,
        metric: str = ...,
        metric_kwds: dict[str, Any] | None = ...,
        random_seed: int = ...,
        init: Literal["random", "spectral"] = ...,
        alignment_regularization: float = ...,
        alignment_learning_rate: float = ...,
        alignment_epochs: int | None = ...,
        recenter_interval: int = ...,
    ) -> None: ...

    def get_params(self, deep: bool = ...) -> dict[str, Any]: ...

    def set_params(self, **params: Any) -> AlignedUmap: ...

    def fit_transform_identity(self, datasets: Sequence[DenseMatrixLike]) -> list[Float32Array]: ...

    def fit_transform(
        self,
        datasets: Sequence[DenseMatrixLike],
        relations: Sequence[DenseMatrixLike | IndexRows],
    ) -> list[Float32Array]: ...


@overload
def fit_transform(
    data: MatrixInput,
    y: LabelInput | None = ...,
    *,
    output_dens: Literal[True],
    **kwargs: Any,
) -> DensOutput:
    """Embed a dataset and return ``(embedding, rad_orig, rad_emb)``."""


@overload
def fit_transform(
    data: MatrixInput,
    y: LabelInput | None = ...,
    *,
    n_neighbors: int = ...,
    n_components: int = ...,
    n_epochs: int | None = ...,
    metric: Literal[
        "euclidean",
        "l2",
        "manhattan",
        "l1",
        "cosine",
        "chebyshev",
        "linfinity",
        "linf",
        "minkowski",
        "correlation",
        "canberra",
        "braycurtis",
        "bray_curtis",
    ] = ...,
    metric_kwds: dict[str, Any] | None = ...,
    learning_rate: float = ...,
    min_dist: float = ...,
    spread: float = ...,
    local_connectivity: float = ...,
    set_op_mix_ratio: float = ...,
    repulsion_strength: float = ...,
    negative_sample_rate: int = ...,
    random_seed: int = ...,
    init: Literal["random", "spectral"] = ...,
    ann_mode: Literal["auto", "exact", "approximate"] = ...,
    use_approximate_knn: bool = ...,
    approx_knn_candidates: int = ...,
    approx_knn_iters: int = ...,
    approx_knn_threshold: int = ...,
    densmap: bool = ...,
    dens_lambda: float = ...,
    dens_frac: float = ...,
    dens_var_shift: float = ...,
    output_dens: Literal[False] = ...,
    target_metric: Literal["categorical"] | None = ...,
    target_weight: float = ...,
    target_n_neighbors: int | None = ...,
) -> Float32Array:
    """Embed a dataset in one call.

    Parameters
    ----------
    data :
        2D dense array-like input, a sequence of dense rows, or a CSR-like
        sparse matrix.
    y :
        Optional label array used only when ``target_metric="categorical"``.
    n_neighbors, n_components, n_epochs, metric, learning_rate, min_dist,
    spread, local_connectivity, set_op_mix_ratio, repulsion_strength,
    negative_sample_rate, random_seed, init, ann_mode, metric_kwds,
    use_approximate_knn, approx_knn_candidates, approx_knn_iters,
    approx_knn_threshold, densmap, dens_lambda, dens_frac,
    dens_var_shift, output_dens, target_metric, target_weight,
    target_n_neighbors :
        Model hyperparameters forwarded to ``Umap(...)``.

    Returns
    -------
    Float32Array
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


@overload
def fit_transform(
    data: MatrixInput,
    y: LabelInput | None = ...,
    **kwargs: Unpack[UmapKwargs],
) -> Float32Array | DensOutput:
    """Embed a dataset in one call.

    Parameters
    ----------
    data :
        2D dense array-like input, a sequence of dense rows, or a CSR-like
        sparse matrix.
    y :
        Optional label array used only when ``target_metric="categorical"``.
    **kwargs :
        Hyperparameters forwarded to ``Umap(...)``. Common keys include
        ``n_neighbors``, ``n_components``, ``metric``, ``init``, and
        ``random_seed``.

    Returns
    -------
    Float32Array
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
