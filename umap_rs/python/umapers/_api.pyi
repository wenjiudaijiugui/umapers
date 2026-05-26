from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal, Protocol, SupportsFloat, SupportsIndex, TypedDict, Unpack, overload

import numpy as np
import numpy.typing as npt

Float32Array = npt.NDArray[np.float32]


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
    """Structural type for CSR sparse matrices accepted by the thin binding.

    English:
    Any object that behaves like a SciPy CSR matrix.

    中文：
    任何行为上兼容 SciPy CSR 矩阵的对象。
    """

    format: str
    shape: tuple[int, int]
    indptr: Sequence[SupportsIndex] | DenseMatrixLike
    indices: Sequence[SupportsIndex] | DenseMatrixLike
    data: Sequence[SupportsFloat] | DenseMatrixLike


DenseRows = Sequence[Sequence[SupportsFloat]]
IndexRows = Sequence[Sequence[SupportsIndex]]
MatrixInput = DenseMatrixLike | DenseRows | CsrMatrixLike


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
    metric: Literal["euclidean", "manhattan", "cosine"]
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

    @overload
    def __init__(
        self,
        *,
        n_neighbors: int = ...,
        n_components: int = ...,
        n_epochs: int | None = ...,
        metric: Literal["euclidean", "manhattan", "cosine"] = ...,
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
    def __init__(self, **kwargs: Any) -> None:
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

    def fit(self, data: MatrixInput) -> Umap:
        """Fit the model on dense or CSR input and return `self`.

        English:
        `data` should be a 2D dense array-like object or a CSR-like sparse
        matrix.

        中文：
        `data` 应为二维 dense array-like 输入，或 CSR 风格的稀疏矩阵对象。
        """

    def fit_transform(
        self, data: MatrixInput, *, out: Float32Array | None = ...
    ) -> Float32Array:
        """Fit the model and return the embedding for the training data.

        English:
        `data` should be a 2D dense array-like object or a CSR-like sparse
        matrix. The return value is a `float32` array with shape
        `(n_samples, n_components)`.

        中文：
        `data` 应为二维 dense array-like 输入，或 CSR 风格的稀疏矩阵对象。
        返回值是形状为 `(n_samples, n_components)` 的 `float32` 数组。

        Example / 示例
        ----------------
        >>> import numpy as np
        >>> from umapers import Umap
        >>> x = np.random.default_rng(42).normal(size=(200, 16)).astype(np.float32)
        >>> emb = Umap(n_neighbors=15, n_components=2, init="random").fit_transform(x)
        >>> emb.dtype
        dtype('float32')
        """

    def fit_transform_with_knn(
        self,
        data: DenseMatrixLike,
        knn_indices: DenseMatrixLike | IndexRows,
        knn_dists: DenseMatrixLike | DenseRows,
        *,
        knn_metric: Literal["euclidean", "manhattan", "cosine"] = ...,
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

    def transform(
        self, query: DenseMatrixLike, *, out: Float32Array | None = ...
    ) -> Float32Array:
        """Project new dense samples into the learned embedding space.

        English:
        The result has shape `(n_samples, n_components)`.

        中文：
        返回结果形状为 `(n_samples, n_components)`。
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


@overload
def fit_transform(
    data: MatrixInput,
    *,
    n_neighbors: int = ...,
    n_components: int = ...,
    n_epochs: int | None = ...,
    metric: Literal["euclidean", "manhattan", "cosine"] = ...,
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
) -> Float32Array:
    """Embed a dataset in one call.

    Parameters
    ----------
    data :
        2D dense array-like input, a sequence of dense rows, or a CSR-like
        sparse matrix.
    n_neighbors, n_components, n_epochs, metric, learning_rate, min_dist,
    spread, local_connectivity, set_op_mix_ratio, repulsion_strength,
    negative_sample_rate, random_seed, init, ann_mode,
    use_approximate_knn, approx_knn_candidates, approx_knn_iters,
    approx_knn_threshold :
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
def fit_transform(data: MatrixInput, **kwargs: Any) -> Float32Array:
    """Embed a dataset in one call.

    Parameters
    ----------
    data :
        2D dense array-like input, a sequence of dense rows, or a CSR-like
        sparse matrix.
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
