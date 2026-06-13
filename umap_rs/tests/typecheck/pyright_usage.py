from __future__ import annotations

import numpy as np
import numpy.typing as npt

from umapers import AlignedUmap, ParametricUmap, Umap, UmapKwargs, __version__, fit_transform


rng = np.random.default_rng(42)
x: npt.NDArray[np.float32] = rng.normal(size=(64, 8)).astype(np.float32)
relations: list[npt.NDArray[np.int64]] = [np.array([[0, 0], [1, 1]], dtype=np.int64)]

kwargs: UmapKwargs = {
    "n_neighbors": 8,
    "n_components": 2,
    "init": "spectral",
    "output_dens": False,
}

embedding: npt.NDArray[np.float32] = fit_transform(
    x,
    n_neighbors=8,
    n_components=2,
    init="spectral",
    output_dens=False,
)

dense_model = Umap(**kwargs)
dense_embedding = dense_model.fit_transform(x)
if isinstance(dense_embedding, tuple):
    raise TypeError("output_dens=False should return an embedding array")

density_embedding, rad_orig, rad_emb = fit_transform(
    x,
    n_neighbors=8,
    n_components=2,
    output_dens=True,
)

density_model = Umap(n_neighbors=8, n_components=2, output_dens=True)
density_output = density_model.fit_transform(x)
if not isinstance(density_output, tuple):
    raise TypeError("output_dens=True should return density output")

model_embedding, model_rad_orig, model_rad_emb = density_output
parametric_embedding = ParametricUmap(n_neighbors=8, n_components=2).fit_transform(x)
aligned_embeddings = AlignedUmap(n_neighbors=8, n_components=2).fit_transform([x, x], relations)

version_text: str = __version__
_: tuple[int, ...] = embedding.shape
_: tuple[int, ...] = dense_embedding.shape
_: tuple[int, ...] = density_embedding.shape
_: tuple[int, ...] = rad_orig.shape
_: tuple[int, ...] = rad_emb.shape
_: tuple[int, ...] = model_embedding.shape
_: tuple[int, ...] = model_rad_orig.shape
_: tuple[int, ...] = model_rad_emb.shape
_: tuple[int, ...] = density_model.rad_orig_.shape
_: tuple[int, ...] = density_model.rad_emb_.shape
_: tuple[int, ...] = parametric_embedding.shape
_: tuple[int, ...] = aligned_embeddings[0].shape
