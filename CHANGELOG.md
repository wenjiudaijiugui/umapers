# Changelog

All notable changes to this repository are documented in this file.

The format follows Keep a Changelog and uses semantic-versioned release headings
for user-facing milestones.

## [1.1.4] - 2026-07-13

### Performance

- Reused fitted curve parameters, precomputed the curve-search powers, and
  removed repeated spectral scratch allocations while preserving the fitted
  embedding bit-for-bit on the release regression matrix.
- Parallelized independent dense-to-sparse queries and transform heads, and
  improved contiguous access in the parametric and aligned workflows. The
  transform path remains deterministic across thread counts, but its seeded
  coordinates intentionally differ from `1.1.3` because each query now owns an
  independent random stream.
- Removed avoidable inverse-transform gradient allocations and retained a
  bounded dense spectral fallback to prevent unbounded recovery allocations.

### Packaging

- Included the full BSD license text in Python distributions and declared the
  tested Python support range as 3.9 through 3.13.
- Added release version validation, cross-platform wheel smoke tests, and an
  automated GitHub Release asset step around the PyPI Trusted Publishing flow.
- Fixed the deep benchmark workflow to build inside an activated virtual
  environment.

## [1.1.3] - 2026-06-13

### Fixed

- Stabilized the iterative spectral initialization path for low-dimensional
  nonlinear manifolds just above the exact-spectral cutoff. The larger
  subspace and longer block iteration remove the Swiss Roll quality cliff seen
  around 2049 samples without forcing dense full eigendecomposition.
- Matched `umap-learn` density-output attribute semantics more closely:
  `rad_orig_`, `rad_emb_`, `radii_original_`, and `radii_embedding_` are exposed
  only when `output_dens=True`, and are cleared when estimator parameters are
  reset.
- Broadened the pyright usage probe to cover top-level wrapper exports beyond
  `Umap`, including `ParametricUmap`, `AlignedUmap`, and `__version__`.

## [1.1.2] - 2026-06-12

### Fixed

- Restored exact spectral initialization for low-dimensional connected
  manifolds up to 2048 samples, while keeping the oversampled iterative path
  for larger graphs. This fixes the Swiss Roll quality regression where
  silhouette was markedly below `umap-learn`.
- Exposed densMAP/output-density radii using the `umap-learn`-compatible
  `rad_orig_` and `rad_emb_` fitted attributes, while retaining
  `radii_original_` and `radii_embedding_`.
- Fixed top-level Python stub re-exports so pyright resolves
  `from umapers import Umap, UmapKwargs, fit_transform` from a virtual
  environment, and added a pyright usage probe for density-output typing.

## [1.1.1] - 2026-06-12

### Highlights

- Expanded the Python-facing UMAP surface with categorical supervision,
  additional dense metrics, dense densMAP/output-density support, sparse CSR
  parity paths, aligned/parametric wrappers, ecosystem plotting helpers, and
  ANN diagnostics.
- Retuned spectral initialization and automatic ANN selection from measured
  hot spots, then optimized the standard 2D layout path with a contiguous flat
  embedding loop.
- Added release-quality diagnostics and reports for feature parity,
  real-data clustering metrics, small-dataset hot spots, ANN recall, and
  synthetic runtime scaling against `umap-learn`.
- Current local release benchmarks show `umapers` faster than `umap-learn` on
  29 / 30 synthetic scaling datasets under a release build, while real-data
  clustering metrics remain comparable.

### Fixed

- Removed remaining pre-1.0 name references from release notes, reports,
  benchmark scripts, temporary directory prefixes, and source-distribution
  files.
- Restricted the Python release workflow to `umapers-v*` tags and made PyPI
  publishing idempotent when an artifact already exists.

## [1.1.0] - 2026-06-12

### Notes

- Initial 1.1 package publish. Superseded by `1.1.1`, which carries the same
  feature and performance work with cleaned release naming.

## [1.0.0] - 2026-05-26

### Highlights

- Renamed the Python distribution and import package to `umapers`.
- Bumped the Rust core crate and Python binding to version `1.0.0`.
- Includes the recent dense-distance, ANN, sparse kNN, transform, and memory optimizations that show significant speed and memory advantages over `umap-learn` in local benchmarks.

## [0.3.0] - 2026-04-03

### Highlights

- Published the Python binding on PyPI under the then-current pre-1.0
  distribution name, with the import path `umap_rs`.
- Shipped typed Python package assets for `umap_rs`, including
  `__init__.pyi`, `_api.pyi`, and `py.typed`, plus `UmapKwargs`, richer
  docstrings, hover text, and clearer signature help for the public API.
- Documented the Python API in layers: dense `Umap` and one-shot
  `fit_transform` are the default path, while
  `Umap.fit_transform_with_knn(...)` remains an advanced precomputed-kNN
  interface.
- Standardized public Python keyword naming on `**kwargs` and removed the
  overlapping stub pattern that caused Pyright/Pylance unreachable-overload
  diagnostics.
- Added automated Python release packaging for Linux x86_64, Windows x86_64,
  and macOS arm64 across Python 3.9-3.13, plus a non-blocking Python 3.14
  sdist probe.
- Verified release artifacts with `maturin build --release --sdist`,
  `twine check`, editable-install tests, import smoke tests, and CI wheel/sdist
  smoke installs.

### Naming Notes

- This was the pre-1.0 naming surface. The package was later renamed to
  `umapers`.
- Published support is Python 3.9 through 3.13. Python 3.14 was probed in CI
  from sdist, but it is not yet part of the supported range.
