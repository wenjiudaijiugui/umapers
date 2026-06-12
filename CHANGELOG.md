# Changelog

All notable changes to this repository are documented in this file.

The format follows Keep a Changelog and uses semantic-versioned release headings
for user-facing milestones.

## [1.1.0] - 2026-06-12

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

## [1.0.0] - 2026-05-26

### Highlights

- Renamed the Python distribution and import package to `umapers`.
- Bumped the Rust core crate and Python binding to version `1.0.0`.
- Includes the recent dense-distance, ANN, sparse kNN, transform, and memory optimizations that show significant speed and memory advantages over `umap-learn` in local benchmarks.

## [0.3.0] - 2026-04-03

### Highlights

- Published the Python binding on PyPI under the unified distribution name
  `umap-rs`, with the import path `umap_rs`.
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

- The Python distribution and import path are now unified as `umap-rs` and
  `umap_rs`.
- Published support is Python 3.9 through 3.13. Python 3.14 was probed in CI
  from sdist, but it is not yet part of the supported range.
