# umapers 1.1.2

`umapers 1.1.2` is a patch release for three user-facing gaps found after the
1.1.1 cleanup release.

## Fixes

- Improves Swiss Roll and similar low-dimensional nonlinear manifolds by using
  exact spectral initialization for connected datasets up to `2048 x 8`, while
  retaining the oversampled iterative spectral path for larger graphs.
- Exposes densMAP/output-density fitted radii as `rad_orig_` and `rad_emb_`,
  matching the attribute names users expect from `umap-learn`. The existing
  `radii_original_` and `radii_embedding_` aliases remain available.
- Fixes top-level Python type stub re-exports so pyright resolves
  `Umap`, `UmapKwargs`, and `fit_transform` from installed virtual
  environments. The top-level `fit_transform(..., output_dens=True)` overload
  now resolves to the density-output tuple.

## 1.1 Release Context

The 1.1 line turns the previous compatibility work into a broader, measured
Python/Rust surface and fixes the benchmark interpretation issues that
previously made large-data runtime look worse than it was.

## Highlights

- Broader Python API:
  - categorical supervised and semi-supervised dense UMAP
  - additional dense metrics and sparse-compatible metrics
  - dense densMAP/output-density support
  - sparse CSR/CSC/COO inputs through the Python wrapper
  - aligned and parametric convenience wrappers
  - optional plotting and trustworthiness helpers
- Performance and quality:
  - spectral initialization now uses exact eigendecomposition for low-dimensional
    manifolds where it protects layout quality, and an oversampled iterative path
    where dense eigendecomposition would dominate runtime
  - auto ANN selection is delayed until the measured crossover point instead of
    switching at the public threshold too early
  - standard 2D layout optimization now uses a contiguous flat embedding path
  - ANN updates are parallelized and have recall diagnostics
- Release evidence:
  - feature parity report for direct and ecosystem scenarios
  - real-data clustering report with ARI/NMI/AMI/homogeneity/completeness/
    V-measure/silhouette visualization
  - small-dataset hot-spot report
  - 30-dataset synthetic runtime scaling report
  - large synthetic runtime probe up to `20000 x 192`

## Current Local Evidence

- `reports/current_feature_parity_report.md`
- `reports/current_clustering_analysis_report.md`
- `reports/current_clustering_metrics.svg`
- `reports/synthetic_runtime_scaling_report.md`
- `reports/synthetic_large_runtime_probe.md`
- `reports/small_dataset_hotspot_report.md`

The latest local release-build synthetic scaling report shows `umapers` faster
than `umap-learn` on 29 / 30 datasets. The largest dataset in that report,
`4200 x 96`, runs at `0.67x` of `umap-learn` time. The larger probe from
`4200 x 96` to `20000 x 192` reports `0.68x` to `0.73x` of `umap-learn` time.

## Install

```bash
pip install umapers
```

```python
from umapers import Umap, fit_transform
```

## Validation

The local release gate is:

```bash
scripts/validate_release.sh
```

Before publishing, build the Python release artifacts and check their metadata:

```bash
maturin build --release --sdist --manifest-path umap_rs/Cargo.toml --out dist
uvx twine check dist/*
```
