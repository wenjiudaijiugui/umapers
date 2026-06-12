# rust_umap

A ground-up Rust implementation of the core UMAP pipeline for reproducible algorithm study.

## What is implemented

- KNN search with multiple metrics:
  - Exact all-pairs baseline for `euclidean`, `manhattan`/`l1`, and `cosine`
  - Optional NN-Descent-style approximate mode for larger datasets
- Smooth KNN distance estimation (`sigma`, `rho`) with binary search
- Fuzzy simplicial set construction and symmetric fuzzy union
- Automatic `(a, b)` curve parameter fitting from `(spread, min_dist)`
- Two initialization strategies:
  - `InitMethod::Spectral` (normalized graph Laplacian eigenvectors)
  - `InitMethod::Random`
- Spectral initialization for disconnected graphs via component-aware layout
- Stochastic layout optimization with negative sampling
- End-to-end training APIs:
  - `fit_transform`
  - `fit`
- New point embedding API:
  - `transform`
- Sparse CSR input MVP:
  - `SparseCsrMatrix` + `fit_sparse_csr` / `fit_transform_sparse_csr`
  - Exact sparse kNN for `euclidean` / `manhattan` / `cosine` (no full dense distance matrix materialization)
  - Sparse-trained `transform` supports dense query input for `euclidean` / `manhattan` / `cosine`
  - `fit_csv` / `bench_fit_csv` support `--csr-indptr/--csr-indices/--csr-data/--csr-n-cols`
  - Sparse `inverse_transform` is intentionally unsupported in this MVP
- Approximate inverse mapping API:
  - `inverse_transform` (Euclidean metric in embedding space)
  - Exact training-embedding lookups are mapped back to their original samples
- CLI utilities:
  - `fit_csv`
  - `bench_fit_csv`
  - optional `--metric euclidean|manhattan|cosine`
  - optional `--knn-metric ...` guard in precomputed-kNN mode

## What is intentionally out of scope (for this version)

- Full `pynndescent`-equivalent ANN quality/performance parity
- Sparse-trained inverse transform

## Current scope boundary

The crate README and repository README use the same scope contract for the current documented surface:

- Supported core workflow: single-dataset non-parametric `UmapModel` / `Umap` fit, `fit_transform`, `transform`, and dense-trained `inverse_transform`
- Supported input modes: dense input, precomputed-kNN fit with metric match, and sparse CSR fit / `fit_transform_sparse_csr`
- Supported sparse follow-up path: dense-query `transform` after sparse training for `euclidean`, `manhattan`, and `cosine`

The following remain intentionally outside the current boundary:

- `inverse_transform` after sparse training
- Python binding parity for crate-only experimental or auxiliary surfaces such as parametric UMAP, aligned UMAP, CLI binaries, or benchmark helpers
- Full ANN parity with Python `umap-learn` + `pynndescent`

The repository-level Python binding is intentionally kept thin over these Rust
paths: Python normalizes I/O and forwards calls, while validation and compute
stay in Rust wherever practical.

See `../docs/adr/ADR-L8-scope-alignment.md` for the L8 alignment record.

## Quick start

```bash
cd rust_umap
cargo test
cargo run --release
```

`cargo run` executes a toy dataset embedding and then transforms a small query subset.

## CI-facing validation

Repository automation keeps benchmark validation staged by workflow/job:

1. `.github/workflows/ci.yml`: `rust-build-test` -> `consistency-smoke` -> `ann-e2e-smoke` -> `wave1-smoke` -> `no-regression-smoke` (euclidean/manhattan/cosine matrix).
2. `.github/workflows/ecosystem-python-binding.yml`: `binding-smoke-and-benchmark` for binding tests and ecosystem smoke/gate outputs.
3. `.github/workflows/deep-benchmark-report.yml`: optional manual/scheduled `optimization-report` after consistency and no-regression jobs.

The Rust crate remains the unit under test in all staged checks.

For Wave 3 release-prep convergence, run the repository-level orchestrator from the repo root:

```bash
PYTHON_BIN="$(command -v python3 || command -v python)"
$PYTHON_BIN benchmarks/release_prep_regression.py \
  --candidate-root "$(pwd -P)" \
  --baseline-root /path/to/umapers-baseline \
  --gate-config benchmarks/gate_thresholds.json \
  --metrics euclidean,manhattan,cosine \
  --output-json benchmarks/release-prep-regression.local.json
```

This wraps `wave1-smoke`, `ann-e2e-smoke`, `consistency-smoke`, and per-metric `no-regression-smoke`
into one summary artifact and exit code.

For CSV-driven runs:

```bash
cargo run --release --bin fit_csv -- \
  data.csv embedding.csv \
  15 2 200 42 spectral false 30 10 4096 fit \
  --metric cosine
```

For sparse CSR input (`fit` mode), pass CSR arrays directly:

```bash
cargo run --release --bin fit_csv -- \
  dummy.csv embedding.csv \
  15 2 200 42 random false 30 10 4096 fit \
  --metric euclidean \
  --csr-indptr indptr.csv \
  --csr-indices indices.csv \
  --csr-data values.csv \
  --csr-n-cols 1800
```

For precomputed kNN input, add explicit metric guard:

```bash
cargo run --release --bin fit_csv -- \
  data.csv embedding.csv \
  15 2 200 42 spectral false 30 10 4096 fit_precomputed \
  knn_idx.csv knn_dist.csv \
  --metric cosine \
  --knn-metric cosine
```

Precomputed-kNN contract (strict validation):

- each row length must be `>= n_neighbors` and index/dist row lengths must match
- indices must be in-range and unique within each row (duplicate indices are rejected)
- distances must be finite, `>= 0`, and non-decreasing within each row

To benchmark repeated fits:

```bash
cargo run --release --bin bench_fit_csv -- \
  data.csv embedding.csv \
  15 2 200 42 spectral false 30 10 4096 1 5 \
  --metric manhattan
```

To benchmark sparse CSR fits against `umap-learn` (consistency + speed + memory):

```bash
PYTHON_BIN="$(command -v python3 || command -v python)"
uv run --with numpy --with scipy --with scikit-learn --with umap-learn \
  "$PYTHON_BIN" rust_umap/benchmarks/eval_sparse_csr_vs_umap_learn.py --dataset all
```

## Minimal library usage

```rust
use rust_umap::{InitMethod, Metric, UmapModel, UmapParams};

let data: Vec<Vec<f32>> = vec![
    vec![0.0, 0.1, 0.2],
    vec![0.2, 0.1, 0.0],
    // ...
];

let mut model = UmapModel::new(UmapParams {
    init: InitMethod::Spectral,
    metric: Metric::Cosine,
    ..UmapParams::default()
});

let embedding = model.fit_transform(&data)?;
let new_points = vec![vec![0.1, 0.1, 0.2]];
let transformed = model.transform(&new_points)?;
let reconstructed = model.inverse_transform(&transformed)?;
```

Minimal sparse usage:

```rust
use rust_umap::{fit_transform_sparse_csr, Metric, SparseCsrMatrix, UmapParams};

let csr = SparseCsrMatrix::new(
    3,
    5,
    vec![0, 2, 3, 5],
    vec![0, 3, 1, 0, 4],
    vec![1.0, 2.0, 3.0, 4.0, 5.0],
)?;

let embedding = fit_transform_sparse_csr(
    csr,
    UmapParams {
        metric: Metric::Euclidean,
        ..UmapParams::default()
    },
)?;
```

## Notes on fidelity

The core equations for graph construction and layout optimization are aligned with standard UMAP behavior.
This implementation prioritizes clarity and reproducibility over large-scale performance.

## Inverse benchmark helpers

For inverse-transform quality and Euclidean fit no-regression checks:

```bash
PYTHON_BIN="$(command -v python3 || command -v python)"
"$PYTHON_BIN" rust_umap/benchmarks/eval_inverse_quality.py --dataset all
"$PYTHON_BIN" rust_umap/benchmarks/eval_euclidean_fit_regression.py --dataset all \
  --output-json rust_umap/benchmarks/current_fit_regression.json
"$PYTHON_BIN" rust_umap/benchmarks/eval_euclidean_fit_regression.py --dataset all \
  --baseline-report rust_umap/benchmarks/current_fit_regression.json
```

`eval_inverse_quality.py` now fails on asymmetric success/failure by default to avoid biased comparisons.
