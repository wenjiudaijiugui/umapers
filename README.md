# umap-rs

[中文版本](README_CN.md)

A Rust-first UMAP implementation with fairness-controlled benchmarking against
Python `umap-learn`.

The repository includes:

- a Rust library and CLI under `rust_umap/`
- reproducible benchmark harnesses under `benchmarks/`
- the `umapers` Python package under `umap_rs/`

## Repository layout

- `rust_umap/`: Rust UMAP crate and CLI binaries
- `umap_rs/`: PyO3 + maturin Python binding published/imported as `umapers`
- `benchmarks/`: fairness-oriented benchmark scripts and reports
- `reports/`: generated benchmark and regression artifacts
- `UMAP_MATHEMATICAL_DOCUMENTATION*.md`: mathematical notes

## Quick start

```bash
cd rust_umap
cargo build --release
cargo test
```

## Python binding

Version `1.0.0` of the Python binding focuses on IDE discoverability:

- public entrypoints ship useful type hints and docstrings
- editor hover and `help()` now describe the main call patterns
- the API surface stays intentionally small and layered

The Python binding is still intentionally thin:

- Python normalizes dense arrays and CSR inputs
- Rust owns validation and compute-heavy paths whenever practical
- precomputed kNN is available, but it is an advanced interface rather than the
  default path

### Install locally

```bash
PYTHON_BIN="$(command -v python3 || command -v python)"
if [ -z "$PYTHON_BIN" ]; then
  echo "python3/python not found" >&2
  exit 1
fi

uv venv --python "$PYTHON_BIN" .venv
uv pip install --python .venv/bin/python --upgrade pip maturin
uv run --python .venv/bin/python maturin develop --manifest-path umap_rs/Cargo.toml
```

### API layers

#### Main API

This is the stable public API that most users should learn first.

- `from umapers import Umap, fit_transform`
- `Umap.fit(data)`
- `Umap.fit_transform(data, out=None)`
- `Umap.transform(query, out=None)`
- `Umap.inverse_transform(embedded_query, out=None)`
- `fit_transform(data, **kwargs)`

Typical dense example:

```python
import numpy as np
from umapers import Umap

rng = np.random.default_rng(42)
x = rng.normal(size=(400, 16)).astype(np.float32)

model = Umap(
    n_neighbors=15,
    n_components=2,
    n_epochs=120,
    metric="euclidean",
    random_seed=42,
    init="random",
)

emb = model.fit_transform(x)
print("embedding shape:", emb.shape, "dtype:", emb.dtype)
```

The main API also supports:

- CSR sparse input for `fit` and `fit_transform`
- `out=` buffers for `fit_transform`, `transform`, and `inverse_transform`
- `ann_mode="auto" | "exact" | "approximate"` as a Python convenience layer

#### Advanced API

`Umap.fit_transform_with_knn(...)` is a public advanced interface.

Use it when you already have an exact or shared kNN graph and want to:

- run fairness-controlled benchmarks
- reuse the same kNN graph across parameter sweeps
- integrate with an external nearest-neighbor pipeline

It expects:

- `data`: shape `(n_samples, n_features)`, converted to `float32`
- `knn_indices`: shape `(n_samples, k)`, converted to `int64`
- `knn_dists`: shape `(n_samples, k)`, converted to `float32`
- `knn_metric`: must match the model metric

Example with a shared exact kNN graph:

```python
import numpy as np
from sklearn.neighbors import NearestNeighbors
from umapers import Umap

x = np.random.default_rng(42).normal(size=(300, 16)).astype(np.float32)
k = 15

nbrs = NearestNeighbors(
    n_neighbors=k + 1,
    algorithm="brute",
    metric="euclidean",
    n_jobs=1,
)
nbrs.fit(x)
dists, idx = nbrs.kneighbors(x)

knn_indices = idx[:, 1 : k + 1].astype(np.int64)
knn_dists = dists[:, 1 : k + 1].astype(np.float32)

model = Umap(
    n_neighbors=k,
    n_components=2,
    metric="euclidean",
    random_seed=42,
    init="random",
    use_approximate_knn=False,
)

emb = model.fit_transform_with_knn(
    x,
    knn_indices,
    knn_dists,
    knn_metric="euclidean",
)
```

This interface is intentionally narrower than a generic graph API:

- it accepts precomputed kNN, not arbitrary sparse graphs
- it is useful, but not the recommended quickstart path
- for strict binding comparisons, treat `algo_exact_shared_knn_exact` as the
  fairness anchor

#### Internal API

The following are internal implementation details and do not carry a public
compatibility guarantee:

- `umapers._umapers.UmapCore`
- `umapers._api`
- helper functions and `_`-prefixed symbols inside the binding package

## Current scope boundary

The documented and benchmarked Python binding surface currently includes:

- dense single-dataset `Umap` workflows
- dense-trained `inverse_transform`
- precomputed kNN fit for dense input as an advanced API
- CSR sparse `fit` / `fit_transform` MVP
- dense-query `transform` after sparse training for `euclidean`, `manhattan`,
  and `cosine`

The current boundary is intentionally narrower than full `umap-learn` parity:

- sparse-trained `inverse_transform` is not supported yet
- the Python package does not expose parametric UMAP or aligned UMAP
- ANN quality and performance parity with `pynndescent` is still out of scope
- the binding does not currently expose a generic graph-input API

See `docs/adr/ADR-L8-scope-alignment.md` for the repo-level scope decision.

## Benchmark reports

The ecosystem binding comparison report is available at:

- `benchmarks/report_ecosystem_python_binding.md`
- `benchmarks/report_ecosystem_python_binding.json`

To run the ecosystem comparison locally:

```bash
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || command -v python)}"
"$PYTHON_BIN" benchmarks/compare_ecosystem_python_binding.py \
  --python-bin "$PYTHON_BIN" \
  --warmup 1 \
  --repeats 3 \
  --sample-cap-consistency 2000
```

Outputs:

- `benchmarks/report_ecosystem_python_binding.json`
- `benchmarks/report_ecosystem_python_binding.md`

`e2e_mixed_knn_strategy` is useful for ecosystem smoke. For the strictest
binding comparison, prefer `algo_exact_shared_knn_exact`.

To run the core Rust-vs-Python fairness harness locally:

```bash
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || command -v python)}"
"$PYTHON_BIN" benchmarks/compare_real_impls_fair.py \
  --python-bin "$PYTHON_BIN" \
  --warmup 1 \
  --repeats 3 \
  --sample-cap-consistency 2000
```

## Local validation

```bash
PYTHON_BIN="$(command -v python3 || command -v python)"

cargo test --manifest-path rust_umap/Cargo.toml

uv venv --python "$PYTHON_BIN" .venv
uv pip install --python .venv/bin/python --upgrade pip
uv pip install --python .venv/bin/python -r benchmarks/requirements-bench.txt pytest maturin

uv run --python .venv/bin/python python -m py_compile \
  benchmarks/compare_real_impls_fair.py \
  benchmarks/compare_ecosystem_python_binding.py \
  benchmarks/run_umap_rs.py \
  benchmarks/run_umap_rs_algo.py

uv run --python .venv/bin/python maturin develop --manifest-path umap_rs/Cargo.toml
uv run --python .venv/bin/python python -I -m pytest -q umap_rs/tests/test_binding.py
```

For the full local regression and release-prep workflow, see the benchmark
scripts under `benchmarks/` and the generated reports under `reports/`.
