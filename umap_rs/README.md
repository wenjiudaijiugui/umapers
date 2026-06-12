# umapers

`umapers` is a Python UMAP library for dimensionality reduction, clustering workflows, and embedding visualization.

`umap-learn` is the established Python implementation used as the compatibility and benchmark baseline for this project. For matching parameters and workflows, `umapers` keeps the user-visible semantics familiar while running its own fitting and transform code instead of importing `umap-learn` at runtime.

The public API is intentionally small: fit an embedding, transform new rows, check quality, and reproduce runs with fixed seeds.

## Installation

```bash
pip install umapers
```

Optional extras for plotting and diagnostics:

```bash
pip install "umapers[plot,diagnostics]"
```

## Quickstart

```python
import numpy as np
from umapers import Umap

x = np.random.default_rng(42).normal(size=(1000, 32)).astype(np.float32)

embedding = Umap(
    n_neighbors=15,
    n_components=2,
    n_epochs=200,
    random_seed=42,
).fit_transform(x)
```

Fit once and transform new rows:

```python
model = Umap(random_seed=42).fit(x)
query_embedding = model.transform(x[:10])
```

Use labels to run categorical supervised UMAP:

```python
labels = np.random.default_rng(7).integers(0, 4, size=x.shape[0])

embedding = Umap(
    target_metric="categorical",
    target_weight=0.5,
    random_seed=42,
).fit_transform(x, y=labels)
```

## Results

The `1.1.1` release reports compare `umapers` with `umap-learn` on feature coverage, clustering quality, and runtime. On the real datasets tested here, downstream clustering quality is similar. On the synthetic scaling benchmarks, `umapers` is strongest on medium and large matrices; for very small datasets, fixed overhead can make `umap-learn` faster.

- Clustering quality: [current_clustering_analysis_report.md](https://github.com/wenjiudaijiugui/umapers/blob/main/reports/current_clustering_analysis_report.md)
- Runtime scaling: [synthetic_runtime_scaling_report.md](https://github.com/wenjiudaijiugui/umapers/blob/main/reports/synthetic_runtime_scaling_report.md)
- Large-data probe: [synthetic_large_runtime_probe.md](https://github.com/wenjiudaijiugui/umapers/blob/main/reports/synthetic_large_runtime_probe.md)
- Real-data runtime: [runtime_vs_dataset_size.md](https://github.com/wenjiudaijiugui/umapers/blob/main/reports/runtime_vs_dataset_size.md)
- Feature parity: [current_feature_parity_report.md](https://github.com/wenjiudaijiugui/umapers/blob/main/reports/current_feature_parity_report.md)

## Supported Workflows

| Need | API |
|---|---|
| Fit a 2D or nD embedding | `Umap(...).fit_transform(data)` |
| Fit once, then transform new rows | `model.fit(data)` then `model.transform(query)` |
| Approximate inverse mapping | `model.inverse_transform(embedding)` |
| Categorical supervised embedding | `target_metric="categorical"`, pass `y=` |
| Dense densMAP radii | `densmap=True`, `output_dens=True` |
| Sparse input | pass CSR/CSC/COO input to `fit` or `fit_transform` |
| Plot an embedding | `umapers.plot.points(...)` |
| Compute trustworthiness | `umapers.diagnostics.trustworthiness_report(...)` |
| Reuse a precomputed kNN graph | `Umap.fit_transform_with_knn(...)` |
| Parametric convenience workflow | `ParametricUmap` |
| Aligned convenience workflow | `AlignedUmap` |

## Known Gaps

- sparse-trained `inverse_transform` is not supported yet
- `ParametricUmap` is a lightweight convenience workflow, not a Keras/TensorFlow replacement
- `AlignedUmap` covers only part of the relation modes available in `umap-learn`
- approximate-neighbor behavior is not identical to `pynndescent`
- arbitrary precomputed graph input is not exposed as a general public API

## Source Build

```bash
pip install --upgrade pip maturin
maturin develop --release --manifest-path umap_rs/Cargo.toml
python -I -m pytest -q umap_rs/tests/test_binding.py
```
