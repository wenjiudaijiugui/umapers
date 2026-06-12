# Current umap-rs Feature Parity Report

- generated_at: `2026-06-12T18:55:40+0800`
- python: `3.13.9 (main, Oct 31 2025, 23:02:44) [Clang 21.1.4 ]`
- umapers: `1.1.0`
- umap_learn: `0.5.12`

## Scenario Summary

| scenario | comparability | umap-rs | umap-learn | key metric |
|---|---|---:|---:|---|
| `dense_fit_transform_digits` | direct | ok | ok | rs trust=0.944; learn trust=0.947 |
| `transform_inverse_digits` | direct | ok | ok | rs acc=0.922; learn acc=0.922 |
| `categorical_supervised_wine` | direct | ok | ok | rs trust=0.918; learn trust=0.915 |
| `expanded_metric_sweep_breast_cancer` | direct | ok | ok |  |
| `densmap_output_breast_cancer` | direct | ok | ok | rs trust=0.888; learn trust=0.888 |
| `sparse_digits_csr_csc_coo` | direct | ok | ok | rs trust=0.959; learn trust=0.955 |
| `precomputed_knn_wine` | library-native precomputed contracts, not byte-identical graph input | ok | ok | rs trust=0.938; learn trust=0.938 |
| `approximate_ann_digits` | algorithmic ANN comparison; backends differ | ok | ok | rs trust=0.939; learn trust=0.947; rs ann recall=0.973 |
| `parametric_breast_cancer` | optional extension comparison; umap-learn requires TensorFlow stack | ok | fail: RuntimeError |  |
| `aligned_digits_batches` | aligned API semantics differ but relation task is equivalent | ok | ok |  |
| `ecosystem_helpers_wine` | ecosystem utility comparison | ok | ok |  |
| `sparse_inverse_boundary` | boundary behavior | ok | ok |  |

## Notes

- Timing is sequential side-by-side timing to avoid CPU contention inside one scenario.
- Precomputed-kNN comparison uses each library's native contract, because the accepted self-neighbor convention differs.
- Optional extension failures are reported as environment facts, not counted as core algorithm failures.
