# Runtime vs Dataset Size

![Runtime vs dataset size](runtime_vs_dataset_size.svg)

| dataset | samples | features | matrix size | umapers sec | umap-learn sec | rs / learn |
|---|---:|---:|---:|---:|---:|---:|
| `iris` | 150 | 4 | 600 | 0.040 +/- 0.004 | 0.056 +/- 0.003 | 0.72x |
| `wine` | 178 | 13 | 2,314 | 0.050 +/- 0.001 | 0.060 +/- 0.001 | 0.82x |
| `breast_cancer` | 569 | 30 | 17,070 | 0.132 +/- 0.002 | 0.269 +/- 0.006 | 0.49x |
| `digits` | 1797 | 64 | 115,008 | 0.382 +/- 0.005 | 4.241 +/- 3.687 | 0.09x |
