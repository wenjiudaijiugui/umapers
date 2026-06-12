# Runtime vs Dataset Size

![Runtime vs dataset size](runtime_vs_dataset_size.svg)

| dataset | samples | features | matrix size | umap-rs sec | umap-learn sec | rs / learn |
|---|---:|---:|---:|---:|---:|---:|
| `iris` | 150 | 4 | 600 | 0.043 +/- 0.005 | 0.055 +/- 0.003 | 0.78x |
| `wine` | 178 | 13 | 2,314 | 0.053 +/- 0.002 | 0.068 +/- 0.002 | 0.78x |
| `breast_cancer` | 569 | 30 | 17,070 | 0.135 +/- 0.003 | 0.273 +/- 0.006 | 0.49x |
| `digits` | 1797 | 64 | 115,008 | 0.411 +/- 0.004 | 4.433 +/- 3.894 | 0.09x |
