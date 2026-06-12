# Runtime vs Dataset Size

![Runtime vs dataset size](runtime_vs_dataset_size.svg)

| dataset | samples | features | matrix size | umapers sec | umap-learn sec | rs / learn |
|---|---:|---:|---:|---:|---:|---:|
| `iris` | 150 | 4 | 600 | 0.041 +/- 0.004 | 0.057 +/- 0.004 | 0.73x |
| `wine` | 178 | 13 | 2,314 | 0.051 +/- 0.003 | 0.061 +/- 0.002 | 0.84x |
| `breast_cancer` | 569 | 30 | 17,070 | 0.129 +/- 0.001 | 0.259 +/- 0.004 | 0.50x |
| `digits` | 1797 | 64 | 115,008 | 0.383 +/- 0.007 | 4.156 +/- 3.672 | 0.09x |
