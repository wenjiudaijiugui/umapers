# Synthetic Large Runtime Probe

- generated_at: `2026-06-12T18:52:19+0800`
- python: `3.13.9`
- umapers: `1.1.0`
- umap_learn: `0.5.12`
- build_profile: `release`
- seed: `13`
- UMAP: `n_neighbors=15`, `n_epochs=150`, `init=spectral`, `metric=euclidean`
- warmup: `small path plus discarded >4096-sample large path`

![Synthetic large runtime probe](synthetic_large_runtime_probe.svg)

| dataset | samples | features | matrix size | rs auto sec | auto kNN | auto opt | auto ANN | rs approx sec | rs exact sec | learn sec | auto / learn | auto / exact |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| `class_4200x96` | 4200 | 96 | 403,200 | 0.635 | 0.023 | 0.583 | `no` | 0.698 | 0.638 | 0.967 | 0.66x | 1.00x |
| `class_5000x112` | 5000 | 112 | 560,000 | 0.757 | 0.035 | 0.690 | `no` | 0.843 | 0.766 | 1.130 | 0.67x | 0.99x |
| `class_6000x128` | 6000 | 128 | 768,000 | 0.915 | 0.052 | 0.830 | `no` | 1.040 | 0.896 | 1.362 | 0.67x | 1.02x |
| `class_8000x128` | 8000 | 128 | 1,024,000 | 1.230 | 0.074 | 1.114 | `no` | 1.347 | 1.210 | 1.796 | 0.68x | 1.02x |
| `class_10000x160` | 10000 | 160 | 1,600,000 | 1.556 | 0.115 | 1.388 | `no` | 1.688 | 1.560 | 2.364 | 0.66x | 1.00x |
| `class_16000x160` | 16000 | 160 | 2,560,000 | 2.575 | 0.249 | 2.241 | `no` | 2.705 | 2.560 | 3.834 | 0.67x | 1.01x |
| `class_20000x192` | 20000 | 192 | 3,840,000 | 3.390 | 0.311 | 2.969 | `yes` | 3.422 | 3.434 | 4.829 | 0.70x | 0.99x |
