# Synthetic Large Runtime Probe

- generated_at: `2026-06-12T19:49:40+0800`
- python: `3.13.9`
- umapers: `1.1.1`
- umap_learn: `0.5.12`
- build_profile: `release`
- seed: `13`
- UMAP: `n_neighbors=15`, `n_epochs=150`, `init=spectral`, `metric=euclidean`
- warmup: `small path plus discarded >4096-sample large path`

![Synthetic large runtime probe](synthetic_large_runtime_probe.svg)

| dataset | samples | features | matrix size | rs auto sec | auto kNN | auto opt | auto ANN | rs approx sec | rs exact sec | learn sec | auto / learn | auto / exact |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| `class_4200x96` | 4200 | 96 | 403,200 | 0.636 | 0.025 | 0.576 | `no` | 0.699 | 0.613 | 0.938 | 0.68x | 1.04x |
| `class_5000x112` | 5000 | 112 | 560,000 | 0.752 | 0.033 | 0.683 | `no` | 0.837 | 0.740 | 1.130 | 0.67x | 1.02x |
| `class_6000x128` | 6000 | 128 | 768,000 | 0.925 | 0.051 | 0.840 | `no` | 1.016 | 0.883 | 1.343 | 0.69x | 1.05x |
| `class_8000x128` | 8000 | 128 | 1,024,000 | 1.220 | 0.071 | 1.105 | `no` | 1.343 | 1.194 | 1.817 | 0.67x | 1.02x |
| `class_10000x160` | 10000 | 160 | 1,600,000 | 1.580 | 0.103 | 1.425 | `no` | 1.718 | 1.531 | 2.327 | 0.68x | 1.03x |
| `class_16000x160` | 16000 | 160 | 2,560,000 | 2.562 | 0.228 | 2.248 | `no` | 2.707 | 2.621 | 3.834 | 0.67x | 0.98x |
| `class_20000x192` | 20000 | 192 | 3,840,000 | 3.392 | 0.299 | 2.982 | `yes` | 3.382 | 3.345 | 4.659 | 0.73x | 1.01x |
