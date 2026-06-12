# Small Dataset Hotspot Report

- generated_at: `2026-06-12T18:49:25+0800`
- dataset: `sklearn.load_wine` `178 x 13`
- repeats: `80`, warmup: `10`
- UMAP: `n_neighbors=15`, `n_epochs=200`, `init=spectral`, `ann_mode=auto`
- semantic max_abs_diff_vs_fit_transform: `0`
- memory VmRSS before/after: `132.66015625` / `132.91015625` MB
- memory VmHWM before/after: `132.66015625` / `132.91015625` MB

![Small dataset hotspot breakdown](small_dataset_hotspot_breakdown.svg)

## Grouped Hotspots

| group | mean ms | 95% CI +/- ms | share of pipeline |
|---|---:|---:|---:|
| `layout_optimize` | 29.568 | 0.224 | 56.1% |
| `curve_params` | 10.842 | 0.227 | 20.6% |
| `kmeans` | 9.022 | 0.780 | 17.1% |
| `knn` | 1.063 | 0.123 | 2.0% |
| `graph_prepare` | 0.341 | 0.006 | 0.6% |
| `spectral_init` | 0.284 | 0.006 | 0.5% |
| `output_store` | 0.013 | 0.001 | 0.0% |
| `validation` | 0.001 | 0.000 | 0.0% |

## Stage Timings

| stage | mean ms | std ms | 95% CI +/- ms | median ms | p05 ms | p95 ms | CV | share of pipeline |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `pipeline_total_sec` | 52.714 | 4.394 | 0.963 | 52.632 | 47.388 | 57.123 | 0.083 | 100.0% |
| `total_sec` | 42.111 | 1.695 | 0.371 | 41.875 | 39.990 | 45.091 | 0.040 | 79.9% |
| `optimize_sec` | 29.568 | 1.024 | 0.224 | 29.067 | 28.621 | 31.400 | 0.035 | 56.1% |
| `curve_params_sec` | 10.842 | 1.037 | 0.227 | 10.829 | 9.612 | 12.403 | 0.096 | 20.6% |
| `kmeans_sec` | 9.022 | 3.559 | 0.780 | 9.509 | 4.966 | 11.537 | 0.394 | 17.1% |
| `knn_sec` | 1.063 | 0.560 | 0.123 | 1.022 | 0.603 | 1.359 | 0.527 | 2.0% |
| `init_sec` | 0.284 | 0.030 | 0.006 | 0.273 | 0.265 | 0.361 | 0.104 | 0.5% |
| `smooth_knn_sec` | 0.144 | 0.020 | 0.004 | 0.136 | 0.135 | 0.173 | 0.140 | 0.3% |
| `symmetrize_sec` | 0.134 | 0.012 | 0.003 | 0.128 | 0.126 | 0.162 | 0.092 | 0.3% |
| `knn_validate_trim_sec` | 0.052 | 0.010 | 0.002 | 0.049 | 0.043 | 0.069 | 0.197 | 0.1% |
| `output_copy_sec` | 0.009 | 0.004 | 0.001 | 0.009 | 0.007 | 0.014 | 0.431 | 0.0% |
| `membership_sec` | 0.009 | 0.006 | 0.001 | 0.008 | 0.008 | 0.012 | 0.648 | 0.0% |
| `store_state_sec` | 0.003 | 0.001 | 0.000 | 0.003 | 0.002 | 0.004 | 0.189 | 0.0% |
| `prune_sec` | 0.002 | 0.000 | 0.000 | 0.002 | 0.002 | 0.003 | 0.105 | 0.0% |
| `validation_sec` | 0.001 | 0.000 | 0.000 | 0.001 | 0.001 | 0.002 | 0.234 | 0.0% |
| `density_embedding_sec` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.813 | 0.0% |
| `density_original_sec` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.093 | 0.0% |
| `target_intersection_sec` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | nan | 0.0% |

## Quality

| metric | mean | std |
|---|---:|---:|
| `adjusted_rand` | 0.774176 | 0.000000 |
| `normalized_mutual_info` | 0.774016 | 0.000000 |
