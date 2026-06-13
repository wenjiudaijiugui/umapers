# umapers 1.1.3

`umapers 1.1.3` is a follow-up patch for issues that were only partially fixed
in `1.1.2`.

## Fixes

- Improves spectral initialization quality for low-dimensional nonlinear
  manifolds above the exact-spectral cutoff. The iterative connected-graph path
  now uses a wider subspace and more iterations, removing the Swiss Roll
  silhouette cliff observed just above 2048 samples.
- Aligns densMAP/output-density fitted attributes with `umap-learn` semantics:
  density radii are exposed as `rad_orig_` and `rad_emb_` only when
  `output_dens=True`, with the existing `radii_original_` and
  `radii_embedding_` aliases kept in the same lifecycle.
- Extends the pyright usage probe to cover additional top-level exports and
  wrapper workflows, rather than only the basic `Umap` import path.

## Validation Scope

This patch has been checked with targeted real comparisons, not only unit
tests:

- Swiss Roll at 2049, 3000, and 8000 samples against `umap-learn`
- S-curve, moons, circles, and digits against `umap-learn`
- densMAP/output-density return values and fitted attributes against
  `umap-learn`
- pyright checks in an installed virtual environment

Before publishing, rerun the release gate and the broader runtime reports to
confirm the wider iterative spectral subspace has not regressed large-data
scaling.
