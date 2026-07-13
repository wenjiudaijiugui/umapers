# umapers 1.1.4

`umapers 1.1.4` is a performance and release-hardening patch. It keeps the
public API stable while reducing repeated work in core, sparse, parametric,
aligned, transform, and inverse-transform paths.

## Performance

- Reuses fitted curve parameters and precomputes the curve-search powers,
  reducing the parameter-solve microbenchmark by about 79%.
- Reuses spectral scratch buffers while keeping the 2047-2050 sample boundary
  outputs bit-for-bit equal to `1.1.3`.
- Parallelizes independent transform heads and dense-to-sparse queries. The
  measured transform path is about 3.3 times faster and the sparse query probe
  is at least 5.9 times faster on the release test machine.
- Improves contiguous access in `ParametricUmap` and bounds `AlignedUmap`
  concurrency while avoiding unnecessary retained model state.

## Reproducibility Note

Fitted embeddings remain bit-for-bit equal to `1.1.3` across the release
regression matrix. `transform` uses independent per-query random streams in
this release, so the same seed does not reproduce the exact transformed
coordinates produced by `1.1.3`. Results within `1.1.4` remain bit-for-bit
stable across repeated calls and across one or four Rayon threads; held-out
neighborhood quality did not regress in the release validation.

## Packaging and Release

- Python support is explicitly bounded to CPython 3.9 through 3.13 until the
  PyO3 dependency is upgraded for Python 3.14.
- Wheels and the source distribution include the full BSD 3-Clause license.
- The release workflow validates the tag against all package and lock-file
  versions, smokes every built wheel on its target platform, publishes through
  PyPI Trusted Publishing, and creates the matching GitHub Release assets.

## Validation Scope

- Rust release tests, Python binding tests, Clippy, formatting, and benchmark
  regression gates
- Swiss Roll around the 2048/2049 spectral boundary across multiple seeds
- circles, digits, moons, S-curve, and Swiss Roll against `umap-learn`
- isolated wheel and source-distribution builds plus install smoke tests
