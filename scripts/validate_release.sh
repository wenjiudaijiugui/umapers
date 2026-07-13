#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python3 || command -v python || true)"
fi
if [[ -z "$PYTHON_BIN" ]]; then
  echo "python3/python not found; set PYTHON_BIN" >&2
  exit 1
fi

VALIDATION_VENV="${VALIDATION_VENV:-.venv}"
VENV_PYTHON="$VALIDATION_VENV/bin/python"
DIST_DIR="$(mktemp -d "${TMPDIR:-/tmp}/umapers-release-dist.XXXXXX")"
trap 'rm -rf "$DIST_DIR"' EXIT

cargo fmt --manifest-path rust_umap/Cargo.toml --check
cargo fmt --manifest-path umap_rs/Cargo.toml --check
cargo test --release --locked --manifest-path rust_umap/Cargo.toml
cargo test --release --locked --manifest-path umap_rs/Cargo.toml
cargo clippy --locked --manifest-path rust_umap/Cargo.toml --all-targets -- -D warnings
cargo clippy --locked --manifest-path umap_rs/Cargo.toml --all-targets -- -D warnings

if command -v uv >/dev/null 2>&1; then
  uv venv --python "$PYTHON_BIN" "$VALIDATION_VENV"
  uv pip install --python "$VENV_PYTHON" --upgrade pip maturin==1.14.1 pytest twine==6.2.0 'tomli; python_version < "3.11"'
  uv pip install --python "$VENV_PYTHON" -r benchmarks/requirements-bench.txt
  "$VENV_PYTHON" scripts/check_release_version.py
  uv run --python "$VENV_PYTHON" maturin develop --release --locked --manifest-path umap_rs/Cargo.toml
  uv run --python "$VENV_PYTHON" python -I -m pytest -q umap_rs/tests/test_binding.py
  uv run --python "$VENV_PYTHON" python -m pytest -q benchmarks/tests/test_release_prep_regression.py
  uv run --python "$VENV_PYTHON" maturin sdist --manifest-path umap_rs/Cargo.toml --out "$DIST_DIR"
  uv run --python "$VENV_PYTHON" maturin build --release --locked --manifest-path umap_rs/Cargo.toml --out "$DIST_DIR"
  uv run --python "$VENV_PYTHON" python -m twine check "$DIST_DIR"/*
else
  "$PYTHON_BIN" -m venv "$VALIDATION_VENV"
  "$VENV_PYTHON" -m pip install --upgrade pip maturin==1.14.1 pytest twine==6.2.0 'tomli; python_version < "3.11"'
  "$VENV_PYTHON" -m pip install -r benchmarks/requirements-bench.txt
  "$VENV_PYTHON" scripts/check_release_version.py
  VIRTUAL_ENV="$VALIDATION_VENV" PATH="$ROOT_DIR/$VALIDATION_VENV/bin:$PATH" \
    maturin develop --release --locked --manifest-path umap_rs/Cargo.toml
  "$VENV_PYTHON" -I -m pytest -q umap_rs/tests/test_binding.py
  "$VENV_PYTHON" -m pytest -q benchmarks/tests/test_release_prep_regression.py
  VIRTUAL_ENV="$VALIDATION_VENV" PATH="$ROOT_DIR/$VALIDATION_VENV/bin:$PATH" \
    maturin sdist --manifest-path umap_rs/Cargo.toml --out "$DIST_DIR"
  VIRTUAL_ENV="$VALIDATION_VENV" PATH="$ROOT_DIR/$VALIDATION_VENV/bin:$PATH" \
    maturin build --release --locked --manifest-path umap_rs/Cargo.toml --out "$DIST_DIR"
  "$VENV_PYTHON" -m twine check "$DIST_DIR"/*
fi

"$VENV_PYTHON" scripts/check_python_dist.py --dist-dir "$DIST_DIR"
