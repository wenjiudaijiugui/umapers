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

cargo test --manifest-path rust_umap/Cargo.toml
cargo test --manifest-path umap_rs/Cargo.toml
cargo clippy --manifest-path rust_umap/Cargo.toml --all-targets -- -D warnings
cargo clippy --manifest-path umap_rs/Cargo.toml --all-targets -- -D warnings

if command -v uv >/dev/null 2>&1; then
  uv venv --python "$PYTHON_BIN" "$VALIDATION_VENV"
  uv pip install --python "$VENV_PYTHON" --upgrade pip maturin pytest
  uv pip install --python "$VENV_PYTHON" -r benchmarks/requirements-bench.txt
  uv run --python "$VENV_PYTHON" maturin develop --release --manifest-path umap_rs/Cargo.toml
  uv run --python "$VENV_PYTHON" python -I -m pytest -q umap_rs/tests/test_binding.py
else
  "$PYTHON_BIN" -m venv "$VALIDATION_VENV"
  "$VENV_PYTHON" -m pip install --upgrade pip maturin pytest
  "$VENV_PYTHON" -m pip install -r benchmarks/requirements-bench.txt
  VIRTUAL_ENV="$VALIDATION_VENV" PATH="$ROOT_DIR/$VALIDATION_VENV/bin:$PATH" \
    maturin develop --release --manifest-path umap_rs/Cargo.toml
  "$VENV_PYTHON" -I -m pytest -q umap_rs/tests/test_binding.py
fi
