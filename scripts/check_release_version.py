#!/usr/bin/env python3
"""Validate release version consistency across manifests, locks, and docs."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.9/3.10 helper path
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise SystemExit("Python 3.11+ or the 'tomli' package is required") from exc


ROOT = Path(__file__).resolve().parents[1]
TAG_RE = re.compile(r"^umapers-v(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$")


def load_toml(relative_path: str) -> dict[str, Any]:
    with (ROOT / relative_path).open("rb") as handle:
        return tomllib.load(handle)


def lock_package_version(relative_path: str, package_name: str) -> str:
    packages = load_toml(relative_path).get("package", [])
    matches = [package for package in packages if package.get("name") == package_name]
    if len(matches) != 1:
        raise ValueError(
            f"{relative_path}: expected one {package_name!r} package, found {len(matches)}"
        )
    version = matches[0].get("version")
    if not isinstance(version, str):
        raise ValueError(f"{relative_path}: {package_name!r} has no string version")
    return version


def release_version() -> str:
    manifest_versions = {
        "rust_umap/Cargo.toml": load_toml("rust_umap/Cargo.toml")["package"]["version"],
        "umap_rs/Cargo.toml": load_toml("umap_rs/Cargo.toml")["package"]["version"],
        "umap_rs/pyproject.toml": load_toml("umap_rs/pyproject.toml")["project"]["version"],
    }
    lock_versions = {
        "rust_umap/Cargo.lock:rust_umap": lock_package_version(
            "rust_umap/Cargo.lock", "rust_umap"
        ),
        "umap_rs/Cargo.lock:rust_umap": lock_package_version(
            "umap_rs/Cargo.lock", "rust_umap"
        ),
        "umap_rs/Cargo.lock:umapers": lock_package_version(
            "umap_rs/Cargo.lock", "umapers"
        ),
        "umap_rs/uv.lock:umapers": lock_package_version("umap_rs/uv.lock", "umapers"),
    }
    versions = {**manifest_versions, **lock_versions}
    unique_versions = set(versions.values())
    if len(unique_versions) != 1:
        details = "\n".join(f"  {source}: {version}" for source, version in versions.items())
        raise ValueError(f"release versions disagree:\n{details}")

    version = unique_versions.pop()
    if not re.fullmatch(r"(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)", version):
        raise ValueError(f"release version must be strict SemVer, got {version!r}")
    return version


def validate_docs(version: str) -> None:
    release_note = ROOT / "docs" / "releases" / f"umapers-v{version}.md"
    if not release_note.is_file():
        raise ValueError(f"missing release note: {release_note.relative_to(ROOT)}")
    if f"# umapers {version}" not in release_note.read_text(encoding="utf-8"):
        raise ValueError(f"release note heading does not name {version}")

    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    if f"## [{version}]" not in changelog:
        raise ValueError(f"CHANGELOG.md has no {version} heading")

    readme_checks = {
        "README.md": f"As of `{version}`:",
        "README_CN.md": f"截至 `{version}`：",
    }
    for relative_path, expected in readme_checks.items():
        content = (ROOT / relative_path).read_text(encoding="utf-8")
        if expected not in content:
            raise ValueError(f"{relative_path} does not declare the {version} known-gap scope")


def validate_python_metadata() -> None:
    project = load_toml("umap_rs/pyproject.toml")["project"]
    if project.get("requires-python") != ">=3.9,<3.14":
        raise ValueError("umap_rs/pyproject.toml must declare Python >=3.9,<3.14")
    if project.get("license") != {"file": "LICENSE"}:
        raise ValueError("umap_rs/pyproject.toml must package LICENSE by file")
    if (ROOT / "LICENSE").read_bytes() != (ROOT / "umap_rs" / "LICENSE").read_bytes():
        raise ValueError("root LICENSE and umap_rs/LICENSE differ")


def validate_tag(tag: str, version: str) -> None:
    normalized = tag.removeprefix("refs/tags/")
    match = TAG_RE.fullmatch(normalized)
    if match is None:
        raise ValueError(
            f"release tag must match umapers-vMAJOR.MINOR.PATCH, got {tag!r}"
        )
    tag_version = ".".join(match.groups())
    if tag_version != version:
        raise ValueError(f"tag version {tag_version} does not match package version {version}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", help="release tag, for example umapers-v1.1.4")
    parser.add_argument(
        "--print-version",
        action="store_true",
        help="print only the validated version",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        version = release_version()
        validate_docs(version)
        validate_python_metadata()
        if args.tag:
            validate_tag(args.tag, version)
    except (KeyError, OSError, TypeError, ValueError) as exc:
        print(f"release version validation failed: {exc}", file=sys.stderr)
        return 1

    if args.print_version:
        print(version)
    else:
        suffix = f" and tag {args.tag}" if args.tag else ""
        print(f"release version {version}{suffix} is consistent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
