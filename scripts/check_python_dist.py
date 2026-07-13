#!/usr/bin/env python3
"""Validate release metadata and license contents in Python distributions."""

from __future__ import annotations

import argparse
import sys
import tarfile
import zipfile
from email.parser import Parser
from pathlib import Path, PurePosixPath

from check_release_version import ROOT, release_version


EXPECTED_REQUIRES_PYTHON = ">=3.9,<3.14"
LICENSE_BYTES = (ROOT / "umap_rs" / "LICENSE").read_bytes()


def safe_member(name: str) -> bool:
    path = PurePosixPath(name)
    return not path.is_absolute() and ".." not in path.parts


def validate_metadata(raw_metadata: bytes, source: Path, version: str) -> None:
    metadata = Parser().parsestr(raw_metadata.decode("utf-8"))
    if metadata["Name"] != "umapers":
        raise ValueError(f"{source.name}: unexpected Name {metadata['Name']!r}")
    if metadata["Version"] != version:
        raise ValueError(f"{source.name}: unexpected Version {metadata['Version']!r}")
    requires_python = (metadata["Requires-Python"] or "").replace(" ", "")
    if requires_python != EXPECTED_REQUIRES_PYTHON:
        raise ValueError(
            f"{source.name}: unexpected Requires-Python "
            f"{metadata['Requires-Python']!r}"
        )


def validate_wheel(path: Path, version: str) -> None:
    if not path.name.startswith(f"umapers-{version}-"):
        raise ValueError(f"wheel filename does not contain version {version}: {path.name}")
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if any(not safe_member(name) for name in names):
            raise ValueError(f"{path.name}: unsafe archive member")
        metadata_names = [name for name in names if name.endswith(".dist-info/METADATA")]
        if len(metadata_names) != 1:
            raise ValueError(f"{path.name}: expected one METADATA file")
        validate_metadata(archive.read(metadata_names[0]), path, version)

        license_names = [name for name in names if PurePosixPath(name).name == "LICENSE"]
        if not license_names:
            raise ValueError(f"{path.name}: LICENSE is missing")
        if not any(archive.read(name) == LICENSE_BYTES for name in license_names):
            raise ValueError(f"{path.name}: packaged LICENSE does not match umap_rs/LICENSE")


def validate_sdist(path: Path, version: str) -> None:
    expected_name = f"umapers-{version}.tar.gz"
    if path.name != expected_name:
        raise ValueError(f"sdist must be named {expected_name}, got {path.name}")
    with tarfile.open(path, "r:gz") as archive:
        members = archive.getmembers()
        if any(not safe_member(member.name) for member in members):
            raise ValueError(f"{path.name}: unsafe archive member")
        metadata_members = [member for member in members if member.name.endswith("/PKG-INFO")]
        if len(metadata_members) != 1:
            raise ValueError(f"{path.name}: expected one PKG-INFO file")
        metadata_handle = archive.extractfile(metadata_members[0])
        if metadata_handle is None:
            raise ValueError(f"{path.name}: cannot read PKG-INFO")
        validate_metadata(metadata_handle.read(), path, version)

        license_members = [
            member for member in members if PurePosixPath(member.name).name == "LICENSE"
        ]
        if not license_members:
            raise ValueError(f"{path.name}: LICENSE is missing")
        matching_license = False
        for member in license_members:
            license_handle = archive.extractfile(member)
            if license_handle is not None and license_handle.read() == LICENSE_BYTES:
                matching_license = True
                break
        if not matching_license:
            raise ValueError(f"{path.name}: packaged LICENSE does not match umap_rs/LICENSE")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dist-dir", type=Path, required=True)
    parser.add_argument("--expected-count", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dist_dir = args.dist_dir.resolve()
    artifacts = sorted(path for path in dist_dir.iterdir() if path.is_file())
    wheels = [path for path in artifacts if path.suffix == ".whl"]
    sdists = [path for path in artifacts if path.name.endswith(".tar.gz")]
    try:
        if args.expected_count is not None and len(artifacts) != args.expected_count:
            raise ValueError(
                f"expected {args.expected_count} artifacts, found {len(artifacts)}"
            )
        if not wheels:
            raise ValueError("no wheel found")
        if len(sdists) != 1:
            raise ValueError(f"expected one sdist, found {len(sdists)}")
        if len(wheels) + len(sdists) != len(artifacts):
            raise ValueError("dist directory contains unexpected files")

        version = release_version()
        for wheel in wheels:
            validate_wheel(wheel, version)
        validate_sdist(sdists[0], version)
    except (OSError, tarfile.TarError, ValueError, zipfile.BadZipFile) as exc:
        print(f"Python distribution validation failed: {exc}", file=sys.stderr)
        return 1

    print(
        f"validated {len(wheels)} wheel(s) and one sdist for umapers {version}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
