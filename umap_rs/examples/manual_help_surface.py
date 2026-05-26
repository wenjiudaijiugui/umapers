from __future__ import annotations

import inspect
from importlib import resources

from umapers import Umap, __version__, fit_transform


def _doc_summary(obj: object) -> str:
    doc = inspect.getdoc(obj)
    if not doc:
        return "<missing>"
    return doc.splitlines()[0]


def main() -> None:
    print("umapers version:", __version__)
    print("runtime signature for fit_transform:", inspect.signature(fit_transform))
    print()

    public_objects = [
        ("Umap", Umap),
        ("Umap.__init__", Umap.__init__),
        ("Umap.fit", Umap.fit),
        ("Umap.fit_transform", Umap.fit_transform),
        ("Umap.fit_transform_with_knn", Umap.fit_transform_with_knn),
        ("Umap.transform", Umap.transform),
        ("Umap.inverse_transform", Umap.inverse_transform),
        ("fit_transform", fit_transform),
    ]

    for name, obj in public_objects:
        summary = _doc_summary(obj)
        print(f"{name}: {summary}")
        assert summary != "<missing>", f"missing docstring for {name}"

    print()

    package_root = resources.files("umapers")
    for filename in ("py.typed", "__init__.pyi", "_api.pyi"):
        resource = package_root / filename
        print(f"{filename}: {resource}")
        assert resource.is_file(), f"missing package resource: {filename}"

    print()
    print("manual_help_surface.py checks passed")


if __name__ == "__main__":
    main()
