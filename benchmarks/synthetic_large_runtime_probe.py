#!/usr/bin/env python3
"""Large synthetic UMAP runtime probe for auto/exact/approximate kNN choices."""

from __future__ import annotations

import argparse
import json
import math
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler

from umapers import Umap


RS_COLOR = "#2563eb"
LEARN_COLOR = "#dc2626"
EXACT_COLOR = "#16a34a"
APPROX_COLOR = "#9333ea"
GRID_COLOR = "#d4d4d8"
TEXT_COLOR = "#18181b"
MUTED_COLOR = "#71717a"

SPECS = [
    ("class_4200x96", 4200, 96, 10),
    ("class_5000x112", 5000, 112, 10),
    ("class_6000x128", 6000, 128, 10),
    ("class_8000x128", 8000, 128, 10),
    ("class_10000x160", 10000, 160, 10),
    ("class_16000x160", 16000, 160, 10),
    ("class_20000x192", 20000, 192, 10),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path, default=Path("reports/synthetic_large_runtime_probe.json"))
    parser.add_argument("--output-md", type=Path, default=Path("reports/synthetic_large_runtime_probe.md"))
    parser.add_argument("--output-svg", type=Path, default=Path("reports/synthetic_large_runtime_probe.svg"))
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--n-epochs", type=int, default=150)
    parser.add_argument("--init", choices=["spectral", "random"], default="spectral")
    parser.add_argument("--build-profile", default="release")
    parser.add_argument("--no-warmup", action="store_true")
    return parser.parse_args()


def version_of(package: str) -> str:
    try:
        from importlib.metadata import version

        return version(package)
    except Exception as exc:  # pragma: no cover
        return f"unavailable: {type(exc).__name__}: {exc}"


def timed(fn: Any) -> tuple[Any, float]:
    started = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - started


def make_dataset(n_samples: int, n_features: int, n_classes: int, seed: int) -> np.ndarray:
    informative = max(2, min(n_features, n_classes * 3))
    redundant = min(max(0, n_features - informative), n_features // 5)
    x, _ = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=informative,
        n_redundant=redundant,
        n_repeated=0,
        n_classes=n_classes,
        n_clusters_per_class=1,
        class_sep=0.92,
        random_state=seed,
    )
    x = StandardScaler().fit_transform(x.astype(np.float32)).astype(np.float32)
    return np.ascontiguousarray(x)


def make_umap_learn(seed: int, n_neighbors: int, n_epochs: int, init: str) -> Any:
    import umap

    return umap.UMAP(
        n_neighbors=n_neighbors,
        n_components=2,
        n_epochs=n_epochs,
        metric="euclidean",
        init=init,
        random_state=seed,
    )


def run_umapers(x: np.ndarray, seed: int, n_neighbors: int, n_epochs: int, init: str, ann_mode: str) -> dict[str, Any]:
    model = Umap(
        n_neighbors=min(n_neighbors, x.shape[0] - 1),
        n_components=2,
        n_epochs=n_epochs,
        metric="euclidean",
        init=init,
        random_seed=seed,
        ann_mode=ann_mode,
    )
    result, elapsed = timed(lambda: model.profile_fit_transform(x))
    timings = result["timings"]
    return {
        "time_sec": elapsed,
        "knn_sec": float(timings["knn_sec"]),
        "optimize_sec": float(timings["optimize_sec"]),
        "used_approximate_knn": bool(result["used_approximate_knn"]),
    }


def run_spec(name: str, n_samples: int, n_features: int, n_classes: int, args: argparse.Namespace) -> dict[str, Any]:
    x = make_dataset(n_samples, n_features, n_classes, args.seed)
    auto = run_umapers(x, args.seed, args.n_neighbors, args.n_epochs, args.init, "auto")
    approx = run_umapers(x, args.seed, args.n_neighbors, args.n_epochs, args.init, "approximate")
    exact = run_umapers(x, args.seed, args.n_neighbors, args.n_epochs, args.init, "exact")
    _, learn_time = timed(
        lambda: make_umap_learn(args.seed, min(args.n_neighbors, x.shape[0] - 1), args.n_epochs, args.init)
        .fit_transform(x)
        .astype(np.float32)
    )
    return {
        "name": name,
        "n_samples": n_samples,
        "n_features": n_features,
        "matrix_size": n_samples * n_features,
        "umapers_auto": auto,
        "umapers_approximate": approx,
        "umapers_exact": exact,
        "umap_learn": {"time_sec": learn_time},
        "auto_over_learn": auto["time_sec"] / learn_time if learn_time > 0 else float("nan"),
        "auto_over_exact": auto["time_sec"] / exact["time_sec"] if exact["time_sec"] > 0 else float("nan"),
    }


def warmup(args: argparse.Namespace) -> None:
    small = make_dataset(80, 8, 3, 123)
    Umap(
        n_neighbors=min(args.n_neighbors, small.shape[0] - 1),
        n_components=2,
        n_epochs=20,
        init=args.init,
        random_seed=123,
        ann_mode="auto",
    ).fit_transform(small)
    make_umap_learn(123, min(args.n_neighbors, small.shape[0] - 1), 20, args.init).fit_transform(small)

    large = make_dataset(4200, 96, 10, 321)
    Umap(
        n_neighbors=min(args.n_neighbors, large.shape[0] - 1),
        n_components=2,
        n_epochs=20,
        init=args.init,
        random_seed=321,
        ann_mode="auto",
    ).fit_transform(large)
    make_umap_learn(321, min(args.n_neighbors, large.shape[0] - 1), 20, args.init).fit_transform(large)


def scale_log(value: float, src_min: float, src_max: float, dst_min: float, dst_max: float) -> float:
    log_value = math.log10(value)
    log_min = math.log10(src_min)
    log_max = math.log10(src_max)
    if log_max == log_min:
        return (dst_min + dst_max) / 2.0
    return dst_min + (log_value - log_min) * (dst_max - dst_min) / (log_max - log_min)


def scale_linear(value: float, src_min: float, src_max: float, dst_min: float, dst_max: float) -> float:
    if src_max == src_min:
        return (dst_min + dst_max) / 2.0
    return dst_min + (value - src_min) * (dst_max - dst_min) / (src_max - src_min)


def polyline(points: list[tuple[float, float]]) -> str:
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in points)


def make_svg(rows: list[dict[str, Any]]) -> str:
    width = 1120
    height = 690
    left = 94
    right = 44
    top = 78
    bottom = 94
    plot_w = width - left - right
    plot_h = height - top - bottom
    x_min = min(row["matrix_size"] for row in rows)
    x_max = max(row["matrix_size"] for row in rows)
    y_max = max(
        max(
            row["umapers_auto"]["time_sec"],
            row["umapers_approximate"]["time_sec"],
            row["umapers_exact"]["time_sec"],
            row["umap_learn"]["time_sec"],
        )
        for row in rows
    ) * 1.12

    def x(row: dict[str, Any]) -> float:
        return scale_log(row["matrix_size"], x_min, x_max, left, left + plot_w)

    def y(value: float) -> float:
        return scale_linear(value, 0.0, y_max, top + plot_h, top)

    series = [
        ("umapers auto", RS_COLOR, [(x(row), y(row["umapers_auto"]["time_sec"])) for row in rows]),
        ("umapers exact", EXACT_COLOR, [(x(row), y(row["umapers_exact"]["time_sec"])) for row in rows]),
        ("umapers approx", APPROX_COLOR, [(x(row), y(row["umapers_approximate"]["time_sec"])) for row in rows]),
        ("umap-learn", LEARN_COLOR, [(x(row), y(row["umap_learn"]["time_sec"])) for row in rows]),
    ]
    tick_count = 5
    y_step = y_max / tick_count
    y_ticks = [idx * y_step for idx in range(tick_count + 1)]

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{left}" y="34" font-family="Inter, Arial, sans-serif" font-size="24" font-weight="700" fill="{TEXT_COLOR}">Synthetic Large Runtime Probe</text>',
        f'<text x="{left}" y="58" font-family="Inter, Arial, sans-serif" font-size="13" fill="{MUTED_COLOR}">Auto, exact, approximate, and umap-learn on larger synthetic classification matrices.</text>',
    ]
    for tick in y_ticks:
        ty = y(tick)
        lines.append(f'<line x1="{left}" y1="{ty:.2f}" x2="{left + plot_w}" y2="{ty:.2f}" stroke="{GRID_COLOR}" stroke-width="1"/>')
        lines.append(
            f'<text x="{left - 12}" y="{ty + 4:.2f}" text-anchor="end" font-family="Inter, Arial, sans-serif" font-size="12" fill="{MUTED_COLOR}">{tick:.1f}s</text>'
        )

    for row in rows:
        tx = x(row)
        lines.append(f'<line x1="{tx:.2f}" y1="{top}" x2="{tx:.2f}" y2="{top + plot_h}" stroke="#f4f4f5" stroke-width="1"/>')
        lines.append(
            f'<text x="{tx:.2f}" y="{top + plot_h + 24}" text-anchor="middle" font-family="Inter, Arial, sans-serif" font-size="11" fill="{TEXT_COLOR}">{row["n_samples"]}x{row["n_features"]}</text>'
        )
        lines.append(
            f'<text x="{tx:.2f}" y="{top + plot_h + 42}" text-anchor="middle" font-family="Inter, Arial, sans-serif" font-size="10" fill="{MUTED_COLOR}">{row["matrix_size"]:,}</text>'
        )

    axis_y = top + plot_h
    lines.extend(
        [
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{axis_y}" stroke="{TEXT_COLOR}" stroke-width="1.2"/>',
            f'<line x1="{left}" y1="{axis_y}" x2="{left + plot_w}" y2="{axis_y}" stroke="{TEXT_COLOR}" stroke-width="1.2"/>',
            f'<text x="28" y="{top + plot_h / 2:.2f}" transform="rotate(-90 28 {top + plot_h / 2:.2f})" text-anchor="middle" font-family="Inter, Arial, sans-serif" font-size="13" fill="{TEXT_COLOR}">runtime seconds</text>',
            f'<text x="{left + plot_w / 2:.2f}" y="{height - 22}" text-anchor="middle" font-family="Inter, Arial, sans-serif" font-size="13" fill="{TEXT_COLOR}">dataset matrix size, log scale</text>',
        ]
    )
    for label, color, points in series:
        lines.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.8" points="{polyline(points)}"/>')
        for px, py in points:
            lines.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="4.5" fill="{color}"/>')

    legend_x = left + plot_w - 220
    legend_y = top + 12
    lines.append(f'<rect x="{legend_x}" y="{legend_y}" width="194" height="104" rx="6" fill="#ffffff" stroke="{GRID_COLOR}"/>')
    for idx, (label, color, _) in enumerate(series):
        y0 = legend_y + 22 + idx * 22
        lines.append(f'<line x1="{legend_x + 14}" y1="{y0}" x2="{legend_x + 48}" y2="{y0}" stroke="{color}" stroke-width="3"/>')
        lines.append(f'<circle cx="{legend_x + 31}" cy="{y0}" r="4.5" fill="{color}"/>')
        lines.append(f'<text x="{legend_x + 60}" y="{y0 + 4}" font-family="Inter, Arial, sans-serif" font-size="12" fill="{TEXT_COLOR}">{label}</text>')

    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def write_markdown(report: dict[str, Any], path: Path, svg_path: Path) -> None:
    lines = [
        "# Synthetic Large Runtime Probe",
        "",
        f"- generated_at: `{report['generated_at']}`",
        f"- python: `{report['environment']['python']}`",
        f"- umapers: `{report['environment']['umapers']}`",
        f"- umap_learn: `{report['environment']['umap_learn']}`",
        f"- build_profile: `{report['settings']['build_profile']}`",
        f"- seed: `{report['settings']['seed']}`",
        f"- UMAP: `n_neighbors={report['settings']['n_neighbors']}`, `n_epochs={report['settings']['n_epochs']}`, `init={report['settings']['init']}`, `metric=euclidean`",
        f"- warmup: `{report['settings']['warmup_note']}`",
        "",
        f"![Synthetic large runtime probe]({svg_path.name})",
        "",
        "| dataset | samples | features | matrix size | rs auto sec | auto kNN | auto opt | auto ANN | rs approx sec | rs exact sec | learn sec | auto / learn | auto / exact |",
        "|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in report["datasets"]:
        auto = row["umapers_auto"]
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['name']}`",
                    str(row["n_samples"]),
                    str(row["n_features"]),
                    f"{row['matrix_size']:,}",
                    f"{auto['time_sec']:.3f}",
                    f"{auto['knn_sec']:.3f}",
                    f"{auto['optimize_sec']:.3f}",
                    f"`{'yes' if auto['used_approximate_knn'] else 'no'}`",
                    f"{row['umapers_approximate']['time_sec']:.3f}",
                    f"{row['umapers_exact']['time_sec']:.3f}",
                    f"{row['umap_learn']['time_sec']:.3f}",
                    f"{row['auto_over_learn']:.2f}x",
                    f"{row['auto_over_exact']:.2f}x",
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not args.no_warmup:
        warmup(args)
    rows = []
    for idx, spec in enumerate(SPECS, start=1):
        name, n_samples, n_features, n_classes = spec
        print(f"running {idx:02d}/{len(SPECS)} {name} ({n_samples} x {n_features})", flush=True)
        rows.append(run_spec(name, n_samples, n_features, n_classes, args))
    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "environment": {
            "python": platform.python_version(),
            "umapers": version_of("umapers"),
            "umap_learn": version_of("umap-learn"),
        },
        "settings": {
            "seed": args.seed,
            "n_neighbors": args.n_neighbors,
            "n_epochs": args.n_epochs,
            "init": args.init,
            "build_profile": args.build_profile,
            "warmup_note": "small path plus discarded >4096-sample large path" if not args.no_warmup else "disabled",
        },
        "datasets": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    args.output_svg.write_text(make_svg(rows), encoding="utf-8")
    write_markdown(report, args.output_md, args.output_svg)
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_svg}")
    print(f"wrote {args.output_md}")


if __name__ == "__main__":
    main()
