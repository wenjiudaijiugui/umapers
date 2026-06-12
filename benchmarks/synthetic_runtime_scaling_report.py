#!/usr/bin/env python3
"""Synthetic runtime scaling report for umap-rs and umap-learn."""

from __future__ import annotations

import argparse
import json
import math
import platform
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.datasets import make_classification
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler

from umapers import Umap


RS_COLOR = "#2563eb"
LEARN_COLOR = "#dc2626"
GRID_COLOR = "#d4d4d8"
TEXT_COLOR = "#18181b"
MUTED_COLOR = "#71717a"


@dataclass(frozen=True)
class SyntheticSpec:
    name: str
    kind: str
    n_samples: int
    n_features: int
    n_classes: int
    cluster_std: float = 1.0
    class_sep: float = 1.5


SPECS: list[SyntheticSpec] = [
    SyntheticSpec("syn_001_blobs_120x4", "blobs", 120, 4, 3, cluster_std=0.85),
    SyntheticSpec("syn_002_class_150x6", "classification", 150, 6, 3, class_sep=1.4),
    SyntheticSpec("syn_003_blobs_180x8", "blobs", 180, 8, 4, cluster_std=0.95),
    SyntheticSpec("syn_004_class_220x10", "classification", 220, 10, 4, class_sep=1.3),
    SyntheticSpec("syn_005_blobs_260x12", "blobs", 260, 12, 5, cluster_std=1.05),
    SyntheticSpec("syn_006_class_320x16", "classification", 320, 16, 5, class_sep=1.25),
    SyntheticSpec("syn_007_blobs_400x8", "blobs", 400, 8, 4, cluster_std=1.00),
    SyntheticSpec("syn_008_class_480x20", "classification", 480, 20, 5, class_sep=1.2),
    SyntheticSpec("syn_009_blobs_560x12", "blobs", 560, 12, 6, cluster_std=1.10),
    SyntheticSpec("syn_010_class_650x24", "classification", 650, 24, 6, class_sep=1.15),
    SyntheticSpec("syn_011_blobs_750x16", "blobs", 750, 16, 6, cluster_std=1.15),
    SyntheticSpec("syn_012_class_850x32", "classification", 850, 32, 6, class_sep=1.12),
    SyntheticSpec("syn_013_blobs_950x20", "blobs", 950, 20, 7, cluster_std=1.20),
    SyntheticSpec("syn_014_class_1100x32", "classification", 1100, 32, 7, class_sep=1.10),
    SyntheticSpec("syn_015_blobs_1250x24", "blobs", 1250, 24, 7, cluster_std=1.20),
    SyntheticSpec("syn_016_class_1400x40", "classification", 1400, 40, 7, class_sep=1.08),
    SyntheticSpec("syn_017_blobs_1600x32", "blobs", 1600, 32, 8, cluster_std=1.25),
    SyntheticSpec("syn_018_class_1800x48", "classification", 1800, 48, 8, class_sep=1.05),
    SyntheticSpec("syn_019_blobs_2000x40", "blobs", 2000, 40, 8, cluster_std=1.25),
    SyntheticSpec("syn_020_class_2200x56", "classification", 2200, 56, 8, class_sep=1.02),
    SyntheticSpec("syn_021_blobs_2400x48", "blobs", 2400, 48, 8, cluster_std=1.30),
    SyntheticSpec("syn_022_class_2600x64", "classification", 2600, 64, 8, class_sep=1.00),
    SyntheticSpec("syn_023_blobs_2800x56", "blobs", 2800, 56, 9, cluster_std=1.30),
    SyntheticSpec("syn_024_class_3000x72", "classification", 3000, 72, 9, class_sep=0.98),
    SyntheticSpec("syn_025_blobs_3200x64", "blobs", 3200, 64, 9, cluster_std=1.35),
    SyntheticSpec("syn_026_class_3400x80", "classification", 3400, 80, 9, class_sep=0.96),
    SyntheticSpec("syn_027_blobs_3600x72", "blobs", 3600, 72, 10, cluster_std=1.35),
    SyntheticSpec("syn_028_class_3800x88", "classification", 3800, 88, 10, class_sep=0.94),
    SyntheticSpec("syn_029_blobs_4000x80", "blobs", 4000, 80, 10, cluster_std=1.40),
    SyntheticSpec("syn_030_class_4200x96", "classification", 4200, 96, 10, class_sep=0.92),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path, default=Path("reports/synthetic_runtime_scaling_report.json"))
    parser.add_argument("--output-md", type=Path, default=Path("reports/synthetic_runtime_scaling_report.md"))
    parser.add_argument("--output-svg", type=Path, default=Path("reports/synthetic_runtime_scaling_curve.svg"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[13])
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--n-epochs", type=int, default=150)
    parser.add_argument("--init", choices=["spectral", "random"], default="spectral")
    parser.add_argument("--build-profile", default="unspecified")
    parser.add_argument("--no-warmup", action="store_true")
    return parser.parse_args()


def version_of(package: str) -> str:
    try:
        from importlib.metadata import version

        return version(package)
    except Exception as exc:  # pragma: no cover - report best effort
        return f"unavailable: {type(exc).__name__}: {exc}"


def make_dataset(spec: SyntheticSpec, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if spec.kind == "blobs":
        x, y = make_blobs(
            n_samples=spec.n_samples,
            n_features=spec.n_features,
            centers=spec.n_classes,
            cluster_std=spec.cluster_std,
            random_state=seed,
        )
    elif spec.kind == "classification":
        n_informative = min(max(spec.n_classes * 2, 4), spec.n_features)
        n_redundant = min(max(spec.n_features // 5, 0), max(spec.n_features - n_informative, 0))
        x, y = make_classification(
            n_samples=spec.n_samples,
            n_features=spec.n_features,
            n_informative=n_informative,
            n_redundant=n_redundant,
            n_repeated=0,
            n_classes=spec.n_classes,
            n_clusters_per_class=1,
            class_sep=spec.class_sep,
            flip_y=0.01,
            random_state=seed,
        )
    else:
        raise ValueError(f"unknown synthetic dataset kind: {spec.kind}")

    x = StandardScaler().fit_transform(np.asarray(x, dtype=np.float32)).astype(np.float32)
    return x, np.asarray(y)


def timed(fn: Any) -> tuple[Any, float]:
    started = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - started


def cluster_scores(embedding: np.ndarray, y_true: np.ndarray, n_classes: int, seed: int) -> dict[str, float]:
    labels = KMeans(n_clusters=n_classes, n_init=10, random_state=seed).fit_predict(embedding)
    return {
        "adjusted_rand": float(adjusted_rand_score(y_true, labels)),
        "normalized_mutual_info": float(normalized_mutual_info_score(y_true, labels)),
    }


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


def warmup(n_neighbors: int, init: str) -> None:
    x, _ = make_dataset(SyntheticSpec("warmup", "blobs", 80, 8, 3), 123)
    Umap(
        n_neighbors=min(n_neighbors, x.shape[0] - 1),
        n_components=2,
        n_epochs=20,
        init=init,
        random_seed=123,
        ann_mode="auto",
    ).fit_transform(x)
    make_umap_learn(123, min(n_neighbors, x.shape[0] - 1), 20, init).fit_transform(x)

    # Warm the large-data path as well. umap-learn switches from exact pairwise
    # distances to PyNNDescent at 4096 samples, and the first call includes
    # Numba/PyNNDescent compilation work that should not be charged to the
    # largest benchmark point.
    x, _ = make_dataset(SyntheticSpec("large_warmup", "classification", 4200, 96, 10), 321)
    Umap(
        n_neighbors=min(n_neighbors, x.shape[0] - 1),
        n_components=2,
        n_epochs=20,
        init=init,
        random_seed=321,
        ann_mode="auto",
    ).fit_transform(x)
    make_umap_learn(321, min(n_neighbors, x.shape[0] - 1), 20, init).fit_transform(x)


def run_one(
    spec: SyntheticSpec,
    seed: int,
    n_neighbors: int,
    n_epochs: int,
    init: str,
) -> dict[str, Any]:
    x, y = make_dataset(spec, seed)
    k = min(n_neighbors, x.shape[0] - 1)

    def run_rs() -> dict[str, Any]:
        emb = Umap(
            n_neighbors=k,
            n_components=2,
            n_epochs=n_epochs,
            metric="euclidean",
            init=init,
            random_seed=seed,
            ann_mode="auto",
        ).fit_transform(x)
        return cluster_scores(emb, y, spec.n_classes, seed)

    rs_scores, rs_time = timed(run_rs)

    def run_learn() -> dict[str, Any]:
        emb = make_umap_learn(seed, k, n_epochs, init).fit_transform(x).astype(np.float32)
        return cluster_scores(emb, y, spec.n_classes, seed)

    learn_scores, learn_time = timed(run_learn)

    return {
        "seed": seed,
        "umap_rs": {"time_sec": rs_time, "scores": rs_scores},
        "umap_learn": {"time_sec": learn_time, "scores": learn_scores},
    }


def aggregate(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=0))}


def aggregate_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for method in ("umap_rs", "umap_learn"):
        out[method] = {
            "time_sec": aggregate([run[method]["time_sec"] for run in runs]),
            "scores": {
                "adjusted_rand": aggregate([run[method]["scores"]["adjusted_rand"] for run in runs]),
                "normalized_mutual_info": aggregate(
                    [run[method]["scores"]["normalized_mutual_info"] for run in runs]
                ),
            },
        }
    return out


def run_spec(
    spec: SyntheticSpec,
    seeds: list[int],
    n_neighbors: int,
    n_epochs: int,
    init: str,
) -> dict[str, Any]:
    runs = [run_one(spec, seed, n_neighbors, n_epochs, init) for seed in seeds]
    aggregate_row = aggregate_runs(runs)
    rs_time = aggregate_row["umap_rs"]["time_sec"]["mean"]
    learn_time = aggregate_row["umap_learn"]["time_sec"]["mean"]
    return {
        "name": spec.name,
        "kind": spec.kind,
        "n_samples": spec.n_samples,
        "n_features": spec.n_features,
        "n_classes": spec.n_classes,
        "matrix_size": spec.n_samples * spec.n_features,
        "runs": runs,
        "aggregate": aggregate_row,
        "rs_over_learn": rs_time / learn_time if learn_time > 0 else float("nan"),
    }


def fmt_mean_std(value: dict[str, float]) -> str:
    return f"{value['mean']:.3f} +/- {value['std']:.3f}"


def log_scale(value: float, src_min: float, src_max: float, dst_min: float, dst_max: float) -> float:
    lo = math.log10(src_min)
    hi = math.log10(src_max)
    val = math.log10(value)
    if hi == lo:
        return (dst_min + dst_max) / 2.0
    return dst_min + (val - lo) * (dst_max - dst_min) / (hi - lo)


def polyline(points: list[tuple[float, float]]) -> str:
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in points)


def make_svg(rows: list[dict[str, Any]]) -> str:
    width = 1180
    height = 720
    left = 98
    right = 42
    top = 86
    bottom = 104
    plot_w = width - left - right
    plot_h = height - top - bottom

    x_min = min(row["matrix_size"] for row in rows)
    x_max = max(row["matrix_size"] for row in rows)
    all_times = [
        row["aggregate"][method]["time_sec"]["mean"]
        for row in rows
        for method in ("umap_rs", "umap_learn")
    ]
    y_min = max(min(all_times) / 1.35, 1e-3)
    y_max = max(all_times) * 1.45

    def x(row: dict[str, Any]) -> float:
        return log_scale(row["matrix_size"], x_min, x_max, left, left + plot_w)

    def y(value: float) -> float:
        return log_scale(value, y_min, y_max, top + plot_h, top)

    rs_points = [(x(row), y(row["aggregate"]["umap_rs"]["time_sec"]["mean"])) for row in rows]
    learn_points = [(x(row), y(row["aggregate"]["umap_learn"]["time_sec"]["mean"])) for row in rows]

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{left}" y="34" font-family="Inter, Arial, sans-serif" font-size="24" font-weight="700" fill="{TEXT_COLOR}">Synthetic UMAP Runtime Scaling</text>',
        f'<text x="{left}" y="58" font-family="Inter, Arial, sans-serif" font-size="13" fill="{MUTED_COLOR}">30 synthetic datasets; x-axis is samples x features; y-axis is runtime in seconds; both axes use log scale.</text>',
    ]

    y_ticks = [0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50]
    for tick in y_ticks:
        if tick < y_min or tick > y_max:
            continue
        ty = y(tick)
        lines.append(
            f'<line x1="{left}" y1="{ty:.2f}" x2="{left + plot_w}" y2="{ty:.2f}" stroke="{GRID_COLOR}" stroke-width="1"/>'
        )
        lines.append(
            f'<text x="{left - 12}" y="{ty + 4:.2f}" text-anchor="end" font-family="Inter, Arial, sans-serif" font-size="12" fill="{MUTED_COLOR}">{tick:g}s</text>'
        )

    label_indices = {0, 5, 10, 15, 20, 25, len(rows) - 1}
    for idx, row in enumerate(rows):
        tx = x(row)
        lines.append(
            f'<line x1="{tx:.2f}" y1="{top}" x2="{tx:.2f}" y2="{top + plot_h}" stroke="#f4f4f5" stroke-width="1"/>'
        )
        if idx in label_indices:
            lines.append(
                f'<text x="{tx:.2f}" y="{top + plot_h + 26}" text-anchor="middle" font-family="Inter, Arial, sans-serif" font-size="11" fill="{TEXT_COLOR}">{row["n_samples"]}x{row["n_features"]}</text>'
            )
            lines.append(
                f'<text x="{tx:.2f}" y="{top + plot_h + 43}" text-anchor="middle" font-family="Inter, Arial, sans-serif" font-size="10" fill="{MUTED_COLOR}">{row["matrix_size"]:,}</text>'
            )

    axis_y = top + plot_h
    lines.extend(
        [
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{axis_y}" stroke="{TEXT_COLOR}" stroke-width="1.2"/>',
            f'<line x1="{left}" y1="{axis_y}" x2="{left + plot_w}" y2="{axis_y}" stroke="{TEXT_COLOR}" stroke-width="1.2"/>',
            f'<text x="30" y="{top + plot_h / 2:.2f}" transform="rotate(-90 30 {top + plot_h / 2:.2f})" text-anchor="middle" font-family="Inter, Arial, sans-serif" font-size="13" fill="{TEXT_COLOR}">runtime seconds, log scale</text>',
            f'<text x="{left + plot_w / 2:.2f}" y="{height - 26}" text-anchor="middle" font-family="Inter, Arial, sans-serif" font-size="13" fill="{TEXT_COLOR}">dataset matrix size, log scale</text>',
            f'<polyline fill="none" stroke="{RS_COLOR}" stroke-width="2.8" points="{polyline(rs_points)}"/>',
            f'<polyline fill="none" stroke="{LEARN_COLOR}" stroke-width="2.8" points="{polyline(learn_points)}"/>',
        ]
    )

    for (rx, ry), (lx, ly) in zip(rs_points, learn_points, strict=True):
        lines.append(f'<circle cx="{rx:.2f}" cy="{ry:.2f}" r="3.8" fill="{RS_COLOR}"/>')
        lines.append(f'<circle cx="{lx:.2f}" cy="{ly:.2f}" r="3.8" fill="{LEARN_COLOR}"/>')

    legend_x = left + plot_w - 220
    legend_y = top + 12
    lines.extend(
        [
            f'<rect x="{legend_x}" y="{legend_y}" width="190" height="64" rx="6" fill="#ffffff" stroke="{GRID_COLOR}"/>',
            f'<line x1="{legend_x + 16}" y1="{legend_y + 22}" x2="{legend_x + 50}" y2="{legend_y + 22}" stroke="{RS_COLOR}" stroke-width="3"/>',
            f'<circle cx="{legend_x + 33}" cy="{legend_y + 22}" r="5" fill="{RS_COLOR}"/>',
            f'<text x="{legend_x + 62}" y="{legend_y + 26}" font-family="Inter, Arial, sans-serif" font-size="13" fill="{TEXT_COLOR}">umap-rs</text>',
            f'<line x1="{legend_x + 16}" y1="{legend_y + 46}" x2="{legend_x + 50}" y2="{legend_y + 46}" stroke="{LEARN_COLOR}" stroke-width="3"/>',
            f'<circle cx="{legend_x + 33}" cy="{legend_y + 46}" r="5" fill="{LEARN_COLOR}"/>',
            f'<text x="{legend_x + 62}" y="{legend_y + 50}" font-family="Inter, Arial, sans-serif" font-size="13" fill="{TEXT_COLOR}">umap-learn</text>',
        ]
    )

    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def write_markdown(report: dict[str, Any], path: Path, svg_path: Path) -> None:
    rows = report["datasets"]
    faster_count = sum(row["rs_over_learn"] < 1.0 for row in rows)
    first_faster = next((row for row in rows if row["rs_over_learn"] < 1.0), None)
    lines = [
        "# Synthetic Runtime Scaling Report",
        "",
        f"- generated_at: `{report['generated_at']}`",
        f"- python: `{report['environment']['python']}`",
        f"- umapers: `{report['environment']['umapers']}`",
        f"- umap_learn: `{report['environment']['umap_learn']}`",
        f"- build_profile: `{report['settings']['build_profile']}`",
        f"- datasets: `{len(rows)}`",
        f"- seeds: `{report['settings']['seeds']}`",
        f"- UMAP: `n_neighbors={report['settings']['n_neighbors']}`, `n_epochs={report['settings']['n_epochs']}`, `init={report['settings']['init']}`, `metric=euclidean`",
        f"- warmup: `{report['settings']['warmup']}`",
        f"- warmup_note: `{report['settings']['warmup_note']}`",
        "",
        f"![Synthetic runtime scaling curve]({svg_path.name})",
        "",
        "## Summary",
        "",
        f"- umap-rs is faster on {faster_count} / {len(rows)} synthetic datasets.",
    ]
    if first_faster is not None:
        lines.append(
            f"- First faster point: `{first_faster['name']}` at matrix size `{first_faster['matrix_size']:,}` "
            f"({first_faster['n_samples']} x {first_faster['n_features']}), ratio `{first_faster['rs_over_learn']:.2f}x`."
        )
    lines.extend(
        [
            "",
            "## Runtime Table",
            "",
            "| dataset | kind | samples | features | matrix size | classes | umap-rs sec | umap-learn sec | rs / learn | rs ARI | learn ARI |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        rs = row["aggregate"]["umap_rs"]
        learn = row["aggregate"]["umap_learn"]
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['name']}`",
                    f"`{row['kind']}`",
                    str(row["n_samples"]),
                    str(row["n_features"]),
                    f"{row['matrix_size']:,}",
                    str(row["n_classes"]),
                    fmt_mean_std(rs["time_sec"]),
                    fmt_mean_std(learn["time_sec"]),
                    f"{row['rs_over_learn']:.2f}x",
                    fmt_mean_std(rs["scores"]["adjusted_rand"]),
                    fmt_mean_std(learn["scores"]["adjusted_rand"]),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    warnings.filterwarnings("ignore", message="n_jobs value .* overridden .*")
    if not args.no_warmup:
        warmup(args.n_neighbors, args.init)

    rows = []
    for idx, spec in enumerate(SPECS, start=1):
        print(
            f"running {idx:02d}/{len(SPECS)} {spec.name} "
            f"({spec.n_samples} x {spec.n_features})",
            flush=True,
        )
        rows.append(run_spec(spec, args.seeds, args.n_neighbors, args.n_epochs, args.init))

    rows.sort(key=lambda row: row["matrix_size"])
    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "environment": {
            "python": platform.python_version(),
            "umapers": version_of("umapers"),
            "umap_learn": version_of("umap-learn"),
        },
        "settings": {
            "seeds": args.seeds,
            "n_neighbors": args.n_neighbors,
            "n_epochs": args.n_epochs,
            "init": args.init,
            "build_profile": args.build_profile,
            "warmup": not args.no_warmup,
            "warmup_note": (
                "small path plus discarded >4096-sample large path"
                if not args.no_warmup
                else "disabled"
            ),
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
