#!/usr/bin/env python3
"""Small-dataset hotspot report for umap-rs module timings."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_wine
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler

from umapers import Umap


TIMING_KEYS = [
    "validation_sec",
    "knn_sec",
    "curve_params_sec",
    "knn_validate_trim_sec",
    "smooth_knn_sec",
    "density_original_sec",
    "membership_sec",
    "symmetrize_sec",
    "target_intersection_sec",
    "prune_sec",
    "init_sec",
    "optimize_sec",
    "density_embedding_sec",
    "output_copy_sec",
    "store_state_sec",
    "total_sec",
    "kmeans_sec",
    "pipeline_total_sec",
]


MODULE_GROUPS = {
    "validation": ["validation_sec"],
    "knn": ["knn_sec"],
    "curve_params": ["curve_params_sec"],
    "graph_prepare": [
        "knn_validate_trim_sec",
        "smooth_knn_sec",
        "membership_sec",
        "symmetrize_sec",
        "prune_sec",
    ],
    "spectral_init": ["init_sec"],
    "layout_optimize": ["optimize_sec"],
    "output_store": ["output_copy_sec", "store_state_sec"],
    "kmeans": ["kmeans_sec"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path, default=Path("reports/small_dataset_hotspot_report.json"))
    parser.add_argument("--output-md", type=Path, default=Path("reports/small_dataset_hotspot_report.md"))
    parser.add_argument("--output-svg", type=Path, default=Path("reports/small_dataset_hotspot_breakdown.svg"))
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=80)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--n-epochs", type=int, default=200)
    parser.add_argument("--init", choices=["spectral", "random"], default="spectral")
    return parser.parse_args()


def version_of(package: str) -> str:
    try:
        from importlib.metadata import version

        return version(package)
    except Exception as exc:  # pragma: no cover
        return f"unavailable: {type(exc).__name__}: {exc}"


def load_dataset() -> tuple[np.ndarray, np.ndarray]:
    data = load_wine()
    x = StandardScaler().fit_transform(data.data.astype(np.float32)).astype(np.float32)
    y = np.asarray(data.target)
    return np.ascontiguousarray(x), y


def read_proc_status() -> dict[str, float | None]:
    out: dict[str, float | None] = {
        "vmrss_mb": None,
        "vmhwm_mb": None,
    }
    try:
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                out["vmrss_mb"] = float(line.split()[1]) / 1024.0
            elif line.startswith("VmHWM:"):
                out["vmhwm_mb"] = float(line.split()[1]) / 1024.0
    except OSError:
        pass
    return out


def score_embedding(embedding: np.ndarray, y: np.ndarray, seed: int) -> tuple[dict[str, float], float]:
    started = time.perf_counter()
    labels = KMeans(n_clusters=int(np.unique(y).size), n_init=20, random_state=seed).fit_predict(embedding)
    elapsed = time.perf_counter() - started
    return {
        "adjusted_rand": float(adjusted_rand_score(y, labels)),
        "normalized_mutual_info": float(normalized_mutual_info_score(y, labels)),
    }, elapsed


def run_profile_once(
    x: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
    n_neighbors: int,
    n_epochs: int,
    init: str,
) -> dict[str, Any]:
    model = Umap(
        n_neighbors=n_neighbors,
        n_components=2,
        n_epochs=n_epochs,
        metric="euclidean",
        init=init,
        random_seed=seed,
        ann_mode="auto",
    )
    started = time.perf_counter()
    result = model.profile_fit_transform(x)
    scores, kmeans_sec = score_embedding(result["embedding"], y, seed)
    pipeline_total_sec = time.perf_counter() - started
    timings = dict(result["timings"])
    timings["kmeans_sec"] = kmeans_sec
    timings["pipeline_total_sec"] = pipeline_total_sec
    return {
        "timings": timings,
        "scores": scores,
        "n_edges": int(result["n_edges"]),
        "used_approximate_knn": bool(result["used_approximate_knn"]),
    }


def mean(values: list[float]) -> float:
    return float(statistics.fmean(values))


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    sorted_values = sorted(values)
    pos = (len(sorted_values) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(sorted_values[lo])
    frac = pos - lo
    return float(sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac)


def summarize(values: list[float]) -> dict[str, float]:
    n = len(values)
    avg = mean(values)
    std = float(statistics.pstdev(values)) if n > 1 else 0.0
    sem = std / math.sqrt(n) if n > 0 else float("nan")
    ci95 = 1.96 * sem
    return {
        "n": float(n),
        "mean": avg,
        "std": std,
        "median": float(statistics.median(values)),
        "p05": percentile(values, 0.05),
        "p95": percentile(values, 0.95),
        "ci95_half_width": ci95,
        "cv": std / avg if avg > 0 else float("nan"),
    }


def semantic_check(x: np.ndarray, seed: int, n_neighbors: int, n_epochs: int, init: str) -> dict[str, Any]:
    a = Umap(
        n_neighbors=n_neighbors,
        n_components=2,
        n_epochs=n_epochs,
        metric="euclidean",
        init=init,
        random_seed=seed,
        ann_mode="auto",
    ).fit_transform(x)
    b = Umap(
        n_neighbors=n_neighbors,
        n_components=2,
        n_epochs=n_epochs,
        metric="euclidean",
        init=init,
        random_seed=seed,
        ann_mode="auto",
    ).profile_fit_transform(x)["embedding"]
    return {
        "shape": list(b.shape),
        "finite": bool(np.all(np.isfinite(b))),
        "max_abs_diff_vs_fit_transform": float(np.max(np.abs(a - b))),
    }


def make_group_summary(stage_summary: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    groups: dict[str, dict[str, float]] = {}
    pipeline_mean = stage_summary["pipeline_total_sec"]["mean"]
    for group, keys in MODULE_GROUPS.items():
        group_mean = sum(stage_summary[key]["mean"] for key in keys)
        ci = math.sqrt(sum(stage_summary[key]["ci95_half_width"] ** 2 for key in keys))
        groups[group] = {
            "mean": group_mean,
            "ci95_half_width": ci,
            "share_of_pipeline": group_mean / pipeline_mean if pipeline_mean > 0 else float("nan"),
        }
    return groups


def make_svg(group_summary: dict[str, dict[str, float]], output: Path) -> None:
    width = 1060
    height = 520
    left = 210
    right = 52
    top = 52
    row_h = 44
    max_mean = max(row["mean"] for row in group_summary.values())
    colors = {
        "validation": "#64748b",
        "knn": "#0ea5e9",
        "curve_params": "#a855f7",
        "graph_prepare": "#14b8a6",
        "spectral_init": "#dc2626",
        "layout_optimize": "#f97316",
        "output_store": "#84cc16",
        "kmeans": "#6366f1",
    }
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="36" y="30" font-family="Inter, Arial, sans-serif" font-size="22" font-weight="700" fill="#18181b">Small Dataset Hotspot Breakdown</text>',
    ]
    for idx, (group, row) in enumerate(group_summary.items()):
        y = top + idx * row_h
        bar_w = 0 if max_mean <= 0 else row["mean"] / max_mean * (width - left - right)
        lines.append(
            f'<text x="{left - 14}" y="{y + 22}" text-anchor="end" font-family="Inter, Arial, sans-serif" font-size="13" fill="#18181b">{group}</text>'
        )
        lines.append(
            f'<rect x="{left}" y="{y + 6}" width="{bar_w:.2f}" height="22" rx="3" fill="{colors.get(group, "#71717a")}"/>'
        )
        lines.append(
            f'<text x="{left + bar_w + 8:.2f}" y="{y + 22}" font-family="Inter, Arial, sans-serif" font-size="12" fill="#18181b">{row["mean"] * 1000:.2f} ms ({row["share_of_pipeline"] * 100:.1f}%)</text>'
        )
    lines.append("</svg>")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_markdown(report: dict[str, Any], path: Path, svg_path: Path) -> None:
    stage_summary = report["stage_summary"]
    group_summary = report["group_summary"]
    lines = [
        "# Small Dataset Hotspot Report",
        "",
        f"- generated_at: `{report['generated_at']}`",
        f"- dataset: `{report['dataset']['name']}` `{report['dataset']['n_samples']} x {report['dataset']['n_features']}`",
        f"- repeats: `{report['settings']['repeats']}`, warmup: `{report['settings']['warmup']}`",
        f"- UMAP: `n_neighbors={report['settings']['n_neighbors']}`, `n_epochs={report['settings']['n_epochs']}`, `init={report['settings']['init']}`, `ann_mode=auto`",
        f"- semantic max_abs_diff_vs_fit_transform: `{report['semantic_check']['max_abs_diff_vs_fit_transform']:.6g}`",
        f"- memory VmRSS before/after: `{report['memory']['before']['vmrss_mb']}` / `{report['memory']['after']['vmrss_mb']}` MB",
        f"- memory VmHWM before/after: `{report['memory']['before']['vmhwm_mb']}` / `{report['memory']['after']['vmhwm_mb']}` MB",
        "",
        f"![Small dataset hotspot breakdown]({svg_path.name})",
        "",
        "## Grouped Hotspots",
        "",
        "| group | mean ms | 95% CI +/- ms | share of pipeline |",
        "|---|---:|---:|---:|",
    ]
    for group, row in sorted(
        group_summary.items(), key=lambda item: item[1]["share_of_pipeline"], reverse=True
    ):
        lines.append(
            f"| `{group}` | {row['mean'] * 1000:.3f} | {row['ci95_half_width'] * 1000:.3f} | {row['share_of_pipeline'] * 100:.1f}% |"
        )

    lines.extend(
        [
            "",
            "## Stage Timings",
            "",
            "| stage | mean ms | std ms | 95% CI +/- ms | median ms | p05 ms | p95 ms | CV | share of pipeline |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    pipeline_mean = stage_summary["pipeline_total_sec"]["mean"]
    for stage in sorted(TIMING_KEYS, key=lambda key: stage_summary[key]["mean"], reverse=True):
        row = stage_summary[stage]
        share = row["mean"] / pipeline_mean if pipeline_mean > 0 else float("nan")
        lines.append(
            f"| `{stage}` | {row['mean'] * 1000:.3f} | {row['std'] * 1000:.3f} | "
            f"{row['ci95_half_width'] * 1000:.3f} | {row['median'] * 1000:.3f} | "
            f"{row['p05'] * 1000:.3f} | {row['p95'] * 1000:.3f} | {row['cv']:.3f} | {share * 100:.1f}% |"
        )

    lines.extend(
        [
            "",
            "## Quality",
            "",
            "| metric | mean | std |",
            "|---|---:|---:|",
        ]
    )
    for metric, row in report["score_summary"].items():
        lines.append(f"| `{metric}` | {row['mean']:.6f} | {row['std']:.6f} |")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    warnings.filterwarnings("ignore", message="n_jobs value .* overridden .*")
    x, y = load_dataset()

    for _ in range(args.warmup):
        run_profile_once(
            x,
            y,
            seed=args.seed,
            n_neighbors=args.n_neighbors,
            n_epochs=args.n_epochs,
            init=args.init,
        )

    before_mem = read_proc_status()
    runs = [
        run_profile_once(
            x,
            y,
            seed=args.seed,
            n_neighbors=args.n_neighbors,
            n_epochs=args.n_epochs,
            init=args.init,
        )
        for _ in range(args.repeats)
    ]
    after_mem = read_proc_status()

    stage_summary = {
        key: summarize([float(run["timings"][key]) for run in runs]) for key in TIMING_KEYS
    }
    score_summary = {
        "adjusted_rand": summarize([float(run["scores"]["adjusted_rand"]) for run in runs]),
        "normalized_mutual_info": summarize(
            [float(run["scores"]["normalized_mutual_info"]) for run in runs]
        ),
    }
    group_summary = make_group_summary(stage_summary)
    semantic = semantic_check(x, args.seed, args.n_neighbors, args.n_epochs, args.init)

    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "environment": {
            "python": platform.python_version(),
            "umapers": version_of("umapers"),
            "umap_learn": version_of("umap-learn"),
            "pid": os.getpid(),
        },
        "dataset": {
            "name": "sklearn.load_wine",
            "n_samples": int(x.shape[0]),
            "n_features": int(x.shape[1]),
            "n_classes": int(np.unique(y).size),
        },
        "settings": {
            "warmup": args.warmup,
            "repeats": args.repeats,
            "seed": args.seed,
            "n_neighbors": args.n_neighbors,
            "n_epochs": args.n_epochs,
            "init": args.init,
        },
        "semantic_check": semantic,
        "memory": {
            "before": before_mem,
            "after": after_mem,
        },
        "stage_summary": stage_summary,
        "group_summary": group_summary,
        "score_summary": score_summary,
        "runs": runs,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    make_svg(group_summary, args.output_svg)
    write_markdown(report, args.output_md, args.output_svg)
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_svg}")
    print(f"wrote {args.output_md}")


if __name__ == "__main__":
    main()
