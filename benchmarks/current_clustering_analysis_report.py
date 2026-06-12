#!/usr/bin/env python3
"""Real-data dimensionality-reduction clustering report for umapers."""

from __future__ import annotations

import argparse
import html
import json
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_digits
from sklearn.datasets import load_iris
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import completeness_score
from sklearn.metrics import homogeneity_score
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import silhouette_score
from sklearn.metrics import v_measure_score
from sklearn.preprocessing import StandardScaler

from umapers import Umap


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    loader: Any


DATASETS = [
    DatasetSpec("digits", load_digits),
    DatasetSpec("wine", load_wine),
    DatasetSpec("breast_cancer", load_breast_cancer),
    DatasetSpec("iris", load_iris),
]

METHOD_ORDER = ["raw_kmeans", "pca2_kmeans", "umap_rs2_kmeans", "umap_learn2_kmeans"]
METHOD_LABELS = {
    "raw_kmeans": "Raw",
    "pca2_kmeans": "PCA2",
    "umap_rs2_kmeans": "umapers",
    "umap_learn2_kmeans": "umap-learn",
}
METHOD_COLORS = {
    "raw_kmeans": "#6b7280",
    "pca2_kmeans": "#16a34a",
    "umap_rs2_kmeans": "#2563eb",
    "umap_learn2_kmeans": "#dc2626",
}
METRIC_ORDER = [
    ("adjusted_rand", "ARI"),
    ("normalized_mutual_info", "NMI"),
    ("adjusted_mutual_info", "AMI"),
    ("homogeneity", "Homogeneity"),
    ("completeness", "Completeness"),
    ("v_measure", "V-measure"),
    ("silhouette", "Silhouette"),
]
TEXT_COLOR = "#111827"
MUTED_COLOR = "#4b5563"
GRID_COLOR = "#e5e7eb"
AXIS_COLOR = "#9ca3af"


def timed(fn: Any) -> tuple[Any, float]:
    started = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - started


def load_scaled(spec: DatasetSpec) -> tuple[np.ndarray, np.ndarray]:
    data = spec.loader()
    x = np.asarray(data.data, dtype=np.float32)
    y = np.asarray(data.target)
    x = StandardScaler().fit_transform(x).astype(np.float32)
    return x, y


def cluster_and_score(
    representation: np.ndarray,
    y_true: np.ndarray,
    n_clusters: int,
    seed: int,
) -> dict[str, Any]:
    labels = KMeans(n_clusters=n_clusters, n_init=20, random_state=seed).fit_predict(representation)
    unique = np.unique(labels)
    silhouette = float("nan")
    if 1 < len(unique) < len(labels):
        silhouette = float(silhouette_score(representation, labels))

    return {
        "adjusted_rand": float(adjusted_rand_score(y_true, labels)),
        "normalized_mutual_info": float(normalized_mutual_info_score(y_true, labels)),
        "adjusted_mutual_info": float(adjusted_mutual_info_score(y_true, labels)),
        "homogeneity": float(homogeneity_score(y_true, labels)),
        "completeness": float(completeness_score(y_true, labels)),
        "v_measure": float(v_measure_score(y_true, labels)),
        "silhouette": silhouette,
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


def run_dataset(
    spec: DatasetSpec,
    seeds: list[int],
    n_neighbors: int,
    n_epochs: int,
    init: str,
) -> dict[str, Any]:
    x, y = load_scaled(spec)
    n_clusters = int(np.unique(y).size)
    runs: list[dict[str, Any]] = []

    for seed in seeds:
        raw_scores, raw_time = timed(lambda: cluster_and_score(x, y, n_clusters, seed))
        runs.append(
            {
                "seed": seed,
                "method": "raw_kmeans",
                "time_sec": raw_time,
                "scores": raw_scores,
            }
        )

        def pca_run() -> dict[str, Any]:
            pca = PCA(n_components=2, random_state=seed)
            emb = pca.fit_transform(x).astype(np.float32)
            scores = cluster_and_score(emb, y, n_clusters, seed)
            scores["explained_variance_ratio_sum"] = float(pca.explained_variance_ratio_.sum())
            return scores

        pca_scores, pca_time = timed(pca_run)
        runs.append(
            {
                "seed": seed,
                "method": "pca2_kmeans",
                "time_sec": pca_time,
                "scores": pca_scores,
            }
        )

        def rs_run() -> dict[str, Any]:
            emb = Umap(
                n_neighbors=n_neighbors,
                n_components=2,
                n_epochs=n_epochs,
                metric="euclidean",
                init=init,
                random_seed=seed,
                ann_mode="auto",
            ).fit_transform(x)
            return cluster_and_score(emb, y, n_clusters, seed)

        rs_scores, rs_time = timed(rs_run)
        runs.append(
            {
                "seed": seed,
                "method": "umap_rs2_kmeans",
                "time_sec": rs_time,
                "scores": rs_scores,
            }
        )

        def ul_run() -> dict[str, Any]:
            emb = make_umap_learn(seed, n_neighbors, n_epochs, init).fit_transform(x).astype(np.float32)
            return cluster_and_score(emb, y, n_clusters, seed)

        ul_scores, ul_time = timed(ul_run)
        runs.append(
            {
                "seed": seed,
                "method": "umap_learn2_kmeans",
                "time_sec": ul_time,
                "scores": ul_scores,
            }
        )

    return {
        "dataset": spec.name,
        "n_samples": int(x.shape[0]),
        "n_features": int(x.shape[1]),
        "n_classes": n_clusters,
        "runs": runs,
        "aggregate": aggregate_runs(runs),
    }


def aggregate_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    methods = sorted({row["method"] for row in runs})
    out: dict[str, Any] = {}
    for method in methods:
        method_runs = [row for row in runs if row["method"] == method]
        metrics = sorted(method_runs[0]["scores"].keys())
        scores: dict[str, Any] = {}
        for metric in metrics:
            values = np.asarray([row["scores"][metric] for row in method_runs], dtype=np.float64)
            scores[metric] = {
                "mean": float(np.nanmean(values)),
                "std": float(np.nanstd(values, ddof=0)),
            }
        times = np.asarray([row["time_sec"] for row in method_runs], dtype=np.float64)
        out[method] = {
            "time_sec": {
                "mean": float(times.mean()),
                "std": float(times.std(ddof=0)),
            },
            "scores": scores,
        }
    return out


def fmt_mean_std(value: dict[str, float], digits: int = 3) -> str:
    return f"{value['mean']:.{digits}f} +/- {value['std']:.{digits}f}"


def metric_range(report: dict[str, Any], metric: str) -> tuple[float, float]:
    values = []
    for ds in report["datasets"]:
        for method in METHOD_ORDER:
            metric_value = ds["aggregate"][method]["scores"][metric]
            values.extend([metric_value["mean"] - metric_value["std"], metric_value["mean"] + metric_value["std"]])

    finite = [value for value in values if np.isfinite(value)]
    if not finite:
        return 0.0, 1.0
    if min(finite) >= 0.0 and max(finite) <= 1.0:
        return 0.0, 1.0
    return min(-1.0, min(finite)), max(1.0, max(finite))


def make_metrics_svg(report: dict[str, Any]) -> str:
    datasets = report["datasets"]
    width = 1260
    panel_w = 370
    panel_h = 230
    cols = 3
    rows = (len(METRIC_ORDER) + cols - 1) // cols
    left_margin = 54
    top_margin = 104
    panel_gap_x = 36
    panel_gap_y = 56
    height = top_margin + rows * panel_h + (rows - 1) * panel_gap_y + 58
    plot_left_pad = 46
    plot_right_pad = 10
    plot_top_pad = 30
    plot_bottom_pad = 42

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{left_margin}" y="38" font-family="Inter, Arial, sans-serif" font-size="24" font-weight="700" fill="{TEXT_COLOR}">Real-data clustering metrics</text>',
        f'<text x="{left_margin}" y="64" font-family="Inter, Arial, sans-serif" font-size="13" fill="{MUTED_COLOR}">Mean score across seeds; whiskers show standard deviation.</text>',
    ]

    legend_y = 84
    for idx, method in enumerate(METHOD_ORDER):
        x = left_margin + idx * 136
        color = METHOD_COLORS[method]
        label = html.escape(METHOD_LABELS[method])
        lines.extend(
            [
                f'<rect x="{x}" y="{legend_y - 10}" width="14" height="14" rx="2" fill="{color}"/>',
                f'<text x="{x + 22}" y="{legend_y + 2}" font-family="Inter, Arial, sans-serif" font-size="13" fill="{TEXT_COLOR}">{label}</text>',
            ]
        )

    for metric_idx, (metric, metric_label) in enumerate(METRIC_ORDER):
        col = metric_idx % cols
        row = metric_idx // cols
        panel_x = left_margin + col * (panel_w + panel_gap_x)
        panel_y = top_margin + row * (panel_h + panel_gap_y)
        plot_x = panel_x + plot_left_pad
        plot_y = panel_y + plot_top_pad
        plot_w = panel_w - plot_left_pad - plot_right_pad
        plot_h = panel_h - plot_top_pad - plot_bottom_pad
        y_min, y_max = metric_range(report, metric)
        value_span = y_max - y_min

        def y_pos(value: float) -> float:
            return plot_y + plot_h - ((value - y_min) / value_span) * plot_h

        baseline_y = y_pos(0.0)
        ticks = [-1.0, -0.5, 0.0, 0.5, 1.0] if y_min < 0.0 else [0.0, 0.25, 0.5, 0.75, 1.0]

        lines.append(
            f'<text x="{panel_x}" y="{panel_y + 14}" font-family="Inter, Arial, sans-serif" font-size="16" font-weight="700" fill="{TEXT_COLOR}">{html.escape(metric_label)}</text>'
        )
        for tick in ticks:
            if tick < y_min or tick > y_max:
                continue
            y = y_pos(tick)
            lines.extend(
                [
                    f'<line x1="{plot_x}" y1="{y:.2f}" x2="{plot_x + plot_w}" y2="{y:.2f}" stroke="{GRID_COLOR}" stroke-width="1"/>',
                    f'<text x="{plot_x - 8}" y="{y + 4:.2f}" font-family="Inter, Arial, sans-serif" font-size="10" text-anchor="end" fill="{MUTED_COLOR}">{tick:.2g}</text>',
                ]
            )
        lines.extend(
            [
                f'<line x1="{plot_x}" y1="{baseline_y:.2f}" x2="{plot_x + plot_w}" y2="{baseline_y:.2f}" stroke="{AXIS_COLOR}" stroke-width="1"/>',
                f'<line x1="{plot_x}" y1="{plot_y}" x2="{plot_x}" y2="{plot_y + plot_h}" stroke="{AXIS_COLOR}" stroke-width="1"/>',
                f'<line x1="{plot_x}" y1="{plot_y + plot_h}" x2="{plot_x + plot_w}" y2="{plot_y + plot_h}" stroke="{AXIS_COLOR}" stroke-width="1"/>',
            ]
        )

        group_w = plot_w / max(1, len(datasets))
        bar_gap = 2.0
        bar_w = min(14.0, (group_w - 16.0) / len(METHOD_ORDER) - bar_gap)
        total_bar_w = len(METHOD_ORDER) * bar_w + (len(METHOD_ORDER) - 1) * bar_gap

        for ds_idx, ds in enumerate(datasets):
            group_center = plot_x + group_w * ds_idx + group_w / 2.0
            for method_idx, method in enumerate(METHOD_ORDER):
                metric_value = ds["aggregate"][method]["scores"][metric]
                mean = metric_value["mean"]
                std = metric_value["std"]
                if not np.isfinite(mean):
                    continue
                bar_x = group_center - total_bar_w / 2.0 + method_idx * (bar_w + bar_gap)
                value_y = y_pos(mean)
                rect_y = min(value_y, baseline_y)
                rect_h = abs(baseline_y - value_y)
                color = METHOD_COLORS[method]
                lines.append(
                    f'<rect x="{bar_x:.2f}" y="{rect_y:.2f}" width="{bar_w:.2f}" height="{rect_h:.2f}" fill="{color}"/>'
                )
                if np.isfinite(std) and std > 0.0:
                    y_low = y_pos(max(y_min, mean - std))
                    y_high = y_pos(min(y_max, mean + std))
                    x_mid = bar_x + bar_w / 2.0
                    lines.extend(
                        [
                            f'<line x1="{x_mid:.2f}" y1="{y_low:.2f}" x2="{x_mid:.2f}" y2="{y_high:.2f}" stroke="{TEXT_COLOR}" stroke-width="1"/>',
                            f'<line x1="{x_mid - 3:.2f}" y1="{y_low:.2f}" x2="{x_mid + 3:.2f}" y2="{y_low:.2f}" stroke="{TEXT_COLOR}" stroke-width="1"/>',
                            f'<line x1="{x_mid - 3:.2f}" y1="{y_high:.2f}" x2="{x_mid + 3:.2f}" y2="{y_high:.2f}" stroke="{TEXT_COLOR}" stroke-width="1"/>',
                        ]
                    )

            dataset_label = html.escape(ds["dataset"].replace("_", " "))
            lines.append(
                f'<text x="{group_center:.2f}" y="{plot_y + plot_h + 18}" font-family="Inter, Arial, sans-serif" font-size="10" text-anchor="middle" fill="{MUTED_COLOR}">{dataset_label}</text>'
            )

    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def write_markdown(report: dict[str, Any], path: Path, svg_path: Path) -> None:
    lines = [
        "# Current umapers Clustering Analysis Report",
        "",
        f"- generated_at: `{report['generated_at']}`",
        f"- python: `{report['environment']['python']}`",
        f"- umapers: `{report['environment']['umapers']}`",
        f"- umap_learn: `{report['environment']['umap_learn']}`",
        f"- seeds: `{report['settings']['seeds']}`",
        f"- UMAP: `n_neighbors={report['settings']['n_neighbors']}`, `n_epochs={report['settings']['n_epochs']}`, `init={report['settings']['init']}`, `metric=euclidean`",
        "",
        "Labels are used only for evaluation. UMAP runs are unsupervised; KMeans is run on each representation with the known number of classes.",
        "",
        f"![Current clustering metrics]({svg_path.name})",
        "",
        "## Runtime Summary",
        "",
        "| dataset | size | raw KMeans | PCA2 + KMeans | umapers + KMeans | umap-learn + KMeans | rs / learn |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for ds in report["datasets"]:
        aggregate = ds["aggregate"]
        size = f"{ds['n_samples']} x {ds['n_features']}"
        rs_time = aggregate["umap_rs2_kmeans"]["time_sec"]["mean"]
        learn_time = aggregate["umap_learn2_kmeans"]["time_sec"]["mean"]
        ratio = rs_time / learn_time if learn_time > 0 else float("nan")
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{ds['dataset']}`",
                    f"`{size}`",
                    fmt_mean_std(aggregate["raw_kmeans"]["time_sec"]),
                    fmt_mean_std(aggregate["pca2_kmeans"]["time_sec"]),
                    fmt_mean_std(aggregate["umap_rs2_kmeans"]["time_sec"]),
                    fmt_mean_std(aggregate["umap_learn2_kmeans"]["time_sec"]),
                    f"{ratio:.2f}x",
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
        "## Aggregate Metrics",
        "",
        "| dataset | size | method | ARI | NMI | AMI | V-measure | silhouette | time_sec |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
    )

    for ds in report["datasets"]:
        aggregate = ds["aggregate"]
        size = f"{ds['n_samples']} x {ds['n_features']}"
        for method in METHOD_ORDER:
            row = aggregate[method]
            scores = row["scores"]
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"`{ds['dataset']}`",
                        f"`{size}`",
                        f"`{method}`",
                        fmt_mean_std(scores["adjusted_rand"]),
                        fmt_mean_std(scores["normalized_mutual_info"]),
                        fmt_mean_std(scores["adjusted_mutual_info"]),
                        fmt_mean_std(scores["v_measure"]),
                        fmt_mean_std(scores["silhouette"]),
                        fmt_mean_std(row["time_sec"]),
                    ]
                )
                + " |"
            )

    lines.extend(
        [
            "",
            "## Best ARI By Dataset",
            "",
            "| dataset | best method | ARI | umapers ARI | umap-learn ARI |",
            "|---|---|---:|---:|---:|",
        ]
    )

    for ds in report["datasets"]:
        aggregate = ds["aggregate"]
        best_method = max(
            METHOD_ORDER,
            key=lambda method: aggregate[method]["scores"]["adjusted_rand"]["mean"],
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{ds['dataset']}`",
                    f"`{best_method}`",
                    fmt_mean_std(aggregate[best_method]["scores"]["adjusted_rand"]),
                    fmt_mean_std(aggregate["umap_rs2_kmeans"]["scores"]["adjusted_rand"]),
                    fmt_mean_std(aggregate["umap_learn2_kmeans"]["scores"]["adjusted_rand"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Dataset Shapes",
            "",
            "| dataset | samples | features | classes |",
            "|---|---:|---:|---:|",
        ]
    )
    for ds in report["datasets"]:
        lines.append(
            f"| `{ds['dataset']}` | {ds['n_samples']} | {ds['n_features']} | {ds['n_classes']} |"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path, default=Path("reports/current_clustering_analysis_report.json"))
    parser.add_argument("--output-md", type=Path, default=Path("reports/current_clustering_analysis_report.md"))
    parser.add_argument("--output-svg", type=Path, default=Path("reports/current_clustering_metrics.svg"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 23, 37])
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--n-epochs", type=int, default=200)
    parser.add_argument("--init", choices=["spectral", "random"], default="spectral")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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
        },
        "datasets": [],
    }

    for spec in DATASETS:
        print(f"running {spec.name}")
        report["datasets"].append(
            run_dataset(
                spec,
                seeds=args.seeds,
                n_neighbors=args.n_neighbors,
                n_epochs=args.n_epochs,
                init=args.init,
            )
        )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    args.output_svg.write_text(make_metrics_svg(report), encoding="utf-8")
    write_markdown(report, args.output_md, args.output_svg)
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_svg}")
    print(f"wrote {args.output_md}")


def version_of(package: str) -> str:
    try:
        from importlib.metadata import version

        return version(package)
    except Exception as exc:  # pragma: no cover - report best effort
        return f"unavailable: {type(exc).__name__}: {exc}"


if __name__ == "__main__":
    main()
