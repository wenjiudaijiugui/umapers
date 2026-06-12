#!/usr/bin/env python3
"""Plot umapers and umap-learn runtime against dataset matrix size."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


RS_COLOR = "#2563eb"
LEARN_COLOR = "#dc2626"
GRID_COLOR = "#d4d4d8"
TEXT_COLOR = "#18181b"
MUTED_COLOR = "#71717a"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-json",
        type=Path,
        default=Path("reports/current_clustering_analysis_report.json"),
    )
    parser.add_argument(
        "--output-svg",
        type=Path,
        default=Path("reports/runtime_vs_dataset_size.svg"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("reports/runtime_vs_dataset_size.md"),
    )
    return parser.parse_args()


def load_points(path: Path) -> list[dict[str, Any]]:
    report = json.loads(path.read_text(encoding="utf-8"))
    points: list[dict[str, Any]] = []
    for dataset in report["datasets"]:
        aggregate = dataset["aggregate"]
        rs = aggregate["umap_rs2_kmeans"]["time_sec"]
        learn = aggregate["umap_learn2_kmeans"]["time_sec"]
        samples = int(dataset["n_samples"])
        features = int(dataset["n_features"])
        points.append(
            {
                "dataset": dataset["dataset"],
                "samples": samples,
                "features": features,
                "size": samples * features,
                "rs_mean": float(rs["mean"]),
                "rs_std": float(rs["std"]),
                "learn_mean": float(learn["mean"]),
                "learn_std": float(learn["std"]),
            }
        )
    points.sort(key=lambda row: row["size"])
    return points


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


def fmt(value: float) -> str:
    return f"{value:.3f}"


def make_svg(points: list[dict[str, Any]]) -> str:
    width = 1120
    height = 720
    left = 96
    right = 48
    top = 82
    bottom = 96
    plot_w = width - left - right
    plot_h = height - top - bottom
    x_min = min(row["size"] for row in points)
    x_max = max(row["size"] for row in points)
    y_max = max(max(row["rs_mean"], row["learn_mean"]) for row in points) * 1.12

    def x(row: dict[str, Any]) -> float:
        return scale_log(row["size"], x_min, x_max, left, left + plot_w)

    def y(value: float) -> float:
        return scale_linear(value, 0.0, y_max, top + plot_h, top)

    rs_points = [(x(row), y(row["rs_mean"])) for row in points]
    learn_points = [(x(row), y(row["learn_mean"])) for row in points]
    y_ticks = [0.0, 1.0, 2.0, 3.0, 4.0]
    if y_max > 4.5:
        y_ticks.append(5.0)

    lines: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{left}" y="34" font-family="Inter, Arial, sans-serif" font-size="24" font-weight="700" fill="{TEXT_COLOR}">UMAP Runtime vs Dataset Size</text>',
        f'<text x="{left}" y="58" font-family="Inter, Arial, sans-serif" font-size="13" fill="{MUTED_COLOR}">x-axis is matrix size: samples x features. Runtime includes 2D UMAP plus KMeans, mean over three seeds.</text>',
    ]

    for tick in y_ticks:
        if tick > y_max:
            continue
        ty = y(tick)
        lines.append(
            f'<line x1="{left}" y1="{ty:.2f}" x2="{left + plot_w}" y2="{ty:.2f}" stroke="{GRID_COLOR}" stroke-width="1"/>'
        )
        lines.append(
            f'<text x="{left - 12}" y="{ty + 4:.2f}" text-anchor="end" font-family="Inter, Arial, sans-serif" font-size="12" fill="{MUTED_COLOR}">{tick:.0f}s</text>'
        )

    for row in points:
        tx = x(row)
        lines.append(
            f'<line x1="{tx:.2f}" y1="{top}" x2="{tx:.2f}" y2="{top + plot_h}" stroke="#f4f4f5" stroke-width="1"/>'
        )
        label = f"{row['dataset']} ({row['samples']}x{row['features']})"
        lines.append(
            f'<text x="{tx:.2f}" y="{top + plot_h + 28}" text-anchor="middle" font-family="Inter, Arial, sans-serif" font-size="12" fill="{TEXT_COLOR}">{label}</text>'
        )
        lines.append(
            f'<text x="{tx:.2f}" y="{top + plot_h + 46}" text-anchor="middle" font-family="Inter, Arial, sans-serif" font-size="11" fill="{MUTED_COLOR}">{row["size"]:,} values</text>'
        )

    axis_y = top + plot_h
    lines.extend(
        [
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{axis_y}" stroke="{TEXT_COLOR}" stroke-width="1.2"/>',
            f'<line x1="{left}" y1="{axis_y}" x2="{left + plot_w}" y2="{axis_y}" stroke="{TEXT_COLOR}" stroke-width="1.2"/>',
            f'<text x="28" y="{top + plot_h / 2:.2f}" transform="rotate(-90 28 {top + plot_h / 2:.2f})" text-anchor="middle" font-family="Inter, Arial, sans-serif" font-size="13" fill="{TEXT_COLOR}">runtime seconds</text>',
            f'<text x="{left + plot_w / 2:.2f}" y="{height - 24}" text-anchor="middle" font-family="Inter, Arial, sans-serif" font-size="13" fill="{TEXT_COLOR}">dataset matrix size, log scale</text>',
            f'<polyline fill="none" stroke="{RS_COLOR}" stroke-width="3" points="{polyline(rs_points)}"/>',
            f'<polyline fill="none" stroke="{LEARN_COLOR}" stroke-width="3" points="{polyline(learn_points)}"/>',
        ]
    )

    for row, (rx, ry), (lx, ly) in zip(points, rs_points, learn_points, strict=True):
        lines.append(f'<circle cx="{rx:.2f}" cy="{ry:.2f}" r="5" fill="{RS_COLOR}"/>')
        lines.append(f'<circle cx="{lx:.2f}" cy="{ly:.2f}" r="5" fill="{LEARN_COLOR}"/>')
        lines.append(
            f'<text x="{rx:.2f}" y="{ry - 12:.2f}" text-anchor="middle" font-family="Inter, Arial, sans-serif" font-size="11" fill="{RS_COLOR}">{fmt(row["rs_mean"])}s</text>'
        )
        lines.append(
            f'<text x="{lx:.2f}" y="{ly + 20:.2f}" text-anchor="middle" font-family="Inter, Arial, sans-serif" font-size="11" fill="{LEARN_COLOR}">{fmt(row["learn_mean"])}s</text>'
        )

    legend_x = left + plot_w - 220
    legend_y = top + 12
    lines.extend(
        [
            f'<rect x="{legend_x}" y="{legend_y}" width="190" height="64" rx="6" fill="#ffffff" stroke="{GRID_COLOR}"/>',
            f'<line x1="{legend_x + 16}" y1="{legend_y + 22}" x2="{legend_x + 50}" y2="{legend_y + 22}" stroke="{RS_COLOR}" stroke-width="3"/>',
            f'<circle cx="{legend_x + 33}" cy="{legend_y + 22}" r="5" fill="{RS_COLOR}"/>',
            f'<text x="{legend_x + 62}" y="{legend_y + 26}" font-family="Inter, Arial, sans-serif" font-size="13" fill="{TEXT_COLOR}">umapers</text>',
            f'<line x1="{legend_x + 16}" y1="{legend_y + 46}" x2="{legend_x + 50}" y2="{legend_y + 46}" stroke="{LEARN_COLOR}" stroke-width="3"/>',
            f'<circle cx="{legend_x + 33}" cy="{legend_y + 46}" r="5" fill="{LEARN_COLOR}"/>',
            f'<text x="{legend_x + 62}" y="{legend_y + 50}" font-family="Inter, Arial, sans-serif" font-size="13" fill="{TEXT_COLOR}">umap-learn</text>',
        ]
    )

    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def write_markdown(points: list[dict[str, Any]], path: Path, svg_path: Path) -> None:
    rel_svg = svg_path.name
    lines = [
        "# Runtime vs Dataset Size",
        "",
        f"![Runtime vs dataset size]({rel_svg})",
        "",
        "| dataset | samples | features | matrix size | umapers sec | umap-learn sec | rs / learn |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in points:
        ratio = row["rs_mean"] / row["learn_mean"] if row["learn_mean"] > 0 else float("nan")
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['dataset']}`",
                    str(row["samples"]),
                    str(row["features"]),
                    f"{row['size']:,}",
                    f"{row['rs_mean']:.3f} +/- {row['rs_std']:.3f}",
                    f"{row['learn_mean']:.3f} +/- {row['learn_std']:.3f}",
                    f"{ratio:.2f}x",
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    points = load_points(args.input_json)
    args.output_svg.parent.mkdir(parents=True, exist_ok=True)
    args.output_svg.write_text(make_svg(points), encoding="utf-8")
    write_markdown(points, args.output_md, args.output_svg)
    print(f"wrote {args.output_svg}")
    print(f"wrote {args.output_md}")


if __name__ == "__main__":
    main()
