#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
BENCH_DIR = ROOT / "benchmarks"
sys.path.insert(0, str(BENCH_DIR))

from gate_config import THRESHOLDS_PATH, emit_report, gate_report, load_gate_config
from rust_build_utils import build_release_bins

THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
    "NUMBA_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "PYTHONHASHSEED": "0",
}

DEFAULT_CANDIDATE_ROOT = Path(os.environ.get("UMAP_BENCH_CANDIDATE_ROOT", str(ROOT)))
DEFAULT_BASELINE_ROOT = os.environ.get("UMAP_BENCH_BASELINE_ROOT", "").strip()
_GNU_TIME_BIN_CACHE: Optional[str] = None


def _unique_non_empty(items: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in items:
        value = item.strip()
        if not value or value in seen:
            continue
        out.append(value)
        seen.add(value)
    return out


def _time_candidates() -> List[str]:
    override = os.environ.get("UMAP_BENCH_TIME_BIN", "").strip()
    candidates: List[str] = []
    if override:
        candidates.append(override)

    for exe in ["gtime", "time"]:
        resolved = shutil.which(exe)
        if resolved:
            candidates.append(resolved)

    candidates.extend(["/usr/bin/time", "gtime", "time"])
    return _unique_non_empty(candidates)


def _probe_gnu_time(candidate: str) -> Tuple[bool, str]:
    cmd = [candidate, "-v", "true"]
    try:
        proc = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            env=os.environ.copy(),
        )
    except FileNotFoundError:
        return False, "not found"
    except PermissionError:
        return False, "not executable"
    except OSError as exc:
        return False, f"probe failed: {exc}"

    out = f"{proc.stdout}\n{proc.stderr}"
    if "Maximum resident set size" in out:
        return True, "ok"

    version_hint = ""
    try:
        ver = subprocess.run(
            [candidate, "--version"],
            check=False,
            capture_output=True,
            text=True,
            env=os.environ.copy(),
        )
        ver_out = f"{ver.stdout}\n{ver.stderr}"
        first_line = next((line.strip() for line in ver_out.splitlines() if line.strip()), "")
        if first_line:
            version_hint = f" (version: {first_line[:80]})"
    except OSError:
        pass

    return False, f"missing GNU max RSS field in '-v' output{version_hint}"


def resolve_time_bin() -> str:
    global _GNU_TIME_BIN_CACHE
    if _GNU_TIME_BIN_CACHE:
        return _GNU_TIME_BIN_CACHE

    attempted: List[str] = []
    for candidate in _time_candidates():
        ok, reason = _probe_gnu_time(candidate)
        attempted.append(f"{candidate}: {reason}")
        if ok:
            _GNU_TIME_BIN_CACHE = candidate
            return candidate

    attempted_str = "; ".join(attempted) if attempted else "none"
    raise RuntimeError(
        "Unable to find GNU time with '-v' max RSS reporting. "
        f"Tried: {attempted_str}. "
        "Set UMAP_BENCH_TIME_BIN to a GNU time executable path. "
        "Install hint: Debian/Ubuntu `apt-get install time`; macOS `brew install gnu-time`."
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "CI no-regression gate for Rust fit timing and RSS with randomized paired order "
            "and robust ratio statistics"
        )
    )
    p.add_argument("--gate-config", default=str(THRESHOLDS_PATH))
    p.add_argument("--candidate-root", default=str(DEFAULT_CANDIDATE_ROOT))
    p.add_argument("--baseline-root", default=DEFAULT_BASELINE_ROOT or None)
    p.add_argument("--warmup", type=int, default=None)
    p.add_argument("--repeats", type=int, default=None)
    p.add_argument("--paired-runs", type=int, default=None)
    p.add_argument("--time-ratio-threshold", type=float, default=None)
    p.add_argument("--rss-ratio-threshold", type=float, default=None)
    p.add_argument("--tail-slack", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--metric", choices=["euclidean", "manhattan", "cosine"], default="euclidean")
    p.add_argument(
        "--candidate-bin",
        default="",
        help="prebuilt candidate bench_fit_csv path (skip candidate build when set)",
    )
    p.add_argument(
        "--baseline-bin",
        default="",
        help="prebuilt baseline bench_fit_csv path (skip baseline build when set)",
    )
    p.add_argument("--output-json", default=None)
    args = p.parse_args()
    config = load_gate_config("no_regression_smoke", args.gate_config)
    if args.warmup is None:
        args.warmup = int(config["warmup"])
    if args.repeats is None:
        args.repeats = int(config["repeats"])
    if args.paired_runs is None:
        args.paired_runs = int(config["paired_runs"])
    if args.time_ratio_threshold is None:
        args.time_ratio_threshold = float(config["time_ratio_threshold"])
    if args.rss_ratio_threshold is None:
        args.rss_ratio_threshold = float(config["rss_ratio_threshold"])
    if args.tail_slack is None:
        args.tail_slack = float(config["tail_slack"])
    if args.seed is None:
        args.seed = int(config["seed"])
    return args


def build_release(root: Path) -> Path:
    return build_release_bins(root, need_fit_bin=False).bench_fit_csv


def generate_dataset(seed: int, n_samples: int = 1400, n_features: int = 24) -> np.ndarray:
    import numpy as np

    rng = np.random.default_rng(seed)
    centers = np.stack(
        [
            np.linspace(-2.0, 2.0, n_features, dtype=np.float32),
            np.linspace(2.0, -2.0, n_features, dtype=np.float32),
            np.zeros(n_features, dtype=np.float32),
        ]
    )
    labels = rng.integers(0, len(centers), size=n_samples)
    noise = rng.normal(loc=0.0, scale=0.4, size=(n_samples, n_features)).astype(np.float32)
    x = centers[labels] + noise
    x -= x.mean(axis=0, keepdims=True)
    x /= x.std(axis=0, keepdims=True) + 1e-6
    return x.astype(np.float32)


def parse_rss_mb(time_file: Path) -> float:
    for line in time_file.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = line.strip()
        if stripped.startswith("Maximum resident set size (kbytes):"):
            return float(stripped.split(":", 1)[1].strip()) / 1024.0
    raise RuntimeError(f"failed to parse max RSS from {time_file}")


def run_bench(
    binary: Path,
    data_path: Path,
    output_path: Path,
    time_path: Path,
    warmup: int,
    repeats: int,
    seed: int,
    metric: str,
) -> Dict[str, float]:
    time_bin = resolve_time_bin()
    cmd = [
        time_bin,
        "-v",
        "-o",
        str(time_path),
        str(binary),
        str(data_path),
        str(output_path),
        "15",
        "2",
        "120",
        str(seed),
        "random",
        "false",
        "30",
        "10",
        "4096",
        str(warmup),
        str(repeats),
        "--metric",
        metric,
    ]
    env = os.environ.copy()
    env.update(THREAD_ENV)

    t0 = time.perf_counter()
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True, env=env)
    wall = time.perf_counter() - t0
    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    return {
        "fit_mean_sec": float(payload["fit_mean_sec"]),
        "fit_std_sec": float(payload["fit_std_sec"]),
        "process_wall_sec": wall,
        "process_max_rss_mb": parse_rss_mb(time_path),
    }


def ratio_stats(candidate_vals: List[float], baseline_vals: List[float]) -> Dict[str, float]:
    import numpy as np

    c = np.asarray(candidate_vals, dtype=np.float64)
    b = np.asarray(baseline_vals, dtype=np.float64)
    if c.size != b.size:
        raise ValueError("candidate and baseline sample sizes must match for paired ratio stats")
    if c.size == 0:
        raise ValueError("empty sample set for ratio stats")
    if np.any(b <= 0.0):
        raise ValueError("baseline contains non-positive values, ratio undefined")

    ratio = c / b
    return {
        "median": float(np.median(ratio)),
        "p75": float(np.percentile(ratio, 75.0)),
        "mean": float(np.mean(ratio)),
        "std": float(np.std(ratio)),
        "samples": ratio.tolist(),
    }


def main() -> None:
    args = parse_args()
    config = load_gate_config("no_regression_smoke", args.gate_config)
    import numpy as np

    if args.paired_runs < 1:
        raise SystemExit("--paired-runs must be >= 1")
    if not args.baseline_root:
        raise SystemExit("--baseline-root is required (or set UMAP_BENCH_BASELINE_ROOT)")

    candidate_root = Path(args.candidate_root).resolve()
    baseline_root = Path(args.baseline_root).resolve()

    candidate_bin = Path(args.candidate_bin).resolve() if args.candidate_bin else build_release(candidate_root)
    baseline_bin = Path(args.baseline_bin).resolve() if args.baseline_bin else build_release(baseline_root)
    if not candidate_bin.exists():
        raise SystemExit(f"candidate binary not found: {candidate_bin}")
    if not baseline_bin.exists():
        raise SystemExit(f"baseline binary not found: {baseline_bin}")

    rng = np.random.default_rng(args.seed ^ 0x5EEDC0DE)

    per_impl: Dict[str, List[Dict[str, float]]] = {"baseline": [], "candidate": []}
    paired_order: List[List[str]] = []

    with tempfile.TemporaryDirectory(prefix="umapers-ci-regression-") as tmpdir:
        tmp = Path(tmpdir)
        x = generate_dataset(args.seed)
        data_path = tmp / "dataset.csv"
        np.savetxt(data_path, x, delimiter=",", fmt="%.8f")

        for pair_idx in range(args.paired_runs):
            order = ["baseline", "candidate"]
            if int(rng.integers(0, 2)) == 1:
                order.reverse()
            paired_order.append(order)

            for impl in order:
                bin_path = baseline_bin if impl == "baseline" else candidate_bin
                run_metrics = run_bench(
                    bin_path,
                    data_path,
                    tmp / f"{impl}_pair{pair_idx}.out.csv",
                    tmp / f"{impl}_pair{pair_idx}.time.txt",
                    args.warmup,
                    args.repeats,
                    args.seed + pair_idx,
                    args.metric,
                )
                per_impl[impl].append(run_metrics)

    baseline_fit = [x["fit_mean_sec"] for x in per_impl["baseline"]]
    candidate_fit = [x["fit_mean_sec"] for x in per_impl["candidate"]]
    baseline_rss = [x["process_max_rss_mb"] for x in per_impl["baseline"]]
    candidate_rss = [x["process_max_rss_mb"] for x in per_impl["candidate"]]

    time_ratio = ratio_stats(candidate_fit, baseline_fit)
    rss_ratio = ratio_stats(candidate_rss, baseline_rss)

    summary = {
        "metric": args.metric,
        "paired_runs": args.paired_runs,
        "paired_order": paired_order,
        "candidate": per_impl["candidate"],
        "baseline": per_impl["baseline"],
        "ratios": {
            "fit_mean_sec": time_ratio,
            "process_max_rss_mb": rss_ratio,
        },
        "thresholds": {
            "fit_mean_sec_median": args.time_ratio_threshold,
            "fit_mean_sec_p75": args.time_ratio_threshold * (1.0 + args.tail_slack),
            "process_max_rss_mb_median": args.rss_ratio_threshold,
            "process_max_rss_mb_p75": args.rss_ratio_threshold * (1.0 + args.tail_slack),
        },
        "thread_env": THREAD_ENV,
        "system": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
    }
    failures: List[str] = []
    if time_ratio["median"] > args.time_ratio_threshold:
        failures.append(
            f"candidate fit_mean_sec median ratio {time_ratio['median']:.4f} exceeded threshold {args.time_ratio_threshold:.4f}"
        )
    if time_ratio["p75"] > args.time_ratio_threshold * (1.0 + args.tail_slack):
        failures.append(
            "candidate fit_mean_sec p75 ratio "
            f"{time_ratio['p75']:.4f} exceeded threshold {args.time_ratio_threshold * (1.0 + args.tail_slack):.4f}"
        )

    if rss_ratio["median"] > args.rss_ratio_threshold:
        failures.append(
            f"candidate process_max_rss_mb median ratio {rss_ratio['median']:.4f} exceeded threshold {args.rss_ratio_threshold:.4f}"
        )
    if rss_ratio["p75"] > args.rss_ratio_threshold * (1.0 + args.tail_slack):
        failures.append(
            "candidate process_max_rss_mb p75 ratio "
            f"{rss_ratio['p75']:.4f} exceeded threshold {args.rss_ratio_threshold * (1.0 + args.tail_slack):.4f}"
        )

    report = gate_report(
        gate="no_regression_smoke",
        strict=bool(config["strict"]),
        overall_pass=not failures,
        thresholds=summary["thresholds"],
        failures=failures,
        details=summary,
    )
    emit_report(report, args.output_json)

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
