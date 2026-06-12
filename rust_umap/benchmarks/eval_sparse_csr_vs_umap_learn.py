#!/usr/bin/env python3
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
import umap
from scipy.linalg import orthogonal_procrustes
from scipy.spatial.distance import pdist
from sklearn.datasets import load_digits
from sklearn.manifold import trustworthiness


ROOT = Path(__file__).resolve().parents[2]
CRATE_DIR = ROOT / "rust_umap"
BENCH_BIN = CRATE_DIR / "target" / "release" / "bench_fit_csv"

THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
    "NUMBA_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "PYTHONHASHSEED": "0",
    "LC_ALL": "C",
    "LANG": "C",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark sparse CSR fit path (Rust vs umap-learn) on consistency/speed/memory"
    )
    p.add_argument(
        "--dataset",
        choices=["synthetic", "digits_sparse", "all"],
        default="all",
        help="datasets to benchmark",
    )
    p.add_argument("--n-neighbors", type=int, default=15)
    p.add_argument("--n-components", type=int, default=2)
    p.add_argument("--n-epochs", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument(
        "--run-order",
        choices=["balanced", "rust_first", "python_first"],
        default="balanced",
        help=(
            "execution order strategy: balanced runs both [rust->python] and [python->rust] "
            "for order debiasing"
        ),
    )
    p.add_argument("--synthetic-samples", type=int, default=1400)
    p.add_argument("--synthetic-features", type=int, default=1800)
    p.add_argument("--synthetic-density", type=float, default=0.01)
    p.add_argument(
        "--max-consistency-samples",
        type=int,
        default=800,
        help="subsample size for pairwise-distance consistency metrics",
    )
    p.add_argument("--output-json", default="", help="optional report output path")

    # Internal-only mode: run umap-learn in a subprocess so /usr/bin/time can isolate RSS.
    p.add_argument("--internal-python-fit", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--input-npz", default="", help=argparse.SUPPRESS)
    p.add_argument("--output-csv", default="", help=argparse.SUPPRESS)
    return p.parse_args()


def parse_max_rss_mb(text: str) -> Optional[float]:
    match = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", text)
    if not match:
        return None
    return float(match.group(1)) / 1024.0


def parse_elapsed_wall_sec(text: str) -> Optional[float]:
    match = re.search(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\):\s*([0-9:.]+)", text)
    if not match:
        return None
    token = match.group(1).strip()
    parts = token.split(":")
    try:
        if len(parts) == 3:
            hours = float(parts[0])
            minutes = float(parts[1])
            seconds = float(parts[2])
            return hours * 3600.0 + minutes * 60.0 + seconds
        if len(parts) == 2:
            minutes = float(parts[0])
            seconds = float(parts[1])
            return minutes * 60.0 + seconds
        return float(parts[0])
    except ValueError:
        return None


def parse_cpu_sec(text: str, key: str) -> Optional[float]:
    match = re.search(rf"{re.escape(key)}:\s*([0-9.]+)", text)
    if not match:
        return None
    return float(match.group(1))


def parse_time_verbose_report(text: str) -> Dict[str, Optional[float]]:
    return {
        "process_max_rss_mb": parse_max_rss_mb(text),
        "process_elapsed_wall_sec": parse_elapsed_wall_sec(text),
        "process_user_cpu_sec": parse_cpu_sec(text, "User time (seconds)"),
        "process_system_cpu_sec": parse_cpu_sec(text, "System time (seconds)"),
    }


def read_self_status_kb(field: str) -> Optional[int]:
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith(field):
                    parts = line.split()
                    if len(parts) >= 2:
                        return int(parts[1])
    except (OSError, ValueError):
        return None
    return None


def kb_to_mb(value_kb: Optional[int]) -> Optional[float]:
    if value_kb is None:
        return None
    return float(value_kb) / 1024.0


def safe_mean(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def safe_std(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return float(np.std(np.asarray(values, dtype=np.float64), ddof=0))


def safe_ratio(num: Optional[float], den: Optional[float]) -> Optional[float]:
    if num is None or den is None or den <= 0.0:
        return None
    return float(num / den)


def write_csv_vector(path: Path, values: np.ndarray, fmt: str) -> None:
    arr = np.asarray(values).reshape(1, -1)
    np.savetxt(path, arr, delimiter=",", fmt=fmt)


def load_embeddings(path: Path) -> np.ndarray:
    emb = np.loadtxt(path, delimiter=",", dtype=np.float32)
    if emb.ndim == 1:
        emb = emb.reshape(1, -1)
    return emb


def make_synthetic_sparse(
    n_samples: int, n_features: int, density: float, seed: int
) -> sp.csr_matrix:
    rng = np.random.default_rng(seed)

    def _data_rvs(k: int) -> np.ndarray:
        return rng.normal(loc=0.0, scale=1.0, size=k).astype(np.float32)

    x = sp.random(
        n_samples,
        n_features,
        density=density,
        format="csr",
        random_state=seed,
        data_rvs=_data_rvs,
    ).astype(np.float32)
    x.sum_duplicates()
    x.sort_indices()
    return x


def load_dataset(name: str, args: argparse.Namespace) -> sp.csr_matrix:
    if name == "synthetic":
        return make_synthetic_sparse(
            n_samples=args.synthetic_samples,
            n_features=args.synthetic_features,
            density=args.synthetic_density,
            seed=args.seed,
        )
    if name == "digits_sparse":
        x = load_digits().data.astype(np.float32)
        x = sp.csr_matrix(x)
        x.sum_duplicates()
        x.sort_indices()
        return x
    raise ValueError(f"unsupported dataset {name}")


def python_model_config(args: argparse.Namespace) -> Dict[str, object]:
    return {
        "n_neighbors": args.n_neighbors,
        "n_components": args.n_components,
        "metric": "euclidean",
        "n_epochs": args.n_epochs,
        "learning_rate": 1.0,
        "init": "random",
        "min_dist": 0.1,
        "spread": 1.0,
        "set_op_mix_ratio": 1.0,
        "local_connectivity": 1.0,
        "repulsion_strength": 1.0,
        "negative_sample_rate": 5,
        "random_state": args.seed,
        "transform_seed": args.seed,
        "low_memory": True,
        "force_approximation_algorithm": False,
        "n_jobs": 1,
        "verbose": False,
    }


def detect_python_neighbor_path(
    model: umap.UMAP, args: argparse.Namespace, x: sp.csr_matrix
) -> Dict[str, object]:
    knn_search_index = getattr(
        model,
        "_knn_search_index",
        getattr(model, "knn_search_index", None),
    )
    observed_small_data = bool(getattr(model, "_small_data", False))

    if observed_small_data:
        observed_path = "exact_small_data"
    elif knn_search_index is None:
        observed_path = "exact_no_knn_index"
    else:
        observed_path = "approx_knn_index"

    return {
        "library": "umap-learn",
        "version": getattr(umap, "__version__", "unknown"),
        "requested_metric": "euclidean",
        "requested_force_approximation_algorithm": False,
        "requested_low_memory": True,
        "requested_n_jobs": 1,
        "n_samples": int(x.shape[0]),
        "n_features": int(x.shape[1]),
        "observed_small_data": observed_small_data,
        "observed_knn_search_index_type": (
            type(knn_search_index).__name__ if knn_search_index is not None else None
        ),
        "observed_neighbor_search_path": observed_path,
    }


def run_rust_sparse(
    x: sp.csr_matrix, args: argparse.Namespace
) -> Tuple[Dict[str, object], np.ndarray]:
    with tempfile.TemporaryDirectory(prefix="umapers-sparse-rust-") as tmpdir:
        tmp = Path(tmpdir)
        dummy_input = tmp / "dummy.csv"
        out_csv = tmp / "rust_embedding.csv"
        indptr_csv = tmp / "indptr.csv"
        indices_csv = tmp / "indices.csv"
        values_csv = tmp / "values.csv"
        time_txt = tmp / "rust_time.txt"

        dummy_input.write_text("0\n", encoding="utf-8")
        write_csv_vector(indptr_csv, x.indptr, "%d")
        write_csv_vector(indices_csv, x.indices, "%d")
        write_csv_vector(values_csv, x.data.astype(np.float32), "%.8f")

        cmd = [
            "/usr/bin/time",
            "-v",
            "-o",
            str(time_txt),
            str(BENCH_BIN),
            str(dummy_input),
            str(out_csv),
            str(args.n_neighbors),
            str(args.n_components),
            str(args.n_epochs),
            str(args.seed),
            "random",
            "false",
            "30",
            "10",
            "4096",
            str(args.warmup),
            str(args.repeats),
            "--metric",
            "euclidean",
            "--csr-indptr",
            str(indptr_csv),
            "--csr-indices",
            str(indices_csv),
            "--csr-data",
            str(values_csv),
            "--csr-n-cols",
            str(x.shape[1]),
        ]
        env = os.environ.copy()
        env.update(THREAD_ENV)

        outer_t0 = time.perf_counter()
        stdout = subprocess.check_output(cmd, text=True, env=env)
        outer_elapsed = time.perf_counter() - outer_t0

        payload = json.loads(stdout)
        time_verbose = parse_time_verbose_report(time_txt.read_text(encoding="utf-8"))
        embedding = load_embeddings(out_csv)

        return (
            {
                "fit_times_sec": [float(v) for v in payload["fit_times_sec"]],
                "fit_mean_sec": float(payload["fit_mean_sec"]),
                "fit_std_sec": float(payload["fit_std_sec"]),
                "warmup": int(payload.get("warmup", args.warmup)),
                "repeats": int(payload.get("repeats", args.repeats)),
                "process_max_rss_mb": time_verbose["process_max_rss_mb"],
                "process_elapsed_wall_sec": (
                    time_verbose["process_elapsed_wall_sec"]
                    if time_verbose["process_elapsed_wall_sec"] is not None
                    else float(outer_elapsed)
                ),
                "process_elapsed_outer_sec": float(outer_elapsed),
                "process_user_cpu_sec": time_verbose["process_user_cpu_sec"],
                "process_system_cpu_sec": time_verbose["process_system_cpu_sec"],
                "algorithm_phase_memory_proxy_available": bool(
                    payload.get("algorithm_phase_memory_proxy_available", False)
                ),
                "algorithm_phase_peak_rss_delta_mb": payload.get(
                    "algorithm_phase_peak_rss_delta_mb"
                ),
                "algorithm_phase_vmrss_before_mb": payload.get(
                    "algorithm_phase_vmrss_before_mb"
                ),
                "algorithm_phase_vmrss_after_mb": payload.get(
                    "algorithm_phase_vmrss_after_mb"
                ),
                "algorithm_phase_vmhwm_before_mb": payload.get(
                    "algorithm_phase_vmhwm_before_mb"
                ),
                "algorithm_phase_vmhwm_after_mb": payload.get(
                    "algorithm_phase_vmhwm_after_mb"
                ),
                "raw_payload": payload,
            },
            embedding,
        )


def run_python_internal(args: argparse.Namespace) -> None:
    x = sp.load_npz(args.input_npz).tocsr().astype(np.float32)

    fit_times: List[float] = []
    embedding: Optional[np.ndarray] = None
    observed_ref_path: Optional[Dict[str, object]] = None

    vmhwm_before_kb = read_self_status_kb("VmHWM:")
    vmrss_before_kb = read_self_status_kb("VmRSS:")

    total = args.warmup + args.repeats
    for run_idx in range(total):
        model = umap.UMAP(**python_model_config(args))
        t0 = time.perf_counter()
        embedding = model.fit_transform(x)
        dt = time.perf_counter() - t0
        if run_idx >= args.warmup:
            fit_times.append(float(dt))
        observed_ref_path = detect_python_neighbor_path(model, args, x)

    vmhwm_after_kb = read_self_status_kb("VmHWM:")
    vmrss_after_kb = read_self_status_kb("VmRSS:")

    if embedding is None:
        raise RuntimeError("internal python fit did not produce embedding")

    np.savetxt(args.output_csv, embedding.astype(np.float32), delimiter=",", fmt="%.8f")

    fit_mean = safe_mean(fit_times)
    fit_std = safe_std(fit_times)

    algo_mem_proxy_available = (
        vmhwm_before_kb is not None
        and vmhwm_after_kb is not None
        and vmrss_before_kb is not None
        and vmrss_after_kb is not None
    )
    peak_delta_mb = None
    if vmhwm_before_kb is not None and vmhwm_after_kb is not None:
        peak_delta_mb = float(max(0, vmhwm_after_kb - vmhwm_before_kb)) / 1024.0

    payload: Dict[str, object] = {
        "fit_times_sec": fit_times,
        "fit_mean_sec": fit_mean,
        "fit_std_sec": fit_std,
        "warmup": int(args.warmup),
        "repeats": int(args.repeats),
        "algorithm_phase_memory_proxy_available": bool(algo_mem_proxy_available),
        "algorithm_phase_peak_rss_delta_mb": peak_delta_mb,
        "algorithm_phase_vmrss_before_mb": kb_to_mb(vmrss_before_kb),
        "algorithm_phase_vmrss_after_mb": kb_to_mb(vmrss_after_kb),
        "algorithm_phase_vmhwm_before_mb": kb_to_mb(vmhwm_before_kb),
        "algorithm_phase_vmhwm_after_mb": kb_to_mb(vmhwm_after_kb),
        "python_reference_path": observed_ref_path,
    }
    print(json.dumps(payload))


def run_python_sparse(
    x: sp.csr_matrix, args: argparse.Namespace
) -> Tuple[Dict[str, object], np.ndarray]:
    with tempfile.TemporaryDirectory(prefix="umapers-sparse-py-") as tmpdir:
        tmp = Path(tmpdir)
        input_npz = tmp / "input_sparse.npz"
        output_csv = tmp / "py_embedding.csv"
        time_txt = tmp / "py_time.txt"
        sp.save_npz(input_npz, x)

        cmd = [
            "/usr/bin/time",
            "-v",
            "-o",
            str(time_txt),
            sys.executable,
            str(__file__),
            "--internal-python-fit",
            "--input-npz",
            str(input_npz),
            "--output-csv",
            str(output_csv),
            "--n-neighbors",
            str(args.n_neighbors),
            "--n-components",
            str(args.n_components),
            "--n-epochs",
            str(args.n_epochs),
            "--seed",
            str(args.seed),
            "--warmup",
            str(args.warmup),
            "--repeats",
            str(args.repeats),
        ]
        env = os.environ.copy()
        env.update(THREAD_ENV)

        outer_t0 = time.perf_counter()
        stdout = subprocess.check_output(cmd, text=True, env=env)
        outer_elapsed = time.perf_counter() - outer_t0

        payload = json.loads(stdout)
        time_verbose = parse_time_verbose_report(time_txt.read_text(encoding="utf-8"))
        embedding = load_embeddings(output_csv)

        return (
            {
                "fit_times_sec": [float(v) for v in payload["fit_times_sec"]],
                "fit_mean_sec": float(payload["fit_mean_sec"]),
                "fit_std_sec": float(payload["fit_std_sec"]),
                "warmup": int(payload.get("warmup", args.warmup)),
                "repeats": int(payload.get("repeats", args.repeats)),
                "process_max_rss_mb": time_verbose["process_max_rss_mb"],
                "process_elapsed_wall_sec": (
                    time_verbose["process_elapsed_wall_sec"]
                    if time_verbose["process_elapsed_wall_sec"] is not None
                    else float(outer_elapsed)
                ),
                "process_elapsed_outer_sec": float(outer_elapsed),
                "process_user_cpu_sec": time_verbose["process_user_cpu_sec"],
                "process_system_cpu_sec": time_verbose["process_system_cpu_sec"],
                "algorithm_phase_memory_proxy_available": bool(
                    payload.get("algorithm_phase_memory_proxy_available", False)
                ),
                "algorithm_phase_peak_rss_delta_mb": payload.get(
                    "algorithm_phase_peak_rss_delta_mb"
                ),
                "algorithm_phase_vmrss_before_mb": payload.get(
                    "algorithm_phase_vmrss_before_mb"
                ),
                "algorithm_phase_vmrss_after_mb": payload.get(
                    "algorithm_phase_vmrss_after_mb"
                ),
                "algorithm_phase_vmhwm_before_mb": payload.get(
                    "algorithm_phase_vmhwm_before_mb"
                ),
                "algorithm_phase_vmhwm_after_mb": payload.get(
                    "algorithm_phase_vmhwm_after_mb"
                ),
                "python_reference_path": payload.get("python_reference_path"),
                "raw_payload": payload,
            },
            embedding,
        )


def procrustes_rmse(rust_emb: np.ndarray, py_emb: np.ndarray) -> float:
    rust = rust_emb - rust_emb.mean(axis=0, keepdims=True)
    py = py_emb - py_emb.mean(axis=0, keepdims=True)
    rust /= np.linalg.norm(rust) + 1e-12
    py /= np.linalg.norm(py) + 1e-12
    rotation, _ = orthogonal_procrustes(py, rust)
    py_aligned = py @ rotation
    return float(np.sqrt(np.mean((rust - py_aligned) ** 2)))


def pairwise_distance_corr(
    rust_emb: np.ndarray, py_emb: np.ndarray, max_samples: int, seed: int
) -> float:
    n = rust_emb.shape[0]
    if n <= 2:
        return 1.0
    rng = np.random.default_rng(seed)
    m = min(n, max_samples)
    idx = rng.choice(n, size=m, replace=False)
    rust_d = pdist(rust_emb[idx], metric="euclidean")
    py_d = pdist(py_emb[idx], metric="euclidean")
    if rust_d.size == 0 or py_d.size == 0:
        return 1.0
    corr = np.corrcoef(rust_d, py_d)[0, 1]
    return float(corr)


def benchmark_orders(mode: str) -> List[Tuple[str, str]]:
    if mode == "rust_first":
        return [("rust", "python")]
    if mode == "python_first":
        return [("python", "rust")]
    return [("rust", "python"), ("python", "rust")]


def flatten_fit_times(runs: Sequence[Dict[str, object]]) -> List[float]:
    values: List[float] = []
    for run in runs:
        values.extend(float(v) for v in run["fit_times_sec"])
    return values


def aggregate_impl_runs(
    impl_name: str, runs: Sequence[Dict[str, object]]
) -> Dict[str, object]:
    fit_times = flatten_fit_times(runs)
    fit_means_by_trial = [float(run["fit_mean_sec"]) for run in runs]

    process_wall_by_trial = [float(run["process_elapsed_wall_sec"]) for run in runs]
    process_rss_by_trial = [
        float(run["process_max_rss_mb"])
        for run in runs
        if run["process_max_rss_mb"] is not None
    ]

    algo_peak_delta_by_trial = [
        float(run["algorithm_phase_peak_rss_delta_mb"])
        for run in runs
        if run.get("algorithm_phase_memory_proxy_available")
        and run.get("algorithm_phase_peak_rss_delta_mb") is not None
    ]

    return {
        "impl": impl_name,
        "num_trials": len(runs),
        "order_trace": [
            {
                "order_trial": int(run["order_trial"]),
                "order_sequence": run["order_sequence"],
                "order_position": int(run["order_position"]),
            }
            for run in runs
        ],
        "algorithm_fit_times_sec": fit_times,
        "algorithm_fit_mean_sec": safe_mean(fit_times),
        "algorithm_fit_std_sec": safe_std(fit_times),
        "algorithm_fit_mean_sec_by_trial": fit_means_by_trial,
        "algorithm_fit_mean_sec_of_trials": safe_mean(fit_means_by_trial),
        "process_wall_total_sec_by_trial": process_wall_by_trial,
        "process_wall_total_mean_sec": safe_mean(process_wall_by_trial),
        "process_wall_total_std_sec": safe_std(process_wall_by_trial),
        "process_max_rss_mb_by_trial": process_rss_by_trial,
        "process_max_rss_mean_mb": safe_mean(process_rss_by_trial),
        "process_max_rss_std_mb": safe_std(process_rss_by_trial),
        "algorithm_phase_peak_rss_delta_mb_by_trial": algo_peak_delta_by_trial,
        "algorithm_phase_peak_rss_delta_mean_mb": safe_mean(algo_peak_delta_by_trial),
        "algorithm_phase_peak_rss_delta_std_mb": safe_std(algo_peak_delta_by_trial),
        "algorithm_phase_memory_proxy_available_trials": len(algo_peak_delta_by_trial),
        "algorithm_phase_memory_proxy_total_trials": len(runs),
        "runs": list(runs),
    }


def evaluate_dataset(name: str, x: sp.csr_matrix, args: argparse.Namespace) -> Dict[str, object]:
    orders = benchmark_orders(args.run_order)
    runs_by_impl: Dict[str, List[Dict[str, object]]] = {"rust": [], "python": []}
    first_embedding: Dict[str, np.ndarray] = {}

    for order_trial, order in enumerate(orders):
        for order_position, impl in enumerate(order):
            if impl == "rust":
                payload, embedding = run_rust_sparse(x, args)
            elif impl == "python":
                payload, embedding = run_python_sparse(x, args)
            else:
                raise ValueError(f"unknown impl {impl}")

            payload["order_trial"] = order_trial
            payload["order_sequence"] = list(order)
            payload["order_position"] = order_position
            runs_by_impl[impl].append(payload)

            if impl not in first_embedding:
                first_embedding[impl] = embedding

    rust_embedding = first_embedding["rust"]
    py_embedding = first_embedding["python"]

    rust_trust = trustworthiness(x, rust_embedding, n_neighbors=args.n_neighbors)
    py_trust = trustworthiness(x, py_embedding, n_neighbors=args.n_neighbors)
    embed_corr = pairwise_distance_corr(
        rust_embedding, py_embedding, args.max_consistency_samples, args.seed
    )
    align_rmse = procrustes_rmse(rust_embedding, py_embedding)

    rust_agg = aggregate_impl_runs("rust", runs_by_impl["rust"])
    py_agg = aggregate_impl_runs("python", runs_by_impl["python"])

    rust_algo_mean = rust_agg["algorithm_fit_mean_sec"]
    py_algo_mean = py_agg["algorithm_fit_mean_sec"]
    rust_process_mean = rust_agg["process_wall_total_mean_sec"]
    py_process_mean = py_agg["process_wall_total_mean_sec"]

    rust_rss_mean = rust_agg["process_max_rss_mean_mb"]
    py_rss_mean = py_agg["process_max_rss_mean_mb"]

    rust_algo_rss_proxy_mean = rust_agg["algorithm_phase_peak_rss_delta_mean_mb"]
    py_algo_rss_proxy_mean = py_agg["algorithm_phase_peak_rss_delta_mean_mb"]

    python_ref_paths = [
        run.get("python_reference_path")
        for run in runs_by_impl["python"]
        if run.get("python_reference_path")
    ]
    observed_py_modes = [
        path.get("observed_neighbor_search_path")
        for path in python_ref_paths
        if isinstance(path, dict)
    ]

    return {
        "shape": [int(x.shape[0]), int(x.shape[1])],
        "nnz": int(x.nnz),
        "density": float(x.nnz / (x.shape[0] * x.shape[1])),
        "run_order": {
            "strategy": args.run_order,
            "orders": [list(order) for order in orders],
            "total_trials_per_impl": len(runs_by_impl["rust"]),
        },
        "consistency": {
            "trustworthiness_rust": float(rust_trust),
            "trustworthiness_python": float(py_trust),
            "trustworthiness_delta_rust_minus_python": float(rust_trust - py_trust),
            "embedding_pairwise_distance_corr": embed_corr,
            "procrustes_rmse": align_rmse,
        },
        "speed": {
            "algorithm": {
                "rust_fit_mean_sec": rust_algo_mean,
                "rust_fit_std_sec": rust_agg["algorithm_fit_std_sec"],
                "python_fit_mean_sec": py_algo_mean,
                "python_fit_std_sec": py_agg["algorithm_fit_std_sec"],
                "python_over_rust_ratio": safe_ratio(py_algo_mean, rust_algo_mean),
                "notes": "Algorithm timing uses warmup-trimmed fit repeats only (same warmup/repeats for Rust and Python).",
            },
            "process": {
                "rust_wall_total_mean_sec": rust_process_mean,
                "rust_wall_total_std_sec": rust_agg["process_wall_total_std_sec"],
                "python_wall_total_mean_sec": py_process_mean,
                "python_wall_total_std_sec": py_agg["process_wall_total_std_sec"],
                "python_over_rust_ratio": safe_ratio(py_process_mean, rust_process_mean),
                "notes": "Process wall time from /usr/bin/time includes startup/runtime overhead + warmup + measured repeats.",
            },
            # Backward-compatible aliases (algorithm timing).
            "rust_fit_mean_sec": rust_algo_mean,
            "python_fit_mean_sec": py_algo_mean,
            "python_over_rust_ratio": safe_ratio(py_algo_mean, rust_algo_mean),
        },
        "memory": {
            "rss_sampling": {
                "process_max_rss_source": "/usr/bin/time -v Maximum resident set size (kbytes)",
                "algorithm_phase_proxy_source": "in-process /proc/self/status VmHWM delta across fit loop",
                "limitations": [
                    "process_max_rss includes startup/runtime overhead and one-time allocations",
                    "algorithm_phase VmHWM delta is a proxy; allocator reuse and prior high-water marks can hide transient peaks",
                    "cold start effects are reduced by separate algorithm/process reporting and balanced run order",
                ],
            },
            "process_max_rss_mb": {
                "rust_mean_mb": rust_rss_mean,
                "rust_std_mb": rust_agg["process_max_rss_std_mb"],
                "python_mean_mb": py_rss_mean,
                "python_std_mb": py_agg["process_max_rss_std_mb"],
                "python_over_rust_ratio": safe_ratio(py_rss_mean, rust_rss_mean),
            },
            "algorithm_phase_peak_rss_delta_mb_proxy": {
                "rust_mean_mb": rust_algo_rss_proxy_mean,
                "rust_std_mb": rust_agg["algorithm_phase_peak_rss_delta_std_mb"],
                "rust_available_trials": rust_agg[
                    "algorithm_phase_memory_proxy_available_trials"
                ],
                "python_mean_mb": py_algo_rss_proxy_mean,
                "python_std_mb": py_agg["algorithm_phase_peak_rss_delta_std_mb"],
                "python_available_trials": py_agg[
                    "algorithm_phase_memory_proxy_available_trials"
                ],
                "python_over_rust_ratio": safe_ratio(
                    py_algo_rss_proxy_mean, rust_algo_rss_proxy_mean
                ),
            },
            # Backward-compatible aliases (process max RSS).
            "rust_process_max_rss_mb": rust_rss_mean,
            "python_process_max_rss_mb": py_rss_mean,
            "python_over_rust_ratio": safe_ratio(py_rss_mean, rust_rss_mean),
        },
        "reference_paths": {
            "rust": {
                "sparse_knn_path": "exact_sparse_csr_euclidean",
                "use_approximate_knn": False,
            },
            "python": {
                "requested_force_approximation_algorithm": False,
                "observed_neighbor_search_paths": observed_py_modes,
                "observed_neighbor_search_path_consistent": (
                    len(set(observed_py_modes)) <= 1 if observed_py_modes else True
                ),
                "observed_effective_neighbor_search_path": (
                    observed_py_modes[0] if observed_py_modes else None
                ),
                "trial_details": python_ref_paths,
            },
        },
        "rust_summary": rust_agg,
        "python_summary": py_agg,
        # Backward-compatible raw fields.
        "rust_raw": runs_by_impl["rust"][0]["raw_payload"],
        "python_raw": runs_by_impl["python"][0]["raw_payload"],
    }


def main() -> None:
    args = parse_args()

    if args.internal_python_fit:
        run_python_internal(args)
        return

    datasets = ["synthetic", "digits_sparse"] if args.dataset == "all" else [args.dataset]

    subprocess.run(
        ["cargo", "build", "--release", "--quiet", "--bin", "bench_fit_csv"],
        cwd=CRATE_DIR,
        check=True,
    )

    report: Dict[str, object] = {
        "config": {
            "metric": "euclidean",
            "n_neighbors": args.n_neighbors,
            "n_components": args.n_components,
            "n_epochs": args.n_epochs,
            "seed": args.seed,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "run_order": args.run_order,
            "thread_env": THREAD_ENV,
            "timing_definitions": {
                "algorithm_fit": "warmup-trimmed fit loop timing",
                "process_total": "subprocess wall time including startup/runtime overhead",
            },
            "memory_definitions": {
                "process_max_rss": "Maximum resident set size from /usr/bin/time -v",
                "algorithm_phase_proxy": "VmHWM delta (after fit loop minus before fit loop)",
            },
        },
        "datasets": {},
    }

    for name in datasets:
        x = load_dataset(name, args)
        report["datasets"][name] = evaluate_dataset(name, x, args)

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
