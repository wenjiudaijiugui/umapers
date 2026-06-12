#!/usr/bin/env python3
"""Release-prep regression orchestrator for Wave 3 convergence."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from .rust_build_utils import ReleaseBins, build_release_bins
except ImportError:  # pragma: no cover - direct script execution path
    from rust_build_utils import ReleaseBins, build_release_bins


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GATE_CONFIG = ROOT / "benchmarks" / "gate_thresholds.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Wave 3 release-prep regression gates "
            "(wave1/ann/consistency/no-regression) with a unified summary."
        )
    )
    parser.add_argument(
        "--candidate-root",
        default=str(ROOT),
        help="candidate repository root (default: current repo root)",
    )
    parser.add_argument(
        "--baseline-root",
        default=os.environ.get("UMAP_BENCH_BASELINE_ROOT", "").strip(),
        help="baseline repository root for no-regression gate",
    )
    parser.add_argument(
        "--python-bin",
        default=sys.executable,
        help="python executable used to invoke child gate scripts",
    )
    parser.add_argument(
        "--metrics",
        default="euclidean,manhattan,cosine",
        help="comma-separated metrics for no-regression gate",
    )
    parser.add_argument(
        "--gate-config",
        default=str(DEFAULT_GATE_CONFIG),
        help="gate config path for frozen thresholds and defaults",
    )
    parser.add_argument(
        "--output-json",
        default="release-prep-regression.json",
        help="output JSON summary path",
    )
    return parser.parse_args()


def _tail(text: str, max_chars: int = 2000) -> str:
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def _parse_metrics(raw: str) -> List[str]:
    metrics = [part.strip() for part in raw.split(",") if part.strip()]
    if not metrics:
        raise SystemExit("--metrics must contain at least one metric")
    invalid = [metric for metric in metrics if metric not in {"euclidean", "manhattan", "cosine"}]
    if invalid:
        raise SystemExit(
            "unsupported metric(s): "
            + ", ".join(invalid)
            + " (expected euclidean, manhattan, cosine)"
        )
    return metrics


def _ensure_paths(candidate_root: Path, baseline_root: Optional[Path]) -> None:
    if not candidate_root.exists():
        raise SystemExit(f"candidate root not found: {candidate_root}")
    if not (candidate_root / "benchmarks").exists():
        raise SystemExit(f"candidate root does not look like the umapers repo: {candidate_root}")
    if baseline_root is None:
        raise SystemExit("--baseline-root is required for no-regression gate")
    if not baseline_root.exists():
        raise SystemExit(f"baseline root not found: {baseline_root}")
    if candidate_root.resolve() == baseline_root.resolve():
        raise SystemExit("candidate-root and baseline-root must be different directories")


def _ensure_gate_config(path: Optional[Path]) -> None:
    if path is None:
        return
    if not path.exists():
        raise SystemExit(f"gate config not found: {path}")


def _script_help_text(
    python_bin: str, script_path: Path, cache: Dict[str, str]
) -> str:
    key = str(script_path)
    cached = cache.get(key)
    if cached is not None:
        return cached
    proc = subprocess.run(
        [python_bin, str(script_path), "--help"],
        cwd=script_path.parent.parent,
        check=False,
        capture_output=True,
        text=True,
    )
    help_text = f"{proc.stdout}\n{proc.stderr}"
    cache[key] = help_text
    return help_text


def _supports_option(
    python_bin: str, script_path: Path, option_name: str, cache: Dict[str, str]
) -> bool:
    return option_name in _script_help_text(python_bin, script_path, cache)


def _extract_json_payload(text: str) -> Optional[Dict[str, Any]]:
    decoder = json.JSONDecoder()
    last_obj: Optional[Dict[str, Any]] = None
    for idx, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, end = decoder.raw_decode(text[idx:])
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        if text[idx + end :].strip():
            continue
        last_obj = obj
    return last_obj


def _result_pass(payload: Optional[Dict[str, Any]], returncode: int) -> bool:
    if returncode != 0:
        return False
    if payload is not None and isinstance(payload.get("overall_pass"), bool):
        return bool(payload["overall_pass"])
    return True


def _write_payload_if_needed(path: Path, payload: Optional[Dict[str, Any]]) -> None:
    if payload is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _path_for_command(path: Path, cwd: Path) -> str:
    resolved = path.resolve()
    try:
        rel = resolved.relative_to(cwd.resolve())
    except ValueError:
        return str(resolved)
    rel_str = str(rel)
    return rel_str if rel_str else "."


def _candidate_rust_bin_args(candidate_bins: ReleaseBins, cwd: Path) -> List[str]:
    if candidate_bins.fit_csv is None:
        raise RuntimeError("candidate fit_csv binary is required for skip-rust-build gates")
    return [
        "--rust-fit-bin",
        _path_for_command(candidate_bins.fit_csv, cwd),
        "--rust-bench-bin",
        _path_for_command(candidate_bins.bench_fit_csv, cwd),
        "--skip-rust-build",
    ]


def _run_gate(
    gate_id: str,
    python_bin: str,
    script_path: Path,
    script_args: List[str],
    cwd: Path,
    artifact_path: Path,
    gate_config: Optional[Path],
    help_cache: Dict[str, str],
) -> Dict[str, Any]:
    command = [python_bin, _path_for_command(script_path, cwd), *script_args]
    gate_config_forwarded = False
    output_json_forwarded = False

    if gate_config is not None and _supports_option(python_bin, script_path, "--gate-config", help_cache):
        command.extend(["--gate-config", _path_for_command(gate_config, cwd)])
        gate_config_forwarded = True

    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    stale_artifact_removed = False
    if artifact_path.exists():
        artifact_path.unlink()
        stale_artifact_removed = True

    if _supports_option(python_bin, script_path, "--output-json", help_cache):
        command.extend(["--output-json", _path_for_command(artifact_path, cwd)])
        output_json_forwarded = True

    t0 = time.perf_counter()
    proc = subprocess.run(
        command,
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )
    elapsed = time.perf_counter() - t0

    payload: Optional[Dict[str, Any]] = None
    if output_json_forwarded and artifact_path.exists():
        try:
            payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = None
    if payload is None:
        payload = _extract_json_payload(proc.stdout)
        if payload is not None:
            _write_payload_if_needed(artifact_path, payload)

    passed = _result_pass(payload, proc.returncode)
    failures: List[str] = []
    if not passed:
        if payload is not None and isinstance(payload.get("failures"), list):
            failures.extend(str(item) for item in payload["failures"])
        if not failures:
            failures.append(f"{gate_id} exited with code {proc.returncode}")

    return {
        "id": gate_id,
        "pass": passed,
        "elapsed_sec": elapsed,
        "command": " ".join(shlex.quote(tok) for tok in command),
        "artifact_path": _path_for_command(artifact_path, cwd),
        "returncode": proc.returncode,
        "gate_config_forwarded": gate_config_forwarded,
        "output_json_forwarded": output_json_forwarded,
        "stale_artifact_removed": stale_artifact_removed,
        "stdout_tail": _tail(proc.stdout),
        "stderr_tail": _tail(proc.stderr),
        "failures": failures,
        "payload": payload,
    }


def main() -> int:
    args = parse_args()

    candidate_root = Path(args.candidate_root).resolve()
    baseline_root = Path(args.baseline_root).resolve() if args.baseline_root else None
    gate_config = Path(args.gate_config).resolve() if args.gate_config else None
    metrics = _parse_metrics(args.metrics)
    _ensure_paths(candidate_root, baseline_root)
    _ensure_gate_config(gate_config)
    assert baseline_root is not None

    output_json = Path(args.output_json).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    artifacts_dir = output_json.parent / f"{output_json.stem}.artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    bench_dir = candidate_root / "benchmarks"
    help_cache: Dict[str, str] = {}

    wave1_script = bench_dir / "ci_wave1_smoke.py"
    ann_script = bench_dir / "ci_ann_smoke.py"
    consistency_script = bench_dir / "ci_consistency_smoke.py"
    no_regression_script = bench_dir / "ci_no_regression.py"
    with ThreadPoolExecutor(max_workers=2) as executor:
        candidate_future = executor.submit(build_release_bins, candidate_root, need_fit_bin=True)
        baseline_future = executor.submit(build_release_bins, baseline_root, need_fit_bin=False)
        candidate_bins = candidate_future.result()
        baseline_bins = baseline_future.result()

    results: List[Dict[str, Any]] = []
    failures: List[str] = []

    results.append(
        _run_gate(
            gate_id="wave1_smoke",
            python_bin=args.python_bin,
            script_path=wave1_script,
            script_args=[
                "--manifest-path",
                _path_for_command(candidate_root / "rust_umap" / "Cargo.toml", candidate_root),
            ],
            cwd=candidate_root,
            artifact_path=artifacts_dir / "wave1-smoke.json",
            gate_config=gate_config,
            help_cache=help_cache,
        )
    )

    results.append(
        _run_gate(
            gate_id="ann_e2e_smoke",
            python_bin=args.python_bin,
            script_path=ann_script,
            script_args=[
                "--python-bin",
                args.python_bin,
                *_candidate_rust_bin_args(candidate_bins, candidate_root),
            ],
            cwd=candidate_root,
            artifact_path=artifacts_dir / "ann-e2e-smoke.json",
            gate_config=gate_config,
            help_cache=help_cache,
        )
    )

    results.append(
        _run_gate(
            gate_id="consistency_smoke",
            python_bin=args.python_bin,
            script_path=consistency_script,
            script_args=[
                "--python-bin",
                args.python_bin,
                *_candidate_rust_bin_args(candidate_bins, candidate_root),
            ],
            cwd=candidate_root,
            artifact_path=artifacts_dir / "consistency-smoke.json",
            gate_config=gate_config,
            help_cache=help_cache,
        )
    )

    for metric in metrics:
        results.append(
            _run_gate(
                gate_id=f"no_regression_smoke:{metric}",
                python_bin=args.python_bin,
                script_path=no_regression_script,
                script_args=[
                    "--candidate-root",
                    _path_for_command(candidate_root, candidate_root),
                    "--baseline-root",
                    _path_for_command(baseline_root, candidate_root),
                    "--candidate-bin",
                    _path_for_command(candidate_bins.bench_fit_csv, candidate_root),
                    "--baseline-bin",
                    _path_for_command(baseline_bins.bench_fit_csv, candidate_root),
                    "--metric",
                    metric,
                ],
                cwd=candidate_root,
                artifact_path=artifacts_dir / f"no-regression-smoke-{metric}.json",
                gate_config=gate_config,
                help_cache=help_cache,
            )
        )

    overall_pass = True
    for result in results:
        if not result["pass"]:
            overall_pass = False
            failures.extend(result["failures"])

    summary = {
        "schema_version": 1,
        "gate": "release_prep_regression",
        "strict": True,
        "overall_pass": overall_pass,
        "timestamp_unix": int(time.time()),
        "candidate_root": _path_for_command(candidate_root, candidate_root),
        "baseline_root": _path_for_command(baseline_root, candidate_root)
        if baseline_root is not None
        else "",
        "metrics": metrics,
        "gate_config": _path_for_command(gate_config, candidate_root) if gate_config is not None else "",
        "artifacts_dir": _path_for_command(artifacts_dir, candidate_root),
        "build_reuse": {
            "candidate_fit_csv": _path_for_command(candidate_bins.fit_csv, candidate_root)
            if candidate_bins.fit_csv is not None
            else "",
            "candidate_bench_fit_csv": _path_for_command(
                candidate_bins.bench_fit_csv, candidate_root
            ),
            "baseline_bench_fit_csv": _path_for_command(
                baseline_bins.bench_fit_csv, candidate_root
            ),
        },
        "failures": failures,
        "results": results,
    }

    output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
