#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import time
import traceback
import warnings
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("NUMBA_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
from scipy import sparse
from sklearn.datasets import load_breast_cancer, load_digits, load_wine
from sklearn.manifold import trustworthiness
from sklearn.metrics import accuracy_score, mean_squared_error, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.preprocessing import StandardScaler

from umapers import AlignedUmap, ParametricUmap, Umap, fit_transform as rs_fit_transform

try:
    import umap
except Exception as exc:  # pragma: no cover - script reports this explicitly
    umap = None
    UMAP_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"
else:
    UMAP_IMPORT_ERROR = ""


Record = dict[str, Any]


def to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (float, int, str, bool)) or value is None:
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return str(value)
        return value
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return str(value)


def timed(fn: Callable[[], dict[str, Any]]) -> Record:
    start = time.perf_counter()
    try:
        payload = fn()
    except Exception as exc:  # pragma: no cover - report path
        return {
            "ok": False,
            "time_sec": time.perf_counter() - start,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback_tail": traceback.format_exc().splitlines()[-8:],
        }
    return {"ok": True, "time_sec": time.perf_counter() - start, **payload}


def scaled_dataset(name: str, limit: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    if name == "digits":
        data = load_digits()
    elif name == "wine":
        data = load_wine()
    elif name == "breast_cancer":
        data = load_breast_cancer()
    else:
        raise ValueError(f"unknown dataset {name!r}")

    x = StandardScaler().fit_transform(data.data.astype(np.float32)).astype(np.float32)
    y = data.target.astype(np.int64)
    if limit is not None and x.shape[0] > limit:
        rng = np.random.default_rng(42)
        keep = rng.choice(x.shape[0], size=limit, replace=False)
        keep.sort()
        x = x[keep]
        y = y[keep]
    return x, y


def quality(x: np.ndarray, y: np.ndarray | None, emb: np.ndarray, n_neighbors: int) -> dict[str, Any]:
    report: dict[str, Any] = {
        "shape": list(emb.shape),
        "finite": bool(np.all(np.isfinite(emb))),
        "trustworthiness": float(
            trustworthiness(x, emb, n_neighbors=min(n_neighbors, max(1, x.shape[0] - 2)))
        ),
    }
    if y is not None and len(np.unique(y)) > 1 and emb.shape[0] >= 3:
        report["embedding_silhouette"] = float(silhouette_score(emb, y))
    return report


def compare_result(
    scenario: str,
    description: str,
    dataset: str,
    rs_fn: Callable[[], dict[str, Any]],
    ul_fn: Callable[[], dict[str, Any]] | None,
    *,
    comparability: str = "direct",
) -> Record:
    row: Record = {
        "scenario": scenario,
        "description": description,
        "dataset": dataset,
        "comparability": comparability,
        "umap_rs": timed(rs_fn),
    }
    if ul_fn is None:
        row["umap_learn"] = {
            "ok": False,
            "skipped": True,
            "reason": "no directly comparable umap-learn core API in this environment",
        }
    elif umap is None:
        row["umap_learn"] = {
            "ok": False,
            "skipped": True,
            "reason": UMAP_IMPORT_ERROR,
        }
    else:
        row["umap_learn"] = timed(ul_fn)
    return row


def make_umap_learn(**kwargs: Any) -> Any:
    if umap is None:
        raise RuntimeError(UMAP_IMPORT_ERROR)
    defaults = {
        "random_state": 42,
        "n_jobs": 1,
        "low_memory": True,
    }
    defaults.update(kwargs)
    return umap.UMAP(**defaults)


def warmup() -> dict[str, Any]:
    x = np.random.default_rng(7).normal(size=(40, 5)).astype(np.float32)
    out: dict[str, Any] = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        timed(lambda: {"shape": list(Umap(n_neighbors=5, n_epochs=10, random_seed=7, init="random").fit_transform(x).shape)})
        if umap is not None:
            out["umap_learn"] = timed(
                lambda: {
                    "shape": list(
                        make_umap_learn(n_neighbors=5, n_epochs=10, init="random").fit_transform(x).shape
                    )
                }
            )
        else:
            out["umap_learn"] = {"ok": False, "error": UMAP_IMPORT_ERROR}
    return out


def dense_fit_transform_case() -> Record:
    x, y = scaled_dataset("digits", 420)
    k = 12
    epochs = 60

    def rs() -> dict[str, Any]:
        emb = Umap(
            n_neighbors=k,
            n_components=2,
            n_epochs=epochs,
            metric="euclidean",
            random_seed=11,
            init="random",
            ann_mode="exact",
        ).fit_transform(x)
        top = rs_fit_transform(
            x[:120],
            n_neighbors=k,
            n_components=2,
            n_epochs=epochs,
            random_seed=11,
            init="random",
            ann_mode="exact",
        )
        return {**quality(x, y, emb, k), "top_level_shape": list(top.shape)}

    def ul() -> dict[str, Any]:
        emb = make_umap_learn(
            n_neighbors=k,
            n_components=2,
            n_epochs=epochs,
            metric="euclidean",
            init="random",
        ).fit_transform(x)
        return quality(x, y, emb.astype(np.float32), k)

    return compare_result(
        "dense_fit_transform_digits",
        "Dense unsupervised embedding on the sklearn digits dataset.",
        "sklearn.load_digits",
        rs,
        ul,
    )


def transform_inverse_out_case() -> Record:
    x, y = scaled_dataset("digits", 360)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=5, stratify=y
    )
    k = 10
    epochs = 60

    def rs() -> dict[str, Any]:
        model = Umap(
            n_neighbors=k,
            n_components=2,
            n_epochs=epochs,
            metric="euclidean",
            random_seed=13,
            init="random",
            ann_mode="exact",
        )
        out = np.empty((x_train.shape[0], 2), dtype=np.float32)
        emb_train = model.fit_transform(x_train, out=out)
        query_out = np.empty((x_test.shape[0], 2), dtype=np.float32)
        emb_test = model.transform(x_test, out=query_out)
        reconstructed = model.inverse_transform(emb_test[:24])
        pred = KNeighborsClassifier(n_neighbors=5).fit(emb_train, y_train).predict(emb_test)
        return {
            "fit_out_identity": bool(emb_train is out),
            "transform_out_identity": bool(emb_test is query_out),
            "query_shape": list(emb_test.shape),
            "query_finite": bool(np.all(np.isfinite(emb_test))),
            "embedding_knn_accuracy": float(accuracy_score(y_test, pred)),
            "inverse_mse_first24": float(mean_squared_error(x_test[:24], reconstructed)),
        }

    def ul() -> dict[str, Any]:
        model = make_umap_learn(
            n_neighbors=k,
            n_components=2,
            n_epochs=epochs,
            metric="euclidean",
            init="random",
        )
        emb_train = model.fit_transform(x_train)
        emb_test = model.transform(x_test)
        reconstructed = model.inverse_transform(emb_test[:24])
        pred = KNeighborsClassifier(n_neighbors=5).fit(emb_train, y_train).predict(emb_test)
        return {
            "query_shape": list(emb_test.shape),
            "query_finite": bool(np.all(np.isfinite(emb_test))),
            "embedding_knn_accuracy": float(accuracy_score(y_test, pred)),
            "inverse_mse_first24": float(mean_squared_error(x_test[:24], reconstructed)),
        }

    return compare_result(
        "transform_inverse_digits",
        "Fit on train digits, transform held-out digits, and reconstruct a small query batch.",
        "sklearn.load_digits",
        rs,
        ul,
    )


def supervised_case() -> Record:
    x, y = scaled_dataset("wine", None)
    k = 12
    epochs = 70

    def rs() -> dict[str, Any]:
        emb = Umap(
            n_neighbors=k,
            n_components=2,
            n_epochs=epochs,
            metric="euclidean",
            target_metric="categorical",
            target_weight=0.6,
            random_seed=17,
            init="random",
            ann_mode="exact",
        ).fit_transform(x, y)
        return quality(x, y, emb, k)

    def ul() -> dict[str, Any]:
        emb = make_umap_learn(
            n_neighbors=k,
            n_components=2,
            n_epochs=epochs,
            metric="euclidean",
            target_metric="categorical",
            target_weight=0.6,
            init="random",
        ).fit_transform(x, y)
        return quality(x, y, emb.astype(np.float32), k)

    return compare_result(
        "categorical_supervised_wine",
        "Categorical supervised embedding on sklearn wine labels.",
        "sklearn.load_wine",
        rs,
        ul,
    )


def metric_sweep_case() -> Record:
    x, y = scaled_dataset("breast_cancer", 220)
    k = 10
    epochs = 45
    metrics: list[tuple[str, dict[str, Any] | None]] = [
        ("euclidean", None),
        ("manhattan", None),
        ("cosine", None),
        ("chebyshev", None),
        ("minkowski", {"p": 3.0}),
        ("correlation", None),
        ("canberra", None),
        ("braycurtis", None),
    ]

    def run_rs() -> dict[str, Any]:
        rows: dict[str, Any] = {}
        for metric, metric_kwds in metrics:
            emb = Umap(
                n_neighbors=k,
                n_components=2,
                n_epochs=epochs,
                metric=metric,
                metric_kwds=metric_kwds,
                random_seed=19,
                init="random",
                ann_mode="exact",
            ).fit_transform(x)
            rows[metric] = quality(x, y, emb, k)
        return rows

    def run_ul() -> dict[str, Any]:
        rows: dict[str, Any] = {}
        for metric, metric_kwds in metrics:
            kwargs: dict[str, Any] = {
                "n_neighbors": k,
                "n_components": 2,
                "n_epochs": epochs,
                "metric": metric,
                "init": "random",
            }
            if metric_kwds is not None:
                kwargs["metric_kwds"] = metric_kwds
            emb = make_umap_learn(**kwargs).fit_transform(x)
            rows[metric] = quality(x, y, emb.astype(np.float32), k)
        return rows

    return compare_result(
        "expanded_metric_sweep_breast_cancer",
        "Dense metric sweep across currently exposed metric names.",
        "sklearn.load_breast_cancer",
        run_rs,
        run_ul,
    )


def densmap_case() -> Record:
    x, y = scaled_dataset("breast_cancer", 260)
    k = 12
    epochs = 55

    def summarize(result: Any) -> dict[str, Any]:
        emb, radii_original, radii_embedding = result
        return {
            **quality(x, y, emb.astype(np.float32), k),
            "radii_original_finite": bool(np.all(np.isfinite(radii_original))),
            "radii_embedding_finite": bool(np.all(np.isfinite(radii_embedding))),
            "radii_original_shape": list(np.asarray(radii_original).shape),
            "radii_embedding_shape": list(np.asarray(radii_embedding).shape),
        }

    def rs() -> dict[str, Any]:
        return summarize(
            Umap(
                n_neighbors=k,
                n_components=2,
                n_epochs=epochs,
                metric="euclidean",
                densmap=True,
                output_dens=True,
                random_seed=23,
                init="random",
                ann_mode="exact",
            ).fit_transform(x)
        )

    def ul() -> dict[str, Any]:
        return summarize(
            make_umap_learn(
                n_neighbors=k,
                n_components=2,
                n_epochs=epochs,
                metric="euclidean",
                densmap=True,
                output_dens=True,
                init="random",
            ).fit_transform(x)
        )

    return compare_result(
        "densmap_output_breast_cancer",
        "densMAP with output radii on breast cancer features.",
        "sklearn.load_breast_cancer",
        rs,
        ul,
    )


def sparse_case() -> Record:
    x, y = scaled_dataset("digits", 360)
    x_sparse_source = x.copy()
    x_sparse_source[np.abs(x_sparse_source) < 0.25] = 0.0
    csr = sparse.csr_matrix(x_sparse_source)
    csc = sparse.csc_matrix(x_sparse_source)
    coo_query = sparse.coo_matrix(x_sparse_source[:40])
    k = 10
    epochs = 55

    def rs() -> dict[str, Any]:
        model = Umap(
            n_neighbors=k,
            n_components=2,
            n_epochs=epochs,
            metric="cosine",
            random_seed=29,
            init="random",
            ann_mode="exact",
        )
        emb = model.fit_transform(csr)
        query = model.transform(coo_query)
        csc_emb = Umap(
            n_neighbors=k,
            n_components=2,
            n_epochs=epochs,
            metric="cosine",
            random_seed=29,
            init="random",
            ann_mode="exact",
        ).fit_transform(csc)
        return {
            **quality(x, y, emb, k),
            "sparsity": float(1.0 - csr.nnz / (csr.shape[0] * csr.shape[1])),
            "sparse_query_shape": list(query.shape),
            "sparse_query_finite": bool(np.all(np.isfinite(query))),
            "csc_fit_shape": list(csc_emb.shape),
        }

    def ul() -> dict[str, Any]:
        model = make_umap_learn(
            n_neighbors=k,
            n_components=2,
            n_epochs=epochs,
            metric="cosine",
            init="random",
        )
        emb = model.fit_transform(csr)
        query = model.transform(coo_query)
        csc_emb = make_umap_learn(
            n_neighbors=k,
            n_components=2,
            n_epochs=epochs,
            metric="cosine",
            init="random",
        ).fit_transform(csc)
        return {
            **quality(x, y, emb.astype(np.float32), k),
            "sparsity": float(1.0 - csr.nnz / (csr.shape[0] * csr.shape[1])),
            "sparse_query_shape": list(query.shape),
            "sparse_query_finite": bool(np.all(np.isfinite(query))),
            "csc_fit_shape": list(csc_emb.shape),
        }

    return compare_result(
        "sparse_digits_csr_csc_coo",
        "Sparse CSR fit plus sparse COO query on thresholded real digits features.",
        "sklearn.load_digits thresholded to sparse",
        rs,
        ul,
    )


def precomputed_knn_case() -> Record:
    x, y = scaled_dataset("wine", None)
    k = 12
    epochs = 60
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="brute", metric="euclidean", n_jobs=1)
    nbrs.fit(x)
    dists_all, idx_all = nbrs.kneighbors(x)
    rs_idx = idx_all[:, 1 : k + 1].astype(np.int64)
    rs_dist = dists_all[:, 1 : k + 1].astype(np.float32)
    ul_idx = idx_all[:, :k].astype(np.int64)
    ul_dist = dists_all[:, :k].astype(np.float32)

    def rs() -> dict[str, Any]:
        emb = Umap(
            n_neighbors=k,
            n_components=2,
            n_epochs=epochs,
            metric="euclidean",
            random_seed=31,
            init="random",
            ann_mode="exact",
        ).fit_transform_with_knn(x, rs_idx, rs_dist, knn_metric="euclidean")
        return quality(x, y, emb, k)

    def ul() -> dict[str, Any]:
        emb = make_umap_learn(
            n_neighbors=k,
            n_components=2,
            n_epochs=epochs,
            metric="euclidean",
            init="random",
            precomputed_knn=(ul_idx, ul_dist, None),
        ).fit_transform(x)
        return quality(x, y, emb.astype(np.float32), k)

    return compare_result(
        "precomputed_knn_wine",
        "Library-native precomputed exact kNN path. Contracts differ: umap-rs receives neighbors excluding self; umap-learn receives its native self-inclusive kNN tuple.",
        "sklearn.load_wine",
        rs,
        ul,
        comparability="library-native precomputed contracts, not byte-identical graph input",
    )


def ann_case() -> Record:
    x, y = scaled_dataset("digits", 500)
    k = 12
    epochs = 60

    def rs() -> dict[str, Any]:
        model = Umap(
            n_neighbors=k,
            n_components=2,
            n_epochs=epochs,
            metric="euclidean",
            random_seed=37,
            init="random",
            ann_mode="approximate",
        )
        diag = model.knn_diagnostics(x)
        emb = model.fit_transform(x)
        return {
            **quality(x, y, emb, k),
            "knn_mean_recall": float(diag["mean_recall"]),
            "knn_worst_decile_recall": float(diag["worst_decile_recall"]),
        }

    def ul() -> dict[str, Any]:
        emb = make_umap_learn(
            n_neighbors=k,
            n_components=2,
            n_epochs=epochs,
            metric="euclidean",
            init="random",
        ).fit_transform(x)
        return quality(x, y, emb.astype(np.float32), k)

    return compare_result(
        "approximate_ann_digits",
        "Approximate-neighbor embedding on digits. umap-rs additionally reports exact-vs-approx kNN recall.",
        "sklearn.load_digits",
        rs,
        ul,
        comparability="algorithmic ANN comparison; backends differ",
    )


def parametric_case() -> Record:
    x, y = scaled_dataset("breast_cancer", 220)
    x_train, x_test = train_test_split(x, test_size=0.2, random_state=3, stratify=y)
    k = 10

    def rs() -> dict[str, Any]:
        model = ParametricUmap(
            n_neighbors=k,
            n_components=2,
            n_epochs=45,
            metric="euclidean",
            hidden_dim=24,
            train_epochs=35,
            batch_size=64,
            random_seed=41,
            train_mode="optimized",
        )
        emb = model.fit_transform(x_train)
        query = model.transform(x_test)
        return {
            "train_shape": list(emb.shape),
            "query_shape": list(query.shape),
            "train_finite": bool(np.all(np.isfinite(emb))),
            "query_finite": bool(np.all(np.isfinite(query))),
            "n_features_in": int(model.n_features_in_),
        }

    def ul() -> dict[str, Any]:
        try:
            parametric_mod = importlib.import_module("umap.parametric_umap")
            cls = getattr(parametric_mod, "ParametricUMAP")
        except Exception as exc:
            raise RuntimeError(
                f"umap-learn ParametricUMAP unavailable in this environment: {type(exc).__name__}: {exc}"
            ) from exc
        emb = cls(n_neighbors=k, n_components=2, n_epochs=45, random_state=41).fit_transform(x_train)
        return {"train_shape": list(emb.shape), "train_finite": bool(np.all(np.isfinite(emb)))}

    return compare_result(
        "parametric_breast_cancer",
        "Parametric training/inference on breast cancer features.",
        "sklearn.load_breast_cancer",
        rs,
        ul,
        comparability="optional extension comparison; umap-learn requires TensorFlow stack",
    )


def aligned_case() -> Record:
    x, _ = scaled_dataset("digits", 160)
    x0 = x[:90]
    x1 = x[20:110].copy()
    relation_array = np.column_stack([np.arange(70, dtype=np.int64), np.arange(20, 90, dtype=np.int64)])
    relation_dict = {int(left): int(right) for left, right in relation_array}
    k = 8
    epochs = 45

    def rs() -> dict[str, Any]:
        embeddings = AlignedUmap(
            n_neighbors=k,
            n_components=2,
            n_epochs=epochs,
            metric="euclidean",
            random_seed=43,
            init="random",
            alignment_regularization=0.3,
            alignment_epochs=25,
        ).fit_transform([x0, x1], [relation_array])
        gap = np.linalg.norm(embeddings[0][relation_array[:, 0]] - embeddings[1][relation_array[:, 1]], axis=1)
        return {
            "shapes": [list(e.shape) for e in embeddings],
            "finite": bool(all(np.all(np.isfinite(e)) for e in embeddings)),
            "aligned_pair_mean_gap": float(gap.mean()),
        }

    def ul() -> dict[str, Any]:
        if umap is None or not hasattr(umap, "AlignedUMAP"):
            raise RuntimeError("umap-learn AlignedUMAP unavailable")
        embeddings = umap.AlignedUMAP(
            n_neighbors=k,
            n_components=2,
            n_epochs=epochs,
            metric="euclidean",
            random_state=43,
            alignment_regularisation=0.3,
            alignment_window_size=1,
        ).fit_transform([x0, x1], relations=[relation_dict])
        gap = np.linalg.norm(embeddings[0][relation_array[:, 0]] - embeddings[1][relation_array[:, 1]], axis=1)
        return {
            "shapes": [list(np.asarray(e).shape) for e in embeddings],
            "finite": bool(all(np.all(np.isfinite(e)) for e in embeddings)),
            "aligned_pair_mean_gap": float(gap.mean()),
        }

    return compare_result(
        "aligned_digits_batches",
        "Two related digits batches with explicit index relations.",
        "sklearn.load_digits shifted batches",
        rs,
        ul,
        comparability="aligned API semantics differ but relation task is equivalent",
    )


def ecosystem_helpers_case() -> Record:
    x, y = scaled_dataset("wine", None)
    emb = Umap(
        n_neighbors=10,
        n_components=2,
        n_epochs=40,
        random_seed=47,
        init="random",
        ann_mode="exact",
    ).fit_transform(x)

    def rs() -> dict[str, Any]:
        diagnostics = importlib.import_module("umapers.diagnostics")
        report = diagnostics.trustworthiness_report(x, emb, n_neighbors=10)
        plot_status: dict[str, Any]
        try:
            plot = importlib.import_module("umapers.plot")
            ax = plot.points(emb, labels=y)
            plot_status = {"ok": True, "axes_type": type(ax).__name__}
        except Exception as exc:
            plot_status = {
                "ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        return {
            "diagnostics": report,
            "plot_points": plot_status,
        }

    def ul() -> dict[str, Any]:
        try:
            plot_mod = importlib.import_module("umap.plot")
            status = {"ok": True, "module": str(plot_mod)}
        except Exception as exc:
            status = {"ok": False, "error_type": type(exc).__name__, "error": str(exc)}
        return {
            "plot_module": status,
            "note": "umap-learn trustworthiness is usually consumed through sklearn.manifold.trustworthiness, not a UMAP class method.",
        }

    return compare_result(
        "ecosystem_helpers_wine",
        "Optional diagnostics/plotting helpers around a real wine embedding.",
        "sklearn.load_wine",
        rs,
        ul,
        comparability="ecosystem utility comparison",
    )


def unsupported_boundary_case() -> Record:
    x, _ = scaled_dataset("digits", 120)
    x[np.abs(x) < 0.2] = 0.0
    csr = sparse.csr_matrix(x)

    def rs() -> dict[str, Any]:
        model = Umap(
            n_neighbors=8,
            n_components=2,
            n_epochs=35,
            metric="euclidean",
            random_seed=53,
            init="random",
            ann_mode="exact",
        )
        emb = model.fit_transform(csr)
        try:
            model.inverse_transform(emb[:8])
        except Exception as exc:
            return {
                "sparse_fit_shape": list(emb.shape),
                "inverse_error_type": type(exc).__name__,
                "inverse_error": str(exc),
            }
        return {"sparse_fit_shape": list(emb.shape), "inverse_error": ""}

    def ul() -> dict[str, Any]:
        model = make_umap_learn(
            n_neighbors=8,
            n_components=2,
            n_epochs=35,
            metric="euclidean",
            init="random",
        )
        emb = model.fit_transform(csr)
        try:
            inv = model.inverse_transform(emb[:8])
        except Exception as exc:
            return {
                "sparse_fit_shape": list(emb.shape),
                "inverse_error_type": type(exc).__name__,
                "inverse_error": str(exc),
            }
        return {
            "sparse_fit_shape": list(emb.shape),
            "inverse_shape": list(np.asarray(inv).shape),
            "inverse_finite": bool(np.all(np.isfinite(inv))),
        }

    return compare_result(
        "sparse_inverse_boundary",
        "Sparse-trained inverse_transform boundary check.",
        "sklearn.load_digits thresholded to sparse",
        rs,
        ul,
        comparability="boundary behavior",
    )


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Current umap-rs Feature Parity Report",
        "",
        f"- generated_at: `{report['generated_at']}`",
        f"- python: `{report['environment']['python']}`",
        f"- umapers: `{report['environment']['umapers_version']}`",
        f"- umap_learn: `{report['environment']['umap_learn_version']}`",
        "",
        "## Scenario Summary",
        "",
        "| scenario | comparability | umap-rs | umap-learn | key metric |",
        "|---|---|---:|---:|---|",
    ]
    for row in report["scenarios"]:
        rs = row["umap_rs"]
        ul = row["umap_learn"]
        rs_status = "ok" if rs.get("ok") else f"fail: {rs.get('error_type', 'skipped')}"
        ul_status = "ok" if ul.get("ok") else f"fail: {ul.get('error_type', 'skipped')}"
        key_parts = []
        if rs.get("ok") and "trustworthiness" in rs:
            key_parts.append(f"rs trust={rs['trustworthiness']:.3f}")
        if ul.get("ok") and "trustworthiness" in ul:
            key_parts.append(f"learn trust={ul['trustworthiness']:.3f}")
        if rs.get("ok") and "embedding_knn_accuracy" in rs:
            key_parts.append(f"rs acc={rs['embedding_knn_accuracy']:.3f}")
        if ul.get("ok") and "embedding_knn_accuracy" in ul:
            key_parts.append(f"learn acc={ul['embedding_knn_accuracy']:.3f}")
        if rs.get("ok") and "knn_mean_recall" in rs:
            key_parts.append(f"rs ann recall={rs['knn_mean_recall']:.3f}")
        lines.append(
            f"| `{row['scenario']}` | {row['comparability']} | {rs_status} | {ul_status} | {'; '.join(key_parts)} |"
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Timing is sequential side-by-side timing to avoid CPU contention inside one scenario.",
            "- Precomputed-kNN comparison uses each library's native contract, because the accepted self-neighbor convention differs.",
            "- Optional extension failures are reported as environment facts, not counted as core algorithm failures.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Exercise current umap-rs features against umap-learn on real datasets.")
    parser.add_argument("--output-json", type=Path, default=Path("reports/current_feature_parity_report.json"))
    parser.add_argument("--output-md", type=Path, default=Path("reports/current_feature_parity_report.md"))
    parser.add_argument("--skip-warmup", action="store_true")
    args = parser.parse_args()

    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    import platform
    import sys
    import umapers

    report: dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "environment": {
            "python": sys.version.replace("\n", " "),
            "platform": platform.platform(),
            "umapers_version": getattr(umapers, "__version__", "unknown"),
            "umap_learn_version": getattr(umap, "__version__", "unavailable") if umap is not None else "unavailable",
            "umap_learn_import_error": UMAP_IMPORT_ERROR,
        },
        "warmup": {} if args.skip_warmup else warmup(),
        "scenarios": [],
    }

    scenario_fns = [
        dense_fit_transform_case,
        transform_inverse_out_case,
        supervised_case,
        metric_sweep_case,
        densmap_case,
        sparse_case,
        precomputed_knn_case,
        ann_case,
        parametric_case,
        aligned_case,
        ecosystem_helpers_case,
        unsupported_boundary_case,
    ]
    for fn in scenario_fns:
        print(f"running {fn.__name__}", flush=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            report["scenarios"].append(fn())

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(to_jsonable(report), indent=2, sort_keys=True), encoding="utf-8")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    write_markdown(to_jsonable(report), args.output_md)
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_md}")


if __name__ == "__main__":
    main()
