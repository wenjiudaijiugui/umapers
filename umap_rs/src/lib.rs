#![allow(clippy::useless_conversion)]

use numpy::ndarray::{Array2, ArrayView1};
use numpy::PyUntypedArrayMethods;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyReadwriteArray2};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use rust_umap::{
    approximate_knn_recall_diagnostics as approximate_knn_recall_diagnostics_rows,
    approximate_knn_recall_diagnostics_dense, fit_transform as fit_transform_stateless_rows,
    fit_transform_dense as fit_transform_stateless_dense,
    fit_transform_sparse_csr as fit_transform_sparse_csr_stateless, AlignedUmapError,
    AlignedUmapModel, AlignedUmapParams, AlignmentRelation, DenseMatrix, FitTimingBreakdown,
    InitMethod, KnnRecallDiagnostics, Metric, ParametricTrainMode, ParametricUmapModel,
    ParametricUmapParams, ProfiledFit, SparseCsrMatrix, SupervisedTarget, UmapError, UmapModel,
    UmapParams,
};

type PyKnnRows = (Vec<Vec<usize>>, Vec<Vec<f32>>);

fn parse_metric(metric: &str, metric_p: Option<f32>) -> PyResult<Metric> {
    match metric.to_ascii_lowercase().as_str() {
        "euclidean" | "l2" => Ok(Metric::Euclidean),
        "manhattan" | "l1" => Ok(Metric::Manhattan),
        "cosine" => Ok(Metric::Cosine),
        "chebyshev" | "linfinity" | "linf" => Ok(Metric::Chebyshev),
        "minkowski" => {
            let p = metric_p.ok_or_else(|| {
                PyValueError::new_err("metric='minkowski' requires metric_kwds={'p': ...}")
            })?;
            if !p.is_finite() || p <= 0.0 {
                return Err(PyValueError::new_err(
                    "metric_kwds['p'] must be finite and > 0 for minkowski",
                ));
            }
            Ok(Metric::Minkowski { p })
        }
        "correlation" => Ok(Metric::Correlation),
        "canberra" => Ok(Metric::Canberra),
        "braycurtis" | "bray_curtis" => Ok(Metric::BrayCurtis),
        _ => Err(PyValueError::new_err(format!(
            "unsupported metric '{metric}', expected euclidean|manhattan|cosine|chebyshev|minkowski|correlation|canberra|braycurtis"
        ))),
    }
}

fn metric_name(metric: Metric) -> &'static str {
    match metric {
        Metric::Euclidean => "euclidean",
        Metric::Manhattan => "manhattan",
        Metric::Cosine => "cosine",
        Metric::Chebyshev => "chebyshev",
        Metric::Minkowski { .. } => "minkowski",
        Metric::Correlation => "correlation",
        Metric::Canberra => "canberra",
        Metric::BrayCurtis => "braycurtis",
    }
}

fn diagnostics_to_pydict<'py>(
    py: Python<'py>,
    diagnostics: KnnRecallDiagnostics,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new_bound(py);
    out.set_item("n_samples", diagnostics.n_samples)?;
    out.set_item("n_features", diagnostics.n_features)?;
    out.set_item("n_neighbors", diagnostics.n_neighbors)?;
    out.set_item("metric", metric_name(diagnostics.metric))?;
    if let Metric::Minkowski { p } = diagnostics.metric {
        out.set_item("metric_p", p)?;
    } else {
        out.set_item("metric_p", py.None())?;
    }
    out.set_item("candidate_pool", diagnostics.candidate_pool)?;
    out.set_item("n_iters", diagnostics.n_iters)?;
    out.set_item("seed", diagnostics.seed)?;
    out.set_item("mean_recall", diagnostics.mean_recall)?;
    out.set_item("min_recall", diagnostics.min_recall)?;
    out.set_item("worst_decile_recall", diagnostics.worst_decile_recall)?;
    out.set_item(
        "per_row_recall",
        PyArray1::from_vec_bound(py, diagnostics.per_row_recall),
    )?;
    Ok(out)
}

fn fit_timings_to_pydict<'py>(
    py: Python<'py>,
    timings: &FitTimingBreakdown,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new_bound(py);
    out.set_item("validation_sec", timings.validation_sec)?;
    out.set_item("knn_sec", timings.knn_sec)?;
    out.set_item("curve_params_sec", timings.curve_params_sec)?;
    out.set_item("knn_validate_trim_sec", timings.knn_validate_trim_sec)?;
    out.set_item("smooth_knn_sec", timings.smooth_knn_sec)?;
    out.set_item("density_original_sec", timings.density_original_sec)?;
    out.set_item("membership_sec", timings.membership_sec)?;
    out.set_item("symmetrize_sec", timings.symmetrize_sec)?;
    out.set_item("target_intersection_sec", timings.target_intersection_sec)?;
    out.set_item("prune_sec", timings.prune_sec)?;
    out.set_item("init_sec", timings.init_sec)?;
    out.set_item("optimize_sec", timings.optimize_sec)?;
    out.set_item("density_embedding_sec", timings.density_embedding_sec)?;
    out.set_item("output_copy_sec", timings.output_copy_sec)?;
    out.set_item("store_state_sec", timings.store_state_sec)?;
    out.set_item("total_sec", timings.total_sec)?;
    Ok(out)
}

fn profiled_fit_to_pydict<'py>(
    py: Python<'py>,
    profile: ProfiledFit,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new_bound(py);
    out.set_item("embedding", dense_to_numpy(py, profile.embedding)?)?;
    out.set_item("timings", fit_timings_to_pydict(py, &profile.timings)?)?;
    out.set_item("used_approximate_knn", profile.used_approximate_knn)?;
    out.set_item("n_samples", profile.n_samples)?;
    out.set_item("n_features", profile.n_features)?;
    out.set_item("n_epochs", profile.n_epochs)?;
    out.set_item("n_edges", profile.n_edges)?;
    Ok(out)
}

fn parse_init(init: &str) -> PyResult<InitMethod> {
    match init.to_ascii_lowercase().as_str() {
        "random" => Ok(InitMethod::Random),
        "spectral" => Ok(InitMethod::Spectral),
        _ => Err(PyValueError::new_err(format!(
            "unsupported init '{init}', expected random|spectral"
        ))),
    }
}

fn parse_parametric_train_mode(train_mode: &str) -> PyResult<ParametricTrainMode> {
    match train_mode.to_ascii_lowercase().as_str() {
        "naive" => Ok(ParametricTrainMode::Naive),
        "optimized" => Ok(ParametricTrainMode::Optimized),
        _ => Err(PyValueError::new_err(format!(
            "unsupported train_mode '{train_mode}', expected naive|optimized"
        ))),
    }
}

fn map_umap_error(err: UmapError) -> PyErr {
    match err {
        UmapError::NotFitted => PyRuntimeError::new_err(err.to_string()),
        UmapError::EmptyData
        | UmapError::NeedAtLeastTwoSamples
        | UmapError::InconsistentDimensions { .. }
        | UmapError::FeatureMismatch { .. }
        | UmapError::EmbeddingDimensionMismatch { .. }
        | UmapError::InvalidParameter(_) => PyValueError::new_err(err.to_string()),
    }
}

fn map_aligned_error(err: AlignedUmapError) -> PyErr {
    match err {
        AlignedUmapError::Umap(UmapError::NotFitted) => PyRuntimeError::new_err(err.to_string()),
        _ => PyValueError::new_err(err.to_string()),
    }
}

fn ensure_nonzero_columns(name: &str, n_cols: usize) -> PyResult<()> {
    if n_cols == 0 {
        return Err(PyValueError::new_err(format!(
            "{name} must have at least one column"
        )));
    }
    Ok(())
}

fn array2_dims(data: &PyReadonlyArray2<'_, f32>, name: &str) -> PyResult<(usize, usize)> {
    let dims = data.as_array().dim();
    ensure_nonzero_columns(name, dims.1)?;
    Ok(dims)
}

fn array2_f32_to_rows(data: &PyReadonlyArray2<'_, f32>, name: &str) -> PyResult<Vec<Vec<f32>>> {
    let view = data.as_array();
    let (n_rows, n_cols) = view.dim();
    ensure_nonzero_columns(name, n_cols)?;

    if let Ok(slice) = data.as_slice() {
        let mut rows = Vec::with_capacity(n_rows);
        for row in slice.chunks_exact(n_cols) {
            rows.push(row.to_vec());
        }
        return Ok(rows);
    }

    let mut rows = Vec::with_capacity(n_rows);
    for row in view.outer_iter() {
        rows.push(row.to_vec());
    }
    Ok(rows)
}

fn array2_i64_to_usize_rows(
    data: &PyReadonlyArray2<'_, i64>,
    name: &str,
) -> PyResult<Vec<Vec<usize>>> {
    let view = data.as_array();
    let (n_rows, n_cols) = view.dim();
    ensure_nonzero_columns(name, n_cols)?;

    if let Ok(slice) = data.as_slice() {
        let mut rows = Vec::with_capacity(n_rows);
        for row in slice.chunks_exact(n_cols) {
            let mut out_row = Vec::with_capacity(n_cols);
            for &idx in row {
                if idx < 0 {
                    return Err(PyValueError::new_err(
                        "knn indices must be non-negative integers",
                    ));
                }
                out_row.push(idx as usize);
            }
            rows.push(out_row);
        }
        return Ok(rows);
    }

    let mut rows = Vec::with_capacity(n_rows);
    for row in view.outer_iter() {
        let mut out_row = Vec::with_capacity(n_cols);
        for &idx in row {
            if idx < 0 {
                return Err(PyValueError::new_err(
                    "knn indices must be non-negative integers",
                ));
            }
            out_row.push(idx as usize);
        }
        rows.push(out_row);
    }
    Ok(rows)
}

fn i64_slice_to_usize_vec(slice: &[i64], name: &str) -> PyResult<Vec<usize>> {
    let mut out = Vec::with_capacity(slice.len());
    for &value in slice {
        if value < 0 {
            return Err(PyValueError::new_err(format!(
                "{name} must contain non-negative integers"
            )));
        }
        out.push(value as usize);
    }
    Ok(out)
}

fn array1_i64_to_usize_vec(data: &PyReadonlyArray1<'_, i64>, name: &str) -> PyResult<Vec<usize>> {
    let slice = data
        .as_slice()
        .map_err(|_| PyValueError::new_err(format!("{name} must be contiguous")))?;
    i64_slice_to_usize_vec(slice, name)
}

fn array1_i64_to_vec(data: &PyReadonlyArray1<'_, i64>, name: &str) -> PyResult<Vec<i64>> {
    let slice = data
        .as_slice()
        .map_err(|_| PyValueError::new_err(format!("{name} must be contiguous")))?;
    Ok(slice.to_vec())
}

fn relation_from_array(
    data: &PyReadonlyArray2<'_, i64>,
    name: &str,
) -> PyResult<AlignmentRelation> {
    let shape = data.shape();
    if shape.len() != 2 || shape[1] != 2 {
        return Err(PyValueError::new_err(format!(
            "{name} must have shape (n_pairs, 2)"
        )));
    }
    let mut pairs = Vec::with_capacity(shape[0]);
    if let Ok(slice) = data.as_slice() {
        for pair in slice.chunks_exact(2) {
            if pair[0] < 0 || pair[1] < 0 {
                return Err(PyValueError::new_err(format!(
                    "{name} indices must be non-negative"
                )));
            }
            pairs.push((pair[0] as usize, pair[1] as usize));
        }
    } else {
        let arr = data.as_array();
        for row in 0..shape[0] {
            let left = arr[[row, 0]];
            let right = arr[[row, 1]];
            if left < 0 || right < 0 {
                return Err(PyValueError::new_err(format!(
                    "{name} indices must be non-negative"
                )));
            }
            pairs.push((left as usize, right as usize));
        }
    }
    Ok(AlignmentRelation::new(pairs))
}

fn embeddings_to_pylist<'py>(
    py: Python<'py>,
    embeddings: Vec<Vec<Vec<f32>>>,
    n_components: usize,
) -> PyResult<Bound<'py, PyList>> {
    let out = PyList::empty_bound(py);
    for embedding in embeddings {
        out.append(rows_to_numpy(py, embedding, n_components)?)?;
    }
    Ok(out)
}

fn array1_f32_to_vec(data: &PyReadonlyArray1<'_, f32>, name: &str) -> PyResult<Vec<f32>> {
    let slice = data
        .as_slice()
        .map_err(|_| PyValueError::new_err(format!("{name} must be contiguous")))?;
    Ok(slice.to_vec())
}

fn supervised_target_from_py(
    labels: &PyReadonlyArray1<'_, i64>,
    target_weight: f32,
    target_n_neighbors: Option<usize>,
) -> PyResult<SupervisedTarget> {
    let labels = array1_i64_to_vec(labels, "y")?;
    Ok(SupervisedTarget::categorical(
        labels,
        target_weight,
        target_n_neighbors,
    ))
}

fn sparse_csr_from_py(
    indptr: &PyReadonlyArray1<'_, i64>,
    indices: &PyReadonlyArray1<'_, i64>,
    data: &PyReadonlyArray1<'_, f32>,
    n_cols: usize,
) -> PyResult<SparseCsrMatrix> {
    let indptr = array1_i64_to_usize_vec(indptr, "indptr")?;
    let indices = array1_i64_to_usize_vec(indices, "indices")?;
    let values = array1_f32_to_vec(data, "data")?;
    if indptr.is_empty() {
        return Err(PyValueError::new_err("indptr cannot be empty"));
    }
    let n_rows = indptr.len() - 1;
    SparseCsrMatrix::new(n_rows, n_cols, indptr, indices, values).map_err(map_umap_error)
}

fn validate_precomputed_shapes(
    data_n_rows: usize,
    knn_indices: &PyReadonlyArray2<'_, i64>,
    knn_dists: &PyReadonlyArray2<'_, f32>,
    n_neighbors: usize,
) -> PyResult<()> {
    if knn_indices.shape() != knn_dists.shape() {
        return Err(PyValueError::new_err(
            "knn_indices and knn_dists must have identical shapes",
        ));
    }
    if knn_indices.shape()[0] != data_n_rows {
        return Err(PyValueError::new_err(
            "knn row count must match data row count",
        ));
    }
    if knn_indices.shape()[1] < n_neighbors {
        return Err(PyValueError::new_err(format!(
            "knn columns must be >= n_neighbors ({n_neighbors})"
        )));
    }
    Ok(())
}

fn validate_precomputed_distance_values(knn_dists: &PyReadonlyArray2<'_, f32>) -> PyResult<()> {
    for &dist in knn_dists.as_array() {
        if !dist.is_finite() {
            return Err(PyValueError::new_err(
                "knn_dists must contain only finite values",
            ));
        }
        if dist < 0.0 {
            return Err(PyValueError::new_err("knn_dists must be non-negative"));
        }
    }
    Ok(())
}

fn precomputed_knn_rows_from_arrays(
    knn_indices: &PyReadonlyArray2<'_, i64>,
    knn_dists: &PyReadonlyArray2<'_, f32>,
) -> PyResult<PyKnnRows> {
    let knn_idx_rows = array2_i64_to_usize_rows(knn_indices, "knn_indices")?;
    let knn_dist_rows = array2_f32_to_rows(knn_dists, "knn_dists")?;
    Ok((knn_idx_rows, knn_dist_rows))
}

enum F32MatrixInput<'py> {
    Slice {
        values: &'py [f32],
        n_rows: usize,
        n_cols: usize,
    },
    Rows {
        rows: Vec<Vec<f32>>,
    },
}

impl<'py> F32MatrixInput<'py> {
    fn from_py(data: &'py PyReadonlyArray2<'py, f32>, name: &str) -> PyResult<Self> {
        let (n_rows, n_cols) = array2_dims(data, name)?;
        if let Ok(values) = data.as_slice() {
            Ok(Self::Slice {
                values,
                n_rows,
                n_cols,
            })
        } else {
            Ok(Self::Rows {
                rows: array2_f32_to_rows(data, name)?,
            })
        }
    }
}

fn rows_to_numpy<'py>(
    py: Python<'py>,
    rows: Vec<Vec<f32>>,
    empty_cols: usize,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let n_rows = rows.len();
    let n_cols = if n_rows == 0 {
        empty_cols
    } else {
        rows[0].len()
    };

    let mut flat = Vec::with_capacity(n_rows.saturating_mul(n_cols));
    for row in rows {
        if row.len() != n_cols {
            return Err(PyRuntimeError::new_err(
                "inconsistent output row width from rust_umap core",
            ));
        }
        flat.extend_from_slice(&row);
    }

    let arr = Array2::from_shape_vec((n_rows, n_cols), flat)
        .map_err(|err| PyRuntimeError::new_err(format!("failed to build output array: {err}")))?;
    Ok(PyArray2::from_owned_array_bound(py, arr))
}

fn dense_to_numpy<'py>(py: Python<'py>, dense: DenseMatrix) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let (flat, n_rows, n_cols) = dense.into_raw_parts();
    let arr = Array2::from_shape_vec((n_rows, n_cols), flat)
        .map_err(|err| PyRuntimeError::new_err(format!("failed to build output array: {err}")))?;
    Ok(PyArray2::from_owned_array_bound(py, arr))
}

fn copy_rows_into_out(
    rows: &[Vec<f32>],
    empty_cols: usize,
    mut out: PyReadwriteArray2<'_, f32>,
) -> PyResult<()> {
    let n_rows = rows.len();
    let n_cols = if n_rows == 0 {
        empty_cols
    } else {
        rows[0].len()
    };

    let mut out_view = out.as_array_mut();
    if out_view.nrows() != n_rows || out_view.ncols() != n_cols {
        return Err(PyValueError::new_err(format!(
            "output buffer shape mismatch: expected ({n_rows}, {n_cols}), got ({}, {})",
            out_view.nrows(),
            out_view.ncols()
        )));
    }

    if let Some(out_slice) = out_view.as_slice_memory_order_mut() {
        for (i, row) in rows.iter().enumerate() {
            if row.len() != n_cols {
                return Err(PyRuntimeError::new_err(
                    "inconsistent output row width from rust_umap core",
                ));
            }
            let start = i * n_cols;
            let end = start + n_cols;
            out_slice[start..end].copy_from_slice(row);
        }
        return Ok(());
    }

    for (i, row) in rows.iter().enumerate() {
        if row.len() != n_cols {
            return Err(PyRuntimeError::new_err(
                "inconsistent output row width from rust_umap core",
            ));
        }
        out_view
            .row_mut(i)
            .assign(&ArrayView1::from(row.as_slice()));
    }

    Ok(())
}

fn copy_dense_into_out(dense: &DenseMatrix, mut out: PyReadwriteArray2<'_, f32>) -> PyResult<()> {
    let n_rows = dense.n_rows();
    let n_cols = dense.n_cols();

    let mut out_view = out.as_array_mut();
    if out_view.nrows() != n_rows || out_view.ncols() != n_cols {
        return Err(PyValueError::new_err(format!(
            "output buffer shape mismatch: expected ({n_rows}, {n_cols}), got ({}, {})",
            out_view.nrows(),
            out_view.ncols()
        )));
    }

    if let Some(out_slice) = out_view.as_slice_memory_order_mut() {
        out_slice.copy_from_slice(dense.as_slice());
        return Ok(());
    }

    for i in 0..n_rows {
        let start = i * n_cols;
        let end = start + n_cols;
        out_view
            .row_mut(i)
            .assign(&ArrayView1::from(&dense.as_slice()[start..end]));
    }

    Ok(())
}

#[pyclass(name = "UmapCore", module = "umapers._umapers")]
struct PyUmapCore {
    inner: UmapModel,
}

#[allow(clippy::too_many_arguments)]
#[pymethods]
impl PyUmapCore {
    #[new]
    #[pyo3(signature = (
        n_neighbors = 15,
        n_components = 2,
        n_epochs = None,
        metric = "euclidean",
        metric_p = None,
        learning_rate = 1.0,
        min_dist = 0.1,
        spread = 1.0,
        local_connectivity = 1.0,
        set_op_mix_ratio = 1.0,
        repulsion_strength = 1.0,
        negative_sample_rate = 5,
        random_seed = 42,
        init = "spectral",
        use_approximate_knn = true,
        approx_knn_candidates = 50,
        approx_knn_iters = 14,
        approx_knn_threshold = 4096,
        densmap = false,
        dens_lambda = 2.0,
        dens_frac = 0.3,
        dens_var_shift = 0.1,
        output_dens = false,
    ))]
    fn new(
        n_neighbors: usize,
        n_components: usize,
        n_epochs: Option<usize>,
        metric: &str,
        metric_p: Option<f32>,
        learning_rate: f32,
        min_dist: f32,
        spread: f32,
        local_connectivity: f32,
        set_op_mix_ratio: f32,
        repulsion_strength: f32,
        negative_sample_rate: usize,
        random_seed: u64,
        init: &str,
        use_approximate_knn: bool,
        approx_knn_candidates: usize,
        approx_knn_iters: usize,
        approx_knn_threshold: usize,
        densmap: bool,
        dens_lambda: f32,
        dens_frac: f32,
        dens_var_shift: f32,
        output_dens: bool,
    ) -> PyResult<Self> {
        let metric = parse_metric(metric, metric_p)?;
        let init = parse_init(init)?;

        let params = UmapParams {
            n_neighbors,
            n_components,
            n_epochs,
            metric,
            learning_rate,
            min_dist,
            spread,
            local_connectivity,
            set_op_mix_ratio,
            repulsion_strength,
            negative_sample_rate,
            random_seed,
            init,
            use_approximate_knn,
            approx_knn_candidates,
            approx_knn_iters,
            approx_knn_threshold,
            densmap,
            dens_lambda,
            dens_frac,
            dens_var_shift,
            output_dens,
        };

        Ok(Self {
            inner: UmapModel::new(params),
        })
    }

    fn fit(&mut self, py: Python<'_>, data: PyReadonlyArray2<'_, f32>) -> PyResult<()> {
        match F32MatrixInput::from_py(&data, "data")? {
            F32MatrixInput::Slice {
                values,
                n_rows,
                n_cols,
            } => py
                .allow_threads(|| self.inner.fit_dense(values, n_rows, n_cols))
                .map_err(map_umap_error),
            F32MatrixInput::Rows { rows, .. } => py
                .allow_threads(|| self.inner.fit(&rows))
                .map_err(map_umap_error),
        }
    }

    #[pyo3(signature = (data, y, target_weight, target_n_neighbors = None))]
    fn fit_supervised(
        &mut self,
        py: Python<'_>,
        data: PyReadonlyArray2<'_, f32>,
        y: PyReadonlyArray1<'_, i64>,
        target_weight: f32,
        target_n_neighbors: Option<usize>,
    ) -> PyResult<()> {
        let target = supervised_target_from_py(&y, target_weight, target_n_neighbors)?;
        match F32MatrixInput::from_py(&data, "data")? {
            F32MatrixInput::Slice {
                values,
                n_rows,
                n_cols,
            } => py
                .allow_threads(|| {
                    self.inner
                        .fit_dense_supervised(values, n_rows, n_cols, &target)
                })
                .map_err(map_umap_error),
            F32MatrixInput::Rows { rows, .. } => py
                .allow_threads(|| self.inner.fit_supervised(&rows, &target))
                .map_err(map_umap_error),
        }
    }

    fn fit_sparse_csr(
        &mut self,
        py: Python<'_>,
        indptr: PyReadonlyArray1<'_, i64>,
        indices: PyReadonlyArray1<'_, i64>,
        data: PyReadonlyArray1<'_, f32>,
        n_cols: usize,
    ) -> PyResult<()> {
        let csr = sparse_csr_from_py(&indptr, &indices, &data, n_cols)?;
        py.allow_threads(|| self.inner.fit_sparse_csr(csr))
            .map_err(map_umap_error)
    }

    fn fit_transform<'py>(
        &mut self,
        py: Python<'py>,
        data: PyReadonlyArray2<'py, f32>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        match F32MatrixInput::from_py(&data, "data")? {
            F32MatrixInput::Slice {
                values,
                n_rows,
                n_cols,
            } => {
                let embedding = py
                    .allow_threads(|| self.inner.fit_transform_dense(values, n_rows, n_cols))
                    .map_err(map_umap_error)?;
                dense_to_numpy(py, embedding)
            }
            F32MatrixInput::Rows { rows, .. } => {
                let embedding = py
                    .allow_threads(|| self.inner.fit_transform(&rows))
                    .map_err(map_umap_error)?;
                rows_to_numpy(py, embedding, self.inner.params().n_components)
            }
        }
    }

    fn profile_fit_transform<'py>(
        &mut self,
        py: Python<'py>,
        data: PyReadonlyArray2<'py, f32>,
    ) -> PyResult<Bound<'py, PyDict>> {
        match F32MatrixInput::from_py(&data, "data")? {
            F32MatrixInput::Slice {
                values,
                n_rows,
                n_cols,
            } => {
                let profile = py
                    .allow_threads(|| {
                        self.inner
                            .profile_fit_transform_dense(values, n_rows, n_cols)
                    })
                    .map_err(map_umap_error)?;
                profiled_fit_to_pydict(py, profile)
            }
            F32MatrixInput::Rows { rows } => {
                let n_rows = rows.len();
                let n_cols = rows.first().map_or(0, |row| row.len());
                let dense = DenseMatrix::from_rows(&rows).map_err(map_umap_error)?;
                let profile = py
                    .allow_threads(|| {
                        self.inner
                            .profile_fit_transform_dense(dense.as_slice(), n_rows, n_cols)
                    })
                    .map_err(map_umap_error)?;
                profiled_fit_to_pydict(py, profile)
            }
        }
    }

    #[pyo3(signature = (data, y, target_weight, target_n_neighbors = None))]
    fn fit_transform_supervised<'py>(
        &mut self,
        py: Python<'py>,
        data: PyReadonlyArray2<'py, f32>,
        y: PyReadonlyArray1<'py, i64>,
        target_weight: f32,
        target_n_neighbors: Option<usize>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let target = supervised_target_from_py(&y, target_weight, target_n_neighbors)?;
        match F32MatrixInput::from_py(&data, "data")? {
            F32MatrixInput::Slice {
                values,
                n_rows,
                n_cols,
            } => {
                let embedding = py
                    .allow_threads(|| {
                        self.inner
                            .fit_transform_dense_supervised(values, n_rows, n_cols, &target)
                    })
                    .map_err(map_umap_error)?;
                dense_to_numpy(py, embedding)
            }
            F32MatrixInput::Rows { rows, .. } => {
                let embedding = py
                    .allow_threads(|| self.inner.fit_transform_supervised(&rows, &target))
                    .map_err(map_umap_error)?;
                rows_to_numpy(py, embedding, self.inner.params().n_components)
            }
        }
    }

    fn fit_transform_stateless<'py>(
        &self,
        py: Python<'py>,
        data: PyReadonlyArray2<'py, f32>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let n_components = self.inner.params().n_components;
        match F32MatrixInput::from_py(&data, "data")? {
            F32MatrixInput::Slice {
                values,
                n_rows,
                n_cols,
            } => {
                let params = self.inner.params().clone();
                let embedding = py
                    .allow_threads(move || {
                        fit_transform_stateless_dense(values, n_rows, n_cols, params)
                    })
                    .map_err(map_umap_error)?;
                dense_to_numpy(py, embedding)
            }
            F32MatrixInput::Rows { rows, .. } => {
                let params = self.inner.params().clone();
                let embedding = py
                    .allow_threads(move || fit_transform_stateless_rows(&rows, params))
                    .map_err(map_umap_error)?;
                rows_to_numpy(py, embedding, n_components)
            }
        }
    }

    fn fit_transform_into<'py>(
        &mut self,
        py: Python<'py>,
        data: PyReadonlyArray2<'py, f32>,
        out: PyReadwriteArray2<'py, f32>,
    ) -> PyResult<()> {
        match F32MatrixInput::from_py(&data, "data")? {
            F32MatrixInput::Slice {
                values,
                n_rows,
                n_cols,
            } => {
                let embedding = py
                    .allow_threads(|| self.inner.fit_transform_dense(values, n_rows, n_cols))
                    .map_err(map_umap_error)?;
                copy_dense_into_out(&embedding, out)
            }
            F32MatrixInput::Rows { rows, .. } => {
                let embedding = py
                    .allow_threads(|| self.inner.fit_transform(&rows))
                    .map_err(map_umap_error)?;
                copy_rows_into_out(&embedding, self.inner.params().n_components, out)
            }
        }
    }

    #[pyo3(signature = (data, y, out, target_weight, target_n_neighbors = None))]
    fn fit_transform_supervised_into<'py>(
        &mut self,
        py: Python<'py>,
        data: PyReadonlyArray2<'py, f32>,
        y: PyReadonlyArray1<'py, i64>,
        out: PyReadwriteArray2<'py, f32>,
        target_weight: f32,
        target_n_neighbors: Option<usize>,
    ) -> PyResult<()> {
        let target = supervised_target_from_py(&y, target_weight, target_n_neighbors)?;
        match F32MatrixInput::from_py(&data, "data")? {
            F32MatrixInput::Slice {
                values,
                n_rows,
                n_cols,
            } => {
                let embedding = py
                    .allow_threads(|| {
                        self.inner
                            .fit_transform_dense_supervised(values, n_rows, n_cols, &target)
                    })
                    .map_err(map_umap_error)?;
                copy_dense_into_out(&embedding, out)
            }
            F32MatrixInput::Rows { rows, .. } => {
                let embedding = py
                    .allow_threads(|| self.inner.fit_transform_supervised(&rows, &target))
                    .map_err(map_umap_error)?;
                copy_rows_into_out(&embedding, self.inner.params().n_components, out)
            }
        }
    }

    fn fit_transform_sparse_csr<'py>(
        &mut self,
        py: Python<'py>,
        indptr: PyReadonlyArray1<'py, i64>,
        indices: PyReadonlyArray1<'py, i64>,
        data: PyReadonlyArray1<'py, f32>,
        n_cols: usize,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let csr = sparse_csr_from_py(&indptr, &indices, &data, n_cols)?;
        let embedding = py
            .allow_threads(|| self.inner.fit_transform_sparse_csr(csr))
            .map_err(map_umap_error)?;
        rows_to_numpy(py, embedding, self.inner.params().n_components)
    }

    fn fit_transform_sparse_csr_stateless<'py>(
        &self,
        py: Python<'py>,
        indptr: PyReadonlyArray1<'py, i64>,
        indices: PyReadonlyArray1<'py, i64>,
        data: PyReadonlyArray1<'py, f32>,
        n_cols: usize,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let csr = sparse_csr_from_py(&indptr, &indices, &data, n_cols)?;
        let params = self.inner.params().clone();
        let embedding = py
            .allow_threads(move || fit_transform_sparse_csr_stateless(csr, params))
            .map_err(map_umap_error)?;
        rows_to_numpy(py, embedding, self.inner.params().n_components)
    }

    fn fit_transform_sparse_csr_into<'py>(
        &mut self,
        py: Python<'py>,
        indptr: PyReadonlyArray1<'py, i64>,
        indices: PyReadonlyArray1<'py, i64>,
        data: PyReadonlyArray1<'py, f32>,
        n_cols: usize,
        out: PyReadwriteArray2<'py, f32>,
    ) -> PyResult<()> {
        let csr = sparse_csr_from_py(&indptr, &indices, &data, n_cols)?;
        let embedding = py
            .allow_threads(|| self.inner.fit_transform_sparse_csr(csr))
            .map_err(map_umap_error)?;
        copy_rows_into_out(&embedding, self.inner.params().n_components, out)
    }

    #[pyo3(signature = (
        data,
        knn_indices,
        knn_dists,
        knn_metric = "euclidean",
        validate_precomputed = true
    ))]
    fn fit_transform_with_knn<'py>(
        &mut self,
        py: Python<'py>,
        data: PyReadonlyArray2<'py, f32>,
        knn_indices: PyReadonlyArray2<'py, i64>,
        knn_dists: PyReadonlyArray2<'py, f32>,
        knn_metric: &str,
        validate_precomputed: bool,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let data = F32MatrixInput::from_py(&data, "data")?;
        let knn_metric = parse_metric(knn_metric, None)?;

        match data {
            F32MatrixInput::Slice {
                values,
                n_rows,
                n_cols,
            } => {
                if let (Ok(knn_idx_slice), Ok(knn_dist_slice)) =
                    (knn_indices.as_slice(), knn_dists.as_slice())
                {
                    let knn_idx_shape = knn_indices.shape();
                    let knn_dist_shape = knn_dists.shape();
                    let embedding = py
                        .allow_threads(|| {
                            self.inner.fit_transform_with_knn_metric_dense_i64_flat(
                                values,
                                n_rows,
                                n_cols,
                                knn_idx_slice,
                                knn_idx_shape[0],
                                knn_idx_shape[1],
                                knn_dist_slice,
                                knn_dist_shape[0],
                                knn_dist_shape[1],
                                knn_metric,
                                validate_precomputed,
                            )
                        })
                        .map_err(map_umap_error)?;
                    dense_to_numpy(py, embedding)
                } else {
                    validate_precomputed_shapes(
                        n_rows,
                        &knn_indices,
                        &knn_dists,
                        self.inner.params().n_neighbors,
                    )?;
                    if validate_precomputed {
                        validate_precomputed_distance_values(&knn_dists)?;
                    }
                    let (knn_idx_rows, knn_dist_rows) =
                        precomputed_knn_rows_from_arrays(&knn_indices, &knn_dists)?;
                    let embedding = py
                        .allow_threads(|| {
                            self.inner.fit_transform_with_knn_metric_dense(
                                values,
                                n_rows,
                                n_cols,
                                &knn_idx_rows,
                                &knn_dist_rows,
                                knn_metric,
                            )
                        })
                        .map_err(map_umap_error)?;
                    dense_to_numpy(py, embedding)
                }
            }
            F32MatrixInput::Rows { rows, .. } => {
                validate_precomputed_shapes(
                    rows.len(),
                    &knn_indices,
                    &knn_dists,
                    self.inner.params().n_neighbors,
                )?;
                if validate_precomputed {
                    validate_precomputed_distance_values(&knn_dists)?;
                }
                let (knn_idx_rows, knn_dist_rows) =
                    precomputed_knn_rows_from_arrays(&knn_indices, &knn_dists)?;
                let embedding = py
                    .allow_threads(|| {
                        self.inner.fit_transform_with_knn_metric(
                            &rows,
                            &knn_idx_rows,
                            &knn_dist_rows,
                            knn_metric,
                        )
                    })
                    .map_err(map_umap_error)?;

                rows_to_numpy(py, embedding, self.inner.params().n_components)
            }
        }
    }

    #[pyo3(signature = (
        data,
        knn_indices,
        knn_dists,
        out,
        knn_metric = "euclidean",
        validate_precomputed = true
    ))]
    fn fit_transform_with_knn_into<'py>(
        &mut self,
        py: Python<'py>,
        data: PyReadonlyArray2<'py, f32>,
        knn_indices: PyReadonlyArray2<'py, i64>,
        knn_dists: PyReadonlyArray2<'py, f32>,
        out: PyReadwriteArray2<'py, f32>,
        knn_metric: &str,
        validate_precomputed: bool,
    ) -> PyResult<()> {
        let data = F32MatrixInput::from_py(&data, "data")?;
        let knn_metric = parse_metric(knn_metric, None)?;

        match data {
            F32MatrixInput::Slice {
                values,
                n_rows,
                n_cols,
            } => {
                if let (Ok(knn_idx_slice), Ok(knn_dist_slice)) =
                    (knn_indices.as_slice(), knn_dists.as_slice())
                {
                    let knn_idx_shape = knn_indices.shape();
                    let knn_dist_shape = knn_dists.shape();
                    let embedding = py
                        .allow_threads(|| {
                            self.inner.fit_transform_with_knn_metric_dense_i64_flat(
                                values,
                                n_rows,
                                n_cols,
                                knn_idx_slice,
                                knn_idx_shape[0],
                                knn_idx_shape[1],
                                knn_dist_slice,
                                knn_dist_shape[0],
                                knn_dist_shape[1],
                                knn_metric,
                                validate_precomputed,
                            )
                        })
                        .map_err(map_umap_error)?;
                    copy_dense_into_out(&embedding, out)
                } else {
                    validate_precomputed_shapes(
                        n_rows,
                        &knn_indices,
                        &knn_dists,
                        self.inner.params().n_neighbors,
                    )?;
                    if validate_precomputed {
                        validate_precomputed_distance_values(&knn_dists)?;
                    }
                    let (knn_idx_rows, knn_dist_rows) =
                        precomputed_knn_rows_from_arrays(&knn_indices, &knn_dists)?;
                    let embedding = py
                        .allow_threads(|| {
                            self.inner.fit_transform_with_knn_metric_dense(
                                values,
                                n_rows,
                                n_cols,
                                &knn_idx_rows,
                                &knn_dist_rows,
                                knn_metric,
                            )
                        })
                        .map_err(map_umap_error)?;
                    copy_dense_into_out(&embedding, out)
                }
            }
            F32MatrixInput::Rows { rows, .. } => {
                validate_precomputed_shapes(
                    rows.len(),
                    &knn_indices,
                    &knn_dists,
                    self.inner.params().n_neighbors,
                )?;
                if validate_precomputed {
                    validate_precomputed_distance_values(&knn_dists)?;
                }
                let (knn_idx_rows, knn_dist_rows) =
                    precomputed_knn_rows_from_arrays(&knn_indices, &knn_dists)?;
                let embedding = py
                    .allow_threads(|| {
                        self.inner.fit_transform_with_knn_metric(
                            &rows,
                            &knn_idx_rows,
                            &knn_dist_rows,
                            knn_metric,
                        )
                    })
                    .map_err(map_umap_error)?;

                copy_rows_into_out(&embedding, self.inner.params().n_components, out)
            }
        }
    }

    fn knn_recall_diagnostics<'py>(
        &self,
        py: Python<'py>,
        data: PyReadonlyArray2<'py, f32>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let params = self.inner.params().clone();
        let diagnostics = match F32MatrixInput::from_py(&data, "data")? {
            F32MatrixInput::Slice {
                values,
                n_rows,
                n_cols,
            } => py
                .allow_threads(|| {
                    approximate_knn_recall_diagnostics_dense(values, n_rows, n_cols, params)
                })
                .map_err(map_umap_error)?,
            F32MatrixInput::Rows { rows, .. } => py
                .allow_threads(|| approximate_knn_recall_diagnostics_rows(&rows, params))
                .map_err(map_umap_error)?,
        };
        diagnostics_to_pydict(py, diagnostics)
    }

    fn transform<'py>(
        &self,
        py: Python<'py>,
        query: PyReadonlyArray2<'py, f32>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        match F32MatrixInput::from_py(&query, "query")? {
            F32MatrixInput::Slice {
                values,
                n_rows,
                n_cols,
            } => {
                let out = py
                    .allow_threads(|| self.inner.transform_dense(values, n_rows, n_cols))
                    .map_err(map_umap_error)?;
                dense_to_numpy(py, out)
            }
            F32MatrixInput::Rows { rows, .. } => {
                let out = py
                    .allow_threads(|| self.inner.transform(&rows))
                    .map_err(map_umap_error)?;
                rows_to_numpy(py, out, self.inner.params().n_components)
            }
        }
    }

    fn transform_into<'py>(
        &self,
        py: Python<'py>,
        query: PyReadonlyArray2<'py, f32>,
        out: PyReadwriteArray2<'py, f32>,
    ) -> PyResult<()> {
        match F32MatrixInput::from_py(&query, "query")? {
            F32MatrixInput::Slice {
                values,
                n_rows,
                n_cols,
            } => {
                let transformed = py
                    .allow_threads(|| self.inner.transform_dense(values, n_rows, n_cols))
                    .map_err(map_umap_error)?;
                copy_dense_into_out(&transformed, out)
            }
            F32MatrixInput::Rows { rows, .. } => {
                let transformed = py
                    .allow_threads(|| self.inner.transform(&rows))
                    .map_err(map_umap_error)?;
                copy_rows_into_out(&transformed, self.inner.params().n_components, out)
            }
        }
    }

    fn transform_sparse_csr<'py>(
        &self,
        py: Python<'py>,
        indptr: PyReadonlyArray1<'py, i64>,
        indices: PyReadonlyArray1<'py, i64>,
        data: PyReadonlyArray1<'py, f32>,
        n_cols: usize,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let csr = sparse_csr_from_py(&indptr, &indices, &data, n_cols)?;
        let transformed = py
            .allow_threads(|| self.inner.transform_sparse_csr(csr))
            .map_err(map_umap_error)?;
        rows_to_numpy(py, transformed, self.inner.params().n_components)
    }

    fn transform_sparse_csr_into<'py>(
        &self,
        py: Python<'py>,
        indptr: PyReadonlyArray1<'py, i64>,
        indices: PyReadonlyArray1<'py, i64>,
        data: PyReadonlyArray1<'py, f32>,
        n_cols: usize,
        out: PyReadwriteArray2<'py, f32>,
    ) -> PyResult<()> {
        let csr = sparse_csr_from_py(&indptr, &indices, &data, n_cols)?;
        let transformed = py
            .allow_threads(|| self.inner.transform_sparse_csr(csr))
            .map_err(map_umap_error)?;
        copy_rows_into_out(&transformed, self.inner.params().n_components, out)
    }

    fn inverse_transform<'py>(
        &self,
        py: Python<'py>,
        embedded_query: PyReadonlyArray2<'py, f32>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        match F32MatrixInput::from_py(&embedded_query, "embedded_query")? {
            F32MatrixInput::Slice {
                values,
                n_rows,
                n_cols,
            } => {
                let out = py
                    .allow_threads(|| self.inner.inverse_transform_dense(values, n_rows, n_cols))
                    .map_err(map_umap_error)?;
                dense_to_numpy(py, out)
            }
            F32MatrixInput::Rows { rows, .. } => {
                let out = py
                    .allow_threads(|| self.inner.inverse_transform(&rows))
                    .map_err(map_umap_error)?;
                rows_to_numpy(py, out, self.inner.n_features().unwrap_or(0))
            }
        }
    }

    fn inverse_transform_into<'py>(
        &self,
        py: Python<'py>,
        embedded_query: PyReadonlyArray2<'py, f32>,
        out: PyReadwriteArray2<'py, f32>,
    ) -> PyResult<()> {
        match F32MatrixInput::from_py(&embedded_query, "embedded_query")? {
            F32MatrixInput::Slice {
                values,
                n_rows,
                n_cols,
            } => {
                let reconstructed = py
                    .allow_threads(|| self.inner.inverse_transform_dense(values, n_rows, n_cols))
                    .map_err(map_umap_error)?;
                copy_dense_into_out(&reconstructed, out)
            }
            F32MatrixInput::Rows { rows, .. } => {
                let reconstructed = py
                    .allow_threads(|| self.inner.inverse_transform(&rows))
                    .map_err(map_umap_error)?;
                copy_rows_into_out(&reconstructed, self.inner.n_features().unwrap_or(0), out)
            }
        }
    }

    #[getter]
    fn n_components(&self) -> usize {
        self.inner.params().n_components
    }

    #[getter]
    fn n_neighbors(&self) -> usize {
        self.inner.params().n_neighbors
    }

    #[getter]
    fn n_features(&self) -> Option<usize> {
        self.inner.n_features()
    }

    #[getter]
    fn embedding<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyArray2<f32>>>> {
        match self.inner.embedding() {
            Some(embedding) => {
                rows_to_numpy(py, embedding.to_vec(), self.inner.params().n_components).map(Some)
            }
            None => Ok(None),
        }
    }

    #[getter]
    fn radii_original<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyArray1<f32>>>> {
        match self.inner.radii_original() {
            Some(radii) => Ok(Some(PyArray1::from_slice_bound(py, radii))),
            None => Ok(None),
        }
    }

    #[getter]
    fn radii_embedding<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyArray1<f32>>>> {
        match self.inner.radii_embedding() {
            Some(radii) => Ok(Some(PyArray1::from_slice_bound(py, radii))),
            None => Ok(None),
        }
    }
}

#[pyclass(name = "ParametricUmapCore", module = "umapers._umapers")]
struct PyParametricUmapCore {
    inner: ParametricUmapModel,
}

#[allow(clippy::too_many_arguments)]
#[pymethods]
impl PyParametricUmapCore {
    #[new]
    #[pyo3(signature = (
        n_neighbors = 15,
        n_components = 2,
        n_epochs = None,
        metric = "euclidean",
        metric_p = None,
        hidden_dim = 64,
        train_epochs = 120,
        batch_size = 128,
        inference_batch_size = 1024,
        learning_rate = 0.01,
        weight_decay = 0.0001,
        pairwise_loss_weight = 0.0,
        pairwise_pairs_per_batch = 32,
        standardize_input = true,
        random_seed = 42,
        train_mode = "optimized",
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        n_neighbors: usize,
        n_components: usize,
        n_epochs: Option<usize>,
        metric: &str,
        metric_p: Option<f32>,
        hidden_dim: usize,
        train_epochs: usize,
        batch_size: usize,
        inference_batch_size: usize,
        learning_rate: f32,
        weight_decay: f32,
        pairwise_loss_weight: f32,
        pairwise_pairs_per_batch: usize,
        standardize_input: bool,
        random_seed: u64,
        train_mode: &str,
    ) -> PyResult<Self> {
        let metric = parse_metric(metric, metric_p)?;
        let train_mode = parse_parametric_train_mode(train_mode)?;
        let umap_params = UmapParams {
            n_neighbors,
            n_components,
            n_epochs,
            metric,
            random_seed,
            use_approximate_knn: false,
            ..UmapParams::default()
        };
        let params = ParametricUmapParams {
            umap_params,
            hidden_dim,
            train_epochs,
            batch_size,
            inference_batch_size,
            learning_rate,
            weight_decay,
            pairwise_loss_weight,
            pairwise_pairs_per_batch,
            standardize_input,
            seed: random_seed,
            train_mode,
        };
        Ok(Self {
            inner: ParametricUmapModel::new(params),
        })
    }

    fn fit(&mut self, py: Python<'_>, data: PyReadonlyArray2<'_, f32>) -> PyResult<()> {
        let rows = array2_f32_to_rows(&data, "data")?;
        py.allow_threads(|| self.inner.fit(&rows))
            .map_err(map_umap_error)
    }

    fn fit_transform<'py>(
        &mut self,
        py: Python<'py>,
        data: PyReadonlyArray2<'py, f32>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let rows = array2_f32_to_rows(&data, "data")?;
        let embedding = py
            .allow_threads(|| self.inner.fit_transform(&rows))
            .map_err(map_umap_error)?;
        rows_to_numpy(py, embedding, self.inner.params().umap_params.n_components)
    }

    fn transform<'py>(
        &self,
        py: Python<'py>,
        query: PyReadonlyArray2<'py, f32>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let rows = array2_f32_to_rows(&query, "query")?;
        let embedding = py
            .allow_threads(|| self.inner.transform(&rows))
            .map_err(map_umap_error)?;
        rows_to_numpy(py, embedding, self.inner.params().umap_params.n_components)
    }

    #[getter]
    fn teacher_embedding<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Option<Bound<'py, PyArray2<f32>>>> {
        match self.inner.teacher_embedding() {
            Some(embedding) => rows_to_numpy(
                py,
                embedding.to_vec(),
                self.inner.params().umap_params.n_components,
            )
            .map(Some),
            None => Ok(None),
        }
    }

    #[getter]
    fn n_features(&self) -> Option<usize> {
        self.inner.n_features()
    }
}

#[pyclass(name = "AlignedUmapCore", module = "umapers._umapers")]
struct PyAlignedUmapCore {
    inner: AlignedUmapModel,
}

#[allow(clippy::too_many_arguments)]
#[pymethods]
impl PyAlignedUmapCore {
    #[new]
    #[pyo3(signature = (
        n_neighbors = 15,
        n_components = 2,
        n_epochs = None,
        metric = "euclidean",
        metric_p = None,
        random_seed = 42,
        init = "spectral",
        alignment_regularization = 0.08,
        alignment_learning_rate = 0.25,
        alignment_epochs = None,
        recenter_interval = 5,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        n_neighbors: usize,
        n_components: usize,
        n_epochs: Option<usize>,
        metric: &str,
        metric_p: Option<f32>,
        random_seed: u64,
        init: &str,
        alignment_regularization: f32,
        alignment_learning_rate: f32,
        alignment_epochs: Option<usize>,
        recenter_interval: usize,
    ) -> PyResult<Self> {
        let metric = parse_metric(metric, metric_p)?;
        let init = parse_init(init)?;
        let umap = UmapParams {
            n_neighbors,
            n_components,
            n_epochs,
            metric,
            random_seed,
            init,
            use_approximate_knn: false,
            ..UmapParams::default()
        };
        let params = AlignedUmapParams {
            umap,
            alignment_regularization,
            alignment_learning_rate,
            alignment_epochs,
            recenter_interval,
        };
        Ok(Self {
            inner: AlignedUmapModel::new(params),
        })
    }

    fn fit_transform_identity<'py>(
        &mut self,
        py: Python<'py>,
        datasets: Vec<PyReadonlyArray2<'py, f32>>,
    ) -> PyResult<Bound<'py, PyList>> {
        let slices = datasets
            .iter()
            .enumerate()
            .map(|(idx, data)| array2_f32_to_rows(data, &format!("datasets[{idx}]")))
            .collect::<PyResult<Vec<Vec<Vec<f32>>>>>()?;
        let n_components = self.inner.params().umap.n_components;
        let embeddings = py
            .allow_threads(|| self.inner.fit_transform_identity(&slices))
            .map_err(map_aligned_error)?;
        embeddings_to_pylist(py, embeddings, n_components)
    }

    fn fit_transform<'py>(
        &mut self,
        py: Python<'py>,
        datasets: Vec<PyReadonlyArray2<'py, f32>>,
        relations: Vec<PyReadonlyArray2<'py, i64>>,
    ) -> PyResult<Bound<'py, PyList>> {
        let slices = datasets
            .iter()
            .enumerate()
            .map(|(idx, data)| array2_f32_to_rows(data, &format!("datasets[{idx}]")))
            .collect::<PyResult<Vec<Vec<Vec<f32>>>>>()?;
        let relations = relations
            .iter()
            .enumerate()
            .map(|(idx, relation)| relation_from_array(relation, &format!("relations[{idx}]")))
            .collect::<PyResult<Vec<AlignmentRelation>>>()?;
        let n_components = self.inner.params().umap.n_components;
        let embeddings = py
            .allow_threads(|| self.inner.fit_transform(&slices, &relations))
            .map_err(map_aligned_error)?;
        embeddings_to_pylist(py, embeddings, n_components)
    }
}

#[pymodule]
fn _umapers(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyUmapCore>()?;
    m.add_class::<PyParametricUmapCore>()?;
    m.add_class::<PyAlignedUmapCore>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
