use numpy::ndarray::{Array2, ArrayView1};
use numpy::PyUntypedArrayMethods;
use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyReadwriteArray2};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use rust_umap::{
    fit_transform as fit_transform_stateless_rows,
    fit_transform_dense as fit_transform_stateless_dense,
    fit_transform_sparse_csr as fit_transform_sparse_csr_stateless, DenseMatrix, InitMethod,
    Metric, SparseCsrMatrix, UmapError, UmapModel, UmapParams,
};

fn parse_metric(metric: &str) -> PyResult<Metric> {
    match metric.to_ascii_lowercase().as_str() {
        "euclidean" => Ok(Metric::Euclidean),
        "manhattan" | "l1" => Ok(Metric::Manhattan),
        "cosine" => Ok(Metric::Cosine),
        _ => Err(PyValueError::new_err(format!(
            "unsupported metric '{metric}', expected euclidean|manhattan|cosine"
        ))),
    }
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

fn array1_f32_to_vec(data: &PyReadonlyArray1<'_, f32>, name: &str) -> PyResult<Vec<f32>> {
    let slice = data
        .as_slice()
        .map_err(|_| PyValueError::new_err(format!("{name} must be contiguous")))?;
    Ok(slice.to_vec())
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
) -> PyResult<(Vec<Vec<usize>>, Vec<Vec<f32>>)> {
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

#[pyclass(name = "UmapCore", module = "umap_rs._umap_rs")]
struct PyUmapCore {
    inner: UmapModel,
}

#[pymethods]
impl PyUmapCore {
    #[new]
    #[pyo3(signature = (
        n_neighbors = 15,
        n_components = 2,
        n_epochs = None,
        metric = "euclidean",
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
        approx_knn_candidates = 30,
        approx_knn_iters = 10,
        approx_knn_threshold = 4096,
    ))]
    fn new(
        n_neighbors: usize,
        n_components: usize,
        n_epochs: Option<usize>,
        metric: &str,
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
    ) -> PyResult<Self> {
        let metric = parse_metric(metric)?;
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
        let knn_metric = parse_metric(knn_metric)?;

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
        let knn_metric = parse_metric(knn_metric)?;

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
}

#[pymodule]
fn _umap_rs(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyUmapCore>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
