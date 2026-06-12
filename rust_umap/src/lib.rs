use nalgebra::{DMatrix, SymmetricEigen};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Normal};
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::time::Instant;

pub mod aligned;
pub use aligned::{AlignedUmapError, AlignedUmapModel, AlignedUmapParams, AlignmentRelation};
pub mod parametric;
pub use parametric::{ParametricTrainMode, ParametricUmapModel, ParametricUmapParams};
pub mod sparse;
pub use sparse::SparseCsrMatrix;

const SMOOTH_K_TOLERANCE: f32 = 1e-5;
const MIN_K_DIST_SCALE: f32 = 1e-3;
const DEFAULT_BANDWIDTH: f32 = 1.0;
const INIT_MAX_COORD: f32 = 10.0;
const INIT_NOISE: f32 = 1e-4;
const AUTO_EXACT_LOW_DIM_MAX_FEATURES: usize = 16;
const AUTO_EXACT_LOW_DIM_SAMPLE_MULTIPLIER: usize = 4;
const AUTO_EXACT_GENERAL_SAMPLE_MULTIPLIER: usize = 4;
// Try the sparse iterative spectral path for every non-trivial connected graph.
// Dense full eigendecomposition remains the fallback inside `spectral_embedding_from_edges`.
const SPECTRAL_ITERATIVE_CONNECTED_THRESHOLD: usize = 0;
const SPECTRAL_ITERATIVE_COMPONENT_THRESHOLD: usize = 128;
const SPECTRAL_ITERATIVE_MAX_ITERS: usize = 32;
const SPECTRAL_ORTHO_EPS: f64 = 1e-12;
const DENSITY_GRADIENT_SCALE: f32 = 0.1;

type KnnRows = (Vec<Vec<usize>>, Vec<Vec<f32>>);
type EmbeddingRows = Vec<Vec<f32>>;
type FitArtifacts = (
    EmbeddingRows,
    Vec<f32>,
    Vec<f32>,
    Option<Vec<f32>>,
    Option<Vec<f32>>,
);
type WeightedUndirectedEdge = (usize, usize, f64);
type SymmetrizedUndirectedGraph = (Vec<WeightedUndirectedEdge>, Vec<f64>);

#[derive(Debug, Clone, Default)]
pub struct FitTimingBreakdown {
    pub validation_sec: f64,
    pub knn_sec: f64,
    pub curve_params_sec: f64,
    pub knn_validate_trim_sec: f64,
    pub smooth_knn_sec: f64,
    pub density_original_sec: f64,
    pub membership_sec: f64,
    pub symmetrize_sec: f64,
    pub target_intersection_sec: f64,
    pub prune_sec: f64,
    pub init_sec: f64,
    pub optimize_sec: f64,
    pub density_embedding_sec: f64,
    pub output_copy_sec: f64,
    pub store_state_sec: f64,
    pub total_sec: f64,
}

#[derive(Debug, Clone)]
pub struct ProfiledFit {
    pub embedding: DenseMatrix,
    pub timings: FitTimingBreakdown,
    pub used_approximate_knn: bool,
    pub n_samples: usize,
    pub n_features: usize,
    pub n_epochs: usize,
    pub n_edges: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum InitMethod {
    Random,
    #[default]
    Spectral,
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum Metric {
    #[default]
    Euclidean,
    Manhattan,
    Cosine,
    Chebyshev,
    Minkowski {
        p: f32,
    },
    Correlation,
    Canberra,
    BrayCurtis,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TargetMetric {
    Categorical,
}

#[derive(Debug, Clone)]
pub struct SupervisedTarget {
    pub labels: Vec<i64>,
    pub metric: TargetMetric,
    pub weight: f32,
    pub n_neighbors: Option<usize>,
}

impl SupervisedTarget {
    pub fn categorical(labels: Vec<i64>, weight: f32, n_neighbors: Option<usize>) -> Self {
        Self {
            labels,
            metric: TargetMetric::Categorical,
            weight,
            n_neighbors,
        }
    }
}

#[derive(Debug, Clone)]
pub struct UmapParams {
    pub n_neighbors: usize,
    pub n_components: usize,
    pub n_epochs: Option<usize>,
    pub metric: Metric,
    pub learning_rate: f32,
    pub min_dist: f32,
    pub spread: f32,
    pub local_connectivity: f32,
    pub set_op_mix_ratio: f32,
    pub repulsion_strength: f32,
    pub negative_sample_rate: usize,
    pub random_seed: u64,
    pub init: InitMethod,
    pub use_approximate_knn: bool,
    pub approx_knn_candidates: usize,
    pub approx_knn_iters: usize,
    pub approx_knn_threshold: usize,
    pub densmap: bool,
    pub dens_lambda: f32,
    pub dens_frac: f32,
    pub dens_var_shift: f32,
    pub output_dens: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct KnnRecallDiagnostics {
    pub n_samples: usize,
    pub n_features: usize,
    pub n_neighbors: usize,
    pub metric: Metric,
    pub candidate_pool: usize,
    pub n_iters: usize,
    pub seed: u64,
    pub mean_recall: f32,
    pub min_recall: f32,
    pub worst_decile_recall: f32,
    pub per_row_recall: Vec<f32>,
}

impl Default for UmapParams {
    fn default() -> Self {
        Self {
            n_neighbors: 15,
            n_components: 2,
            n_epochs: None,
            metric: Metric::default(),
            learning_rate: 1.0,
            min_dist: 0.1,
            spread: 1.0,
            local_connectivity: 1.0,
            set_op_mix_ratio: 1.0,
            repulsion_strength: 1.0,
            negative_sample_rate: 5,
            random_seed: 42,
            init: InitMethod::default(),
            use_approximate_knn: true,
            approx_knn_candidates: 50,
            approx_knn_iters: 14,
            approx_knn_threshold: 4096,
            densmap: false,
            dens_lambda: 2.0,
            dens_frac: 0.3,
            dens_var_shift: 0.1,
            output_dens: false,
        }
    }
}

#[derive(Debug)]
pub enum UmapError {
    EmptyData,
    NeedAtLeastTwoSamples,
    InconsistentDimensions {
        row: usize,
        expected: usize,
        got: usize,
    },
    FeatureMismatch {
        expected: usize,
        got: usize,
    },
    EmbeddingDimensionMismatch {
        expected: usize,
        got: usize,
    },
    NotFitted,
    InvalidParameter(String),
}

impl Display for UmapError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            UmapError::EmptyData => write!(f, "input data is empty"),
            UmapError::NeedAtLeastTwoSamples => write!(f, "need at least two samples"),
            UmapError::InconsistentDimensions { row, expected, got } => {
                write!(
                    f,
                    "inconsistent row dimension at row {row}: expected {expected}, got {got}"
                )
            }
            UmapError::FeatureMismatch { expected, got } => {
                write!(f, "feature mismatch: expected {expected}, got {got}")
            }
            UmapError::EmbeddingDimensionMismatch { expected, got } => {
                write!(
                    f,
                    "embedding dimension mismatch: expected {expected}, got {got}"
                )
            }
            UmapError::NotFitted => write!(f, "model is not fitted yet"),
            UmapError::InvalidParameter(msg) => write!(f, "invalid parameter: {msg}"),
        }
    }
}

impl Error for UmapError {}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Edge {
    head: usize,
    tail: usize,
    weight: f32,
}

#[derive(Debug, Clone)]
pub struct DenseMatrix {
    data: Vec<f32>,
    n_rows: usize,
    n_cols: usize,
}

impl DenseMatrix {
    pub fn from_rows(rows: &[Vec<f32>]) -> Result<Self, UmapError> {
        let (n_rows, n_cols) = validate_data(rows)?;
        let mut data = Vec::with_capacity(n_rows.saturating_mul(n_cols));
        for row in rows {
            data.extend_from_slice(row);
        }
        Ok(Self {
            data,
            n_rows,
            n_cols,
        })
    }

    pub fn from_flat(data: &[f32], n_rows: usize, n_cols: usize) -> Result<Self, UmapError> {
        let view = DenseMatrixView::new(data, n_rows, n_cols)?;
        validate_data(&view)?;
        Ok(Self {
            data: data.to_vec(),
            n_rows,
            n_cols,
        })
    }

    pub fn from_row_vectors(mut rows: Vec<Vec<f32>>) -> Result<Self, UmapError> {
        let (n_rows, n_cols) = validate_data(rows.as_slice())?;
        let mut data = Vec::with_capacity(n_rows.saturating_mul(n_cols));
        for row in rows.iter_mut() {
            data.append(row);
        }
        Ok(Self {
            data,
            n_rows,
            n_cols,
        })
    }

    fn from_rows_validated(rows: &[Vec<f32>], n_rows: usize, n_cols: usize) -> Self {
        let mut data = Vec::with_capacity(n_rows.saturating_mul(n_cols));
        for row in rows {
            data.extend_from_slice(row);
        }
        Self {
            data,
            n_rows,
            n_cols,
        }
    }

    fn from_flat_validated(data: &[f32], n_rows: usize, n_cols: usize) -> Self {
        Self {
            data: data.to_vec(),
            n_rows,
            n_cols,
        }
    }

    fn from_row_vectors_validated(mut rows: Vec<Vec<f32>>, n_rows: usize, n_cols: usize) -> Self {
        let mut data = Vec::with_capacity(n_rows.saturating_mul(n_cols));
        for row in rows.iter_mut() {
            data.append(row);
        }
        Self {
            data,
            n_rows,
            n_cols,
        }
    }

    pub fn n_rows(&self) -> usize {
        self.n_rows
    }

    pub fn n_cols(&self) -> usize {
        self.n_cols
    }

    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }

    pub fn into_raw_parts(self) -> (Vec<f32>, usize, usize) {
        (self.data, self.n_rows, self.n_cols)
    }

    fn row_unchecked(&self, idx: usize) -> &[f32] {
        let start = idx * self.n_cols;
        &self.data[start..start + self.n_cols]
    }
}

fn dense_matrix_from_output_rows(
    mut rows: Vec<Vec<f32>>,
    empty_cols: usize,
) -> Result<DenseMatrix, UmapError> {
    let n_rows = rows.len();
    let n_cols = if n_rows == 0 {
        empty_cols
    } else {
        rows[0].len()
    };

    let mut data = Vec::with_capacity(n_rows.saturating_mul(n_cols));
    for (row_idx, row) in rows.iter_mut().enumerate() {
        if row.len() != n_cols {
            return Err(UmapError::InconsistentDimensions {
                row: row_idx,
                expected: n_cols,
                got: row.len(),
            });
        }
        data.append(row);
    }

    Ok(DenseMatrix {
        data,
        n_rows,
        n_cols,
    })
}

struct DenseMatrixView<'a> {
    data: &'a [f32],
    n_rows: usize,
    n_cols: usize,
}

impl<'a> DenseMatrixView<'a> {
    fn new(data: &'a [f32], n_rows: usize, n_cols: usize) -> Result<Self, UmapError> {
        if n_cols == 0 {
            return Err(UmapError::InvalidParameter(
                "dense matrix view must have at least one column".to_string(),
            ));
        }
        if data.len() != n_rows.saturating_mul(n_cols) {
            return Err(UmapError::InvalidParameter(format!(
                "dense matrix view length mismatch: expected {}, got {}",
                n_rows.saturating_mul(n_cols),
                data.len()
            )));
        }
        Ok(Self {
            data,
            n_rows,
            n_cols,
        })
    }
}

struct FlatF32MatrixView<'a> {
    data: &'a [f32],
    n_rows: usize,
    n_cols: usize,
}

impl<'a> FlatF32MatrixView<'a> {
    fn new(data: &'a [f32], n_rows: usize, n_cols: usize) -> Result<Self, UmapError> {
        if data.len() != n_rows.saturating_mul(n_cols) {
            return Err(UmapError::InvalidParameter(format!(
                "flat matrix view length mismatch: expected {}, got {}",
                n_rows.saturating_mul(n_cols),
                data.len()
            )));
        }
        Ok(Self {
            data,
            n_rows,
            n_cols,
        })
    }
}

struct UsizeMatrixView<'a> {
    data: &'a [usize],
    n_rows: usize,
    n_cols: usize,
}

impl<'a> UsizeMatrixView<'a> {
    fn new(data: &'a [usize], n_rows: usize, n_cols: usize) -> Result<Self, UmapError> {
        if data.len() != n_rows.saturating_mul(n_cols) {
            return Err(UmapError::InvalidParameter(format!(
                "usize matrix view length mismatch: expected {}, got {}",
                n_rows.saturating_mul(n_cols),
                data.len()
            )));
        }
        Ok(Self {
            data,
            n_rows,
            n_cols,
        })
    }
}

trait RowMatrix {
    fn n_rows(&self) -> usize;
    fn n_cols(&self) -> usize;
    fn row(&self, idx: usize) -> &[f32];
}

trait IndexRowMatrix {
    fn n_rows(&self) -> usize;
    fn row(&self, idx: usize) -> &[usize];
}

impl RowMatrix for [Vec<f32>] {
    fn n_rows(&self) -> usize {
        self.len()
    }

    fn n_cols(&self) -> usize {
        self.first().map_or(0, Vec::len)
    }

    fn row(&self, idx: usize) -> &[f32] {
        &self[idx]
    }
}

impl RowMatrix for Vec<Vec<f32>> {
    fn n_rows(&self) -> usize {
        self.len()
    }

    fn n_cols(&self) -> usize {
        self.first().map_or(0, Vec::len)
    }

    fn row(&self, idx: usize) -> &[f32] {
        &self[idx]
    }
}

impl RowMatrix for DenseMatrix {
    fn n_rows(&self) -> usize {
        self.n_rows
    }

    fn n_cols(&self) -> usize {
        self.n_cols
    }

    fn row(&self, idx: usize) -> &[f32] {
        self.row_unchecked(idx)
    }
}

impl RowMatrix for DenseMatrixView<'_> {
    fn n_rows(&self) -> usize {
        self.n_rows
    }

    fn n_cols(&self) -> usize {
        self.n_cols
    }

    fn row(&self, idx: usize) -> &[f32] {
        let start = idx * self.n_cols;
        &self.data[start..start + self.n_cols]
    }
}

impl RowMatrix for FlatF32MatrixView<'_> {
    fn n_rows(&self) -> usize {
        self.n_rows
    }

    fn n_cols(&self) -> usize {
        self.n_cols
    }

    fn row(&self, idx: usize) -> &[f32] {
        let start = idx * self.n_cols;
        &self.data[start..start + self.n_cols]
    }
}

impl IndexRowMatrix for [Vec<usize>] {
    fn n_rows(&self) -> usize {
        self.len()
    }

    fn row(&self, idx: usize) -> &[usize] {
        &self[idx]
    }
}

impl IndexRowMatrix for Vec<Vec<usize>> {
    fn n_rows(&self) -> usize {
        self.len()
    }

    fn row(&self, idx: usize) -> &[usize] {
        &self[idx]
    }
}

impl IndexRowMatrix for UsizeMatrixView<'_> {
    fn n_rows(&self) -> usize {
        self.n_rows
    }

    fn row(&self, idx: usize) -> &[usize] {
        let start = idx * self.n_cols;
        &self.data[start..start + self.n_cols]
    }
}

#[derive(Debug, Clone)]
pub struct UmapModel {
    params: UmapParams,
    a: f32,
    b: f32,
    embedding: Option<Vec<Vec<f32>>>,
    training_data_dense: Option<DenseMatrix>,
    training_data_sparse: Option<SparseCsrMatrix>,
    n_features: Option<usize>,
    fit_sigmas: Option<Vec<f32>>,
    fit_rhos: Option<Vec<f32>>,
    fit_radii_original: Option<Vec<f32>>,
    fit_radii_embedding: Option<Vec<f32>>,
}

impl UmapModel {
    pub fn new(params: UmapParams) -> Self {
        Self {
            params,
            a: 0.0,
            b: 0.0,
            embedding: None,
            training_data_dense: None,
            training_data_sparse: None,
            n_features: None,
            fit_sigmas: None,
            fit_rhos: None,
            fit_radii_original: None,
            fit_radii_embedding: None,
        }
    }

    pub fn params(&self) -> &UmapParams {
        &self.params
    }

    pub fn ab_params(&self) -> Option<(f32, f32)> {
        if self.a > 0.0 && self.b > 0.0 {
            Some((self.a, self.b))
        } else {
            None
        }
    }

    pub fn embedding(&self) -> Option<&[Vec<f32>]> {
        self.embedding.as_deref()
    }

    pub fn n_features(&self) -> Option<usize> {
        self.n_features
    }

    pub fn radii_original(&self) -> Option<&[f32]> {
        self.fit_radii_original.as_deref()
    }

    pub fn radii_embedding(&self) -> Option<&[f32]> {
        self.fit_radii_embedding.as_deref()
    }

    pub fn fit(&mut self, data: &[Vec<f32>]) -> Result<(), UmapError> {
        let (n_samples, n_features) = validate_data(data)?;
        validate_params(&self.params, n_samples, n_features)?;
        let (knn_indices, knn_dists) = build_fit_knn(&self.params, data, n_samples, n_features);
        self.fit_with_knn_prevalidated(data, n_features, &knn_indices, &knn_dists)
    }

    pub fn fit_supervised(
        &mut self,
        data: &[Vec<f32>],
        target: &SupervisedTarget,
    ) -> Result<(), UmapError> {
        let (n_samples, n_features) = validate_data(data)?;
        validate_params(&self.params, n_samples, n_features)?;
        validate_supervised_target(target, n_samples)?;
        let (knn_indices, knn_dists) = build_fit_knn(&self.params, data, n_samples, n_features);
        self.fit_with_knn_supervised_prevalidated(
            data,
            n_features,
            &knn_indices,
            &knn_dists,
            target,
        )
    }

    pub fn fit_transform(&mut self, data: &[Vec<f32>]) -> Result<Vec<Vec<f32>>, UmapError> {
        let (n_samples, n_features) = validate_data(data)?;
        validate_params(&self.params, n_samples, n_features)?;

        let (knn_indices, knn_dists) = build_fit_knn(&self.params, data, n_samples, n_features);

        self.fit_transform_with_knn_prevalidated(data, n_features, &knn_indices, &knn_dists)
    }

    pub fn fit_transform_supervised(
        &mut self,
        data: &[Vec<f32>],
        target: &SupervisedTarget,
    ) -> Result<Vec<Vec<f32>>, UmapError> {
        let (n_samples, n_features) = validate_data(data)?;
        validate_params(&self.params, n_samples, n_features)?;
        validate_supervised_target(target, n_samples)?;

        let (knn_indices, knn_dists) = build_fit_knn(&self.params, data, n_samples, n_features);

        self.fit_transform_with_knn_supervised_prevalidated(
            data,
            n_features,
            &knn_indices,
            &knn_dists,
            target,
        )
    }

    pub fn fit_owned(&mut self, data: Vec<Vec<f32>>) -> Result<(), UmapError> {
        let (n_samples, n_features) = validate_data(&data)?;
        validate_params(&self.params, n_samples, n_features)?;
        let (knn_indices, knn_dists) = build_fit_knn(&self.params, &data, n_samples, n_features);

        self.fit_with_knn_owned_prevalidated(data, n_features, &knn_indices, &knn_dists)
    }

    pub fn fit_transform_owned(&mut self, data: Vec<Vec<f32>>) -> Result<Vec<Vec<f32>>, UmapError> {
        let (n_samples, n_features) = validate_data(&data)?;
        validate_params(&self.params, n_samples, n_features)?;
        let (knn_indices, knn_dists) = build_fit_knn(&self.params, &data, n_samples, n_features);

        self.fit_transform_with_knn_owned_prevalidated(data, n_features, &knn_indices, &knn_dists)
    }

    pub fn fit_dense(
        &mut self,
        data: &[f32],
        n_rows: usize,
        n_cols: usize,
    ) -> Result<(), UmapError> {
        let data_view = DenseMatrixView::new(data, n_rows, n_cols)?;
        let (n_samples, n_features) = validate_data(&data_view)?;
        validate_params(&self.params, n_samples, n_features)?;
        let (knn_indices, knn_dists) =
            build_fit_knn(&self.params, &data_view, n_samples, n_features);
        let (embedding, sigmas, rhos, radii_original, radii_embedding) =
            self.build_dense_fit_artifacts(&data_view, &knn_indices, &knn_dists)?;
        let training_data = DenseMatrix::from_flat_validated(data, n_rows, n_cols);
        self.store_dense_fit_state(
            embedding,
            training_data,
            n_features,
            sigmas,
            rhos,
            radii_original,
            radii_embedding,
        );
        Ok(())
    }

    pub fn fit_dense_supervised(
        &mut self,
        data: &[f32],
        n_rows: usize,
        n_cols: usize,
        target: &SupervisedTarget,
    ) -> Result<(), UmapError> {
        let data_view = DenseMatrixView::new(data, n_rows, n_cols)?;
        let (n_samples, n_features) = validate_data(&data_view)?;
        validate_params(&self.params, n_samples, n_features)?;
        validate_supervised_target(target, n_samples)?;
        let (knn_indices, knn_dists) =
            build_fit_knn(&self.params, &data_view, n_samples, n_features);
        let (embedding, sigmas, rhos, radii_original, radii_embedding) = self
            .build_dense_fit_artifacts_with_target(
                &data_view,
                &knn_indices,
                &knn_dists,
                Some(target),
            )?;
        let training_data = DenseMatrix::from_flat_validated(data, n_rows, n_cols);
        self.store_dense_fit_state(
            embedding,
            training_data,
            n_features,
            sigmas,
            rhos,
            radii_original,
            radii_embedding,
        );
        Ok(())
    }

    pub fn fit_transform_dense(
        &mut self,
        data: &[f32],
        n_rows: usize,
        n_cols: usize,
    ) -> Result<DenseMatrix, UmapError> {
        let data_view = DenseMatrixView::new(data, n_rows, n_cols)?;
        let (n_samples, n_features) = validate_data(&data_view)?;
        validate_params(&self.params, n_samples, n_features)?;
        let (knn_indices, knn_dists) =
            build_fit_knn(&self.params, &data_view, n_samples, n_features);
        let (embedding, sigmas, rhos, radii_original, radii_embedding) =
            self.build_dense_fit_artifacts(&data_view, &knn_indices, &knn_dists)?;
        let out = dense_matrix_from_output_rows(embedding.clone(), self.params.n_components)?;
        let training_data = DenseMatrix::from_flat_validated(data, n_rows, n_cols);
        self.store_dense_fit_state(
            embedding,
            training_data,
            n_features,
            sigmas,
            rhos,
            radii_original,
            radii_embedding,
        );
        Ok(out)
    }

    pub fn profile_fit_transform_dense(
        &mut self,
        data: &[f32],
        n_rows: usize,
        n_cols: usize,
    ) -> Result<ProfiledFit, UmapError> {
        let total_start = Instant::now();
        let mut timings = FitTimingBreakdown::default();

        let stage = Instant::now();
        let data_view = DenseMatrixView::new(data, n_rows, n_cols)?;
        let (n_samples, n_features) = validate_data(&data_view)?;
        validate_params(&self.params, n_samples, n_features)?;
        timings.validation_sec = stage.elapsed().as_secs_f64();

        let used_approximate_knn = should_use_approximate_knn(&self.params, n_samples, n_features);
        let stage = Instant::now();
        let (knn_indices, knn_dists) = if used_approximate_knn {
            approximate_nearest_neighbors(
                &data_view,
                self.params.n_neighbors,
                self.params.metric,
                self.params.approx_knn_candidates,
                self.params.approx_knn_iters,
                self.params.random_seed ^ 0xC0FE_FEED_1234_ABCD,
            )
        } else {
            exact_nearest_neighbors(&data_view, self.params.n_neighbors, self.params.metric)
        };
        timings.knn_sec = stage.elapsed().as_secs_f64();

        let n_epochs = self
            .params
            .n_epochs
            .unwrap_or(if n_samples <= 10_000 { 500 } else { 200 });

        let stage = Instant::now();
        let (a, b) = find_ab_params(self.params.spread, self.params.min_dist);
        self.a = a;
        self.b = b;
        timings.curve_params_sec = stage.elapsed().as_secs_f64();

        let stage = Instant::now();
        let (knn_indices_trimmed, knn_dists_trimmed) = validate_and_trim_precomputed_knn(
            &knn_indices,
            &knn_dists,
            n_samples,
            self.params.n_neighbors,
        )?;
        timings.knn_validate_trim_sec = stage.elapsed().as_secs_f64();

        let stage = Instant::now();
        let (sigmas, rhos) = smooth_knn_dist(
            &knn_dists_trimmed,
            self.params.n_neighbors as f32,
            self.params.local_connectivity,
            DEFAULT_BANDWIDTH,
            knn_rows_include_self(&knn_indices_trimmed, &knn_dists_trimmed),
        );
        timings.smooth_knn_sec = stage.elapsed().as_secs_f64();

        let compute_density_outputs = self.params.densmap || self.params.output_dens;
        let stage = Instant::now();
        let radii_original = if compute_density_outputs {
            Some(local_radii_from_knn(
                &knn_indices_trimmed,
                &knn_dists_trimmed,
                self.params.dens_var_shift,
            ))
        } else {
            None
        };
        let density_state = if self.params.densmap && self.params.dens_lambda > 0.0 {
            Some(DensityState {
                original_radii: radii_original
                    .clone()
                    .expect("densmap should compute original radii"),
                start_epoch: density_start_epoch(n_epochs, self.params.dens_frac),
                lambda: self.params.dens_lambda,
                var_shift: self.params.dens_var_shift,
            })
        } else {
            None
        };
        timings.density_original_sec = stage.elapsed().as_secs_f64();

        let stage = Instant::now();
        let directed =
            compute_membership_strengths(&knn_indices_trimmed, &knn_dists_trimmed, &sigmas, &rhos);
        timings.membership_sec = stage.elapsed().as_secs_f64();

        let stage = Instant::now();
        let mut edges = symmetrize_fuzzy_graph(&directed, self.params.set_op_mix_ratio);
        timings.symmetrize_sec = stage.elapsed().as_secs_f64();

        // Dense unsupervised profiling has no target graph; keep a zero-valued
        // stage so reports stay schema-compatible with supervised profiling.
        timings.target_intersection_sec = 0.0;

        let stage = Instant::now();
        prune_edges(&mut edges, n_epochs);
        let n_edges = edges.len();
        timings.prune_sec = stage.elapsed().as_secs_f64();

        let stage = Instant::now();
        let mut embedding = initialize_embedding(
            &data_view,
            self.params.n_components,
            &edges,
            self.params.init,
            self.params.random_seed,
        );
        timings.init_sec = stage.elapsed().as_secs_f64();

        let stage = Instant::now();
        optimize_layout_training(
            &mut embedding,
            &edges,
            n_epochs,
            a,
            b,
            self.params.learning_rate,
            self.params.negative_sample_rate,
            self.params.repulsion_strength,
            self.params.random_seed ^ 0x9E37_79B9_7F4A_7C15,
            density_state.as_ref(),
        );
        timings.optimize_sec = stage.elapsed().as_secs_f64();

        let stage = Instant::now();
        let radii_embedding = if compute_density_outputs {
            Some(embedding_radii_from_knn(
                &embedding,
                &knn_indices_trimmed,
                self.params.dens_var_shift,
            ))
        } else {
            None
        };
        timings.density_embedding_sec = stage.elapsed().as_secs_f64();

        let stage = Instant::now();
        let out = dense_matrix_from_output_rows(embedding.clone(), self.params.n_components)?;
        timings.output_copy_sec = stage.elapsed().as_secs_f64();

        let stage = Instant::now();
        let training_data = DenseMatrix::from_flat_validated(data, n_rows, n_cols);
        self.store_dense_fit_state(
            embedding,
            training_data,
            n_features,
            sigmas,
            rhos,
            radii_original,
            radii_embedding,
        );
        timings.store_state_sec = stage.elapsed().as_secs_f64();
        timings.total_sec = total_start.elapsed().as_secs_f64();

        Ok(ProfiledFit {
            embedding: out,
            timings,
            used_approximate_knn,
            n_samples,
            n_features,
            n_epochs,
            n_edges,
        })
    }

    pub fn fit_transform_dense_supervised(
        &mut self,
        data: &[f32],
        n_rows: usize,
        n_cols: usize,
        target: &SupervisedTarget,
    ) -> Result<DenseMatrix, UmapError> {
        let data_view = DenseMatrixView::new(data, n_rows, n_cols)?;
        let (n_samples, n_features) = validate_data(&data_view)?;
        validate_params(&self.params, n_samples, n_features)?;
        validate_supervised_target(target, n_samples)?;
        let (knn_indices, knn_dists) =
            build_fit_knn(&self.params, &data_view, n_samples, n_features);
        let (embedding, sigmas, rhos, radii_original, radii_embedding) = self
            .build_dense_fit_artifacts_with_target(
                &data_view,
                &knn_indices,
                &knn_dists,
                Some(target),
            )?;
        let out = dense_matrix_from_output_rows(embedding.clone(), self.params.n_components)?;
        let training_data = DenseMatrix::from_flat_validated(data, n_rows, n_cols);
        self.store_dense_fit_state(
            embedding,
            training_data,
            n_features,
            sigmas,
            rhos,
            radii_original,
            radii_embedding,
        );
        Ok(out)
    }

    pub fn fit_sparse_csr(&mut self, data: SparseCsrMatrix) -> Result<(), UmapError> {
        let n_samples = data.n_rows();
        let n_features = data.n_cols();
        if n_samples == 0 {
            return Err(UmapError::EmptyData);
        }
        if n_samples < 2 {
            return Err(UmapError::NeedAtLeastTwoSamples);
        }

        validate_params(&self.params, n_samples, n_features)?;
        validate_sparse_metric(self.params.metric)?;
        validate_sparse_density_params(&self.params)?;
        let (knn_indices, knn_dists) =
            sparse::exact_nearest_neighbors(&data, self.params.n_neighbors, self.params.metric);
        self.fit_sparse_csr_prevalidated(data, &knn_indices, &knn_dists)
    }

    pub fn fit_transform_sparse_csr(
        &mut self,
        data: SparseCsrMatrix,
    ) -> Result<Vec<Vec<f32>>, UmapError> {
        let n_samples = data.n_rows();
        let n_features = data.n_cols();
        if n_samples == 0 {
            return Err(UmapError::EmptyData);
        }
        if n_samples < 2 {
            return Err(UmapError::NeedAtLeastTwoSamples);
        }

        validate_params(&self.params, n_samples, n_features)?;
        validate_sparse_metric(self.params.metric)?;
        validate_sparse_density_params(&self.params)?;
        let (knn_indices, knn_dists) =
            sparse::exact_nearest_neighbors(&data, self.params.n_neighbors, self.params.metric);
        self.fit_transform_with_knn_sparse_prevalidated(data, &knn_indices, &knn_dists)
    }

    pub fn fit_transform_with_knn(
        &mut self,
        data: &[Vec<f32>],
        knn_indices: &[Vec<usize>],
        knn_dists: &[Vec<f32>],
    ) -> Result<Vec<Vec<f32>>, UmapError> {
        self.fit_transform_with_knn_metric(data, knn_indices, knn_dists, self.params.metric)
    }

    pub fn fit_transform_with_knn_metric(
        &mut self,
        data: &[Vec<f32>],
        knn_indices: &[Vec<usize>],
        knn_dists: &[Vec<f32>],
        knn_metric: Metric,
    ) -> Result<Vec<Vec<f32>>, UmapError> {
        if knn_metric != self.params.metric {
            return Err(UmapError::InvalidParameter(format!(
                "precomputed knn metric ({knn_metric:?}) must match model metric ({:?})",
                self.params.metric
            )));
        }
        let (n_samples, n_features) = validate_data(data)?;
        validate_params(&self.params, n_samples, n_features)?;
        self.fit_transform_with_knn_prevalidated(data, n_features, knn_indices, knn_dists)
    }

    pub fn fit_transform_with_knn_metric_owned(
        &mut self,
        data: Vec<Vec<f32>>,
        knn_indices: &[Vec<usize>],
        knn_dists: &[Vec<f32>],
        knn_metric: Metric,
    ) -> Result<Vec<Vec<f32>>, UmapError> {
        if knn_metric != self.params.metric {
            return Err(UmapError::InvalidParameter(format!(
                "precomputed knn metric ({knn_metric:?}) must match model metric ({:?})",
                self.params.metric
            )));
        }
        let (n_samples, n_features) = validate_data(&data)?;
        validate_params(&self.params, n_samples, n_features)?;
        self.fit_transform_with_knn_owned_prevalidated(data, n_features, knn_indices, knn_dists)
    }

    pub fn fit_transform_with_knn_metric_dense(
        &mut self,
        data: &[f32],
        n_rows: usize,
        n_cols: usize,
        knn_indices: &[Vec<usize>],
        knn_dists: &[Vec<f32>],
        knn_metric: Metric,
    ) -> Result<DenseMatrix, UmapError> {
        if knn_metric != self.params.metric {
            return Err(UmapError::InvalidParameter(format!(
                "precomputed knn metric ({knn_metric:?}) must match model metric ({:?})",
                self.params.metric
            )));
        }

        let data_view = DenseMatrixView::new(data, n_rows, n_cols)?;
        let (n_samples, n_features) = validate_data(&data_view)?;
        validate_params(&self.params, n_samples, n_features)?;

        let (embedding, sigmas, rhos, radii_original, radii_embedding) =
            self.build_dense_fit_artifacts(&data_view, knn_indices, knn_dists)?;
        let out = dense_matrix_from_output_rows(embedding.clone(), self.params.n_components)?;
        let training_data = DenseMatrix::from_flat_validated(data, n_rows, n_cols);
        self.store_dense_fit_state(
            embedding,
            training_data,
            n_features,
            sigmas,
            rhos,
            radii_original,
            radii_embedding,
        );
        Ok(out)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn fit_transform_with_knn_metric_dense_flat(
        &mut self,
        data: &[f32],
        n_rows: usize,
        n_cols: usize,
        knn_indices: &[usize],
        knn_dists: &[f32],
        knn_cols: usize,
        knn_metric: Metric,
    ) -> Result<DenseMatrix, UmapError> {
        if knn_metric != self.params.metric {
            return Err(UmapError::InvalidParameter(format!(
                "precomputed knn metric ({knn_metric:?}) must match model metric ({:?})",
                self.params.metric
            )));
        }

        let data_view = DenseMatrixView::new(data, n_rows, n_cols)?;
        let knn_index_view = UsizeMatrixView::new(knn_indices, n_rows, knn_cols)?;
        let knn_dist_view = FlatF32MatrixView::new(knn_dists, n_rows, knn_cols)?;
        let (n_samples, n_features) = validate_data(&data_view)?;
        validate_params(&self.params, n_samples, n_features)?;

        let (embedding, sigmas, rhos, radii_original, radii_embedding) =
            self.build_dense_fit_artifacts(&data_view, &knn_index_view, &knn_dist_view)?;
        let out = dense_matrix_from_output_rows(embedding.clone(), self.params.n_components)?;
        let training_data = DenseMatrix::from_flat_validated(data, n_rows, n_cols);
        self.store_dense_fit_state(
            embedding,
            training_data,
            n_features,
            sigmas,
            rhos,
            radii_original,
            radii_embedding,
        );
        Ok(out)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn fit_transform_with_knn_metric_dense_i64_flat(
        &mut self,
        data: &[f32],
        n_rows: usize,
        n_cols: usize,
        knn_indices: &[i64],
        knn_indices_rows: usize,
        knn_indices_cols: usize,
        knn_dists: &[f32],
        knn_dists_rows: usize,
        knn_dists_cols: usize,
        knn_metric: Metric,
        validate_precomputed: bool,
    ) -> Result<DenseMatrix, UmapError> {
        if knn_metric != self.params.metric {
            return Err(UmapError::InvalidParameter(format!(
                "precomputed knn metric ({knn_metric:?}) must match model metric ({:?})",
                self.params.metric
            )));
        }

        let data_view = DenseMatrixView::new(data, n_rows, n_cols)?;
        let (n_samples, n_features) = validate_data(&data_view)?;
        validate_params(&self.params, n_samples, n_features)?;
        validate_precomputed_array_shapes(
            n_samples,
            knn_indices_rows,
            knn_indices_cols,
            knn_dists_rows,
            knn_dists_cols,
            self.params.n_neighbors,
        )?;
        if validate_precomputed {
            validate_precomputed_distance_values_flat(knn_dists)?;
        }

        let knn_idx_rows =
            precomputed_knn_indices_from_i64_flat(knn_indices, knn_indices_rows, knn_indices_cols)?;
        let knn_dist_view = FlatF32MatrixView::new(knn_dists, knn_dists_rows, knn_dists_cols)?;
        let (embedding, sigmas, rhos, radii_original, radii_embedding) =
            self.build_dense_fit_artifacts(&data_view, &knn_idx_rows, &knn_dist_view)?;
        let out = dense_matrix_from_output_rows(embedding.clone(), self.params.n_components)?;
        let training_data = DenseMatrix::from_flat_validated(data, n_rows, n_cols);
        self.store_dense_fit_state(
            embedding,
            training_data,
            n_features,
            sigmas,
            rhos,
            radii_original,
            radii_embedding,
        );
        Ok(out)
    }

    fn fit_transform_with_knn_prevalidated(
        &mut self,
        data: &[Vec<f32>],
        n_features: usize,
        knn_indices: &[Vec<usize>],
        knn_dists: &[Vec<f32>],
    ) -> Result<Vec<Vec<f32>>, UmapError> {
        let (embedding, sigmas, rhos, radii_original, radii_embedding) =
            self.build_dense_fit_artifacts(data, knn_indices, knn_dists)?;
        let training_data = DenseMatrix::from_rows_validated(data, data.len(), n_features);

        self.store_dense_fit_state(
            embedding.clone(),
            training_data,
            n_features,
            sigmas,
            rhos,
            radii_original,
            radii_embedding,
        );
        Ok(embedding)
    }

    fn fit_transform_with_knn_supervised_prevalidated(
        &mut self,
        data: &[Vec<f32>],
        n_features: usize,
        knn_indices: &[Vec<usize>],
        knn_dists: &[Vec<f32>],
        target: &SupervisedTarget,
    ) -> Result<Vec<Vec<f32>>, UmapError> {
        let (embedding, sigmas, rhos, radii_original, radii_embedding) =
            self.build_dense_fit_artifacts_with_target(data, knn_indices, knn_dists, Some(target))?;
        let training_data = DenseMatrix::from_rows_validated(data, data.len(), n_features);

        self.store_dense_fit_state(
            embedding.clone(),
            training_data,
            n_features,
            sigmas,
            rhos,
            radii_original,
            radii_embedding,
        );
        Ok(embedding)
    }

    fn fit_with_knn_prevalidated(
        &mut self,
        data: &[Vec<f32>],
        n_features: usize,
        knn_indices: &[Vec<usize>],
        knn_dists: &[Vec<f32>],
    ) -> Result<(), UmapError> {
        let (embedding, sigmas, rhos, radii_original, radii_embedding) =
            self.build_dense_fit_artifacts(data, knn_indices, knn_dists)?;
        let training_data = DenseMatrix::from_rows_validated(data, data.len(), n_features);
        self.store_dense_fit_state(
            embedding,
            training_data,
            n_features,
            sigmas,
            rhos,
            radii_original,
            radii_embedding,
        );
        Ok(())
    }

    fn fit_with_knn_supervised_prevalidated(
        &mut self,
        data: &[Vec<f32>],
        n_features: usize,
        knn_indices: &[Vec<usize>],
        knn_dists: &[Vec<f32>],
        target: &SupervisedTarget,
    ) -> Result<(), UmapError> {
        let (embedding, sigmas, rhos, radii_original, radii_embedding) =
            self.build_dense_fit_artifacts_with_target(data, knn_indices, knn_dists, Some(target))?;
        let training_data = DenseMatrix::from_rows_validated(data, data.len(), n_features);
        self.store_dense_fit_state(
            embedding,
            training_data,
            n_features,
            sigmas,
            rhos,
            radii_original,
            radii_embedding,
        );
        Ok(())
    }

    fn fit_with_knn_owned_prevalidated(
        &mut self,
        data: Vec<Vec<f32>>,
        n_features: usize,
        knn_indices: &[Vec<usize>],
        knn_dists: &[Vec<f32>],
    ) -> Result<(), UmapError> {
        let (embedding, sigmas, rhos, radii_original, radii_embedding) =
            self.build_dense_fit_artifacts(&data, knn_indices, knn_dists)?;
        let n_rows = data.len();
        let training_data = DenseMatrix::from_row_vectors_validated(data, n_rows, n_features);
        self.store_dense_fit_state(
            embedding,
            training_data,
            n_features,
            sigmas,
            rhos,
            radii_original,
            radii_embedding,
        );
        Ok(())
    }

    fn fit_transform_with_knn_owned_prevalidated(
        &mut self,
        data: Vec<Vec<f32>>,
        n_features: usize,
        knn_indices: &[Vec<usize>],
        knn_dists: &[Vec<f32>],
    ) -> Result<Vec<Vec<f32>>, UmapError> {
        let (embedding, sigmas, rhos, radii_original, radii_embedding) =
            self.build_dense_fit_artifacts(&data, knn_indices, knn_dists)?;
        let n_rows = data.len();
        let training_data = DenseMatrix::from_row_vectors_validated(data, n_rows, n_features);
        let out = embedding.clone();
        self.store_dense_fit_state(
            embedding,
            training_data,
            n_features,
            sigmas,
            rhos,
            radii_original,
            radii_embedding,
        );
        Ok(out)
    }

    fn build_dense_fit_artifacts<M, I, D>(
        &mut self,
        data: &M,
        knn_indices: &I,
        knn_dists: &D,
    ) -> Result<FitArtifacts, UmapError>
    where
        M: RowMatrix + ?Sized,
        I: IndexRowMatrix + ?Sized,
        D: RowMatrix + ?Sized,
    {
        self.build_dense_fit_artifacts_with_target(data, knn_indices, knn_dists, None)
    }

    fn build_dense_fit_artifacts_with_target<M, I, D>(
        &mut self,
        data: &M,
        knn_indices: &I,
        knn_dists: &D,
        target: Option<&SupervisedTarget>,
    ) -> Result<FitArtifacts, UmapError>
    where
        M: RowMatrix + ?Sized,
        I: IndexRowMatrix + ?Sized,
        D: RowMatrix + ?Sized,
    {
        let n_samples = data.n_rows();
        let n_epochs = self
            .params
            .n_epochs
            .unwrap_or(if n_samples <= 10_000 { 500 } else { 200 });

        let (a, b) = find_ab_params(self.params.spread, self.params.min_dist);
        self.a = a;
        self.b = b;

        let (knn_indices_trimmed, knn_dists_trimmed) = validate_and_trim_precomputed_knn(
            knn_indices,
            knn_dists,
            n_samples,
            self.params.n_neighbors,
        )?;

        let (sigmas, rhos) = smooth_knn_dist(
            &knn_dists_trimmed,
            self.params.n_neighbors as f32,
            self.params.local_connectivity,
            DEFAULT_BANDWIDTH,
            knn_rows_include_self(&knn_indices_trimmed, &knn_dists_trimmed),
        );
        let compute_density_outputs = self.params.densmap || self.params.output_dens;
        let radii_original = if compute_density_outputs {
            Some(local_radii_from_knn(
                &knn_indices_trimmed,
                &knn_dists_trimmed,
                self.params.dens_var_shift,
            ))
        } else {
            None
        };
        let density_state = if self.params.densmap && self.params.dens_lambda > 0.0 {
            Some(DensityState {
                original_radii: radii_original
                    .clone()
                    .expect("densmap should compute original radii"),
                start_epoch: density_start_epoch(n_epochs, self.params.dens_frac),
                lambda: self.params.dens_lambda,
                var_shift: self.params.dens_var_shift,
            })
        } else {
            None
        };

        let directed =
            compute_membership_strengths(&knn_indices_trimmed, &knn_dists_trimmed, &sigmas, &rhos);
        let mut edges = symmetrize_fuzzy_graph(&directed, self.params.set_op_mix_ratio);
        if let Some(target) = target
            && target.weight > 0.0
        {
            edges = apply_categorical_target_intersection(&edges, target);
        }
        prune_edges(&mut edges, n_epochs);

        let mut embedding = initialize_embedding(
            data,
            self.params.n_components,
            &edges,
            self.params.init,
            self.params.random_seed,
        );

        optimize_layout_training(
            &mut embedding,
            &edges,
            n_epochs,
            a,
            b,
            self.params.learning_rate,
            self.params.negative_sample_rate,
            self.params.repulsion_strength,
            self.params.random_seed ^ 0x9E37_79B9_7F4A_7C15,
            density_state.as_ref(),
        );
        let radii_embedding = if compute_density_outputs {
            Some(embedding_radii_from_knn(
                &embedding,
                &knn_indices_trimmed,
                self.params.dens_var_shift,
            ))
        } else {
            None
        };
        Ok((embedding, sigmas, rhos, radii_original, radii_embedding))
    }

    fn fit_transform_with_knn_sparse_prevalidated(
        &mut self,
        data: SparseCsrMatrix,
        knn_indices: &[Vec<usize>],
        knn_dists: &[Vec<f32>],
    ) -> Result<Vec<Vec<f32>>, UmapError> {
        let n_features = data.n_cols();
        let (embedding, sigmas, rhos, _, _) =
            self.build_sparse_fit_artifacts(&data, knn_indices, knn_dists)?;
        self.store_sparse_fit_state(embedding.clone(), data, n_features, sigmas, rhos);
        Ok(embedding)
    }

    fn fit_sparse_csr_prevalidated(
        &mut self,
        data: SparseCsrMatrix,
        knn_indices: &[Vec<usize>],
        knn_dists: &[Vec<f32>],
    ) -> Result<(), UmapError> {
        let n_features = data.n_cols();
        let (embedding, sigmas, rhos, _, _) =
            self.build_sparse_fit_artifacts(&data, knn_indices, knn_dists)?;
        self.store_sparse_fit_state(embedding, data, n_features, sigmas, rhos);
        Ok(())
    }

    fn build_sparse_fit_artifacts(
        &mut self,
        data: &SparseCsrMatrix,
        knn_indices: &[Vec<usize>],
        knn_dists: &[Vec<f32>],
    ) -> Result<FitArtifacts, UmapError> {
        let n_samples = data.n_rows();
        let n_epochs = self
            .params
            .n_epochs
            .unwrap_or(if n_samples <= 10_000 { 500 } else { 200 });

        let (a, b) = find_ab_params(self.params.spread, self.params.min_dist);
        self.a = a;
        self.b = b;

        let (knn_indices_trimmed, knn_dists_trimmed) = validate_and_trim_precomputed_knn(
            knn_indices,
            knn_dists,
            n_samples,
            self.params.n_neighbors,
        )?;

        let (sigmas, rhos) = smooth_knn_dist(
            &knn_dists_trimmed,
            self.params.n_neighbors as f32,
            self.params.local_connectivity,
            DEFAULT_BANDWIDTH,
            knn_rows_include_self(&knn_indices_trimmed, &knn_dists_trimmed),
        );

        let directed =
            compute_membership_strengths(&knn_indices_trimmed, &knn_dists_trimmed, &sigmas, &rhos);
        let mut edges = symmetrize_fuzzy_graph(&directed, self.params.set_op_mix_ratio);
        prune_edges(&mut edges, n_epochs);

        let mut embedding = initialize_embedding_sparse(
            data,
            self.params.n_components,
            &edges,
            self.params.init,
            self.params.random_seed,
        );

        optimize_layout_training(
            &mut embedding,
            &edges,
            n_epochs,
            a,
            b,
            self.params.learning_rate,
            self.params.negative_sample_rate,
            self.params.repulsion_strength,
            self.params.random_seed ^ 0x9E37_79B9_7F4A_7C15,
            None,
        );

        Ok((embedding, sigmas, rhos, None, None))
    }

    #[allow(clippy::too_many_arguments)]
    fn store_dense_fit_state(
        &mut self,
        embedding: Vec<Vec<f32>>,
        training_data: DenseMatrix,
        n_features: usize,
        sigmas: Vec<f32>,
        rhos: Vec<f32>,
        radii_original: Option<Vec<f32>>,
        radii_embedding: Option<Vec<f32>>,
    ) {
        self.embedding = Some(embedding);
        self.training_data_dense = Some(training_data);
        self.training_data_sparse = None;
        self.n_features = Some(n_features);
        self.fit_sigmas = Some(sigmas);
        self.fit_rhos = Some(rhos);
        self.fit_radii_original = radii_original;
        self.fit_radii_embedding = radii_embedding;
    }

    fn store_sparse_fit_state(
        &mut self,
        embedding: Vec<Vec<f32>>,
        training_data_sparse: SparseCsrMatrix,
        n_features: usize,
        sigmas: Vec<f32>,
        rhos: Vec<f32>,
    ) {
        self.embedding = Some(embedding);
        self.training_data_dense = None;
        self.training_data_sparse = Some(training_data_sparse);
        self.n_features = Some(n_features);
        self.fit_sigmas = Some(sigmas);
        self.fit_rhos = Some(rhos);
        self.fit_radii_original = None;
        self.fit_radii_embedding = None;
    }

    pub fn transform(&self, query: &[Vec<f32>]) -> Result<Vec<Vec<f32>>, UmapError> {
        self.transform_rows(query)
    }

    pub fn transform_dense(
        &self,
        query: &[f32],
        n_rows: usize,
        n_cols: usize,
    ) -> Result<DenseMatrix, UmapError> {
        let query_view = DenseMatrixView::new(query, n_rows, n_cols)?;
        let out = self.transform_rows(&query_view)?;
        dense_matrix_from_output_rows(out, self.params.n_components)
    }

    fn transform_rows<Q: RowMatrix + Sync + ?Sized>(
        &self,
        query: &Q,
    ) -> Result<Vec<Vec<f32>>, UmapError> {
        let train_embedding = self.embedding.as_ref().ok_or(UmapError::NotFitted)?;
        let expected_features = self.n_features.ok_or(UmapError::NotFitted)?;
        let train_data_dense = self.training_data_dense.as_ref();
        let train_data_sparse = self.training_data_sparse.as_ref();
        if train_data_dense.is_none() && train_data_sparse.is_none() {
            return Err(UmapError::NotFitted);
        }

        if query.n_rows() == 0 {
            return Ok(Vec::new());
        }

        for idx in 0..query.n_rows() {
            let row = query.row(idx);
            if row.len() != expected_features {
                if idx == 0 {
                    return Err(UmapError::FeatureMismatch {
                        expected: expected_features,
                        got: row.len(),
                    });
                }
                return Err(UmapError::InconsistentDimensions {
                    row: idx,
                    expected: expected_features,
                    got: row.len(),
                });
            }
            if row.iter().any(|v| !v.is_finite()) {
                return Err(UmapError::InvalidParameter(format!(
                    "input contains non-finite value at row {idx}"
                )));
            }
        }

        let n_train = if let Some(train_data) = train_data_dense {
            train_data.n_rows()
        } else if let Some(train_data) = train_data_sparse {
            train_data.n_rows()
        } else {
            return Err(UmapError::NotFitted);
        };
        let n_neighbors = self.params.n_neighbors.min(n_train);
        let n_epochs = match self.params.n_epochs {
            None => {
                if query.n_rows() <= 10_000 {
                    100
                } else {
                    30
                }
            }
            Some(e) => e.max(1),
        };

        let (indices, dists) = if let Some(train_data) = train_data_dense {
            exact_nearest_neighbors_to_reference(query, train_data, n_neighbors, self.params.metric)
        } else if let Some(train_data) = train_data_sparse {
            let query_rows = (0..query.n_rows())
                .map(|idx| query.row(idx).to_vec())
                .collect::<Vec<Vec<f32>>>();
            sparse::exact_nearest_neighbors_dense_query(
                &query_rows,
                train_data,
                n_neighbors,
                self.params.metric,
            )?
        } else {
            return Err(UmapError::NotFitted);
        };

        Ok(self.transform_from_knn(
            query.n_rows(),
            train_embedding,
            indices,
            dists,
            n_neighbors,
            n_epochs,
        ))
    }

    pub fn transform_sparse_csr(&self, query: SparseCsrMatrix) -> Result<Vec<Vec<f32>>, UmapError> {
        let train_embedding = self.embedding.as_ref().ok_or(UmapError::NotFitted)?;
        let expected_features = self.n_features.ok_or(UmapError::NotFitted)?;
        if query.n_cols() != expected_features {
            return Err(UmapError::FeatureMismatch {
                expected: expected_features,
                got: query.n_cols(),
            });
        }
        let train_data = self.training_data_sparse.as_ref().ok_or_else(|| {
            if self.training_data_dense.is_some() {
                UmapError::InvalidParameter(
                    "sparse query transform is supported only for sparse-fitted models".to_string(),
                )
            } else {
                UmapError::NotFitted
            }
        })?;

        if query.n_rows() == 0 {
            return Ok(Vec::new());
        }

        let n_neighbors = self.params.n_neighbors.min(train_data.n_rows());
        let n_epochs = match self.params.n_epochs {
            None => {
                if query.n_rows() <= 10_000 {
                    100
                } else {
                    30
                }
            }
            Some(e) => e.max(1),
        };
        let (indices, dists) = sparse::exact_nearest_neighbors_sparse_query(
            &query,
            train_data,
            n_neighbors,
            self.params.metric,
        )?;

        Ok(self.transform_from_knn(
            query.n_rows(),
            train_embedding,
            indices,
            dists,
            n_neighbors,
            n_epochs,
        ))
    }

    fn transform_from_knn(
        &self,
        query_n_rows: usize,
        train_embedding: &[Vec<f32>],
        indices: Vec<Vec<usize>>,
        dists: Vec<Vec<f32>>,
        n_neighbors: usize,
        n_epochs: usize,
    ) -> Vec<Vec<f32>> {
        let adjusted_local_connectivity = (self.params.local_connectivity - 1.0).max(0.0);
        let (sigmas, rhos) = smooth_knn_dist(
            &dists,
            n_neighbors as f32,
            adjusted_local_connectivity,
            DEFAULT_BANDWIDTH,
            false,
        );

        let mut edges = compute_membership_strengths_bipartite(&indices, &dists, &sigmas, &rhos);
        prune_edges(&mut edges, n_epochs);

        let mut embedding = init_graph_transform(
            query_n_rows,
            self.params.n_components,
            &edges,
            train_embedding,
        );

        optimize_layout_transform(
            &mut embedding,
            train_embedding,
            &edges,
            n_epochs,
            self.a,
            self.b,
            self.params.learning_rate / 4.0,
            self.params.negative_sample_rate,
            self.params.repulsion_strength,
            self.params.random_seed ^ 0xA24B_AED4_0B53_C117,
        );

        embedding
    }

    pub fn inverse_transform(
        &self,
        embedded_query: &[Vec<f32>],
    ) -> Result<Vec<Vec<f32>>, UmapError> {
        self.inverse_transform_rows(embedded_query)
    }

    pub fn inverse_transform_dense(
        &self,
        embedded_query: &[f32],
        n_rows: usize,
        n_cols: usize,
    ) -> Result<DenseMatrix, UmapError> {
        let query_view = DenseMatrixView::new(embedded_query, n_rows, n_cols)?;
        let out = self.inverse_transform_rows(&query_view)?;
        dense_matrix_from_output_rows(out, self.n_features.unwrap_or(0))
    }

    fn inverse_transform_rows<Q: RowMatrix + Sync + ?Sized>(
        &self,
        embedded_query: &Q,
    ) -> Result<Vec<Vec<f32>>, UmapError> {
        let train_data = if let Some(train_data) = self.training_data_dense.as_ref() {
            train_data
        } else if self.training_data_sparse.is_some() {
            return Err(UmapError::InvalidParameter(
                "inverse_transform is not supported for sparse-trained models yet".to_string(),
            ));
        } else {
            return Err(UmapError::NotFitted);
        };
        let train_embedding = self.embedding.as_ref().ok_or(UmapError::NotFitted)?;
        let fit_sigmas = self.fit_sigmas.as_ref().ok_or(UmapError::NotFitted)?;
        let fit_rhos = self.fit_rhos.as_ref().ok_or(UmapError::NotFitted)?;
        self.n_features.ok_or(UmapError::NotFitted)?;

        if embedded_query.n_rows() == 0 {
            return Ok(Vec::new());
        }

        for idx in 0..embedded_query.n_rows() {
            let row = embedded_query.row(idx);
            if row.len() != self.params.n_components {
                if idx == 0 {
                    return Err(UmapError::EmbeddingDimensionMismatch {
                        expected: self.params.n_components,
                        got: row.len(),
                    });
                }
                return Err(UmapError::InconsistentDimensions {
                    row: idx,
                    expected: self.params.n_components,
                    got: row.len(),
                });
            }
            if row.iter().any(|v| !v.is_finite()) {
                return Err(UmapError::InvalidParameter(format!(
                    "input contains non-finite value at row {idx}"
                )));
            }
        }

        let n_train = train_data.n_rows();
        let n_neighbors = inverse_neighbor_count(self.params.n_neighbors, n_train);
        let n_epochs = match self.params.n_epochs {
            None => {
                if embedded_query.n_rows() <= 10_000 {
                    100
                } else {
                    30
                }
            }
            Some(e) => (e / 3).max(1),
        };

        let (indices, dists) = exact_nearest_neighbors_to_reference(
            embedded_query,
            train_embedding,
            n_neighbors,
            Metric::Euclidean,
        );

        let mut graph_edges = Vec::with_capacity(embedded_query.n_rows() * n_neighbors);
        let mut init_weights = vec![vec![0.0_f32; n_neighbors]; embedded_query.n_rows()];
        let mut fixed_rows = vec![false; embedded_query.n_rows()];
        for i in 0..embedded_query.n_rows() {
            let mut row_sum = 0.0_f32;
            for j in 0..n_neighbors {
                let tail = indices[i][j];
                let dist = dists[i][j];
                let weight = 1.0 / (1.0 + self.a * dist.powf(2.0 * self.b));
                if weight > 0.0 && weight.is_finite() {
                    init_weights[i][j] = weight;
                    row_sum += weight;
                    graph_edges.push(Edge {
                        head: i,
                        tail,
                        weight,
                    });
                }
            }
            if row_sum > 0.0 {
                for w in init_weights[i].iter_mut() {
                    *w /= row_sum;
                }
            }
            fixed_rows[i] = dists[i][0] <= 1e-6;
        }

        let mut inv_points = init_inverse_points(
            embedded_query,
            &indices,
            &init_weights,
            &fixed_rows,
            train_embedding,
            train_data,
        );

        optimize_layout_inverse(
            &mut inv_points,
            train_data,
            &graph_edges,
            &fixed_rows,
            fit_sigmas,
            fit_rhos,
            n_epochs,
            self.params.learning_rate / 4.0,
            self.params.negative_sample_rate,
            self.params.repulsion_strength,
            self.params.random_seed ^ 0xD1B5_4A32_D192_ED03,
        );

        Ok(inv_points)
    }
}

pub fn fit_transform(data: &[Vec<f32>], params: UmapParams) -> Result<Vec<Vec<f32>>, UmapError> {
    let mut model = UmapModel::new(params);
    let (n_samples, n_features) = validate_data(data)?;
    validate_params(&model.params, n_samples, n_features)?;
    let (knn_indices, knn_dists) = build_fit_knn(&model.params, data, n_samples, n_features);
    let (embedding, _, _, _, _) =
        model.build_dense_fit_artifacts(data, &knn_indices, &knn_dists)?;
    Ok(embedding)
}

pub fn fit_transform_dense(
    data: &[f32],
    n_rows: usize,
    n_cols: usize,
    params: UmapParams,
) -> Result<DenseMatrix, UmapError> {
    let mut model = UmapModel::new(params);
    let data_view = DenseMatrixView::new(data, n_rows, n_cols)?;
    let (n_samples, n_features) = validate_data(&data_view)?;
    validate_params(&model.params, n_samples, n_features)?;
    let (knn_indices, knn_dists) = build_fit_knn(&model.params, &data_view, n_samples, n_features);
    let (embedding, _, _, _, _) =
        model.build_dense_fit_artifacts(&data_view, &knn_indices, &knn_dists)?;
    dense_matrix_from_output_rows(embedding, model.params.n_components)
}

pub fn fit_transform_sparse_csr(
    data: SparseCsrMatrix,
    params: UmapParams,
) -> Result<Vec<Vec<f32>>, UmapError> {
    let mut model = UmapModel::new(params);
    let n_samples = data.n_rows();
    let n_features = data.n_cols();
    if n_samples == 0 {
        return Err(UmapError::EmptyData);
    }
    if n_samples < 2 {
        return Err(UmapError::NeedAtLeastTwoSamples);
    }

    validate_params(&model.params, n_samples, n_features)?;
    validate_sparse_metric(model.params.metric)?;
    validate_sparse_density_params(&model.params)?;
    let (knn_indices, knn_dists) =
        sparse::exact_nearest_neighbors(&data, model.params.n_neighbors, model.params.metric);
    let (embedding, _, _, _, _) =
        model.build_sparse_fit_artifacts(&data, &knn_indices, &knn_dists)?;
    Ok(embedding)
}

fn validate_and_trim_precomputed_knn<I, D>(
    knn_indices: &I,
    knn_dists: &D,
    n_samples: usize,
    n_neighbors: usize,
) -> Result<KnnRows, UmapError>
where
    I: IndexRowMatrix + ?Sized,
    D: RowMatrix + ?Sized,
{
    if knn_indices.n_rows() != n_samples || knn_dists.n_rows() != n_samples {
        return Err(UmapError::InvalidParameter(
            "precomputed knn row count must match number of samples".to_string(),
        ));
    }

    let mut idx_trimmed = Vec::with_capacity(n_samples);
    let mut dist_trimmed = Vec::with_capacity(n_samples);
    let mut seen = HashSet::with_capacity(n_neighbors * 2);

    for row_idx in 0..n_samples {
        let idx_row = knn_indices.row(row_idx);
        let dist_row = knn_dists.row(row_idx);

        if idx_row.len() < n_neighbors || dist_row.len() < n_neighbors {
            return Err(UmapError::InvalidParameter(
                "precomputed knn columns must be >= n_neighbors".to_string(),
            ));
        }
        if idx_row.len() != dist_row.len() {
            return Err(UmapError::InvalidParameter(
                "precomputed knn index/dist row lengths must match".to_string(),
            ));
        }

        let idx_row = &idx_row[..n_neighbors];
        let dist_row = &dist_row[..n_neighbors];
        seen.clear();
        let mut prev = f32::NEG_INFINITY;

        for col_idx in 0..n_neighbors {
            let idx = idx_row[col_idx];
            let dist = dist_row[col_idx];
            if idx >= n_samples {
                return Err(UmapError::InvalidParameter(format!(
                    "precomputed knn index out of range at row {row_idx}, col {col_idx}"
                )));
            }
            if !dist.is_finite() || dist < 0.0 {
                return Err(UmapError::InvalidParameter(format!(
                    "precomputed knn distance must be finite and >= 0 at row {row_idx}, col {col_idx}"
                )));
            }
            if !seen.insert(idx) {
                return Err(UmapError::InvalidParameter(format!(
                    "precomputed knn row {row_idx} contains duplicate index {idx}"
                )));
            }
            if col_idx > 0 && dist + SMOOTH_K_TOLERANCE < prev {
                return Err(UmapError::InvalidParameter(format!(
                    "precomputed knn distances must be non-decreasing within each row; row {row_idx} has dist[{col_idx}]={dist} < dist[{}]={prev}",
                    col_idx - 1
                )));
            }
            prev = dist;
        }

        idx_trimmed.push(idx_row.to_vec());
        dist_trimmed.push(dist_row.to_vec());
    }

    Ok((idx_trimmed, dist_trimmed))
}

fn validate_precomputed_array_shapes(
    data_n_rows: usize,
    knn_indices_rows: usize,
    knn_indices_cols: usize,
    knn_dists_rows: usize,
    knn_dists_cols: usize,
    n_neighbors: usize,
) -> Result<(), UmapError> {
    if knn_indices_rows != knn_dists_rows || knn_indices_cols != knn_dists_cols {
        return Err(UmapError::InvalidParameter(
            "knn_indices and knn_dists must have identical shapes".to_string(),
        ));
    }
    if knn_indices_rows != data_n_rows {
        return Err(UmapError::InvalidParameter(
            "knn row count must match data row count".to_string(),
        ));
    }
    if knn_indices_cols < n_neighbors {
        return Err(UmapError::InvalidParameter(format!(
            "knn columns must be >= n_neighbors ({n_neighbors})"
        )));
    }
    Ok(())
}

fn validate_precomputed_distance_values_flat(knn_dists: &[f32]) -> Result<(), UmapError> {
    for &dist in knn_dists {
        if !dist.is_finite() {
            return Err(UmapError::InvalidParameter(
                "knn_dists must contain only finite values".to_string(),
            ));
        }
        if dist < 0.0 {
            return Err(UmapError::InvalidParameter(
                "knn_dists must be non-negative".to_string(),
            ));
        }
    }
    Ok(())
}

fn precomputed_knn_indices_from_i64_flat(
    knn_indices: &[i64],
    n_rows: usize,
    n_cols: usize,
) -> Result<Vec<Vec<usize>>, UmapError> {
    let expected_len = n_rows.saturating_mul(n_cols);
    if knn_indices.len() != expected_len {
        return Err(UmapError::InvalidParameter(format!(
            "flat knn_indices length mismatch: expected {expected_len}, got {}",
            knn_indices.len()
        )));
    }

    let mut rows = Vec::with_capacity(n_rows);
    for row in knn_indices.chunks_exact(n_cols) {
        let mut out_row = Vec::with_capacity(n_cols);
        for &idx in row {
            if idx < 0 {
                return Err(UmapError::InvalidParameter(
                    "knn indices must be non-negative integers".to_string(),
                ));
            }
            out_row.push(idx as usize);
        }
        rows.push(out_row);
    }
    Ok(rows)
}

fn validate_data<M: RowMatrix + ?Sized>(data: &M) -> Result<(usize, usize), UmapError> {
    if data.n_rows() == 0 {
        return Err(UmapError::EmptyData);
    }
    if data.n_rows() < 2 {
        return Err(UmapError::NeedAtLeastTwoSamples);
    }

    let n_features = data.n_cols();
    if n_features == 0 {
        return Err(UmapError::InvalidParameter(
            "input rows must have at least one feature".to_string(),
        ));
    }

    for row_idx in 0..data.n_rows() {
        let row = data.row(row_idx);
        if row.iter().any(|v| !v.is_finite()) {
            return Err(UmapError::InvalidParameter(format!(
                "input contains non-finite value at row {row_idx}"
            )));
        }
    }

    Ok((data.n_rows(), n_features))
}

fn validate_params(
    params: &UmapParams,
    n_samples: usize,
    _n_features: usize,
) -> Result<(), UmapError> {
    validate_metric(params.metric)?;
    if params.n_neighbors < 2 {
        return Err(UmapError::InvalidParameter(
            "n_neighbors must be >= 2".to_string(),
        ));
    }
    if params.n_neighbors >= n_samples {
        return Err(UmapError::InvalidParameter(format!(
            "n_neighbors ({}) must be smaller than number of samples ({n_samples})",
            params.n_neighbors
        )));
    }
    if params.n_components == 0 {
        return Err(UmapError::InvalidParameter(
            "n_components must be >= 1".to_string(),
        ));
    }
    if !params.learning_rate.is_finite() || params.learning_rate <= 0.0 {
        return Err(UmapError::InvalidParameter(
            "learning_rate must be finite and > 0".to_string(),
        ));
    }
    if !params.min_dist.is_finite() || params.min_dist < 0.0 {
        return Err(UmapError::InvalidParameter(
            "min_dist must be finite and >= 0".to_string(),
        ));
    }
    if !params.spread.is_finite() || params.spread <= 0.0 {
        return Err(UmapError::InvalidParameter(
            "spread must be finite and > 0".to_string(),
        ));
    }
    if params.min_dist > params.spread {
        return Err(UmapError::InvalidParameter(
            "min_dist must be <= spread".to_string(),
        ));
    }
    if !params.set_op_mix_ratio.is_finite() || !(0.0..=1.0).contains(&params.set_op_mix_ratio) {
        return Err(UmapError::InvalidParameter(
            "set_op_mix_ratio must be finite and in [0, 1]".to_string(),
        ));
    }
    if !params.repulsion_strength.is_finite() || params.repulsion_strength < 0.0 {
        return Err(UmapError::InvalidParameter(
            "repulsion_strength must be finite and >= 0".to_string(),
        ));
    }
    if params.negative_sample_rate == 0 {
        return Err(UmapError::InvalidParameter(
            "negative_sample_rate must be >= 1".to_string(),
        ));
    }
    if !params.local_connectivity.is_finite() || params.local_connectivity < 0.0 {
        return Err(UmapError::InvalidParameter(
            "local_connectivity must be finite and >= 0".to_string(),
        ));
    }
    if params.approx_knn_candidates == 0 {
        return Err(UmapError::InvalidParameter(
            "approx_knn_candidates must be >= 1".to_string(),
        ));
    }
    if params.approx_knn_iters == 0 {
        return Err(UmapError::InvalidParameter(
            "approx_knn_iters must be >= 1".to_string(),
        ));
    }
    if !params.dens_lambda.is_finite() || params.dens_lambda < 0.0 {
        return Err(UmapError::InvalidParameter(
            "dens_lambda must be finite and >= 0".to_string(),
        ));
    }
    if !params.dens_frac.is_finite() || !(0.0..=1.0).contains(&params.dens_frac) {
        return Err(UmapError::InvalidParameter(
            "dens_frac must be finite and in [0, 1]".to_string(),
        ));
    }
    if !params.dens_var_shift.is_finite() || params.dens_var_shift <= 0.0 {
        return Err(UmapError::InvalidParameter(
            "dens_var_shift must be finite and > 0".to_string(),
        ));
    }
    Ok(())
}

fn validate_metric(metric: Metric) -> Result<(), UmapError> {
    if let Metric::Minkowski { p } = metric
        && (!p.is_finite() || p <= 0.0)
    {
        return Err(UmapError::InvalidParameter(
            "minkowski metric requires finite p > 0".to_string(),
        ));
    }
    Ok(())
}

fn validate_sparse_metric(metric: Metric) -> Result<(), UmapError> {
    validate_metric(metric)?;
    if matches!(metric, Metric::Canberra | Metric::BrayCurtis) {
        return Err(UmapError::InvalidParameter(format!(
            "{metric:?} metric is currently supported for dense input only"
        )));
    }
    Ok(())
}

fn validate_sparse_density_params(params: &UmapParams) -> Result<(), UmapError> {
    if params.densmap || params.output_dens {
        return Err(UmapError::InvalidParameter(
            "densmap and output_dens are currently supported for dense input only".to_string(),
        ));
    }
    Ok(())
}

fn validate_supervised_target(
    target: &SupervisedTarget,
    n_samples: usize,
) -> Result<(), UmapError> {
    if target.labels.len() != n_samples {
        return Err(UmapError::InvalidParameter(format!(
            "target label length ({}) must match number of samples ({n_samples})",
            target.labels.len()
        )));
    }
    if !target.weight.is_finite() || !(0.0..=1.0).contains(&target.weight) {
        return Err(UmapError::InvalidParameter(
            "target_weight must be finite and in [0, 1]".to_string(),
        ));
    }
    if let Some(n_neighbors) = target.n_neighbors
        && n_neighbors == 0
    {
        return Err(UmapError::InvalidParameter(
            "target_n_neighbors must be >= 1 when provided".to_string(),
        ));
    }
    Ok(())
}

fn should_use_approximate_knn(params: &UmapParams, n_samples: usize, n_features: usize) -> bool {
    if !params.use_approximate_knn || n_samples <= params.approx_knn_threshold {
        return false;
    }

    // The in-crate ANN implementation is a lightweight NN-descent variant. It
    // is now parallel, but still lacks PyNNDescent's RP-forest initialization
    // and full candidate-update machinery. Auto mode therefore waits past the
    // public threshold before switching; low-dimensional data gets a larger
    // exact window because the exact rayon path remains competitive longer.
    if params.approx_knn_threshold > 0 {
        let exact_limit = if n_features <= AUTO_EXACT_LOW_DIM_MAX_FEATURES {
            params
                .approx_knn_threshold
                .saturating_mul(AUTO_EXACT_LOW_DIM_SAMPLE_MULTIPLIER)
        } else {
            params
                .approx_knn_threshold
                .saturating_mul(AUTO_EXACT_GENERAL_SAMPLE_MULTIPLIER)
        };
        if n_samples <= exact_limit {
            return false;
        }
    }

    true
}

fn build_fit_knn<M: RowMatrix + Sync + ?Sized>(
    params: &UmapParams,
    data: &M,
    n_samples: usize,
    n_features: usize,
) -> KnnRows {
    if should_use_approximate_knn(params, n_samples, n_features) {
        approximate_nearest_neighbors(
            data,
            params.n_neighbors,
            params.metric,
            params.approx_knn_candidates,
            params.approx_knn_iters,
            params.random_seed ^ 0xC0FE_FEED_1234_ABCD,
        )
    } else {
        exact_nearest_neighbors(data, params.n_neighbors, params.metric)
    }
}

fn row_knn_recall_excluding_self(row_idx: usize, exact_row: &[usize], approx_row: &[usize]) -> f32 {
    let exact_set = exact_row
        .iter()
        .copied()
        .filter(|&idx| idx != row_idx)
        .collect::<HashSet<usize>>();
    if exact_set.is_empty() {
        return 1.0;
    }

    let matched = approx_row
        .iter()
        .copied()
        .filter(|&idx| idx != row_idx && exact_set.contains(&idx))
        .count();
    matched as f32 / exact_set.len() as f32
}

fn summarize_knn_recall(
    exact_indices: &[Vec<usize>],
    approx_indices: &[Vec<usize>],
) -> (f32, f32, f32, Vec<f32>) {
    debug_assert_eq!(exact_indices.len(), approx_indices.len());
    let per_row_recall = exact_indices
        .iter()
        .zip(approx_indices.iter())
        .enumerate()
        .map(|(row_idx, (exact_row, approx_row))| {
            row_knn_recall_excluding_self(row_idx, exact_row, approx_row)
        })
        .collect::<Vec<f32>>();

    if per_row_recall.is_empty() {
        return (1.0, 1.0, 1.0, per_row_recall);
    }

    let mean_recall = per_row_recall.iter().sum::<f32>() / per_row_recall.len() as f32;
    let min_recall = per_row_recall.iter().copied().fold(f32::INFINITY, f32::min);
    let mut sorted = per_row_recall.clone();
    sorted.sort_by(|lhs, rhs| lhs.total_cmp(rhs));
    let worst_count = sorted.len().div_ceil(10).max(1);
    let worst_decile_recall = sorted[..worst_count].iter().sum::<f32>() / worst_count as f32;

    (mean_recall, min_recall, worst_decile_recall, per_row_recall)
}

fn approximate_knn_recall_diagnostics_rows<M: RowMatrix + Sync + ?Sized>(
    data: &M,
    n_samples: usize,
    n_features: usize,
    params: &UmapParams,
) -> Result<KnnRecallDiagnostics, UmapError> {
    validate_params(params, n_samples, n_features)?;
    let n_neighbors = params.n_neighbors.min(n_samples);
    let candidate_pool = params
        .approx_knn_candidates
        .max(n_neighbors)
        .min(n_samples - 1);
    let seed = params.random_seed ^ 0xC0FE_FEED_1234_ABCD;

    let (exact_indices, _) = exact_nearest_neighbors(data, n_neighbors, params.metric);
    let (approx_indices, _) = approximate_nearest_neighbors(
        data,
        n_neighbors,
        params.metric,
        params.approx_knn_candidates,
        params.approx_knn_iters,
        seed,
    );
    let (mean_recall, min_recall, worst_decile_recall, per_row_recall) =
        summarize_knn_recall(&exact_indices, &approx_indices);

    Ok(KnnRecallDiagnostics {
        n_samples,
        n_features,
        n_neighbors,
        metric: params.metric,
        candidate_pool,
        n_iters: params.approx_knn_iters,
        seed,
        mean_recall,
        min_recall,
        worst_decile_recall,
        per_row_recall,
    })
}

pub fn approximate_knn_recall_diagnostics(
    data: &[Vec<f32>],
    params: UmapParams,
) -> Result<KnnRecallDiagnostics, UmapError> {
    let (n_samples, n_features) = validate_data(data)?;
    approximate_knn_recall_diagnostics_rows(data, n_samples, n_features, &params)
}

pub fn approximate_knn_recall_diagnostics_dense(
    data: &[f32],
    n_rows: usize,
    n_cols: usize,
    params: UmapParams,
) -> Result<KnnRecallDiagnostics, UmapError> {
    let data_view = DenseMatrixView::new(data, n_rows, n_cols)?;
    let (n_samples, n_features) = validate_data(&data_view)?;
    approximate_knn_recall_diagnostics_rows(&data_view, n_samples, n_features, &params)
}

#[inline(always)]
fn euclidean_distance(x: &[f32], y: &[f32]) -> f32 {
    squared_distance(x, y).sqrt()
}

#[inline(always)]
fn squared_distance(x: &[f32], y: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if x.len() >= 64 && std::is_x86_feature_detected!("avx2") {
            // SAFETY: guarded by runtime AVX2 feature detection above.
            return unsafe { squared_distance_avx2(x, y) };
        }
    }
    squared_distance_scalar(x, y)
}

#[inline(always)]
fn squared_distance_scalar(x: &[f32], y: &[f32]) -> f32 {
    let mut acc = 0.0_f32;
    let mut idx = 0;
    while idx < x.len() {
        let d = x[idx] - y[idx];
        acc += d * d;
        idx += 1;
    }
    acc
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn squared_distance_avx2(x: &[f32], y: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let mut acc = _mm256_setzero_ps();
    let mut idx = 0;
    while idx + 8 <= x.len() {
        let xv = unsafe { _mm256_loadu_ps(x.as_ptr().add(idx)) };
        let yv = unsafe { _mm256_loadu_ps(y.as_ptr().add(idx)) };
        let diff = _mm256_sub_ps(xv, yv);
        acc = _mm256_add_ps(acc, _mm256_mul_ps(diff, diff));
        idx += 8;
    }

    let mut lanes = [0.0_f32; 8];
    unsafe { _mm256_storeu_ps(lanes.as_mut_ptr(), acc) };
    let mut sum = lanes.iter().sum::<f32>();
    while idx < x.len() {
        let d = x[idx] - y[idx];
        sum += d * d;
        idx += 1;
    }
    sum
}

#[inline]
fn manhattan_distance(x: &[f32], y: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if x.len() >= 64 && std::is_x86_feature_detected!("avx2") {
            // SAFETY: guarded by runtime AVX2 feature detection above.
            return unsafe { manhattan_distance_avx2(x, y) };
        }
    }
    manhattan_distance_scalar(x, y)
}

#[inline]
fn manhattan_distance_scalar(x: &[f32], y: &[f32]) -> f32 {
    let mut acc = 0.0_f32;
    let mut idx = 0;
    while idx < x.len() {
        acc += (x[idx] - y[idx]).abs();
        idx += 1;
    }
    acc
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn manhattan_distance_avx2(x: &[f32], y: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let sign_mask = _mm256_set1_ps(-0.0);
    let mut acc = _mm256_setzero_ps();
    let mut idx = 0;
    while idx + 8 <= x.len() {
        let xv = unsafe { _mm256_loadu_ps(x.as_ptr().add(idx)) };
        let yv = unsafe { _mm256_loadu_ps(y.as_ptr().add(idx)) };
        let diff = _mm256_sub_ps(xv, yv);
        let abs = _mm256_andnot_ps(sign_mask, diff);
        acc = _mm256_add_ps(acc, abs);
        idx += 8;
    }

    let mut lanes = [0.0_f32; 8];
    unsafe { _mm256_storeu_ps(lanes.as_mut_ptr(), acc) };
    let mut sum = lanes.iter().sum::<f32>();
    while idx < x.len() {
        sum += (x[idx] - y[idx]).abs();
        idx += 1;
    }
    sum
}

#[inline]
fn chebyshev_distance(x: &[f32], y: &[f32]) -> f32 {
    let mut max_diff = 0.0_f32;
    let mut idx = 0;
    while idx < x.len() {
        max_diff = max_diff.max((x[idx] - y[idx]).abs());
        idx += 1;
    }
    max_diff
}

#[inline]
fn minkowski_distance(x: &[f32], y: &[f32], p: f32) -> f32 {
    let mut acc = 0.0_f32;
    let mut idx = 0;
    while idx < x.len() {
        acc += (x[idx] - y[idx]).abs().powf(p);
        idx += 1;
    }
    acc.powf(1.0 / p)
}

#[inline]
fn canberra_distance(x: &[f32], y: &[f32]) -> f32 {
    let mut acc = 0.0_f32;
    let mut idx = 0;
    while idx < x.len() {
        let denom = x[idx].abs() + y[idx].abs();
        if denom > 0.0 {
            acc += (x[idx] - y[idx]).abs() / denom;
        }
        idx += 1;
    }
    acc
}

#[inline]
fn bray_curtis_distance(x: &[f32], y: &[f32]) -> f32 {
    let mut numerator = 0.0_f32;
    let mut denominator = 0.0_f32;
    let mut idx = 0;
    while idx < x.len() {
        numerator += (x[idx] - y[idx]).abs();
        denominator += (x[idx] + y[idx]).abs();
        idx += 1;
    }
    if denominator == 0.0 {
        0.0
    } else {
        numerator / denominator
    }
}

#[allow(dead_code)]
#[inline]
fn cosine_distance(x: &[f32], y: &[f32]) -> f32 {
    cosine_distance_with_norms(x, y, l2_norm(x), l2_norm(y))
}

#[inline]
fn l2_norm(x: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if x.len() >= 64 && std::is_x86_feature_detected!("avx2") {
            // SAFETY: guarded by runtime AVX2 feature detection above.
            return unsafe { l2_norm_avx2(x) };
        }
    }
    l2_norm_scalar(x)
}

#[inline]
fn l2_norm_scalar(x: &[f32]) -> f32 {
    let mut acc = 0.0_f32;
    let mut idx = 0;
    while idx < x.len() {
        acc += x[idx] * x[idx];
        idx += 1;
    }
    acc.sqrt()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn l2_norm_avx2(x: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let mut acc = _mm256_setzero_ps();
    let mut idx = 0;
    while idx + 8 <= x.len() {
        let xv = unsafe { _mm256_loadu_ps(x.as_ptr().add(idx)) };
        acc = _mm256_add_ps(acc, _mm256_mul_ps(xv, xv));
        idx += 8;
    }

    let mut lanes = [0.0_f32; 8];
    unsafe { _mm256_storeu_ps(lanes.as_mut_ptr(), acc) };
    let mut sum = lanes.iter().sum::<f32>();
    while idx < x.len() {
        sum += x[idx] * x[idx];
        idx += 1;
    }
    sum.sqrt()
}

fn compute_l2_norms<M: RowMatrix + ?Sized>(data: &M) -> Vec<f32> {
    (0..data.n_rows())
        .map(|row_idx| l2_norm(data.row(row_idx)))
        .collect()
}

#[inline]
fn dot_product_scalar(x: &[f32], y: &[f32]) -> f32 {
    let mut dot = 0.0_f32;
    let mut idx = 0;
    while idx < x.len() {
        dot += x[idx] * y[idx];
        idx += 1;
    }
    dot
}

#[inline]
fn correlation_distance(x: &[f32], y: &[f32]) -> f32 {
    let mut sum_x = 0.0_f32;
    let mut sum_y = 0.0_f32;
    let mut sq_x = 0.0_f32;
    let mut sq_y = 0.0_f32;
    let mut dot = 0.0_f32;

    let mut idx = 0;
    while idx < x.len() {
        let xv = x[idx];
        let yv = y[idx];
        sum_x += xv;
        sum_y += yv;
        sq_x += xv * xv;
        sq_y += yv * yv;
        dot += xv * yv;
        idx += 1;
    }

    correlation_distance_from_summaries(x.len(), sum_x, sq_x, sum_y, sq_y, dot)
}

#[inline]
pub(crate) fn correlation_distance_from_summaries(
    n_features: usize,
    sum_x: f32,
    sq_x: f32,
    sum_y: f32,
    sq_y: f32,
    dot: f32,
) -> f32 {
    let n = n_features as f32;
    let centered_dot = dot - (sum_x * sum_y / n);
    let x_var = (sq_x - (sum_x * sum_x / n)).max(0.0);
    let y_var = (sq_y - (sum_y * sum_y / n)).max(0.0);
    let x_norm = x_var.sqrt();
    let y_norm = y_var.sqrt();

    if x_norm == 0.0 && y_norm == 0.0 {
        0.0
    } else if x_norm == 0.0 || y_norm == 0.0 {
        1.0
    } else {
        let corr = (centered_dot / (x_norm * y_norm)).clamp(-1.0, 1.0);
        1.0 - corr
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_product_avx2(x: &[f32], y: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let mut acc = _mm256_setzero_ps();
    let mut idx = 0;
    while idx + 8 <= x.len() {
        let xv = unsafe { _mm256_loadu_ps(x.as_ptr().add(idx)) };
        let yv = unsafe { _mm256_loadu_ps(y.as_ptr().add(idx)) };
        acc = _mm256_add_ps(acc, _mm256_mul_ps(xv, yv));
        idx += 8;
    }

    let mut lanes = [0.0_f32; 8];
    unsafe { _mm256_storeu_ps(lanes.as_mut_ptr(), acc) };
    let mut dot = lanes.iter().sum::<f32>();
    while idx < x.len() {
        dot += x[idx] * y[idx];
        idx += 1;
    }
    dot
}

#[inline]
fn cosine_distance_with_norms(x: &[f32], y: &[f32], x_norm: f32, y_norm: f32) -> f32 {
    let dot = {
        #[cfg(target_arch = "x86_64")]
        {
            if x.len() >= 64 && std::is_x86_feature_detected!("avx2") {
                // SAFETY: guarded by runtime AVX2 feature detection above.
                unsafe { dot_product_avx2(x, y) }
            } else {
                dot_product_scalar(x, y)
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            dot_product_scalar(x, y)
        }
    };

    if x_norm == 0.0 && y_norm == 0.0 {
        0.0
    } else if x_norm == 0.0 || y_norm == 0.0 {
        1.0
    } else {
        let cosine_sim = (dot / (x_norm * y_norm)).clamp(-1.0, 1.0);
        1.0 - cosine_sim
    }
}

#[inline(always)]
fn neighbor_cmp(lhs_idx: usize, lhs_dist: f32, rhs_idx: usize, rhs_dist: f32) -> Ordering {
    lhs_dist
        .total_cmp(&rhs_dist)
        .then_with(|| lhs_idx.cmp(&rhs_idx))
}

#[inline(always)]
fn push_top_k_neighbor(
    row_indices: &mut Vec<usize>,
    row_dists: &mut Vec<f32>,
    idx: usize,
    dist: f32,
    k: usize,
) {
    debug_assert_eq!(row_indices.len(), row_dists.len());
    if row_indices.len() == k
        && neighbor_cmp(idx, dist, row_indices[k - 1], row_dists[k - 1]) != Ordering::Less
    {
        return;
    }

    let mut insert_at = row_indices.len();
    while insert_at > 0
        && neighbor_cmp(
            idx,
            dist,
            row_indices[insert_at - 1],
            row_dists[insert_at - 1],
        ) == Ordering::Less
    {
        insert_at -= 1;
    }

    row_indices.insert(insert_at, idx);
    row_dists.insert(insert_at, dist);

    if row_indices.len() > k {
        row_indices.pop();
        row_dists.pop();
    }
}

fn exact_top_k_neighbors<F>(
    n_candidates: usize,
    n_neighbors: usize,
    mut distance_at: F,
) -> (Vec<usize>, Vec<f32>)
where
    F: FnMut(usize) -> f32,
{
    let mut row_indices = Vec::with_capacity(n_neighbors);
    let mut row_dists = Vec::with_capacity(n_neighbors);

    for idx in 0..n_candidates {
        let dist = distance_at(idx);
        push_top_k_neighbor(&mut row_indices, &mut row_dists, idx, dist, n_neighbors);
    }

    (row_indices, row_dists)
}

#[inline]
fn sqrt_row_dists(row_dists: &mut [f32]) {
    for dist in row_dists {
        *dist = dist.sqrt();
    }
}

fn exact_top_k_neighbors_euclidean<M: RowMatrix + ?Sized>(
    data: &M,
    row_idx: usize,
    n_candidates: usize,
    n_neighbors: usize,
) -> (Vec<usize>, Vec<f32>) {
    let x = data.row(row_idx);
    let (row_indices, mut row_dists) = exact_top_k_neighbors(n_candidates, n_neighbors, |j| {
        squared_distance(x, data.row(j))
    });
    sqrt_row_dists(&mut row_dists);
    (row_indices, row_dists)
}

fn exact_top_k_neighbors_to_reference_euclidean<R: RowMatrix + ?Sized>(
    x: &[f32],
    reference: &R,
    n_neighbors: usize,
) -> (Vec<usize>, Vec<f32>) {
    let (row_indices, mut row_dists) =
        exact_top_k_neighbors(reference.n_rows(), n_neighbors, |j| {
            squared_distance(x, reference.row(j))
        });
    sqrt_row_dists(&mut row_dists);
    (row_indices, row_dists)
}

fn exact_nearest_neighbors_euclidean<M: RowMatrix + Sync + ?Sized>(
    data: &M,
    n_neighbors: usize,
) -> (Vec<Vec<usize>>, Vec<Vec<f32>>) {
    let n_samples = data.n_rows();
    (0..n_samples)
        .into_par_iter()
        .map(|i| exact_top_k_neighbors_euclidean(data, i, n_samples, n_neighbors))
        .unzip()
}

fn exact_nearest_neighbors_cosine<M: RowMatrix + Sync + ?Sized>(
    data: &M,
    n_neighbors: usize,
) -> (Vec<Vec<usize>>, Vec<Vec<f32>>) {
    let n_samples = data.n_rows();
    let norms = compute_l2_norms(data);

    (0..n_samples)
        .into_par_iter()
        .map(|i| {
            exact_top_k_neighbors(n_samples, n_neighbors, |j| {
                cosine_distance_with_norms(data.row(i), data.row(j), norms[i], norms[j])
            })
        })
        .unzip()
}

fn exact_nearest_neighbors_by<M, F>(
    data: &M,
    n_neighbors: usize,
    distance_fn: F,
) -> (Vec<Vec<usize>>, Vec<Vec<f32>>)
where
    M: RowMatrix + Sync + ?Sized,
    F: Fn(&[f32], &[f32]) -> f32 + Copy + Sync + Send,
{
    let n_samples = data.n_rows();
    (0..n_samples)
        .into_par_iter()
        .map(|i| {
            exact_top_k_neighbors(n_samples, n_neighbors, |j| {
                distance_fn(data.row(i), data.row(j))
            })
        })
        .unzip()
}

fn exact_nearest_neighbors<M: RowMatrix + Sync + ?Sized>(
    data: &M,
    n_neighbors: usize,
    metric: Metric,
) -> (Vec<Vec<usize>>, Vec<Vec<f32>>) {
    match metric {
        Metric::Euclidean => exact_nearest_neighbors_euclidean(data, n_neighbors),
        Metric::Manhattan => exact_nearest_neighbors_by(data, n_neighbors, manhattan_distance),
        Metric::Cosine => exact_nearest_neighbors_cosine(data, n_neighbors),
        Metric::Chebyshev => exact_nearest_neighbors_by(data, n_neighbors, chebyshev_distance),
        Metric::Minkowski { p } => {
            exact_nearest_neighbors_by(data, n_neighbors, move |x, y| minkowski_distance(x, y, p))
        }
        Metric::Correlation => exact_nearest_neighbors_by(data, n_neighbors, correlation_distance),
        Metric::Canberra => exact_nearest_neighbors_by(data, n_neighbors, canberra_distance),
        Metric::BrayCurtis => exact_nearest_neighbors_by(data, n_neighbors, bray_curtis_distance),
    }
}

// Approximate kNN candidate builders below de-duplicate indices before distance
// evaluation, so the sorted top-k step can avoid another per-row HashSet.
fn sorted_unique_neighbors(mut pairs: Vec<(usize, f32)>, k: usize) -> Vec<(usize, f32)> {
    pairs.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
    let keep = pairs.len().min(k);
    let mut out = Vec::with_capacity(k);
    out.extend_from_slice(&pairs[..keep]);
    out
}

#[derive(Clone)]
struct FlatNeighbors {
    indices: Vec<usize>,
    dists: Vec<f32>,
    row_len: usize,
}

impl FlatNeighbors {
    fn new(n_rows: usize, row_len: usize) -> Self {
        Self {
            indices: vec![0; n_rows.saturating_mul(row_len)],
            dists: vec![0.0; n_rows.saturating_mul(row_len)],
            row_len,
        }
    }

    fn from_rows(rows: Vec<Vec<(usize, f32)>>, row_len: usize) -> Self {
        let mut out = Self::new(rows.len(), row_len);
        for (row_idx, row) in rows.iter().enumerate() {
            out.set_row(row_idx, row);
        }
        out
    }

    fn from_updates(rows: Vec<(Vec<(usize, f32)>, bool)>, row_len: usize) -> (Self, bool) {
        let changed = rows.iter().any(|(_, row_changed)| *row_changed);
        let mut out = Self::new(rows.len(), row_len);
        for (row_idx, (row, _)) in rows.iter().enumerate() {
            out.set_row(row_idx, row);
        }
        (out, changed)
    }

    #[inline]
    fn row_range(&self, row_idx: usize) -> std::ops::Range<usize> {
        let start = row_idx * self.row_len;
        start..start + self.row_len
    }

    #[inline]
    fn row_indices(&self, row_idx: usize) -> &[usize] {
        &self.indices[self.row_range(row_idx)]
    }

    fn set_row(&mut self, row_idx: usize, row: &[(usize, f32)]) {
        debug_assert_eq!(row.len(), self.row_len);
        let range = self.row_range(row_idx);
        for (offset, &(idx, dist)) in row.iter().enumerate() {
            let pos = range.start + offset;
            self.indices[pos] = idx;
            self.dists[pos] = dist;
        }
    }

    fn row_eq(&self, row_idx: usize, row: &[(usize, f32)]) -> bool {
        if row.len() != self.row_len {
            return false;
        }
        let range = self.row_range(row_idx);
        row.iter().enumerate().all(|(offset, &(idx, dist))| {
            let pos = range.start + offset;
            self.indices[pos] == idx && self.dists[pos] == dist
        })
    }

    fn into_knn_rows(self, n_rows: usize) -> KnnRows {
        let mut indices = Vec::with_capacity(n_rows);
        let mut dists = Vec::with_capacity(n_rows);
        for row_idx in 0..n_rows {
            let range = self.row_range(row_idx);
            indices.push(self.indices[range.clone()].to_vec());
            dists.push(self.dists[range].to_vec());
        }
        (indices, dists)
    }
}

fn mix_approx_seed(seed: u64, row_idx: usize, iteration: usize) -> u64 {
    let mut value = seed
        ^ (row_idx as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ (iteration as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

fn push_unique_candidate(out: &mut Vec<usize>, idx: usize) -> bool {
    if out.contains(&idx) {
        false
    } else {
        out.push(idx);
        true
    }
}

fn sample_unique_candidates<R: Rng + ?Sized>(
    rng: &mut R,
    n_samples: usize,
    row_idx: usize,
    target_len: usize,
    out: &mut Vec<usize>,
) {
    out.clear();
    out.push(row_idx);
    while out.len() < target_len {
        let idx = rng.gen_range(0..n_samples);
        push_unique_candidate(out, idx);
    }
}

#[allow(clippy::too_many_arguments)]
fn collect_approx_candidate_indices<R: Rng + ?Sized>(
    neighbors: &FlatNeighbors,
    row_idx: usize,
    n_samples: usize,
    n_neighbors: usize,
    max_candidates: usize,
    rng: &mut R,
    out: &mut Vec<usize>,
) {
    out.clear();

    for &idx in neighbors.row_indices(row_idx) {
        push_unique_candidate(out, idx);
        for &idx2 in neighbors.row_indices(idx) {
            push_unique_candidate(out, idx2);
        }
    }
    push_unique_candidate(out, row_idx);

    let exploration = (n_neighbors / 2).max(4).min(max_candidates);
    for _ in 0..exploration {
        let idx = rng.gen_range(0..n_samples);
        push_unique_candidate(out, idx);
    }

    out.sort_unstable();
    if out.len() > max_candidates {
        for s in 0..max_candidates {
            let r = rng.gen_range(s..out.len());
            out.swap(s, r);
        }
        out.truncate(max_candidates);
    }
}

fn approximate_nearest_neighbors_with_distance<F>(
    n_samples: usize,
    n_neighbors: usize,
    candidate_pool: usize,
    n_iters: usize,
    seed: u64,
    distance_fn: F,
) -> (Vec<Vec<usize>>, Vec<Vec<f32>>)
where
    F: Fn(usize, usize) -> f32 + Copy + Sync + Send,
{
    let pool = candidate_pool.max(n_neighbors).min(n_samples - 1);
    let max_candidates = candidate_pool.max(n_neighbors) * 4 + 1;

    let initial_rows = (0..n_samples)
        .into_par_iter()
        .map(|i| {
            let mut rng = SmallRng::seed_from_u64(mix_approx_seed(seed, i, 0));
            let mut candidate_indices = Vec::with_capacity(max_candidates.max(pool + 1));
            sample_unique_candidates(&mut rng, n_samples, i, pool + 1, &mut candidate_indices);

            let candidates = candidate_indices
                .iter()
                .map(|&j| (j, distance_fn(i, j)))
                .collect::<Vec<(usize, f32)>>();
            sorted_unique_neighbors(candidates, n_neighbors)
        })
        .collect::<Vec<Vec<(usize, f32)>>>();

    let mut neighbors = FlatNeighbors::from_rows(initial_rows, n_neighbors);

    for iteration in 0..n_iters {
        let updates = (0..n_samples)
            .into_par_iter()
            .map(|i| {
                let mut rng = SmallRng::seed_from_u64(mix_approx_seed(seed, i, iteration + 1));
                let mut candidate_indices = Vec::with_capacity(max_candidates.max(pool + 1));
                collect_approx_candidate_indices(
                    &neighbors,
                    i,
                    n_samples,
                    n_neighbors,
                    max_candidates,
                    &mut rng,
                    &mut candidate_indices,
                );

                let candidate_pairs = candidate_indices
                    .iter()
                    .map(|&j| (j, distance_fn(i, j)))
                    .collect::<Vec<(usize, f32)>>();
                let updated = sorted_unique_neighbors(candidate_pairs, n_neighbors);
                let changed = !neighbors.row_eq(i, &updated);
                (updated, changed)
            })
            .collect::<Vec<(Vec<(usize, f32)>, bool)>>();

        let (next_neighbors, changed) = FlatNeighbors::from_updates(updates, n_neighbors);
        neighbors = next_neighbors;
        if !changed {
            break;
        }
    }

    neighbors.into_knn_rows(n_samples)
}

fn approximate_nearest_neighbors_euclidean<M: RowMatrix + Sync + ?Sized>(
    data: &M,
    n_neighbors: usize,
    candidate_pool: usize,
    n_iters: usize,
    seed: u64,
) -> (Vec<Vec<usize>>, Vec<Vec<f32>>) {
    let n_samples = data.n_rows();
    approximate_nearest_neighbors_with_distance(
        n_samples,
        n_neighbors,
        candidate_pool,
        n_iters,
        seed,
        |i, j| euclidean_distance(data.row(i), data.row(j)),
    )
}

fn approximate_nearest_neighbors_by<M, F>(
    data: &M,
    n_neighbors: usize,
    candidate_pool: usize,
    n_iters: usize,
    seed: u64,
    distance_fn: F,
) -> (Vec<Vec<usize>>, Vec<Vec<f32>>)
where
    M: RowMatrix + Sync + ?Sized,
    F: Fn(&[f32], &[f32]) -> f32 + Copy + Sync + Send,
{
    let n_samples = data.n_rows();
    approximate_nearest_neighbors_with_distance(
        n_samples,
        n_neighbors,
        candidate_pool,
        n_iters,
        seed,
        |i, j| distance_fn(data.row(i), data.row(j)),
    )
}

fn approximate_nearest_neighbors_cosine<M: RowMatrix + Sync + ?Sized>(
    data: &M,
    n_neighbors: usize,
    candidate_pool: usize,
    n_iters: usize,
    seed: u64,
) -> (Vec<Vec<usize>>, Vec<Vec<f32>>) {
    let n_samples = data.n_rows();
    let norms = compute_l2_norms(data);
    approximate_nearest_neighbors_with_distance(
        n_samples,
        n_neighbors,
        candidate_pool,
        n_iters,
        seed,
        |i, j| cosine_distance_with_norms(data.row(i), data.row(j), norms[i], norms[j]),
    )
}

fn approximate_nearest_neighbors<M: RowMatrix + Sync + ?Sized>(
    data: &M,
    n_neighbors: usize,
    metric: Metric,
    candidate_pool: usize,
    n_iters: usize,
    seed: u64,
) -> (Vec<Vec<usize>>, Vec<Vec<f32>>) {
    match metric {
        Metric::Euclidean => approximate_nearest_neighbors_euclidean(
            data,
            n_neighbors,
            candidate_pool,
            n_iters,
            seed,
        ),
        Metric::Manhattan => approximate_nearest_neighbors_by(
            data,
            n_neighbors,
            candidate_pool,
            n_iters,
            seed,
            manhattan_distance,
        ),
        Metric::Cosine => {
            approximate_nearest_neighbors_cosine(data, n_neighbors, candidate_pool, n_iters, seed)
        }
        Metric::Chebyshev => approximate_nearest_neighbors_by(
            data,
            n_neighbors,
            candidate_pool,
            n_iters,
            seed,
            chebyshev_distance,
        ),
        Metric::Minkowski { p } => approximate_nearest_neighbors_by(
            data,
            n_neighbors,
            candidate_pool,
            n_iters,
            seed,
            move |x, y| minkowski_distance(x, y, p),
        ),
        Metric::Correlation => approximate_nearest_neighbors_by(
            data,
            n_neighbors,
            candidate_pool,
            n_iters,
            seed,
            correlation_distance,
        ),
        Metric::Canberra => approximate_nearest_neighbors_by(
            data,
            n_neighbors,
            candidate_pool,
            n_iters,
            seed,
            canberra_distance,
        ),
        Metric::BrayCurtis => approximate_nearest_neighbors_by(
            data,
            n_neighbors,
            candidate_pool,
            n_iters,
            seed,
            bray_curtis_distance,
        ),
    }
}

fn exact_nearest_neighbors_to_reference_euclidean<Q, R>(
    query: &Q,
    reference: &R,
    n_neighbors: usize,
) -> (Vec<Vec<usize>>, Vec<Vec<f32>>)
where
    Q: RowMatrix + Sync + ?Sized,
    R: RowMatrix + Sync + ?Sized,
{
    (0..query.n_rows())
        .into_par_iter()
        .map(|i| {
            let x = query.row(i);
            exact_top_k_neighbors_to_reference_euclidean(x, reference, n_neighbors)
        })
        .unzip()
}

fn exact_nearest_neighbors_to_reference_by<Q, R, F>(
    query: &Q,
    reference: &R,
    n_neighbors: usize,
    distance_fn: F,
) -> (Vec<Vec<usize>>, Vec<Vec<f32>>)
where
    Q: RowMatrix + Sync + ?Sized,
    R: RowMatrix + Sync + ?Sized,
    F: Fn(&[f32], &[f32]) -> f32 + Copy + Sync + Send,
{
    (0..query.n_rows())
        .into_par_iter()
        .map(|i| {
            let x = query.row(i);
            exact_top_k_neighbors(reference.n_rows(), n_neighbors, |j| {
                distance_fn(x, reference.row(j))
            })
        })
        .unzip()
}

fn exact_nearest_neighbors_to_reference_cosine<Q, R>(
    query: &Q,
    reference: &R,
    n_neighbors: usize,
) -> (Vec<Vec<usize>>, Vec<Vec<f32>>)
where
    Q: RowMatrix + Sync + ?Sized,
    R: RowMatrix + Sync + ?Sized,
{
    let query_norms = compute_l2_norms(query);
    let ref_norms = compute_l2_norms(reference);

    (0..query.n_rows())
        .into_par_iter()
        .map(|i| {
            let x = query.row(i);
            exact_top_k_neighbors(reference.n_rows(), n_neighbors, |j| {
                cosine_distance_with_norms(x, reference.row(j), query_norms[i], ref_norms[j])
            })
        })
        .unzip()
}

fn exact_nearest_neighbors_to_reference<Q, R>(
    query: &Q,
    reference: &R,
    n_neighbors: usize,
    metric: Metric,
) -> (Vec<Vec<usize>>, Vec<Vec<f32>>)
where
    Q: RowMatrix + Sync + ?Sized,
    R: RowMatrix + Sync + ?Sized,
{
    match metric {
        Metric::Euclidean => {
            exact_nearest_neighbors_to_reference_euclidean(query, reference, n_neighbors)
        }
        Metric::Manhattan => exact_nearest_neighbors_to_reference_by(
            query,
            reference,
            n_neighbors,
            manhattan_distance,
        ),
        Metric::Cosine => {
            exact_nearest_neighbors_to_reference_cosine(query, reference, n_neighbors)
        }
        Metric::Chebyshev => exact_nearest_neighbors_to_reference_by(
            query,
            reference,
            n_neighbors,
            chebyshev_distance,
        ),
        Metric::Minkowski { p } => {
            exact_nearest_neighbors_to_reference_by(query, reference, n_neighbors, move |x, y| {
                minkowski_distance(x, y, p)
            })
        }
        Metric::Correlation => exact_nearest_neighbors_to_reference_by(
            query,
            reference,
            n_neighbors,
            correlation_distance,
        ),
        Metric::Canberra => exact_nearest_neighbors_to_reference_by(
            query,
            reference,
            n_neighbors,
            canberra_distance,
        ),
        Metric::BrayCurtis => exact_nearest_neighbors_to_reference_by(
            query,
            reference,
            n_neighbors,
            bray_curtis_distance,
        ),
    }
}

#[inline]
fn knn_rows_include_self(knn_indices: &[Vec<usize>], knn_dists: &[Vec<f32>]) -> bool {
    if knn_indices.len() != knn_dists.len() {
        return false;
    }

    knn_indices
        .iter()
        .zip(knn_dists.iter())
        .enumerate()
        .all(|(row_idx, (idx_row, dist_row))| {
            !idx_row.is_empty()
                && !dist_row.is_empty()
                && idx_row[0] == row_idx
                && dist_row[0].abs() <= SMOOTH_K_TOLERANCE
        })
}

#[inline]
fn inverse_neighbor_count(params_n_neighbors: usize, n_train: usize) -> usize {
    params_n_neighbors.min(n_train).max(1)
}

fn smooth_knn_dist(
    distances: &[Vec<f32>],
    k: f32,
    local_connectivity: f32,
    bandwidth: f32,
    distances_include_self: bool,
) -> (Vec<f32>, Vec<f32>) {
    let target = k.log2() * bandwidth;
    let n_samples = distances.len();

    let mean_distances = {
        let mut total = 0.0_f32;
        let mut count = 0_usize;
        for row in distances {
            let start = usize::from(distances_include_self && !row.is_empty());
            for &d in row.iter().skip(start) {
                total += d;
                count += 1;
            }
        }
        if count > 0 { total / count as f32 } else { 0.0 }
    };

    let mut sigmas = vec![0.0_f32; n_samples];
    let mut rhos = vec![0.0_f32; n_samples];
    let mut non_zero_dists = Vec::new();

    for i in 0..n_samples {
        let ith_distances = &distances[i];
        let neighbor_start = usize::from(distances_include_self && !ith_distances.is_empty());
        let active_distances = &ith_distances[neighbor_start..];

        non_zero_dists.clear();
        non_zero_dists.extend(active_distances.iter().copied().filter(|d| *d > 0.0));

        let mean_ith = if active_distances.is_empty() {
            0.0
        } else {
            active_distances.iter().sum::<f32>() / active_distances.len() as f32
        };
        if non_zero_dists.len() as f32 >= local_connectivity {
            let index = local_connectivity.floor() as usize;
            let interpolation = local_connectivity - index as f32;

            if index > 0 {
                rhos[i] = non_zero_dists[index - 1];
                if interpolation > SMOOTH_K_TOLERANCE && index < non_zero_dists.len() {
                    rhos[i] += interpolation * (non_zero_dists[index] - non_zero_dists[index - 1]);
                }
            } else if let Some(first) = non_zero_dists.first() {
                rhos[i] = interpolation * *first;
            }
        } else if let Some(max_dist) = non_zero_dists.iter().copied().reduce(f32::max) {
            rhos[i] = max_dist;
        }

        let mut lo = 0.0_f32;
        let mut hi = f32::INFINITY;
        let mut mid = 1.0_f32;

        for _ in 0..64 {
            let mut psum = 0.0_f32;

            for &dist in active_distances {
                let d = dist - rhos[i];
                if d > 0.0 {
                    psum += (-(d / mid)).exp();
                } else {
                    psum += 1.0;
                }
            }

            if (psum - target).abs() < SMOOTH_K_TOLERANCE {
                break;
            }

            if psum > target {
                hi = mid;
                mid = (lo + hi) / 2.0;
            } else {
                lo = mid;
                if hi.is_infinite() {
                    mid *= 2.0;
                } else {
                    mid = (lo + hi) / 2.0;
                }
            }
        }

        let min_scale = if rhos[i] > 0.0 {
            MIN_K_DIST_SCALE * mean_ith
        } else {
            MIN_K_DIST_SCALE * mean_distances
        };

        sigmas[i] = mid.max(min_scale);
    }

    (sigmas, rhos)
}

fn compute_membership_strengths(
    knn_indices: &[Vec<usize>],
    knn_dists: &[Vec<f32>],
    sigmas: &[f32],
    rhos: &[f32],
) -> Vec<Edge> {
    let n_samples = knn_indices.len();
    let n_neighbors = knn_indices[0].len();

    let mut directed = Vec::with_capacity(n_samples * n_neighbors);

    for i in 0..n_samples {
        for j in 0..n_neighbors {
            let neighbor = knn_indices[i][j];
            let dist = knn_dists[i][j];

            let val = if neighbor == i {
                0.0
            } else if dist - rhos[i] <= 0.0 || sigmas[i] == 0.0 {
                1.0
            } else {
                (-(dist - rhos[i]) / sigmas[i]).exp()
            };

            if val > 0.0 {
                directed.push(Edge {
                    head: i,
                    tail: neighbor,
                    weight: val,
                });
            }
        }
    }

    directed
}

fn compute_membership_strengths_bipartite(
    indices: &[Vec<usize>],
    dists: &[Vec<f32>],
    sigmas: &[f32],
    rhos: &[f32],
) -> Vec<Edge> {
    let n_samples = indices.len();
    let n_neighbors = indices[0].len();

    let mut edges = Vec::with_capacity(n_samples * n_neighbors);

    for i in 0..n_samples {
        for j in 0..n_neighbors {
            let tail = indices[i][j];
            let dist = dists[i][j];

            let val = if dist - rhos[i] <= 0.0 || sigmas[i] == 0.0 {
                1.0
            } else {
                (-(dist - rhos[i]) / sigmas[i]).exp()
            };

            if val > 0.0 {
                edges.push(Edge {
                    head: i,
                    tail,
                    weight: val,
                });
            }
        }
    }

    edges
}

fn symmetrize_fuzzy_graph(directed: &[Edge], set_op_mix_ratio: f32) -> Vec<Edge> {
    let mut by_pair = Vec::with_capacity(directed.len());

    for edge in directed {
        if edge.head == edge.tail {
            continue;
        }
        if edge.head < edge.tail {
            by_pair.push((edge.head, edge.tail, edge.weight, 0.0_f32));
        } else {
            by_pair.push((edge.tail, edge.head, 0.0_f32, edge.weight));
        }
    }

    by_pair.sort_by_key(|&(i, j, _, _)| (i, j));

    let mut edges = Vec::with_capacity(by_pair.len().saturating_mul(2));
    let mut idx = 0;
    while idx < by_pair.len() {
        let (i, j, _, _) = by_pair[idx];
        let mut w_ij = 0.0_f32;
        let mut w_ji = 0.0_f32;

        while idx < by_pair.len() && by_pair[idx].0 == i && by_pair[idx].1 == j {
            if by_pair[idx].2 > 0.0 {
                w_ij = by_pair[idx].2;
            }
            if by_pair[idx].3 > 0.0 {
                w_ji = by_pair[idx].3;
            }
            idx += 1;
        }

        let prod = w_ij * w_ji;
        let sym = set_op_mix_ratio * (w_ij + w_ji - prod) + (1.0 - set_op_mix_ratio) * prod;

        if sym > 0.0 {
            edges.push(Edge {
                head: i,
                tail: j,
                weight: sym,
            });
            edges.push(Edge {
                head: j,
                tail: i,
                weight: sym,
            });
        }
    }

    edges.sort_by_key(|e| (e.head, e.tail));
    edges
}

#[cfg(test)]
fn build_categorical_target_membership_strengths(
    target: &SupervisedTarget,
    n_samples: usize,
    default_n_neighbors: usize,
) -> Vec<Edge> {
    match target.metric {
        TargetMetric::Categorical => {}
    }

    let mut by_label = HashMap::<i64, Vec<usize>>::new();
    for (idx, &label) in target.labels.iter().enumerate().take(n_samples) {
        if label == -1 {
            continue;
        }
        by_label.entry(label).or_default().push(idx);
    }

    let target_n_neighbors = target.n_neighbors.unwrap_or(default_n_neighbors);
    let mut edges = Vec::new();
    for group in by_label.values() {
        if group.len() < 2 {
            continue;
        }
        let row_neighbors = target_n_neighbors.min(group.len() - 1);
        for (pos, &head) in group.iter().enumerate() {
            for offset in 1..=row_neighbors {
                let tail = group[(pos + offset) % group.len()];
                if head != tail {
                    edges.push(Edge {
                        head,
                        tail,
                        weight: 1.0,
                    });
                }
            }
        }
    }

    edges.sort_by_key(|edge| (edge.head, edge.tail));
    edges
}

fn apply_categorical_target_intersection(
    data_edges: &[Edge],
    target: &SupervisedTarget,
) -> Vec<Edge> {
    if target.weight <= 0.0 || data_edges.is_empty() {
        return data_edges.to_vec();
    }

    let far_dist = if target.weight < 1.0 {
        2.5 / (1.0 - target.weight)
    } else {
        1.0e12
    };
    let unknown_scale = (-1.0_f32).exp();
    let far_scale = (-far_dist).exp();

    let mut scaled = Vec::with_capacity(data_edges.len());
    for edge in data_edges {
        let lhs = target.labels[edge.head];
        let rhs = target.labels[edge.tail];
        let scale = if lhs == -1 || rhs == -1 {
            unknown_scale
        } else if lhs != rhs {
            far_scale
        } else {
            1.0
        };
        let weight = edge.weight * scale;
        if weight > 0.0 && weight.is_finite() {
            scaled.push(Edge {
                head: edge.head,
                tail: edge.tail,
                weight,
            });
        }
    }

    reset_local_connectivity_edges(&scaled, target.labels.len())
}

fn reset_local_connectivity_edges(edges: &[Edge], n_vertices: usize) -> Vec<Edge> {
    if edges.is_empty() {
        return Vec::new();
    }

    let mut row_max = vec![0.0_f32; n_vertices];
    for edge in edges {
        row_max[edge.head] = row_max[edge.head].max(edge.weight);
    }

    let directed = edges
        .iter()
        .filter_map(|edge| {
            let max_weight = row_max[edge.head];
            if max_weight <= 0.0 || !max_weight.is_finite() {
                return None;
            }
            Some(Edge {
                head: edge.head,
                tail: edge.tail,
                weight: edge.weight / max_weight,
            })
        })
        .collect::<Vec<Edge>>();

    symmetrize_fuzzy_graph(&directed, 1.0)
}

#[cfg(test)]
fn combine_fuzzy_graphs(
    data_edges: &[Edge],
    target_edges: &[Edge],
    target_weight: f32,
) -> Vec<Edge> {
    if target_weight <= 0.0 || target_edges.is_empty() {
        return data_edges.to_vec();
    }

    let mut by_pair = HashMap::<(usize, usize), (f32, f32)>::with_capacity(
        data_edges.len().saturating_add(target_edges.len()),
    );
    for edge in data_edges {
        by_pair
            .entry((edge.head, edge.tail))
            .and_modify(|weights| weights.0 = weights.0.max(edge.weight))
            .or_insert((edge.weight, 0.0));
    }
    for edge in target_edges {
        by_pair
            .entry((edge.head, edge.tail))
            .and_modify(|weights| weights.1 = weights.1.max(edge.weight))
            .or_insert((0.0, edge.weight));
    }

    let data_weight = 1.0 - target_weight;
    let mut edges = by_pair
        .into_iter()
        .filter_map(|((head, tail), (data, target))| {
            let weight = data_weight * data + target_weight * target;
            if weight > 0.0 && weight.is_finite() {
                Some(Edge { head, tail, weight })
            } else {
                None
            }
        })
        .collect::<Vec<Edge>>();
    edges.sort_by_key(|e| (e.head, e.tail));
    edges
}

fn prune_edges(edges: &mut Vec<Edge>, n_epochs: usize) {
    if edges.is_empty() {
        return;
    }

    let max_weight = edges
        .iter()
        .map(|e| e.weight)
        .fold(f32::NEG_INFINITY, f32::max);

    let denom = if n_epochs > 10 {
        n_epochs as f32
    } else {
        500.0
    };

    let threshold = max_weight / denom;
    edges.retain(|e| e.weight >= threshold && e.weight > 0.0);
}

fn random_init(
    n_samples: usize,
    n_components: usize,
    seed: u64,
    low: f32,
    high: f32,
) -> Vec<Vec<f32>> {
    let mut rng = SmallRng::seed_from_u64(seed);
    (0..n_samples)
        .map(|_| {
            (0..n_components)
                .map(|_| rng.gen_range(low..high))
                .collect::<Vec<f32>>()
        })
        .collect()
}

fn initialize_embedding<M: RowMatrix + ?Sized>(
    data: &M,
    n_components: usize,
    edges: &[Edge],
    init: InitMethod,
    seed: u64,
) -> Vec<Vec<f32>> {
    let n_samples = data.n_rows();
    match init {
        InitMethod::Random => random_init(n_samples, n_components, seed, -10.0, 10.0),
        InitMethod::Spectral => spectral_init(data, n_components, edges, seed)
            .unwrap_or_else(|| random_init(n_samples, n_components, seed, -10.0, 10.0)),
    }
}

fn initialize_embedding_sparse(
    data: &SparseCsrMatrix,
    n_components: usize,
    edges: &[Edge],
    init: InitMethod,
    seed: u64,
) -> Vec<Vec<f32>> {
    let n_samples = data.n_rows();
    match init {
        InitMethod::Random => random_init(n_samples, n_components, seed, -10.0, 10.0),
        InitMethod::Spectral => spectral_init_sparse(data, n_components, edges, seed)
            .unwrap_or_else(|| random_init(n_samples, n_components, seed, -10.0, 10.0)),
    }
}

fn spectral_init<M: RowMatrix + ?Sized>(
    data: &M,
    n_components: usize,
    edges: &[Edge],
    seed: u64,
) -> Option<Vec<Vec<f32>>> {
    let n_samples = data.n_rows();
    if n_samples <= n_components + 1 || edges.is_empty() {
        return None;
    }

    let components = connected_components_from_edges(n_samples, edges);
    if components.len() > 1 {
        return multi_component_spectral_init(data, n_components, edges, &components, seed);
    }

    spectral_init_connected(n_samples, n_components, edges, seed)
}

fn spectral_init_sparse(
    data: &SparseCsrMatrix,
    n_components: usize,
    edges: &[Edge],
    seed: u64,
) -> Option<Vec<Vec<f32>>> {
    let n_samples = data.n_rows();
    if n_samples <= n_components + 1 || edges.is_empty() {
        return None;
    }

    let components = connected_components_from_edges(n_samples, edges);
    if components.len() > 1 {
        return multi_component_spectral_init_sparse(data, n_components, edges, &components, seed);
    }

    spectral_init_connected(n_samples, n_components, edges, seed)
}

fn spectral_embedding_from_laplacian(
    laplacian: DMatrix<f64>,
    embedding_dim: usize,
    drop_first: bool,
) -> Option<Vec<Vec<f32>>> {
    let n_samples = laplacian.nrows();
    if n_samples == 0 || laplacian.ncols() != n_samples {
        return None;
    }

    let eig = SymmetricEigen::new(laplacian);
    let mut order: Vec<usize> = (0..n_samples).collect();
    order.sort_by(|&i, &j| {
        eig.eigenvalues[i]
            .partial_cmp(&eig.eigenvalues[j])
            .unwrap_or(Ordering::Equal)
    });

    let offset = usize::from(drop_first);
    if order.len() <= embedding_dim + offset {
        return None;
    }

    let mut coords = vec![vec![0.0_f32; embedding_dim]; n_samples];
    for out_col in 0..embedding_dim {
        let eig_col = order[out_col + offset];
        for (row, coord_row) in coords.iter_mut().enumerate().take(n_samples) {
            coord_row[out_col] = eig.eigenvectors[(row, eig_col)] as f32;
        }
    }

    if coords
        .iter()
        .flat_map(|row| row.iter())
        .all(|value| value.is_finite())
    {
        Some(coords)
    } else {
        None
    }
}

fn normalized_laplacian_from_affinity(affinity: &[Vec<f64>]) -> Option<DMatrix<f64>> {
    let n_samples = affinity.len();
    if n_samples == 0 {
        return None;
    }
    if affinity.iter().any(|row| row.len() != n_samples) {
        return None;
    }

    let mut degrees = vec![0.0_f64; n_samples];
    for i in 0..n_samples {
        degrees[i] = affinity[i].iter().sum();
    }
    if degrees.iter().all(|degree| *degree <= 0.0) {
        return None;
    }

    let inv_sqrt_degrees = degrees
        .iter()
        .map(|&degree| {
            if degree > 0.0 {
                degree.sqrt().recip()
            } else {
                0.0
            }
        })
        .collect::<Vec<f64>>();

    let mut laplacian = DMatrix::<f64>::identity(n_samples, n_samples);
    for i in 0..n_samples {
        if degrees[i] <= 0.0 {
            continue;
        }
        let inv_i = inv_sqrt_degrees[i];
        for j in 0..n_samples {
            if degrees[j] <= 0.0 {
                continue;
            }
            let weight = affinity[i][j];
            if weight > 0.0 {
                laplacian[(i, j)] -= weight * inv_i * inv_sqrt_degrees[j];
            }
        }
    }

    Some(laplacian)
}

fn normalized_laplacian_from_edges(n_samples: usize, edges: &[Edge]) -> Option<DMatrix<f64>> {
    if n_samples == 0 || edges.is_empty() {
        return None;
    }

    let mut laplacian = DMatrix::<f64>::identity(n_samples, n_samples);
    let mut degrees = vec![0.0_f64; n_samples];
    let mut undirected_pairs = Vec::<(usize, usize)>::with_capacity(edges.len());

    for edge in edges {
        if edge.head == edge.tail {
            continue;
        }
        let weight = edge.weight.max(0.0) as f64;
        if weight <= 0.0 {
            continue;
        }

        let (i, j) = if edge.head < edge.tail {
            (edge.head, edge.tail)
        } else {
            (edge.tail, edge.head)
        };
        let prev = laplacian[(i, j)];
        if weight > prev {
            if prev == 0.0 {
                undirected_pairs.push((i, j));
            }
            let delta = weight - prev;
            laplacian[(i, j)] = weight;
            laplacian[(j, i)] = weight;
            degrees[i] += delta;
            degrees[j] += delta;
        }
    }

    if degrees.iter().all(|degree| *degree <= 0.0) {
        return None;
    }

    let inv_sqrt_degrees = degrees
        .iter()
        .map(|&degree| {
            if degree > 0.0 {
                degree.sqrt().recip()
            } else {
                0.0
            }
        })
        .collect::<Vec<f64>>();

    for (i, j) in undirected_pairs {
        let normalized = laplacian[(i, j)] * inv_sqrt_degrees[i] * inv_sqrt_degrees[j];
        laplacian[(i, j)] = -normalized;
        laplacian[(j, i)] = -normalized;
    }

    for i in 0..n_samples {
        laplacian[(i, i)] = 1.0;
    }

    Some(laplacian)
}

fn spectral_embedding_from_affinity_raw(
    affinity: &[Vec<f64>],
    embedding_dim: usize,
    drop_first: bool,
) -> Option<Vec<Vec<f32>>> {
    let laplacian = normalized_laplacian_from_affinity(affinity)?;
    spectral_embedding_from_laplacian(laplacian, embedding_dim, drop_first)
}

#[derive(Clone)]
struct SpectralBlock {
    data: Vec<f64>,
    n_rows: usize,
    n_cols: usize,
}

impl SpectralBlock {
    fn zeros(n_rows: usize, n_cols: usize) -> Self {
        Self {
            data: vec![0.0_f64; n_rows.saturating_mul(n_cols)],
            n_rows,
            n_cols,
        }
    }

    fn row(&self, row: usize) -> &[f64] {
        let start = row * self.n_cols;
        &self.data[start..start + self.n_cols]
    }

    fn row_mut(&mut self, row: usize) -> &mut [f64] {
        let start = row * self.n_cols;
        &mut self.data[start..start + self.n_cols]
    }
}

fn symmetrized_undirected_edges(
    n_samples: usize,
    edges: &[Edge],
) -> Option<SymmetrizedUndirectedGraph> {
    if n_samples == 0 || edges.is_empty() {
        return None;
    }

    let mut by_pair = HashMap::<(usize, usize), f64>::with_capacity(edges.len());
    for edge in edges {
        if edge.head == edge.tail {
            continue;
        }
        let weight = edge.weight.max(0.0) as f64;
        if weight <= 0.0 {
            continue;
        }
        let pair = if edge.head < edge.tail {
            (edge.head, edge.tail)
        } else {
            (edge.tail, edge.head)
        };
        by_pair
            .entry(pair)
            .and_modify(|current| {
                if weight > *current {
                    *current = weight;
                }
            })
            .or_insert(weight);
    }

    if by_pair.is_empty() {
        return None;
    }

    let mut degrees = vec![0.0_f64; n_samples];
    let mut undirected_edges = by_pair
        .into_iter()
        .map(|((i, j), weight)| {
            degrees[i] += weight;
            degrees[j] += weight;
            (i, j, weight)
        })
        .collect::<Vec<WeightedUndirectedEdge>>();
    undirected_edges.sort_unstable_by_key(|&(i, j, _)| (i, j));

    if degrees.iter().all(|degree| *degree <= 0.0) {
        return None;
    }

    Some((undirected_edges, degrees))
}

fn orthonormalize_columns(block: &mut SpectralBlock) -> bool {
    if block.n_rows == 0 {
        return false;
    }
    if block.n_cols == 0 {
        return false;
    }

    for col in 0..block.n_cols {
        for _ in 0..2 {
            for prev in 0..col {
                let mut dot = 0.0_f64;
                for row in 0..block.n_rows {
                    let row_offset = row * block.n_cols;
                    dot += block.data[row_offset + col] * block.data[row_offset + prev];
                }
                for row in 0..block.n_rows {
                    let row_offset = row * block.n_cols;
                    block.data[row_offset + col] -= dot * block.data[row_offset + prev];
                }
            }
        }

        let mut norm_sq = 0.0_f64;
        for row in 0..block.n_rows {
            let value = block.data[row * block.n_cols + col];
            norm_sq += value * value;
        }
        if !norm_sq.is_finite() || norm_sq <= SPECTRAL_ORTHO_EPS {
            return false;
        }
        let inv_norm = norm_sq.sqrt().recip();
        for row in 0..block.n_rows {
            block.data[row * block.n_cols + col] *= inv_norm;
        }
    }

    true
}

fn shifted_normalized_adjacency_mul(
    block: &SpectralBlock,
    undirected_edges: &[(usize, usize, f64)],
    inv_sqrt_degrees: &[f64],
) -> SpectralBlock {
    let mut out = block.clone();

    for &(i, j, weight) in undirected_edges {
        let normalized = weight * inv_sqrt_degrees[i] * inv_sqrt_degrees[j];
        if normalized == 0.0 {
            continue;
        }
        let row_i_offset = i * block.n_cols;
        let row_j_offset = j * block.n_cols;
        for col in 0..block.n_cols {
            let x_i = block.data[row_i_offset + col];
            let x_j = block.data[row_j_offset + col];
            out.data[row_i_offset + col] += normalized * x_j;
            out.data[row_j_offset + col] += normalized * x_i;
        }
    }

    out
}

fn spectral_embedding_from_edges_iterative(
    n_samples: usize,
    embedding_dim: usize,
    edges: &[Edge],
    seed: u64,
) -> Option<Vec<Vec<f32>>> {
    let subspace_dim = embedding_dim + 1;
    if n_samples <= subspace_dim || edges.is_empty() {
        return None;
    }

    let (undirected_edges, degrees) = symmetrized_undirected_edges(n_samples, edges)?;
    let inv_sqrt_degrees = degrees
        .iter()
        .map(|&degree| {
            if degree > 0.0 {
                degree.sqrt().recip()
            } else {
                0.0
            }
        })
        .collect::<Vec<f64>>();

    let mut rng = SmallRng::seed_from_u64(seed ^ 0x7A5B_3D91_CE42_A117);
    let mut block = SpectralBlock::zeros(n_samples, subspace_dim);
    for (row, degree) in degrees.iter().enumerate().take(n_samples) {
        let block_row = block.row_mut(row);
        block_row[0] = degree.sqrt();
        for value in block_row.iter_mut().take(subspace_dim).skip(1) {
            *value = rng.gen_range(-1.0_f64..1.0_f64);
        }
    }
    if !orthonormalize_columns(&mut block) {
        return None;
    }

    for _ in 0..SPECTRAL_ITERATIVE_MAX_ITERS {
        let mut next =
            shifted_normalized_adjacency_mul(&block, &undirected_edges, &inv_sqrt_degrees);
        if !orthonormalize_columns(&mut next) {
            return None;
        }
        block = next;
    }

    let projected = shifted_normalized_adjacency_mul(&block, &undirected_edges, &inv_sqrt_degrees);
    let mut ritz = DMatrix::<f64>::zeros(subspace_dim, subspace_dim);
    for i in 0..subspace_dim {
        for j in i..subspace_dim {
            let mut value = 0.0_f64;
            for row in 0..n_samples {
                value += block.row(row)[i] * projected.row(row)[j];
            }
            ritz[(i, j)] = value;
            ritz[(j, i)] = value;
        }
    }

    let eig = SymmetricEigen::new(ritz);
    let mut order: Vec<usize> = (0..subspace_dim).collect();
    order.sort_by(|&i, &j| {
        eig.eigenvalues[j]
            .partial_cmp(&eig.eigenvalues[i])
            .unwrap_or(Ordering::Equal)
    });
    if order.len() <= embedding_dim {
        return None;
    }

    let mut coords = vec![vec![0.0_f32; embedding_dim]; n_samples];
    for out_col in 0..embedding_dim {
        let eig_col = order[out_col + 1];
        for (row, coord_row) in coords.iter_mut().enumerate().take(n_samples) {
            let mut value = 0.0_f64;
            let block_row = block.row(row);
            for (basis, &block_value) in block_row.iter().enumerate().take(subspace_dim) {
                value += block_value * eig.eigenvectors[(basis, eig_col)];
            }
            coord_row[out_col] = value as f32;
        }
    }

    if coords
        .iter()
        .flat_map(|row| row.iter())
        .all(|value| value.is_finite())
    {
        Some(coords)
    } else {
        None
    }
}

fn spectral_embedding_from_edges(
    n_samples: usize,
    embedding_dim: usize,
    edges: &[Edge],
    seed: u64,
    iterative_threshold: usize,
) -> Option<Vec<Vec<f32>>> {
    if n_samples <= embedding_dim + 1 || edges.is_empty() {
        return None;
    }

    if n_samples >= iterative_threshold {
        spectral_embedding_from_edges_iterative(n_samples, embedding_dim, edges, seed).or_else(
            || {
                let laplacian = normalized_laplacian_from_edges(n_samples, edges)?;
                spectral_embedding_from_laplacian(laplacian, embedding_dim, true)
            },
        )
    } else {
        let laplacian = normalized_laplacian_from_edges(n_samples, edges)?;
        spectral_embedding_from_laplacian(laplacian, embedding_dim, true)
    }
}

fn spectral_init_connected(
    n_samples: usize,
    n_components: usize,
    edges: &[Edge],
    seed: u64,
) -> Option<Vec<Vec<f32>>> {
    let mut coords = spectral_embedding_from_edges(
        n_samples,
        n_components,
        edges,
        seed,
        SPECTRAL_ITERATIVE_CONNECTED_THRESHOLD,
    )?;
    noisy_scale_coords(&mut coords, seed ^ 0x5DEECE66D, INIT_MAX_COORD, INIT_NOISE);

    if coords.iter().flat_map(|r| r.iter()).all(|v| v.is_finite()) {
        Some(coords)
    } else {
        None
    }
}

fn connected_components_from_edges(n_samples: usize, edges: &[Edge]) -> Vec<Vec<usize>> {
    let mut parent = (0..n_samples).collect::<Vec<usize>>();
    let mut rank = vec![0_u8; n_samples];

    fn find(parent: &mut [usize], x: usize) -> usize {
        let mut root = x;
        while parent[root] != root {
            root = parent[root];
        }
        let mut node = x;
        while parent[node] != root {
            let next = parent[node];
            parent[node] = root;
            node = next;
        }
        root
    }

    fn union(parent: &mut [usize], rank: &mut [u8], a: usize, b: usize) {
        let root_a = find(parent, a);
        let root_b = find(parent, b);
        if root_a == root_b {
            return;
        }
        if rank[root_a] < rank[root_b] {
            parent[root_a] = root_b;
        } else if rank[root_a] > rank[root_b] {
            parent[root_b] = root_a;
        } else {
            parent[root_b] = root_a;
            rank[root_a] += 1;
        }
    }

    for edge in edges {
        if edge.head != edge.tail {
            union(&mut parent, &mut rank, edge.head, edge.tail);
        }
    }

    let mut groups = HashMap::<usize, Vec<usize>>::new();
    for node in 0..n_samples {
        let root = find(&mut parent, node);
        groups.entry(root).or_default().push(node);
    }

    let mut components = groups.into_values().collect::<Vec<Vec<usize>>>();
    for component in components.iter_mut() {
        component.sort_unstable();
    }
    components.sort_by_key(|component| component[0]);
    components
}

fn remap_component_edges(
    component: &[usize],
    component_labels: &[usize],
    component_id: usize,
    edges: &[Edge],
) -> Vec<Edge> {
    let mut mapping = vec![usize::MAX; component_labels.len()];
    for (local_idx, &global_idx) in component.iter().enumerate() {
        mapping[global_idx] = local_idx;
    }

    let mut out = Vec::new();
    for edge in edges {
        if component_labels[edge.head] == component_id
            && component_labels[edge.tail] == component_id
        {
            out.push(Edge {
                head: mapping[edge.head],
                tail: mapping[edge.tail],
                weight: edge.weight,
            });
        }
    }
    out
}

fn component_centroids_dense<M: RowMatrix + ?Sized>(
    data: &M,
    components: &[Vec<usize>],
) -> Vec<Vec<f64>> {
    let n_features = data.n_cols();
    let mut centroids = Vec::with_capacity(components.len());

    for component in components {
        let mut centroid = vec![0.0_f64; n_features];
        for &idx in component {
            for (feature_idx, &value) in data.row(idx).iter().enumerate() {
                centroid[feature_idx] += value as f64;
            }
        }
        let scale = 1.0 / component.len() as f64;
        for value in centroid.iter_mut() {
            *value *= scale;
        }
        centroids.push(centroid);
    }

    centroids
}

fn component_centroids_sparse(data: &SparseCsrMatrix, components: &[Vec<usize>]) -> Vec<Vec<f64>> {
    let n_features = data.n_cols();
    let mut centroids = Vec::with_capacity(components.len());

    for component in components {
        let mut centroid = vec![0.0_f64; n_features];
        for &idx in component {
            let (row_indices, row_values) = data.row(idx);
            for (&feature_idx, &value) in row_indices.iter().zip(row_values.iter()) {
                centroid[feature_idx] += value as f64;
            }
        }
        let scale = 1.0 / component.len() as f64;
        for value in centroid.iter_mut() {
            *value *= scale;
        }
        centroids.push(centroid);
    }

    centroids
}

fn squared_distance_f64(x: &[f64], y: &[f64]) -> f64 {
    x.iter()
        .zip(y.iter())
        .map(|(a, b)| {
            let d = a - b;
            d * d
        })
        .sum()
}

fn meta_component_layout_from_centroids(
    centroids: &[Vec<f64>],
    embedding_dim: usize,
    seed: u64,
) -> Option<Vec<Vec<f32>>> {
    let n_graph_components = centroids.len();
    if n_graph_components == 0 {
        return None;
    }
    if n_graph_components == 1 {
        return Some(vec![vec![0.0; embedding_dim]]);
    }

    if n_graph_components <= 2 * embedding_dim {
        let k = n_graph_components.div_ceil(2);
        let mut layout = vec![vec![0.0_f32; embedding_dim]; n_graph_components];
        for (i, row) in layout.iter_mut().enumerate().take(n_graph_components) {
            let axis = i % k;
            row[axis] = if i < k { 1.0 } else { -1.0 };
        }
        return Some(layout);
    }

    let mut affinity = vec![vec![0.0_f64; n_graph_components]; n_graph_components];
    for i in 0..n_graph_components {
        affinity[i][i] = 1.0;
        for j in (i + 1)..n_graph_components {
            let dist2 = squared_distance_f64(&centroids[i], &centroids[j]);
            let weight = (-dist2).exp();
            affinity[i][j] = weight;
            affinity[j][i] = weight;
        }
    }

    spectral_embedding_from_affinity(&affinity, embedding_dim, true, seed ^ 0xC85E_7D6A_DA31_8F21)
}

fn meta_component_layout<M: RowMatrix + ?Sized>(
    data: &M,
    embedding_dim: usize,
    components: &[Vec<usize>],
    seed: u64,
) -> Option<Vec<Vec<f32>>> {
    let centroids = component_centroids_dense(data, components);
    meta_component_layout_from_centroids(&centroids, embedding_dim, seed)
}

fn meta_component_layout_sparse(
    data: &SparseCsrMatrix,
    embedding_dim: usize,
    components: &[Vec<usize>],
    seed: u64,
) -> Option<Vec<Vec<f32>>> {
    let centroids = component_centroids_sparse(data, components);
    meta_component_layout_from_centroids(&centroids, embedding_dim, seed)
}

fn spectral_embedding_from_affinity(
    affinity: &[Vec<f64>],
    embedding_dim: usize,
    drop_first: bool,
    seed: u64,
) -> Option<Vec<Vec<f32>>> {
    let mut coords = spectral_embedding_from_affinity_raw(affinity, embedding_dim, drop_first)?;
    noisy_scale_coords(&mut coords, seed, INIT_MAX_COORD, INIT_NOISE);
    if coords
        .iter()
        .flat_map(|row| row.iter())
        .all(|value| value.is_finite())
    {
        Some(coords)
    } else {
        None
    }
}

fn spectral_init_connected_raw(
    n_samples: usize,
    n_components: usize,
    edges: &[Edge],
    seed: u64,
) -> Option<Vec<Vec<f32>>> {
    spectral_embedding_from_edges(
        n_samples,
        n_components,
        edges,
        seed,
        SPECTRAL_ITERATIVE_COMPONENT_THRESHOLD,
    )
}

fn add_noise(coords: &mut [Vec<f32>], seed: u64, noise: f32) {
    if coords.is_empty() || coords[0].is_empty() {
        return;
    }

    let normal = Normal::new(0.0_f64, noise as f64).expect("normal distribution should be valid");
    let mut rng = SmallRng::seed_from_u64(seed);
    for row in coords.iter_mut() {
        for value in row.iter_mut() {
            *value += normal.sample(&mut rng) as f32;
        }
    }
}

fn multi_component_spectral_init<M: RowMatrix + ?Sized>(
    data: &M,
    embedding_dim: usize,
    edges: &[Edge],
    components: &[Vec<usize>],
    seed: u64,
) -> Option<Vec<Vec<f32>>> {
    let component_layout = meta_component_layout(
        data,
        embedding_dim,
        components,
        seed ^ 0xA13F_52A9_2D4C_B801,
    )?;
    multi_component_spectral_init_with_layout(
        data.n_rows(),
        embedding_dim,
        edges,
        components,
        &component_layout,
        seed,
    )
}

fn multi_component_spectral_init_sparse(
    data: &SparseCsrMatrix,
    embedding_dim: usize,
    edges: &[Edge],
    components: &[Vec<usize>],
    seed: u64,
) -> Option<Vec<Vec<f32>>> {
    let component_layout = meta_component_layout_sparse(
        data,
        embedding_dim,
        components,
        seed ^ 0xA13F_52A9_2D4C_B801,
    )?;
    multi_component_spectral_init_with_layout(
        data.n_rows(),
        embedding_dim,
        edges,
        components,
        &component_layout,
        seed,
    )
}

fn multi_component_spectral_init_with_layout(
    n_samples: usize,
    embedding_dim: usize,
    edges: &[Edge],
    components: &[Vec<usize>],
    component_layout: &[Vec<f32>],
    seed: u64,
) -> Option<Vec<Vec<f32>>> {
    if component_layout.len() != components.len() {
        return None;
    }

    let mut component_labels = vec![usize::MAX; n_samples];
    for (component_id, component) in components.iter().enumerate() {
        for &idx in component {
            component_labels[idx] = component_id;
        }
    }

    let mut result = vec![vec![0.0_f32; embedding_dim]; n_samples];

    for (component_id, component) in components.iter().enumerate() {
        let anchor = &component_layout[component_id];
        let mut data_range = f32::INFINITY;
        for (other_id, other_anchor) in component_layout.iter().enumerate() {
            if other_id == component_id {
                continue;
            }
            let dist = euclidean_distance(anchor, other_anchor);
            if dist > 0.0 {
                data_range = data_range.min(dist / 2.0);
            }
        }
        if !data_range.is_finite() || data_range <= 0.0 {
            data_range = 1.0;
        }

        let local_coords =
            if component.len() < 2 * embedding_dim || component.len() <= embedding_dim + 1 {
                random_init(
                    component.len(),
                    embedding_dim,
                    seed ^ ((component_id as u64 + 1) * 0x9E37_79B9),
                    -data_range,
                    data_range,
                )
            } else {
                let component_edges =
                    remap_component_edges(component, &component_labels, component_id, edges);
                let mut coords = spectral_init_connected_raw(
                    component.len(),
                    embedding_dim,
                    &component_edges,
                    seed ^ (component_id as u64 + 1).wrapping_mul(0xBF58_476D_1CE4_E5B9),
                )?;
                let max_abs = coords
                    .iter()
                    .flat_map(|row| row.iter())
                    .map(|value| value.abs())
                    .fold(0.0_f32, f32::max);
                let expansion = if max_abs > 0.0 {
                    data_range / max_abs
                } else {
                    1.0
                };
                for row in coords.iter_mut() {
                    for value in row.iter_mut() {
                        *value *= expansion;
                    }
                }
                add_noise(
                    &mut coords,
                    seed ^ ((component_id as u64 + 1) * 0x94D0_49BB),
                    INIT_NOISE,
                );
                coords
            };

        for (local_idx, &global_idx) in component.iter().enumerate() {
            for dim in 0..embedding_dim {
                result[global_idx][dim] = local_coords[local_idx][dim] + anchor[dim];
            }
        }
    }

    if result
        .iter()
        .flat_map(|row| row.iter())
        .all(|value| value.is_finite())
    {
        Some(result)
    } else {
        None
    }
}

fn noisy_scale_coords(coords: &mut [Vec<f32>], seed: u64, max_coord: f32, noise: f32) {
    if coords.is_empty() || coords[0].is_empty() {
        return;
    }

    let max_abs = coords
        .iter()
        .flat_map(|r| r.iter())
        .map(|v| v.abs())
        .fold(0.0_f32, f32::max);

    let expansion = if max_abs > 0.0 {
        max_coord / max_abs
    } else {
        1.0
    };

    let normal = Normal::new(0.0_f64, noise as f64).expect("normal distribution should be valid");
    let mut rng = SmallRng::seed_from_u64(seed);

    for row in coords.iter_mut() {
        for value in row.iter_mut() {
            *value = *value * expansion + normal.sample(&mut rng) as f32;
        }
    }
}

fn init_graph_transform(
    n_new_samples: usize,
    dim: usize,
    graph_edges: &[Edge],
    base_embedding: &[Vec<f32>],
) -> Vec<Vec<f32>> {
    let mut rows: Vec<Vec<(usize, f32)>> = vec![Vec::new(); n_new_samples];
    for edge in graph_edges {
        rows[edge.head].push((edge.tail, edge.weight));
    }

    let mut result = vec![vec![0.0_f32; dim]; n_new_samples];

    for i in 0..n_new_samples {
        if rows[i].is_empty() {
            result[i].fill(f32::NAN);
            continue;
        }

        if let Some((col_idx, _)) = rows[i]
            .iter()
            .copied()
            .find(|(_, weight)| (*weight - 1.0).abs() < f32::EPSILON)
        {
            result[i].clone_from_slice(&base_embedding[col_idx]);
            continue;
        }

        let row_sum: f32 = rows[i].iter().map(|(_, w)| *w).sum();
        if row_sum <= 0.0 {
            result[i].fill(f32::NAN);
            continue;
        }

        for (col_idx, weight) in rows[i].iter().copied() {
            let w = weight / row_sum;
            for d in 0..dim {
                result[i][d] += w * base_embedding[col_idx][d];
            }
        }
    }

    result
}

fn init_inverse_points<Q, T>(
    embedded_query: &Q,
    neighbor_indices: &[Vec<usize>],
    normalized_weights: &[Vec<f32>],
    fixed_rows: &[bool],
    train_embedding: &[Vec<f32>],
    train_data: &T,
) -> Vec<Vec<f32>>
where
    Q: RowMatrix + ?Sized,
    T: RowMatrix + ?Sized,
{
    let n_queries = embedded_query.n_rows();
    let low_dim = embedded_query.n_cols();
    let high_dim = train_data.n_cols();
    let mut result = vec![vec![0.0_f32; high_dim]; n_queries];

    for i in 0..n_queries {
        if fixed_rows[i] {
            result[i].clone_from_slice(train_data.row(neighbor_indices[i][0]));
            continue;
        }

        let mut z_bar = vec![0.0_f32; low_dim];
        let mut x_bar = vec![0.0_f32; high_dim];
        let mut weight_sum = 0.0_f32;

        for (&idx, &weight) in neighbor_indices[i].iter().zip(normalized_weights[i].iter()) {
            if weight <= 0.0 || !weight.is_finite() {
                continue;
            }
            weight_sum += weight;
            for d in 0..low_dim {
                z_bar[d] += weight * train_embedding[idx][d];
            }
            let train_row = train_data.row(idx);
            for (d, x_bar_d) in x_bar.iter_mut().enumerate().take(high_dim) {
                *x_bar_d += weight * train_row[d];
            }
        }

        if weight_sum <= 0.0 {
            result[i].fill(f32::NAN);
            continue;
        }

        let mut gram = DMatrix::<f32>::zeros(low_dim, low_dim);
        let mut rhs = DMatrix::<f32>::zeros(low_dim, high_dim);
        let mut non_zero = 0usize;
        for (&idx, &weight) in neighbor_indices[i].iter().zip(normalized_weights[i].iter()) {
            if weight <= 0.0 || !weight.is_finite() {
                continue;
            }
            non_zero += 1;
            for a in 0..low_dim {
                let z_a = train_embedding[idx][a] - z_bar[a];
                for b in 0..low_dim {
                    let z_b = train_embedding[idx][b] - z_bar[b];
                    gram[(a, b)] += weight * z_a * z_b;
                }
                for d in 0..high_dim {
                    let x_d = train_data.row(idx)[d] - x_bar[d];
                    rhs[(a, d)] += weight * z_a * x_d;
                }
            }
        }

        for d in 0..low_dim {
            gram[(d, d)] += 1e-3;
        }

        let mut pred = x_bar.clone();
        if non_zero > low_dim {
            let lu = gram.lu();
            if let Some(beta) = lu.solve(&rhs) {
                let mut valid = true;
                for d in 0..high_dim {
                    let mut value = x_bar[d];
                    for a in 0..low_dim {
                        value += (embedded_query.row(i)[a] - z_bar[a]) * beta[(a, d)];
                    }
                    if !value.is_finite() {
                        valid = false;
                        break;
                    }
                    pred[d] = value;
                }
                if !valid {
                    pred.clone_from_slice(&x_bar);
                }
            }
        }

        result[i] = pred;
    }

    result
}

fn make_epochs_per_sample(weights: &[f32], n_epochs: usize) -> Vec<f32> {
    if weights.is_empty() {
        return Vec::new();
    }

    let max_weight = weights
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max)
        .max(1e-12);

    weights
        .iter()
        .copied()
        .map(|w| {
            let n_samples = n_epochs as f32 * (w / max_weight);
            if n_samples > 0.0 {
                n_epochs as f32 / n_samples
            } else {
                f32::INFINITY
            }
        })
        .collect()
}

fn clip(value: f32) -> f32 {
    value.clamp(-4.0, 4.0)
}

fn flatten_embedding_rows(embedding: &[Vec<f32>], dim: usize) -> Vec<f32> {
    let mut flat = Vec::with_capacity(embedding.len().saturating_mul(dim));
    for row in embedding {
        debug_assert_eq!(row.len(), dim);
        flat.extend_from_slice(row);
    }
    flat
}

fn copy_flat_embedding_to_rows(flat: &[f32], embedding: &mut [Vec<f32>], dim: usize) {
    for (row_idx, row) in embedding.iter_mut().enumerate() {
        let start = row_idx * dim;
        row.copy_from_slice(&flat[start..start + dim]);
    }
}

#[inline]
fn flat_squared_distance(flat: &[f32], left: usize, right: usize, dim: usize) -> f32 {
    let mut sq = 0.0_f32;
    let left_start = left * dim;
    let right_start = right * dim;
    for d in 0..dim {
        let diff = flat[left_start + d] - flat[right_start + d];
        sq += diff * diff;
    }
    sq
}

fn two_rows_mut<T>(rows: &mut [Vec<T>], i: usize, j: usize) -> (&mut Vec<T>, &mut Vec<T>) {
    assert!(i != j, "indices must be different");
    if i < j {
        let (left, right) = rows.split_at_mut(j);
        (&mut left[i], &mut right[0])
    } else {
        let (left, right) = rows.split_at_mut(i);
        (&mut right[0], &mut left[j])
    }
}

fn is_finite_row(row: &[f32]) -> bool {
    row.iter().all(|v| v.is_finite())
}

fn euclidean_distance_with_grad(x: &[f32], y: &[f32]) -> (f32, Vec<f32>) {
    let mut sq = 0.0_f32;
    for (a, b) in x.iter().zip(y.iter()) {
        let d = *a - *b;
        sq += d * d;
    }
    let dist = sq.sqrt();
    let denom = 1e-6 + dist;
    let grad = x
        .iter()
        .zip(y.iter())
        .map(|(a, b)| (*a - *b) / denom)
        .collect::<Vec<f32>>();
    (dist, grad)
}

struct DensityState {
    original_radii: Vec<f32>,
    start_epoch: usize,
    lambda: f32,
    var_shift: f32,
}

fn density_start_epoch(n_epochs: usize, dens_frac: f32) -> usize {
    let active = ((n_epochs as f32) * dens_frac).ceil() as usize;
    n_epochs.saturating_sub(active.max(1))
}

fn local_radii_from_knn<I, D>(knn_indices: &I, knn_dists: &D, var_shift: f32) -> Vec<f32>
where
    I: IndexRowMatrix + ?Sized,
    D: RowMatrix + ?Sized,
{
    (0..knn_dists.n_rows())
        .map(|row_idx| {
            let idx_row = knn_indices.row(row_idx);
            let dist_row = knn_dists.row(row_idx);
            let mut sum = 0.0_f32;
            let mut count = 0usize;
            for (&neighbor, &dist) in idx_row.iter().zip(dist_row.iter()) {
                if neighbor == row_idx && dist.abs() <= SMOOTH_K_TOLERANCE {
                    continue;
                }
                sum += (dist.max(0.0) + var_shift).ln();
                count += 1;
            }
            if count == 0 {
                var_shift.ln()
            } else {
                sum / count as f32
            }
        })
        .collect()
}

fn embedding_radii_from_knn<I>(embedding: &[Vec<f32>], knn_indices: &I, var_shift: f32) -> Vec<f32>
where
    I: IndexRowMatrix + ?Sized,
{
    (0..embedding.len())
        .map(|row_idx| {
            let idx_row = knn_indices.row(row_idx);
            let mut sum = 0.0_f32;
            let mut count = 0usize;
            for &neighbor in idx_row {
                if neighbor == row_idx {
                    continue;
                }
                let dist = euclidean_distance(&embedding[row_idx], &embedding[neighbor]);
                sum += (dist.max(0.0) + var_shift).ln();
                count += 1;
            }
            if count == 0 {
                var_shift.ln()
            } else {
                sum / count as f32
            }
        })
        .collect()
}

fn apply_density_gradient(
    embedding: &mut [Vec<f32>],
    head: usize,
    tail: usize,
    alpha: f32,
    density: &DensityState,
) {
    if head == tail {
        return;
    }

    let current_dist = euclidean_distance(&embedding[head], &embedding[tail]);
    let target_radius = ((density.original_radii[head] + density.original_radii[tail]) * 0.5)
        .exp()
        .max(1e-6);
    let coeff = DENSITY_GRADIENT_SCALE * density.lambda * (current_dist - target_radius)
        / (current_dist + density.var_shift);
    let (current, other) = two_rows_mut(embedding, head, tail);
    for d in 0..current.len() {
        let direction = (current[d] - other[d]) / (current_dist + 1e-6);
        let grad = clip(coeff * direction);
        current[d] -= grad * alpha;
        other[d] += grad * alpha;
    }
}

#[allow(clippy::too_many_arguments)]
fn optimize_layout_training_flat(
    embedding: &mut [Vec<f32>],
    edges: &[Edge],
    n_epochs: usize,
    a: f32,
    b: f32,
    initial_alpha: f32,
    negative_sample_rate: usize,
    repulsion_strength: f32,
    seed: u64,
) {
    if edges.is_empty() || embedding.is_empty() || n_epochs == 0 {
        return;
    }

    let dim = embedding[0].len();
    let n_vertices = embedding.len();
    let mut flat = flatten_embedding_rows(embedding, dim);

    let heads = edges.iter().map(|edge| edge.head).collect::<Vec<usize>>();
    let tails = edges.iter().map(|edge| edge.tail).collect::<Vec<usize>>();
    let weights = edges.iter().map(|edge| edge.weight).collect::<Vec<f32>>();
    let epochs_per_sample = make_epochs_per_sample(&weights, n_epochs);
    let mut epoch_of_next_sample = epochs_per_sample.clone();

    let epochs_per_negative_sample = epochs_per_sample
        .iter()
        .map(|eps| *eps / negative_sample_rate as f32)
        .collect::<Vec<f32>>();
    let mut epoch_of_next_negative_sample = epochs_per_negative_sample.clone();

    let mut rng = SmallRng::seed_from_u64(seed);
    let repulsion_factor = 2.0 * repulsion_strength * b;
    let attraction_factor = -2.0 * a * b;

    if dim == 2 {
        for epoch in 0..n_epochs {
            let alpha = initial_alpha * (1.0 - epoch as f32 / n_epochs as f32);
            let epoch_f = epoch as f32;

            for edge_idx in 0..edges.len() {
                if epoch_of_next_sample[edge_idx] > epoch_f {
                    continue;
                }

                let head = heads[edge_idx];
                let tail = tails[edge_idx];

                if head != tail {
                    let head_start = head * 2;
                    let tail_start = tail * 2;
                    let dx = flat[head_start] - flat[tail_start];
                    let dy = flat[head_start + 1] - flat[tail_start + 1];
                    let dist_squared = dx * dx + dy * dy;
                    if dist_squared > 0.0 {
                        let dist_pow_b = dist_squared.powf(b);
                        let grad_coeff = attraction_factor * (dist_pow_b / dist_squared)
                            / (a * dist_pow_b + 1.0);
                        let grad_x = clip(grad_coeff * dx);
                        let grad_y = clip(grad_coeff * dy);
                        flat[head_start] += grad_x * alpha;
                        flat[head_start + 1] += grad_y * alpha;
                        flat[tail_start] -= grad_x * alpha;
                        flat[tail_start + 1] -= grad_y * alpha;
                    }
                }

                epoch_of_next_sample[edge_idx] += epochs_per_sample[edge_idx];

                let eps_neg = epochs_per_negative_sample[edge_idx];
                if !eps_neg.is_finite() || eps_neg <= 0.0 {
                    continue;
                }

                let n_neg_samples = ((epoch_f - epoch_of_next_negative_sample[edge_idx]) / eps_neg)
                    .floor()
                    .max(0.0) as usize;

                for _ in 0..n_neg_samples {
                    let neg_idx = rng.gen_range(0..n_vertices);
                    if neg_idx == head {
                        continue;
                    }

                    let head_start = head * 2;
                    let neg_start = neg_idx * 2;
                    let dx = flat[head_start] - flat[neg_start];
                    let dy = flat[head_start + 1] - flat[neg_start + 1];
                    let dist_squared = dx * dx + dy * dy;
                    if dist_squared > 0.0 {
                        let dist_pow_b = dist_squared.powf(b);
                        let grad_coeff =
                            repulsion_factor / ((0.001 + dist_squared) * (a * dist_pow_b + 1.0));
                        if grad_coeff > 0.0 {
                            let grad_x = clip(grad_coeff * dx);
                            let grad_y = clip(grad_coeff * dy);
                            flat[head_start] += grad_x * alpha;
                            flat[head_start + 1] += grad_y * alpha;
                        }
                    }
                }

                epoch_of_next_negative_sample[edge_idx] += n_neg_samples as f32 * eps_neg;
            }
        }

        copy_flat_embedding_to_rows(&flat, embedding, dim);
        return;
    }

    for epoch in 0..n_epochs {
        let alpha = initial_alpha * (1.0 - epoch as f32 / n_epochs as f32);
        let epoch_f = epoch as f32;

        for edge_idx in 0..edges.len() {
            if epoch_of_next_sample[edge_idx] > epoch_f {
                continue;
            }

            let head = heads[edge_idx];
            let tail = tails[edge_idx];

            if head != tail {
                let dist_squared = flat_squared_distance(&flat, head, tail, dim);
                if dist_squared > 0.0 {
                    let dist_pow_b = dist_squared.powf(b);
                    let grad_coeff =
                        attraction_factor * (dist_pow_b / dist_squared) / (a * dist_pow_b + 1.0);
                    let head_start = head * dim;
                    let tail_start = tail * dim;
                    for d in 0..dim {
                        let head_pos = head_start + d;
                        let tail_pos = tail_start + d;
                        let head_value = flat[head_pos];
                        let tail_value = flat[tail_pos];
                        let grad = clip(grad_coeff * (head_value - tail_value));
                        flat[head_pos] = head_value + grad * alpha;
                        flat[tail_pos] = tail_value - grad * alpha;
                    }
                }
            }

            epoch_of_next_sample[edge_idx] += epochs_per_sample[edge_idx];

            let eps_neg = epochs_per_negative_sample[edge_idx];
            if !eps_neg.is_finite() || eps_neg <= 0.0 {
                continue;
            }

            let n_neg_samples = ((epoch_f - epoch_of_next_negative_sample[edge_idx]) / eps_neg)
                .floor()
                .max(0.0) as usize;

            for _ in 0..n_neg_samples {
                let neg_idx = rng.gen_range(0..n_vertices);
                if neg_idx == head {
                    continue;
                }

                let dist_squared = flat_squared_distance(&flat, head, neg_idx, dim);
                if dist_squared > 0.0 {
                    let dist_pow_b = dist_squared.powf(b);
                    let grad_coeff =
                        repulsion_factor / ((0.001 + dist_squared) * (a * dist_pow_b + 1.0));
                    if grad_coeff > 0.0 {
                        let head_start = head * dim;
                        let neg_start = neg_idx * dim;
                        for d in 0..dim {
                            let head_pos = head_start + d;
                            let neg_pos = neg_start + d;
                            let head_value = flat[head_pos];
                            let neg_value = flat[neg_pos];
                            let grad = clip(grad_coeff * (head_value - neg_value));
                            flat[head_pos] = head_value + grad * alpha;
                        }
                    }
                }
            }

            epoch_of_next_negative_sample[edge_idx] += n_neg_samples as f32 * eps_neg;
        }
    }

    copy_flat_embedding_to_rows(&flat, embedding, dim);
}

#[allow(clippy::too_many_arguments)]
fn optimize_layout_training(
    embedding: &mut Vec<Vec<f32>>,
    edges: &[Edge],
    n_epochs: usize,
    a: f32,
    b: f32,
    initial_alpha: f32,
    negative_sample_rate: usize,
    repulsion_strength: f32,
    seed: u64,
    density: Option<&DensityState>,
) {
    if edges.is_empty() || embedding.is_empty() || n_epochs == 0 {
        return;
    }

    if density.is_none() {
        optimize_layout_training_flat(
            embedding.as_mut_slice(),
            edges,
            n_epochs,
            a,
            b,
            initial_alpha,
            negative_sample_rate,
            repulsion_strength,
            seed,
        );
        return;
    }

    let dim = embedding[0].len();
    let n_vertices = embedding.len();

    let weights: Vec<f32> = edges.iter().map(|e| e.weight).collect();
    let epochs_per_sample = make_epochs_per_sample(&weights, n_epochs);
    let mut epoch_of_next_sample = epochs_per_sample.clone();

    let epochs_per_negative_sample: Vec<f32> = epochs_per_sample
        .iter()
        .map(|eps| *eps / negative_sample_rate as f32)
        .collect();
    let mut epoch_of_next_negative_sample = epochs_per_negative_sample.clone();

    let mut rng = SmallRng::seed_from_u64(seed);

    for epoch in 0..n_epochs {
        let alpha = initial_alpha * (1.0 - epoch as f32 / n_epochs as f32);

        for (edge_idx, edge) in edges.iter().enumerate() {
            if epoch_of_next_sample[edge_idx] > epoch as f32 {
                continue;
            }

            let head = edge.head;
            let tail = edge.tail;

            if head != tail {
                let dist_squared = squared_distance(&embedding[head], &embedding[tail]);
                let grad_coeff = if dist_squared > 0.0 {
                    let dist_pow_b = dist_squared.powf(b);
                    -2.0 * a * b * dist_squared.powf(b - 1.0) / (a * dist_pow_b + 1.0)
                } else {
                    0.0
                };

                let (current, other) = two_rows_mut(embedding.as_mut_slice(), head, tail);
                for d in 0..dim {
                    let grad = clip(grad_coeff * (current[d] - other[d]));
                    current[d] += grad * alpha;
                    other[d] -= grad * alpha;
                }
            }
            if let Some(density) = density
                && epoch >= density.start_epoch
            {
                apply_density_gradient(embedding.as_mut_slice(), head, tail, alpha, density);
            }

            epoch_of_next_sample[edge_idx] += epochs_per_sample[edge_idx];

            let eps_neg = epochs_per_negative_sample[edge_idx];
            if !eps_neg.is_finite() || eps_neg <= 0.0 {
                continue;
            }

            let n_neg_samples = ((epoch as f32 - epoch_of_next_negative_sample[edge_idx]) / eps_neg)
                .floor()
                .max(0.0) as usize;

            for _ in 0..n_neg_samples {
                let neg_idx = rng.gen_range(0..n_vertices);
                if neg_idx == head {
                    continue;
                }

                let dist_squared = squared_distance(&embedding[head], &embedding[neg_idx]);

                let grad_coeff = if dist_squared > 0.0 {
                    let dist_pow_b = dist_squared.powf(b);
                    2.0 * repulsion_strength * b / ((0.001 + dist_squared) * (a * dist_pow_b + 1.0))
                } else {
                    0.0
                };

                if grad_coeff > 0.0 {
                    let (current, other) = two_rows_mut(embedding.as_mut_slice(), head, neg_idx);
                    for d in 0..dim {
                        let grad = clip(grad_coeff * (current[d] - other[d]));
                        current[d] += grad * alpha;
                    }
                }
            }

            epoch_of_next_negative_sample[edge_idx] += n_neg_samples as f32 * eps_neg;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn optimize_layout_transform(
    embedding: &mut [Vec<f32>],
    base_embedding: &[Vec<f32>],
    edges: &[Edge],
    n_epochs: usize,
    a: f32,
    b: f32,
    initial_alpha: f32,
    negative_sample_rate: usize,
    repulsion_strength: f32,
    seed: u64,
) {
    if edges.is_empty() || embedding.is_empty() || n_epochs == 0 {
        return;
    }

    let dim = embedding[0].len();
    let n_vertices = base_embedding.len();

    let weights: Vec<f32> = edges.iter().map(|e| e.weight).collect();
    let epochs_per_sample = make_epochs_per_sample(&weights, n_epochs);
    let mut epoch_of_next_sample = epochs_per_sample.clone();

    let epochs_per_negative_sample: Vec<f32> = epochs_per_sample
        .iter()
        .map(|eps| *eps / negative_sample_rate as f32)
        .collect();
    let mut epoch_of_next_negative_sample = epochs_per_negative_sample.clone();

    let mut rng = SmallRng::seed_from_u64(seed);

    for epoch in 0..n_epochs {
        let alpha = initial_alpha * (1.0 - epoch as f32 / n_epochs as f32);

        for (edge_idx, edge) in edges.iter().enumerate() {
            if epoch_of_next_sample[edge_idx] > epoch as f32 {
                continue;
            }

            let head = edge.head;
            let tail = edge.tail;

            if !is_finite_row(&embedding[head]) {
                continue;
            }

            let dist_squared = squared_distance(&embedding[head], &base_embedding[tail]);
            let grad_coeff = if dist_squared > 0.0 {
                let dist_pow_b = dist_squared.powf(b);
                -2.0 * a * b * dist_squared.powf(b - 1.0) / (a * dist_pow_b + 1.0)
            } else {
                0.0
            };

            {
                let current = &mut embedding[head];
                let other = &base_embedding[tail];
                for d in 0..dim {
                    let grad = clip(grad_coeff * (current[d] - other[d]));
                    current[d] += grad * alpha;
                }
            }

            epoch_of_next_sample[edge_idx] += epochs_per_sample[edge_idx];

            let eps_neg = epochs_per_negative_sample[edge_idx];
            if !eps_neg.is_finite() || eps_neg <= 0.0 {
                continue;
            }

            let n_neg_samples = ((epoch as f32 - epoch_of_next_negative_sample[edge_idx]) / eps_neg)
                .floor()
                .max(0.0) as usize;

            for _ in 0..n_neg_samples {
                let neg_idx = rng.gen_range(0..n_vertices);

                let dist_squared = squared_distance(&embedding[head], &base_embedding[neg_idx]);
                let grad_coeff = if dist_squared > 0.0 {
                    let dist_pow_b = dist_squared.powf(b);
                    2.0 * repulsion_strength * b / ((0.001 + dist_squared) * (a * dist_pow_b + 1.0))
                } else {
                    0.0
                };

                if grad_coeff > 0.0 {
                    let current = &mut embedding[head];
                    let other = &base_embedding[neg_idx];
                    for d in 0..dim {
                        let grad = clip(grad_coeff * (current[d] - other[d]));
                        current[d] += grad * alpha;
                    }
                }
            }

            epoch_of_next_negative_sample[edge_idx] += n_neg_samples as f32 * eps_neg;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn optimize_layout_inverse<T: RowMatrix + ?Sized>(
    head_embedding: &mut [Vec<f32>],
    tail_embedding: &T,
    edges: &[Edge],
    fixed_rows: &[bool],
    sigmas: &[f32],
    rhos: &[f32],
    n_epochs: usize,
    initial_alpha: f32,
    negative_sample_rate: usize,
    repulsion_strength: f32,
    seed: u64,
) {
    if edges.is_empty() || head_embedding.is_empty() || n_epochs == 0 {
        return;
    }

    let dim = head_embedding[0].len();
    let n_vertices = tail_embedding.n_rows();

    let weights: Vec<f32> = edges.iter().map(|e| e.weight).collect();
    let epochs_per_sample = make_epochs_per_sample(&weights, n_epochs);
    let mut epoch_of_next_sample = epochs_per_sample.clone();

    let epochs_per_negative_sample: Vec<f32> = epochs_per_sample
        .iter()
        .map(|eps| *eps / negative_sample_rate as f32)
        .collect();
    let mut epoch_of_next_negative_sample = epochs_per_negative_sample.clone();

    let mut rng = SmallRng::seed_from_u64(seed);

    for epoch in 0..n_epochs {
        let alpha = initial_alpha * (1.0 - epoch as f32 / n_epochs as f32);

        for (edge_idx, edge) in edges.iter().enumerate() {
            if epoch_of_next_sample[edge_idx] > epoch as f32 {
                continue;
            }

            let head = edge.head;
            let tail = edge.tail;
            if fixed_rows[head] || !is_finite_row(&head_embedding[head]) {
                continue;
            }

            let (_, grad_dist_output) =
                euclidean_distance_with_grad(&head_embedding[head], tail_embedding.row(tail));

            let wl = edge.weight;
            let grad_coeff = -(1.0 / (wl * sigmas[tail] + 1e-6));

            {
                let current = &mut head_embedding[head];
                for d in 0..dim {
                    let grad_d = clip(grad_coeff * grad_dist_output[d]);
                    current[d] += grad_d * alpha;
                }
            }

            epoch_of_next_sample[edge_idx] += epochs_per_sample[edge_idx];

            let eps_neg = epochs_per_negative_sample[edge_idx];
            if !eps_neg.is_finite() || eps_neg <= 0.0 {
                continue;
            }

            let n_neg_samples = ((epoch as f32 - epoch_of_next_negative_sample[edge_idx]) / eps_neg)
                .floor()
                .max(0.0) as usize;

            for _ in 0..n_neg_samples {
                let neg_tail = rng.gen_range(0..n_vertices);
                let (dist_neg, grad_neg) = euclidean_distance_with_grad(
                    &head_embedding[head],
                    tail_embedding.row(neg_tail),
                );

                let wh = (-(dist_neg - rhos[neg_tail]).max(1e-6) / (sigmas[neg_tail] + 1e-6)).exp();
                let grad_coeff =
                    -repulsion_strength * ((0.0 - wh) / ((1.0 - wh) * sigmas[neg_tail] + 1e-6));

                {
                    let current = &mut head_embedding[head];
                    for d in 0..dim {
                        let grad_d = clip(grad_coeff * grad_neg[d]);
                        current[d] += grad_d * alpha;
                    }
                }
            }

            epoch_of_next_negative_sample[edge_idx] += n_neg_samples as f32 * eps_neg;
        }
    }
}

fn curve_loss(a: f32, b: f32, xv: &[f32], yv: &[f32]) -> f32 {
    let mut loss = 0.0_f32;
    let exponent = 2.0 * b;

    for (&x, &y) in xv.iter().zip(yv.iter()) {
        let model = 1.0 / (1.0 + a * x.powf(exponent));
        let e = model - y;
        loss += e * e;
    }

    loss / xv.len() as f32
}

pub fn find_ab_params(spread: f32, min_dist: f32) -> (f32, f32) {
    let n = 300;
    let xmax = spread * 3.0;

    let mut xv = Vec::with_capacity(n);
    let mut yv = Vec::with_capacity(n);

    for i in 0..n {
        let x = xmax * i as f32 / (n as f32 - 1.0);
        xv.push(x);
        let y = if x < min_dist {
            1.0
        } else {
            (-(x - min_dist) / spread).exp()
        };
        yv.push(y);
    }

    let mut best_a = 1.0_f32;
    let mut best_b = 1.0_f32;
    let mut best_loss = f32::INFINITY;

    for bi in 0..=70 {
        let b = 0.3 + 2.7 * bi as f32 / 70.0;
        for ai in 0..=120 {
            let log_a = -3.0 + 6.0 * ai as f32 / 120.0;
            let a = 10_f32.powf(log_a);
            let loss = curve_loss(a, b, &xv, &yv);
            if loss < best_loss {
                best_loss = loss;
                best_a = a;
                best_b = b;
            }
        }
    }

    let mut step_a = (best_a * 0.35).max(0.05);
    let mut step_b = 0.2_f32;

    for _ in 0..80 {
        let candidates = [
            (best_a + step_a, best_b),
            ((best_a - step_a).max(1e-6), best_b),
            (best_a, best_b + step_b),
            (best_a, (best_b - step_b).max(1e-6)),
            (best_a + step_a, best_b + step_b),
            (best_a + step_a, (best_b - step_b).max(1e-6)),
            ((best_a - step_a).max(1e-6), best_b + step_b),
            ((best_a - step_a).max(1e-6), (best_b - step_b).max(1e-6)),
        ];

        let mut improved = false;
        for (a, b) in candidates {
            let loss = curve_loss(a, b, &xv, &yv);
            if loss < best_loss {
                best_loss = loss;
                best_a = a;
                best_b = b;
                improved = true;
            }
        }

        if !improved {
            step_a *= 0.7;
            step_b *= 0.7;
            if step_a < 1e-5 && step_b < 1e-5 {
                break;
            }
        }
    }

    (best_a, best_b)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_data(n: usize, dim: usize) -> Vec<Vec<f32>> {
        let mut data = Vec::with_capacity(n);
        for i in 0..n {
            let cluster_shift = if i < n / 2 { 0.0 } else { 5.0 };
            let row = (0..dim)
                .map(|d| {
                    let t = (i as f32 + 1.3 * d as f32) / n as f32;
                    cluster_shift + (10.0 * t).sin() * 0.2 + (7.0 * t).cos() * 0.1
                })
                .collect::<Vec<f32>>();
            data.push(row);
        }
        data
    }

    fn sparse_like_data(n: usize, dim: usize) -> Vec<Vec<f32>> {
        let mut data = Vec::with_capacity(n);
        for i in 0..n {
            let mut row = vec![0.0_f32; dim];
            for t in 0..5 {
                let col = (i * 17 + t * 13 + (i % 3) * 7) % dim;
                row[col] = ((i + 1) as f32 * (t + 2) as f32) * 0.017 + col as f32 * 1e-4;
            }
            data.push(row);
        }
        data
    }

    #[test]
    fn default_approximate_knn_params_use_quality_tuned_recall_budget() {
        let params = UmapParams::default();
        assert!(params.use_approximate_knn);
        assert_eq!(params.approx_knn_candidates, 50);
        assert_eq!(params.approx_knn_iters, 14);
    }

    #[test]
    fn auto_approximate_knn_delays_crossover_beyond_public_threshold() {
        let params = UmapParams {
            use_approximate_knn: true,
            approx_knn_threshold: 4096,
            ..UmapParams::default()
        };

        assert!(!should_use_approximate_knn(&params, 4200, 96));
        assert!(!should_use_approximate_knn(&params, 16_384, 160));
        assert!(should_use_approximate_knn(&params, 16_385, 160));
        assert!(!should_use_approximate_knn(&params, 16_384, 16));
        assert!(should_use_approximate_knn(&params, 16_385, 16));

        let forced = UmapParams {
            approx_knn_threshold: 0,
            ..params.clone()
        };
        assert!(should_use_approximate_knn(&forced, 4200, 96));

        let exact = UmapParams {
            use_approximate_knn: false,
            ..params
        };
        assert!(!should_use_approximate_knn(&exact, 100_000, 160));
    }

    fn dense_to_csr(data: &[Vec<f32>]) -> SparseCsrMatrix {
        let n_rows = data.len();
        let n_cols = data[0].len();
        let mut indptr = Vec::with_capacity(n_rows + 1);
        let mut indices = Vec::new();
        let mut values = Vec::new();
        indptr.push(0);
        for row in data {
            for (col, &v) in row.iter().enumerate() {
                if v != 0.0 {
                    indices.push(col);
                    values.push(v);
                }
            }
            indptr.push(indices.len());
        }
        SparseCsrMatrix::new(n_rows, n_cols, indptr, indices, values)
            .expect("dense_to_csr should produce a valid CSR matrix")
    }

    fn disconnected_component_data(
        n_components: usize,
        points_per_component: usize,
    ) -> (Vec<Vec<f32>>, Vec<usize>) {
        let mut data = Vec::with_capacity(n_components * points_per_component);
        let mut labels = Vec::with_capacity(n_components * points_per_component);

        for component in 0..n_components {
            let shift = component as f32 * 50.0;
            for point_idx in 0..points_per_component {
                let t = 2.0 * std::f32::consts::PI * point_idx as f32 / points_per_component as f32;
                data.push(vec![
                    shift + t.cos(),
                    t.sin(),
                    (2.0 * t).cos(),
                    (2.0 * t).sin(),
                    -1.0 + 2.0 * point_idx as f32 / points_per_component as f32,
                    component as f32 * 0.1,
                    0.0,
                    0.0,
                ]);
                labels.push(component);
            }
        }

        (data, labels)
    }

    fn component_centroid(embedding: &[Vec<f32>], labels: &[usize], component: usize) -> Vec<f32> {
        let dim = embedding[0].len();
        let mut centroid = vec![0.0_f32; dim];
        let mut count = 0.0_f32;
        for (row, &label) in embedding.iter().zip(labels.iter()) {
            if label == component {
                count += 1.0;
                for (dim_idx, &value) in row.iter().enumerate() {
                    centroid[dim_idx] += value;
                }
            }
        }
        for value in centroid.iter_mut() {
            *value /= count;
        }
        centroid
    }

    fn component_std_norm(embedding: &[Vec<f32>], labels: &[usize], component: usize) -> f32 {
        let centroid = component_centroid(embedding, labels, component);
        let dim = embedding[0].len();
        let mut variances = vec![0.0_f32; dim];
        let mut count = 0.0_f32;

        for (row, &label) in embedding.iter().zip(labels.iter()) {
            if label == component {
                count += 1.0;
                for dim_idx in 0..dim {
                    let diff = row[dim_idx] - centroid[dim_idx];
                    variances[dim_idx] += diff * diff;
                }
            }
        }

        variances
            .iter()
            .map(|sum_sq| (sum_sq / count).sqrt())
            .map(|std| std * std)
            .sum::<f32>()
            .sqrt()
    }

    fn min_centroid_distance(embedding: &[Vec<f32>], labels: &[usize], n_components: usize) -> f32 {
        let component_centroids = (0..n_components)
            .map(|component| component_centroid(embedding, labels, component))
            .collect::<Vec<Vec<f32>>>();

        let mut min_distance = f32::INFINITY;
        for i in 0..component_centroids.len() {
            for j in (i + 1)..component_centroids.len() {
                min_distance = min_distance.min(euclidean_distance(
                    &component_centroids[i],
                    &component_centroids[j],
                ));
            }
        }
        min_distance
    }

    fn max_component_std(embedding: &[Vec<f32>], labels: &[usize], n_components: usize) -> f32 {
        (0..n_components)
            .map(|component| component_std_norm(embedding, labels, component))
            .fold(0.0_f32, f32::max)
    }

    fn separation_ratio(embedding: &[Vec<f32>], labels: &[usize], n_components: usize) -> f32 {
        let min_sep = min_centroid_distance(embedding, labels, n_components);
        let max_std = max_component_std(embedding, labels, n_components);
        min_sep / (max_std + 1e-6)
    }

    fn assert_all_finite(points: &[Vec<f32>]) {
        assert!(
            points
                .iter()
                .flat_map(|row| row.iter())
                .all(|v| v.is_finite())
        );
    }

    fn knn_recall_excluding_self(exact: &[Vec<usize>], approx: &[Vec<usize>]) -> f32 {
        assert_eq!(exact.len(), approx.len());
        let mut matched = 0usize;
        let mut total = 0usize;
        for (exact_row, approx_row) in exact.iter().zip(approx.iter()) {
            assert_eq!(exact_row.len(), approx_row.len());
            if exact_row.len() <= 1 {
                continue;
            }
            let exact_set = exact_row
                .iter()
                .skip(1)
                .copied()
                .collect::<HashSet<usize>>();
            for idx in approx_row.iter().skip(1) {
                if exact_set.contains(idx) {
                    matched += 1;
                }
                total += 1;
            }
        }
        if total == 0 {
            1.0
        } else {
            matched as f32 / total as f32
        }
    }

    #[test]
    fn fit_transform_returns_expected_shape_and_finite_values() {
        let data = synthetic_data(80, 8);
        let params = UmapParams {
            n_neighbors: 12,
            n_components: 2,
            n_epochs: Some(60),
            init: InitMethod::Random,
            random_seed: 7,
            ..UmapParams::default()
        };

        let mut model = UmapModel::new(params);
        let embedding = model
            .fit_transform(&data)
            .expect("fit_transform should succeed");

        assert_eq!(embedding.len(), 80);
        assert_eq!(embedding[0].len(), 2);
        assert_all_finite(&embedding);
    }

    #[test]
    fn expanded_dense_metric_distances_match_reference_values() {
        let x = vec![1.0_f32, -2.0, 0.0, 4.0];
        let y = vec![-1.0_f32, 2.0, 0.0, 1.0];

        assert!((chebyshev_distance(&x, &y) - 4.0).abs() <= 1e-6);
        assert!((minkowski_distance(&x, &y, 3.0) - 99.0_f32.powf(1.0 / 3.0)).abs() <= 1e-6);
        assert!((canberra_distance(&x, &y) - 2.6).abs() <= 1e-6);
        assert!((bray_curtis_distance(&x, &y) - 1.8).abs() <= 1e-6);

        let corr_x = vec![1.0_f32, 2.0, 3.0];
        let corr_y = vec![1.0_f32, 5.0, 7.0];
        let expected_corr_distance = 1.0 - 6.0 / (2.0_f32.sqrt() * (168.0_f32 / 9.0).sqrt());
        assert!((correlation_distance(&corr_x, &corr_y) - expected_corr_distance).abs() <= 1e-6);

        let constant = vec![2.0_f32, 2.0, 2.0];
        assert_eq!(correlation_distance(&constant, &constant), 0.0);
        assert_eq!(correlation_distance(&constant, &corr_y), 1.0);
    }

    #[test]
    fn expanded_dense_metrics_fit_and_transform() {
        let data = synthetic_data(72, 7);
        let query = data.iter().take(8).cloned().collect::<Vec<_>>();

        for metric in [
            Metric::Chebyshev,
            Metric::Minkowski { p: 3.0 },
            Metric::Correlation,
            Metric::Canberra,
            Metric::BrayCurtis,
        ] {
            let params = UmapParams {
                n_neighbors: 8,
                n_components: 2,
                n_epochs: Some(30),
                metric,
                init: InitMethod::Random,
                random_seed: 113,
                use_approximate_knn: false,
                ..UmapParams::default()
            };
            let mut model = UmapModel::new(params);
            let embedding = model
                .fit_transform(&data)
                .expect("expanded metric fit should succeed");
            let transformed = model
                .transform(&query)
                .expect("expanded metric transform should succeed");

            assert_eq!(embedding.len(), data.len());
            assert_eq!(transformed.len(), query.len());
            assert_all_finite(&embedding);
            assert_all_finite(&transformed);
        }
    }

    #[test]
    fn densmap_lambda_zero_matches_standard_umap() {
        let data = synthetic_data(72, 7);
        let params = UmapParams {
            n_neighbors: 8,
            n_components: 2,
            n_epochs: Some(30),
            init: InitMethod::Random,
            random_seed: 141,
            use_approximate_knn: false,
            ..UmapParams::default()
        };
        let dens_params = UmapParams {
            densmap: true,
            dens_lambda: 0.0,
            ..params.clone()
        };

        let mut standard = UmapModel::new(params);
        let standard_embedding = standard
            .fit_transform(&data)
            .expect("standard fit should succeed");

        let mut dens = UmapModel::new(dens_params);
        let dens_embedding = dens
            .fit_transform(&data)
            .expect("zero-lambda densmap fit should succeed");

        assert_eq!(dens_embedding, standard_embedding);
    }

    #[test]
    fn output_dens_returns_finite_radii() {
        let data = synthetic_data(64, 6);
        let params = UmapParams {
            n_neighbors: 8,
            n_components: 2,
            n_epochs: Some(30),
            init: InitMethod::Random,
            random_seed: 143,
            use_approximate_knn: false,
            output_dens: true,
            ..UmapParams::default()
        };

        let mut model = UmapModel::new(params);
        let embedding = model
            .fit_transform(&data)
            .expect("output_dens fit should succeed");
        let radii_original = model
            .radii_original()
            .expect("original radii should be stored");
        let radii_embedding = model
            .radii_embedding()
            .expect("embedding radii should be stored");

        assert_eq!(embedding.len(), data.len());
        assert_eq!(radii_original.len(), data.len());
        assert_eq!(radii_embedding.len(), data.len());
        assert!(radii_original.iter().all(|v| v.is_finite()));
        assert!(radii_embedding.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn densmap_rejects_invalid_density_params() {
        let data = synthetic_data(32, 5);
        for params in [
            UmapParams {
                dens_lambda: -1.0,
                ..UmapParams::default()
            },
            UmapParams {
                dens_frac: 1.5,
                ..UmapParams::default()
            },
            UmapParams {
                dens_var_shift: 0.0,
                ..UmapParams::default()
            },
        ] {
            let mut model = UmapModel::new(UmapParams {
                n_neighbors: 6,
                n_epochs: Some(20),
                init: InitMethod::Random,
                use_approximate_knn: false,
                ..params
            });
            let err = model
                .fit_transform(&data)
                .expect_err("invalid density params should fail");
            assert!(matches!(err, UmapError::InvalidParameter(_)));
        }
    }

    #[test]
    fn deterministic_with_same_seed() {
        let data = synthetic_data(60, 6);
        let params = UmapParams {
            n_neighbors: 10,
            n_components: 2,
            n_epochs: Some(40),
            init: InitMethod::Random,
            random_seed: 1234,
            ..UmapParams::default()
        };

        let mut model_1 = UmapModel::new(params.clone());
        let mut model_2 = UmapModel::new(params);

        let emb_1 = model_1.fit_transform(&data).expect("model 1 fit failed");
        let emb_2 = model_2.fit_transform(&data).expect("model 2 fit failed");

        assert_eq!(emb_1, emb_2);
    }

    #[test]
    fn spectral_init_and_transform_work() {
        let data = synthetic_data(120, 5);
        let query = data.iter().take(15).cloned().collect::<Vec<_>>();

        let params = UmapParams {
            n_neighbors: 15,
            n_components: 2,
            n_epochs: Some(80),
            init: InitMethod::Spectral,
            random_seed: 9,
            ..UmapParams::default()
        };

        let mut model = UmapModel::new(params);
        let embedding = model.fit_transform(&data).expect("fit should succeed");
        assert_eq!(embedding.len(), 120);

        let transformed = model.transform(&query).expect("transform should succeed");
        assert_eq!(transformed.len(), 15);
        assert_eq!(transformed[0].len(), 2);
        assert_all_finite(&transformed);
    }

    #[test]
    fn transform_requires_fit() {
        let model = UmapModel::new(UmapParams::default());
        let query = synthetic_data(4, 3);
        let err = model.transform(&query).expect_err("must fail before fit");
        assert!(matches!(err, UmapError::NotFitted));
    }

    #[test]
    fn inverse_transform_workflow() {
        let data = synthetic_data(140, 6);
        let params = UmapParams {
            n_neighbors: 15,
            n_components: 2,
            n_epochs: Some(90),
            init: InitMethod::Spectral,
            random_seed: 11,
            ..UmapParams::default()
        };

        let mut model = UmapModel::new(params);
        let embedding = model.fit_transform(&data).expect("fit should succeed");

        let emb_query = embedding.iter().skip(130).cloned().collect::<Vec<_>>();
        let reconstructed = model
            .inverse_transform(&emb_query)
            .expect("inverse_transform should succeed");

        assert_eq!(reconstructed.len(), emb_query.len());
        assert_eq!(reconstructed[0].len(), 6);
        assert_all_finite(&reconstructed);
    }

    #[test]
    fn dense_flat_api_matches_row_api() {
        let data = synthetic_data(128, 6);
        let flat_data = data
            .iter()
            .flat_map(|row| row.iter().copied())
            .collect::<Vec<f32>>();
        let params = UmapParams {
            n_neighbors: 15,
            n_components: 2,
            n_epochs: Some(80),
            init: InitMethod::Spectral,
            random_seed: 19,
            use_approximate_knn: false,
            ..UmapParams::default()
        };

        let mut row_model = UmapModel::new(params.clone());
        let row_embedding = row_model
            .fit_transform(&data)
            .expect("row fit should succeed");

        let mut flat_model = UmapModel::new(params);
        let flat_embedding = flat_model
            .fit_transform_dense(&flat_data, data.len(), data[0].len())
            .expect("flat fit should succeed");

        let row_embedding_flat = row_embedding
            .iter()
            .flat_map(|row| row.iter().copied())
            .collect::<Vec<f32>>();
        assert_eq!(flat_embedding.as_slice(), row_embedding_flat.as_slice());

        let query = data.iter().take(16).cloned().collect::<Vec<_>>();
        let flat_query = query
            .iter()
            .flat_map(|row| row.iter().copied())
            .collect::<Vec<f32>>();

        let row_transformed = row_model
            .transform(&query)
            .expect("row transform should succeed");
        let flat_transformed = flat_model
            .transform_dense(&flat_query, query.len(), query[0].len())
            .expect("flat transform should succeed");
        let row_transformed_flat = row_transformed
            .iter()
            .flat_map(|row| row.iter().copied())
            .collect::<Vec<f32>>();
        assert_eq!(flat_transformed.as_slice(), row_transformed_flat.as_slice());

        let row_emb_query = row_embedding.iter().skip(120).cloned().collect::<Vec<_>>();
        let flat_emb_query = row_emb_query
            .iter()
            .flat_map(|row| row.iter().copied())
            .collect::<Vec<f32>>();

        let row_inverse = row_model
            .inverse_transform(&row_emb_query)
            .expect("row inverse should succeed");
        let flat_inverse = flat_model
            .inverse_transform_dense(&flat_emb_query, row_emb_query.len(), row_emb_query[0].len())
            .expect("flat inverse should succeed");
        let row_inverse_flat = row_inverse
            .iter()
            .flat_map(|row| row.iter().copied())
            .collect::<Vec<f32>>();
        assert_eq!(flat_inverse.as_slice(), row_inverse_flat.as_slice());
    }

    #[test]
    fn inverse_transform_requires_fit() {
        let model = UmapModel::new(UmapParams::default());
        let emb_query = vec![vec![0.1_f32, 0.2_f32]];
        let err = model
            .inverse_transform(&emb_query)
            .expect_err("must fail before fit");
        assert!(matches!(err, UmapError::NotFitted));
    }

    #[test]
    fn sparse_csr_knn_matches_dense_euclidean() {
        let data = sparse_like_data(42, 61);
        let csr = dense_to_csr(&data);
        let k = 12;

        let (dense_idx, dense_dist) = exact_nearest_neighbors(&data, k, Metric::Euclidean);
        let (sparse_idx, sparse_dist) = sparse::exact_nearest_neighbors(&csr, k, Metric::Euclidean);

        assert_eq!(dense_idx, sparse_idx);
        for (row_dense, row_sparse) in dense_dist.iter().zip(sparse_dist.iter()) {
            for (&lhs, &rhs) in row_dense.iter().zip(row_sparse.iter()) {
                assert!(
                    (lhs - rhs).abs() <= 1e-5,
                    "distance mismatch: dense={lhs}, sparse={rhs}"
                );
            }
        }
    }

    #[test]
    fn sparse_csr_fit_transform_and_transform_work() {
        let data = sparse_like_data(72, 80);
        let query = data.iter().take(10).cloned().collect::<Vec<Vec<f32>>>();
        let csr = dense_to_csr(&data);

        let params = UmapParams {
            n_neighbors: 10,
            n_components: 2,
            n_epochs: Some(50),
            metric: Metric::Euclidean,
            init: InitMethod::Random,
            random_seed: 2026,
            use_approximate_knn: false,
            ..UmapParams::default()
        };

        let mut model = UmapModel::new(params);
        let embedding = model
            .fit_transform_sparse_csr(csr)
            .expect("sparse fit should succeed");
        assert_eq!(embedding.len(), data.len());
        assert_eq!(embedding[0].len(), 2);
        assert_all_finite(&embedding);

        let transformed = model
            .transform(&query)
            .expect("sparse transform should succeed");
        assert_eq!(transformed.len(), query.len());
        assert_eq!(transformed[0].len(), 2);
        assert_all_finite(&transformed);
    }

    #[test]
    fn sparse_csr_query_transform_matches_dense_query_transform() {
        let data = sparse_like_data(72, 80);
        let query = data.iter().skip(4).take(10).cloned().collect::<Vec<_>>();
        let csr = dense_to_csr(&data);
        let query_csr = dense_to_csr(&query);

        let params = UmapParams {
            n_neighbors: 10,
            n_components: 2,
            n_epochs: Some(50),
            metric: Metric::Cosine,
            init: InitMethod::Random,
            random_seed: 2027,
            use_approximate_knn: false,
            ..UmapParams::default()
        };

        let mut model = UmapModel::new(params);
        model
            .fit_transform_sparse_csr(csr)
            .expect("sparse fit should succeed");

        let dense_query_embedding = model
            .transform(&query)
            .expect("dense query transform should succeed");
        let sparse_query_embedding = model
            .transform_sparse_csr(query_csr)
            .expect("sparse query transform should succeed");

        assert_eq!(dense_query_embedding.len(), sparse_query_embedding.len());
        for (dense_row, sparse_row) in dense_query_embedding
            .iter()
            .zip(sparse_query_embedding.iter())
        {
            assert_eq!(dense_row.len(), sparse_row.len());
            for (&dense_value, &sparse_value) in dense_row.iter().zip(sparse_row.iter()) {
                assert!(
                    (dense_value - sparse_value).abs() <= 1e-6,
                    "dense query transform {dense_value} != sparse query transform {sparse_value}"
                );
            }
        }
    }

    #[test]
    fn sparse_csr_fit_transform_supports_manhattan_metric() {
        let data = sparse_like_data(68, 90);
        let csr = dense_to_csr(&data);
        let params = UmapParams {
            n_neighbors: 10,
            n_components: 2,
            n_epochs: Some(50),
            metric: Metric::Manhattan,
            init: InitMethod::Random,
            random_seed: 2048,
            use_approximate_knn: false,
            ..UmapParams::default()
        };

        let mut model = UmapModel::new(params);
        let embedding = model
            .fit_transform_sparse_csr(csr)
            .expect("sparse manhattan fit should succeed");
        assert_eq!(embedding.len(), data.len());
        assert_eq!(embedding[0].len(), 2);
        assert_all_finite(&embedding);
    }

    #[test]
    fn sparse_csr_fit_transform_supports_cosine_metric() {
        let data = sparse_like_data(66, 88);
        let query = data.iter().skip(56).cloned().collect::<Vec<Vec<f32>>>();
        let csr = dense_to_csr(&data);
        let params = UmapParams {
            n_neighbors: 10,
            n_components: 2,
            n_epochs: Some(50),
            metric: Metric::Cosine,
            init: InitMethod::Random,
            random_seed: 4096,
            use_approximate_knn: false,
            ..UmapParams::default()
        };

        let mut model = UmapModel::new(params);
        let embedding = model
            .fit_transform_sparse_csr(csr)
            .expect("sparse cosine fit should succeed");
        assert_eq!(embedding.len(), data.len());
        assert_eq!(embedding[0].len(), 2);
        assert_all_finite(&embedding);

        let transformed = model
            .transform(&query)
            .expect("sparse cosine transform should succeed");
        assert_eq!(transformed.len(), query.len());
        assert_eq!(transformed[0].len(), 2);
        assert_all_finite(&transformed);
    }

    #[test]
    fn sparse_csr_fit_transform_supports_expanded_sparse_metrics() {
        let data = sparse_like_data(64, 16);
        let query = data.iter().take(8).cloned().collect::<Vec<Vec<f32>>>();

        for metric in [
            Metric::Chebyshev,
            Metric::Minkowski { p: 3.0 },
            Metric::Correlation,
        ] {
            let csr = dense_to_csr(&data);
            let params = UmapParams {
                n_neighbors: 8,
                n_components: 2,
                n_epochs: Some(35),
                metric,
                init: InitMethod::Random,
                random_seed: 131,
                use_approximate_knn: false,
                ..UmapParams::default()
            };

            let mut model = UmapModel::new(params);
            let embedding = model
                .fit_transform_sparse_csr(csr)
                .expect("expanded sparse metric fit should succeed");
            let transformed = model
                .transform(&query)
                .expect("expanded sparse metric transform should succeed");

            assert_eq!(embedding.len(), data.len());
            assert_eq!(embedding[0].len(), 2);
            assert_eq!(transformed.len(), query.len());
            assert_all_finite(&embedding);
            assert_all_finite(&transformed);
        }
    }

    #[test]
    fn sparse_csr_rejects_dense_only_metrics() {
        let data = sparse_like_data(32, 12);

        for metric in [Metric::Canberra, Metric::BrayCurtis] {
            let csr = dense_to_csr(&data);
            let params = UmapParams {
                n_neighbors: 6,
                n_components: 2,
                n_epochs: Some(20),
                metric,
                init: InitMethod::Random,
                random_seed: 137,
                use_approximate_knn: false,
                ..UmapParams::default()
            };

            let mut model = UmapModel::new(params);
            let err = model
                .fit_transform_sparse_csr(csr)
                .expect_err("dense-only metric should fail for sparse input");

            assert!(
                err.to_string().contains("dense input only"),
                "unexpected error message: {err}"
            );
        }
    }

    #[test]
    fn sparse_csr_rejects_densmap_and_output_dens() {
        let data = sparse_like_data(32, 12);

        for params in [
            UmapParams {
                densmap: true,
                ..UmapParams::default()
            },
            UmapParams {
                output_dens: true,
                ..UmapParams::default()
            },
        ] {
            let csr = dense_to_csr(&data);
            let mut model = UmapModel::new(UmapParams {
                n_neighbors: 6,
                n_components: 2,
                n_epochs: Some(20),
                init: InitMethod::Random,
                random_seed: 139,
                use_approximate_knn: false,
                ..params
            });
            let err = model
                .fit_transform_sparse_csr(csr)
                .expect_err("sparse densmap/output_dens should fail");
            assert!(
                err.to_string().contains("dense input only"),
                "unexpected error message: {err}"
            );
        }
    }

    #[test]
    fn sparse_csr_random_fit_matches_dense_fit() {
        let data = sparse_like_data(64, 97);
        let csr = dense_to_csr(&data);
        let params = UmapParams {
            n_neighbors: 10,
            n_components: 2,
            n_epochs: Some(40),
            metric: Metric::Euclidean,
            init: InitMethod::Random,
            random_seed: 77,
            use_approximate_knn: false,
            ..UmapParams::default()
        };

        let (knn_idx, knn_dist) =
            sparse::exact_nearest_neighbors(&csr, params.n_neighbors, Metric::Euclidean);

        let mut dense_model = UmapModel::new(params.clone());
        let dense_embedding = dense_model
            .fit_transform_with_knn_metric(&data, &knn_idx, &knn_dist, Metric::Euclidean)
            .expect("dense fit should succeed");

        let mut sparse_model = UmapModel::new(params);
        let sparse_embedding = sparse_model
            .fit_transform_sparse_csr(csr)
            .expect("sparse fit should succeed");

        assert_eq!(dense_embedding, sparse_embedding);
    }

    #[test]
    fn fit_rejects_non_finite_input_values() {
        let mut data = synthetic_data(32, 6);
        data[7][3] = f32::NAN;
        let mut model = UmapModel::new(UmapParams {
            n_neighbors: 8,
            n_components: 2,
            n_epochs: Some(20),
            init: InitMethod::Random,
            random_seed: 77,
            ..UmapParams::default()
        });
        let err = model
            .fit_transform(&data)
            .expect_err("non-finite input must be rejected");
        assert!(matches!(err, UmapError::InvalidParameter(_)));
        assert!(
            err.to_string().contains("non-finite"),
            "unexpected error message: {err}"
        );
    }

    #[test]
    fn fit_rejects_non_finite_hyperparameters() {
        let data = synthetic_data(32, 6);
        let mut params = UmapParams {
            n_neighbors: 8,
            n_components: 2,
            n_epochs: Some(20),
            init: InitMethod::Random,
            random_seed: 77,
            ..UmapParams::default()
        };
        params.learning_rate = f32::INFINITY;

        let mut model = UmapModel::new(params);
        let err = model
            .fit_transform(&data)
            .expect_err("non-finite hyperparameter must be rejected");
        assert!(matches!(err, UmapError::InvalidParameter(_)));
        assert!(
            err.to_string().contains("learning_rate must be finite"),
            "unexpected error message: {err}"
        );
    }

    #[test]
    fn approximate_knn_path_runs() {
        let data = synthetic_data(90, 7);
        let params = UmapParams {
            n_neighbors: 12,
            n_components: 2,
            n_epochs: Some(50),
            init: InitMethod::Random,
            random_seed: 99,
            use_approximate_knn: true,
            approx_knn_candidates: 24,
            approx_knn_iters: 6,
            approx_knn_threshold: 0,
            ..UmapParams::default()
        };

        let mut model = UmapModel::new(params);
        let embedding = model
            .fit_transform(&data)
            .expect("fit should succeed on ANN path");
        assert_eq!(embedding.len(), 90);
        assert_eq!(embedding[0].len(), 2);
        assert_all_finite(&embedding);
    }

    #[test]
    fn approximate_knn_recall_reasonable_euclidean() {
        let data = synthetic_data(240, 8);
        let k = 12;
        let (exact_idx, _) = exact_nearest_neighbors(&data, k, Metric::Euclidean);
        let (approx_idx, _) = approximate_nearest_neighbors(
            &data,
            k,
            Metric::Euclidean,
            30,
            8,
            0x1234_5678_9ABC_DEF0,
        );
        let recall = knn_recall_excluding_self(&exact_idx, &approx_idx);
        assert!(
            recall >= 0.70,
            "euclidean ann recall is too low: got {recall:.4}"
        );
    }

    #[test]
    fn approximate_knn_recall_reasonable_cosine() {
        let data = synthetic_data(220, 9);
        let k = 10;
        let (exact_idx, _) = exact_nearest_neighbors(&data, k, Metric::Cosine);
        let (approx_idx, _) =
            approximate_nearest_neighbors(&data, k, Metric::Cosine, 28, 8, 0xCAFEBABE_DEADC0DE);
        let recall = knn_recall_excluding_self(&exact_idx, &approx_idx);
        assert!(
            recall >= 0.65,
            "cosine ann recall is too low: got {recall:.4}"
        );
    }

    #[test]
    fn approximate_knn_recall_diagnostics_report_tail_quality() {
        let data = synthetic_data(180, 8);
        let params = UmapParams {
            n_neighbors: 10,
            metric: Metric::Euclidean,
            random_seed: 123,
            approx_knn_candidates: 28,
            approx_knn_iters: 8,
            ..UmapParams::default()
        };

        let diagnostics =
            approximate_knn_recall_diagnostics(&data, params).expect("diagnostics should compute");

        assert_eq!(diagnostics.n_samples, data.len());
        assert_eq!(diagnostics.n_features, data[0].len());
        assert_eq!(diagnostics.n_neighbors, 10);
        assert_eq!(diagnostics.metric, Metric::Euclidean);
        assert_eq!(diagnostics.per_row_recall.len(), data.len());
        assert!(diagnostics.mean_recall >= diagnostics.worst_decile_recall);
        assert!(diagnostics.worst_decile_recall >= diagnostics.min_recall);
        assert!(diagnostics.mean_recall >= 0.65);
        assert!(
            diagnostics
                .per_row_recall
                .iter()
                .all(|&recall| (0.0..=1.0).contains(&recall))
        );
    }

    #[test]
    fn fit_transform_supports_manhattan_metric() {
        let data = synthetic_data(96, 8);
        let params = UmapParams {
            n_neighbors: 12,
            n_components: 2,
            n_epochs: Some(60),
            metric: Metric::Manhattan,
            init: InitMethod::Random,
            random_seed: 321,
            use_approximate_knn: false,
            ..UmapParams::default()
        };

        let mut model = UmapModel::new(params);
        let embedding = model
            .fit_transform(&data)
            .expect("fit should succeed for manhattan metric");

        assert_eq!(embedding.len(), data.len());
        assert_eq!(embedding[0].len(), 2);
        assert_all_finite(&embedding);
    }

    #[test]
    fn approximate_knn_path_supports_cosine_metric() {
        let data = synthetic_data(84, 10);
        let params = UmapParams {
            n_neighbors: 10,
            n_components: 2,
            n_epochs: Some(50),
            metric: Metric::Cosine,
            init: InitMethod::Random,
            random_seed: 222,
            use_approximate_knn: true,
            approx_knn_candidates: 24,
            approx_knn_iters: 6,
            approx_knn_threshold: 0,
            ..UmapParams::default()
        };

        let mut model = UmapModel::new(params);
        let embedding = model
            .fit_transform(&data)
            .expect("fit should succeed for cosine ANN path");

        assert_eq!(embedding.len(), data.len());
        assert_eq!(embedding[0].len(), 2);
        assert_all_finite(&embedding);
    }

    #[test]
    fn precomputed_cosine_knn_matches_direct_fit() {
        let data = synthetic_data(72, 7);
        let params = UmapParams {
            n_neighbors: 12,
            n_components: 2,
            n_epochs: Some(50),
            metric: Metric::Cosine,
            init: InitMethod::Random,
            random_seed: 444,
            use_approximate_knn: false,
            ..UmapParams::default()
        };

        let mut direct_model = UmapModel::new(params.clone());
        let direct_embedding = direct_model
            .fit_transform(&data)
            .expect("direct cosine fit should succeed");

        let (knn_indices, knn_dists) =
            exact_nearest_neighbors(&data, params.n_neighbors, params.metric);
        let mut precomputed_model = UmapModel::new(params);
        let precomputed_embedding = precomputed_model
            .fit_transform_with_knn(&data, &knn_indices, &knn_dists)
            .expect("precomputed cosine fit should succeed");

        assert_eq!(direct_embedding, precomputed_embedding);
        assert_all_finite(&precomputed_embedding);
    }

    #[test]
    fn precomputed_dense_flat_knn_matches_row_path() {
        let data = synthetic_data(72, 7);
        let params = UmapParams {
            n_neighbors: 12,
            n_components: 2,
            n_epochs: Some(50),
            metric: Metric::Euclidean,
            init: InitMethod::Random,
            random_seed: 445,
            use_approximate_knn: false,
            ..UmapParams::default()
        };

        let (knn_indices, knn_dists) =
            exact_nearest_neighbors(&data, params.n_neighbors, Metric::Euclidean);
        let flat_data = data.iter().flatten().copied().collect::<Vec<_>>();
        let flat_knn_indices = knn_indices.iter().flatten().copied().collect::<Vec<_>>();
        let flat_knn_dists = knn_dists.iter().flatten().copied().collect::<Vec<_>>();

        let mut row_model = UmapModel::new(params.clone());
        let row_embedding = row_model
            .fit_transform_with_knn_metric(&data, &knn_indices, &knn_dists, Metric::Euclidean)
            .expect("row precomputed fit should succeed");

        let mut flat_model = UmapModel::new(params);
        let flat_embedding = flat_model
            .fit_transform_with_knn_metric_dense_flat(
                &flat_data,
                data.len(),
                data[0].len(),
                &flat_knn_indices,
                &flat_knn_dists,
                knn_indices[0].len(),
                Metric::Euclidean,
            )
            .expect("flat precomputed fit should succeed");

        let row_embedding_flat = row_embedding.iter().flatten().copied().collect::<Vec<_>>();
        assert_eq!(flat_embedding.n_rows(), row_embedding.len());
        assert_eq!(flat_embedding.n_cols(), row_embedding[0].len());
        assert_eq!(flat_embedding.as_slice(), row_embedding_flat.as_slice());
    }

    #[test]
    fn precomputed_dense_i64_flat_knn_matches_row_path() {
        let data = synthetic_data(72, 7);
        let params = UmapParams {
            n_neighbors: 12,
            n_components: 2,
            n_epochs: Some(50),
            metric: Metric::Euclidean,
            init: InitMethod::Random,
            random_seed: 446,
            use_approximate_knn: false,
            ..UmapParams::default()
        };

        let (knn_indices, knn_dists) =
            exact_nearest_neighbors(&data, params.n_neighbors, Metric::Euclidean);
        let flat_data = data.iter().flatten().copied().collect::<Vec<_>>();
        let flat_knn_indices = knn_indices
            .iter()
            .flatten()
            .map(|&idx| idx as i64)
            .collect::<Vec<_>>();
        let flat_knn_dists = knn_dists.iter().flatten().copied().collect::<Vec<_>>();

        let mut row_model = UmapModel::new(params.clone());
        let row_embedding = row_model
            .fit_transform_with_knn_metric(&data, &knn_indices, &knn_dists, Metric::Euclidean)
            .expect("row precomputed fit should succeed");

        let mut flat_model = UmapModel::new(params);
        let flat_embedding = flat_model
            .fit_transform_with_knn_metric_dense_i64_flat(
                &flat_data,
                data.len(),
                data[0].len(),
                &flat_knn_indices,
                knn_indices.len(),
                knn_indices[0].len(),
                &flat_knn_dists,
                knn_dists.len(),
                knn_dists[0].len(),
                Metric::Euclidean,
                true,
            )
            .expect("i64 flat precomputed fit should succeed");

        let row_embedding_flat = row_embedding.iter().flatten().copied().collect::<Vec<_>>();
        assert_eq!(flat_embedding.n_rows(), row_embedding.len());
        assert_eq!(flat_embedding.n_cols(), row_embedding[0].len());
        assert_eq!(flat_embedding.as_slice(), row_embedding_flat.as_slice());
    }

    #[test]
    fn smooth_knn_dist_handles_rows_without_self_neighbor() {
        let dists_with_self = vec![vec![0.0, 0.2, 0.4, 0.8], vec![0.0, 0.1, 0.5, 0.9]];
        let dists_without_self = vec![vec![0.2, 0.4, 0.8], vec![0.1, 0.5, 0.9]];

        let (sigmas_self, rhos_self) =
            smooth_knn_dist(&dists_with_self, 3.0, 1.0, DEFAULT_BANDWIDTH, true);
        let (sigmas_no_self, rhos_no_self) =
            smooth_knn_dist(&dists_without_self, 3.0, 1.0, DEFAULT_BANDWIDTH, false);

        for (&lhs, &rhs) in sigmas_self.iter().zip(sigmas_no_self.iter()) {
            assert!(
                (lhs - rhs).abs() <= 1e-5,
                "sigma mismatch for self/non-self rows: {lhs} vs {rhs}"
            );
        }
        for (&lhs, &rhs) in rhos_self.iter().zip(rhos_no_self.iter()) {
            assert!(
                (lhs - rhs).abs() <= 1e-6,
                "rho mismatch for self/non-self rows: {lhs} vs {rhs}"
            );
        }
    }

    #[test]
    fn smooth_knn_dist_with_zero_rows_matches_self_and_non_self_layouts() {
        let dists_with_self = vec![vec![0.0, 0.0, 0.0, 0.0], vec![0.0, 1.0, 2.0, 3.0]];
        let dists_without_self = vec![vec![0.0, 0.0, 0.0], vec![1.0, 2.0, 3.0]];

        let (sigmas_self, rhos_self) =
            smooth_knn_dist(&dists_with_self, 3.0, 1.0, DEFAULT_BANDWIDTH, true);
        let (sigmas_no_self, rhos_no_self) =
            smooth_knn_dist(&dists_without_self, 3.0, 1.0, DEFAULT_BANDWIDTH, false);

        for (&lhs, &rhs) in sigmas_self.iter().zip(sigmas_no_self.iter()) {
            assert!(
                (lhs - rhs).abs() <= 1e-6,
                "sigma mismatch for self/non-self rows with zero row: {lhs} vs {rhs}"
            );
        }
        for (&lhs, &rhs) in rhos_self.iter().zip(rhos_no_self.iter()) {
            assert!(
                (lhs - rhs).abs() <= 1e-6,
                "rho mismatch for self/non-self rows with zero row: {lhs} vs {rhs}"
            );
        }
    }

    #[test]
    fn inverse_neighbor_count_respects_model_n_neighbors() {
        assert_eq!(inverse_neighbor_count(3, 100), 3);
        assert_eq!(inverse_neighbor_count(15, 7), 7);
        assert_eq!(inverse_neighbor_count(2, 2), 2);
        assert_eq!(inverse_neighbor_count(4, 1), 1);
    }

    #[test]
    fn precomputed_euclidean_knn_without_self_stays_finite() {
        let data = synthetic_data(80, 8);
        let params = UmapParams {
            n_neighbors: 10,
            n_components: 2,
            n_epochs: Some(60),
            metric: Metric::Euclidean,
            init: InitMethod::Random,
            random_seed: 123,
            use_approximate_knn: false,
            ..UmapParams::default()
        };

        let (knn_idx_with_self, knn_dist_with_self) =
            exact_nearest_neighbors(&data, params.n_neighbors + 1, Metric::Euclidean);

        let mut knn_idx_no_self = Vec::with_capacity(data.len());
        let mut knn_dist_no_self = Vec::with_capacity(data.len());
        for (row_idx, (idx_row, dist_row)) in knn_idx_with_self
            .iter()
            .zip(knn_dist_with_self.iter())
            .enumerate()
        {
            let mut idx_out = Vec::with_capacity(params.n_neighbors);
            let mut dist_out = Vec::with_capacity(params.n_neighbors);
            for (&idx, &dist) in idx_row.iter().zip(dist_row.iter()) {
                if idx == row_idx && dist.abs() <= SMOOTH_K_TOLERANCE {
                    continue;
                }
                idx_out.push(idx);
                dist_out.push(dist);
                if idx_out.len() == params.n_neighbors {
                    break;
                }
            }
            assert_eq!(
                idx_out.len(),
                params.n_neighbors,
                "failed to build non-self knn row {row_idx}"
            );
            knn_idx_no_self.push(idx_out);
            knn_dist_no_self.push(dist_out);
        }

        let mut model = UmapModel::new(params);
        let embedding = model
            .fit_transform_with_knn_metric(
                &data,
                &knn_idx_no_self,
                &knn_dist_no_self,
                Metric::Euclidean,
            )
            .expect("precomputed non-self knn fit should succeed");

        assert_eq!(embedding.len(), data.len());
        assert_eq!(embedding[0].len(), 2);
        assert_all_finite(&embedding);
    }

    #[test]
    fn precomputed_knn_rejects_invalid_distances() {
        let data = synthetic_data(64, 6);
        let params = UmapParams {
            n_neighbors: 10,
            n_components: 2,
            n_epochs: Some(40),
            metric: Metric::Euclidean,
            init: InitMethod::Random,
            random_seed: 2027,
            use_approximate_knn: false,
            ..UmapParams::default()
        };
        let (knn_idx, mut knn_dist) =
            exact_nearest_neighbors(&data, params.n_neighbors, Metric::Euclidean);

        knn_dist[0][1] = -0.5;
        let mut model = UmapModel::new(params.clone());
        let err = model
            .fit_transform_with_knn_metric(&data, &knn_idx, &knn_dist, Metric::Euclidean)
            .expect_err("negative distance must be rejected");
        assert!(matches!(err, UmapError::InvalidParameter(_)));

        let (knn_idx2, mut knn_dist2) =
            exact_nearest_neighbors(&data, params.n_neighbors, Metric::Euclidean);
        knn_dist2[0][2] = f32::NAN;
        let mut model = UmapModel::new(params.clone());
        let err = model
            .fit_transform_with_knn_metric(&data, &knn_idx2, &knn_dist2, Metric::Euclidean)
            .expect_err("non-finite distance must be rejected");
        assert!(matches!(err, UmapError::InvalidParameter(_)));

        let (knn_idx3, mut knn_dist3) =
            exact_nearest_neighbors(&data, params.n_neighbors, Metric::Euclidean);
        knn_dist3[0].swap(2, 5);
        let mut model = UmapModel::new(params);
        let err = model
            .fit_transform_with_knn_metric(&data, &knn_idx3, &knn_dist3, Metric::Euclidean)
            .expect_err("unsorted distances must be rejected");
        assert!(matches!(err, UmapError::InvalidParameter(_)));
    }

    #[test]
    fn precomputed_knn_metric_mismatch_is_rejected() {
        let data = synthetic_data(64, 6);
        let params = UmapParams {
            n_neighbors: 12,
            n_components: 2,
            n_epochs: Some(50),
            metric: Metric::Cosine,
            init: InitMethod::Random,
            random_seed: 2026,
            use_approximate_knn: false,
            ..UmapParams::default()
        };

        let (knn_indices, knn_dists) =
            exact_nearest_neighbors(&data, params.n_neighbors, Metric::Euclidean);
        let mut model = UmapModel::new(params);
        let err = model
            .fit_transform_with_knn_metric(&data, &knn_indices, &knn_dists, Metric::Euclidean)
            .expect_err("metric mismatch should fail");
        assert!(matches!(err, UmapError::InvalidParameter(_)));
        assert!(
            err.to_string().contains("precomputed knn metric"),
            "unexpected error message: {err}"
        );
    }

    #[test]
    fn categorical_target_graph_uses_same_label_edges_and_skips_unlabeled() {
        let target = SupervisedTarget::categorical(vec![0, 0, -1, 1, 1], 0.5, Some(1));

        let directed = build_categorical_target_membership_strengths(&target, 5, 3);

        assert_eq!(
            directed,
            vec![
                Edge {
                    head: 0,
                    tail: 1,
                    weight: 1.0,
                },
                Edge {
                    head: 1,
                    tail: 0,
                    weight: 1.0,
                },
                Edge {
                    head: 3,
                    tail: 4,
                    weight: 1.0,
                },
                Edge {
                    head: 4,
                    tail: 3,
                    weight: 1.0,
                },
            ]
        );
    }

    #[test]
    fn combine_fuzzy_graphs_blends_data_and_target_edges() {
        let data_edges = vec![
            Edge {
                head: 0,
                tail: 1,
                weight: 0.25,
            },
            Edge {
                head: 1,
                tail: 2,
                weight: 0.5,
            },
        ];
        let target_edges = vec![
            Edge {
                head: 0,
                tail: 1,
                weight: 1.0,
            },
            Edge {
                head: 2,
                tail: 3,
                weight: 1.0,
            },
        ];

        let combined = combine_fuzzy_graphs(&data_edges, &target_edges, 0.4);

        let expected = [
            Edge {
                head: 0,
                tail: 1,
                weight: 0.55,
            },
            Edge {
                head: 1,
                tail: 2,
                weight: 0.3,
            },
            Edge {
                head: 2,
                tail: 3,
                weight: 0.4,
            },
        ];
        assert_eq!(combined.len(), expected.len());
        for (actual, expected) in combined.iter().zip(expected.iter()) {
            assert_eq!((actual.head, actual.tail), (expected.head, expected.tail));
            assert!(
                (actual.weight - expected.weight).abs() <= 1e-6,
                "weight mismatch for edge {} -> {}: {} vs {}",
                actual.head,
                actual.tail,
                actual.weight,
                expected.weight
            );
        }
        assert_eq!(
            combine_fuzzy_graphs(&data_edges, &target_edges, 0.0),
            data_edges
        );
    }

    #[test]
    fn categorical_target_intersection_modulates_existing_feature_edges() {
        let data_edges = vec![
            Edge {
                head: 0,
                tail: 1,
                weight: 0.8,
            },
            Edge {
                head: 1,
                tail: 0,
                weight: 0.8,
            },
            Edge {
                head: 0,
                tail: 2,
                weight: 0.7,
            },
            Edge {
                head: 2,
                tail: 0,
                weight: 0.7,
            },
            Edge {
                head: 1,
                tail: 2,
                weight: 0.4,
            },
            Edge {
                head: 2,
                tail: 1,
                weight: 0.4,
            },
            Edge {
                head: 2,
                tail: 3,
                weight: 0.8,
            },
            Edge {
                head: 3,
                tail: 2,
                weight: 0.8,
            },
            Edge {
                head: 0,
                tail: 3,
                weight: 0.6,
            },
            Edge {
                head: 3,
                tail: 0,
                weight: 0.6,
            },
        ];
        let target = SupervisedTarget::categorical(vec![0, 0, 1, 1], 0.6, None);

        let intersected = apply_categorical_target_intersection(&data_edges, &target);
        let existing_pairs = data_edges
            .iter()
            .map(|edge| (edge.head, edge.tail))
            .collect::<HashSet<_>>();
        assert!(
            intersected
                .iter()
                .all(|edge| existing_pairs.contains(&(edge.head, edge.tail)))
        );

        let weight = |head, tail| {
            intersected
                .iter()
                .find(|edge| edge.head == head && edge.tail == tail)
                .map(|edge| edge.weight)
                .unwrap_or(0.0)
        };
        assert!(weight(0, 1) > weight(0, 2));
        assert!(weight(1, 0) > weight(1, 2));
        assert!(weight(2, 3) > weight(2, 0));
        assert!(weight(3, 2) > weight(3, 0));
    }

    #[test]
    fn categorical_target_weight_zero_matches_unsupervised_embedding() {
        let data = synthetic_data(80, 8);
        let labels = (0..data.len())
            .map(|idx| if idx < data.len() / 2 { 0 } else { 1 })
            .collect::<Vec<i64>>();
        let params = UmapParams {
            n_neighbors: 10,
            n_components: 2,
            n_epochs: Some(40),
            init: InitMethod::Random,
            random_seed: 101,
            use_approximate_knn: false,
            ..UmapParams::default()
        };

        let mut unsupervised = UmapModel::new(params.clone());
        let base = unsupervised
            .fit_transform(&data)
            .expect("unsupervised fit should succeed");

        let mut supervised = UmapModel::new(params);
        let target = SupervisedTarget::categorical(labels, 0.0, None);
        let with_zero_weight = supervised
            .fit_transform_supervised(&data, &target)
            .expect("zero-weight supervised fit should succeed");

        assert_eq!(base, with_zero_weight);
    }

    #[test]
    fn categorical_supervision_changes_embedding_when_weight_positive() {
        let data = synthetic_data(80, 8);
        let labels = (0..data.len())
            .map(|idx| if idx < data.len() / 2 { 0 } else { 1 })
            .collect::<Vec<i64>>();
        let params = UmapParams {
            n_neighbors: 10,
            n_components: 2,
            n_epochs: Some(40),
            init: InitMethod::Random,
            random_seed: 103,
            use_approximate_knn: false,
            ..UmapParams::default()
        };

        let mut unsupervised = UmapModel::new(params.clone());
        let base = unsupervised
            .fit_transform(&data)
            .expect("unsupervised fit should succeed");

        let mut supervised = UmapModel::new(params);
        let target = SupervisedTarget::categorical(labels, 0.7, None);
        let supervised_embedding = supervised
            .fit_transform_supervised(&data, &target)
            .expect("categorical supervised fit should succeed");

        assert_ne!(base, supervised_embedding);
        assert_all_finite(&supervised_embedding);
    }

    #[test]
    fn categorical_supervision_rejects_label_length_mismatch() {
        let data = synthetic_data(32, 6);
        let params = UmapParams {
            n_neighbors: 6,
            n_components: 2,
            n_epochs: Some(20),
            init: InitMethod::Random,
            random_seed: 107,
            use_approximate_knn: false,
            ..UmapParams::default()
        };
        let target = SupervisedTarget::categorical(vec![0; data.len() - 1], 0.5, None);

        let mut model = UmapModel::new(params);
        let err = model
            .fit_transform_supervised(&data, &target)
            .expect_err("mismatched y length should fail");

        assert!(
            err.to_string().contains("target label length"),
            "unexpected error message: {err}"
        );
    }

    #[test]
    fn categorical_supervision_accepts_unlabeled_minus_one() {
        let data = synthetic_data(48, 6);
        let labels = (0..data.len())
            .map(|idx| {
                if idx % 5 == 0 {
                    -1
                } else if idx < data.len() / 2 {
                    0
                } else {
                    1
                }
            })
            .collect::<Vec<i64>>();
        let params = UmapParams {
            n_neighbors: 8,
            n_components: 2,
            n_epochs: Some(30),
            init: InitMethod::Random,
            random_seed: 109,
            use_approximate_knn: false,
            ..UmapParams::default()
        };
        let target = SupervisedTarget::categorical(labels, 0.5, None);

        let mut model = UmapModel::new(params);
        let embedding = model
            .fit_transform_supervised(&data, &target)
            .expect("-1 labels should be treated as unlabeled");

        assert_eq!(embedding.len(), data.len());
        assert_all_finite(&embedding);
    }

    #[test]
    fn normalized_laplacian_from_edges_matches_dense_affinity_path() {
        let n_samples = 5;
        let edges = vec![
            Edge {
                head: 0,
                tail: 1,
                weight: 0.4,
            },
            Edge {
                head: 1,
                tail: 0,
                weight: 0.7,
            },
            Edge {
                head: 1,
                tail: 2,
                weight: 0.8,
            },
            Edge {
                head: 2,
                tail: 3,
                weight: 0.6,
            },
            Edge {
                head: 3,
                tail: 4,
                weight: 0.2,
            },
            Edge {
                head: 4,
                tail: 3,
                weight: 0.5,
            },
            Edge {
                head: 0,
                tail: 4,
                weight: 0.3,
            },
        ];

        let mut affinity = vec![vec![0.0_f64; n_samples]; n_samples];
        for edge in &edges {
            if edge.head == edge.tail {
                continue;
            }
            let val = edge.weight.max(0.0) as f64;
            if val > affinity[edge.head][edge.tail] {
                affinity[edge.head][edge.tail] = val;
            }
        }
        let mut i = 0usize;
        while i < n_samples {
            let (head, tail) = affinity.split_at_mut(i + 1);
            let row_i = head
                .last_mut()
                .expect("split_at_mut(i + 1) must leave row i in the head slice");
            for (offset, row_j) in tail.iter_mut().enumerate() {
                let j = i + 1 + offset;
                let sym = row_i[j].max(row_j[i]);
                row_i[j] = sym;
                row_j[i] = sym;
            }
            i += 1;
        }

        let from_edges =
            normalized_laplacian_from_edges(n_samples, &edges).expect("edge laplacian");
        let from_affinity =
            normalized_laplacian_from_affinity(&affinity).expect("affinity laplacian");

        for i in 0..n_samples {
            for j in 0..n_samples {
                let diff = (from_edges[(i, j)] - from_affinity[(i, j)]).abs();
                assert!(
                    diff < 1e-12,
                    "laplacian mismatch at ({i}, {j}): {} vs {}",
                    from_edges[(i, j)],
                    from_affinity[(i, j)]
                );
            }
        }
    }

    #[test]
    fn spectral_init_disconnected_not_random_fallback() {
        let (data, labels) = disconnected_component_data(4, 60);
        let n_neighbors = 10;
        let (knn_indices, knn_dists) =
            exact_nearest_neighbors(&data, n_neighbors, Metric::Euclidean);
        let (sigmas, rhos) =
            smooth_knn_dist(&knn_dists, n_neighbors as f32, 1.0, DEFAULT_BANDWIDTH, true);
        let directed = compute_membership_strengths(&knn_indices, &knn_dists, &sigmas, &rhos);
        let edges = symmetrize_fuzzy_graph(&directed, 1.0);

        let spectral = spectral_init(&data, 2, &edges, 42).expect("spectral init should succeed");
        let random = random_init(data.len(), 2, 42, -10.0, 10.0);

        assert_ne!(
            spectral, random,
            "spectral disconnected initialization should not degenerate to pure random init"
        );

        let spectral_ratio = separation_ratio(&spectral, &labels, 4);
        let random_ratio = separation_ratio(&random, &labels, 4);
        assert!(
            spectral_ratio > random_ratio * 1.2,
            "expected disconnected spectral init to separate components better than random: spectral={spectral_ratio}, random={random_ratio}"
        );
    }

    #[test]
    fn sparse_spectral_init_disconnected_not_random_fallback() {
        let (data, labels) = disconnected_component_data(4, 60);
        let csr = dense_to_csr(&data);
        let n_neighbors = 10;
        let (knn_indices, knn_dists) =
            sparse::exact_nearest_neighbors(&csr, n_neighbors, Metric::Euclidean);
        let (sigmas, rhos) =
            smooth_knn_dist(&knn_dists, n_neighbors as f32, 1.0, DEFAULT_BANDWIDTH, true);
        let directed = compute_membership_strengths(&knn_indices, &knn_dists, &sigmas, &rhos);
        let edges = symmetrize_fuzzy_graph(&directed, 1.0);

        let spectral =
            spectral_init_sparse(&csr, 2, &edges, 42).expect("sparse spectral init should succeed");
        let random = random_init(data.len(), 2, 42, -10.0, 10.0);

        assert_ne!(
            spectral, random,
            "sparse disconnected spectral initialization should not degenerate to pure random init"
        );

        let spectral_ratio = separation_ratio(&spectral, &labels, 4);
        let random_ratio = separation_ratio(&random, &labels, 4);
        assert!(
            spectral_ratio > random_ratio * 1.2,
            "expected sparse disconnected spectral init to separate components better than random: spectral={spectral_ratio}, random={random_ratio}"
        );
    }

    #[test]
    fn spectral_init_connected_raw_iterative_is_deterministic() {
        let data = synthetic_data(SPECTRAL_ITERATIVE_COMPONENT_THRESHOLD + 48, 6);
        let n_neighbors = 15;
        let (knn_indices, knn_dists) =
            exact_nearest_neighbors(&data, n_neighbors, Metric::Euclidean);
        let (sigmas, rhos) =
            smooth_knn_dist(&knn_dists, n_neighbors as f32, 1.0, DEFAULT_BANDWIDTH, true);
        let directed = compute_membership_strengths(&knn_indices, &knn_dists, &sigmas, &rhos);
        let edges = symmetrize_fuzzy_graph(&directed, 1.0);

        let coords_a = spectral_init_connected_raw(data.len(), 2, &edges, 77)
            .expect("iterative raw spectral init should succeed");
        let coords_b = spectral_init_connected_raw(data.len(), 2, &edges, 77)
            .expect("iterative raw spectral init should be reproducible");

        assert_eq!(coords_a, coords_b);
        assert_all_finite(&coords_a);
    }

    #[test]
    fn spectral_init_large_disconnected_components_stays_structured() {
        let component_size = SPECTRAL_ITERATIVE_COMPONENT_THRESHOLD + 24;
        let (data, labels) = disconnected_component_data(3, component_size);
        let n_neighbors = 12;
        let (knn_indices, knn_dists) =
            exact_nearest_neighbors(&data, n_neighbors, Metric::Euclidean);
        let (sigmas, rhos) =
            smooth_knn_dist(&knn_dists, n_neighbors as f32, 1.0, DEFAULT_BANDWIDTH, true);
        let directed = compute_membership_strengths(&knn_indices, &knn_dists, &sigmas, &rhos);
        let edges = symmetrize_fuzzy_graph(&directed, 1.0);

        let spectral = spectral_init(&data, 2, &edges, 42).expect("spectral init should succeed");
        let random = random_init(data.len(), 2, 42, -10.0, 10.0);

        assert_all_finite(&spectral);
        let spectral_ratio = separation_ratio(&spectral, &labels, 3);
        let random_ratio = separation_ratio(&random, &labels, 3);
        assert!(
            spectral_ratio > random_ratio * 1.2,
            "expected large disconnected spectral init to separate components better than random: spectral={spectral_ratio}, random={random_ratio}"
        );
    }

    #[test]
    fn spectral_init_handles_disconnected_components() {
        let (data, labels) = disconnected_component_data(4, 60);
        let params = UmapParams {
            n_neighbors: 10,
            n_components: 2,
            n_epochs: Some(60),
            init: InitMethod::Spectral,
            random_seed: 42,
            use_approximate_knn: false,
            ..UmapParams::default()
        };

        let mut model = UmapModel::new(params);
        let embedding = model.fit_transform(&data).expect("fit should succeed");

        let max_std = max_component_std(&embedding, &labels, 4);
        let min_distance = min_centroid_distance(&embedding, &labels, 4);
        let sep_ratio = separation_ratio(&embedding, &labels, 4);

        assert!(
            max_std > 0.05,
            "expected each disconnected component to keep non-trivial internal spread, got {max_std}"
        );
        assert!(
            min_distance > 0.8,
            "expected disconnected component centroids to remain separated after optimization, got {min_distance}"
        );
        assert!(
            sep_ratio > 1.5,
            "expected inter-component separation ratio to stay robust, got {sep_ratio}"
        );
    }

    #[test]
    fn sparse_spectral_init_handles_disconnected_components() {
        let (data, labels) = disconnected_component_data(4, 60);
        let csr = dense_to_csr(&data);
        let params = UmapParams {
            n_neighbors: 10,
            n_components: 2,
            n_epochs: Some(60),
            metric: Metric::Euclidean,
            init: InitMethod::Spectral,
            random_seed: 42,
            use_approximate_knn: false,
            ..UmapParams::default()
        };

        let mut model = UmapModel::new(params);
        let embedding = model
            .fit_transform_sparse_csr(csr)
            .expect("sparse fit should succeed");

        let max_std = max_component_std(&embedding, &labels, 4);
        let min_distance = min_centroid_distance(&embedding, &labels, 4);
        let sep_ratio = separation_ratio(&embedding, &labels, 4);

        assert!(
            max_std > 0.05,
            "expected each sparse disconnected component to keep non-trivial internal spread, got {max_std}"
        );
        assert!(
            min_distance > 0.8,
            "expected sparse disconnected component centroids to remain separated after optimization, got {min_distance}"
        );
        assert!(
            sep_ratio > 1.1,
            "expected sparse inter-component separation ratio to stay robust, got {sep_ratio}"
        );
    }
}
