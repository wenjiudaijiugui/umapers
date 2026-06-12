use crate::{UmapError, UmapModel, UmapParams};
use std::error::Error;
use std::fmt::{Display, Formatter};

const DEFAULT_ALIGNMENT_REGULARIZATION: f32 = 0.08;
const DEFAULT_ALIGNMENT_LEARNING_RATE: f32 = 0.25;
const DEFAULT_RECENTER_INTERVAL: usize = 5;

#[derive(Debug, Clone)]
pub struct AlignmentRelation {
    pairs: Vec<(usize, usize)>,
}

impl AlignmentRelation {
    pub fn new(mut pairs: Vec<(usize, usize)>) -> Self {
        pairs.sort_unstable();
        pairs.dedup();
        Self { pairs }
    }

    pub fn identity(n_samples: usize) -> Self {
        let mut pairs = Vec::with_capacity(n_samples);
        for idx in 0..n_samples {
            pairs.push((idx, idx));
        }
        Self { pairs }
    }

    pub fn from_forward_map(map: &[Option<usize>]) -> Self {
        let mut pairs = Vec::new();
        for (left_idx, maybe_right_idx) in map.iter().enumerate() {
            if let Some(right_idx) = maybe_right_idx {
                pairs.push((left_idx, *right_idx));
            }
        }
        Self::new(pairs)
    }

    pub fn pairs(&self) -> &[(usize, usize)] {
        &self.pairs
    }

    pub fn len(&self) -> usize {
        self.pairs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pairs.is_empty()
    }
}

#[derive(Debug, Clone)]
pub struct AlignedUmapParams {
    pub umap: UmapParams,
    pub alignment_regularization: f32,
    pub alignment_learning_rate: f32,
    pub alignment_epochs: Option<usize>,
    pub recenter_interval: usize,
}

impl Default for AlignedUmapParams {
    fn default() -> Self {
        Self {
            umap: UmapParams::default(),
            alignment_regularization: DEFAULT_ALIGNMENT_REGULARIZATION,
            alignment_learning_rate: DEFAULT_ALIGNMENT_LEARNING_RATE,
            alignment_epochs: None,
            recenter_interval: DEFAULT_RECENTER_INTERVAL,
        }
    }
}

#[derive(Debug)]
pub enum AlignedUmapError {
    EmptyDatasets,
    NeedAtLeastTwoDatasets,
    EmptySlice {
        slice: usize,
    },
    FeatureMismatch {
        slice: usize,
        expected: usize,
        got: usize,
    },
    RelationCountMismatch {
        expected: usize,
        got: usize,
    },
    RelationIndexOutOfRange {
        relation: usize,
        left_idx: usize,
        right_idx: usize,
        left_len: usize,
        right_len: usize,
    },
    DatasetSampleMismatch {
        left_slice: usize,
        right_slice: usize,
        left_samples: usize,
        right_samples: usize,
    },
    InvalidParameter(String),
    Umap(UmapError),
}

impl Display for AlignedUmapError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            AlignedUmapError::EmptyDatasets => write!(f, "input datasets are empty"),
            AlignedUmapError::NeedAtLeastTwoDatasets => {
                write!(f, "need at least two slices for aligned UMAP")
            }
            AlignedUmapError::EmptySlice { slice } => {
                write!(f, "slice {slice} is empty")
            }
            AlignedUmapError::FeatureMismatch {
                slice,
                expected,
                got,
            } => {
                write!(
                    f,
                    "slice {slice} feature mismatch: expected {expected}, got {got}"
                )
            }
            AlignedUmapError::RelationCountMismatch { expected, got } => {
                write!(
                    f,
                    "relation count mismatch: expected {expected} relations for adjacent slices, got {got}"
                )
            }
            AlignedUmapError::RelationIndexOutOfRange {
                relation,
                left_idx,
                right_idx,
                left_len,
                right_len,
            } => {
                write!(
                    f,
                    "relation {relation} pair ({left_idx}, {right_idx}) out of range (left_len={left_len}, right_len={right_len})"
                )
            }
            AlignedUmapError::DatasetSampleMismatch {
                left_slice,
                right_slice,
                left_samples,
                right_samples,
            } => {
                write!(
                    f,
                    "identity alignment requires equal sample counts; slices {left_slice} and {right_slice} have {left_samples} and {right_samples} samples"
                )
            }
            AlignedUmapError::InvalidParameter(msg) => write!(f, "invalid parameter: {msg}"),
            AlignedUmapError::Umap(err) => write!(f, "underlying umap error: {err}"),
        }
    }
}

impl Error for AlignedUmapError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Umap(err) => Some(err),
            _ => None,
        }
    }
}

impl From<UmapError> for AlignedUmapError {
    fn from(value: UmapError) -> Self {
        Self::Umap(value)
    }
}

#[derive(Debug, Clone)]
struct PreparedRelation {
    left_slice: usize,
    right_slice: usize,
    pairs: Vec<(usize, usize)>,
    left_inv_degree: Vec<f32>,
    right_inv_degree: Vec<f32>,
}

#[derive(Debug, Clone)]
struct EmbeddingSystemStats {
    centroid: Vec<f32>,
    rms_radius: f32,
}

#[derive(Debug, Clone)]
pub struct AlignedUmapModel {
    params: AlignedUmapParams,
    embeddings: Option<Vec<Vec<Vec<f32>>>>,
}

impl AlignedUmapModel {
    pub fn new(params: AlignedUmapParams) -> Self {
        Self {
            params,
            embeddings: None,
        }
    }

    pub fn params(&self) -> &AlignedUmapParams {
        &self.params
    }

    pub fn embeddings(&self) -> Option<&[Vec<Vec<f32>>]> {
        self.embeddings.as_deref()
    }

    pub fn fit_transform_identity(
        &mut self,
        datasets: &[Vec<Vec<f32>>],
    ) -> Result<Vec<Vec<Vec<f32>>>, AlignedUmapError> {
        let relations = build_identity_relations(datasets)?;
        self.fit_transform(datasets, &relations)
    }

    pub fn fit(
        &mut self,
        datasets: &[Vec<Vec<f32>>],
        relations: &[AlignmentRelation],
    ) -> Result<(), AlignedUmapError> {
        self.fit_transform(datasets, relations).map(|_| ())
    }

    pub fn fit_transform(
        &mut self,
        datasets: &[Vec<Vec<f32>>],
        relations: &[AlignmentRelation],
    ) -> Result<Vec<Vec<Vec<f32>>>, AlignedUmapError> {
        validate_aligned_params(&self.params)?;
        validate_datasets(datasets)?;

        let expected_relations = datasets.len() - 1;
        if relations.len() != expected_relations {
            return Err(AlignedUmapError::RelationCountMismatch {
                expected: expected_relations,
                got: relations.len(),
            });
        }

        let mut embeddings = Vec::with_capacity(datasets.len());
        for (slice_idx, slice_data) in datasets.iter().enumerate() {
            let mut params = self.params.umap.clone();
            params.random_seed = derive_slice_seed(params.random_seed, slice_idx as u64 + 1);
            let mut model = UmapModel::new(params);
            embeddings.push(model.fit_transform(slice_data)?);
        }

        let prepared_relations = prepare_relations(datasets, relations)?;
        if self.params.alignment_regularization > 0.0 && !prepared_relations.is_empty() {
            warmstart_embeddings_with_relations(&mut embeddings, &prepared_relations);
            optimize_aligned_embeddings(&mut embeddings, &prepared_relations, &self.params);
        }

        self.embeddings = Some(embeddings.clone());
        Ok(embeddings)
    }
}

fn derive_slice_seed(base_seed: u64, slice_ordinal: u64) -> u64 {
    let mut z = base_seed ^ slice_ordinal.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn validate_aligned_params(params: &AlignedUmapParams) -> Result<(), AlignedUmapError> {
    if !params.alignment_regularization.is_finite() || params.alignment_regularization < 0.0 {
        return Err(AlignedUmapError::InvalidParameter(
            "alignment_regularization must be finite and >= 0".to_string(),
        ));
    }
    if !params.alignment_learning_rate.is_finite() || params.alignment_learning_rate <= 0.0 {
        return Err(AlignedUmapError::InvalidParameter(
            "alignment_learning_rate must be finite and > 0".to_string(),
        ));
    }
    if let Some(epochs) = params.alignment_epochs
        && epochs == 0
    {
        return Err(AlignedUmapError::InvalidParameter(
            "alignment_epochs must be >= 1 when provided".to_string(),
        ));
    }
    if params.recenter_interval == 0 {
        return Err(AlignedUmapError::InvalidParameter(
            "recenter_interval must be >= 1".to_string(),
        ));
    }
    Ok(())
}

fn validate_datasets(datasets: &[Vec<Vec<f32>>]) -> Result<(), AlignedUmapError> {
    if datasets.is_empty() {
        return Err(AlignedUmapError::EmptyDatasets);
    }
    if datasets.len() < 2 {
        return Err(AlignedUmapError::NeedAtLeastTwoDatasets);
    }

    let first_slice = &datasets[0];
    if first_slice.is_empty() {
        return Err(AlignedUmapError::EmptySlice { slice: 0 });
    }
    let expected_features = first_slice[0].len();

    for (slice_idx, slice) in datasets.iter().enumerate() {
        if slice.is_empty() {
            return Err(AlignedUmapError::EmptySlice { slice: slice_idx });
        }
        for row in slice {
            if row.len() != expected_features {
                return Err(AlignedUmapError::FeatureMismatch {
                    slice: slice_idx,
                    expected: expected_features,
                    got: row.len(),
                });
            }
        }
    }

    Ok(())
}

fn build_identity_relations(
    datasets: &[Vec<Vec<f32>>],
) -> Result<Vec<AlignmentRelation>, AlignedUmapError> {
    let mut relations = Vec::with_capacity(datasets.len().saturating_sub(1));
    for slice_idx in 0..datasets.len().saturating_sub(1) {
        let left_len = datasets[slice_idx].len();
        let right_len = datasets[slice_idx + 1].len();
        if left_len != right_len {
            return Err(AlignedUmapError::DatasetSampleMismatch {
                left_slice: slice_idx,
                right_slice: slice_idx + 1,
                left_samples: left_len,
                right_samples: right_len,
            });
        }
        relations.push(AlignmentRelation::identity(left_len));
    }
    Ok(relations)
}

fn prepare_relations(
    datasets: &[Vec<Vec<f32>>],
    relations: &[AlignmentRelation],
) -> Result<Vec<PreparedRelation>, AlignedUmapError> {
    let mut out = Vec::with_capacity(relations.len());

    for (relation_idx, relation) in relations.iter().enumerate() {
        let left_slice = relation_idx;
        let right_slice = relation_idx + 1;
        let left_len = datasets[left_slice].len();
        let right_len = datasets[right_slice].len();

        let mut valid_pairs = Vec::with_capacity(relation.len());
        let mut left_degree = vec![0usize; left_len];
        let mut right_degree = vec![0usize; right_len];

        for &(left_idx, right_idx) in relation.pairs() {
            if left_idx >= left_len || right_idx >= right_len {
                return Err(AlignedUmapError::RelationIndexOutOfRange {
                    relation: relation_idx,
                    left_idx,
                    right_idx,
                    left_len,
                    right_len,
                });
            }
            valid_pairs.push((left_idx, right_idx));
            left_degree[left_idx] += 1;
            right_degree[right_idx] += 1;
        }

        if valid_pairs.is_empty() {
            continue;
        }

        let left_inv_degree = left_degree
            .into_iter()
            .map(|deg| if deg == 0 { 0.0 } else { 1.0 / deg as f32 })
            .collect::<Vec<f32>>();

        let right_inv_degree = right_degree
            .into_iter()
            .map(|deg| if deg == 0 { 0.0 } else { 1.0 / deg as f32 })
            .collect::<Vec<f32>>();

        out.push(PreparedRelation {
            left_slice,
            right_slice,
            pairs: valid_pairs,
            left_inv_degree,
            right_inv_degree,
        });
    }

    Ok(out)
}

fn optimize_aligned_embeddings(
    embeddings: &mut Vec<Vec<Vec<f32>>>,
    relations: &[PreparedRelation],
    params: &AlignedUmapParams,
) {
    if embeddings.is_empty() {
        return;
    }

    let n_epochs = alignment_epoch_count(params, embeddings);
    if n_epochs == 0 {
        return;
    }

    let n_components = embeddings[0][0].len();
    let mut deltas = embeddings
        .iter()
        .map(|slice| vec![0.0_f32; slice.len() * n_components])
        .collect::<Vec<Vec<f32>>>();

    let Some(target_stats) = embedding_system_stats(embeddings.as_slice()) else {
        return;
    };

    let recenter_interval = params.recenter_interval;

    for epoch in 0..n_epochs {
        for delta in deltas.iter_mut() {
            delta.fill(0.0);
        }

        let step_size = cosine_decay_alignment_lr(params.alignment_learning_rate, epoch, n_epochs);
        if step_size <= 0.0 {
            break;
        }

        let reg = params.alignment_regularization;

        for relation in relations {
            let left_slice_idx = relation.left_slice;
            let right_slice_idx = relation.right_slice;

            let left_embedding = &embeddings[left_slice_idx];
            let right_embedding = &embeddings[right_slice_idx];
            let (left_delta, right_delta) =
                two_vectors_mut(deltas.as_mut_slice(), left_slice_idx, right_slice_idx);

            for &(left_idx, right_idx) in relation.pairs.iter() {
                let left_row = &left_embedding[left_idx];
                let right_row = &right_embedding[right_idx];
                let left_scale = relation.left_inv_degree[left_idx];
                let right_scale = relation.right_inv_degree[right_idx];

                let left_offset = left_idx * n_components;
                let right_offset = right_idx * n_components;

                for dim in 0..n_components {
                    let diff = left_row[dim] - right_row[dim];
                    let grad = reg * diff;
                    left_delta[left_offset + dim] -= grad * left_scale;
                    right_delta[right_offset + dim] += grad * right_scale;
                }
            }
        }

        for (slice_idx, slice) in embeddings.iter_mut().enumerate() {
            let delta = &deltas[slice_idx];
            for (point_idx, row) in slice.iter_mut().enumerate() {
                let offset = point_idx * n_components;
                for dim in 0..n_components {
                    row[dim] += step_size * delta[offset + dim];
                }
            }
        }

        // Apply one shared affine stabilization across all slices to avoid
        // per-slice anchoring that would counteract cross-slice alignment.
        if epoch % recenter_interval == 0 || epoch + 1 == n_epochs {
            enforce_embedding_system_stats(embeddings.as_mut_slice(), &target_stats);
        }
    }
}

fn warmstart_embeddings_with_relations(
    embeddings: &mut [Vec<Vec<f32>>],
    relations: &[PreparedRelation],
) {
    if embeddings.is_empty() {
        return;
    }

    let n_components = embeddings[0][0].len();
    for relation in relations {
        if relation.pairs.is_empty() {
            continue;
        }
        let left_idx = relation.left_slice;
        let right_idx = relation.right_slice;
        let left = &embeddings[left_idx];
        let right = &embeddings[right_idx];

        let mut offset = vec![0.0_f32; n_components];
        let mut weight_sum = 0.0_f32;
        for &(l_idx, r_idx) in relation.pairs.iter() {
            let w = 0.5 * (relation.left_inv_degree[l_idx] + relation.right_inv_degree[r_idx]);
            if w <= 0.0 {
                continue;
            }
            weight_sum += w;
            for dim in 0..n_components {
                offset[dim] += w * (left[l_idx][dim] - right[r_idx][dim]);
            }
        }
        if weight_sum <= 0.0 {
            continue;
        }
        for value in offset.iter_mut().take(n_components) {
            *value /= weight_sum;
        }

        for row in embeddings[right_idx].iter_mut() {
            for dim in 0..n_components {
                row[dim] += offset[dim];
            }
        }
    }
}

fn cosine_decay_alignment_lr(base: f32, epoch: usize, total_epochs: usize) -> f32 {
    if total_epochs <= 1 {
        return base;
    }
    let progress = epoch as f32 / (total_epochs - 1) as f32;
    let factor = 0.05 + 0.95 * 0.5 * (1.0 + (std::f32::consts::PI * progress).cos());
    base * factor
}

fn alignment_epoch_count(params: &AlignedUmapParams, embeddings: &[Vec<Vec<f32>>]) -> usize {
    if let Some(epochs) = params.alignment_epochs {
        return epochs;
    }

    if let Some(base_epochs) = params.umap.n_epochs {
        return (base_epochs / 2).max(1);
    }

    let total_points: usize = embeddings.iter().map(|slice| slice.len()).sum();
    if total_points <= 10_000 { 120 } else { 60 }
}

fn embedding_system_stats(embeddings: &[Vec<Vec<f32>>]) -> Option<EmbeddingSystemStats> {
    let first_slice = embeddings.first()?;
    let first_row = first_slice.first()?;
    let n_components = first_row.len();
    let total_points: usize = embeddings.iter().map(|slice| slice.len()).sum();
    if total_points == 0 {
        return None;
    }

    let mut centroid = vec![0.0_f32; n_components];
    for slice in embeddings {
        for row in slice {
            for dim in 0..n_components {
                centroid[dim] += row[dim];
            }
        }
    }
    for value in centroid.iter_mut() {
        *value /= total_points as f32;
    }

    let mut sum_sq = 0.0_f32;
    for slice in embeddings {
        for row in slice {
            for dim in 0..n_components {
                let delta = row[dim] - centroid[dim];
                sum_sq += delta * delta;
            }
        }
    }

    let rms_radius = (sum_sq / total_points as f32).sqrt();
    Some(EmbeddingSystemStats {
        centroid,
        rms_radius,
    })
}

fn enforce_embedding_system_stats(embeddings: &mut [Vec<Vec<f32>>], target: &EmbeddingSystemStats) {
    if embeddings.is_empty() {
        return;
    }

    let Some(current) = embedding_system_stats(embeddings) else {
        return;
    };
    let n_components = target.centroid.len();

    let scale = if current.rms_radius > 1e-8 && target.rms_radius > 0.0 {
        target.rms_radius / current.rms_radius
    } else {
        1.0
    };

    for slice in embeddings.iter_mut() {
        for row in slice.iter_mut() {
            for (dim, value) in row.iter_mut().enumerate().take(n_components) {
                let centered = *value - current.centroid[dim];
                *value = target.centroid[dim] + centered * scale;
            }
        }
    }
}

fn two_vectors_mut<T>(arr: &mut [Vec<T>], i: usize, j: usize) -> (&mut Vec<T>, &mut Vec<T>) {
    assert!(i != j, "indices must be different");
    if i < j {
        let (left, right) = arr.split_at_mut(j);
        (&mut left[i], &mut right[0])
    } else {
        let (left, right) = arr.split_at_mut(i);
        (&mut right[0], &mut left[j])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{InitMethod, Metric, UmapModel};

    fn make_temporal_slices(
        n_slices: usize,
        n_samples: usize,
        n_features: usize,
    ) -> Vec<Vec<Vec<f32>>> {
        let mut slices = Vec::with_capacity(n_slices);

        for slice_idx in 0..n_slices {
            let mut slice = Vec::with_capacity(n_samples);
            let drift = slice_idx as f32 * 0.22;
            for i in 0..n_samples {
                let t = 2.0 * std::f32::consts::PI * i as f32 / n_samples as f32;
                let base = [
                    (t + drift).cos(),
                    (t + drift).sin(),
                    (2.0 * t + 0.5 * drift).cos() * 0.7,
                    (3.0 * t + drift).sin() * 0.4,
                ];

                let mut row = vec![0.0_f32; n_features];
                for feat_idx in 0..n_features {
                    let b0 = base[feat_idx % base.len()];
                    let b1 = base[(feat_idx + 1) % base.len()];
                    let mix = (feat_idx as f32 + 1.0) * 0.11;
                    row[feat_idx] = b0 * (0.6 + 0.1 * mix) + b1 * (0.35 - 0.03 * mix) + drift;
                }
                slice.push(row);
            }
            slices.push(slice);
        }

        slices
    }

    fn mean_identity_gap(embeddings: &[Vec<Vec<f32>>]) -> f32 {
        let mut gap_sum = 0.0_f32;
        let mut pair_count = 0usize;

        for idx in 0..embeddings.len() - 1 {
            let left = &embeddings[idx];
            let right = &embeddings[idx + 1];
            let n = left.len().min(right.len());
            for point_idx in 0..n {
                let mut sq = 0.0_f32;
                for dim in 0..left[point_idx].len() {
                    let d = left[point_idx][dim] - right[point_idx][dim];
                    sq += d * d;
                }
                gap_sum += sq.sqrt();
                pair_count += 1;
            }
        }

        if pair_count == 0 {
            0.0
        } else {
            gap_sum / pair_count as f32
        }
    }

    fn assert_all_finite(embeddings: &[Vec<Vec<f32>>]) {
        for slice in embeddings {
            for row in slice {
                for &v in row {
                    assert!(v.is_finite(), "non-finite value encountered: {v}");
                }
            }
        }
    }

    fn centroid_and_radius(slice: &[Vec<f32>]) -> (Vec<f32>, f32) {
        let n_points = slice.len();
        let n_components = slice[0].len();
        let mut centroid = vec![0.0_f32; n_components];

        for row in slice {
            for dim in 0..n_components {
                centroid[dim] += row[dim];
            }
        }
        for value in centroid.iter_mut() {
            *value /= n_points as f32;
        }

        let mut sum_sq = 0.0_f32;
        for row in slice {
            for dim in 0..n_components {
                let d = row[dim] - centroid[dim];
                sum_sq += d * d;
            }
        }
        let rms_radius = (sum_sq / n_points as f32).sqrt();
        (centroid, rms_radius)
    }

    fn centroid_shift_l2(left: &[f32], right: &[f32]) -> f32 {
        let mut sum_sq = 0.0_f32;
        for dim in 0..left.len() {
            let d = left[dim] - right[dim];
            sum_sq += d * d;
        }
        sum_sq.sqrt()
    }

    #[test]
    fn aligned_identity_is_reproducible_with_fixed_seed() {
        let slices = make_temporal_slices(3, 72, 9);

        let aligned_params = AlignedUmapParams {
            umap: UmapParams {
                n_neighbors: 12,
                n_components: 2,
                n_epochs: Some(80),
                metric: Metric::Euclidean,
                learning_rate: 1.0,
                min_dist: 0.1,
                spread: 1.0,
                local_connectivity: 1.0,
                set_op_mix_ratio: 1.0,
                repulsion_strength: 1.0,
                negative_sample_rate: 5,
                random_seed: 2026,
                init: InitMethod::Random,
                use_approximate_knn: false,
                approx_knn_candidates: 30,
                approx_knn_iters: 10,
                approx_knn_threshold: 4096,
                ..UmapParams::default()
            },
            alignment_regularization: 0.1,
            alignment_learning_rate: 0.2,
            alignment_epochs: Some(70),
            recenter_interval: 3,
        };

        let mut model_a = AlignedUmapModel::new(aligned_params.clone());
        let mut model_b = AlignedUmapModel::new(aligned_params);

        let emb_a = model_a
            .fit_transform_identity(&slices)
            .expect("first aligned fit should succeed");
        let emb_b = model_b
            .fit_transform_identity(&slices)
            .expect("second aligned fit should succeed");

        assert_eq!(
            emb_a, emb_b,
            "fixed seed should produce deterministic aligned embedding"
        );
        assert_all_finite(&emb_a);
    }

    #[test]
    fn warmstart_reduces_initial_identity_gap() {
        let slices = make_temporal_slices(2, 64, 8);
        let base_umap = UmapParams {
            n_neighbors: 10,
            n_components: 2,
            n_epochs: Some(70),
            metric: Metric::Euclidean,
            learning_rate: 1.0,
            min_dist: 0.1,
            spread: 1.0,
            local_connectivity: 1.0,
            set_op_mix_ratio: 1.0,
            repulsion_strength: 1.0,
            negative_sample_rate: 5,
            random_seed: 42,
            init: InitMethod::Random,
            use_approximate_knn: false,
            approx_knn_candidates: 30,
            approx_knn_iters: 10,
            approx_knn_threshold: 4096,
            ..UmapParams::default()
        };

        let mut embeddings = Vec::with_capacity(slices.len());
        for (slice_idx, slice) in slices.iter().enumerate() {
            let mut params = base_umap.clone();
            params.random_seed = derive_slice_seed(params.random_seed, slice_idx as u64 + 1);
            let mut model = UmapModel::new(params);
            embeddings.push(
                model
                    .fit_transform(slice)
                    .expect("slice fit should succeed"),
            );
        }

        let relations = build_identity_relations(&slices).expect("relations should build");
        let prepared = prepare_relations(&slices, &relations).expect("relations should prepare");

        let gap_before = mean_identity_gap(&embeddings);
        warmstart_embeddings_with_relations(&mut embeddings, &prepared);
        let gap_after = mean_identity_gap(&embeddings);

        assert!(
            gap_after <= gap_before + 1e-6,
            "warm-start should not increase identity gap: before={gap_before}, after={gap_after}"
        );
    }

    #[test]
    fn cosine_decay_alignment_lr_is_monotonic_non_increasing() {
        let base = 0.8_f32;
        let epochs = 15_usize;
        let mut prev = f32::INFINITY;
        for epoch in 0..epochs {
            let lr = cosine_decay_alignment_lr(base, epoch, epochs);
            assert!(
                lr <= prev + 1e-8,
                "alignment lr must be non-increasing, epoch={epoch}, prev={prev}, current={lr}"
            );
            prev = lr;
        }
        let start = cosine_decay_alignment_lr(base, 0, epochs);
        let end = cosine_decay_alignment_lr(base, epochs - 1, epochs);
        assert!(
            (start - base).abs() < 1e-7,
            "start LR should be base: {start} vs {base}"
        );
        assert!(
            (end - base * 0.05).abs() < 1e-6,
            "end LR should decay to 5% of base: {end} vs {}",
            base * 0.05
        );
    }

    #[test]
    fn alignment_regularization_reduces_temporal_gap() {
        let slices = make_temporal_slices(3, 64, 8);

        let base_umap = UmapParams {
            n_neighbors: 10,
            n_components: 2,
            n_epochs: Some(70),
            metric: Metric::Euclidean,
            learning_rate: 1.0,
            min_dist: 0.1,
            spread: 1.0,
            local_connectivity: 1.0,
            set_op_mix_ratio: 1.0,
            repulsion_strength: 1.0,
            negative_sample_rate: 5,
            random_seed: 42,
            init: InitMethod::Random,
            use_approximate_knn: false,
            approx_knn_candidates: 30,
            approx_knn_iters: 10,
            approx_knn_threshold: 4096,
            ..UmapParams::default()
        };

        let mut unaligned = AlignedUmapModel::new(AlignedUmapParams {
            umap: base_umap.clone(),
            alignment_regularization: 0.0,
            alignment_learning_rate: 0.2,
            alignment_epochs: Some(60),
            recenter_interval: 3,
        });

        let mut aligned = AlignedUmapModel::new(AlignedUmapParams {
            umap: base_umap,
            alignment_regularization: 0.12,
            alignment_learning_rate: 0.2,
            alignment_epochs: Some(60),
            recenter_interval: 3,
        });

        let emb_unaligned = unaligned
            .fit_transform_identity(&slices)
            .expect("unaligned fit should succeed");
        let emb_aligned = aligned
            .fit_transform_identity(&slices)
            .expect("aligned fit should succeed");

        let gap_unaligned = mean_identity_gap(&emb_unaligned);
        let gap_aligned = mean_identity_gap(&emb_aligned);

        assert!(
            gap_aligned < gap_unaligned,
            "expected alignment regularization to reduce temporal identity gap: aligned={gap_aligned}, unaligned={gap_unaligned}"
        );
        assert_all_finite(&emb_aligned);
    }

    #[test]
    fn alignment_is_not_slice_stats_anchored_to_initial_state() {
        let slices = make_temporal_slices(3, 64, 8);

        let base_umap = UmapParams {
            n_neighbors: 10,
            n_components: 2,
            n_epochs: Some(70),
            metric: Metric::Euclidean,
            learning_rate: 1.0,
            min_dist: 0.1,
            spread: 1.0,
            local_connectivity: 1.0,
            set_op_mix_ratio: 1.0,
            repulsion_strength: 1.0,
            negative_sample_rate: 5,
            random_seed: 42,
            init: InitMethod::Random,
            use_approximate_knn: false,
            approx_knn_candidates: 30,
            approx_knn_iters: 10,
            approx_knn_threshold: 4096,
            ..UmapParams::default()
        };

        let mut initial_embeddings = Vec::with_capacity(slices.len());
        for (slice_idx, slice) in slices.iter().enumerate() {
            let mut params = base_umap.clone();
            params.random_seed = derive_slice_seed(params.random_seed, slice_idx as u64 + 1);
            let mut model = UmapModel::new(params);
            initial_embeddings.push(
                model
                    .fit_transform(slice)
                    .expect("baseline per-slice umap fit should succeed"),
            );
        }

        let mut aligned = AlignedUmapModel::new(AlignedUmapParams {
            umap: base_umap,
            alignment_regularization: 0.12,
            alignment_learning_rate: 0.2,
            alignment_epochs: Some(60),
            recenter_interval: 1,
        });

        let emb_aligned = aligned
            .fit_transform_identity(&slices)
            .expect("aligned fit should succeed");

        let mut preserved_slices = 0usize;
        for idx in 0..emb_aligned.len() {
            let (centroid_before, radius_before) = centroid_and_radius(&initial_embeddings[idx]);
            let (centroid_after, radius_after) = centroid_and_radius(&emb_aligned[idx]);
            let centroid_shift = centroid_shift_l2(&centroid_before, &centroid_after);
            let radius_delta = (radius_before - radius_after).abs();
            if centroid_shift <= 1e-4 && radius_delta <= 1e-4 {
                preserved_slices += 1;
            }
        }

        assert!(
            preserved_slices < emb_aligned.len(),
            "unexpected per-slice stat anchoring: every slice kept initial centroid/radius"
        );
        assert_all_finite(&emb_aligned);
    }
}
