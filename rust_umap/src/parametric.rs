use crate::{UmapError, UmapModel, UmapParams};
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};

const SCALE_EPS: f32 = 1e-6;

#[inline]
fn forward_hidden_layer(input: &[f32], weights: &[f32], bias: &[f32], output: &mut [f32]) {
    let hidden = output.len();
    debug_assert_eq!(bias.len(), hidden);
    debug_assert_eq!(weights.len(), input.len() * hidden);

    output.copy_from_slice(bias);
    for (feature, &value) in input.iter().enumerate() {
        let weight_row = &weights[feature * hidden..(feature + 1) * hidden];
        for hidden_idx in 0..hidden {
            output[hidden_idx] += value * weight_row[hidden_idx];
        }
    }
    for value in output.iter_mut() {
        *value = value.tanh();
    }
}

#[inline]
fn forward_output_layer(input: &[f32], weights: &[f32], bias: &[f32], output: &mut [f32]) {
    let output_dim = output.len();
    debug_assert_eq!(bias.len(), output_dim);
    debug_assert_eq!(weights.len(), input.len() * output_dim);

    output.copy_from_slice(bias);
    for (hidden_idx, &value) in input.iter().enumerate() {
        let weight_row = &weights[hidden_idx * output_dim..(hidden_idx + 1) * output_dim];
        for output_idx in 0..output_dim {
            output[output_idx] += value * weight_row[output_idx];
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ParametricTrainMode {
    Naive,
    #[default]
    Optimized,
}

#[derive(Debug, Clone)]
pub struct ParametricUmapParams {
    pub umap_params: UmapParams,
    pub hidden_dim: usize,
    pub train_epochs: usize,
    pub batch_size: usize,
    pub inference_batch_size: usize,
    pub learning_rate: f32,
    pub weight_decay: f32,
    pub pairwise_loss_weight: f32,
    pub pairwise_pairs_per_batch: usize,
    pub standardize_input: bool,
    pub seed: u64,
    pub train_mode: ParametricTrainMode,
}

impl Default for ParametricUmapParams {
    fn default() -> Self {
        Self {
            umap_params: UmapParams {
                n_epochs: Some(200),
                use_approximate_knn: false,
                ..UmapParams::default()
            },
            hidden_dim: 64,
            train_epochs: 120,
            batch_size: 128,
            inference_batch_size: 1024,
            learning_rate: 0.01,
            weight_decay: 1e-4,
            // Keep pairwise distillation opt-in so legacy behavior remains unchanged by default.
            pairwise_loss_weight: 0.0,
            pairwise_pairs_per_batch: 32,
            standardize_input: true,
            seed: 42,
            train_mode: ParametricTrainMode::Optimized,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ParametricUmapModel {
    params: ParametricUmapParams,
    n_features: Option<usize>,
    input_mean: Vec<f32>,
    input_scale: Vec<f32>,
    w1: Vec<f32>,
    b1: Vec<f32>,
    w2: Vec<f32>,
    b2: Vec<f32>,
    teacher_embedding: Option<Vec<Vec<f32>>>,
}

impl ParametricUmapModel {
    pub fn new(params: ParametricUmapParams) -> Self {
        Self {
            params,
            n_features: None,
            input_mean: Vec::new(),
            input_scale: Vec::new(),
            w1: Vec::new(),
            b1: Vec::new(),
            w2: Vec::new(),
            b2: Vec::new(),
            teacher_embedding: None,
        }
    }

    pub fn params(&self) -> &ParametricUmapParams {
        &self.params
    }

    pub fn teacher_embedding(&self) -> Option<&[Vec<f32>]> {
        self.teacher_embedding.as_deref()
    }

    pub fn n_features(&self) -> Option<usize> {
        self.n_features
    }

    pub fn fit(&mut self, data: &[Vec<f32>]) -> Result<(), UmapError> {
        self.fit_transform(data).map(|_| ())
    }

    pub fn fit_transform(&mut self, data: &[Vec<f32>]) -> Result<Vec<Vec<f32>>, UmapError> {
        validate_parametric_params(&self.params)?;

        let (n_samples, n_features) = validate_dense_data(data)?;
        let output_dim = self.params.umap_params.n_components;
        if output_dim == 0 {
            return Err(UmapError::InvalidParameter(
                "parametric model requires umap_params.n_components >= 1".to_string(),
            ));
        }

        let mut teacher_params = self.params.umap_params.clone();
        teacher_params.random_seed = self.params.seed;
        if teacher_params.n_epochs.is_none() {
            teacher_params.n_epochs = Some(200);
        }

        let mut teacher_model = UmapModel::new(teacher_params);
        let teacher_embedding = teacher_model.fit_transform(data)?;

        let targets = flatten_rows(&teacher_embedding);
        let (inputs, mean, scale) = prepare_input_matrix(data, self.params.standardize_input);

        self.n_features = Some(n_features);
        self.input_mean = mean;
        self.input_scale = scale;
        self.init_network(n_features, output_dim);

        match self.params.train_mode {
            ParametricTrainMode::Naive => {
                self.train_naive_full_batch(&inputs, &targets, n_samples, n_features, output_dim)
            }
            ParametricTrainMode::Optimized => {
                self.train_optimized_minibatch(&inputs, &targets, n_samples, n_features, output_dim)
            }
        }

        self.teacher_embedding = Some(teacher_embedding);
        self.predict_internal(&inputs, n_samples, n_features)
    }

    pub fn transform(&self, query: &[Vec<f32>]) -> Result<Vec<Vec<f32>>, UmapError> {
        let expected_features = self.n_features.ok_or(UmapError::NotFitted)?;
        if query.is_empty() {
            return Ok(Vec::new());
        }
        validate_query_dimensions(query, expected_features)?;

        let inputs = apply_scaler(
            query,
            expected_features,
            &self.input_mean,
            &self.input_scale,
            self.params.standardize_input,
        );
        self.predict_internal(&inputs, query.len(), expected_features)
    }

    fn init_network(&mut self, n_features: usize, output_dim: usize) {
        let hidden = self.params.hidden_dim;
        let mut rng = SmallRng::seed_from_u64(self.params.seed ^ 0x57D1_CE4A_A2D1_9B63);

        self.w1 = vec![0.0; n_features * hidden];
        self.b1 = vec![0.0; hidden];
        self.w2 = vec![0.0; hidden * output_dim];
        self.b2 = vec![0.0; output_dim];

        let limit_1 = (6.0_f32 / (n_features + hidden).max(1) as f32).sqrt();
        let limit_2 = (6.0_f32 / (hidden + output_dim).max(1) as f32).sqrt();

        for weight in self.w1.iter_mut() {
            *weight = rng.gen_range(-limit_1..limit_1);
        }
        for weight in self.w2.iter_mut() {
            *weight = rng.gen_range(-limit_2..limit_2);
        }
    }

    #[allow(clippy::needless_range_loop)]
    fn train_naive_full_batch(
        &mut self,
        inputs: &[f32],
        targets: &[f32],
        n_samples: usize,
        n_features: usize,
        output_dim: usize,
    ) {
        let hidden = self.params.hidden_dim;
        let inv_n = 1.0 / n_samples as f32;
        let lr = self.params.learning_rate;
        let wd = self.params.weight_decay;

        let mut hidden_all = vec![0.0_f32; n_samples * hidden];
        let mut output_all = vec![0.0_f32; n_samples * output_dim];

        let mut grad_w1 = vec![0.0_f32; self.w1.len()];
        let mut grad_b1 = vec![0.0_f32; self.b1.len()];
        let mut grad_w2 = vec![0.0_f32; self.w2.len()];
        let mut grad_b2 = vec![0.0_f32; self.b2.len()];

        let mut delta_out_all = vec![0.0_f32; n_samples * output_dim];
        let mut delta_hidden = vec![0.0_f32; hidden];

        for epoch in 0..self.params.train_epochs {
            let epoch_lr = cosine_decay_lr(lr, epoch, self.params.train_epochs);

            hidden_all.fill(0.0);
            output_all.fill(0.0);
            grad_w1.fill(0.0);
            grad_b1.fill(0.0);
            grad_w2.fill(0.0);
            grad_b2.fill(0.0);
            delta_out_all.fill(0.0);

            for sample_idx in 0..n_samples {
                let x_row = &inputs[sample_idx * n_features..(sample_idx + 1) * n_features];
                let hidden_row = &mut hidden_all[sample_idx * hidden..(sample_idx + 1) * hidden];
                let output_row =
                    &mut output_all[sample_idx * output_dim..(sample_idx + 1) * output_dim];

                forward_hidden_layer(x_row, &self.w1, &self.b1, hidden_row);
                forward_output_layer(hidden_row, &self.w2, &self.b2, output_row);
                for o in 0..output_dim {
                    let sum = output_row[o];
                    let target = targets[sample_idx * output_dim + o];
                    delta_out_all[sample_idx * output_dim + o] = sum - target;
                }
            }

            if self.params.pairwise_loss_weight > 0.0
                && self.params.pairwise_pairs_per_batch > 0
                && n_samples > 1
            {
                let max_pairs = n_samples * (n_samples - 1) / 2;
                let pair_count = self.params.pairwise_pairs_per_batch.min(max_pairs).max(1);
                let pair_scale = self.params.pairwise_loss_weight / pair_count as f32;

                for pair_idx in 0..pair_count {
                    let i = (pair_idx * 9973 + epoch * 37) % n_samples;
                    let mut j = (pair_idx * 3251 + epoch * 73 + 1) % n_samples;
                    if j == i {
                        j = (j + 1) % n_samples;
                    }

                    let i_off = i * output_dim;
                    let j_off = j * output_dim;
                    for o in 0..output_dim {
                        let student_diff = output_all[i_off + o] - output_all[j_off + o];
                        let teacher_diff = targets[i_off + o] - targets[j_off + o];
                        let pair_err = student_diff - teacher_diff;
                        delta_out_all[i_off + o] += pair_scale * pair_err;
                        delta_out_all[j_off + o] -= pair_scale * pair_err;
                    }
                }
            }

            for sample_idx in 0..n_samples {
                let x_row = &inputs[sample_idx * n_features..(sample_idx + 1) * n_features];
                let hidden_row = &hidden_all[sample_idx * hidden..(sample_idx + 1) * hidden];
                let delta_out_row =
                    &delta_out_all[sample_idx * output_dim..(sample_idx + 1) * output_dim];

                for o in 0..output_dim {
                    grad_b2[o] += delta_out_row[o];
                }
                for h in 0..hidden {
                    let weight_offset = h * output_dim;
                    let hidden_value = hidden_row[h];
                    for o in 0..output_dim {
                        grad_w2[weight_offset + o] += hidden_value * delta_out_row[o];
                    }
                }

                for h in 0..hidden {
                    let mut sum = 0.0_f32;
                    for o in 0..output_dim {
                        sum += delta_out_row[o] * self.w2[h * output_dim + o];
                    }
                    let dh = sum * (1.0 - hidden_row[h] * hidden_row[h]);
                    delta_hidden[h] = dh;
                    grad_b1[h] += dh;
                }

                for f in 0..n_features {
                    let x = x_row[f];
                    let base = f * hidden;
                    for h in 0..hidden {
                        grad_w1[base + h] += x * delta_hidden[h];
                    }
                }
            }

            for idx in 0..self.w1.len() {
                let grad = grad_w1[idx] * inv_n + wd * self.w1[idx];
                self.w1[idx] -= epoch_lr * grad;
            }
            for idx in 0..self.b1.len() {
                let grad = grad_b1[idx] * inv_n;
                self.b1[idx] -= epoch_lr * grad;
            }
            for idx in 0..self.w2.len() {
                let grad = grad_w2[idx] * inv_n + wd * self.w2[idx];
                self.w2[idx] -= epoch_lr * grad;
            }
            for idx in 0..self.b2.len() {
                let grad = grad_b2[idx] * inv_n;
                self.b2[idx] -= epoch_lr * grad;
            }
        }
    }

    #[allow(clippy::needless_range_loop)]
    fn train_optimized_minibatch(
        &mut self,
        inputs: &[f32],
        targets: &[f32],
        n_samples: usize,
        n_features: usize,
        output_dim: usize,
    ) {
        let hidden = self.params.hidden_dim;
        let batch_size = self.params.batch_size.max(1).min(n_samples);
        let wd = self.params.weight_decay;

        let mut rng = SmallRng::seed_from_u64(self.params.seed ^ 0x6A09_E667_F3BC_C909);
        let mut order: Vec<usize> = (0..n_samples).collect();

        let mut hidden_buf = vec![0.0_f32; batch_size * hidden];
        let mut output_buf = vec![0.0_f32; batch_size * output_dim];
        let mut delta_out = vec![0.0_f32; batch_size * output_dim];
        let mut delta_hidden = vec![0.0_f32; batch_size * hidden];

        let mut grad_w1 = vec![0.0_f32; self.w1.len()];
        let mut grad_b1 = vec![0.0_f32; self.b1.len()];
        let mut grad_w2 = vec![0.0_f32; self.w2.len()];
        let mut grad_b2 = vec![0.0_f32; self.b2.len()];

        for epoch in 0..self.params.train_epochs {
            let epoch_lr =
                cosine_decay_lr(self.params.learning_rate, epoch, self.params.train_epochs);
            order.shuffle(&mut rng);

            for chunk in order.chunks(batch_size) {
                let bsz = chunk.len();
                grad_w1.fill(0.0);
                grad_b1.fill(0.0);
                grad_w2.fill(0.0);
                grad_b2.fill(0.0);

                for (bi, &sample_idx) in chunk.iter().enumerate() {
                    let x_row = &inputs[sample_idx * n_features..(sample_idx + 1) * n_features];
                    let y_row = &targets[sample_idx * output_dim..(sample_idx + 1) * output_dim];

                    let hidden_row = &mut hidden_buf[bi * hidden..(bi + 1) * hidden];
                    let output_row = &mut output_buf[bi * output_dim..(bi + 1) * output_dim];
                    let delta_out_row = &mut delta_out[bi * output_dim..(bi + 1) * output_dim];

                    forward_hidden_layer(x_row, &self.w1, &self.b1, hidden_row);
                    forward_output_layer(hidden_row, &self.w2, &self.b2, output_row);
                    for o in 0..output_dim {
                        delta_out_row[o] = output_row[o] - y_row[o];
                    }
                }

                if self.params.pairwise_loss_weight > 0.0
                    && self.params.pairwise_pairs_per_batch > 0
                    && bsz > 1
                {
                    let max_pairs = bsz * (bsz - 1) / 2;
                    let pair_count = self.params.pairwise_pairs_per_batch.min(max_pairs).max(1);
                    let pair_scale = self.params.pairwise_loss_weight / pair_count as f32;

                    for _ in 0..pair_count {
                        let i = rng.gen_range(0..bsz);
                        let mut j = rng.gen_range(0..(bsz - 1));
                        if j >= i {
                            j += 1;
                        }

                        let i_off = i * output_dim;
                        let j_off = j * output_dim;
                        let i_target_off = chunk[i] * output_dim;
                        let j_target_off = chunk[j] * output_dim;
                        for o in 0..output_dim {
                            let student_diff = output_buf[i_off + o] - output_buf[j_off + o];
                            let teacher_diff =
                                targets[i_target_off + o] - targets[j_target_off + o];
                            let pair_err = student_diff - teacher_diff;
                            delta_out[i_off + o] += pair_scale * pair_err;
                            delta_out[j_off + o] -= pair_scale * pair_err;
                        }
                    }
                }

                for bi in 0..bsz {
                    let hidden_row = &hidden_buf[bi * hidden..(bi + 1) * hidden];
                    let delta_out_row = &delta_out[bi * output_dim..(bi + 1) * output_dim];
                    for o in 0..output_dim {
                        grad_b2[o] += delta_out_row[o];
                    }
                    for h in 0..hidden {
                        let weight_offset = h * output_dim;
                        let hidden_value = hidden_row[h];
                        for o in 0..output_dim {
                            grad_w2[weight_offset + o] += hidden_value * delta_out_row[o];
                        }
                    }
                }

                for (bi, &sample_idx) in chunk.iter().enumerate() {
                    let x_row = &inputs[sample_idx * n_features..(sample_idx + 1) * n_features];
                    let hidden_row = &hidden_buf[bi * hidden..(bi + 1) * hidden];
                    let delta_out_row = &delta_out[bi * output_dim..(bi + 1) * output_dim];
                    let delta_hidden_row = &mut delta_hidden[bi * hidden..(bi + 1) * hidden];

                    for h in 0..hidden {
                        let mut sum = 0.0_f32;
                        for o in 0..output_dim {
                            sum += delta_out_row[o] * self.w2[h * output_dim + o];
                        }
                        let dh = sum * (1.0 - hidden_row[h] * hidden_row[h]);
                        delta_hidden_row[h] = dh;
                        grad_b1[h] += dh;
                    }

                    for f in 0..n_features {
                        let x = x_row[f];
                        let base = f * hidden;
                        for h in 0..hidden {
                            grad_w1[base + h] += x * delta_hidden_row[h];
                        }
                    }
                }

                let scale = 1.0 / bsz as f32;
                for idx in 0..self.w1.len() {
                    let grad = grad_w1[idx] * scale + wd * self.w1[idx];
                    self.w1[idx] -= epoch_lr * grad;
                }
                for idx in 0..self.b1.len() {
                    self.b1[idx] -= epoch_lr * (grad_b1[idx] * scale);
                }
                for idx in 0..self.w2.len() {
                    let grad = grad_w2[idx] * scale + wd * self.w2[idx];
                    self.w2[idx] -= epoch_lr * grad;
                }
                for idx in 0..self.b2.len() {
                    self.b2[idx] -= epoch_lr * (grad_b2[idx] * scale);
                }
            }
        }
    }

    #[allow(clippy::needless_range_loop)]
    fn predict_internal(
        &self,
        inputs: &[f32],
        n_samples: usize,
        n_features: usize,
    ) -> Result<Vec<Vec<f32>>, UmapError> {
        if self.w1.is_empty() || self.w2.is_empty() {
            return Err(UmapError::NotFitted);
        }

        let hidden = self.params.hidden_dim;
        let output_dim = self.params.umap_params.n_components;
        let batch_size = self
            .params
            .inference_batch_size
            .max(1)
            .min(n_samples.max(1));

        let mut output = vec![vec![0.0_f32; output_dim]; n_samples];
        let mut hidden_buf = vec![0.0_f32; hidden];

        let mut start = 0usize;
        while start < n_samples {
            let end = (start + batch_size).min(n_samples);
            let bsz = end - start;

            for bi in 0..bsz {
                let sample_idx = start + bi;
                let x_row = &inputs[sample_idx * n_features..(sample_idx + 1) * n_features];
                forward_hidden_layer(x_row, &self.w1, &self.b1, &mut hidden_buf);
                forward_output_layer(&hidden_buf, &self.w2, &self.b2, &mut output[sample_idx]);
            }

            start = end;
        }

        let all_finite = output
            .iter()
            .flat_map(|row| row.iter())
            .all(|value| value.is_finite());
        if !all_finite {
            return Err(UmapError::InvalidParameter(
                "parametric forward produced non-finite outputs".to_string(),
            ));
        }

        Ok(output)
    }
}

fn cosine_decay_lr(base: f32, epoch: usize, total_epochs: usize) -> f32 {
    if total_epochs <= 1 {
        return base;
    }
    let progress = epoch as f32 / (total_epochs - 1) as f32;
    let factor = 0.1 + 0.9 * 0.5 * (1.0 + (std::f32::consts::PI * (1.0 - progress)).cos());
    base * factor
}

fn validate_parametric_params(params: &ParametricUmapParams) -> Result<(), UmapError> {
    if params.hidden_dim == 0 {
        return Err(UmapError::InvalidParameter(
            "hidden_dim must be >= 1".to_string(),
        ));
    }
    if params.train_epochs == 0 {
        return Err(UmapError::InvalidParameter(
            "train_epochs must be >= 1".to_string(),
        ));
    }
    if params.batch_size == 0 {
        return Err(UmapError::InvalidParameter(
            "batch_size must be >= 1".to_string(),
        ));
    }
    if params.inference_batch_size == 0 {
        return Err(UmapError::InvalidParameter(
            "inference_batch_size must be >= 1".to_string(),
        ));
    }
    if !params.learning_rate.is_finite() || params.learning_rate <= 0.0 {
        return Err(UmapError::InvalidParameter(
            "learning_rate must be finite and > 0".to_string(),
        ));
    }
    if !params.pairwise_loss_weight.is_finite() || params.pairwise_loss_weight < 0.0 {
        return Err(UmapError::InvalidParameter(
            "pairwise_loss_weight must be finite and >= 0".to_string(),
        ));
    }
    if !params.weight_decay.is_finite() || params.weight_decay < 0.0 {
        return Err(UmapError::InvalidParameter(
            "weight_decay must be finite and >= 0".to_string(),
        ));
    }
    Ok(())
}

fn validate_dense_data(data: &[Vec<f32>]) -> Result<(usize, usize), UmapError> {
    if data.is_empty() {
        return Err(UmapError::EmptyData);
    }
    if data.len() < 2 {
        return Err(UmapError::NeedAtLeastTwoSamples);
    }

    let n_features = data[0].len();
    if n_features == 0 {
        return Err(UmapError::InvalidParameter(
            "input rows must have at least one feature".to_string(),
        ));
    }

    for (row_idx, row) in data.iter().enumerate().skip(1) {
        if row.len() != n_features {
            return Err(UmapError::InconsistentDimensions {
                row: row_idx,
                expected: n_features,
                got: row.len(),
            });
        }
    }

    Ok((data.len(), n_features))
}

fn validate_query_dimensions(
    query: &[Vec<f32>],
    expected_features: usize,
) -> Result<(), UmapError> {
    for (row_idx, row) in query.iter().enumerate() {
        if row.len() != expected_features {
            if row_idx == 0 {
                return Err(UmapError::FeatureMismatch {
                    expected: expected_features,
                    got: row.len(),
                });
            }
            return Err(UmapError::InconsistentDimensions {
                row: row_idx,
                expected: expected_features,
                got: row.len(),
            });
        }
    }
    Ok(())
}

fn prepare_input_matrix(
    data: &[Vec<f32>],
    standardize_input: bool,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let n_samples = data.len();
    let n_features = data[0].len();

    if !standardize_input {
        let mut flat = vec![0.0_f32; n_samples * n_features];
        for (row_idx, row) in data.iter().enumerate() {
            for (feature_idx, &value) in row.iter().enumerate() {
                flat[row_idx * n_features + feature_idx] = value;
            }
        }
        return (flat, vec![0.0; n_features], vec![1.0; n_features]);
    }

    let mut mean = vec![0.0_f32; n_features];
    for row in data {
        for (feature_idx, &value) in row.iter().enumerate() {
            mean[feature_idx] += value;
        }
    }
    let inv_n = 1.0 / n_samples as f32;
    for value in mean.iter_mut() {
        *value *= inv_n;
    }

    let mut var = vec![0.0_f32; n_features];
    for row in data {
        for feature_idx in 0..n_features {
            let diff = row[feature_idx] - mean[feature_idx];
            var[feature_idx] += diff * diff;
        }
    }

    let mut scale = vec![1.0_f32; n_features];
    for feature_idx in 0..n_features {
        let std = (var[feature_idx] * inv_n).sqrt();
        scale[feature_idx] = if std > SCALE_EPS { std } else { 1.0 };
    }

    let mut flat = vec![0.0_f32; n_samples * n_features];
    for (row_idx, row) in data.iter().enumerate() {
        for feature_idx in 0..n_features {
            flat[row_idx * n_features + feature_idx] =
                (row[feature_idx] - mean[feature_idx]) / scale[feature_idx];
        }
    }

    (flat, mean, scale)
}

fn apply_scaler(
    data: &[Vec<f32>],
    n_features: usize,
    mean: &[f32],
    scale: &[f32],
    standardize_input: bool,
) -> Vec<f32> {
    let mut flat = vec![0.0_f32; data.len() * n_features];
    for (row_idx, row) in data.iter().enumerate() {
        for feature_idx in 0..n_features {
            let raw = row[feature_idx];
            let value = if standardize_input {
                (raw - mean[feature_idx]) / scale[feature_idx]
            } else {
                raw
            };
            flat[row_idx * n_features + feature_idx] = value;
        }
    }
    flat
}

fn flatten_rows(rows: &[Vec<f32>]) -> Vec<f32> {
    if rows.is_empty() {
        return Vec::new();
    }
    let n_cols = rows[0].len();
    let mut flat = vec![0.0_f32; rows.len() * n_cols];
    for (row_idx, row) in rows.iter().enumerate() {
        for (col_idx, &value) in row.iter().enumerate() {
            flat[row_idx * n_cols + col_idx] = value;
        }
    }
    flat
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::InitMethod;

    fn synthetic_data(n: usize, dim: usize) -> Vec<Vec<f32>> {
        let mut data = Vec::with_capacity(n);
        for i in 0..n {
            let cluster_shift = if i < n / 2 { 0.0 } else { 3.5 };
            let row = (0..dim)
                .map(|d| {
                    let t = (i as f32 + 1.7 * d as f32) / n as f32;
                    cluster_shift + (9.0 * t).sin() * 0.2 + (5.0 * t).cos() * 0.1
                })
                .collect::<Vec<f32>>();
            data.push(row);
        }
        data
    }

    fn assert_finite(points: &[Vec<f32>]) {
        assert!(
            points
                .iter()
                .flat_map(|row| row.iter())
                .all(|value| value.is_finite())
        );
    }

    #[test]
    fn contiguous_forward_loops_match_reference_accumulation_exactly() {
        let input = [0.25_f32, -0.75, 1.5];
        let hidden_bias = [0.1_f32, -0.2, 0.3, -0.4];
        let hidden_weights = [
            0.5_f32, -0.1, 0.2, 0.7, -0.4, 0.8, -0.3, 0.6, 0.9, 0.2, -0.5, -0.7,
        ];
        let output_bias = [0.05_f32, -0.15];
        let output_weights = [0.2_f32, -0.3, 0.4, 0.6, -0.8, 0.1, 0.7, -0.5];

        let mut expected_hidden = vec![0.0_f32; hidden_bias.len()];
        for hidden_idx in 0..expected_hidden.len() {
            let mut sum = hidden_bias[hidden_idx];
            for feature in 0..input.len() {
                sum +=
                    input[feature] * hidden_weights[feature * expected_hidden.len() + hidden_idx];
            }
            expected_hidden[hidden_idx] = sum.tanh();
        }

        let mut actual_hidden = vec![0.0_f32; hidden_bias.len()];
        forward_hidden_layer(&input, &hidden_weights, &hidden_bias, &mut actual_hidden);
        assert_eq!(actual_hidden, expected_hidden);

        let mut expected_output = vec![0.0_f32; output_bias.len()];
        for output_idx in 0..expected_output.len() {
            let mut sum = output_bias[output_idx];
            for hidden_idx in 0..expected_hidden.len() {
                sum += expected_hidden[hidden_idx]
                    * output_weights[hidden_idx * expected_output.len() + output_idx];
            }
            expected_output[output_idx] = sum;
        }

        let mut actual_output = vec![0.0_f32; output_bias.len()];
        forward_output_layer(
            &actual_hidden,
            &output_weights,
            &output_bias,
            &mut actual_output,
        );
        assert_eq!(actual_output, expected_output);
    }

    #[test]
    fn parametric_fit_transform_is_deterministic_with_same_seed() {
        let data = synthetic_data(64, 6);

        let umap_params = UmapParams {
            n_neighbors: 12,
            n_components: 2,
            n_epochs: Some(80),
            init: InitMethod::Random,
            use_approximate_knn: false,
            ..UmapParams::default()
        };

        let params = ParametricUmapParams {
            umap_params,
            hidden_dim: 48,
            train_epochs: 60,
            batch_size: 16,
            inference_batch_size: 64,
            learning_rate: 0.01,
            weight_decay: 1e-4,
            pairwise_loss_weight: 0.1,
            pairwise_pairs_per_batch: 16,
            standardize_input: true,
            seed: 777,
            train_mode: ParametricTrainMode::Optimized,
        };

        let mut model_a = ParametricUmapModel::new(params.clone());
        let mut model_b = ParametricUmapModel::new(params);

        let emb_a = model_a
            .fit_transform(&data)
            .expect("parametric fit should succeed");
        let emb_b = model_b
            .fit_transform(&data)
            .expect("parametric fit should succeed");

        assert_eq!(emb_a, emb_b);
        assert_finite(&emb_a);
    }

    #[test]
    fn parametric_transform_supports_new_samples() {
        let data = synthetic_data(80, 8);
        let query = data.iter().skip(70).cloned().collect::<Vec<_>>();

        let mut params = ParametricUmapParams::default();
        params.umap_params.n_neighbors = 10;
        params.umap_params.n_components = 2;
        params.umap_params.n_epochs = Some(70);
        params.umap_params.init = InitMethod::Random;
        params.umap_params.use_approximate_knn = false;
        params.hidden_dim = 32;
        params.train_epochs = 50;
        params.batch_size = 20;
        params.pairwise_loss_weight = 0.1;
        params.pairwise_pairs_per_batch = 16;
        params.seed = 2026;
        params.train_mode = ParametricTrainMode::Optimized;

        let mut model = ParametricUmapModel::new(params);
        let train_emb = model.fit_transform(&data).expect("fit should succeed");
        let query_emb = model.transform(&query).expect("transform should succeed");

        assert_eq!(train_emb.len(), data.len());
        assert_eq!(query_emb.len(), query.len());
        assert_eq!(query_emb[0].len(), 2);
        assert!(model.teacher_embedding().is_some());
        assert_finite(&query_emb);
    }

    #[test]
    fn parametric_rejects_invalid_pairwise_loss_weight() {
        let data = synthetic_data(24, 4);
        let params = ParametricUmapParams {
            pairwise_loss_weight: f32::NAN,
            ..ParametricUmapParams::default()
        };

        let mut model = ParametricUmapModel::new(params);
        let err = model
            .fit_transform(&data)
            .expect_err("non-finite pairwise_loss_weight should fail");
        assert!(matches!(err, UmapError::InvalidParameter(_)));
        assert!(err.to_string().contains("pairwise_loss_weight"));
    }

    #[test]
    fn parametric_rejects_negative_pairwise_loss_weight() {
        let data = synthetic_data(24, 4);
        let params = ParametricUmapParams {
            pairwise_loss_weight: -0.01,
            ..ParametricUmapParams::default()
        };

        let mut model = ParametricUmapModel::new(params);
        let err = model
            .fit_transform(&data)
            .expect_err("negative pairwise_loss_weight should fail");
        assert!(matches!(err, UmapError::InvalidParameter(_)));
        assert!(err.to_string().contains("pairwise_loss_weight"));
    }

    #[test]
    fn parametric_rejects_non_finite_learning_rate() {
        let data = synthetic_data(24, 4);
        let params = ParametricUmapParams {
            learning_rate: f32::INFINITY,
            ..ParametricUmapParams::default()
        };

        let mut model = ParametricUmapModel::new(params);
        let err = model
            .fit_transform(&data)
            .expect_err("non-finite learning_rate should fail");
        assert!(matches!(err, UmapError::InvalidParameter(_)));
        assert!(err.to_string().contains("learning_rate"));
    }

    #[test]
    fn parametric_rejects_non_finite_weight_decay() {
        let data = synthetic_data(24, 4);
        let params = ParametricUmapParams {
            weight_decay: f32::NAN,
            ..ParametricUmapParams::default()
        };

        let mut model = ParametricUmapModel::new(params);
        let err = model
            .fit_transform(&data)
            .expect_err("non-finite weight_decay should fail");
        assert!(matches!(err, UmapError::InvalidParameter(_)));
        assert!(err.to_string().contains("weight_decay"));
    }

    #[test]
    fn parametric_defaults_keep_pairwise_distillation_disabled() {
        let params = ParametricUmapParams::default();
        assert_eq!(params.pairwise_loss_weight, 0.0);
        assert_eq!(params.pairwise_pairs_per_batch, 32);
    }
}
