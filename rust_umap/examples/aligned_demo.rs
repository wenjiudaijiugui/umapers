use rust_umap::{AlignedUmapModel, AlignedUmapParams, InitMethod, Metric, UmapParams};

fn make_temporal_slices(
    n_slices: usize,
    n_samples: usize,
    n_features: usize,
) -> Vec<Vec<Vec<f32>>> {
    let mut slices = Vec::with_capacity(n_slices);

    for slice_idx in 0..n_slices {
        let mut slice = Vec::with_capacity(n_samples);
        let drift = slice_idx as f32 * 0.2;
        for i in 0..n_samples {
            let t = 2.0 * std::f32::consts::PI * i as f32 / n_samples as f32;
            let base = [
                (t + drift).cos(),
                (t + drift).sin(),
                (2.0 * t + 0.3 * drift).cos() * 0.6,
                (3.0 * t + drift).sin() * 0.5,
            ];

            let mut row = vec![0.0_f32; n_features];
            for feat_idx in 0..n_features {
                let b0 = base[feat_idx % base.len()];
                let b1 = base[(feat_idx + 1) % base.len()];
                row[feat_idx] = b0 * 0.7 + b1 * 0.3 + drift;
            }
            slice.push(row);
        }
        slices.push(slice);
    }

    slices
}

fn mean_identity_gap(embeddings: &[Vec<Vec<f32>>]) -> f32 {
    if embeddings.len() < 2 {
        return 0.0;
    }

    let mut sum = 0.0_f32;
    let mut count = 0usize;
    for idx in 0..embeddings.len() - 1 {
        let left = &embeddings[idx];
        let right = &embeddings[idx + 1];
        let n = left.len().min(right.len());
        for row_idx in 0..n {
            let mut sq = 0.0_f32;
            for d in 0..left[row_idx].len() {
                let diff = left[row_idx][d] - right[row_idx][d];
                sq += diff * diff;
            }
            sum += sq.sqrt();
            count += 1;
        }
    }

    if count == 0 { 0.0 } else { sum / count as f32 }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let slices = make_temporal_slices(3, 128, 10);

    let mut model = AlignedUmapModel::new(AlignedUmapParams {
        umap: UmapParams {
            n_neighbors: 15,
            n_components: 2,
            n_epochs: Some(120),
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
        },
        alignment_regularization: 0.1,
        alignment_learning_rate: 0.2,
        alignment_epochs: Some(80),
        recenter_interval: 4,
    });

    let embeddings = model.fit_transform_identity(&slices)?;

    println!("aligned embedding slices: {}", embeddings.len());
    println!(
        "slice[0] shape: {} x {}",
        embeddings[0].len(),
        embeddings[0][0].len()
    );
    println!(
        "mean adjacent identity gap: {:.6}",
        mean_identity_gap(&embeddings)
    );

    for (slice_idx, slice) in embeddings.iter().enumerate() {
        let row = &slice[0];
        println!(
            "slice[{slice_idx}] first point: [{:.6}, {:.6}]",
            row[0], row[1]
        );
    }

    Ok(())
}
