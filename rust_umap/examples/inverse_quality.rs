use rust_umap::{InitMethod, Metric, UmapModel, UmapParams};
use std::env;
use std::error::Error;
use std::fs;
use std::path::Path;
use std::time::Instant;

fn read_csv(path: &Path) -> Result<Vec<Vec<f32>>, Box<dyn Error>> {
    let text = fs::read_to_string(path)?;
    let mut rows = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let row = line
            .split(',')
            .map(|x| x.trim().parse::<f32>())
            .collect::<Result<Vec<_>, _>>()?;
        rows.push(row);
    }
    Ok(rows)
}

fn rmse(a: &[Vec<f32>], b: &[Vec<f32>]) -> f32 {
    let mut sum_sq = 0.0_f64;
    let mut n = 0usize;
    for (ra, rb) in a.iter().zip(b.iter()) {
        for (xa, xb) in ra.iter().zip(rb.iter()) {
            let d = *xa as f64 - *xb as f64;
            sum_sq += d * d;
            n += 1;
        }
    }
    (sum_sq / n as f64).sqrt() as f32
}

fn mae(a: &[Vec<f32>], b: &[Vec<f32>]) -> f32 {
    let mut sum_abs = 0.0_f64;
    let mut n = 0usize;
    for (ra, rb) in a.iter().zip(b.iter()) {
        for (xa, xb) in ra.iter().zip(rb.iter()) {
            sum_abs += (*xa as f64 - *xb as f64).abs();
            n += 1;
        }
    }
    (sum_abs / n as f64) as f32
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        return Err("usage: inverse_quality <train.csv> <query.csv>".into());
    }

    let train = read_csv(Path::new(&args[1]))?;
    let query = read_csv(Path::new(&args[2]))?;

    let params = UmapParams {
        n_neighbors: 15,
        n_components: 2,
        n_epochs: Some(200),
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

    let mut model = UmapModel::new(params);

    let t0 = Instant::now();
    let train_embedding = model.fit_transform(&train)?;
    let fit_time_sec = t0.elapsed().as_secs_f64();

    let t1 = Instant::now();
    let train_reconstruction = model.inverse_transform(&train_embedding)?;
    let inverse_train_time_sec = t1.elapsed().as_secs_f64();

    let t2 = Instant::now();
    let query_embedding = model.transform(&query)?;
    let transform_query_time_sec = t2.elapsed().as_secs_f64();

    let t3 = Instant::now();
    let query_reconstruction = model.inverse_transform(&query_embedding)?;
    let inverse_query_time_sec = t3.elapsed().as_secs_f64();

    println!(
        concat!(
            "{{",
            "\"fit_time_sec\":{:.9},",
            "\"inverse_train_time_sec\":{:.9},",
            "\"transform_query_time_sec\":{:.9},",
            "\"inverse_query_time_sec\":{:.9},",
            "\"train_rmse\":{:.9},",
            "\"train_mae\":{:.9},",
            "\"query_rmse\":{:.9},",
            "\"query_mae\":{:.9}",
            "}}"
        ),
        fit_time_sec,
        inverse_train_time_sec,
        transform_query_time_sec,
        inverse_query_time_sec,
        rmse(&train, &train_reconstruction),
        mae(&train, &train_reconstruction),
        rmse(&query, &query_reconstruction),
        mae(&query, &query_reconstruction),
    );
    Ok(())
}
