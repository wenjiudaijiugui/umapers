use rust_umap::{
    InitMethod, Metric, ParametricTrainMode, ParametricUmapModel, ParametricUmapParams, UmapParams,
};
use std::env;
use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;
use std::time::Instant;

fn parse_mode(raw: &str) -> Result<ParametricTrainMode, Box<dyn Error>> {
    match raw.to_ascii_lowercase().as_str() {
        "naive" => Ok(ParametricTrainMode::Naive),
        "optimized" => Ok(ParametricTrainMode::Optimized),
        _ => Err(format!("unsupported mode '{raw}', expected naive|optimized").into()),
    }
}

fn read_csv(path: &Path) -> Result<Vec<Vec<f32>>, Box<dyn Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let mut data = Vec::new();
    for line in reader.lines() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        let row = trimmed
            .split(',')
            .map(|x| x.trim().parse::<f32>())
            .collect::<Result<Vec<f32>, _>>()?;
        data.push(row);
    }
    Ok(data)
}

fn write_csv(path: &Path, arr: &[Vec<f32>]) -> Result<(), Box<dyn Error>> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    for row in arr {
        for (j, val) in row.iter().enumerate() {
            if j > 0 {
                writer.write_all(b",")?;
            }
            write!(writer, "{val:.8}")?;
        }
        writer.write_all(b"\n")?;
    }

    Ok(())
}

fn rmse(a: &[Vec<f32>], b: &[Vec<f32>]) -> f32 {
    let mut sum_sq = 0.0_f64;
    let mut n = 0usize;
    for (row_a, row_b) in a.iter().zip(b.iter()) {
        for (x_a, x_b) in row_a.iter().zip(row_b.iter()) {
            let diff = *x_a as f64 - *x_b as f64;
            sum_sq += diff * diff;
            n += 1;
        }
    }
    if n == 0 {
        return 0.0;
    }
    (sum_sq / n as f64).sqrt() as f32
}

fn mae(a: &[Vec<f32>], b: &[Vec<f32>]) -> f32 {
    let mut sum_abs = 0.0_f64;
    let mut n = 0usize;
    for (row_a, row_b) in a.iter().zip(b.iter()) {
        for (x_a, x_b) in row_a.iter().zip(row_b.iter()) {
            sum_abs += (*x_a as f64 - *x_b as f64).abs();
            n += 1;
        }
    }
    if n == 0 {
        return 0.0;
    }
    (sum_abs / n as f64) as f32
}

fn usage() {
    eprintln!(
        "Usage:\n  parametric_eval <train.csv> <query.csv> <train_out.csv> <query_out.csv> \
          <seed> <hidden_dim> <train_epochs> <batch_size> <mode:naive|optimized> \
          [n_neighbors] [umap_epochs]"
    );
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 10 {
        usage();
        return Err("insufficient arguments".into());
    }

    let train_path = Path::new(&args[1]);
    let query_path = Path::new(&args[2]);
    let train_out_path = Path::new(&args[3]);
    let query_out_path = Path::new(&args[4]);

    let seed = args[5].parse::<u64>()?;
    let hidden_dim = args[6].parse::<usize>()?;
    let train_epochs = args[7].parse::<usize>()?;
    let batch_size = args[8].parse::<usize>()?;
    let mode = parse_mode(&args[9])?;

    let n_neighbors = if args.len() >= 11 {
        args[10].parse::<usize>()?
    } else {
        15
    };
    let umap_epochs = if args.len() >= 12 {
        args[11].parse::<usize>()?
    } else {
        200
    };

    let train = read_csv(train_path)?;
    let query = read_csv(query_path)?;

    let umap_params = UmapParams {
        n_neighbors,
        n_components: 2,
        n_epochs: Some(umap_epochs),
        metric: Metric::Euclidean,
        learning_rate: 1.0,
        min_dist: 0.1,
        spread: 1.0,
        local_connectivity: 1.0,
        set_op_mix_ratio: 1.0,
        repulsion_strength: 1.0,
        negative_sample_rate: 5,
        random_seed: seed,
        init: InitMethod::Spectral,
        use_approximate_knn: false,
        approx_knn_candidates: 30,
        approx_knn_iters: 10,
        approx_knn_threshold: 4096,
        ..UmapParams::default()
    };

    let params = ParametricUmapParams {
        umap_params,
        hidden_dim,
        train_epochs,
        batch_size,
        inference_batch_size: batch_size.max(256),
        learning_rate: 0.01,
        weight_decay: 1e-4,
        pairwise_loss_weight: 0.1,
        pairwise_pairs_per_batch: 32,
        standardize_input: false,
        seed,
        train_mode: mode,
    };

    let mut model = ParametricUmapModel::new(params);

    let t0 = Instant::now();
    let train_embedding = model.fit_transform(&train)?;
    let fit_time_sec = t0.elapsed().as_secs_f64();

    let teacher_embedding = model
        .teacher_embedding()
        .ok_or("teacher embedding is unavailable after fit")?;

    let t1 = Instant::now();
    let query_embedding = model.transform(&query)?;
    let transform_query_time_sec = t1.elapsed().as_secs_f64();

    write_csv(train_out_path, &train_embedding)?;
    write_csv(query_out_path, &query_embedding)?;

    let mode_name = match mode {
        ParametricTrainMode::Naive => "naive",
        ParametricTrainMode::Optimized => "optimized",
    };

    println!(
        concat!(
            "{{",
            "\"mode\":\"{}\",",
            "\"seed\":{},",
            "\"hidden_dim\":{},",
            "\"train_epochs\":{},",
            "\"batch_size\":{},",
            "\"n_neighbors\":{},",
            "\"umap_epochs\":{},",
            "\"fit_time_sec\":{:.9},",
            "\"transform_query_time_sec\":{:.9},",
            "\"train_alignment_rmse\":{:.9},",
            "\"train_alignment_mae\":{:.9},",
            "\"n_train\":{},",
            "\"n_query\":{}",
            "}}"
        ),
        mode_name,
        seed,
        hidden_dim,
        train_epochs,
        batch_size,
        n_neighbors,
        umap_epochs,
        fit_time_sec,
        transform_query_time_sec,
        rmse(&train_embedding, teacher_embedding),
        mae(&train_embedding, teacher_embedding),
        train.len(),
        query.len(),
    );

    Ok(())
}
