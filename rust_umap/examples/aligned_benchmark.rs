use rust_umap::{AlignedUmapModel, AlignedUmapParams, InitMethod, Metric, UmapParams};
use std::env;
use std::error::Error;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

struct Config {
    output_dir: PathBuf,
    slice_paths: Vec<PathBuf>,
    n_neighbors: usize,
    n_components: usize,
    n_epochs: usize,
    seed: u64,
    init: InitMethod,
    metric: Metric,
    use_approximate_knn: bool,
    approx_knn_candidates: usize,
    approx_knn_iters: usize,
    approx_knn_threshold: usize,
    alignment_regularization: f32,
    alignment_learning_rate: f32,
    alignment_epochs: usize,
    recenter_interval: usize,
}

fn parse_bool(s: &str) -> Result<bool, Box<dyn Error>> {
    match s.to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "y" => Ok(true),
        "0" | "false" | "no" | "n" => Ok(false),
        _ => Err(format!("cannot parse bool from '{s}'").into()),
    }
}

fn parse_metric(s: &str) -> Result<Metric, Box<dyn Error>> {
    match s.to_ascii_lowercase().as_str() {
        "euclidean" | "l2" => Ok(Metric::Euclidean),
        "manhattan" | "l1" => Ok(Metric::Manhattan),
        "cosine" => Ok(Metric::Cosine),
        "chebyshev" | "linfinity" | "linf" => Ok(Metric::Chebyshev),
        "correlation" => Ok(Metric::Correlation),
        "canberra" => Ok(Metric::Canberra),
        "braycurtis" | "bray_curtis" => Ok(Metric::BrayCurtis),
        _ => Err(format!(
            "unsupported metric '{s}', expected euclidean|manhattan|cosine|chebyshev|correlation|canberra|braycurtis"
        )
        .into()),
    }
}

fn parse_init(s: &str) -> Result<InitMethod, Box<dyn Error>> {
    match s.to_ascii_lowercase().as_str() {
        "random" => Ok(InitMethod::Random),
        "spectral" => Ok(InitMethod::Spectral),
        _ => Err(format!("unsupported init '{s}', expected random|spectral").into()),
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

fn init_name(init: InitMethod) -> &'static str {
    match init {
        InitMethod::Random => "random",
        InitMethod::Spectral => "spectral",
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

fn write_csv(path: &Path, rows: &[Vec<f32>]) -> Result<(), Box<dyn Error>> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    for row in rows {
        for (idx, val) in row.iter().enumerate() {
            if idx > 0 {
                writer.write_all(b",")?;
            }
            write!(writer, "{val:.8}")?;
        }
        writer.write_all(b"\n")?;
    }
    Ok(())
}

fn mean_adjacent_identity_gap(embeddings: &[Vec<Vec<f32>>]) -> f32 {
    if embeddings.len() < 2 {
        return 0.0;
    }

    let mut sum = 0.0_f32;
    let mut count = 0usize;

    for slice_idx in 0..embeddings.len() - 1 {
        let left = &embeddings[slice_idx];
        let right = &embeddings[slice_idx + 1];
        let n = left.len().min(right.len());
        for row_idx in 0..n {
            let mut sq = 0.0_f32;
            for dim in 0..left[row_idx].len() {
                let d = left[row_idx][dim] - right[row_idx][dim];
                sq += d * d;
            }
            sum += sq.sqrt();
            count += 1;
        }
    }

    if count == 0 { 0.0 } else { sum / count as f32 }
}

fn usage() {
    eprintln!(
        "Usage:\n  aligned_benchmark <output_dir> <slice_0.csv> <slice_1.csv> [slice_2.csv ...] \\\n          [--n-neighbors <usize>] [--n-components <usize>] [--n-epochs <usize>] [--seed <u64>] \\\n          [--init random|spectral] [--metric euclidean|manhattan|cosine] \\\n          [--use-approximate-knn true|false] [--approx-knn-candidates <usize>] \\\n          [--approx-knn-iters <usize>] [--approx-knn-threshold <usize>] \\\n          [--alignment-regularization <f32>] [--alignment-learning-rate <f32>] \\\n          [--alignment-epochs <usize>] [--recenter-interval <usize>]"
    );
}

fn parse_args() -> Result<Config, Box<dyn Error>> {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.len() < 3 {
        usage();
        return Err("insufficient arguments".into());
    }

    let output_dir = PathBuf::from(&args[0]);

    let mut idx = 1;
    let mut slice_paths = Vec::new();
    while idx < args.len() && !args[idx].starts_with("--") {
        slice_paths.push(PathBuf::from(&args[idx]));
        idx += 1;
    }

    if slice_paths.len() < 2 {
        usage();
        return Err("need at least two slice csv files".into());
    }

    let mut cfg = Config {
        output_dir,
        slice_paths,
        n_neighbors: 15,
        n_components: 2,
        n_epochs: 200,
        seed: 42,
        init: InitMethod::Random,
        metric: Metric::Euclidean,
        use_approximate_knn: false,
        approx_knn_candidates: 30,
        approx_knn_iters: 10,
        approx_knn_threshold: 4096,
        alignment_regularization: 0.08,
        alignment_learning_rate: 0.25,
        alignment_epochs: 100,
        recenter_interval: 5,
    };

    while idx < args.len() {
        let key = &args[idx];
        idx += 1;
        if idx >= args.len() {
            return Err(format!("missing value for option '{key}'").into());
        }
        let value = &args[idx];
        idx += 1;

        match key.as_str() {
            "--n-neighbors" => cfg.n_neighbors = value.parse::<usize>()?,
            "--n-components" => cfg.n_components = value.parse::<usize>()?,
            "--n-epochs" => cfg.n_epochs = value.parse::<usize>()?,
            "--seed" => cfg.seed = value.parse::<u64>()?,
            "--init" => cfg.init = parse_init(value)?,
            "--metric" => cfg.metric = parse_metric(value)?,
            "--use-approximate-knn" => cfg.use_approximate_knn = parse_bool(value)?,
            "--approx-knn-candidates" => cfg.approx_knn_candidates = value.parse::<usize>()?,
            "--approx-knn-iters" => cfg.approx_knn_iters = value.parse::<usize>()?,
            "--approx-knn-threshold" => cfg.approx_knn_threshold = value.parse::<usize>()?,
            "--alignment-regularization" => cfg.alignment_regularization = value.parse::<f32>()?,
            "--alignment-learning-rate" => cfg.alignment_learning_rate = value.parse::<f32>()?,
            "--alignment-epochs" => cfg.alignment_epochs = value.parse::<usize>()?,
            "--recenter-interval" => cfg.recenter_interval = value.parse::<usize>()?,
            _ => return Err(format!("unsupported option '{key}'").into()),
        }
    }

    Ok(cfg)
}

fn main() -> Result<(), Box<dyn Error>> {
    let cfg = parse_args()?;
    fs::create_dir_all(&cfg.output_dir)?;

    let mut slices = Vec::with_capacity(cfg.slice_paths.len());
    for path in cfg.slice_paths.iter() {
        slices.push(read_csv(path)?);
    }

    let mut model = AlignedUmapModel::new(AlignedUmapParams {
        umap: UmapParams {
            n_neighbors: cfg.n_neighbors,
            n_components: cfg.n_components,
            n_epochs: Some(cfg.n_epochs),
            metric: cfg.metric,
            learning_rate: 1.0,
            min_dist: 0.1,
            spread: 1.0,
            local_connectivity: 1.0,
            set_op_mix_ratio: 1.0,
            repulsion_strength: 1.0,
            negative_sample_rate: 5,
            random_seed: cfg.seed,
            init: cfg.init,
            use_approximate_knn: cfg.use_approximate_knn,
            approx_knn_candidates: cfg.approx_knn_candidates,
            approx_knn_iters: cfg.approx_knn_iters,
            approx_knn_threshold: cfg.approx_knn_threshold,
            ..UmapParams::default()
        },
        alignment_regularization: cfg.alignment_regularization,
        alignment_learning_rate: cfg.alignment_learning_rate,
        alignment_epochs: Some(cfg.alignment_epochs),
        recenter_interval: cfg.recenter_interval,
    });

    let t0 = Instant::now();
    let embeddings = model.fit_transform_identity(&slices)?;
    let fit_time_sec = t0.elapsed().as_secs_f64();

    let mut shapes = Vec::with_capacity(embeddings.len());
    for (slice_idx, embedding) in embeddings.iter().enumerate() {
        let out_path = cfg
            .output_dir
            .join(format!("slice_{slice_idx}_embedding.csv"));
        write_csv(&out_path, embedding)?;

        let shape = format!("[{},{}]", embedding.len(), embedding[0].len());
        shapes.push(shape);
    }

    let alignment_gap = mean_adjacent_identity_gap(&embeddings);

    print!(
        "{{\"fit_time_sec\":{:.9},\"adjacent_identity_gap\":{:.9},\"n_slices\":{},\"metric\":\"{}\",\"init\":\"{}\",\"n_neighbors\":{},\"n_components\":{},\"n_epochs\":{},\"seed\":{},\"use_approximate_knn\":{},\"approx_knn_candidates\":{},\"approx_knn_iters\":{},\"approx_knn_threshold\":{},\"alignment_regularization\":{:.6},\"alignment_learning_rate\":{:.6},\"alignment_epochs\":{},\"recenter_interval\":{},\"slice_shapes\":[",
        fit_time_sec,
        alignment_gap,
        embeddings.len(),
        metric_name(cfg.metric),
        init_name(cfg.init),
        cfg.n_neighbors,
        cfg.n_components,
        cfg.n_epochs,
        cfg.seed,
        cfg.use_approximate_knn,
        cfg.approx_knn_candidates,
        cfg.approx_knn_iters,
        cfg.approx_knn_threshold,
        cfg.alignment_regularization,
        cfg.alignment_learning_rate,
        cfg.alignment_epochs,
        cfg.recenter_interval,
    );
    for (idx, shape) in shapes.iter().enumerate() {
        if idx > 0 {
            print!(",");
        }
        print!("{}", shape);
    }
    println!("]}}");

    Ok(())
}
