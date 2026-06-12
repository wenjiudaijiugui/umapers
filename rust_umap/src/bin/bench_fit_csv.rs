use rust_umap::{Metric, UmapModel};
use std::env;
use std::error::Error;
use std::fs;
use std::path::Path;
use std::time::Instant;

#[path = "../cli_common.rs"]
mod common_cli;
use common_cli::{
    extract_optional_args, parse_umap_args, read_csv, read_csv_usize, read_sparse_csr, write_csv,
};

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
fn usage() {
    eprintln!(
        "Usage:\n  bench_fit_csv <input.csv> <output.csv> <n_neighbors> <n_components> <n_epochs> <seed> \\\n          <init:random|spectral> <use_approx:bool> <approx_candidates> <approx_iters> <approx_threshold> \\\n          <warmup> <repeats> [knn_idx.csv] [knn_dist.csv] [--metric euclidean|manhattan|cosine] [--knn-metric euclidean|manhattan|cosine] \\\n\
          [--ann-mode auto|exact|approximate] \\\n\
          [--learning-rate <f32>] [--min-dist <f32>] [--spread <f32>] \\\n\
          [--local-connectivity <f32>] [--set-op-mix-ratio <f32>] \\\n\
          [--repulsion-strength <f32>] [--negative-sample-rate <usize>]\n\
          Optional CSR input (fit mode): --csr-indptr <path> --csr-indices <path> --csr-data <path> --csr-n-cols <n>"
    );
}

fn mean_std(vals: &[f64]) -> (f64, f64) {
    if vals.is_empty() {
        return (0.0, 0.0);
    }
    let mean = vals.iter().sum::<f64>() / vals.len() as f64;
    let var = vals
        .iter()
        .map(|v| {
            let d = *v - mean;
            d * d
        })
        .sum::<f64>()
        / vals.len() as f64;
    (mean, var.sqrt())
}

fn precomputed_arg_arity(args_len: usize) -> Result<Option<(usize, usize)>, Box<dyn Error>> {
    match args_len.saturating_sub(14) {
        0 => Ok(None),
        2 => Ok(Some((14, 15))),
        1 => Err("precomputed kNN requires both knn_idx.csv and knn_dist.csv".into()),
        _ => Err("unexpected positional arguments after repeats".into()),
    }
}

fn read_proc_status_kb(field: &str) -> Option<u64> {
    let status = fs::read_to_string("/proc/self/status").ok()?;
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix(field) {
            let value = rest.split_whitespace().next()?;
            return value.parse::<u64>().ok();
        }
    }
    None
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut args: Vec<String> = env::args().collect();
    let optional = extract_optional_args(&mut args)?;
    if args.len() < 14 {
        usage();
        return Err("insufficient arguments".into());
    }

    let input_path = Path::new(&args[1]);
    let output_path = Path::new(&args[2]);

    let parsed = parse_umap_args(&args)?;
    let warmup = args[12].parse::<usize>()?;
    let repeats = args[13].parse::<usize>()?;

    let precomputed = if let Some((idx_pos, dist_pos)) = precomputed_arg_arity(args.len())? {
        Some((
            read_csv_usize(Path::new(&args[idx_pos]))?,
            read_csv(Path::new(&args[dist_pos]))?,
        ))
    } else {
        None
    };
    if precomputed.is_none() && optional.knn_metric.is_some() {
        return Err("--knn-metric requires precomputed kNN inputs".into());
    }
    if optional.sparse_input.is_some() && precomputed.is_some() {
        return Err("CSR input cannot be combined with precomputed kNN files".into());
    }
    let knn_metric = optional.knn_metric.unwrap_or(optional.metric);

    let dense_data = if optional.sparse_input.is_some() {
        None
    } else {
        Some(read_csv(input_path)?)
    };
    let sparse_data = if let Some(spec) = optional.sparse_input.as_ref() {
        Some(read_sparse_csr(spec)?)
    } else {
        None
    };

    let mut params = parsed.build_params(optional.metric, optional.ann_mode);
    optional.overrides.apply_to(&mut params)?;

    let total = warmup + repeats;
    let mut times = Vec::with_capacity(repeats);
    let mut last_embedding: Option<Vec<Vec<f32>>> = None;
    let vmhwm_before = read_proc_status_kb("VmHWM:");
    let vmrss_before = read_proc_status_kb("VmRSS:");

    for i in 0..total {
        let mut model = UmapModel::new(params.clone());
        let (embedding, dt) = if let Some((ref idx, ref dist)) = precomputed {
            let data = dense_data
                .as_ref()
                .ok_or("precomputed kNN requires dense input data")?;
            let t0 = Instant::now();
            (
                model.fit_transform_with_knn_metric(data, idx, dist, knn_metric)?,
                t0.elapsed().as_secs_f64(),
            )
        } else if let Some(data) = sparse_data.as_ref() {
            let sparse_input = data.clone();
            let t0 = Instant::now();
            (
                model.fit_transform_sparse_csr(sparse_input)?,
                t0.elapsed().as_secs_f64(),
            )
        } else {
            let data = dense_data
                .as_ref()
                .ok_or("dense input data is unavailable")?;
            let t0 = Instant::now();
            (model.fit_transform(data)?, t0.elapsed().as_secs_f64())
        };
        if i >= warmup {
            times.push(dt);
        }
        last_embedding = Some(embedding);
    }
    let vmhwm_after = read_proc_status_kb("VmHWM:");
    let vmrss_after = read_proc_status_kb("VmRSS:");

    if let Some(embedding) = last_embedding.as_ref() {
        write_csv(output_path, embedding)?;
    }

    let (mean, std) = mean_std(&times);
    let algo_mem_proxy_available = vmhwm_before.is_some()
        && vmhwm_after.is_some()
        && vmrss_before.is_some()
        && vmrss_after.is_some();
    let vmhwm_before_mb = vmhwm_before.unwrap_or(0) as f64 / 1024.0;
    let vmhwm_after_mb = vmhwm_after.unwrap_or(0) as f64 / 1024.0;
    let vmrss_before_mb = vmrss_before.unwrap_or(0) as f64 / 1024.0;
    let vmrss_after_mb = vmrss_after.unwrap_or(0) as f64 / 1024.0;
    let algo_peak_delta_mb = if let (Some(before), Some(after)) = (vmhwm_before, vmhwm_after) {
        after.saturating_sub(before) as f64 / 1024.0
    } else {
        0.0
    };

    print!(
        "{{\"mode\":\"fit\",\
\"input_format\":\"{}\",\
\"metric\":\"{}\",\
\"knn_metric\":\"{}\",\
\"precomputed_knn\":{},\
\"n_neighbors\":{},\
\"n_components\":{},\
\"n_epochs\":{},\
\"seed\":{},\
\"warmup\":{},\
\"repeats\":{},\
\"fit_times_sec\":[",
        if sparse_data.is_some() {
            "csr"
        } else {
            "dense"
        },
        metric_name(params.metric),
        metric_name(knn_metric),
        precomputed.is_some(),
        parsed.n_neighbors,
        parsed.n_components,
        parsed.n_epochs,
        parsed.seed,
        warmup,
        repeats
    );
    for (i, t) in times.iter().enumerate() {
        if i > 0 {
            print!(",");
        }
        print!("{:.9}", t);
    }
    println!(
        "],\"fit_mean_sec\":{:.9},\"fit_std_sec\":{:.9},\
\"algorithm_phase_memory_proxy_available\":{},\
\"algorithm_phase_peak_rss_delta_mb\":{:.9},\
\"algorithm_phase_vmrss_before_mb\":{:.9},\
\"algorithm_phase_vmrss_after_mb\":{:.9},\
\"algorithm_phase_vmhwm_before_mb\":{:.9},\
\"algorithm_phase_vmhwm_after_mb\":{:.9}}}",
        mean,
        std,
        algo_mem_proxy_available,
        algo_peak_delta_mb,
        vmrss_before_mb,
        vmrss_after_mb,
        vmhwm_before_mb,
        vmhwm_after_mb
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::precomputed_arg_arity;

    #[test]
    fn precomputed_arg_arity_accepts_zero_or_two_extra_paths() {
        assert_eq!(precomputed_arg_arity(14).unwrap(), None);
        assert_eq!(precomputed_arg_arity(16).unwrap(), Some((14, 15)));
    }

    #[test]
    fn precomputed_arg_arity_rejects_partial_or_extra_paths() {
        let one_path = precomputed_arg_arity(15).expect_err("single precomputed path should fail");
        assert!(
            one_path
                .to_string()
                .contains("precomputed kNN requires both knn_idx.csv and knn_dist.csv")
        );

        let too_many = precomputed_arg_arity(17).expect_err("extra positional args should fail");
        assert!(
            too_many
                .to_string()
                .contains("unexpected positional arguments after repeats")
        );
    }
}
