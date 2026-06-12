use rust_umap::{InitMethod, Metric, SparseCsrMatrix, UmapParams};
use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum AnnMode {
    #[default]
    Auto,
    Exact,
    Approximate,
}

impl AnnMode {
    pub fn apply_to(self, params: &mut UmapParams) {
        match self {
            Self::Auto => {}
            Self::Exact => {
                params.use_approximate_knn = false;
            }
            Self::Approximate => {
                params.use_approximate_knn = true;
                params.approx_knn_threshold = 0;
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SparseInputArgs {
    pub indptr_path: String,
    pub indices_path: String,
    pub data_path: String,
    pub n_cols: usize,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct UmapParamOverrides {
    pub learning_rate: Option<f32>,
    pub min_dist: Option<f32>,
    pub spread: Option<f32>,
    pub local_connectivity: Option<f32>,
    pub set_op_mix_ratio: Option<f32>,
    pub repulsion_strength: Option<f32>,
    pub negative_sample_rate: Option<usize>,
}

impl UmapParamOverrides {
    pub fn apply_to(self, params: &mut UmapParams) -> Result<(), Box<dyn Error>> {
        fn ensure_finite(name: &str, value: f32) -> Result<(), Box<dyn Error>> {
            if value.is_finite() {
                Ok(())
            } else {
                Err(format!("{name} must be finite").into())
            }
        }

        if let Some(value) = self.learning_rate {
            ensure_finite("--learning-rate", value)?;
            params.learning_rate = value;
        }
        if let Some(value) = self.min_dist {
            ensure_finite("--min-dist", value)?;
            params.min_dist = value;
        }
        if let Some(value) = self.spread {
            ensure_finite("--spread", value)?;
            params.spread = value;
        }
        if let Some(value) = self.local_connectivity {
            ensure_finite("--local-connectivity", value)?;
            params.local_connectivity = value;
        }
        if let Some(value) = self.set_op_mix_ratio {
            ensure_finite("--set-op-mix-ratio", value)?;
            params.set_op_mix_ratio = value;
        }
        if let Some(value) = self.repulsion_strength {
            ensure_finite("--repulsion-strength", value)?;
            params.repulsion_strength = value;
        }
        if let Some(value) = self.negative_sample_rate {
            params.negative_sample_rate = value;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParsedUmapArgs {
    pub n_neighbors: usize,
    pub n_components: usize,
    pub n_epochs: usize,
    pub seed: u64,
    pub init: InitMethod,
    pub use_approximate_knn: bool,
    pub approx_knn_candidates: usize,
    pub approx_knn_iters: usize,
    pub approx_knn_threshold: usize,
}

impl ParsedUmapArgs {
    pub fn build_params(self, metric: Metric, ann_mode: AnnMode) -> UmapParams {
        let mut params = UmapParams {
            n_neighbors: self.n_neighbors,
            n_components: self.n_components,
            n_epochs: Some(self.n_epochs),
            metric,
            random_seed: self.seed,
            init: self.init,
            use_approximate_knn: self.use_approximate_knn,
            approx_knn_candidates: self.approx_knn_candidates,
            approx_knn_iters: self.approx_knn_iters,
            approx_knn_threshold: self.approx_knn_threshold,
            ..UmapParams::default()
        };
        ann_mode.apply_to(&mut params);
        params
    }
}

pub fn parse_umap_args(args: &[String]) -> Result<ParsedUmapArgs, Box<dyn Error>> {
    Ok(ParsedUmapArgs {
        n_neighbors: args[3].parse::<usize>()?,
        n_components: args[4].parse::<usize>()?,
        n_epochs: args[5].parse::<usize>()?,
        seed: args[6].parse::<u64>()?,
        init: parse_init(&args[7])?,
        use_approximate_knn: parse_bool(&args[8])?,
        approx_knn_candidates: args[9].parse::<usize>()?,
        approx_knn_iters: args[10].parse::<usize>()?,
        approx_knn_threshold: args[11].parse::<usize>()?,
    })
}

pub fn parse_ann_mode(s: &str) -> Result<AnnMode, Box<dyn Error>> {
    match s.to_ascii_lowercase().as_str() {
        "auto" => Ok(AnnMode::Auto),
        "exact" => Ok(AnnMode::Exact),
        "approximate" => Ok(AnnMode::Approximate),
        _ => Err(format!("unsupported ann mode '{s}', expected auto|exact|approximate").into()),
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CliOptionalArgs {
    pub metric: Metric,
    pub knn_metric: Option<Metric>,
    pub ann_mode: AnnMode,
    pub sparse_input: Option<SparseInputArgs>,
    pub overrides: UmapParamOverrides,
}

fn parse_flag_value<T>(args: &[String], index: usize, flag: &str) -> Result<T, Box<dyn Error>>
where
    T: std::str::FromStr,
    T::Err: Error + 'static,
{
    if index + 1 >= args.len() {
        return Err(format!("{flag} requires a value").into());
    }
    args[index + 1]
        .parse::<T>()
        .map_err(|e| Box::new(e) as Box<dyn Error>)
}

pub fn parse_bool(s: &str) -> Result<bool, Box<dyn Error>> {
    match s.to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "y" => Ok(true),
        "0" | "false" | "no" | "n" => Ok(false),
        _ => Err(format!("cannot parse bool from '{s}'").into()),
    }
}

pub fn parse_init(s: &str) -> Result<InitMethod, Box<dyn Error>> {
    match s.to_ascii_lowercase().as_str() {
        "random" => Ok(InitMethod::Random),
        "spectral" => Ok(InitMethod::Spectral),
        _ => Err(format!("unsupported init '{s}', expected random|spectral").into()),
    }
}

pub fn parse_metric(s: &str) -> Result<Metric, Box<dyn Error>> {
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

pub fn extract_optional_args(args: &mut Vec<String>) -> Result<CliOptionalArgs, Box<dyn Error>> {
    let mut metric = Metric::Euclidean;
    let mut knn_metric: Option<Metric> = None;
    let mut ann_mode = AnnMode::Auto;
    let mut csr_indptr: Option<String> = None;
    let mut csr_indices: Option<String> = None;
    let mut csr_data: Option<String> = None;
    let mut csr_n_cols: Option<usize> = None;
    let mut overrides = UmapParamOverrides::default();
    let mut i = 1;
    while i < args.len() {
        if args[i] == "--metric" {
            if i + 1 >= args.len() {
                return Err("--metric requires a value".into());
            }
            metric = parse_metric(&args[i + 1])?;
            args.drain(i..=i + 1);
        } else if args[i] == "--knn-metric" {
            if i + 1 >= args.len() {
                return Err("--knn-metric requires a value".into());
            }
            knn_metric = Some(parse_metric(&args[i + 1])?);
            args.drain(i..=i + 1);
        } else if args[i] == "--ann-mode" {
            if i + 1 >= args.len() {
                return Err("--ann-mode requires a value".into());
            }
            ann_mode = parse_ann_mode(&args[i + 1])?;
            args.drain(i..=i + 1);
        } else if args[i] == "--csr-indptr" {
            if i + 1 >= args.len() {
                return Err("--csr-indptr requires a file path".into());
            }
            csr_indptr = Some(args[i + 1].clone());
            args.drain(i..=i + 1);
        } else if args[i] == "--csr-indices" {
            if i + 1 >= args.len() {
                return Err("--csr-indices requires a file path".into());
            }
            csr_indices = Some(args[i + 1].clone());
            args.drain(i..=i + 1);
        } else if args[i] == "--csr-data" {
            if i + 1 >= args.len() {
                return Err("--csr-data requires a file path".into());
            }
            csr_data = Some(args[i + 1].clone());
            args.drain(i..=i + 1);
        } else if args[i] == "--csr-n-cols" {
            if i + 1 >= args.len() {
                return Err("--csr-n-cols requires an integer value".into());
            }
            csr_n_cols = Some(args[i + 1].parse::<usize>()?);
            args.drain(i..=i + 1);
        } else if args[i] == "--learning-rate" {
            overrides.learning_rate = Some(parse_flag_value(args, i, "--learning-rate")?);
            args.drain(i..=i + 1);
        } else if args[i] == "--min-dist" {
            overrides.min_dist = Some(parse_flag_value(args, i, "--min-dist")?);
            args.drain(i..=i + 1);
        } else if args[i] == "--spread" {
            overrides.spread = Some(parse_flag_value(args, i, "--spread")?);
            args.drain(i..=i + 1);
        } else if args[i] == "--local-connectivity" {
            overrides.local_connectivity = Some(parse_flag_value(args, i, "--local-connectivity")?);
            args.drain(i..=i + 1);
        } else if args[i] == "--set-op-mix-ratio" {
            overrides.set_op_mix_ratio = Some(parse_flag_value(args, i, "--set-op-mix-ratio")?);
            args.drain(i..=i + 1);
        } else if args[i] == "--repulsion-strength" {
            overrides.repulsion_strength = Some(parse_flag_value(args, i, "--repulsion-strength")?);
            args.drain(i..=i + 1);
        } else if args[i] == "--negative-sample-rate" {
            overrides.negative_sample_rate =
                Some(parse_flag_value(args, i, "--negative-sample-rate")?);
            args.drain(i..=i + 1);
        } else {
            i += 1;
        }
    }

    let has_any_csr =
        csr_indptr.is_some() || csr_indices.is_some() || csr_data.is_some() || csr_n_cols.is_some();
    let sparse = if has_any_csr {
        let indptr_path = csr_indptr.ok_or("--csr-indptr is required when using CSR input")?;
        let indices_path = csr_indices.ok_or("--csr-indices is required when using CSR input")?;
        let data_path = csr_data.ok_or("--csr-data is required when using CSR input")?;
        let n_cols = csr_n_cols.ok_or("--csr-n-cols is required when using CSR input")?;
        Some(SparseInputArgs {
            indptr_path,
            indices_path,
            data_path,
            n_cols,
        })
    } else {
        None
    };

    Ok(CliOptionalArgs {
        metric,
        knn_metric,
        ann_mode,
        sparse_input: sparse,
        overrides,
    })
}

pub fn read_csv(path: &Path) -> Result<Vec<Vec<f32>>, Box<dyn Error>> {
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

pub fn read_csv_usize(path: &Path) -> Result<Vec<Vec<usize>>, Box<dyn Error>> {
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
            .map(|x| x.trim().parse::<usize>())
            .collect::<Result<Vec<usize>, _>>()?;
        data.push(row);
    }
    Ok(data)
}

fn read_csv_flat_usize(path: &Path) -> Result<Vec<usize>, Box<dyn Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut values = Vec::new();

    for line in reader.lines() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        for token in trimmed.split(',') {
            let token = token.trim();
            if token.is_empty() {
                continue;
            }
            values.push(token.parse::<usize>()?);
        }
    }

    Ok(values)
}

fn read_csv_flat_f32(path: &Path) -> Result<Vec<f32>, Box<dyn Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut values = Vec::new();

    for line in reader.lines() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        for token in trimmed.split(',') {
            let token = token.trim();
            if token.is_empty() {
                continue;
            }
            values.push(token.parse::<f32>()?);
        }
    }

    Ok(values)
}

pub fn read_sparse_csr(spec: &SparseInputArgs) -> Result<SparseCsrMatrix, Box<dyn Error>> {
    let indptr = read_csv_flat_usize(Path::new(&spec.indptr_path))?;
    let indices = read_csv_flat_usize(Path::new(&spec.indices_path))?;
    let values = read_csv_flat_f32(Path::new(&spec.data_path))?;
    if indptr.is_empty() {
        return Err("csr indptr cannot be empty".into());
    }
    let n_rows = indptr.len() - 1;
    SparseCsrMatrix::new(n_rows, spec.n_cols, indptr, indices, values)
        .map_err(|e| -> Box<dyn Error> { Box::new(e) })
}

pub fn write_csv(path: &Path, arr: &[Vec<f32>]) -> Result<(), Box<dyn Error>> {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_optional_args_parses_overrides_and_sparse_input() {
        let mut args = vec![
            "fit_csv".to_string(),
            "in.csv".to_string(),
            "out.csv".to_string(),
            "--metric".to_string(),
            "cosine".to_string(),
            "--knn-metric".to_string(),
            "manhattan".to_string(),
            "--ann-mode".to_string(),
            "exact".to_string(),
            "--learning-rate".to_string(),
            "0.25".to_string(),
            "--min-dist".to_string(),
            "0.05".to_string(),
            "--spread".to_string(),
            "1.7".to_string(),
            "--local-connectivity".to_string(),
            "2.5".to_string(),
            "--set-op-mix-ratio".to_string(),
            "0.6".to_string(),
            "--repulsion-strength".to_string(),
            "1.9".to_string(),
            "--negative-sample-rate".to_string(),
            "11".to_string(),
            "--csr-indptr".to_string(),
            "indptr.csv".to_string(),
            "--csr-indices".to_string(),
            "indices.csv".to_string(),
            "--csr-data".to_string(),
            "data.csv".to_string(),
            "--csr-n-cols".to_string(),
            "128".to_string(),
            "fit".to_string(),
        ];

        let parsed = extract_optional_args(&mut args).expect("flags should parse");
        assert_eq!(parsed.metric, Metric::Cosine);
        assert_eq!(parsed.knn_metric, Some(Metric::Manhattan));
        assert_eq!(parsed.ann_mode, AnnMode::Exact);
        assert_eq!(
            parsed.overrides,
            UmapParamOverrides {
                learning_rate: Some(0.25),
                min_dist: Some(0.05),
                spread: Some(1.7),
                local_connectivity: Some(2.5),
                set_op_mix_ratio: Some(0.6),
                repulsion_strength: Some(1.9),
                negative_sample_rate: Some(11),
            }
        );

        let sparse = parsed
            .sparse_input
            .expect("csr flags should produce sparse input");
        assert_eq!(sparse.indptr_path, "indptr.csv");
        assert_eq!(sparse.indices_path, "indices.csv");
        assert_eq!(sparse.data_path, "data.csv");
        assert_eq!(sparse.n_cols, 128);

        assert_eq!(
            args,
            vec![
                "fit_csv".to_string(),
                "in.csv".to_string(),
                "out.csv".to_string(),
                "fit".to_string(),
            ]
        );
    }

    #[test]
    fn extract_optional_args_rejects_missing_override_value() {
        let mut args = vec!["fit_csv".to_string(), "--learning-rate".to_string()];
        let err = extract_optional_args(&mut args).expect_err("missing value should fail");
        assert!(err.to_string().contains("--learning-rate requires a value"));
    }

    #[test]
    fn extract_optional_args_requires_complete_csr_tuple() {
        let mut args = vec![
            "fit_csv".to_string(),
            "--csr-indptr".to_string(),
            "indptr.csv".to_string(),
        ];
        let err = extract_optional_args(&mut args).expect_err("partial csr input should fail");
        assert!(err.to_string().contains("--csr-indices is required"));
    }

    #[test]
    fn overrides_reject_non_finite_float_values() {
        let mut params = UmapParams::default();
        let err = UmapParamOverrides {
            learning_rate: Some(f32::NAN),
            ..UmapParamOverrides::default()
        }
        .apply_to(&mut params)
        .expect_err("NaN override must fail");
        assert!(err.to_string().contains("--learning-rate must be finite"));
    }

    #[test]
    fn parse_umap_args_reads_shared_positional_fields() {
        let args = vec![
            "fit_csv".to_string(),
            "in.csv".to_string(),
            "out.csv".to_string(),
            "15".to_string(),
            "2".to_string(),
            "120".to_string(),
            "99".to_string(),
            "random".to_string(),
            "true".to_string(),
            "40".to_string(),
            "11".to_string(),
            "2048".to_string(),
        ];

        let parsed = parse_umap_args(&args).expect("shared positional args should parse");
        assert_eq!(
            parsed,
            ParsedUmapArgs {
                n_neighbors: 15,
                n_components: 2,
                n_epochs: 120,
                seed: 99,
                init: InitMethod::Random,
                use_approximate_knn: true,
                approx_knn_candidates: 40,
                approx_knn_iters: 11,
                approx_knn_threshold: 2048,
            }
        );
    }

    #[test]
    fn ann_mode_mapping_respects_auto_exact_and_approximate() {
        let parsed = ParsedUmapArgs {
            n_neighbors: 15,
            n_components: 2,
            n_epochs: 120,
            seed: 99,
            init: InitMethod::Random,
            use_approximate_knn: false,
            approx_knn_candidates: 40,
            approx_knn_iters: 11,
            approx_knn_threshold: 2048,
        };

        let auto = parsed.build_params(Metric::Euclidean, AnnMode::Auto);
        assert!(!auto.use_approximate_knn);
        assert_eq!(auto.approx_knn_threshold, 2048);

        let exact = parsed.build_params(Metric::Euclidean, AnnMode::Exact);
        assert!(!exact.use_approximate_knn);
        assert_eq!(exact.approx_knn_threshold, 2048);

        let approximate = parsed.build_params(Metric::Euclidean, AnnMode::Approximate);
        assert!(approximate.use_approximate_knn);
        assert_eq!(approximate.approx_knn_threshold, 0);
    }
}
