use std::fs::File;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use ndarray::{Array1, Array2};
use ndarray_npy::NpzReader;
use rctd_core::{run_doublet_mode, run_full_mode, run_multi_mode, sync_device, RctdConfig};

use crate::device_sel::ComputeDevice;
use serde::Serialize;

#[derive(Clone, Copy, Debug, Default, clap::ValueEnum)]
pub enum BenchDeconvMode {
    #[default]
    Full,
    Doublet,
    Multi,
}

#[derive(Debug, Clone, clap::Args)]
pub struct BenchNpzArgs {
    /// NPZ written by benchmarks/compare_xenium_python_rust.py (export step).
    #[arg(long)]
    pub input: PathBuf,
    #[arg(long, default_value_t = 4096)]
    pub batch_size: usize,
    /// Optional CSV path for the weight matrix (same format as `rctd run`).
    #[arg(long)]
    pub weights_csv: Option<PathBuf>,
    #[arg(long, value_enum, default_value_t = ComputeDevice::Cpu)]
    pub compute_device: ComputeDevice,
    #[arg(long, value_enum, default_value_t = BenchDeconvMode::Full)]
    pub mode: BenchDeconvMode,
}

#[derive(Serialize)]
struct BenchReport {
    mode: &'static str,
    elapsed_s: f64,
    weights_hash16: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    aux_hash16: Option<String>,
    n_pixels: usize,
    n_genes: usize,
    k_types: usize,
}

fn weights_hash(w: &Array2<f64>) -> String {
    let mut buf = Vec::with_capacity(w.len() * 8);
    for &v in w.iter() {
        let r = (v * 1e8).round() / 1e8;
        buf.extend_from_slice(&r.to_le_bytes());
    }
    let d = md5::compute(&buf);
    format!("{:x}", d)[..16].to_string()
}

fn hash_i32_mat(w: &ndarray::Array2<i32>) -> String {
    let mut buf = Vec::with_capacity(w.len() * 4);
    for &v in w.iter() {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    let d = md5::compute(&buf);
    format!("{:x}", d)[..16].to_string()
}

fn type_names(k: usize) -> Vec<String> {
    (0..k).map(|i| format!("t{i}")).collect()
}

fn doublet_aux_hash(res: &rctd_core::DoubletResult) -> String {
    let mut buf: Vec<u8> =
        Vec::with_capacity(res.weights_doublet.len() * 8 + res.spot_class.len() * 4);
    for &v in res.weights_doublet.iter() {
        let r = (f64::from(v) * 1e6).round() / 1e6;
        buf.extend_from_slice(&r.to_le_bytes());
    }
    for &v in res.spot_class.iter() {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    let d = md5::compute(&buf);
    format!("{:x}", d)[..16].to_string()
}

pub fn run_bench_npz(args: BenchNpzArgs) -> Result<()> {
    let f = File::open(&args.input).with_context(|| format!("open {}", args.input.display()))?;
    let mut npz = NpzReader::new(f).context("read npz")?;

    let counts: Array2<f64> = npz.by_name("counts").context("npz missing 'counts'")?;
    let numi: Array1<f64> = npz.by_name("numi").context("npz missing 'numi'")?;
    let norm_profiles: Array2<f64> = npz
        .by_name("norm_profiles")
        .context("npz missing 'norm_profiles'")?;
    let q_mat: Array2<f64> = npz.by_name("q_mat").context("npz missing 'q_mat'")?;
    let sq_mat: Array2<f64> = npz.by_name("sq_mat").context("npz missing 'sq_mat'")?;
    let x_vals: Array1<f64> = npz.by_name("x_vals").context("npz missing 'x_vals'")?;

    let n = counts.nrows();
    let g = counts.ncols();
    let k = norm_profiles.ncols();
    if numi.len() != n {
        anyhow::bail!("numi length {} != n_pixels {}", numi.len(), n);
    }
    if norm_profiles.nrows() != g {
        anyhow::bail!("norm_profiles rows {} != G {}", norm_profiles.nrows(), g);
    }

    let device = crate::device_sel::resolve(args.compute_device);
    let cfg = RctdConfig::default();
    let names = type_names(k);

    let (elapsed, weights, aux_hash16, mode_str): (f64, Array2<f64>, Option<String>, &'static str) =
        match args.mode {
            BenchDeconvMode::Full => {
                let t0 = Instant::now();
                let res = run_full_mode(
                    &counts,
                    &numi,
                    &norm_profiles,
                    &q_mat,
                    &sq_mat,
                    &x_vals,
                    args.batch_size,
                    &device,
                );
                sync_device(&device);
                let elapsed = t0.elapsed().as_secs_f64();
                (elapsed, res.weights, None, "full")
            }
            BenchDeconvMode::Doublet => {
                let t0 = Instant::now();
                let res = run_doublet_mode(
                    &counts,
                    &numi,
                    &norm_profiles,
                    names,
                    &q_mat,
                    &sq_mat,
                    &x_vals,
                    &cfg,
                    args.batch_size,
                    &device,
                );
                sync_device(&device);
                let elapsed = t0.elapsed().as_secs_f64();
                let aux = doublet_aux_hash(&res);
                (elapsed, res.weights, Some(aux), "doublet")
            }
            BenchDeconvMode::Multi => {
                let t0 = Instant::now();
                let res = run_multi_mode(
                    &counts,
                    &numi,
                    &norm_profiles,
                    names,
                    &q_mat,
                    &sq_mat,
                    &x_vals,
                    &cfg,
                    args.batch_size,
                    &device,
                );
                sync_device(&device);
                let elapsed = t0.elapsed().as_secs_f64();
                let aux = hash_i32_mat(&res.cell_type_indices);
                (elapsed, res.weights, Some(aux), "multi")
            }
        };

    let weights_hash16 = weights_hash(&weights);
    let report = BenchReport {
        mode: mode_str,
        elapsed_s: elapsed,
        weights_hash16,
        aux_hash16,
        n_pixels: n,
        n_genes: g,
        k_types: k,
    };
    println!("{}", serde_json::to_string(&report)?);

    if let Some(path) = args.weights_csv {
        write_weights_csv(&path, &weights)?;
        eprintln!("wrote {}", path.display());
    }

    Ok(())
}

fn write_weights_csv(path: &PathBuf, w: &Array2<f64>) -> Result<()> {
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    use std::io::Write;
    for row in w.rows() {
        let s = row
            .iter()
            .map(|x| format!("{:.8e}", x))
            .collect::<Vec<_>>()
            .join(",");
        writeln!(f, "{s}")?;
    }
    Ok(())
}
