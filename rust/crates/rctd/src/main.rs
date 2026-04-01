mod bench;
mod device_sel;
mod ref_adata;
mod ref_rds;

use std::collections::HashMap;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use anndata::{AnnData, AnnDataOp, Backend};
use anndata_hdf5::H5;
use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use device_sel::ComputeDevice;
use ndarray::{Array1, Array2};
use rctd_core::io_npz::load_q_matrices_npz;
use rctd_core::likelihood_tables::compute_spline_coefficients;
use rctd_core::{run_deconvolution, sync_device, DeconvMode, PreparedData, RctdConfig};

const Q_MATRICES_URL: &str =
    "https://github.com/p-gueguen/rctd-py/releases/download/v0.1.1/q_matrices.npz";

#[derive(Parser)]
#[command(name = "rctd", about = "RCTD spatial deconvolution (Rust + Burn)")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Info,
    /// Run full-mode IRWLS on a Python-exported NPZ (for parity / timing vs PyTorch).
    BenchNpz(bench::BenchNpzArgs),
    Run {
        #[arg(
            long,
            help = "Spatial input: AnnData .h5ad (X = N×G counts, var = genes) or spacexr::SpatialRNA .rds (via Rscript)"
        )]
        spatial: PathBuf,
        #[arg(
            long,
            help = "Reference: .h5ad single-cell + --cell-type-col, or K×G + --ref-rows-are-types; .rds spacexr::Reference (single-cell) or numeric K×G matrix (with --ref-rows-are-types)"
        )]
        reference: PathBuf,
        #[arg(
            long,
            default_value = "cell_type",
            help = "obs column with cell type labels (h5ad single-cell reference only; ignored for spacexr::Reference .rds)"
        )]
        cell_type_col: String,
        #[arg(
            long,
            help = "Reference X is K×G (one row per cell type); obs_names are type labels (legacy)"
        )]
        ref_rows_are_types: bool,
        #[arg(long, default_value_t = 25)]
        ref_cell_min: usize,
        #[arg(long, default_value_t = 100)]
        ref_min_umi: u32,
        #[arg(long, default_value_t = 10000)]
        ref_max_cells_per_type: usize,
        #[arg(long, help = "Optional q_matrices.npz; defaults to ~/.cache/rctd/q_matrices.npz (auto-download on first use)")]
        q_matrices: Option<PathBuf>,
        #[arg(long, default_value = "100")]
        sigma: i32,
        #[arg(long, value_enum, default_value_t = ModeArg::Full)]
        mode: ModeArg,
        #[arg(long, default_value_t = 4096)]
        batch_size: usize,
        #[arg(long, help = "Optional output prefix; writes weights CSV if set")]
        output_prefix: Option<PathBuf>,
        #[arg(
            long,
            value_enum,
            default_value_t = ComputeDevice::Cpu,
            help = "Compute device (GPU requires building with --features wgpu)"
        )]
        compute_device: ComputeDevice,
    },
}

#[derive(Clone, Copy, Debug, clap::ValueEnum)]
enum ModeArg {
    Full,
    Doublet,
    Multi,
}

impl From<ModeArg> for DeconvMode {
    fn from(m: ModeArg) -> Self {
        match m {
            ModeArg::Full => DeconvMode::Full,
            ModeArg::Doublet => DeconvMode::Doublet,
            ModeArg::Multi => DeconvMode::Multi,
        }
    }
}

fn open_h5ad(path: &PathBuf) -> Result<AnnData<H5>> {
    let store = H5::open(path).with_context(|| format!("open {}", path.display()))?;
    AnnData::open(store).with_context(|| format!("read AnnData {}", path.display()))
}

fn input_is_rds(path: &std::path::Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|s| s.eq_ignore_ascii_case("rds"))
}

fn align_genes_spatial_ref(
    spatial_genes: &[String],
    ref_genes: &[String],
    counts: &Array2<f64>,
    profiles_kg: &Array2<f64>,
) -> Result<(Array2<f64>, Array2<f64>)> {
    let map_ref: HashMap<&str, usize> = ref_genes
        .iter()
        .enumerate()
        .map(|(i, g)| (g.as_str(), i))
        .collect();
    let mut genes = Vec::new();
    let mut si = Vec::new();
    let mut ri = Vec::new();
    for (i, g) in spatial_genes.iter().enumerate() {
        if let Some(&j) = map_ref.get(g.as_str()) {
            genes.push(g.clone());
            si.push(i);
            ri.push(j);
        }
    }
    if genes.is_empty() {
        bail!("no overlapping gene names between spatial and reference");
    }
    let n = genes.len();
    let k = profiles_kg.nrows();
    let mut c = Array2::<f64>::zeros((counts.nrows(), n));
    let mut p = Array2::<f64>::zeros((k, n));
    for (new_g, (is, ir)) in si.iter().zip(ri.iter()).enumerate() {
        c.column_mut(new_g).assign(&counts.column(*is));
        p.column_mut(new_g).assign(&profiles_kg.column(*ir));
    }
    let profiles_gk = p.t().to_owned();
    Ok((c, profiles_gk))
}

fn resolve_q_matrices_path(arg: Option<PathBuf>) -> Result<PathBuf> {
    if let Some(p) = arg {
        if !p.exists() {
            bail!("q_matrices.npz not found at {} (pass a valid path or omit --q-matrices to use the cached/downloaded file)", p.display());
        }
        return Ok(p);
    }

    let mut cache_path = dirs_next::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".cache")
        .join("rctd")
        .join("q_matrices.npz");

    if cache_path.exists() {
        return Ok(cache_path);
    }

    if let Some(parent) = cache_path.parent() {
        fs::create_dir_all(parent)?;
    }

    eprintln!(
        "Downloading Q-matrices from {} to {} ...",
        Q_MATRICES_URL,
        cache_path.display()
    );
    let response = ureq::get(Q_MATRICES_URL)
        .call()
        .map_err(|e| anyhow::Error::new(io::Error::new(io::ErrorKind::Other, format!("{e}"))))
        .context("download q_matrices.npz")?;

    let mut reader = response.into_reader();
    let mut file = std::fs::File::create(&cache_path)?;
    io::copy(&mut reader, &mut file)?;
    file.flush()?;
    eprintln!("Saved Q-matrices to {}", cache_path.display());

    Ok(cache_path)
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let cli = Cli::parse();
    match cli.command {
        Commands::BenchNpz(args) => bench::run_bench_npz(args)?,
        Commands::Info => {
            println!(
                "rctd {} (rctd-core path workspace crate)",
                env!("CARGO_PKG_VERSION")
            );
            println!("Default tensor backend: NdArray CPU f64");
            #[cfg(feature = "wgpu")]
            println!("Optional: `cargo build -p rctd --features wgpu` then `--compute-device wgpu` (float32 on GPU)");
            println!("CUDA (NVIDIA): `cargo build -p rctd --features cuda`");
        }
        Commands::Run {
            spatial,
            reference,
            cell_type_col,
            ref_rows_are_types,
            ref_cell_min,
            ref_min_umi,
            ref_max_cells_per_type,
            q_matrices,
            sigma,
            mode,
            batch_size,
            output_prefix,
            compute_device,
        } => {
            let dev = device_sel::resolve(compute_device);
            let (counts, spatial_genes) = if input_is_rds(&spatial) {
                ref_rds::load_spatial_rds(spatial.as_path())
                    .context("spatial .rds (expect spacexr::SpatialRNA; install R, see rctd resources/export_rds_bridge.R)")?
            } else {
                let spatial_ad = open_h5ad(&spatial)?;
                let spatial_genes = spatial_ad.var_names().into_vec();
                let counts = ref_adata::x_to_dense_f64(&spatial_ad)?;
                if counts.ncols() != spatial_genes.len() || counts.nrows() != spatial_ad.n_obs() {
                    bail!(
                        "spatial X shape {:?} does not match n_obs × n_vars",
                        counts.dim()
                    );
                }
                (counts, spatial_genes)
            };

            let (profiles_kg, cell_type_names, ref_genes) = if input_is_rds(&reference) {
                if ref_rows_are_types {
                    ref_rds::load_reference_profiles_rds(reference.as_path()).context(
                        "reference .rds type profiles (matrix with rownames = types, colnames = genes)",
                    )?
                } else {
                    ref_rds::load_reference_sc_rds(
                        reference.as_path(),
                        ref_cell_min,
                        f64::from(ref_min_umi),
                        ref_max_cells_per_type,
                    )
                    .context("reference .rds (expect spacexr::Reference)")?
                }
            } else {
                let ref_ad = open_h5ad(&reference)?;
                let ref_genes = ref_ad.var_names().into_vec();
                let pair = if ref_rows_are_types {
                    let p = ref_adata::x_to_dense_f64(&ref_ad)?;
                    if p.ncols() != ref_genes.len() || p.nrows() != ref_ad.n_obs() {
                        bail!(
                            "reference X shape {:?} does not match n_obs × n_vars ({} × {})",
                            p.dim(),
                            ref_ad.n_obs(),
                            ref_genes.len()
                        );
                    }
                    let names = ref_ad.obs_names().into_vec();
                    if names.len() != p.nrows() {
                        bail!("obs index length does not match reference n_obs");
                    }
                    (p, names)
                } else {
                    let obs = ref_ad.read_obs().context("read reference obs")?;
                    if obs.get_column_index(&cell_type_col).is_none() {
                        bail!(
                            "obs has no column {:?}; use --ref-rows-are-types if reference X is already K×G (one row per type)",
                            cell_type_col
                        );
                    }
                    ref_adata::single_cell_reference_profiles(
                        &ref_ad,
                        &cell_type_col,
                        ref_cell_min,
                        f64::from(ref_min_umi),
                        ref_max_cells_per_type,
                    )
                    .with_context(|| "single-cell reference profiles")?
                };
                (pair.0, pair.1, ref_genes)
            };
            let (counts, profiles_gk_raw) =
                align_genes_spatial_ref(&spatial_genes, &ref_genes, &counts, &profiles_kg)?;
            let norm_profiles = ref_adata::normalize_columns(&profiles_gk_raw);
            if cell_type_names.len() != norm_profiles.ncols() {
                bail!("cell type count does not match norm_profiles columns");
            }
            let numi: Array1<f64> = counts.sum_axis(ndarray::Axis(1));

            let q_path = resolve_q_matrices_path(q_matrices)?;
            let (q_map, x_vals) = load_q_matrices_npz(q_path.as_path())?;
            let q_prefixed = format!("Q_{sigma}");
            let q_mat = q_map
                .get(&q_prefixed)
                .or_else(|| q_map.get(&sigma.to_string()))
                .with_context(|| {
                    format!(
                        "sigma {sigma} not in q_matrices.npz (try keys like Q_{sigma})"
                    )
                })?
                .clone();
            let sq_mat = compute_spline_coefficients(&q_mat, &x_vals);

            let config = RctdConfig::default();
            let data = PreparedData {
                spatial_counts: counts,
                spatial_numi: numi,
                norm_profiles,
                cell_type_names,
                q_mat,
                sq_mat,
                x_vals,
            };

            let bar = indicatif::ProgressBar::new_spinner();
            bar.set_message("running deconvolution…");
            let out = run_deconvolution(&data, &config, mode.into(), batch_size, &dev);
            sync_device(&dev);
            bar.finish_and_clear();

            if let Some(prefix) = output_prefix {
                match out {
                    rctd_core::DeconvolutionOutput::Full(r) => {
                        write_weights_csv(&prefix.with_extension("weights.csv"), &r.weights)?;
                    }
                    rctd_core::DeconvolutionOutput::Doublet(r) => {
                        write_weights_csv(&prefix.with_extension("weights.csv"), &r.weights)?;
                    }
                    rctd_core::DeconvolutionOutput::Multi(r) => {
                        write_weights_csv(&prefix.with_extension("weights.csv"), &r.weights)?;
                    }
                }
                println!("wrote {}", prefix.with_extension("weights.csv").display());
            }
        }
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

#[cfg(test)]
mod tests {
    use super::resolve_q_matrices_path;
    use std::env;
    use std::fs;
    use std::path::PathBuf;
    use tempfile::TempDir;

    #[test]
    fn resolve_q_path_uses_explicit_existing_path() {
        let tmp = TempDir::new().unwrap();
        let file = tmp.path().join("q_matrices.npz");
        fs::write(&file, b"dummy").unwrap();

        let out = resolve_q_matrices_path(Some(file.clone())).unwrap();
        assert_eq!(out, file);
    }

    #[test]
    fn resolve_q_path_errors_on_missing_explicit_path() {
        let tmp = TempDir::new().unwrap();
        let file = tmp.path().join("missing_q.npz");
        let err = resolve_q_matrices_path(Some(file)).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("q_matrices.npz not found"));
    }

    #[test]
    fn resolve_q_path_uses_cached_home_path_when_present() {
        let tmp_home = TempDir::new().unwrap();
        env::set_var("HOME", tmp_home.path());

        let cache_path: PathBuf = tmp_home
            .path()
            .join(".cache")
            .join("rctd")
            .join("q_matrices.npz");
        if let Some(parent) = cache_path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(&cache_path, b"dummy").unwrap();

        let out = resolve_q_matrices_path(None).unwrap();
        assert_eq!(out, cache_path);
    }
}
