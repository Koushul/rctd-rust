mod bench;
mod device_sel;
mod ref_adata;

use std::collections::HashMap;
use std::path::PathBuf;

use anndata::{AnnData, AnnDataOp, Backend};
use anndata_hdf5::H5;
use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use device_sel::ComputeDevice;
use ndarray::{Array1, Array2};
use rctd_core::io_npz::load_q_matrices_npz;
use rctd_core::likelihood_tables::compute_spline_coefficients;
use rctd_core::{run_deconvolution, sync_device, DeconvMode, PreparedData, RctdConfig};

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
        #[arg(long, help = "Spatial AnnData (.h5ad), X = counts (N×G), var = genes")]
        spatial: PathBuf,
        #[arg(
            long,
            help = "Reference .h5ad: single-cell (obs × genes + --cell-type-col) or K×G with --ref-rows-are-types"
        )]
        reference: PathBuf,
        #[arg(
            long,
            default_value = "cell_type",
            help = "obs column with cell type labels (single-cell reference)"
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
        #[arg(long, help = "q_matrices.npz (from rctd cache / release asset)")]
        q_matrices: PathBuf,
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
            let spatial_ad = open_h5ad(&spatial)?;
            let ref_ad = open_h5ad(&reference)?;
            let spatial_genes = spatial_ad.var_names().into_vec();
            let ref_genes = ref_ad.var_names().into_vec();
            let counts = ref_adata::x_to_dense_f64(&spatial_ad)?;
            let (profiles_kg, cell_type_names) = if ref_rows_are_types {
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
            if counts.ncols() != spatial_genes.len() || counts.nrows() != spatial_ad.n_obs() {
                bail!(
                    "spatial X shape {:?} does not match n_obs × n_vars",
                    counts.dim()
                );
            }
            let (counts, profiles_gk_raw) =
                align_genes_spatial_ref(&spatial_genes, &ref_genes, &counts, &profiles_kg)?;
            let norm_profiles = ref_adata::normalize_columns(&profiles_gk_raw);
            if cell_type_names.len() != norm_profiles.ncols() {
                bail!("cell type count does not match norm_profiles columns");
            }
            let numi: Array1<f64> = counts.sum_axis(ndarray::Axis(1));

            let (q_map, x_vals) = load_q_matrices_npz(&q_matrices)?;
            let sigma_key = sigma.to_string();
            let q_mat = q_map
                .get(&sigma_key)
                .with_context(|| format!("sigma {} not in q_matrices.npz", sigma))?
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
