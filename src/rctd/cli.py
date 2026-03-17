"""Command-line interface for rctd-py."""

from __future__ import annotations

import json
import sys

import click


@click.group()
@click.version_option(package_name="rctd-py")
def main():
    """rctd-py: GPU-accelerated cell type deconvolution for spatial transcriptomics."""


@main.command()
@click.option("--json", "use_json", is_flag=True, help="Output as JSON.")
def info(use_json):
    """Show version, device, and environment information."""
    import torch

    from rctd import __version__

    data = {
        "rctd_version": __version__,
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "cuda_devices": [],
    }
    try:
        import anndata

        data["anndata_version"] = anndata.__version__
    except ImportError:
        data["anndata_version"] = "not installed"
    try:
        import numpy

        data["numpy_version"] = numpy.__version__
    except ImportError:
        data["numpy_version"] = "not installed"
    try:
        import scipy

        data["scipy_version"] = scipy.__version__
    except ImportError:
        data["scipy_version"] = "not installed"

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            data["cuda_devices"].append({
                "index": i,
                "name": props.name,
                "vram_mb": props.total_mem // (1024 * 1024),
            })

    if use_json:
        click.echo(json.dumps(data, indent=2))
    else:
        click.echo(f"rctd-py {data['rctd_version']}")
        click.echo(f"Python  {data['python_version']}")
        click.echo(f"PyTorch {data['torch_version']}")
        if data["cuda_available"]:
            for dev in data["cuda_devices"]:
                click.echo(f"CUDA    {dev['name']} ({dev['vram_mb']} MB)")
        else:
            click.echo("CUDA    not available")
        click.echo(f"anndata {data['anndata_version']}")
        click.echo(f"numpy   {data['numpy_version']}")
        click.echo(f"scipy   {data['scipy_version']}")


@main.command()
@click.argument("spatial", type=click.Path(exists=True, dir_okay=False))
@click.argument("reference", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--cell-type-col",
    default="cell_type",
    show_default=True,
    help="Column in reference .obs for cell type labels.",
)
@click.option("--umi-min", default=100, show_default=True, help="Minimum UMI count per pixel.")
@click.option(
    "--cell-min", default=25, show_default=True, help="Minimum cells per cell type in reference."
)
@click.option("--json", "use_json", is_flag=True, help="Output as JSON.")
def validate(spatial, reference, cell_type_col, umi_min, cell_min, use_json):
    """Validate inputs before running RCTD (fast, no GPU needed)."""
    import anndata
    import numpy as np
    from scipy import sparse

    checks = {}
    warnings = []

    # 1. Read spatial
    try:
        sp = anndata.read_h5ad(spatial)
        checks["spatial_readable"] = {
            "pass": True,
            "detail": f"{sp.n_obs} pixels, {sp.n_vars} genes",
        }
    except Exception as e:
        checks["spatial_readable"] = {"pass": False, "detail": str(e)}
        sp = None

    # 2. Read reference
    try:
        ref = anndata.read_h5ad(reference)
        checks["reference_readable"] = {
            "pass": True,
            "detail": f"{ref.n_obs} cells, {ref.n_vars} genes",
        }
    except Exception as e:
        checks["reference_readable"] = {"pass": False, "detail": str(e)}
        ref = None

    # 3. Check cell_type_col
    if ref is not None:
        if cell_type_col in ref.obs.columns:
            types = ref.obs[cell_type_col].unique()
            checks["cell_type_col_exists"] = {
                "pass": True,
                "detail": f"{cell_type_col}, {len(types)} types",
            }
        else:
            available = ", ".join(ref.obs.columns.tolist()[:10])
            checks["cell_type_col_exists"] = {
                "pass": False,
                "detail": f"'{cell_type_col}' not found. Available: [{available}]",
            }

    # 4. Check cell type counts after cell_min filter
    if ref is not None and checks.get("cell_type_col_exists", {}).get("pass"):
        from collections import Counter

        type_counts = Counter(ref.obs[cell_type_col].values)
        valid_types = [ct for ct, n in type_counts.items() if n >= cell_min]
        low_types = [ct for ct, n in type_counts.items() if n < cell_min]
        if len(valid_types) >= 2:
            checks["min_cell_types"] = {
                "pass": True,
                "detail": f"{len(valid_types)} types >= {cell_min} cells",
            }
        else:
            checks["min_cell_types"] = {
                "pass": False,
                "detail": f"Only {len(valid_types)} types have >= {cell_min} cells",
            }
        if low_types:
            warnings.append(
                f"{len(low_types)} cell types have < {cell_min} cells: "
                f"{low_types[:5]}{'...' if len(low_types) > 5 else ''}"
            )

    # 5. Gene overlap
    if sp is not None and ref is not None:
        common = set(sp.var_names) & set(ref.var_names)
        n_common = len(common)
        pct = (n_common / sp.n_vars * 100) if sp.n_vars > 0 else 0
        if n_common >= 50:
            checks["gene_overlap"] = {
                "pass": True,
                "detail": f"{n_common} common genes ({pct:.1f}% of spatial)",
            }
        else:
            checks["gene_overlap"] = {
                "pass": False,
                "detail": f"{n_common} common genes ({pct:.1f}% of spatial) — need >= 50",
            }

    # 6. Pixel count after UMI filter
    numi = None
    if sp is not None:
        X = sp.X
        if sparse.issparse(X):
            numi = np.array(X.sum(axis=1)).flatten()
        else:
            numi = np.array(X.sum(axis=1)).flatten()
        n_pass = int((numi >= umi_min).sum())
        if n_pass > 0:
            checks["pixel_count_after_filter"] = {
                "pass": True,
                "detail": f"{n_pass}/{sp.n_obs} pass UMI >= {umi_min}",
            }
        else:
            checks["pixel_count_after_filter"] = {
                "pass": False,
                "detail": f"0/{sp.n_obs} pass UMI >= {umi_min}. Min UMI: {numi.min():.0f}",
            }
        if n_pass < sp.n_obs * 0.5:
            warnings.append(
                f"Over half of pixels filtered: {n_pass}/{sp.n_obs} pass UMI >= {umi_min}"
            )

    # Summary
    all_pass = all(c["pass"] for c in checks.values())
    status = "pass" if all_pass else "fail"

    # Estimates
    estimates = {}
    if sp is not None and ref is not None and checks.get("gene_overlap", {}).get("pass"):
        n_common = len(set(sp.var_names) & set(ref.var_names))
        n_types_est = (
            len(ref.obs[cell_type_col].unique())
            if checks.get("cell_type_col_exists", {}).get("pass")
            else 0
        )
        n_pix = int((numi >= umi_min).sum()) if numi is not None else sp.n_obs
        estimates = {
            "n_pixels": n_pix,
            "n_genes": n_common,
            "n_cell_types": n_types_est,
        }

    output = {"status": status, "checks": checks, "warnings": warnings, "estimates": estimates}

    if use_json:
        click.echo(json.dumps(output, indent=2))
    else:
        label = "PASS" if all_pass else "FAIL"
        click.echo(f"Validation: {label}")
        for name, check in checks.items():
            icon = "+" if check["pass"] else "x"
            click.echo(f"  [{icon}] {name}: {check['detail']}")
        for w in warnings:
            click.echo(f"  [!] {w}")
        if estimates:
            click.echo(
                f"  Pixels: {estimates['n_pixels']}, "
                f"Genes: {estimates['n_genes']}, "
                f"Types: {estimates['n_cell_types']}"
            )


if __name__ == "__main__":
    main()
