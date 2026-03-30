# rctd-py CLI Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a click-based CLI (`rctd run`, `rctd validate`, `rctd info`) with structured JSON output for AI-agent consumption.

**Architecture:** Single file `src/rctd/cli.py` with lazy imports for fast `--help`. Three commands under a click group. `--json` flag on every command writes structured JSON to stdout while progress goes to stderr. The `run` command uses the `RCTD` class directly (not `run_rctd()`) to access `_pixel_mask` for result re-expansion.

**Tech Stack:** click>=8.0, existing rctd internals (RCTD, Reference, RCTDConfig, result types)

**Design doc:** `docs/plans/2026-03-17-cli-design.md`

---

### Task 1: Add click dependency and entry point

**Files:**
- Modify: `pyproject.toml:24-29` (dependencies) and add new section (entry point)

**Step 1: Modify pyproject.toml**

Add `click>=8.0` to dependencies and add the `[project.scripts]` section:

```toml
# In [project] dependencies list, add:
dependencies = [
    "torch>=2.0",
    "numpy>=1.24",
    "scipy>=1.10",
    "anndata>=0.10",
    "click>=8.0",
]

# After [project.urls] section, add:
[project.scripts]
rctd = "rctd.cli:main"
```

**Step 2: Create minimal cli.py stub**

Create `src/rctd/cli.py` with just the group and a placeholder:

```python
"""Command-line interface for rctd-py."""

from __future__ import annotations

import click


@click.group()
@click.version_option(package_name="rctd-py")
def main():
    """rctd-py: GPU-accelerated cell type deconvolution for spatial transcriptomics."""


if __name__ == "__main__":
    main()
```

**Step 3: Verify entry point works**

Run: `cd ~/git/rctd-py && uv pip install -e ".[dev]" && rctd --version`
Expected: Prints version string (e.g. `rctd-py, version 0.2.1.dev...`)

Run: `rctd --help`
Expected: Shows group help text

**Step 4: Commit**

```bash
git add pyproject.toml src/rctd/cli.py
git commit -m "feat(cli): add click entry point and minimal CLI stub"
```

---

### Task 2: Implement `rctd info` command with tests

**Files:**
- Modify: `src/rctd/cli.py`
- Create: `tests/test_cli.py`

**Step 1: Write the failing tests**

Create `tests/test_cli.py`:

```python
"""Tests for rctd CLI."""

import json

from click.testing import CliRunner

from rctd.cli import main


def test_info_human():
    """rctd info prints human-readable environment info."""
    runner = CliRunner()
    result = runner.invoke(main, ["info"])
    assert result.exit_code == 0
    assert "rctd-py" in result.output
    assert "torch" in result.output.lower() or "PyTorch" in result.output


def test_info_json():
    """rctd info --json prints valid JSON with required keys."""
    runner = CliRunner()
    result = runner.invoke(main, ["info", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert "rctd_version" in data
    assert "python_version" in data
    assert "torch_version" in data
    assert "cuda_available" in data
    assert isinstance(data["cuda_available"], bool)


def test_version():
    """rctd --version prints version string."""
    runner = CliRunner()
    result = runner.invoke(main, ["--version"])
    assert result.exit_code == 0
    assert "version" in result.output.lower()
```

**Step 2: Run tests to verify they fail**

Run: `cd ~/git/rctd-py && python -m pytest tests/test_cli.py -v`
Expected: FAIL — `info` command not found

**Step 3: Implement `rctd info`**

Add to `src/rctd/cli.py`:

```python
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


if __name__ == "__main__":
    main()
```

**Step 4: Run tests to verify they pass**

Run: `cd ~/git/rctd-py && python -m pytest tests/test_cli.py -v`
Expected: All 3 tests PASS

**Step 5: Commit**

```bash
git add src/rctd/cli.py tests/test_cli.py
git commit -m "feat(cli): add rctd info command with JSON output"
```

---

### Task 3: Implement `rctd validate` command with tests

**Files:**
- Modify: `src/rctd/cli.py`
- Modify: `tests/test_cli.py`

**Step 1: Write the failing tests**

Append to `tests/test_cli.py`:

```python
import numpy as np
import anndata
import pytest
from pathlib import Path


@pytest.fixture
def h5ad_pair(tmp_path):
    """Write synthetic spatial and reference h5ad files for CLI testing."""
    rng = np.random.default_rng(42)
    n_genes, n_types, n_ref_cells, n_pixels = 200, 5, 500, 100

    # Reference
    ref_counts = rng.poisson(5, size=(n_ref_cells, n_genes)).astype(np.float32)
    cell_types = [f"Type_{i % n_types}" for i in range(n_ref_cells)]
    ref = anndata.AnnData(
        X=ref_counts,
        obs={"cell_type": cell_types},
    )
    ref.var_names = [f"Gene_{i}" for i in range(n_genes)]
    ref.obs_names = [f"Cell_{i}" for i in range(n_ref_cells)]

    # Spatial
    sp_counts = rng.poisson(10, size=(n_pixels, n_genes)).astype(np.float32)
    sp = anndata.AnnData(X=sp_counts)
    sp.var_names = [f"Gene_{i}" for i in range(n_genes)]
    sp.obs_names = [f"Pixel_{i}" for i in range(n_pixels)]

    ref_path = tmp_path / "reference.h5ad"
    sp_path = tmp_path / "spatial.h5ad"
    ref.write_h5ad(ref_path)
    sp.write_h5ad(sp_path)
    return sp_path, ref_path


def test_validate_pass(h5ad_pair):
    """rctd validate passes with valid inputs."""
    sp_path, ref_path = h5ad_pair
    runner = CliRunner()
    result = runner.invoke(main, ["validate", str(sp_path), str(ref_path)])
    assert result.exit_code == 0
    assert "pass" in result.output.lower() or "PASS" in result.output


def test_validate_json(h5ad_pair):
    """rctd validate --json returns structured JSON."""
    sp_path, ref_path = h5ad_pair
    runner = CliRunner()
    result = runner.invoke(main, ["validate", str(sp_path), str(ref_path), "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["status"] in ("pass", "fail")
    assert "checks" in data


def test_validate_missing_column(h5ad_pair):
    """rctd validate fails when cell_type_col doesn't exist."""
    sp_path, ref_path = h5ad_pair
    runner = CliRunner()
    result = runner.invoke(main, [
        "validate", str(sp_path), str(ref_path), "--cell-type-col", "nonexistent"
    ])
    assert result.exit_code == 0  # validate reports failure, doesn't crash
    assert "fail" in result.output.lower() or "FAIL" in result.output


def test_validate_no_gene_overlap(tmp_path):
    """rctd validate fails when spatial and reference share no genes."""
    rng = np.random.default_rng(42)
    ref = anndata.AnnData(X=rng.poisson(5, size=(100, 50)).astype(np.float32))
    ref.var_names = [f"RefGene_{i}" for i in range(50)]
    ref.obs_names = [f"Cell_{i}" for i in range(100)]
    ref.obs["cell_type"] = [f"Type_{i % 3}" for i in range(100)]

    sp = anndata.AnnData(X=rng.poisson(10, size=(50, 50)).astype(np.float32))
    sp.var_names = [f"SpGene_{i}" for i in range(50)]
    sp.obs_names = [f"Pixel_{i}" for i in range(50)]

    ref.write_h5ad(tmp_path / "ref.h5ad")
    sp.write_h5ad(tmp_path / "sp.h5ad")

    runner = CliRunner()
    result = runner.invoke(main, [
        "validate", str(tmp_path / "sp.h5ad"), str(tmp_path / "ref.h5ad")
    ])
    assert result.exit_code == 0
    assert "fail" in result.output.lower() or "0 common" in result.output.lower()
```

**Step 2: Run tests to verify they fail**

Run: `cd ~/git/rctd-py && python -m pytest tests/test_cli.py::test_validate_pass -v`
Expected: FAIL — `validate` command not found

**Step 3: Implement `rctd validate`**

Add to `src/rctd/cli.py` (after the `info` command, before `if __name__`):

```python
@main.command()
@click.argument("spatial", type=click.Path(exists=True, dir_okay=False))
@click.argument("reference", type=click.Path(exists=True, dir_okay=False))
@click.option("--cell-type-col", default="cell_type", show_default=True,
              help="Column in reference .obs for cell type labels.")
@click.option("--umi-min", default=100, show_default=True,
              help="Minimum UMI count per pixel.")
@click.option("--cell-min", default=25, show_default=True,
              help="Minimum cells per cell type in reference.")
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
        n_types_est = len(ref.obs[cell_type_col].unique()) if checks.get(
            "cell_type_col_exists", {}
        ).get("pass") else 0
        n_pix = int((numi >= umi_min).sum()) if "numi" in dir() else sp.n_obs
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
            click.echo(f"  Pixels: {estimates['n_pixels']}, "
                        f"Genes: {estimates['n_genes']}, "
                        f"Types: {estimates['n_cell_types']}")
```

**Step 4: Run tests to verify they pass**

Run: `cd ~/git/rctd-py && python -m pytest tests/test_cli.py -v -k validate`
Expected: All 4 validate tests PASS

**Step 5: Commit**

```bash
git add src/rctd/cli.py tests/test_cli.py
git commit -m "feat(cli): add rctd validate command with pre-flight checks"
```

---

### Task 4: Implement `rctd run` command with tests

This is the largest task. The `run` command:
1. Reads h5ad files, constructs Reference and RCTD objects
2. Runs deconvolution via the mode-specific pipeline
3. Expands results back to full AnnData shape using `_pixel_mask`
4. Writes the annotated h5ad
5. Optionally prints JSON summary to stdout

**Files:**
- Modify: `src/rctd/cli.py`
- Modify: `tests/test_cli.py`

**Step 1: Write the failing tests**

Append to `tests/test_cli.py`:

```python
@pytest.fixture
def h5ad_pair_for_run(tmp_path):
    """Synthetic data that actually works with RCTD (enough UMI, good profiles)."""
    from tests.conftest import _make_synthetic_reference, _make_synthetic_spatial

    ref_adata, profiles, cell_type_names = _make_synthetic_reference(
        n_genes=200, n_cells=500, n_types=5, seed=42
    )
    spatial_adata, true_weights = _make_synthetic_spatial(
        profiles, n_pixels=100, n_types=5, seed=123
    )

    ref_path = tmp_path / "reference.h5ad"
    sp_path = tmp_path / "spatial.h5ad"
    ref_adata.write_h5ad(ref_path)
    spatial_adata.write_h5ad(sp_path)
    return sp_path, ref_path


@pytest.mark.slow
def test_run_doublet(h5ad_pair_for_run, tmp_path):
    """rctd run writes output h5ad with doublet results."""
    sp_path, ref_path = h5ad_pair_for_run
    out_path = tmp_path / "output.h5ad"
    runner = CliRunner()
    result = runner.invoke(main, [
        "run", str(sp_path), str(ref_path),
        "--mode", "doublet",
        "--output", str(out_path),
        "--device", "cpu",
    ])
    assert result.exit_code == 0, f"CLI failed: {result.output}"
    assert out_path.exists()

    out = anndata.read_h5ad(out_path)
    assert "rctd_weights" in out.obsm
    assert "rctd_spot_class" in out.obs.columns
    assert "rctd_first_type" in out.obs.columns
    assert "rctd_dominant_type" in out.obs.columns
    assert out.obsm["rctd_weights"].shape[0] == out.n_obs
    assert out.uns["rctd_mode"] == "doublet"


@pytest.mark.slow
def test_run_full(h5ad_pair_for_run, tmp_path):
    """rctd run --mode full writes full-mode results."""
    sp_path, ref_path = h5ad_pair_for_run
    out_path = tmp_path / "output_full.h5ad"
    runner = CliRunner()
    result = runner.invoke(main, [
        "run", str(sp_path), str(ref_path),
        "--mode", "full",
        "--output", str(out_path),
        "--device", "cpu",
    ])
    assert result.exit_code == 0, f"CLI failed: {result.output}"
    out = anndata.read_h5ad(out_path)
    assert "rctd_weights" in out.obsm
    assert "rctd_converged" in out.obs.columns
    assert out.uns["rctd_mode"] == "full"


@pytest.mark.slow
def test_run_multi(h5ad_pair_for_run, tmp_path):
    """rctd run --mode multi writes multi-mode results."""
    sp_path, ref_path = h5ad_pair_for_run
    out_path = tmp_path / "output_multi.h5ad"
    runner = CliRunner()
    result = runner.invoke(main, [
        "run", str(sp_path), str(ref_path),
        "--mode", "multi",
        "--output", str(out_path),
        "--device", "cpu",
    ])
    assert result.exit_code == 0, f"CLI failed: {result.output}"
    out = anndata.read_h5ad(out_path)
    assert "rctd_weights" in out.obsm
    assert "rctd_n_types" in out.obs.columns
    assert out.uns["rctd_mode"] == "multi"


@pytest.mark.slow
def test_run_json_output(h5ad_pair_for_run, tmp_path):
    """rctd run --json prints structured JSON summary."""
    sp_path, ref_path = h5ad_pair_for_run
    out_path = tmp_path / "output_json.h5ad"
    runner = CliRunner()
    result = runner.invoke(main, [
        "run", str(sp_path), str(ref_path),
        "--output", str(out_path),
        "--device", "cpu",
        "--json",
    ])
    assert result.exit_code == 0, f"CLI failed: {result.output}"
    data = json.loads(result.output)
    assert data["status"] == "success"
    assert data["mode"] == "doublet"
    assert "input" in data
    assert "results" in data
    assert "summary" in data


@pytest.mark.slow
def test_run_default_output_path(h5ad_pair_for_run):
    """rctd run without --output writes to <spatial_stem>_rctd.h5ad."""
    sp_path, ref_path = h5ad_pair_for_run
    runner = CliRunner()
    result = runner.invoke(main, [
        "run", str(sp_path), str(ref_path), "--device", "cpu",
    ])
    assert result.exit_code == 0, f"CLI failed: {result.output}"
    expected_out = sp_path.parent / "spatial_rctd.h5ad"
    assert expected_out.exists()


@pytest.mark.slow
def test_run_pixel_mask_expansion(h5ad_pair_for_run, tmp_path):
    """Output h5ad has same n_obs as input; filtered pixels have NaN weights."""
    sp_path, ref_path = h5ad_pair_for_run
    out_path = tmp_path / "output_mask.h5ad"
    runner = CliRunner()
    result = runner.invoke(main, [
        "run", str(sp_path), str(ref_path),
        "--output", str(out_path),
        "--device", "cpu",
    ])
    assert result.exit_code == 0, f"CLI failed: {result.output}"
    sp_in = anndata.read_h5ad(sp_path)
    out = anndata.read_h5ad(out_path)
    assert out.n_obs == sp_in.n_obs
```

**Step 2: Run tests to verify they fail**

Run: `cd ~/git/rctd-py && python -m pytest tests/test_cli.py::test_run_doublet -v`
Expected: FAIL — `run` command not found

**Step 3: Implement `rctd run`**

Add to `src/rctd/cli.py` (after `validate`, before `if __name__`).

The implementation has three parts:
- (a) The click command with all flags
- (b) A helper `_build_summary()` that creates mode-specific JSON summary
- (c) A helper `_write_results_to_adata()` that expands results and writes the h5ad

```python
def _build_summary(result, mode, cell_type_names):
    """Build mode-specific summary dict for JSON output."""
    import numpy as np
    from rctd._types import SPOT_CLASS_NAMES

    summary = {}
    if mode == "full":
        converged = result.converged
        summary["n_converged"] = int(converged.sum())
        summary["convergence_rate"] = float(converged.mean())
        dominant_idx = result.weights.argmax(axis=1)
        dominant_names = [cell_type_names[i] for i in dominant_idx]
        from collections import Counter
        summary["dominant_type_counts"] = dict(Counter(dominant_names).most_common())
    elif mode == "doublet":
        from collections import Counter
        class_names = [SPOT_CLASS_NAMES[i] for i in result.spot_class]
        summary["spot_class_counts"] = dict(Counter(class_names))
        first_names = [cell_type_names[i] for i in result.first_type]
        summary["top_singlet_types"] = dict(
            Counter(
                n for n, c in zip(first_names, result.spot_class) if c == 1
            ).most_common(10)
        )
    elif mode == "multi":
        from collections import Counter
        n_types = result.n_types
        summary["n_types_distribution"] = dict(Counter(int(n) for n in n_types))
        summary["mean_n_types"] = float(n_types.mean())
        dominant_idx = result.weights.argmax(axis=1)
        dominant_names = [cell_type_names[i] for i in dominant_idx]
        summary["dominant_type_counts"] = dict(Counter(dominant_names).most_common())
    return summary


def _write_results_to_adata(spatial_adata, result, mode, pixel_mask, config_dict,
                            cell_type_names, version):
    """Expand results to full AnnData shape and write slots."""
    import numpy as np
    import pandas as pd
    from rctd._types import SPOT_CLASS_NAMES

    adata = spatial_adata.copy()
    n_total = adata.n_obs
    n_types = len(cell_type_names)

    # Weights (all modes)
    full_weights = np.full((n_total, n_types), np.nan, dtype=np.float32)
    full_weights[pixel_mask] = result.weights
    adata.obsm["rctd_weights"] = full_weights

    # Dominant type (all modes)
    dominant = np.full(n_total, "filtered", dtype=object)
    dominant_idx = result.weights.argmax(axis=1)
    dominant[pixel_mask] = [cell_type_names[i] for i in dominant_idx]
    adata.obs["rctd_dominant_type"] = pd.Categorical(dominant)

    if mode == "full":
        converged = np.full(n_total, False)
        converged[pixel_mask] = result.converged
        adata.obs["rctd_converged"] = converged

    elif mode == "doublet":
        # Spot class
        spot_class = np.full(n_total, "filtered", dtype=object)
        spot_class[pixel_mask] = [SPOT_CLASS_NAMES[i] for i in result.spot_class]
        adata.obs["rctd_spot_class"] = pd.Categorical(spot_class)

        # First/second type
        first_type = np.full(n_total, "filtered", dtype=object)
        first_type[pixel_mask] = [cell_type_names[i] for i in result.first_type]
        adata.obs["rctd_first_type"] = pd.Categorical(first_type)

        second_type = np.full(n_total, "filtered", dtype=object)
        second_type[pixel_mask] = [cell_type_names[i] for i in result.second_type]
        adata.obs["rctd_second_type"] = pd.Categorical(second_type)

        # Doublet weights
        full_wt_doublet = np.full((n_total, 2), np.nan, dtype=np.float32)
        full_wt_doublet[pixel_mask] = result.weights_doublet
        adata.obsm["rctd_weights_doublet"] = full_wt_doublet

    elif mode == "multi":
        n_types_per_pixel = np.zeros(n_total, dtype=np.int32)
        n_types_per_pixel[pixel_mask] = result.n_types
        adata.obs["rctd_n_types"] = n_types_per_pixel

        max_multi = result.sub_weights.shape[1]
        full_sub_wt = np.full((n_total, max_multi), np.nan, dtype=np.float32)
        full_sub_wt[pixel_mask] = result.sub_weights
        adata.obsm["rctd_sub_weights"] = full_sub_wt

        full_ct_idx = np.full((n_total, max_multi), -1, dtype=np.int32)
        full_ct_idx[pixel_mask] = result.cell_type_indices
        adata.obsm["rctd_cell_type_indices"] = full_ct_idx

    # Metadata
    adata.uns["rctd_mode"] = mode
    adata.uns["rctd_config"] = config_dict
    adata.uns["rctd_version"] = version
    adata.uns["rctd_cell_type_names"] = cell_type_names

    return adata


@main.command()
@click.argument("spatial", type=click.Path(exists=True, dir_okay=False))
@click.argument("reference", type=click.Path(exists=True, dir_okay=False))
@click.option("--cell-type-col", default="cell_type", show_default=True,
              help="Column in reference .obs for cell type labels.")
@click.option("--mode", type=click.Choice(["full", "doublet", "multi"]),
              default="doublet", show_default=True, help="Deconvolution mode.")
@click.option("--output", "-o", default=None,
              help="Output h5ad path. [default: <spatial_stem>_rctd.h5ad]")
@click.option("--json", "use_json", is_flag=True,
              help="Print structured JSON summary to stdout.")
@click.option("--quiet", "-q", is_flag=True, help="Suppress progress messages.")
# RCTDConfig parameters
@click.option("--gene-cutoff", default=0.000125, show_default=True, type=float)
@click.option("--fc-cutoff", default=0.5, show_default=True, type=float)
@click.option("--gene-cutoff-reg", default=0.0002, show_default=True, type=float)
@click.option("--fc-cutoff-reg", default=0.75, show_default=True, type=float)
@click.option("--umi-min", default=100, show_default=True, type=int,
              help="Minimum UMI count per pixel.")
@click.option("--umi-max", default=20_000_000, show_default=True, type=int)
@click.option("--umi-min-sigma", default=300, show_default=True, type=int)
@click.option("--max-multi-types", default=4, show_default=True, type=int)
@click.option("--confidence-threshold", default=5.0, show_default=True, type=float)
@click.option("--doublet-threshold", default=20.0, show_default=True, type=float)
@click.option("--dtype", type=click.Choice(["float32", "float64"]),
              default="float64", show_default=True)
@click.option("--device", type=click.Choice(["auto", "cpu", "cuda"]),
              default="auto", show_default=True)
# Performance
@click.option("--batch-size", default=10000, show_default=True, type=int,
              help="GPU batch size for pixel processing.")
@click.option("--sigma-override", default=None, type=int,
              help="Skip sigma estimation, use this value directly.")
# Reference construction
@click.option("--cell-min", default=25, show_default=True, type=int,
              help="Minimum cells per type in reference.")
@click.option("--n-max-cells", default=10000, show_default=True, type=int,
              help="Max cells per type for downsampling.")
@click.option("--min-umi-ref", default=100, show_default=True, type=int,
              help="Minimum UMI for reference cells.")
def run(spatial, reference, cell_type_col, mode, output, use_json, quiet,
        gene_cutoff, fc_cutoff, gene_cutoff_reg, fc_cutoff_reg,
        umi_min, umi_max, umi_min_sigma, max_multi_types,
        confidence_threshold, doublet_threshold, dtype, device,
        batch_size, sigma_override, cell_min, n_max_cells, min_umi_ref):
    """Run RCTD deconvolution on spatial transcriptomics data."""
    import contextlib
    import io
    import time
    import traceback
    from pathlib import Path

    import anndata

    from rctd import __version__
    from rctd._doublet import run_doublet_mode
    from rctd._full import run_full_mode
    from rctd._multi import run_multi_mode
    from rctd._rctd import RCTD
    from rctd._reference import Reference
    from rctd._types import RCTDConfig

    # Default output path
    if output is None:
        sp = Path(spatial)
        output = str(sp.parent / f"{sp.stem}_rctd.h5ad")

    # Build config
    config = RCTDConfig(
        gene_cutoff=gene_cutoff,
        fc_cutoff=fc_cutoff,
        gene_cutoff_reg=gene_cutoff_reg,
        fc_cutoff_reg=fc_cutoff_reg,
        UMI_min=umi_min,
        UMI_max=umi_max,
        UMI_min_sigma=umi_min_sigma,
        MAX_MULTI_TYPES=max_multi_types,
        CONFIDENCE_THRESHOLD=confidence_threshold,
        DOUBLET_THRESHOLD=doublet_threshold,
        dtype=dtype,
        device=device,
    )
    config_dict = config._asdict()
    # Convert non-serializable types for JSON
    config_dict = {k: v for k, v in config_dict.items()}

    try:
        # Redirect stdout to stderr when --json or --quiet
        if use_json or quiet:
            redirect = contextlib.redirect_stdout(sys.stderr)
        else:
            redirect = contextlib.nullcontext()

        with redirect:
            t_start = time.time()

            # Load data
            click.echo("Loading spatial data...", err=True) if not quiet else None
            spatial_adata = anndata.read_h5ad(spatial)
            click.echo("Loading reference...", err=True) if not quiet else None
            ref_adata = anndata.read_h5ad(reference)
            ref_obj = Reference(ref_adata, cell_type_col=cell_type_col,
                                cell_min=cell_min, n_max_cells=n_max_cells,
                                min_UMI=min_umi_ref)

            # Run RCTD
            rctd_obj = RCTD(spatial_adata, ref_obj, config)
            rctd_obj.fit_platform_effects(sigma_override=sigma_override)

            cell_type_names = rctd_obj.reference.cell_type_names
            n_pixels_total = spatial_adata.n_obs
            n_pixels_filtered = int(rctd_obj._pixel_mask.sum())
            n_genes_common = len(rctd_obj.common_genes)

            print(f"Running in {mode} mode...")

            kwargs = {
                "spatial_counts": rctd_obj.counts,
                "spatial_numi": rctd_obj.nUMI,
                "norm_profiles": rctd_obj.norm_profiles,
                "cell_type_names": cell_type_names,
                "q_mat": rctd_obj.q_mat,
                "sq_mat": rctd_obj.sq_mat,
                "x_vals": rctd_obj.x_vals,
                "batch_size": batch_size,
                "device": config.device,
            }

            if mode == "full":
                result = run_full_mode(**kwargs)
            elif mode == "doublet":
                result = run_doublet_mode(**kwargs, config=config)
            elif mode == "multi":
                result = run_multi_mode(**kwargs, config=config)

            elapsed = time.time() - t_start

            # Write output h5ad
            out_adata = _write_results_to_adata(
                spatial_adata, result, mode, rctd_obj._pixel_mask,
                config_dict, cell_type_names, __version__,
            )
            out_adata.write_h5ad(output)
            print(f"Results written to {output}")

        # JSON output to real stdout
        if use_json:
            import torch

            device_used = "cpu"
            if torch.cuda.is_available() and config.device != "cpu":
                device_used = torch.cuda.get_device_name(0)

            summary = _build_summary(result, mode, cell_type_names)
            json_output = {
                "status": "success",
                "version": __version__,
                "mode": mode,
                "output_path": str(Path(output).resolve()),
                "input": {
                    "spatial_path": str(Path(spatial).resolve()),
                    "reference_path": str(Path(reference).resolve()),
                    "n_pixels_total": n_pixels_total,
                    "n_pixels_after_filter": n_pixels_filtered,
                    "n_genes_common": n_genes_common,
                    "n_cell_types": len(cell_type_names),
                    "cell_type_names": cell_type_names,
                },
                "config": config_dict,
                "results": {
                    "sigma": rctd_obj.sigma,
                    "elapsed_seconds": round(elapsed, 1),
                    "device_used": device_used,
                },
                "summary": summary,
            }
            click.echo(json.dumps(json_output, indent=2, default=str))

    except Exception as e:
        if use_json:
            error_output = {
                "status": "error",
                "error": str(e),
                "traceback": traceback.format_exc(),
            }
            click.echo(json.dumps(error_output, indent=2))
            sys.exit(1)
        else:
            raise
```

**Step 4: Run tests to verify they pass**

Run: `cd ~/git/rctd-py && python -m pytest tests/test_cli.py -v -m slow -x`
Expected: All `test_run_*` tests PASS (may take 30-60s on CPU with synthetic data)

Also run all CLI tests together:
Run: `cd ~/git/rctd-py && python -m pytest tests/test_cli.py -v`
Expected: Non-slow tests pass immediately; slow tests skipped (unless `-m slow` is added)

**Step 5: Commit**

```bash
git add src/rctd/cli.py tests/test_cli.py
git commit -m "feat(cli): add rctd run command with h5ad output and JSON summary"
```

---

### Task 5: Lint and run full test suite

**Step 1: Run ruff**

Run: `cd ~/git/rctd-py && ruff check src/rctd/cli.py tests/test_cli.py`
Expected: No errors (fix any that appear)

Run: `cd ~/git/rctd-py && ruff format src/rctd/cli.py tests/test_cli.py`

**Step 2: Run full existing test suite to ensure no regressions**

Run: `cd ~/git/rctd-py && python -m pytest tests/ -v --ignore=tests/test_concordance.py --ignore=tests/test_performance.py`
Expected: All tests PASS

**Step 3: Commit any lint fixes**

```bash
git add -u
git commit -m "style: lint and format CLI code"
```

---

### Task 6: Add CLI section to README.md

**Files:**
- Modify: `README.md` — add CLI section after Quick Start

**Step 1: Add CLI documentation**

Insert after the "Quick Start" section (after line 43, before "Installation"):

```markdown
## Command Line

After installation, the `rctd` command is available:

```bash
# Check environment and GPU availability
rctd info
rctd info --json

# Validate inputs before running (fast, no GPU needed)
rctd validate spatial.h5ad reference.h5ad

# Run deconvolution
rctd run spatial.h5ad reference.h5ad --mode doublet --output results.h5ad

# All modes
rctd run spatial.h5ad reference.h5ad --mode full
rctd run spatial.h5ad reference.h5ad --mode multi

# JSON output for AI agents / pipelines
rctd run spatial.h5ad reference.h5ad --json --quiet

# GPU control
rctd run spatial.h5ad reference.h5ad --device cuda --batch-size 5000
```

Results are written into a copy of the spatial h5ad with RCTD annotations in `.obsm["rctd_weights"]`, `.obs["rctd_spot_class"]`, etc. Use `rctd run --help` for all options.
```

**Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add CLI usage section to README"
```

---

### Summary of commits

1. `feat(cli): add click entry point and minimal CLI stub`
2. `feat(cli): add rctd info command with JSON output`
3. `feat(cli): add rctd validate command with pre-flight checks`
4. `feat(cli): add rctd run command with h5ad output and JSON summary`
5. `style: lint and format CLI code`
6. `docs: add CLI usage section to README`
