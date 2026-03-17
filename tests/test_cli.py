"""Tests for rctd CLI."""

import json

import anndata
import numpy as np
import pytest
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


# ── validate tests ──────────────────────────────────────────────────


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
    result = runner.invoke(
        main, ["validate", str(sp_path), str(ref_path), "--cell-type-col", "nonexistent"]
    )
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
    result = runner.invoke(
        main, ["validate", str(tmp_path / "sp.h5ad"), str(tmp_path / "ref.h5ad")]
    )
    assert result.exit_code == 0
    assert "fail" in result.output.lower() or "0 common" in result.output.lower()
