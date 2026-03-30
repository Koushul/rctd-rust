# rctd-py CLI Design

Date: 2026-03-17
Status: Approved

## Goal

Add a simple, purpose-built CLI to rctd-py so AI agents (and humans) can invoke deconvolution from the command line with structured JSON output. Inspired by CLI-Anything's AI-readiness patterns (--json flag, self-documenting --help, click-based) but hand-written since rctd-py is already a clean Python library.

## Decisions

- **CLI framework**: click (lightweight, no transitive deps, standard)
- **Output format**: h5ad (results written into spatial AnnData copy)
- **Flag style**: Flat flags (all RCTDConfig params as top-level --flags)
- **No REPL, no session state, no undo/redo** — batch-processing tool
- **Integrated into existing package** (`src/rctd/cli.py`), not a separate package

## Command Structure

```
rctd --version
rctd info [--json]
rctd validate SPATIAL REFERENCE [--cell-type-col cell_type] [--json]
rctd run SPATIAL REFERENCE [--mode doublet] [--output spatial_rctd.h5ad] [--json] [--quiet] [... config flags]
```

### `rctd info`

Reports: rctd-py version, Python/PyTorch/CUDA info, device names and VRAM.

### `rctd validate`

Fast pre-flight checks (no GPU needed):
1. Both h5ad files readable
2. Reference .obs contains cell_type_col
3. >= 2 cell types after cell_min filtering
4. Gene overlap >= 50 genes
5. Spatial pixel count after UMI filtering > 0
6. Estimated memory/VRAM

### `rctd run`

Full deconvolution pipeline. Positional args for the two h5ad files. All RCTDConfig params exposed as optional flags with defaults.

**Key flags:**
- `--mode [full|doublet|multi]` (default: doublet)
- `--output, -o PATH` (default: `<spatial_stem>_rctd.h5ad`)
- `--json` — structured JSON summary to stdout
- `--quiet, -q` — suppress progress messages
- `--device [auto|cpu|cuda]` (default: auto)
- `--dtype [float32|float64]` (default: float64)
- `--batch-size INT` (default: 10000)
- `--sigma-override INT` — skip sigma estimation
- `--cell-type-col TEXT` (default: cell_type)
- `--umi-min`, `--umi-max`, `--gene-cutoff`, `--fc-cutoff`, etc.

## Output h5ad Structure

Results written into a copy of the spatial h5ad. RCTD filters pixels by UMI, so result arrays are expanded back to full AnnData shape (NaN/"filtered" for excluded pixels).

### All modes

| Slot | Key | Description |
|------|-----|-------------|
| `.obsm` | `rctd_weights` | (N, K) cell type weights, NaN for filtered |
| `.obs` | `rctd_dominant_type` | Highest-weight cell type, "filtered" for excluded |
| `.uns` | `rctd_mode`, `rctd_config`, `rctd_version`, `rctd_cell_type_names` | Metadata |

### Doublet mode (additional)

| Slot | Key | Description |
|------|-----|-------------|
| `.obs` | `rctd_spot_class` | "singlet", "doublet_certain", "doublet_uncertain", "reject", "filtered" |
| `.obs` | `rctd_first_type`, `rctd_second_type` | Primary/secondary cell type |
| `.obsm` | `rctd_weights_doublet` | (N, 2) top-2 weights |

### Full mode (additional)

| Slot | Key | Description |
|------|-----|-------------|
| `.obs` | `rctd_converged` | IRWLS convergence flag |

### Multi mode (additional)

| Slot | Key | Description |
|------|-----|-------------|
| `.obs` | `rctd_n_types` | Number of selected types per pixel |
| `.obsm` | `rctd_sub_weights` | (N, 4) selected type weights |
| `.obsm` | `rctd_cell_type_indices` | (N, 4) type indices, -1 padded |

## JSON Output Schema

### `rctd run --json`

```json
{
  "status": "success",
  "version": "0.2.1",
  "mode": "doublet",
  "output_path": "/path/to/spatial_rctd.h5ad",
  "input": {
    "n_pixels_total": 58191,
    "n_pixels_after_filter": 57842,
    "n_genes_common": 372,
    "n_cell_types": 45,
    "cell_type_names": ["Bcell", "CD4", "..."]
  },
  "config": { "UMI_min": 100, "device": "cuda", "dtype": "float64" },
  "results": {
    "sigma": 84,
    "elapsed_seconds": 396.2,
    "device_used": "NVIDIA L40S"
  },
  "summary": { }
}
```

Mode-specific `summary`:
- **full**: `n_converged`, `convergence_rate`, `dominant_type_counts`
- **doublet**: `spot_class_counts`, `top_singlet_types`, `top_doublet_pairs`
- **multi**: `n_types_distribution`, `mean_n_types`, `dominant_type_counts`

On error: `{"status": "error", "error": "message", "traceback": "..."}` + exit code 1.

### `rctd validate --json`

```json
{
  "status": "pass",
  "checks": { "spatial_readable": {"pass": true, "detail": "..."}, "..." : {} },
  "warnings": [],
  "estimates": { "n_pixels": 57842, "estimated_vram_mb": 2600 }
}
```

### `rctd info --json`

```json
{
  "rctd_version": "0.2.1",
  "python_version": "3.12.3",
  "torch_version": "2.5.1",
  "cuda_available": true,
  "cuda_devices": [{"index": 0, "name": "NVIDIA L40S", "vram_mb": 49152}]
}
```

## Implementation Details

- **Single file**: `src/rctd/cli.py` (~250 lines)
- **Lazy imports**: torch/anndata imported inside command functions for fast --help
- **stdout/stderr separation**: Progress to stderr during --json mode, JSON to real stdout
- **Pixel mask**: CLI uses `RCTD` class directly (not `run_rctd()`) to access `_pixel_mask`
- **New dependency**: `click>=8.0`
- **Entry point**: `[project.scripts] rctd = "rctd.cli:main"`

## Files Changed

1. `src/rctd/cli.py` — new (all CLI logic)
2. `pyproject.toml` — add click dep + entry point
3. `tests/test_cli.py` — new (CLI tests via click.testing.CliRunner)
4. `README.md` — add CLI usage section
