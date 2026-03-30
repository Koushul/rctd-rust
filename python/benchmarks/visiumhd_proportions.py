#!/usr/bin/env python3
"""Quick check: cell type proportions from VisiumHD full-mode RCTD."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def load_reference(ref_dir: Path):
    import anndata
    counts = scipy.io.mmread(ref_dir / "ref_counts.mtx").T.tocsc()
    ref_meta = pd.read_csv(ref_dir / "ref_meta.csv")
    genes = pd.read_csv(ref_dir / "ref_genes.txt", header=None)[0].tolist()
    ref_adata = anndata.AnnData(
        X=counts,
        obs=ref_meta.set_index(ref_meta.columns[0]),
        var=pd.DataFrame(index=genes),
    )
    from rctd import Reference
    return Reference(ref_adata, cell_type_col="cell_type")


def main():
    import scanpy as sc
    from rctd import RCTDConfig, run_rctd

    ref_dir = Path("data/mouse_brain")
    spatial_path = "/scratch/pgueguen/visiumhd_mouse_brain/square_008um/filtered_feature_bc_matrix.h5"

    print("Loading reference...")
    reference = load_reference(ref_dir)

    print("Loading spatial...")
    spatial = sc.read_10x_h5(spatial_path)
    spatial.var_names_make_unique()
    print(f"  {spatial.n_obs} spots")

    print("Running RCTD full mode...")
    config = RCTDConfig(UMI_min=20)
    result = run_rctd(spatial, reference, mode="full", config=config)

    # Dominant type per spot
    dominant_idx = np.argmax(result.weights, axis=1)
    names = reference.cell_type_names

    print("\n=== Cell type proportions (by dominant type) ===")
    counts = pd.Series(dominant_idx).value_counts().sort_index()
    total = counts.sum()
    for idx, count in counts.items():
        pct = 100 * count / total
        print(f"  {names[idx]:30s}  {count:>7d}  ({pct:5.1f}%)")

    # Mean weight per type
    mean_w = result.weights.mean(axis=0)
    print("\n=== Mean weight per cell type ===")
    order = np.argsort(-mean_w)
    for idx in order:
        if mean_w[idx] > 0.001:
            print(f"  {names[idx]:30s}  {mean_w[idx]:.4f}")


if __name__ == "__main__":
    main()
