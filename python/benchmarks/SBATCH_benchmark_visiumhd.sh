#!/bin/bash
#SBATCH --job-name=bench_visiumhd
#SBATCH --output=/home/pgueguen/bench_visiumhd_%j.log
#SBATCH --error=/home/pgueguen/bench_visiumhd_%j.log
#SBATCH --time=04:00:00
#SBATCH --mem-per-cpu=16G
#SBATCH --cpus-per-task=8
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --nodelist=fgcz-c-056

echo "=== rctd-py Benchmark: VisiumHD Mouse Brain 8µm (full mode) ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv
echo ""

cd /home/pgueguen/git/rctd-py
export PATH="/home/pgueguen/git/rctd-py/.venv/bin:$PATH"
export PYTHONUNBUFFERED=1

echo "Branch: $(git branch --show-current)"
echo "Commit: $(git log --oneline -1)"
echo ""

# --- Stage VisiumHD data to /scratch for fast I/O ---
SCRATCH_DIR="/scratch/pgueguen/visiumhd_mouse_brain"
mkdir -p "$SCRATCH_DIR"

if [ ! -f "$SCRATCH_DIR/square_008um/filtered_feature_bc_matrix.h5" ]; then
    echo "Downloading VisiumHD binned outputs..."
    curl -L -o "$SCRATCH_DIR/binned_outputs.tar.gz" \
        https://cf.10xgenomics.com/samples/spatial-exp/4.0.1/Visium_HD_Mouse_Brain/Visium_HD_Mouse_Brain_binned_outputs.tar.gz

    echo "Extracting 8µm bin data..."
    tar -xzf "$SCRATCH_DIR/binned_outputs.tar.gz" \
        -C "$SCRATCH_DIR" \
        --strip-components=1 \
        "binned_outputs/square_008um"

    rm -f "$SCRATCH_DIR/binned_outputs.tar.gz"
    echo "Data staged to $SCRATCH_DIR/square_008um/"
    ls -lh "$SCRATCH_DIR/square_008um/"
else
    echo "Data already staged at $SCRATCH_DIR/square_008um/"
fi
echo ""

# --- Run benchmark ---
python benchmarks/bench_visiumhd.py \
    --spatial "$SCRATCH_DIR/square_008um/filtered_feature_bc_matrix.h5" \
    --ref-dir data/mouse_brain \
    --out-dir data/visiumhd_mouse_brain \
    2>&1

echo ""
echo "Completed at $(date)"
