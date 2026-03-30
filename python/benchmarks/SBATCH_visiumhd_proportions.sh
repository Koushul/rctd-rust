#!/bin/bash
#SBATCH --job-name=vhd_props
#SBATCH --output=/home/pgueguen/vhd_proportions_%j.log
#SBATCH --error=/home/pgueguen/vhd_proportions_%j.log
#SBATCH --time=00:30:00
#SBATCH --mem-per-cpu=16G
#SBATCH --cpus-per-task=8
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --nodelist=fgcz-c-056

cd /home/pgueguen/git/rctd-py
export PATH="/home/pgueguen/git/rctd-py/.venv/bin:$PATH"
export PYTHONUNBUFFERED=1

python benchmarks/visiumhd_proportions.py 2>&1
