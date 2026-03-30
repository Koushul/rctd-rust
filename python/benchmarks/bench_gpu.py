"""GPU benchmark for rctd-py IRWLS solver.

Generates deterministic synthetic data and benchmarks the full-mode pipeline.
Reports elapsed time, peak VRAM, and a weights hash for correctness validation.

Usage:
    uv run python benchmarks/bench_gpu.py [--n-pixels 50000] [--n-genes 1000] [--n-types 12] [--warmup 1] [--runs 3]
"""

import argparse
import hashlib
import time

import numpy as np
import torch

from rctd._full import run_full_mode
from rctd._likelihood import build_x_vals, compute_q_matrix, compute_spline_coefficients


def generate_data(n_pixels: int, n_genes: int, n_types: int, seed: int = 2025):
    """Generate deterministic synthetic RCTD input data."""
    rng = np.random.default_rng(seed)

    # Reference profiles (G x K), normalized per cell type
    profiles = rng.exponential(0.01, size=(n_genes, n_types)).astype(np.float64)
    profiles = profiles / profiles.sum(axis=0, keepdims=True)

    # Spatial data
    nUMIs = rng.integers(200, 5000, size=n_pixels).astype(np.float64)

    # Generate counts from known mixtures
    counts = np.zeros((n_pixels, n_genes), dtype=np.float64)
    for i in range(n_pixels):
        true_w = rng.dirichlet(np.ones(n_types))
        lam = (profiles @ true_w) * nUMIs[i]
        counts[i] = rng.poisson(np.clip(lam, 0, 1e6))

    return profiles, counts, nUMIs


def weights_hash(weights: np.ndarray) -> str:
    """Deterministic hash of weight matrix for correctness checking."""
    w_rounded = np.round(weights, 8)
    return hashlib.md5(w_rounded.tobytes()).hexdigest()[:16]


def run_benchmark(profiles, counts, nUMIs, q_mat, sq_mat, x_vals, device, batch_size):
    """Run a single benchmark iteration."""
    torch.cuda.synchronize() if device == "cuda" else None
    torch.cuda.reset_peak_memory_stats() if device == "cuda" else None

    t0 = time.perf_counter()
    result = run_full_mode(
        spatial_counts=counts,
        spatial_numi=nUMIs,
        norm_profiles=profiles,
        cell_type_names=[f"type_{i}" for i in range(profiles.shape[1])],
        q_mat=q_mat,
        sq_mat=sq_mat,
        x_vals=x_vals,
        batch_size=batch_size,
        device=device,
    )
    torch.cuda.synchronize() if device == "cuda" else None
    elapsed = time.perf_counter() - t0

    peak_mb = 0
    if device == "cuda":
        peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    return result, elapsed, peak_mb


def main():
    parser = argparse.ArgumentParser(description="GPU benchmark for rctd-py")
    parser.add_argument("--n-pixels", type=int, default=50000)
    parser.add_argument("--n-genes", type=int, default=1000)
    parser.add_argument("--n-types", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=10000)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"=== rctd-py GPU Benchmark ===")
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"Config: {args.n_pixels} pixels, {args.n_genes} genes, {args.n_types} types")
    print(f"Batch size: {args.batch_size}")
    print()

    # Generate data
    print("Generating synthetic data...")
    profiles, counts, nUMIs = generate_data(args.n_pixels, args.n_genes, args.n_types, args.seed)

    # Build likelihood tables
    print("Building likelihood tables...")
    x_vals = build_x_vals()
    sigma = 100  # standard sigma
    q_mat = compute_q_matrix(sigma / 100.0, x_vals)
    sq_mat = compute_spline_coefficients(q_mat, x_vals)

    # Warmup runs (torch.compile, CUDA caches, etc.)
    if args.warmup > 0:
        print(f"Warmup ({args.warmup} runs)...")
        for i in range(args.warmup):
            _, t, _ = run_benchmark(profiles, counts, nUMIs, q_mat, sq_mat, x_vals, device, args.batch_size)
            print(f"  warmup {i+1}: {t:.2f}s")

    # Timed runs
    print(f"\nBenchmark ({args.runs} runs)...")
    times = []
    peak_vrams = []
    w_hash = None

    for i in range(args.runs):
        result, elapsed, peak_mb = run_benchmark(profiles, counts, nUMIs, q_mat, sq_mat, x_vals, device, args.batch_size)
        times.append(elapsed)
        peak_vrams.append(peak_mb)
        h = weights_hash(result.weights)
        if w_hash is None:
            w_hash = h
        print(f"  run {i+1}: {elapsed:.3f}s  peak_vram={peak_mb:.0f}MB  hash={h}  consistent={'yes' if h == w_hash else 'NO'}")

    print(f"\n=== Results ===")
    print(f"elapsed_s (median): {np.median(times):.3f}")
    print(f"elapsed_s (min):    {min(times):.3f}")
    print(f"elapsed_s (mean):   {np.mean(times):.3f}")
    if device == "cuda":
        print(f"peak_vram_mb:       {max(peak_vrams):.0f}")
    print(f"weights_hash:       {w_hash}")
    print(f"converged:          {result.converged.mean()*100:.1f}%")


if __name__ == "__main__":
    main()
