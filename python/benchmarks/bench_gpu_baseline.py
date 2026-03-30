"""GPU baseline benchmark - uses original solve_irwls_batch (with S_batch materialization).

This directly calls solve_irwls_batch to compare against solve_irwls_batch_shared.
"""

import argparse
import hashlib
import time

import numpy as np
import torch

from rctd._irwls import solve_irwls_batch, solve_irwls_batch_shared
from rctd._likelihood import build_x_vals, compute_q_matrix, compute_spline_coefficients


def generate_data(n_pixels, n_genes, n_types, seed=2025):
    rng = np.random.default_rng(seed)
    profiles = rng.exponential(0.01, size=(n_genes, n_types)).astype(np.float64)
    profiles = profiles / profiles.sum(axis=0, keepdims=True)
    nUMIs = rng.integers(200, 5000, size=n_pixels).astype(np.float64)
    counts = np.zeros((n_pixels, n_genes), dtype=np.float64)
    for i in range(n_pixels):
        true_w = rng.dirichlet(np.ones(n_types))
        lam = (profiles @ true_w) * nUMIs[i]
        counts[i] = rng.poisson(np.clip(lam, 0, 1e6))
    return profiles, counts, nUMIs


def weights_hash(w):
    return hashlib.md5(np.round(w, 8).tobytes()).hexdigest()[:16]


def bench_original(P_gpu, Y_gpu, nUMI_gpu, Q_gpu, SQ_gpu, X_gpu, batch_size, device):
    """Original path: materialize S_batch = nUMI * P then call solve_irwls_batch."""
    N = Y_gpu.shape[0]
    all_w = []
    torch.cuda.synchronize() if device == "cuda" else None
    torch.cuda.reset_peak_memory_stats() if device == "cuda" else None
    t0 = time.perf_counter()

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        Y_b = Y_gpu[start:end]
        nUMI_b = nUMI_gpu[start:end]
        S_batch = nUMI_b[:, None, None] * P_gpu[None, :, :]  # (bs, G, K) materialized
        w, _ = solve_irwls_batch(S_batch, Y_b, nUMI_b, Q_gpu, SQ_gpu, X_gpu, constrain=False)
        all_w.append(w.cpu())

    torch.cuda.synchronize() if device == "cuda" else None
    elapsed = time.perf_counter() - t0
    peak_mb = torch.cuda.max_memory_allocated() / (1024**2) if device == "cuda" else 0
    return torch.cat(all_w).numpy(), elapsed, peak_mb


def bench_shared(P_gpu, Y_gpu, nUMI_gpu, Q_gpu, SQ_gpu, X_gpu, batch_size, device):
    """Optimized path: pass P directly, avoid S_batch materialization."""
    N = Y_gpu.shape[0]
    all_w = []
    torch.cuda.synchronize() if device == "cuda" else None
    torch.cuda.reset_peak_memory_stats() if device == "cuda" else None
    t0 = time.perf_counter()

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        Y_b = Y_gpu[start:end]
        nUMI_b = nUMI_gpu[start:end]
        w, _ = solve_irwls_batch_shared(P_gpu, Y_b, nUMI_b, Q_gpu, SQ_gpu, X_gpu, constrain=False)
        all_w.append(w.cpu())

    torch.cuda.synchronize() if device == "cuda" else None
    elapsed = time.perf_counter() - t0
    peak_mb = torch.cuda.max_memory_allocated() / (1024**2) if device == "cuda" else 0
    return torch.cat(all_w).numpy(), elapsed, peak_mb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-pixels", type=int, default=50000)
    parser.add_argument("--n-genes", type=int, default=1000)
    parser.add_argument("--n-types", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=10000)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== Baseline vs Shared-Profile Benchmark ===")
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Config: {args.n_pixels}px x {args.n_genes}g x {args.n_types}t, bs={args.batch_size}")
    print()

    profiles, counts, nUMIs = generate_data(args.n_pixels, args.n_genes, args.n_types)
    x_vals = build_x_vals()
    q_mat = compute_q_matrix(1.0, x_vals)
    sq_mat = compute_spline_coefficients(q_mat, x_vals)

    P_gpu = torch.tensor(profiles, device=device)
    Y_gpu = torch.tensor(counts, device=device)
    nUMI_gpu = torch.tensor(nUMIs, device=device)
    Q_gpu = torch.tensor(q_mat, device=device)
    SQ_gpu = torch.tensor(sq_mat, device=device)
    X_gpu = torch.tensor(x_vals, device=device)

    for label, fn in [("ORIGINAL (S_batch)", bench_original), ("SHARED (P only)", bench_shared)]:
        print(f"--- {label} ---")
        # Warmup
        for i in range(args.warmup):
            _, t, _ = fn(P_gpu, Y_gpu, nUMI_gpu, Q_gpu, SQ_gpu, X_gpu, args.batch_size, device)
            print(f"  warmup {i+1}: {t:.2f}s")

        # Timed runs
        times = []
        for i in range(args.runs):
            w, t, peak = fn(P_gpu, Y_gpu, nUMI_gpu, Q_gpu, SQ_gpu, X_gpu, args.batch_size, device)
            h = weights_hash(w)
            times.append(t)
            print(f"  run {i+1}: {t:.3f}s  vram={peak:.0f}MB  hash={h}")

        print(f"  median: {np.median(times):.3f}s  min: {min(times):.3f}s")
        print()


if __name__ == "__main__":
    main()
