# autoresearch — rctd-py

Autonomous optimization of the rctd-py GPU-accelerated RCTD solver.

## Setup

1. **Create the branch**: `git checkout -b autoresearch/<tag>` from current main.
2. **Read the key files**:
   - `src/rctd/_irwls.py` — the core IRWLS solver. **This is the primary file you modify.**
   - `src/rctd/_likelihood.py` — cubic spline likelihood (`calc_q_all`). Fair game for optimization.
   - `src/rctd/_simplex.py` — simplex projection. Can be optimized.
   - `src/rctd/_full.py` — full-mode pipeline entry point. Uses `solve_irwls_batch_shared`.
   - `benchmarks/bench_gpu.py` — **READ-ONLY benchmark. Do NOT modify.**
3. **Initialize results.tsv** with header row.
4. **Establish baseline**: Run the benchmark as-is to get the starting `elapsed_s` and `weights_hash`.

## Benchmark

Run the fixed benchmark:

```bash
uv run python benchmarks/bench_gpu.py --n-pixels 50000 --n-genes 1000 --n-types 12 --batch-size 10000 --warmup 1 --runs 3 --device cuda > run.log 2>&1
```

Extract key metrics:
```bash
grep "elapsed_s\|weights_hash\|peak_vram" run.log
```

## Metric

**Primary**: `elapsed_s` (median) — lower is better.
**Correctness constraint**: `weights_hash` MUST match the baseline. If it changes, the optimization broke numerical correctness. Discard immediately.
**Soft constraint**: `peak_vram_mb` should not increase dramatically. Some increase is fine for meaningful speed gains.

## What you CAN modify

- `src/rctd/_irwls.py` — solver, QP, PSD projection, derivatives, convergence
- `src/rctd/_likelihood.py` — `calc_q_all` spline interpolation
- `src/rctd/_simplex.py` — simplex projection
- `src/rctd/_full.py` — batching strategy, data transfer patterns

## What you CANNOT modify

- `benchmarks/bench_gpu.py` — the benchmark is fixed
- Test files under `tests/`
- `pyproject.toml` — no new dependencies

## Logging results

Append to `results.tsv` (tab-separated):

```
commit	elapsed_s	memory_mb	hash	status	description
a1b2c3d	12.345	2048	abc123def456	keep	baseline
b2c3d4e	11.890	2100	abc123def456	keep	reduce QP sweeps to 30
c3d4e5f	12.500	2048	abc123def456	discard	torch.compile on calc_q_all (slower)
d4e5f6g	0.000	0	CRASH	crash	OOM with batch_size=50000
```

## The experiment loop

LOOP FOREVER:

1. Look at current state: branch, last result, what's been tried
2. Modify the solver code with an experimental optimization
3. `git commit -am "experiment: <description>"`
4. Run the benchmark: `uv run python benchmarks/bench_gpu.py --warmup 1 --runs 3 --device cuda > run.log 2>&1`
5. Extract results: `grep "elapsed_s\|weights_hash\|peak_vram" run.log`
6. If empty → crash. Check `tail -n 50 run.log` for stack trace. Fix or discard.
7. Record to `results.tsv`
8. If `elapsed_s` improved AND `weights_hash` matches baseline → **keep** (advance branch)
9. If `elapsed_s` worse OR `weights_hash` changed → **discard** (`git reset --hard HEAD~1`)

**NEVER STOP.** Keep experimenting until manually interrupted.

## Optimization ideas (starter list)

1. **torch.compile on calc_q_all**: Currently only QP solver is compiled. The spline interpolation (`calc_q_all`) is 36% of runtime — try `@torch.compile`
2. **Adaptive QP sweeps**: Use `_solve_box_qp_batch_adaptive_jit` with early exit instead of fixed 50 sweeps
3. **Mixed precision**: Use float16 for prediction/gradient intermediates, keep float64 for weights
4. **Fuse calc_q_all into derivatives**: Avoid reshaping overhead by computing d1/d2 inline
5. **CUDA graph capture**: Wrap the inner IRWLS loop in a CUDA graph for kernel launch elimination
6. **Batch size tuning**: Try larger batches (20k, 50k) to improve GPU utilization
7. **Warm start**: Initialize weights from a fast closed-form estimate (e.g., NNLS) instead of uniform
8. **Lazy PSD**: Skip eigendecomposition when Hessian is already PSD (check eigenvalue signs)
9. **Custom CUDA kernel**: Write a Triton kernel for the Gauss-Seidel coordinate descent
10. **Cholesky instead of eigh**: Use Cholesky decomposition for PSD matrices — cheaper than full eigendecomposition
11. **Precompute P.T and P_outer on GPU**: Ensure these are computed once and reused across batches
12. **Stream/pipeline batches**: Overlap data transfer with computation using CUDA streams
