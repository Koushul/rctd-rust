# Roadmap: 100% Concordance with R spacexr

**Date:** 2026-03-05
**Current concordance:** 99.7% dominant type agreement (Xenium, 58k pixels, K=45, doublet mode)
**Goal:** 100.0% concordance — bit-level matching of weight vectors on the spacexr vignette dataset, and >99.95% on large Xenium datasets.

---

## Where the 0.3% Gap Comes From

The discrepancy is concentrated in **doublet mode** — singlet pixels already achieve ~99.99% agreement. Three root causes are responsible, listed by estimated impact.

### 1. QP Solver Divergence (biggest source)

**R code:** `quadprog::solve.QP(D, d, A, b)` — Goldfarb-Idnani dual active-set algorithm.
**Python code:** `_solve_box_qp()` (`_irwls.py:65-100`) — Gauss-Seidel coordinate descent + simplex projection.

These are fundamentally different algorithms. While both converge to the same optimum for well-conditioned problems, they diverge when:
- The Hessian `D` is ill-conditioned (cond(D) >> 1e6, common for K=45 types)
- Multiple coordinates are simultaneously active at their bounds
- The warm-start point differs (Python uses diagonal approximation; R uses `solve(D, d)`)

The comment at `_irwls.py:87-89` documents this explicitly:
> "Using solve(D, d) blows up when cond(D) >> 1 (e.g. 5e6 for 45 types), producing delta_w ~ 1e5"

**Impact:** This is the primary source of weight differences. Even small delta_w divergences accumulate across IRWLS iterations (up to 50) and compound through the doublet pipeline where pairwise scoring uses IRWLS results.

### 2. Doublet Pair Selection Ties

**R code:** `process_bead_doublet` iterates candidate pairs and selects the pair with lowest score.
**Python code:** `run_doublet_mode` (`_doublet.py:86-157`) does the same, but floating-point accumulation differences from source #1 cause tie-breaking to differ.

When two candidate pairs score within ~1e-4 of each other, the QP solver differences from source #1 can flip which pair is selected. Once a different pair is chosen, the K=2 refit in step 6 produces legitimately different weights.

**Impact:** This is the mechanism that turns small numerical differences into categorical disagreements (different cell type assignments). Affects ~0.3% of pixels.

### 3. Likelihood Interpolation Precision

**R code:** Natural cubic spline interpolation using R's internal spline functions.
**Python code:** `compute_spline_coefficients()` (`_likelihood.py:22-42`) using scipy/numpy matrix inversion.

The tridiagonal matrix inverse `MI` (437×437) is computed differently between R and Python. While both should produce identical results in exact arithmetic, floating-point accumulation differs, leading to O(1e-10) differences in spline coefficients. These propagate through `calc_q_all()` into every IRWLS iteration.

**Impact:** Small per se, but compounds with source #1 over many iterations.

---

## Proposed Fixes (Priority Order)

### Priority 1: Replace QP solver with quadprog port (Medium effort)

Replace `_solve_box_qp()` with a Python implementation that matches R's `quadprog::solve.QP` exactly.

**Options:**
- **Option A (recommended):** Use `qpsolvers` package with `quadprog` backend — this is a Python binding to the same Goldfarb-Idnani algorithm. `pip install qpsolvers quadprog`.
- **Option B:** Port the Fortran source of `quadprog::solve.QP` to PyTorch directly. More work but avoids CPU roundtrip.
- **Option C:** Use `cvxpy` with `OSQP` backend. Different algorithm but may match more closely than Gauss-Seidel.

**Implementation plan:**
1. Add `quadprog` to optional dependencies
2. Create `_qp_goldfarb_idnani()` wrapper that matches R's calling convention
3. Replace `_solve_box_qp()` in single-pixel `solve_irwls()`
4. For batched path (`solve_irwls_batch`), call the QP solver per-pixel on CPU (the QP is tiny: K×K where K≤45)
5. Validate on vignette data (100 beads) — should achieve exact match

**Risk:** The batched CPU QP may be slower than the current GPU Gauss-Seidel. Profile and consider keeping both paths with a `solver="quadprog"` option.

### Priority 2: Match tie-breaking in doublet pair scoring (Small effort)

Ensure the iteration order over candidate pairs matches R exactly.

**Steps:**
1. In R, `combinations(cands, 2)` iterates in a specific order. Verify Python's `itertools.combinations` produces the same order (it should — both produce lexicographic order).
2. When scores tie (within machine epsilon), use the same tie-breaking rule as R:
   - R: first pair in iteration order wins (implicit `<` comparison)
   - Python: currently same (`if sc < min_p_score`), but floating-point differences from source #1 may change which pair is "first"
3. After fixing source #1, verify this resolves automatically.

### Priority 3: Export R's spline coefficients (Small effort)

Instead of computing spline coefficients from scratch in Python, export R's exact coefficients and load them.

**Steps:**
1. In R: compute the Q-matrices and spline coefficients, save as `.npz`
2. In Python: load pre-computed coefficients instead of calling `_get_or_compute_MI()`
3. This eliminates any floating-point divergence in the interpolation tables

**Note:** The pre-computed Q-matrices (`q_matrices.npz`) already exist and are used. The question is whether the spline coefficients computed at runtime from these Q-matrices exactly match R's.

### Priority 4: Validate on vignette data first (Small effort)

Use the 100-bead Slide-seq vignette (same data as tutorial) for per-pixel debugging.

**Steps:**
1. Run R spacexr on vignette, export per-pixel: `weights_doublet`, `spot_class`, `first_type`, `second_type`, intermediate IRWLS weights at each iteration
2. Run Python rctd-py on same data, export same intermediates
3. Compare pixel-by-pixel, iteration-by-iteration to identify first divergence point
4. This small dataset makes debugging tractable (100 pixels × 19 types vs 58k × 45)

### Priority 5: Re-validate on Xenium Region 3 (Small effort)

After fixes, re-run the full 58k-pixel benchmark:
1. Run `scripts/validate_pytorch_vs_spacexr.py` with updated code
2. Update `data/region3/xenium_metrics.json` with new concordance numbers
3. Update README claims if concordance improves

---

## Expected Outcome

| Fix | Expected concordance improvement |
|-----|----------------------------------|
| QP solver (Priority 1) | 99.7% → ~99.95% |
| Tie-breaking (Priority 2) | Resolves automatically with #1 |
| Spline coefficients (Priority 3) | ~99.95% → ~99.99% |
| Combined | ~99.99%+ on large datasets, 100% on vignette |

**True 100%** (bit-identical weights) is achievable on the 100-bead vignette but may remain at 99.99% on 58k-pixel datasets due to irreducible floating-point differences between R and Python linear algebra libraries (LAPACK implementations, FMA instruction differences, etc.). This is acceptable — the practical threshold for "matching" is dominant type agreement >99.95%.

---

## Non-Goals

- **Matching R's performance characteristics:** We intentionally use GPU batching which changes memory access patterns. Numerical equivalence is the goal, not identical execution traces.
- **Supporting R's legacy quirks:** Some R spacexr behaviors (e.g., `norm(as.matrix(x))` defaulting to L1 instead of L2) are already matched. We won't match undocumented R behaviors that are clearly bugs.
