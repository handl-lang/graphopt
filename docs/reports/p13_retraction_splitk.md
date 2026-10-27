# P13 — retraction: the split-k numbers were wrong

**Every "we beat `awq_gemm`" claim in P8, P8b, P8c, P9, P10 and P12 is
withdrawn.** The kernel used a split-K path that accumulates with
`tl.atomic_add`, and it does not produce correct results.

## The bug

`_gemv_v2` with `SPLIT > 1` splits the K reduction across programs and
combines partials with `tl.atomic_add`. Measured against the unpacked
reference on the real `down_proj` weights:

| split | max abs error |
|---|---|
| 1 | **0.0037** ✅ |
| 2 | 3.78 ❌ |
| 4 | 5.48 ❌ |
| 8 | 5.56 ❌ |

Zeroing the output buffer first does not fix it. The partial sums are
being lost or double-counted, so a split kernel was doing *less work than
the reference* — which is exactly why it looked fast: 55–92 µs instead of
the honest 173 µs.

Every config sweep in this block chose `split=4` or `split=8`, because a
broken kernel that computes two-thirds of the sum is faster than a correct
one. The referee could not catch it: **the searches gated on exactness
only for candidates that ran, and the exactness tolerance was checked
against a reference the timing path never re-validated.** P12 dropped the
check entirely, which is how the error survived into a routing table.

## What the corrected measurement says

Cold (deployment) regime, `split=1`, exact to 0.0037:

| layer | ours | `awq_gemm` | torch | winner |
|---|---|---|---|---|
| `down_proj` 896×4864 | 172.9 µs | 208.0 | **146.9** | torch |
| `q_proj` 896×896 | 72.5 | 61.1 | **53.4** | torch |
| `gate_proj` 4864×896 | 81.0 | **67.4** | 126.7 | `awq_gemm` |
| `k_proj` 128×896 | 52.8 | 56.3 | **32.3** | torch |

**We win zero shapes.** Our kernel is the slowest of the three on three of
four, at roughly 14 % of the roofline (172.9 µs against 24.6 µs).

## What survives

- **The referee** (`bench/benchkit/regime.py`): eviction, blocked
  randomization, medians. It is what caught the regime problem and what
  would have caught this if it had been wired into the sweep's gate.
- **The float-op results (P6/P7)**: those used no split and no atomics —
  `grouped_quant_linear` is exact and 2.27–2.87× over dequant-then-matmul
  on real AWQ weights. Unaffected.
- **The negatives**: weights-transfer is speed-neutral in vLLM (P5); the
  MLP-chain fusion ceiling is 1.29 % (P5c); compute-bound shapes are
  capped at 1.03× (frontier).

## The lesson worth keeping

A speed number is only meaningful with an exactness check *on the timed
code path*, run every time, not once at development time. The sweeps
gated candidates on exactness and then timed a *different* configuration
path. Two rules from now on:

1. the exactness check runs inside the timing harness, per candidate,
   immediately before its timing, and a failure aborts the row;
2. any speedup over 2× on a memory-bound kernel is treated as a bug
   until the byte count and the roofline sanity-check it.

## Where that leaves the work

The machinery is sound; the kernel is not. The next honest step is to
make the int4 GEMV *correct and fast* — fix or drop split-k, then measure
against the 24.6 µs roofline rather than against a competitor. Beating
`awq_gemm` is not currently supported by any valid measurement.
