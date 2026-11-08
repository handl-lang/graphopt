# p32 — the reduction Triton piece (T-010 lever, codegen arm)

Date: 2026-11-07.  Scope: `mlopt_discovery.genkernel` reduction Triton
codegen + `mlopt_discovery.play` referee binding audit, measured over
the same 32 reduction-bearing workloads of the 68-workload held-out
zoo (`mlopt_discovery.zoo`) that p31 censused, at the default probe
sizes (d=16, t=8, b=2) and at 64×64×8, fp64 and fp32.

## Question

p31 shipped the reduction tier with `Tier("reduction", …, None)` —
`red_*` claims delivered as the `torch.compile`d spelled body.  Its
verdict closed with: *"the remaining upside would need a real
reduction Triton codegen (a `tl.sum`/`tl.max`-axis kernel over the
spelled elementwise body) to compete with Inductor rather than with
eager — that is a codegen project, not a tier."*  This arm built that
codegen and re-priced the lever: does a generated fused
reduce-kernel beat the compiled-spelled incumbent — and therefore
whole-model `torch.compile` — anywhere in the zoo?

## What was built

`genkernel.reduction_triton_spec` / `reduction_triton_bindings` —
the fourth tier's first real Triton piece, same shape as the
pointwise/scan/decayed arms:

- **Grammar**: head ∈ `{sum, mean, amax, amin, softmax}` (the
  `reduce_heads` subset whose lowering returns a plain tensor —
  `max`/`min` return (values, indices), `std`/`var`/`prod`/… need
  semantics the kernel does not spell), exactly one elementwise
  body child, attr keys ⊆ `{dim, axis, keepdim}`, metavar names
  clear of every kernel-local (`_REDUCTION_LOCALS`).
- **Kernel** (`_gr`, one generated `.py` per handler under the
  kernel cache): one program per outer (non-reduced) index; the
  reduce axis is walked in `BLOCK_R` chunks inside the program —
  no cross-program partials, one launch for body + reduction.
  `sum`/`mean` accumulate `tl.sum` (`mean` divides by the reduced
  element count `RF`), `amax`/`amin` fold `tl.maximum`/`tl.minimum`,
  `softmax` runs the textbook max → sum-exp → normalise three-pass
  form in the same launch, recomputing the body per pass.
- **Arbitrary reduce dims**: the logical index space permutes to
  `[non-red dims] + [red dims]` (`_red_permutation`), so
  `sum(dim=2)` on a rank-4 tensor and scattered multi-dim reduces
  lower directly — no trailing-dim restriction.  Broadcast shapes,
  per-operand effective strides (0 on absent/broadcast dims), and
  non-contiguous inputs ride an `ometa` int64 stride/shape table;
  softmax's input-shaped store uses an extra stride slot because
  `row * R + col` is not the output offset once dims permute.
- **Honest fallback**: non-CUDA operands, non-floating dtypes,
  unnormalisable `dim` attrs, empty reduce spaces, and any shape
  outside the subset evaluate the spelled body inside the binding —
  the denotation is always the spec verbatim.
- **Routing**: `Tier("reduction", reduction_handlers,
  reduction_triton_bindings)`; `play._binding_attempts` now prefers
  the tier's `<tier>_triton_spec` recogniser when reporting declines,
  and `gen_sink` assembles the reduction bindings with the other
  tiers'.  The referee's `kernel_for` consult therefore prices the
  Triton callable wherever it binds — delivery and measurement are
  the same arm.

## Correctness

Manual CUDA checks (all `torch.allclose` vs the spelled
`_eval_spec`): `sum`/`mean`/`amax`/`amin`/`softmax`, trailing and
non-trailing and scattered reduce dims, `keepdim` both ways,
full-tensor reduces, broadcast vector operands, broadcast reduce
dims, non-contiguous inputs, fp32/fp64, multi-chunk reduce axes
(`BLOCK_R` < `R`), CPU fallback, integer-dtype fallback.  Max
errors ~1e-14 fp64 / ~1e-6 fp32, 0.0 on the exact cases.

## Measured payoff — the lever is closed

Probe: `bench/drivers/probes/reduction_triton_probe.py` — per
workload `play._zoo_case` intake, `TRITON_REFEREE_LOG` joined with
`gk.PROFITABLE_LOG` (spelled = eager `_eval_spec`, kernel = the
delivery-path callable, reps=20, interleaved windows).  Ratio is
`t_kernel / t_spelled` — **>1 means the Triton kernel is slower**.

**fp64, 32 reduction-bearing workloads:**

| sizes | bound consults | kept | ratio range | median |
|---|---|---|---|---|
| 16×8×2 (zoo default) | 30 | **0** | 1.83–27.39× | 2.52× |
| 64×64×8 | 30 | **0** | 1.87–28.78× | 2.52× |

**fp32, 64×64×8:** 30 bound consults, 29 timed — **0 kept**;
1.51–24.61×, median 2.70×.  One consult (RWKVMixBlock `red_0`, a
`sum`) bound but failed the referee's verify-before-time gate:
fp32 accumulation order on a ~1e29-magnitude body exceeds the
`allclose(rtol=1e-4, atol=1e-5)` tolerance — an honest drop, not a
miscompile (the spelled reference itself carries the same
magnitude).

Per-head medians (fp64, both sizes — the two runs agree within
noise):

| head | n | median loss | worst |
|---|---|---|---|
| `sum` | 8 | ~2.5× | 7.2× (CapsuleRouting) |
| `softmax` | 15 | ~2.5–3.0× | 28.8× (BlockSparseAttention) |
| `mean` | 6 | ~2.5× | 21.0× (RMSNormPerHead) |
| `amax` | 1 | 23.9× | BlockSparseAttention |

The best case anywhere is RWKVMixBlock `red_0`/`red_1` (`sum` over a
large fused body): **1.91× / ~2× slower** than even the *eager*
spelled eval — and the spelled arm the old tier delivered was
`torch.compile`d, i.e. faster still.  Against p31's arm this is a
strict regression at the referee: the two `red_*` keeps p31 measured
at 64×64×8 (0.561×/0.728× vs eager spelled → delivered 1.206× vs
eager, 0.892× vs whole-model Inductor) now read ~1.9× and drop —
**zero `red_*` claims ship anywhere**, so no gen_probe row exists
to price — and the consequence is honest, not silent: once the
Triton builder seats, the referee prices the callable delivery
would actually ship.  The two `red_*` keeps p31 measured at
64×64×8 (compiled-spelled at 0.561×/0.728× vs eager spelled)
are *inside* the Triton grammar, so they are now priced on the
Triton arm (~1.9×) and drop — a claim cannot keep the
compiled-spelled win while shipping the slower kernel.  Specs
outside the Triton grammar (`std`/`max`/non-elementwise bodies)
never bind and keep the p31 compiled-spelled arm unchanged.

## Why the boundary sits here

The loss is not launch overhead alone — the Triton kernel loses to
*eager* spelled eval (a multi-op Python walk) by ~2× median, so the
generated kernel itself is the slow arm:

- **Serial per-program reduce.** One program per outer row walks the
  whole reduce axis in `BLOCK_R` chunks — for the small outer counts
  these bodies have (norms, routers, attention rows) occupancy is a
  handful of programs; Inductor instead tiles the reduction across
  the full SM count with cross-block partials.
- **Generic stride machinery.** Every element pays int64
  `ometa`-table coordinate reconstruction (`//`, `%`, per-dim
  multiply-add) that a specialised kernel would fold into pointer
  arithmetic; broadcast/strided generality is bought per element.
- **Softmax pays the body three times** (max, sum-exp, store) —
  ~2/3 of its traffic is recomputation Inductor's fused softmax
  avoids by keeping intermediates in registers/shared memory.
- **fp64 is not the excuse**: fp32 reads the same (~2.7× median),
  so the deficit is structural, not the RTX 2050's crippled fp64
  pipes.

Inductor wins because it *is* the same idea done better: one fused
kernel, tiled reductions, layout-specialised addressing, autotuned
block shapes — and it compiles the whole module so neighbouring ops
fuse around the reduction too.

## Verdict

The reduction Triton codegen is **implemented, verified, and
honestly negative**.  Every expressible `red_*` spec binds a real
`triton.jit` kernel (30/30 consults at both sizes and both dtypes);
correctness holds across the full subset matrix; the referee prices
the delivered callable and drops all 30 — median ~2.5× slower than
even eager spelled eval, let alone Inductor.  **No reduction kernel
beats `torch.compile`; the lever is closed.**  The tier stays
seated: in-grammar specs are priced on the Triton arm that
delivery would ship (all drop today), out-of-grammar specs keep
the p31 compiled-spelled arm, and the codegen is the
honest record of where the boundary sits: beating Inductor on
reductions needs tiled cross-program partials and layout-specialised
kernels, i.e. an autotuning project, not a bigger grammar.  The
seated builder is the truthful configuration — unseating it would
re-price compiled-spelled while delivery shipped Triton, the exact
mismatch the `kernel_for` hook exists to prevent.

Files: `packages/mlopt-discovery/src/mlopt_discovery/genkernel.py`,
`packages/mlopt-discovery/src/mlopt_discovery/play.py`,
`tests/test_discovery_{genkernel,play}.py`,
`bench/drivers/probes/reduction_triton_probe.py`,
`docs/reports/red_triton_probe_{64,zoo,64_fp32}.json` (raw sweeps).
