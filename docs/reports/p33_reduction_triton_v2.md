# p33 — reduction Triton codegen v2 (T-010 re-attack)

Date: 2026-11-08.  Scope: `mlopt_discovery.genkernel` reduction Triton
emitter only — same `reduction_triton_spec` grammar, same
`reduction_triton_bindings` interface, same referee pricing path as
p32.  What changed is *what the kernels look like*, attacked in the
order p32's "why the boundary sits here" named: occupancy, addressing,
softmax passes.

## What v2 changed

p32's diagnosis was structural: one program per outer row with a
serial in-program chunk loop (occupancy = outer-row count), generic
int64 `ometa` stride-table decode paid per element, and softmax
recomputing the body three times.  v2 attacks each:

- **Occupancy — tiled grid + cross-program partials.**  The reduce
  ops (`sum`/`mean`/`amax`/`amin`) emit `_gr` + `_gc`: `_gr` is a
  `(row-blocks, chunks)` 2-D grid — each program reduces one
  `[BM, BR]` tile (BM outer rows × one BR column chunk of the reduce
  axis) and writes either the final per-row result (`DIRECT`, the
  `NC == 1` case) or a per-chunk partial at `rows * NC + pid1`;
  `_gc` folds the `NC` partials per row (`mean` divides by `RF`
  once, in the combine).  `_red_tile_dims` picks `(BR, NC, BM)`:
  `BR` = pow2 covering the reduce axis capped at 256, `NC` = chunk
  count, `BM` = pow2 sized for ~512 row-blocks with `BM * BR` ≤ ~2048
  elements and a `BM * BR ≥ 256` fill floor (a `[1, 64]` tile idles
  half a CTA — measured ~2× slower before the floor).
- **Softmax — blocked passes, fused stats.**  `NC == 1` (the whole
  reduce axis fits one tile — the common case): `_grms` computes each
  row's max *and* sum-exp in a single pass over one loaded tile,
  `_grn` stores the normalised values — two launches instead of
  p32's three-pass single launch, and each pass is tiled over
  rows × columns.  `NC > 1`: `_grm`/`_gcm` (partial row max +
  combine), `_grs`/`_gcs` (partial sum-exp against the combined max
  + combine), `_grn` — the split-K form, five launches.
- **Addressing — int32 decode + cached metadata.**  Row/column
  coordinates decode through int32 index arithmetic (was int64
  `//`/`%` per dim per slot); the binding falls back to spelled eval
  when the addressable offset range or the broadcast size would
  exceed int32 — honesty preserved via the same in-binding fallback.
  `_red_ometa` buffers are cached per layout/device/shape/permutation
  (`_RED_OMETA`, bounded at 256) — rebuilding the table per call cost
  a ~14 µs pageable H2D copy that small reductions cannot afford.
- **Literal `pow(x, 2)`** lowers to `x * x` instead of
  `libdevice.pow` (RMSNormPerHead went ~20× → ~1.9× on this alone).

Unchanged: the spec grammar, `_red_permutation`/ometa layout,
`_red_spelled` fallback posture, `Tier("reduction", …,
reduction_triton_bindings)` seating, and the referee's
verify-before-time gate.  Only emitted kernel source and the host
launch choreography differ.

## Correctness

Same posture as p32, re-verified on the v2 kernels: manual CUDA
sweep over `sum`/`mean`/`amax`/`amin`/`softmax`, trailing and
non-trailing and scattered reduce dims, `keepdim`, multi-dim reduces,
full-tensor reduces, broadcast operands, `NC > 1` split paths
(`R > 256`), non-contiguous inputs, fp32/fp64 — all `torch.allclose`
vs spelled, max diff ~1e-14 fp64 / ~1e-6 fp32, exact on the
integer-count cases.  `pytest tests/test_discovery_genkernel.py
-k reduction` — 11 passed.

## Measured payoff

Probe: `bench/drivers/probes/reduction_triton_probe.py`, same
methodology as p32 (`TRITON_REFEREE_LOG` × `PROFITABLE_LOG`,
`t_kernel / t_spelled`, **>1 = Triton slower**).  Raw sweeps:
`docs/reports/red_triton_probe_v2_{64,64_fp32,zoo}.json`.

**fp64, 64×64×8** — v1 → v2:

| | bound | kept | ratio range | median |
|---|---|---|---|---|
| v1 | 30 | 0 | 1.87–28.78× | 2.52× |
| v2 | 30 | **2** | **0.60–2.83×** | **1.83×** |

Per-head medians (fp64, 64×64×8): `sum` 2.46→1.81 (n=8),
`softmax` 2.54→1.91 (n=15), `mean` 2.48→1.76 (n=6), `amax`
23.93→1.59 (n=1).  The kept claims are RWKVMixBlock `red_0`/`red_1`
— **0.60× / 0.71× vs eager spelled**, the p31 keeps returned on the
Triton arm.

**fp32, 64×64×8** — 30 bound, **1 kept**: RWKVMixBlock `red_1`
0.365–0.391×; `red_0` bound but failed verify-before-time (same
fp32 accumulation-order tolerance issue as v1 — honest drop).
Median over the 29 timed consults = 2.14× (0.39–3.06).

**fp64, zoo default 16×8×2** — 30 bound, **0 kept**; median
2.52→2.07×, range 1.37–2.95× (v1: 1.83–27.39×).  At small sizes the
remaining deficit is launch/occupancy floor, not kernel quality —
RWKVMixBlock itself drops to 1.43×/1.52×.

**Delivered-module pricing (`gen_probe`, the number the lever is
priced on):**

| case | claims | baseline (eager) | compiled (Inductor) | delivered gen | vs eager | vs compiled |
|---|---|---|---|---|---|---|
| RWKVMixBlock fp64 64³ | gen_4, red_0, red_1 | 3359.6 µs | 3072.8 µs | 3127.0 µs | **1.074×** | **0.983×** |
| RWKVMixBlock fp32 64³ | red_1 | 1243.3 µs | 263.6 µs | 1506.4 µs | 0.825× | 0.175× |

The fp64 delivery nearly matches whole-model `torch.compile`
(1.7% slower) — versus p31's compiled-spelled delivery at
1.206×/0.892×, the Triton arm is strictly better on both axes.  The
fp32 keep is honest but does **not** survive incumbent pricing:
Inductor's fp32 whole-module fusion (263.6 µs, 4.7× under eager) is
far stronger than anything a per-claim kernel can deliver — the
referee keeps `red_1` on spelled-relative merit (0.37×) while the
module-level truth is a loss.  That asymmetry is the same one p31
faced; it is recorded, not hidden.

## Verdict

v2 converts p32's structural diagnosis into measured wins **where
the shape pays**: occupancy tiling plus split-K partials took the
worst case from 28.8× to 2.8× and the median from 2.52× to 1.83×
(fp64, 64×64×8); two claims keep and the delivered RWKVMixBlock
module runs at 0.983× vs whole-model Inductor — the first
reduction-kernel delivery that does not lose to the incumbent.
But the win is narrow: 2/30 consults at fp64-64, 1/30 at fp32-64
(a keep that loses to compiled anyway), 0/30 at zoo sizes.  Small
reductions still can't amortise launch + tile overhead against
eager spelled eval, and no per-claim kernel can fuse the way
Inductor fuses the whole module.  **The lever is no longer fully
closed** — it now has a paying corner (large fused reduce bodies,
fp64) — but it is not a general win; whether to keep the arm
seated is a cost/benefit call the referee already makes per claim.

Files: `packages/mlopt-discovery/src/mlopt_discovery/genkernel.py`,
`tests/test_discovery_genkernel.py` (unchanged interface, re-verified),
`bench/drivers/probes/reduction_triton_probe.py` (unchanged),
`docs/reports/red_triton_probe_v2_{64,64_fp32,zoo}.json` (raw sweeps).
