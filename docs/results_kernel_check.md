# Kernel board audit — independent check of the landed `kernelopt` work

Measurement-only pass over the kernel board as committed (`3801504`
spike, `54230d5` expansion; `packages/mlopt-kernel/`,
`bench/discovery/kernel.py`, `bench/limits/triton/`,
`--domain kernel` in `mlopt_discovery.play`).  Claims are taken from
`project/notes/kernel-engineer-board.md` and the two commit messages.
Same box the committed numbers were measured on: RTX 2050 / 16-core /
11 GB, torch 2.14.0+cu130, triton 3.8.0, CUDA fp32 where the claim
says so.  Run 2026-10-20 while the tree also carried the parallel
session's uncommitted WIP (the dtype-contract fix in
`torch_bridge.py`/`meta_eval.py`/`decayed_lower.py` and gamekit
moves) — measurements below are HEAD+WIP, and the two runs that
depend on the fix say so.

**Box caveat.**  Load average 5–7 during the runs (shared box).
Dispatch-side columns are noisy accordingly; every kernel-busy and
verified-equal column reproduced inside noise.

## 1. The 11-case kernel table — `python -m bench.discovery.kernel`

Committed (expansion note) vs reproduced, CUDA fp32, 50 reps median:

| case | claim: spelled→sched µs (×) | repro: spelled→sched µs (×) | triton claim | triton repro | verdict |
|---|---|---|---|---|---|
| kernel_silu | 476→209 (2.28×) | 475.0→209.6 (2.266×) | 200 ✓ | 201.5 ✓ | verified |
| kernel_softsign | 665→209 (3.19×) | 666.2→211.7 (3.147×) | 201 ✓ | 201.3 ✓ | verified |
| kernel_fma | 756→563 (1.34×) | 757.3→565.7 (1.339×) | 291 ✓ | 292.3 ✓ | verified |
| kernel_swiglu | 755→302 (2.50×) | 754.3→303.0 (2.489×) | 292 ✓ | 291.6 ✓ | verified |
| kernel_softmax | 668→208 (3.21×) | 667.0→210.0 (3.176×) | 199 ✓ | 199.5 ✓ | verified |
| kernel_layernorm | 1056→209 (5.05×) | 1050.5→210.5 (4.991×) | 200 ✓ | 200.1 ✓ | verified |
| kernel_mm_bias | 899→731 (1.23×) | 856.6→763.2 (1.122×) | 900 ✓ | 940.9 ✓ | drifted, ordering holds |
| kernel_mm_scale_bias | 1114→990 (1.13×) | 1088.5→850.9 (1.279×) | 904 ✓ | 920.7 ✓ | drifted *up*, ordering holds |
| kernel_strength | 662→661 (1.00×) | 681.0→663.4 (1.027×) | 290 ✓ | 291.8 ✓ | verified |
| kernel_mixed | 940→487 (1.93×) | 948.7→492.7 (1.925×) | 290 ✓ | 291.0 ✓ | verified |
| kernel_flat (control) | 661→661 (1.00×) | 662.4→663.2 (0.999×) | skip | skip | verified |

Every row `verified=True` on both the scheduled arm and the triton
arm; `kernel_flat` honestly reports `no triton kernel registered`.
The two mm-epilogue rows moved — `mm_bias` scheduled slower
(763 vs 731 µs), `mm_scale_bias` scheduled *faster* (851 vs 990 µs) —
but the claimed ordering is intact on both: the Inductor-lowered
`kf_mm_*` composite still beats the tutorial-tile hand Triton kernel
at this size (763<941, 851<921).  Compile-cache/warm-up ordering is
the likely cause of the drift; the claim ("Inductor's mm path beats
the hand tile at this size") survives.

**The honest-gap rows reproduce too**: partial fusions leave a
launch — fma 566 vs triton 292, mixed 493 vs 291, strength 663 vs
292 — the leftover-pointwise residue their note names as the next
arm.

## 2. Player leaderboard — `bench.players.run --boards kernel`

`--meta-cases 24 --meta-budget 8` (the note's "24 held-out, budget 8"
protocol; held-out stream is `cases(seed+7919)`):

| player | claim reward @ evals/ep | repro reward @ evals/ep | verdict |
|---|---|---|---|
| linear-greedy | 0.560 @ 2.29 | 0.5598 @ 2.292 | verified (bit-identical) |
| linear | 0.499 @ 3.42 | 0.4992 @ 3.417 | verified (bit-identical) |
| playbook | 0.200 @ 8.0 (drains) | 0.2004 @ 8.0 (drains) | verified |
| linear-cold | 0.145 @ 6.17 | 0.1452 @ 6.167 | verified |
| random | 0.099 @ 5.63 | 0.0993 @ 5.625 | verified |

All 24 episodes end in certified extraction for every arm, matching
the note.  `distilled`/`neural` correctly report does-not-play
(`--list-players`: distilled→contraction only, neural→open only).
`play --domain kernel --player linear-greedy` runs end-to-end
(0.59 vs random 0.30 on a 4-case eval, 4/4 certified).

Not re-run: the "cold REINFORCE plateaus ~0.20–0.28 at 600 cases"
caveat — that is a training run, and `linear-cold`'s 0.145 held-out
is consistent with it without re-measuring.  The spike-era numbers
(linear 0.49 / playbook 0.34 / random 0.08 on the 5-law bank) are a
*different action set* — superseded by the verified 10-law table
above, not checkable as such.

## 3. Unit layer — `tests/test_kernel.py`

29 tests, all pass in 2.8 s (law guards: keepdim/eps/pow-2 declines,
verify-first sink behavior, cyclic-enode extraction constraint).

## 4. The ~1.6×-to-FLA residual — does the kernel board reach it?

`docs/results_wholemodel_gap.md` leaves the retnet residual
(`fused32+graph` 0.454 ms vs `fla-chunk` ~0.28 ms) explicitly to the
kernel board.  Two questions: does the before-side still measure the
same, and can any landed kernel arm move it.

**Before-side re-measure** (`bench/drivers/probes/
wholemodel_gap_probe.py --case retnet --t 2048`, two runs):

| arm | committed | run 1 | run 2 | verdict |
|---|---|---|---|---|
| carrier(gen)64 | 3.209 | 11.767¹ | 3.215 | verified (run 1 was contention) |
| carrier32+graph(run) | 1.151 | 1.156 | 1.210 | verified |
| fused(exec)32 | 0.951 | 1.171 | 1.154 | drifted ~+0.2 ms — dispatch-bound arm² |
| fused64+graph | 2.078 | 2.089 | 2.095 | verified |
| fused32+graph | **0.454** | **0.456** | **0.500** | verified (~0.45–0.50 band) |
| verify fp64 | 1.67e-16 | 1.67e-16 | 1.67e-16 | verified³ |
| verify32 | passed, rel 1.30e-07 | passed, rel 1.30e-07 | passed | verified³ |
| ambient-fp64 check | float32 | float32 | float32 | verified³ |

¹ First run showed 8.8 ms of CPU dispatch on the 205-launch generic
tape with *identical* kernel-busy time (2981 vs 2980 µs); the re-run
lands at the committed number.  The doc itself flags this class —
"dispatch-side noise on a shared box" — and its uncaptured-fp32 row
drifted the same way (2.917→3.726).

² `fused(exec)32` is dispatch-dominated (707 µs dispatch of 1,154 µs
wall); kernel-busy is unchanged (447 vs 445 µs).  Same contention
class, not a regression in the executor.

³ These three depend on the dtype-contract fix, which is **present
only as uncommitted WIP** in this tree (`git diff` on
`torch_bridge.py`/`meta_eval.py`/`decayed_lower.py` is the fix).
They reproduce because the WIP is in the working tree; at bare HEAD
they would not.

`fla-chunk` stays a carried retro number (`fla-core` not installed —
same caveat as their doc).  Residual as measured here:
**0.456–0.500 vs 0.28 ms ≈ 1.6–1.8×** — the ~1.6× claim holds.

**Does a landed kernel arm reach it?  No — structurally, on three
independent checks:**

- **Op coverage.**  The extracted carrier (saturated at T=512/K=4
  with the probe's own `_saturate`; root `concat`, 14 distinct ops)
  uses `decay`, `decay_vec`, `om_decayed`, `split`, `concat`,
  `linear`, `contiguous`, `detach_` — **8 of its 14 ops are outside
  `KernelSink.supported_ops`**.  KernelSink cannot lower the
  extracted retnet term at all, scheduled or not.
- **Law reach.**  Firing the 10-law bank at the extracted carrier:
  only `kern_div_recip` applies (one site — the `/√T`), kernel_cost
  368→349; the scheduled term is still unlowerable.  The fusion
  laws find *no* sites — the carrier's hot terms are decay masks,
  gathers and pair GEMMs, not `a·σ(b)`/`exp/Σexp`/`c·rsqrt` shapes.
- **The other residual (chainfuse MLP tail).**  On the *spelled*
  program `kern_fuse_silu` fires once (`mul(y,σ(y))` inside the
  softplus) — kernel_cost 34→32, a partial fusion of exactly the
  "leftover pointwise" class their note concedes.  On the
  *extracted* term (`applyd` root, `aff_diag` carrier op) zero laws
  fire and the sink again cannot lower.  And regardless of reach,
  the bank is **not wired into the delivery path**: it ships as
  `play --domain kernel` + `KernelSink`, deliberately out of
  `DEFAULT` (`laws.py`: "these rules only pay under the kernel cost
  model"), so `Optimizer.search`/`lower` never sees it.

**Verdict on the residual: unchanged and out of current reach.**  The
kernel board measures and verifies real fusions on its own case
family, but the retnet gap needs a *fused scan/decay kernel* — a
different op vocabulary (`om_decayed`-family sites, gathers,
materialized `s⊙d`) plus a sink that can lower it — and possibly the
recurrent-state schedule choice the gap doc names.  The committed
note is honest about this: "the leftover-pointwise residue is the
next arm."  What this audit adds: even the *pointwise* arm as landed
does not reach the carrier terms — the sink's op table and the
`DEFAULT`-bank firewall both stand between it and the residual.

## 5. Claim ledger

| claim (source) | verdict |
|---|---|
| verified-equal kernels at 1.36–3.13× on 4/5 CUDA cases (`3801504`) | verified — 11-case superset reproduces; strength ties, flat ties |
| fused composites at hand-Triton speed on single fusions (`54230d5`) | verified — 209–212 vs 199–202 µs |
| 10-law bank, all verify-first, guards hold (`54230d5`) | verified — test_kernel 29/29; table all-verified |
| triton adapter: real kernels all 10 cases, honest skip | verified — 10 builders run+verify; flat skips |
| honest gap: partial fusions leave a launch | verified — fma/mixed/strength residue rows reproduce |
| mm epilogues: Inductor beats tutorial tile | verified in ordering; absolute times drifted (763/851 vs 731/990 vs triton 941/921) |
| players: greedy 0.560 > linear 0.499 > playbook 0.200 > cold 0.145 > random 0.099 | verified — bit-identical under the leaderboard protocol |
| cold REINFORCE plateaus ~0.2–0.28 (caveat) | not re-run (training); consistent with cold 0.145 |
| residual ~1.6× to FLA is the kernel board's target (gap doc) | before-side verified (0.456–0.500 vs 0.28 retro); **no landed arm reaches it — see §4** |

*Generated by an independent measurement pass; instrument reuse per
the repo convention (committed probe + bench module), no new
machinery.*
