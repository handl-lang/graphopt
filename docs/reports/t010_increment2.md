# T-010 increment 2 — candidate survey: one implemented-and-measured negative, three census negatives

**Date:** 2026-11-15 · **Scope:** the next codegen gap after the
clamp attr grammar (`t010_codegen_increment.md`), the chain→`scan_k`
delivery route (`applyd_route.md`), the seq Param/projection lift
(`seq_deindex_params.md`), and the deliver runner arm
(`deliver_runner_arm.md`).

**Verdict up front: no codegen change ships this round.**  The
highest-leverage *remaining* candidate — a stride-1 contiguous fast
path in the reduction kernels — was implemented end-to-end,
verified correct on CUDA, and measured **neutral-to-slower** on
every paying-corner shape; the honest result is a documented
negative and the emitter was reverted.  The other three candidates
are census negatives: zero paying sites in the zoo corpus.

## Candidate census (before building)

Method: all 68 `zoo(16, 8, 2)` workloads exported through
`TorchSource().to_ir`; per workload, `_seq_spines` candidate
spines (≥2 steps) were enumerated and each was run through
`_seq_site`/`_seqm_site` directly, with the per-step skeleton ops
(diffed against the seq renderer's alphabet) recorded for every
decline.  Raw output reproduced below; driver was session-local
(`/tmp/seq_census.py`), not committed.

| candidate | corpus evidence | verdict |
|---|---|---|
| (a) stride-1/contiguous fast path in reduction kernels | paying corner exists (fp32 ≥128 fused softmax bodies — `t010_fp32_attr_levers.md`: 3 kept consults at 128×128×16) | **implemented, measured, reverted — honest negative** (below) |
| (b) graph-arm-aware delivery decisions | runner arm shipped (`deliver_runner_arm.md`); see residuals | surveyed — no actionable gap found |
| (c) seqm/seq spec-dedup residual (`applyd_route.md` known residual) | **≤1 seq/seqm site per zoo workload** — no workload mints two distinct chains folding to one spec | 0 paying sites — honest negative |
| (d) non-elementwise seq step ops (`layer_norm`/`concat`/`sum`/…) | every select-bearing declined spine blocks only on `zeros` (alternative mul-factor readings, by-design arbitration); every spine blocked by `layer_norm`/`concat`/`group_norm`/`chunk`/`transpose`/`conv2d`/mask-ops is **select-free** — a depth/residual chain, not a sequence | 0 paying sites — honest negative |

### (d) detail — the "blocked" spines are not sequences

Declined-spine census by workload (spines ≥2 steps):

- **Select-bearing** (real recurrence candidates): MambaSelectiveScan
  (28 spines → 1 seq site), GatedLinearAttention (28 → 1 seq),
  RealGatedLRU (28 → 1 seq), DeltaRuleBlock (12 → 1 seqm),
  GatedDeltaRule (12 → 1 seqm).  Every declined reading's only
  out-of-alphabet skeleton op is `zeros` — the wrong mul-factor
  readings `_seq_step_cands` documents ("the incorrect 'map' is the
  state term … which either fails deindex or renders nothing
  `scan_spec` accepts").  These are arbitration alternatives of
  chains the minted site already covers, not lost sites.
- **Select-free** (depth chains the spine detector picks up):
  SkipInitNet (4 spines, ops `add/linear/mul/relu` — all in
  alphabet, declines at `not operands`), AdaLNBlock (2,
  `layer_norm`+`chunk`), AdaGNConv (2, `group_norm`+`chunk`),
  MLPMixerBlock (2, `layer_norm`+`transpose`), TimeCondConv (2,
  `conv2d`+`group_norm`+`concat`+mask ops), RWKVMixBlock (8,
  `tril`/`arange`/`concat`/`sum`/`slice`/`eq` mask construction).
  Verified directly: **zero `select` nodes in any step term** —
  these are residual/skip connections (`x + a·mlp(x)`), not
  recurrences.  Lifting `layer_norm`/`concat`/`sum`/mask ops into
  the seq grammar would mint **zero** zoo sites; the
  `seq_deindex_params.md` boundary ("non-elementwise step ops …
  still decline") is real but currently unexercised by the corpus.

### (c) detail

`_seq_scan_sites`/`_seqm_scan_sites` output per workload: 0 or 1
site, never 2.  The spec-dedup residual ("a second chain with an
identical spec but different operands delivers spelled") has **no
zoo instance** — it is a synthetic-corpus-only gap today.

### (b) detail — what was checked

`_deliver_runner_arm` (`play.py:436`) is already generic: it
attempts `CudaGraphRunner` on *any* delivered module when every
example input is a CUDA tensor — `plan_tape` is recorded, not
gating.  Residuals examined:

- **No keep-bar on the wrap**: capture success ships the replay
  wrapper unconditionally.  For plan tapes this is ~always right —
  T-034 showed the loss is ~350–850 µs of host dispatch per call vs
  single-digit µs of static-input copies; the only losing shape
  would be a module already at GPU-bound single-kernel cost, which
  `sink.lower` output never is (always an `IRModule` tape).
- **`profitable` prices eager-vs-eager while delivery ships graph.**
  Ordering plausibly *strengthens* for kept claims: capture
  compresses the multi-launch spelled arm far more than the
  already-fused kernel arm, so an eager keep margin can only grow
  under replay — the risky direction (a kept kernel losing to
  captured-spelled) would need spelled GPU time ≈ kernel GPU time,
  which the fold-vs-chain measurements contradict (the kernel keeps
  the accumulator in registers across `t`; spelled re-reads/writes
  it per step).  Not measured to a verdict — flagged as the one
  open costing question, not a demonstrated gap.

## (a) The implemented experiment — stride-1 contiguous loads

### Design (as built, then reverted)

The reduction kernels address operands through the `ometa`
stride-table: `__o{i} = __ro{i}[:,None] + __co{i}[None,:]` with
`__co{i} = cols64 * stride` (NRED==1) — an opaque int64 gather that
Triton cannot prove contiguous, so loads stay scalar gathers.  The
increment added:

- `_red_binding_vecs(xs, rank, red)` — per-operand flag, 1 iff
  `NRED == 1` **and** the operand's effective stride along the
  reduce dim is exactly 1 (same broadcast mapping as `_red_ometa`;
  absent/singleton/`expand`-stride-0 dims and every `NRED > 1` site
  → 0).
- `_red_vec_params(nops)` — `VEC{i}: tl.constexpr` tail on all five
  load-bearing kernels (`_gr`, `_grms`, `_grn`, `_grm`, `_grs`);
  Triton specialises per flag combo.
- `_red_loads` — `if VEC{i}:` branch spelling the *identical*
  offsets as `__ro{i}[:,None].to(tl.int64) + cols64[None,:]` (for a
  stride-1 operand `__co{i}` IS `cols64` — bit-identical math) but
  with the `tl.arange` column pattern visible to the compiler
  instead of buried in the ometa multiply.  (A first cut used
  `tl.max_contiguous(__o, [1, BR])`; measured equally neutral —
  Triton 3.8's `max_contiguous` needs the `multiple_of` alignment
  half, which the arbitrary per-row base offsets cannot prove.)

### Correctness (CUDA, before revert)

`assert_close` vs `_eval_spec` spelled on: `sum(dim=2)` fp64,
transposed (non-contiguous) operand, `softmax(dim=-1)` fp32,
multi-dim reduce `(1,2)` (flags all 0), broadcast `(8,1)` operand —
all pass.  (`tests/test_discovery_genkernel.py -k "reduction"`:
11 passed post-revert; `scripts/pytest-capped.sh` could not execute
in this environment — `systemd-run --user` permission denied — so
the slice ran under plain `.venv/bin/python -m pytest`.)

### Measurement — the paying corner, A/B by flag

Same kernel source, same call; only `_red_binding_vecs` forced to
zeros for the plain arm (constexpr-dead branch ≈ pre-change codegen).
RTX 2050, serial, min of 5×reps after warmup:

| shape / dtype | VEC=1 µs | plain µs | verdict |
|---|---|---|---|
| softmax `add(mul(X1,.125),X2)` 4096×128 fp32 | 80.4–82.0 | 81.0–81.1 | neutral (~1%, noise) |
| same, fp64 | 641–644 | 640–645 | neutral |
| `sum(mul)` 2048×256 fp32 | 57.8–70.4 | 54.8–70.3 | neutral/noisy |
| `sum(mul)` 512×1024 fp32 | 78.4–78.5 | 77.1–77.8 | neutral/slightly worse |
| `sum(mul)` 2048×512 fp32 | 57.8–70.4 | 54.8–70.3 | neutral/noisy |

(Earlier single-run cells read vec 0.928×/0.758× — re-measured
interleaved they were warm-up/clock noise; the careful table above
is the record.)

### Why it cannot pay (root cause, not just a null result)

Vectorised `ld.global.v4` needs *provable 16-byte alignment* of
every row's column run.  The row base `__ro{i}[r]` is an arbitrary
stride-table decode — no `multiple_of` claim is truthful for it,
and Triton correctly refuses to vectorise.  The gather was never
the binding constraint anyway: at the paying-corner shapes the
kernels run ~80 µs (fp32, ≈125 GB/s effective — near the tile's
latency/occupancy floor, not DRAM saturation) and ~644 µs fp64
(libdevice `exp`/`tanh`-class body cost dominates — loads are
noise).  A real stride-specialised win needs *layout-specialised*
codegen (a dedicated contiguous-operand kernel where the whole
offset is `row*ld + arange` with `ld` a launch arg Triton can
reason about), which is the multi-day emitter rework
`t010_fp32_attr_levers.md` already scoped — this increment confirms
the cheap version of it does not pay.

**Reverted**: `git checkout` on `genkernel.py` — no dead machinery
ships for a measured-neutral path (the repo's honest-posture rule:
the referee's verdict is the verdict).

## What this leaves on the table (honest next levers)

- **Layout-specialised reduction kernels** — the real version of
  (a): a contiguous-operand fast path where offsets are affine in
  `arange` with launch-arg strides.  Multi-day; payoff corner still
  the fp32 ≥128 softmax bodies (3 consults / 1 delivered win in the
  last full sweep).
- **`profitable` under the graph arm** — price keeps against
  graph-captured arms (or re-check at deliver) if a flipped-margin
  case is ever measured; today no evidence one exists.
- **Seq-tier grammar for select-bearing step ops** — currently
  *zero* corpus demand (every declined select-bearing spine is an
  arbitration alternative); revisit only if a new recurrence
  spelling enters the zoo.
- Everything else in the agenda's frontier rows is measurement or
  law-side, not codegen.

## Files

- `docs/reports/t010_increment2.md` — this report.
- No package code changed (the stride-1 experiment was reverted
after measurement; `git status` clean apart from this file).
- Session-local: `/tmp/seq_census.py` (the collector census driver).
