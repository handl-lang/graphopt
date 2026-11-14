# Graph-replay recheck — post-`8589162` correctness of prior `CudaGraphRunner` claims

**Date:** 2026-11-14 · **Builds on:** `8589162` (the `_GraphReplayModule`
backwards-copy fix), `t037_resweep_frontier.md`, `resweep_scan_route.md`,
`deliver_runner_arm.md`, `trajectory_scan_impl.md`, `seqt_readout.md`,
`module_loss_gla_mamba.md`, `t030_031_chunkedssd_headroom.md`,
`t032_rescope.md`, `t034_module_costing.md`, `results_wholemodel_gap.md`,
`whisper_experiment{,_r2}.md`, `whisper_decode_loop.md`.

## The bug and what it invalidated

`8589162` fixed `_GraphReplayModule.forward`
(`packages/mlopt-cuda/src/mlopt_cuda/runners.py:157`): pre-fix,
`zip(xs, graph_ins)` ran `caller_tensor.copy_(graph_buffer)` — replay
never saw fresh inputs (frozen on capture-time values) **and** silently
overwrote the caller's tensors.  Carrier executors
(`capture_cuda_graph`) used the correct order and were never affected.

Two consequences for every prior *generic-path* (plain `IRModule`)
`CudaGraphRunner` delivery:

- **Timing claims survive.**  The replay always launched the same
  captured kernel sequence; only the input *values* were stale.  Every
  `graph_us` / `graph_speedup` / `graph_vs_compiled` number stands.
- **Fresh-input correctness claims did not.**  Worse, every
  `graph_max_abs`/`graph_vs_*` numeric check in the probes was
  *vacuous*: `_probe_graph_arm` (`play.py:~3233`) and
  `_deliver_runner_arm` (`play.py:~505`) both compared
  `module(*xin)` vs `wrapped(*xin)` on the **same tensors used for
  capture** — and pre-fix `wrapped` mutated `xin` back to the capture
  values before replaying them, so the delta was 0.0 by construction.
  The reported `graph_max_abs: 0.0` rows proved nothing about fresh
  inputs.

Unit coverage was already strengthened by the fix commit:
`tests/test_runners.py:243` (`test_cuda_graph_runner_captures_generic_irmodule`)
asserts replay-vs-eager on a *fresh* input, non-mutation of caller
tensors, and shape-mismatch fallback; `:464` covers the
`multi_device` per-device cache (same `_GraphReplayModule`).  The gap
this report closes is **end-to-end deliveries** — real minted
extractions, not synthetic `matmul` IRModules.

## Method — `bench/drivers/probes/graph_replay_recheck.py`

For each claim-bearing workload the probe replays `gen_probe`'s
delivery half exactly (`_zoo_case` → `_gen_board` → `_probe_play` →
`board.deliverable` → `hybrid_sink.lower` → `sink.verify` — mint is
nondeterministic, so claim sets drift run to run; the *delivery
shape* is what is under test), wraps the verified module in
`CudaGraphRunner`, and compares three arms on **4 fresh input
variants** per row (scaled+noised and `randn_like` draws of the
capture feed — never the capture tensors, same
shape/dtype/device contract):

- `graph_vs_delivered` — replay vs the eager delivered `IRModule`
- `delivered_vs_base` — eager delivery vs the source model (sanity)

`graph_replay_recheck_gap.py` does the same for the two
`results_wholemodel_gap.md` generic-path sites (retnet
`_ChunkedRetention` carrier fp64/fp32, chainfuse `_mlp_case`
generic + runner-delivered), importing the probe's own fixtures.
The `play.deliver("gen", …)` hook arm was run inline for
`deliver_runner_arm.md`'s case.  Rows: `docs/reports/graph_replay.jsonl`
+ `graph_replay_gap.json`.  RTX 2050 4 GB, serial, ~25 min total.

## Verdicts

### CONFIRMED (correctness re-verified post-fix, bit-exact)

Every generic-path graph-arm delivery re-measured produces
**max |replay − eager-delivered| = 0.0** on all fresh variants — the
captured graph now tracks its inputs exactly.  `delivered_vs_base`
stays inside the verify tolerances (≤ ~3e-5 fp32, ≤ ~2e-15 fp64).

| claim site | rows rechecked | result |
|---|---|---|
| `t037_resweep_frontier.md` "graph beats compile 7/8" + `resweep_scan_route.md` | ChunkedSSD, DeltaRuleBlock, GatedDeltaRule, DecoupledMLA, GatedLinearAttention, MambaSelectiveScan × fp32+fp64 @ zoo(64,32,8) — **the complete set of graph-arm rows in `zoo_chain_sweep{,_fp32}.json`** | 12/12 bit-exact; all `cuda_graph=True` |
| `trajectory_scan_impl.md` / `seqt_readout.md` (seqs/seqt tiers) | HeadedSelectiveScan, BidirectionalScan fp64 @(64,32,8); GatedLinearAttention, MambaSelectiveScan, HeadedSelectiveScan, BidirectionalScan, **RealGatedLRU** fp32 @(16,8,2) — RealGatedLRU *did* deliver here (the fp64 @(64,32,8) row that timed out still has no claim) | 7/7 bit-exact |
| `deliver_runner_arm.md` (`--deliver` hook, `play.deliver("gen")`) | GatedDeltaRule fp32 CUDA, `ScriptedPlayer`, budget 12 — the actual shipped hook, not the probe path | bit-exact on 4 fresh variants; `graph_speedup` 5.43 (report: 4.44 — same arm, different mint/timing) |
| `results_wholemodel_gap.md` generic lane | retnet `_ChunkedRetention` carrier fp64 + fp32 (`carrier*+graph(run)`), chainfuse `_mlp_case` `generic64+graph(run)` + `delivered64+graph(run)` | all bit-exact (graph-vs-eager Δ = 0.0; eager-vs-base ≤ 3.3e-16 fp64 / ≤ 1.2e-7 fp32) |
| `module_loss_gla_mamba.md` graph arm | same four workloads as the frontier set (DeltaRuleBlock, GatedDeltaRule, GatedLinearAttention, MambaSelectiveScan fp32) | covered by the 12/12 above |
| `t030_031` / `t032` / `t034` module-costing reports | ChunkedSSD, DecoupledMLA, GatedDeltaRule — all in the frontier set | covered |

### TIMING-VALID, CORRECTNESS NOT CLAIMED (no re-check needed)

- `whisper_experiment{,_r2}.md` — the encoder `CudaGraphRunner` arms
  are *timing-only* claims ("replay 63.4 ms — no change", "no gain,
  GEMM-bound").  No fresh-input correctness was asserted; the
  delivered modules' `verify=True` gates ran on the eager module.
  `whisper_decode_loop.md` (post-fix) already re-verified the decode
  step end-to-end: 32/32 generated ids match.
- `phase_c_serving.md` / `device_policy.md` — `multi_device` capture
  semantics; the per-device cache delegates to the same fixed
  `_GraphReplayModule`, and `test_runners.py:464` replays fresh
  same-signature inputs per device.

### CAN'T-RECHECK (bounded scope)

- `graph_vs_compiled` ratios were **not re-timed** — they need a
  `torch.compile` baseline per row (tens of seconds each × 19 rows)
  and the bug cannot have moved them: pre-fix replay ran the same
  kernel sequence on stale inputs, so `graph_us` — and hence every
  ratio built on it — is bit-identical pre/post fix *for the same
  captured module*.  Spot re-times confirm the arm still captures
  and pays (below).
- `RealGatedLRU` fp64 @(64,32,8): never produced a delivery row in
  any report (search budget), so there is no claim to re-verify.

## Do the pay numbers still hold?

Yes — structurally (identical kernel sequence ⇒ identical replay
timing) and empirically on the cheap re-time (`_time_module`,
reps=50, no compile arm).  Claimed `gen_graph_us` vs re-measured
`graph_us` (mint drift noted — the current bank delivers different
claim sets than the sweep rows on the seq workloads, so absolute
µs legitimately differ there; the pre-seqt rows reproduce almost
exactly):

| row | claimed graph µs | re-measured | note |
|---|---|---|---|
| DeltaRuleBlock fp32 | 36.9 | 36.4 | reproduces |
| GatedDeltaRule fp32 | 46.5 | 46.6 | reproduces |
| DecoupledMLA fp32 | 120.2 | 119.9 | reproduces |
| DeltaRuleBlock fp64 | 143.6→114.3* | 88.9 | *sweep/superseded; mint drift |
| GatedDeltaRule fp64 | 135.8 | 136.2 | reproduces |
| DecoupledMLA fp64 | 371.4 | 386.6 | within mint noise |
| ChunkedSSD fp64 | 707.0 | 806.1 | mint drift (different claim set) |
| HeadedSelectiveScan fp64 | 420.2 (seqt) | 416.9 | reproduces |
| BidirectionalScan fp64 | 278.3/277.5 | 279.0 | reproduces |
| deliver-hook GatedDeltaRule fp32 | 96.6 | 87.9 | reproduces (5.4× vs eager) |

The "7/8 frontier rows beat `torch.compile`" headline is a
*timing* claim — it stands.  What changes is only the confidence
behind it: pre-fix those graph-wrapped modules would have returned
capture-time-frozen outputs on any input but the first, so the
numbers described a module that was fast **and wrong**; post-fix
the same timings describe a module that is fast **and correct**.

## Published numbers needing correction

**None for pay.**  Every `graph_us`/`graph_speedup`/`graph_vs_compiled`
figure is timing-valid.  The `graph_max_abs: 0.0` fields in
`zoo_chain_sweep{,_fp32}.json` and the probe reports were vacuous
pre-fix; they are now honestly re-verified at 0.0 on fresh inputs
(post-fix the compare is meaningful — replay genuinely tracks
inputs), so no JSON or markdown number changes.  If any document
quoted `graph_max_abs` as *evidence of correctness on unseen inputs*,
that evidence did not exist pre-fix — this report is the correction.

## Files

- `bench/drivers/probes/graph_replay_recheck.py` — the zoo recheck
  driver (`--only/--dtype/--d --t --b/--fresh/--reps/--budget/--out`).
- `bench/drivers/probes/graph_replay_recheck_gap.py` — the
  wholemodel-gap generic-lane recheck.
- `docs/reports/graph_replay.jsonl` — 20 rows (19 probe + 1
  deliver-hook), per-variant deltas.
- `docs/reports/graph_replay_gap.json` — the 5 wholemodel arms.
