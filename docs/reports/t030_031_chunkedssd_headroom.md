# T-030 / T-031 — ChunkedSSD frontier headroom: diagnosis and prototype

Agenda rows `frontier_delivery_gap` on `zoo:ChunkedSSD` (agenda ranks 2–3):
the Amdahl-on-dispatch ceiling says ~2.7–3.0x is available, the delivered
body captures ≈0 of it.  This note traces the measurement, profiles the
delivered module on CUDA, and lands a working prototype of the missing
arm.

## What the numbers mean

`bench/suites/integration/frontier.py` joins two committed artifacts:

- `docs/reports/zoo_chain_sweep.json` (+ `_fp32`, probe sweeps) — per case
  `runs[].{eager_us, compiled_us, gen_us}`, median-reported by
  `bench/drivers/probes/zoo_chain.py` → `play.gen_probe`;
- `docs/reports/zoo_kcount.json` — **eager-only** CUDA-profiler row:
  `device_us`, `wall_us`, `device_share`, `kernels`.

Per row: `F = 1 - device_share`, `ceiling = 1/(1-F)` (perfect dispatch
elision), `captured = (speedup-1)/(ceiling-1)` with
`speedup = median(eager_us)/median(gen_us)`.  Two caveats on the inputs:

- `kernels` is **distinct kernel names** (`profiler.key_averages()`), not
  launches — the launch count is ~25x larger (below).
- `frontier_fp32.json` was generated from `zoo_chain_probe_fp32.json`
  (5 cases, **1 round**) via `--sweep/--kcount/--stem`, but `run_bench`
  writes `inputs` from `DEFAULT_SWEEP`/`DEFAULT_KCOUNT` — the recorded
  provenance is wrong for override runs (`frontier.py:640-645`).  Minor,
  but it makes the fp32 report look like it re-read the fp64 sweep.

The two ChunkedSSD rows:

| report | dtype | rounds | eager µs | delivered µs | speedup | ceiling | captured | vs_compile |
|---|---|---|---|---|---|---|---|---|
| frontier.json | fp64 | 3 | 2618.2 | 2989.8 | 0.876 | 3.02 | −0.062 | 0.286 |
| frontier_fp32.json | fp32 | 1 | 2438.1 | 2215.0 | 1.101 | 2.72 | +0.058 | — |

The fp64 sweep row itself is noisy: runs were speedups 0.956 / **1.36** /
0.876 — round 2 had `eager_us=4771` (an order-of-magnitude excursion, the
documented "unrelated process on the box" failure mode in
`zoo_chain.py`'s docstring).  Later re-sweeps of the same module
(`zoo_chain_sweep_t016_strided_all`, `t026_fp64`) read 1.01–1.03x.  So the
*sign* of the delivered-vs-eager gap is within noise; the *magnitude* of
the uncaptured ceiling is not — that part is structural.

## What the delivered body actually is

`play.gen_probe` → `_zoo_case` → `TorchSource.to_ir` → mint tiers →
`genkernel.profitable` referee → `_gen_board`/`_probe_play` (claim every
offer, saturate, extract) → `hybrid_sink(gen_handlers,
compile_kernels=True)` → `sink.lower(ir)` → verify → `_time_module`.

`sink.lower` is a **per-leaf plan tape**: every uncovered IR node
lowers to an eager torch op.  Only handler-covered sites get fused
compiled kernels.  The frontier "delivered" number is this raw lowering —
**no `Runner` is ever applied** (`runner=` is an `optimize()` delivery
option; this path bypasses `optimize` entirely).

## Live profile (this box, RTX 2050, `zoo(64,32,8)` ChunkedSSD)

Reproduced `gen_probe`'s delivery path and profiled each arm
(`torch.profiler`, CUDA activity, 20–100 reps):

| arm | wall µs | device µs | launches/fwd | distinct kernels |
|---|---|---|---|---|
| eager (fp32) | 1832–2246 | ~478 | **446** | 15 |
| delivered, raw lowering (fp32) | 2496–2792 | ~560 | **418** | 14 |
| `torch.compile` (fp32) | 458–633 | ~145 | 115 | 24 |
| **delivered + `CudaGraphRunner` (fp32)** | **527–565** | — | ~1 replay | — |
| eager (fp64) | 1935 | — | — | — |
| delivered (fp64) | 2440 | — | — | — |
| compiled (fp64) | 572 | — | — | — |
| **delivered + `CudaGraphRunner` (fp64)** | **624** | — | ~1 replay | — |

Three facts fall out:

1. **The delivered body keeps ~94% of eager's launches** (418 vs 446).
   The minted claims cover a handful of sites; the chunk-loop body
   (~15 ops × 8 chunks unrolled, plus projections) lowers per-leaf.
   Dispatch is ~75% of wall — exactly the share the ceiling prices.
2. **The raw delivered module is often *slower* than eager** (the plan
   tape's host dispatch costs more than eager's own).  That is why
   `captured` goes negative on the fp64 row — delivery loses time
   *before* the ceiling is even relevant.
3. **`CudaGraphRunner`'s generic path captures the delivered module
   cleanly** (`stats["cuda_graph"]=True`) and replay takes wall from
   ~2.5 ms to **~0.53–0.62 ms — a 3.1–4.3x speedup over eager, at or
   above the Amdahl ceiling** (the ceiling only models dispatch
   removal; replay also hides the tape's host overhead, and on fp32 the
   run *beat* `torch.compile`: 527 µs vs 633 µs, `graph_vs_compiled
   1.20`).  `graph_max_abs = 0.0` — bit-identical replay.

A manual `torch.cuda.CUDAGraph` capture of the same module failed with
`IndexError: Dimension out of range` — a probe-harness artifact (my
capture wrapper mishandled the module's view/concat path), **not** a
property of the module: the shipped `_GraphReplayModule.capture` recipe
(static buffers + side-stream warmup) succeeds where the naive recipe
doesn't.  Worth noting only as evidence the runner's warmup matters.

## Verdict: actionable, cheap, and already shipped

This is **not** a missing law, missing move, or unreachable bound.  The
headroom is pure launch/dispatch overhead, and the substrate already
ships the exact tool for it — `mlopt_cuda.CudaGraphRunner` (generic
`nn.Module` path built for precisely this "IRModule plan tape" case,
per `packages/mlopt-cuda/src/mlopt_cuda/runners.py` and
`docs/results_wholemodel_gap.md`).  The gap is that the **zoo
measurement path never applies a Runner**: `gen_probe` times
`sink.lower(...)` raw, so every `frontier_delivery_gap` row on a
launch-bound case reads ~0 capture regardless of what delivery *could*
do.

The same mechanism almost certainly explains the sibling proposals —
DecoupledMLA, DeltaRuleBlock, GatedDeltaRule, CapsuleRouting, FNOBlock,
RotaryPair are all `launch-bound`/`launch-leaning` rows with
`captured ≤ ~0.5`.  A graph arm should be checked across all of them
before assuming per-case causes.

### Prototype landed (uncommitted)

`packages/mlopt-discovery/src/mlopt_discovery/play.py`:

- `_probe_delivery` now returns `(rep, t_gen, module)` (single caller).
- New `_probe_graph_arm`: lazily imports `CudaGraphRunner` (optional dep
  — `mlopt-cuda` is not in mlopt-discovery's requirements), applies it to
  the *same verified module*, times the replay, and records
  `gen_graph_us`, `graph_speedup`, `graph_vs_compiled`, `graph_max_abs`.
  Honest decline → no keys (capture failure, non-CUDA input, or
  `stats["cuda_graph"]` unset all record nothing).  `gen_us`/`speedup`
  are unchanged — the new keys are an additive arm, not a redefinition
  of "delivered".

End-to-end validation through the real driver:

```
.venv/bin/python -m bench.drivers.probes.zoo_chain \
    --only ChunkedSSD --isolate --rounds 1 --dtype fp32
→ gen_us 2791.7 (0.804x), compiled_us 632.8,
  gen_graph_us 526.9, graph_speedup 4.262, graph_vs_compiled 1.201,
  graph_max_abs 0.0
```

`pytest tests/test_discovery_genkernel.py -k TestProbe`: 3 passed.
`ruff check` + `ruff format`: clean.

## What's still missing (the bigger arm, if wanted)

1. **Re-measure**: a `zoo_chain` sweep with the new arm (68 workloads,
   isolated, ~hours) then a frontier variant that reads
   `gen_graph_us` — `Row` needs a `delivered_graph_us` field and a
   second `captured` column.  Until then the committed frontier rows
   still quote the raw-lowering number.
2. **Delivery semantics decision**: should `gen_probe`'s "delivered"
   *be* the runner-wrapped module (i.e. graph capture is part of
   delivery, matching `optimize(..., runner=CudaGraphRunner())`), or
   stay a separate arm?  The probe currently answers both by reporting
   both — the report layer has to choose which one `captured` prices.
3. **Compose with compile**: on fp32 the graph-wrapped delivery already
   beat Inductor here; on rows where fusion matters more (device-time
   bound), `torch.compile` + graph is the remaining gap — Inductor can
   emit cudagraphs itself, so "compile the delivered body" is the
   cheapest next arm to price.
4. **Shape dynamism**: `CudaGraphRunner` bakes the capture-time shape
   and falls back to eager on mismatch (`_GraphReplayModule._matches`) —
   fine for fixed-shape serving, but a deliverable contract needs the
   fallback policy stated (it already degrades gracefully).
5. **Mint nondeterminism observed**: the claims offered for ChunkedSSD
   differed across identical runs (`[]`, `[gen_11, gen_13]`,
   `[gen_10..14, synop_*]`, `[decayed_0, gen_13]`, `[synop_93e57c595e]`)
   — likely set-iteration ordering in the mint/proposal path.  Doesn't
   change this verdict but makes single-round sweep rows less
   reproducible than the 3-round median intends.
