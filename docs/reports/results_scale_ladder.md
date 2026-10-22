# Scale ladder — attention-block terms, T = 512 → 4096

Pure measurement: zoo attention blocks pushed through the shipped pipeline (export → e-graph saturation → extraction → lower → verify) and the `bench/referees` gauntlet (oracle → pay → sweep → cert), each stage timed, doubling T until the box says stop. Driver: `bench/drivers/probes/scale_ladder.py`; raw rows: `results_scale_ladder.json`.

- date: 2026-10-22 · git: `2aa6ba5` (dirty tree; driver committed alongside this doc) · device: cpu (fp64, 12 threads)
- protocol: `{"d": 16, "b": 2, "ts": [512,1024,2048,4096], "seed": 0, "per_case_s": 480, "min_avail_gb": 2.5, "max_enodes": 100000, "max_memory_mb": 6144, "max_iterations": 100, "rules": "default_rules()", "n_rules": 273}`
- box: 11 GB host, shared with parallel agent sessions (~5 GiB MemAvailable at run start); `VmHWM` sampled per stage — cumulative peak, not per-stage
- workloads: `zoo(d=16, t=T, b=2)` — two *unrolled* spellings (`GatedLinearAttention`, `MambaSelectiveScan`: `steps=t`, so the traced term grows ~linearly in T) and three *symbolic* attention blocks (`ChunkedRetention`, `ALiBiSlopeAttention`, `RWKVMixBlock`: the term is T-invariant, only the example tensors grow)

## The ladder

`s` columns are wall seconds; `nodes` = distinct subterms of the exported IR root; `enodes`/`stop` from `eg.run` stats; `peak MiB` = the rung's max `VmHWM` (a cumulative watermark — the process floor was ~1.3 GB after the first rung).

| workload | T | nodes | enodes | iters | stop | export s | search s | extract s | verify s | pay-leg s | peak MiB | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GatedLinearAttention | 512 | 5 648 | 7 696 | 2 | fixed_point | 14.8 | 134.2 | 0.9 | 0.1 | 42.3 | 1 234 | ok |
| GatedLinearAttention | 1024 | 11 280 | — | — | (killed by alarm) | 25.7 | 454.5 | — | — | — | 1 302 | **timeout** |
| MambaSelectiveScan | 512 | 5 139 | 7 187 | 2 | fixed_point | 9.6 | 124.9 | 0.5 | 0.5 | 41.1 | 1 302 | ok |
| MambaSelectiveScan | 1024 | 10 259 | — | — | (killed by alarm) | 34.4 | 445.6 | — | — | — | 1 333 | **timeout** |
| ChunkedRetention | 512 | 27 | 35 | 2 | fixed_point | 0.1 | 0.1 | 0.0 | 0.1 | 0.2 | 1 333 | ok |
| ChunkedRetention | 1024 | 27 | 35 | 2 | fixed_point | 0.0 | 0.1 | 0.0 | 0.0 | 0.1 | 1 346 | ok |
| ChunkedRetention | 2048 | 26 | 35 | 2 | fixed_point | 0.1 | 0.1 | 0.0 | 0.3 | 0.4 | 1 795 | ok |
| ChunkedRetention | 4096 | 26 | 35 | 2 | fixed_point | 0.1 | 0.1 | 0.0 | 0.7 | 1.2 | 3 911 | ok |
| ALiBiSlopeAttention | 512 | 34 | 57 | 3 | fixed_point | 0.1 | 0.1 | 0.0 | 0.1 | 0.0 | 3 911 | ok |
| ALiBiSlopeAttention | 1024 | 34 | 57 | 3 | fixed_point | 0.1 | 0.1 | 0.0 | 0.2 | 0.0 | 3 911 | ok |
| ALiBiSlopeAttention | 2048 | 34 | 57 | 3 | fixed_point | 0.1 | 0.1 | 0.0 | 0.9 | 0.0 | 3 911 | ok |
| ALiBiSlopeAttention | 4096 | 34 | 57 | 3 | fixed_point | 0.1 | 0.3 | 0.0 | 6.6 | 0.0 | 7 033 | ok |
| RWKVMixBlock | 512 | 81 | 90 | 3 | fixed_point | 0.2 | 0.2 | 0.0 | 0.3 | 0.0 | 7 033 | ok |
| RWKVMixBlock | 1024 | 82 | 90 | 3 | fixed_point | 0.1 | 0.2 | 0.0 | 0.9 | 0.0 | 7 033 | ok — **verify refused (NaN)** |
| RWKVMixBlock | 2048 | 83 | 90 | 3 | fixed_point | 0.1 | 0.2 | 0.0 | 34.8 | 0.0 | 7 737 | ok — **verify refused (NaN)** |
| RWKVMixBlock | 4096 | — | — | — | — | — | — | — | — | — | — | **killed (kernel OOM)** |

## Laws still firing at scale

Identical fired-law sets at every rung per workload — T changes nothing about *which* laws fire; it only changes how *many* sites exist on the unrolled spellings.

| workload | fired laws (fires at T=512) | modeled cost before → after |
|---|---|---|
| GatedLinearAttention | `decay_state_lift` (512), `decay_state_lift_post` (512) | −4.5 % |
| MambaSelectiveScan | `decay_state_lift` (512), `decay_state_lift_post` (512) | −4.9 % |
| ChunkedRetention | `om_decayed_lift` (2), `decay_lift`, `sub_to_add`, `mul_div_swap`, `matmul_div_distribute` | −37.9 % (512) → −60.4 % (4096) |
| ALiBiSlopeAttention | `sdpa_fold_addmul` (2), `decay_state_lift_post` (3), `om_lift`, `unsq_mul` (2), `mul_unsq_pad_l/r`, `sub_unsq_pad_l`, `add_unsq_pad_r`, `sub_to_add` (2), `naturality_scalar_rev` | −80.4 % (512) → −93.8 % (4096) |
| RWKVMixBlock | `sub_to_add` (2), `decay_state_lift_post` (2), `pow_to_square`, `sub_unsq_pad_l`, `square_expand` | −0.4 % → −0.05 % |

The unrolled blocks fire `decay_state_lift`/`_post` once per step — 512 fires each at T=512, so the fire *count* scales with T while the law *set* does not.  On the symbolic blocks the yield grows with T: the folds' payoff concentrates in the T² work (ALiBi's `sdpa_fold_addmul` + carrier lifts reach −94 % modeled at T=4096; ChunkedRetention −60 %).  Every delivered module verified at max_rel ≲ 5e-14 **except RWKV at T ≥ 1024** (see below).

## Gauntlet legs at scale

Each rung re-judged the workload's busiest shipped law through `run_component` on a shared `ClaimEvidence` (the memoized-verdict seam `judge` uses).  Legs stop at the first refusal, as `judge` does.

| leg | scales with | measured |
|---|---|---|
| oracle | term size (real-match scan) + a T-blind synthesized window | ≪ 0.2 s everywhere |
| pay | term size **and** real compute — one-rule e-graph on the probe term, then lower + `sink.verify` per changed case | **≈ 42 s** on the ~5.5 k-node unrolled terms; ≲ 1 s on symbolic |
| sweep | synthesized adversarial window (T-blind) + real-match region | ≲ 0.7 s |
| cert | `verify_law` on the claim's two sides — T-blind by construction | ~ms; **refused every carrier claim** (`not derivable` — the carrier laws are not derivable under the `ALL_RULES` bank; expected, not a regression) |

Verdict details worth noting: the RWKV claim (`sub_to_add`) fails the pay leg honestly at every rung — `fires=5 changed=0 paid=0` — so sweep/cert never ran for it; the carrier claims (`decay_state_lift`, `om_decayed_lift`, `decay_state_lift_post`) pass oracle + pay + sweep and refuse only at cert.

## Where it breaks — three distinct walls

1. **The e-graph wall is term size, not T, and it is CPU time not memory.**
   Saturation reaches `fixed_point` in 2–3 iterations on every measured rung; the cost is the first-scan matcher pass (273 rules × classes).  On the ~5.1–5.6 k-node unrolled terms that pass costs ~125–134 s with ~1.3 GB peak — and on the ~10–11 k-node T=1024 terms it did not finish inside the 480 s rung budget (>445 s in `search`, still ~1.3 GB).  Under the shipped default budgets, **`search` saturates the unrolled T=512 spellings in ~2 minutes and does not saturate T=1024 inside a 480 s budget** — the unrolled attention spellings hit the same matcher wall the `slow`-marked saturation tests document, an order of magnitude below their term sizes.

2. **The memory wall is not in the e-graph at all — it is the referee leg's real forward of the spelled program.**
   `RWKVMixBlock` at T=4096 was kernel-OOM-killed (SIGKILL — the rung could not even flush its own stage clocks; the pre-rung `running` row the driver now writes exists for this).  Forensic isolation: `export_to_ir` re-run standalone completes in 1.8 s (the trace is symbolic — it never materializes activations), and running the lowered module on the `(2, 4096, 16)` feed dies: the spelled wkv attention materializes a `(B,T,T,d)` = 2·4096·4096·16 fp64 attention tensor ≈ **17.2 GB** plus intermediates, ~3× the available memory.  The e-graph had saturated that same term in 0.2 s at 90 enodes.  The pipeline's symbolic stages scale fine; the *spelled program* does not — an honest limit of verifying a deliberately-unfused spelling, not of the search machinery.

3. **The numerics wall shows up before the memory wall: RWKV verify refuses at T ≥ 1024 with `max_rel = NaN`.**
   The unscaled `exp(bias + k)` attention overflows fp64 at long T; both the reference and optimized lowerings faithfully reproduce the spelled computation, and `sink.verify` correctly refuses to certify a NaN output.  The pipeline is semantics-faithful — the spelled program itself is unstable at scale.

## Honest claims

- Under the shipped default budgets (273 rules, `symmetry_budget=2048`, `max_enodes=100k`, `max_memory_mb=6 GiB`), e-graph saturation is **T-invariant on symbolic attention terms**: identical nodes/enodes/fired-law sets at every rung; saturation stays ≲ 0.3 s through T=4096.
- The pipeline end-to-end survives T=4096 on the symbolic blocks: ALiBi verify 6.6 s (peak 7.0 GB), ChunkedRetention 0.7 s (peak 3.9 GB).
- On unrolled attention spellings, `search` saturates T=512 (~5.1–5.6 k nodes) in ~2 min and exceeds a 480 s budget at T=1024 (~10–11 k nodes) — the breaking point is matcher time, not memory.
- The box's OOM boundary was hit once, at `RWKVMixBlock` T=4096, in the **verify leg** — the first stage that executes the spelled program — on the ~17.2 GB `(B,T,T,d)` attention tensor.
- The `cert` leg is T-blind and refused all three carrier claims (~ms); `pay` is the only gauntlet leg whose cost tracks term size (~42 s at ~5.5 k nodes).

## Method and caveats

- `search s` bundles the shipped `Optimizer.search` call (export → saturate → non-local passes → first extraction); `export s` is the same `to_ir` step timed standalone, and `extract s` a second `extract_best` on the saturated graph — a lower bound on in-search extraction.  The saturating pass itself is ≈ `search − export − extract`.
- `peak MiB` is `VmHWM` at the stage boundary — monotone, so a rung's column is the cumulative high-water mark, not a per-stage allocation.
- `verify` = `sink.verify` of the delivered module against a fresh reference lowering of `res.ir` (the `lower(..., verify=True)` referee leg, split for timing); `lower` itself cost ≲ 0.15 s on every completed rung.
- The per-rung alarm (480 s) is a *budget*, not a mechanism limit: "timeout" means the rung's `search` did not finish inside it, documented as such — no saturation bound was tightened and no stage was skipped silently.
- Rungs were seed-pinned per (workload, T) and run single-process; the box was shared, so the wall times are ± tens of percent, not microbenchmarks.
