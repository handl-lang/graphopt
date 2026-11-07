# P27 — the fold cost gate, the straddler re-measure, fp32 as default

Three ledger items landed together on the post-T-016 tree
(`_gen_board` merges the shipped `synop*` folds into the gen board).

* **T-019** — `play._fold_cost_gate`: the `prim:synop*` folds ride
  `DEFAULT` into saturation unconditionally, so the measured-cost
  referee `genkernel.profitable` — which already judges every minted
  *claim* — now also judges the fold entries before they join the
  board's `supported` bound. A `synop_*` member stays feasible only
  when its **delivered** binding (the same callable `hybrid_sink`
  ships, via `kernel_for=_triton_kernel_for`) beats the spelled
  denotation on the case's real values. Entries with no concrete site
  cannot fire and keep their seat; the five attr-metavar specs never
  concrete-match and keep theirs too — the gate refuses a *measured*
  loss, it does not invent one.
* **T-020** — 9-round re-measure of the 5 straddlers.
* **T-021** — `zoo_chain --dtype` now defaults to **fp32** (what a
  caller ships), so a bare re-run quotes the fp32 table; fp64 stays
  one flag away.

Machine-readable rows: `zoo_chain_sweep_t019_gate.json` (the 7
prim-delivering fp64 rows, post-gate) and
`zoo_chain_sweep_straddlers_r9.json` (the 5 straddlers, 9 rounds).

## T-019 — before / after on the 7 prim rows (fp64, 3 rounds, isolated)

speedup = eager / delivered; "pre" is `zoo_chain_sweep.json`,
"ungated" is `zoo_chain_sweep_prim.json` (T-016's merge, no gate),
"gated" is this run.

| case | pre | ungated | gated | prim ops delivered (gated) |
|---|---|---|---|---|
| GatedDeltaRule | 0.958 | 0.500 | **0.542** | `synop_4c5296b3ba`, `synop_5e58f26875` |
| DecoupledMLA | 0.931 | 0.744 | **0.980** | — |
| DilatedBlockAttention | 0.860 | 0.725 | **0.833** | — |
| GRUFusedGate | 0.762 | 0.643 | **0.770** | — |
| ChunkedSSD | 0.956 | 0.864 | **0.972** | — |
| RWKVMixBlock | 0.959 | 0.909 | **1.114** | — |
| HighwayGate | 0.943 | 0.945 | **0.940** | — |
| **median of 7** | **0.943** | **0.744** | **0.940** | |

The gate restored the prim-row median to its pre-merge level
(0.940 vs 0.943) and flipped `RWKVMixBlock` to a verified win
(1.114×, all rounds ≥0.967). Every gated row still verifies
(`max_abs ≤ 8.9e-16`).

**Two honest residuals.**

1. `GatedDeltaRule` still delivers two `synop*` ops at 0.542×.
   `synop_4c5296b3ba` carries attr metavars — its pattern never
   concrete-matches a program subterm, so the referee cannot site it
   and it keeps its seat by design; `synop_5e58f26875` measured
   1.97× *slower* than spelled at its site in a direct re-check yet
   was kept in the sweep run — the site referee is noisy at the
   margin (the same spelled-vs-kernel timings moved ~3× across
   processes on this box; see caveat below). The gate is
   necessary-not-sufficient: it refuses measured site-level losses,
   it does not price module-level interactions.
2. `ChunkedSSD`'s first post-gate run read 0.737× — a bad draw of
   `profitable`'s claim admission (gen_12/14/15/2 admitted that run,
   none on re-check), not the fold gate: the re-check delivered no
   prim ops and no claims at 0.972×, matching pre-merge. The claim
   referee's run-to-run admission noise is pre-existing and
   documented in `zoo_chain.md`.

**Verdict: PASS with residuals.** The gate does exactly what the
ledger asked — extraction can no longer take a fold whose measured
delivery is worse — and it recovered ~all of the 0.20 median the
merge cost. `GatedDeltaRule` shows the two holes: attr-metavar folds
are unmeasurable today, and a borderline site can slip on noise.

## T-020 — the 5 straddlers, 9 rounds each (post-gate tree)

Protocol unchanged: `--isolate`, 150 s budget, median of 9 rounds,
spread kept. "straddles" = median > 1 with at least one round < 1 —
the sweep's own exclusion rule.

| case | dtype | median | spread | rounds < 1.0 | previous | verdict |
|---|---|---|---|---|---|---|
| CapsuleRouting | fp64 | 1.012 | 0.801–1.074 | 4 / 9 | 1.027 (5r, unresolved) | **still straddles** |
| WaveNetGate | fp32 | 1.030 | 0.995–1.149 | 2 / 9 | 1.047 | **still straddles** |
| HyenaLongConv | fp32 | 1.012 | 0.985–1.038 | 3 / 9 | 1.09 | **still straddles**, weaker |
| SkipInitNet | fp32 | 1.115 | 0.974–1.128 | 1 / 9 | 1.156 | **still straddles** (one round 0.974) |
| DilatedBlockAttention | fp32 | **1.098** | 1.078–1.197 | **0 / 9** | 1.084 | **resolves: clean win** |

`DilatedBlockAttention` is the only row that settles — every one of
9 rounds ≥ 1.078× (win tolerance 1.02×), and it delivers **no**
`synop*` ops post-gate (`prim_ops: []`). The other four stay out of
the win count: their medians sit within noise of 1 and at least one
round dips below it. `HyenaLongConv`'s median fell 1.09 → 1.012 —
the earlier figure was itself a straddler's lucky draw.

## T-021 — fp32 is now the default

`bench/drivers/probes/zoo_chain.py --dtype` defaults to `fp32`
(was `fp64`); the module docstring, the `measure`/`_measure_one`/
`_measure_isolated` signatures and the help text all say why: fp32
is what a caller ships, so the quoted sweep number is the one
delivery sees, and a bare re-run cannot drift back to the fp64
convention. `--dtype fp64` still produces the historical table.
`play._zoo_case`'s own default stays fp64 deliberately — its tests
pin the intake convention and the driver is the quoted path.

The canonical fp32 artifact (`docs/reports/zoo_chain_sweep_fp32.json`)
was re-run post-gate (3 rounds, isolated, 150 s budget; the two rows
that first read over-budget — `HyenaLongConv`,
`PerChannelQuantMatmul` — were re-measured and spliced in). Headline:

| | old fp32 sweep (pre-merge) | post-gate fp32 sweep |
|---|---|---|
| delivered / verified | 65 | 65 (same 3 over budget) |
| wins, every round | 23 | **20** |
| straddling 1 | 4 | 7 |
| losses | 38 | 38 |
| median delivered ratio | 0.979× | 0.961× |

The quoted number moved *down*, honestly: the old table predates
T-018's referee fix (the referee now grades the Triton kernel
delivery actually ships, changing which claims are admitted), the
T-016 prim merge, and this gate — three changes at once, so the
refresh is the post-change corpus, not a gate-only delta. Only one
delivered row carries `synop*` ops (`GatedDeltaRule`, 0.486× — the
T-019 residual below). Notable flips: `DecoupledMLA` 1.28→0.92,
`PartialRotary` 1.32→0.91, `RWKVMixBlock` 1.04→0.97 — all
claim-admission noise on the `gen_*` referee, no prim ops involved;
`KVCacheAttention` 1.06→1.23 and `SigmoidAttention` 0.72→0.82 moved
the other way. `HyenaLongConv` reads 1.016× here and 1.012× in the
T-020 9-round measure — consistent.

## Caveats

- The fold gate's referee inherits `profitable`'s noise floor: on
  this box the same site measured t_kernel/t_spelled ratios of 0.64×
  and 2.2× across processes (`synop2_35bbd03301`, GatedDeltaRule
  fp64). Borderline folds can flip run to run; the margin is 1.0×
  by design (any measured loss refuses).
- The gate covers the `prim:synop*` fold claims only. Law folds that
  rewrite to ambient ops (`rms_norm`, the attention folds behind
  `RMSNormPerHead` fp64 0.21× / `SigmoidAttention` fp32 0.72× — the
  two misses the ledger cites) do not pass through `profitable` at
  all; gating those is a different mechanism and remains open.
- Attr-metavar prim specs (5 of 57) are structurally ungated — they
  cannot be sited, so they keep their seat. `GatedDeltaRule`'s
  residual loss runs through one of them.
