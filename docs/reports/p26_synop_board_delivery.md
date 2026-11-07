# P26 — `synop*` on the zoo board: deliverable, verified, still losing

T-016's first step is landed and measured. `play._gen_board` now merges
the shipped `synop_*`/`synop2_*`/`synop3_*` folds into the gen board's
handler table (`shipped_primitives.handler_entries()` — one row per
minted op, kernel = the op itself, `attr_mvars` declared for the
attr-metavar specs), widens `supported` to the op names, and reseats
the case's sink on the merged table. `genkernel.gen_bindings` honours
`attr_mvars` — the arriving call's attrs are `prim_mint.spec_bind`'d
back into the spelled denotation spec before `_eval_spec` — and
`hybrid_sink`'s `triton_bindings` tier now reaches the elementwise
prims (real generated `triton.jit` source, audited in
`triton_binding_audit*.json`). The spec-walk `register()` binding is
overridden on this path: the referee grades the fused callable.

Machine-readable sweep: `docs/reports/zoo_chain_sweep_prim.json` —
the same 46 fp64 losing rows of `zoo_chain_sweep.json`, re-run through
`play.gen_probe` after the merge (2026-11-07, post-T-018). The task's
"median 0.979×" is the fp32 all-corpus figure; the 46-row set is the
fp64 loser list, whose baseline median is 0.945.

## Numbers (speedup = eager / delivered, >1 wins)

| set | n | median before | median after |
|---|---|---|---|
| all 46 re-swept rows | 46 | 0.945 | **0.942** |
| rows whose delivered term carries `synop*` ops | 7 | 0.943 | **0.744** |
| rows with no prim op delivered | 39 | 0.947 | 0.947 |

The 7 prim-delivering rows, all `verified` (max_abs ≤ 8.9e-16, fp64):

| case | prim ops in delivered term | before → after |
|---|---|---|
| GatedDeltaRule | `synop2_35bbd03301`, `synop_5e58f26875`, `synop_bb8c54c201` | 0.958 → 0.500 |
| DecoupledMLA | `synop_bb8c54c201` | 0.931 → 0.744 |
| DilatedBlockAttention | `synop_bb8c54c201` | 0.860 → 0.725 |
| GRUFusedGate | `synop_bb8c54c201` | 0.762 → 0.643 |
| ChunkedSSD | `synop_03889d8fe1`, `synop_39f76b66f8`, `synop_93e57c595e` | 0.956 → 0.864 |
| RWKVMixBlock | `synop_bb8c54c201` | 0.959 → 0.909 |
| HighwayGate | `synop_bb8c54c201` | 0.943 → 0.945 |

## Verdict

**The plumbing is delivered; the pay is not — and now it is measured
*against* the fused path, not the spec interpreter.** Before this
change a folded `synop_*` member could never win extraction on the zoo
path (no sink lowered it); now 7 of 46 losing rows extract verified
`synop*` terms and every one of them runs *slower* than the spelled
source — the prim-row median fell 0.943 → 0.744 while the 39 untouched
rows are flat at 0.947. `synop_bb8c54c201` (the GLA fold) dominates the
delivered set. This is the same verdict `prim_delivery.md` reached on
the spec-walk binding, reproduced on the fused-body codegen: at these
shapes the minted fold's one-dispatch billing does not survive contact
with wall-clock. It is also direct evidence for T-019 — extraction now
*routinely prefers* members that lose measured time, so the fold cost
gate in `profitable` is the binding constraint, not deliverability.

Caveats: fp64 only (the fp32 loser set was not re-swept); per-row noise
on an occupied box is ~±0.05, but the prim-row median move (−0.20) is
far outside it. Delivery correctness is gated inside the timing path —
`gen_probe` verifies every delivered module before timing it, and all
46 rows report `verified: true`.

## Next

1. **T-019 is now the pay lever**: `profitable` admits folds on the
   cost model; the zoo says the admitted set loses wall-clock. A
   measured gate (time delivered vs spelled once per admitted claim)
   would have kept all 7 of these rows on their pre-merge extractions.
2. Re-sweep the fp32 loser set (37 rows, `zoo_chain_sweep_fp32.json`)
   under the same merge before quoting a dtype-general number.
