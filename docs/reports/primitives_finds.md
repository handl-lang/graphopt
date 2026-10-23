# Primitive finds — what the machine actually minted

`overnight.py --corpus zoo-prim` (68-module corpus, 2026-10-23) mined recurring
manual subgraphs, synthesized a new `synop_*` primitive for each, and minted the
fold `manual_spelling -> synop_*`. All 21 admitted folds classify
**machine-original**: certified, measured paying, underivable from the 242-rule
bank — novel by construction, since no catalog contains laws about an op that
did not exist.

## What the primitives are

Decoded from `results_overnight_prim_v2.json` `prim_ops` (alpha-normalized
patterns, bind counts across distinct zoo modules):

| class | synops | shape | modules |
|---|---|---|---|
| gated-accumulator tree | `5e58f26875` `4f8fefdd38` `5147826430` `d5b128282a` `39f76b66f8` `571fb768b0` `62d36aefdb` `629f9ef13b` `2b8cfd1c90` `f3290a1163` `6c709f774a` `03889d8fe1` `b246dcb838` `889d740672` `93e57c595e` | `add(mul .., add(mul .., ..))` / `mul v0 (add ..)` — sum-of-gated-products skeletons in every bracketing | GatedDeltaRule, GatedLinearAttention, MambaSelectiveScan, RealGatedLRU |
| broadcast select-gate | `5f1bba5798` `336e34145b` `00cd48d9a2` | `mul(unsq(select v dim1), unsq(select(linear ..) dim1))` — the SSM recurrence step | DeltaRuleBlock + the same 4 |
| residual + broadcast linear | `4c657cb2c0` | `add v0 (mul (unsq v1) (unsq (select (linear ..) ..)))` | GatedLinearAttention, MambaSelectiveScan |
| gated-add + broadcast | `976f9836d7` `f08a53f9a0` `4c5296b3ba` `98a002d64b` | mul-add tree with an `unsq` broadcast leaf | 3 SSM-family modules |
| **sum of two products** | `bb8c54c201` | `add(mul v0 v1, mul v2 v3)` | **14 modules** |

## The honest reading

- `bb8c54c201` binds 14 modules because `a*b + c*d` is the most common
  elementwise skeleton in the corpus — gates, residuals, bilinear terms. The
  machine named it as a primitive and the fold pays (launch 0.80x, one op
  instead of three).
- The SSM cluster is the interesting find: **four different published
  architectures** — DeltaNet, Gated Linear Attention, Mamba selective scan,
  Griffin's RG-LRU — share the same "sum of gated products" computational
  core, and the machine found that shared skeleton by corpus antiunification,
  not by being told it exists. Sixteen depth/bracketing variants of it survive
  the full gauntlet as one named primitive each.
- Every fold is verified fp64-equal at all bound sites (16/16 equal per
  primitive), cert is replayable (`premise-anchored: synop_*.expand`), and pay
  is measured on real lowering.

## What "machine-original" means here — and does not

- **Is**: certified + beats the registered baseline + the op is new, so no
  catalog *can* contain the fold. The closed-world claim holds.
- **Is also**: a real discovery in the weak sense — the shared SSM skeleton is
  a true structural fact about the corpus that no human encoded for it.
- **Is not**: "unknown to humanity". Elementwise mul-add fusion is what every
  fuser does implicitly; the 14-module `a*b+c*d` is FMA-pair-shaped. What is
  new is the *named primitive + certified law* form, not the semantic.
- **Open**: whether any `synop_*` corresponds to something a fused-kernel
  engineer would call a real new kernel. The 8 handler-refused cases (no
  `ew`-family implementation verified within budget) are the honest frontier —
  a minted primitive with a working fused body is the claim to chase next.

## Caveats

- All 21 instances record `corpus:MambaSelectiveScan` / `corpus:HighwayGate` —
  the reported site is the first bound module, not the full bind set (the
  `mods` lists above are the real cross-module counts).
- Arena verdict `false` for all folds: the arena cannot eval unbound patterns;
  the gauntlet is the judge (documented in the driver).

## Shipped

The 21 admitted folds are no longer process-local: they ship in
`mlopt_discovery.shipped_primitives` — the verbatim mint records
materialize the `prim:synop_*` fold laws (each carrying its
`synop_*.expand` definitional premise), `register()` seats the op
definitions and the torch denotations (the binding replays the spelled
subgraph — sound by construction), and `prim_default()` deploys the
set through `search`'s `rules=` seam.  They stay **opt-in** — never
in core `DEFAULT` — because a `synop_*`'s meaning is a torch binding
and pushing them into the torch-free substrate (or `DEFAULT`) would
change extraction for consumers that can never lower the ops.
Domain-side provenance ledger: `shipped_primitives.MACHINE_MINTED`
(prov=machine).  See the "Shipped" note in
`results_overnight_prim_v2.md` for the re-measured deliverability.
