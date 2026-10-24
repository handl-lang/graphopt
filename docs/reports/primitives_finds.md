# Primitive finds — what the machine actually minted

`overnight.py --corpus zoo-prim` (68-module corpus, 2026-10-23) mined recurring
manual subgraphs, synthesized a new `synop_*` primitive for each, and minted the
fold `manual_spelling -> synop_*`. All 21 admitted folds classify
**machine-original**: certified, measured paying, underivable from the then
242-rule bank (`known_bank()` now counts 265 — the folds below seated) — novel
by construction, since no catalog contains laws about an op that did not exist.

*Since this file was written* (the depth-1 run, `results_overnight_prim_v2`/`v4`):
the widened `--prim-deep` hunt admitted **48/48** deeper shapes — all
machine-original, disjoint names — and `--prim-bootstrap`, which seats the
shipped folds in the corpus e-graph as vocabulary, admitted **2 depth-2
`synop2_*` folds** (`results_overnight_prim_deep.md`,
`results_overnight_prim_bootstrap.md`; the bootstrap run's other 46 re-admit the
deep set — **71 distinct folds across runs**).  23 folds (21 + 2) ship in
`DEFAULT`.

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
  engineer would call a real new kernel.  The index-family wall this used to
  name has fallen — twice: the v4 `gather` handler family mints verified
  bodies for **24/24** (the 4 select/linear refusals included), and
  `family_synth.mint_families` then recovered `gather` *itself* as a minted
  descriptor (`fam_8c9a7ed8`, parity-verified)
  (`docs/reports/prim_handlers.md`, `docs/reports/family_synth.md`).

## Caveats

- All 21 instances record `corpus:MambaSelectiveScan` / `corpus:HighwayGate` —
  the reported site is the first bound module, not the full bind set (the
  `mods` lists above are the real cross-module counts).
- Arena verdict `false` for all folds: the arena cannot eval unbound patterns;
  the gauntlet is the judge (documented in the driver).

## Shipped

The admitted folds are no longer process-local — they ship
**default-on**: the 23 fold laws (21 `prim:synop_*` + 2
`prim:synop2_*`) and their 23 `.expand` definitional premises embed
verbatim in `mlopt_core.laws.minted` (`_PRIM_RECORDS` +
`_PRIM2_RECORDS` — torch-free pattern data, the laws not the
semantics) and ride `MINTED_RULES` into `ALL_RULES` / `DEFAULT`, so
`search(rules=DEFAULT)` fires them with no opt-in.  The
`MACHINE_SHIPPED` provenance ledger counts **59 rows** (13 laws + 23
folds + 23 premises); the `.expand` premises stay off the saturation
lists — they are the cert leg's anchored-derivation axioms, carried on
`rule.premises` (re-expanding a `synop_*` re-adds the unfolded
spelling the fold collapses).

What stays domain-side is exactly the piece a torch-free core cannot
ship — the **denotation**: `mlopt_discovery.shipped_primitives`'s
`register()` seats one torch binding per op on the ambient
`torch_bridge._IR_TO_TORCH` table (the binding replays the spelled
subgraph — sound by construction; a `synop2_*` binding spells the
fully re-expanded manual form), so every `OpTable.full()`-backed
`TorchSink` lowers them.  Domain-side ledger:
`shipped_primitives.MACHINE_MINTED` (prov=machine).  See the "Shipped"
note in
`results_overnight_prim_v2.md` for the re-measured deliverability.
