# Retro: grammar gaps — closing the measured refusals, and the pay wall

Date: 2026-10-14
Context: the ADR-0007 layer-2 handler grammar
(`catopt_discovery.handler_synth`) mints a non-elementwise atom by
proposing a descriptor from a structured template vocabulary and
verifying it fp64 against the pre-mint spec.  The composed loop
(`tools/atom_synth_loop.py`) runs need -> handler -> laws ->
admission -> pay.  Five refusals were measured; this round closes
four and names the fifth precisely.

Changed code lives in `packages/catopt-discovery/src/catopt_discovery/handler_synth.py`
plus its test file; `tools/probes/grammar_probe.py` is the census
instrument.  Nothing under `catopt-core` changed.

## The refusals, before -> after (mint verdict at budget 512)

| spec | before | after |
|---|---|---|
| `cumsum(X, dim=1)` verbatim | 0 candidates | MINTED `("fold_emit",1.0,"X",0.0,"$acc",-2)` tried=3 |
| `H0 + cumsum(X, dim=1)` | 0 candidates | MINTED `("fold_emit",1.0,"X","H0","$acc",-2)` tried=3 |
| `sum(X, dim=-1)` | 0 candidates | MINTED `("reduce","sum","X",-1)` tried=35 |
| `mean(X, dim=-1)` | 0 candidates | MINTED `("reduce","mean","X",-1)` tried=37 |
| `softmax` | 0 candidates | MINTED `("norm","div",("exp","X"),"sum",("exp","X"),-1,True)` tried=38 |
| SelectiveScan verbatim (2 metas) | 512-budget refusal | MINTED `("fold_emit",("sigmoid","A"),"B",0.0,"$acc",-2)` tried=44 |

The extensions, each verified through the unchanged fp64 gate:

* **`reduce` family** — `("reduce", op, sub, axis)` over
  sum/mean/prod/amax/amin on the spec's own `dim` attrs plus -1/-2.
* **identity sides** — an empty fold a/b partition spells its side's
  identity (1.0 for a, 0.0 for b); with the scalar-operand lift
  (`_as_tensor`, commit 311fac1) these are live again, so a
  carry-less recurrence is spellable.
* **neutral carries** — a fold may start from 0.0/1.0
  (`FOLD_CARRIES`), so a spec with no explicit carry meta reaches its
  canonical form.
* **`norm` family** — `("norm", op, lhs, red_op, rhs, axis, keepdim)`,
  a value combined with a reduction of (another) value; the composite
  the `reduce` family's whole-spec operand cannot spell.

## The budget heuristics, confirmed not relaxed

The `fold`/`fold_emit` **family** keeps its `>= 3` meta floor: the
one/two-meta mints all come from the `fold_emit` **seed**
(`_canonical_emit_seeds`), and relaxing the floor only explodes the
space — `cumsum(X)`'s `fold_emit` column grows from 36 (the seed) to
147700 candidates, a two-meta spec's from 144 to >200000, every
battery spec mints the *same* body at the *same* index, and the
`applyd`-nest `fold_zip` mint regresses out of a 1500 budget.

The emit projection depth is capped at one non-accumulator op
(`_MAX_EMIT_OPS`): the emitted-scan corpus emit is `$acc` and the
next useful form is one op; a two-op projection is outside the cap.

## Dead space

The old dead region (bare-float fold fillers) was already lifted.
This round found and removed two more: a **non-`keepdim`
normalization** cannot broadcast its denominator back, and the
**both-identity fold** (`a=1.0, b=0.0`) has no step axis.  Measured
on the cumsum spec: 143 of the first 600 candidates raised, 141 of
them non-keepdim norm bodies; after deriving `keepdim` off the spec
(`_spec_keepdim`), gating the norm family to normalization-shaped
specs, and dropping the both-identity forms, **0 raise**.  The
residual raises on two-meta specs are shape-incompatible `contract`
einsums — a domain-shape refusal, not dead space.

## The pay wall

The driver admits every handler need (cumprod 3 laws, cumsum 8,
logcumsumexp 1, scan_fold 2) but **no atom pays** (delta 0.0 on
every corpus term):

* **`scan_fold` — fire 0.**  Its only law that connects the atom to
  the corpus site, `sf.corpus_refold`, is `rejected-undeclarable`
  (`envs=164 eq=3 bad=1 real=1`): "no declarable conjunction of <= 3
  covers 35 equal / 42 bad sites".  The two certified laws
  (`sf.scale`, `sf.split_same`) are algebraic identities with no real
  sites, so nothing fires.  This is the **stabilization gap**.
* **`cumsum` — fire 1, delta 0.**  One certified law
  (`cum.add_const`, the `cumsum(add(x, c))` shape) fires once on one
  corpus term, but its rewrite is cost-neutral: it does not lower the
  extracted FLOPs.  `cumsum` is ambient (already bound), so minting
  it adds no extraction power either.

## The stabilization gap, measured

The corpus SelectiveScan site is
`mul(cumprod(sigmoid(A)), cumsum(B / clamp_min(cumprod(sigmoid(A)), 1e-6)))`;
the minted atom is the *ideal* unclamped recurrence
`fold_emit(sigmoid(A), B, 0.0, $acc, -2)`.  Evaluating both in fp64
across a sweep of the gate magnitude:

| A-scale | min gate | clamp active | mint err |
|---|---|---|---|
| 1.0 | 1.34e-4 | no | 4.4e-16 |
| 5.0 | 3.17e-13 | yes | 2.57e+0 |
| 10.0 | 1.95e-25 | yes | 2.55e+0 |
| 30.0 | 8.07e-75 | yes | 3.09e+0 |

The mint is exact while the cumulative gate stays above `1e-6` and
diverges by O(1) the moment it underflows.  The verification domain
(`randn` A -> sigmoid ~ 0.5, gate ~ 1e-4) sits in the inactive
region, so the mint passes; the real site's gate underflows, so the
recognition law is false there.

**Can the grammar express a stabilized fold?  No — not as an affine
recurrence.**  The clamp acts on the *accumulator* (the cumulative
product), not on a fixed operand: the stabilized scan is a
two-state, non-linear recurrence

    P_t = clamp_min(sigmoid(A_t) * P_{t-1}, eps)
    S_t = S_{t-1} + B_t / P_t
    out_t = P_t * S_t

whose state update is non-affine.  The `fold`/`fold_emit` semantics
is `acc = a_t * acc + b_t` with a_t, b_t fixed elementwise exprs;
expressing the clamp needs a **non-linear update function** and a
**second coupled state** — a semantics extension, not a vocabulary
one.  The atom must therefore stay ideal, and the corpus's clamped
site is outside its reach.

## Residue

* Stabilized (clamped) scans — a non-affine, two-state recurrence;
  the affine fold grammar cannot close it (above).
* `log_softmax` (a real invisible corpus op, 2 sites) — a
  normalization whose reduction result is wrapped in a unary
  (`log . sum . exp`) or a `logsumexp` reduction; the norm template
  covers a bare reduction operand only, and `logsumexp` has no torch
  binding.
* Cost-neutral recognition: a certified law that fires without
  lowering extracted cost (the cumsum case) is an honest zero, not a
  bug — the atom's value would be its *execution*, which an ambient
  binding already provides.
