# Retro — the `$leaf:` auto-compose seam + the `batch_norm` LHS law

**Date**: 2026-10-18.  **Closes**: the two named limits in
`project/adrs/0012-tensor-derives.md` / `GOALS.md` §"Still open" —
(a) `$leaf:` rules can't auto-compose in `meta.synthesize_rules`
(the `@i:` placeholder machinery was attr-only), and (b) no law LHS
named `batch_norm` (`weightspace-fold.md` gate 5 — content, not
capability).

## Item 1 — term-level placeholders

The ADR's resolver seam, landed as the *twin* of the attr machinery
rather than a grammar widening.  A `$leaf:`-produced RHS term metavar
now travels through composition exactly the way a derived attr does:

- `_synthesizable` admits a rule whose RHS term metavar is unbound
  when `rule.derive` can produce it (the attr clause's mirror; the
  earlier blanket `return False` was the measured refusal).
- `_fire_guarded` keeps *every* derive output symbolic under the `@i:`
  namespace — `"$attr:"` keys become `"@i:X"` attr placeholders, bare
  (`$leaf:`) keys become `"@i:X"` *leaf* placeholders — and records
  the concrete value in `conc` either way.
- `_concretize_attrs` resolves placeholder *leaves* (not just attr
  values) so a second parent's guards/derives see the minted `Param`/
  `Const`, never the string.
- `_compose_guards` namespaces term outputs identically:
  `eff1["@1:X"]` lets a `pat2` entry reference the parent's derived
  leaf, and the composite `derive` emits `out["@i:X"]` — the
  instantiate-time resolution the ADR asked for, re-running the
  parent's spec through the pushed `TensorAlgebra` (`source`/`ops`/
  `mint`) per firing.  Nothing is baked in — the earlier silent bake
  of a concrete `Param` into a derived RHS (reachable only by
  cheating `_synthesizable`) is structurally impossible now.
- `_rhs_derive_placeholders` scans leaf positions too, and
  `_validate_candidate` admits `@i:`-prefixed RHS term metavars when
  the candidate derives — unresolved placeholders veto the firing,
  which validation measures as refusal.

Nested composition namespaces correctly by construction
(`"@1:@2:X"`), the same convention attr placeholders already use.

**Measured.**  `tests/test_batch_norm_expand.py`:
`synthesize_rules([mark→marked(a,eps:$leaf:), marked(u,k)→sink(u,k)],
[mark(v)])` emits a candidate whose RHS carries `"@1:eps"` and whose
firing lands `Const(2.0)` — before the seam the parent was filtered
by `_synthesizable` (refusal count 1 → 0).  The twin seats are both
covered (`"@2:eps"` when the `$leaf:` rule is r2), and
`batch_norm_expand` itself — a real `$leaf:` rule — composes.

## Item 2 — the `batch_norm` law family

`laws/tensor.py` gains `BATCH_NORM_RULES` — three arity variants of
the eval-mode expansion `batch_norm → γ·((x−μ)·rsqrt(σ²+ε)) + β`
(affine / weight-only / affine-free — the operand arities aten
exports after dropping `None`s).  Everything is data:

- `cond`: `training=False` (eval only — training computes batch
  stats, different math), concrete `x`/`rm`, channel-extent and
  per-channel shape agreement, `2 ≤ rank(x) ≤ 6`.
- `dspec`: `CH = (1,) + shape(rm) + take(ones, rank(x)−2)` — the
  `(1, C, 1…)` broadcast target derived per-rank, one law covering
  ranks 2–6 (`take` is Python-clamped; the `≤ 6` guard is what keeps
  a longer rank from minting a short target silently); `$leaf:EPS`
  mints `Const(eps)` through the scalar side of ADR 0012's channel —
  no tensor algebra needed.

**Held out of DEFAULT** as an opt-in list — the admission floor (the
guarded sweep the promoted laws cleared) has not run; `ALL_RULES` is
unchanged, extraction behavior for every existing caller unchanged.
The fused member stays cheapest under extraction (priced 4 <
the ~7-pass spelled subgraph), so the family is optionality, not a
regression, wherever it is opted in.

**Measured.**  fp64-exact vs `F.batch_norm` through the torch bridge
(max|Δ| = 7.3e-13 on (2,6,4,4)); `training=True`, stats/x channel
mismatch, rank-1 and rank-7 inputs, and unshaped `x` all decline —
the strictness contract, never a fabricated member.

## What stays open

- The conv+BN *weight fold* itself: `W' = W·γ·rsqrt(σ²+ε)`,
  `b' = (b−μ)·scale + β` — now reachable (stats are visible as terms,
  `tval`+`mint` can mint the folded `Param`), but the law is
  un-authored — it wants the tensor-valued `$leaf:` parents to
  auto-compose, which this seam enables; a follow-up measurement
  should try `synthesize_rules`/`dspec_synth` over a real
  conv→batch_norm seed rather than hand-writing it.
- `DEFAULT` admission for `BATCH_NORM_RULES` — pending the guarded
  sweep (off this box's budget).
- The mint-determinism contract remains documented-not-enforced
  (ADR 0012 (b)).
