# Retro — premise-anchored `derivable`: the composite's minting proof counts

**Date**: 2026-10-19.  **Follows**: `binding-env-cap.md` (its named
policy question: "should `corpus.base_rules` carry a stored object's
premise closure, so `derivable` reflects the premise-anchored proof?")
and `lemma-cert-admission.md`.  **Commit**: this commit's parent —
`evidence: premise-anchored derivability for stored composites`.

## The decision

**Yes — with a site-verification guard.**  A stored composite's
`derivable` leg now evaluates against `corpus.base_rules` plus its
**transitive premise closure**: the record's `guard.parents` field
resolved through the store member-wise — the same resolution
`admit_object` performs for guard transport (`ca42c16`): each premise
must be a row of the same store that rebuilds at *full declared
strength* (`missing_hooks` minus `transported` empty), or — when it
names a base rule — the axiom itself.  A member that resolves short of
that bar (no row, weaker-than-declared admit, cycle) empties the whole
closure: a partial premise set is not the derivation the object was
minted under.

The alternative — axioms-only `derivable` — is *sound but blind*: it
under-reports a composite whose entire honest claim is "derivable
under the premises I was minted from".  The alternative the question
warns against — unconditionally trusting the closure — is circular:
`verify_law`'s certificate proves each step is a valid *application*
of a named rule, not that the named rule is *true*.  A false stored
rule in the closure would let a false composite fake derivability.

## The anti-gaming bar

The anchored credit is granted only when the reported certificate is
standalone-replayable **and every step naming a non-axiom rule
site-verifies**: `eval_instance(step.lhs, step.rhs)` on the recorded
concrete instance — the exact terms the premise rewrote between —
under the same fp64 random-env oracle `num_true` uses.  A premise true
at the site the proof uses is sufficient (`derivable` is a
per-instance verdict); a premise false at its applied site, an
unverifiable step, or a non-replayable witness all revoke the credit —
`measure` tries `base_rules` first, so the revoked verdict falls back
to the honest axiom-only answer.  The flag surfaces as
`Evidence.premise_anchored` and `derivable=True(premise-anchored)` in
the truth-gate detail.

## Re-measured: the conv+BN composite

`lab/probes/conv_fold_premise_pay_probe.py`, same mint (974 emitted,
8-rule premise closure, one-case corpus):

| stage | before (`0721cec`) | now |
|---|---|---|
| measure | `derivable=False` under `ALL_RULES` | `derivable=True(premise-anchored)` — replayable cert, premise steps `conv_channel_shift_sp` + `syn_…_209` site-verify |
| truth | **FAIL** (starved sweep, no waiver) | pass — `numeric=True`, `1eq/0ne` real site, derivable waives starvation |
| typed-pay | not reached | pass — `fires=1 paid=1 verify_fail=0` |
| closure | not reached | **FAIL** — `enode=6.00x` vs the 2.0 cap |

The gauntlet now refuses at **closure**, not truth: the fold expands
the real conv+BN term into its spelled form inside the probe e-graph
(6× enode growth on a one-term graph).  That is a *different*, honest
gate — expansion cost — and the correct residual blocker: the object
is proved (premise-anchored) and pays (17 438.7 vs 34 874.3), but its
closure safety is a corpus-level property the anchor does not touch.
Whether an expansion composite should carry a different closure bar is
the next named policy question — deliberately not decided here.

## What the tests pin

`tests/test_discovery_gauntlet.py` (+6): the flip on a minted
`square→pow` composite under an axiom-thin corpus; transitivity
through a stored `syn_*` intermediate; refusal when a premise row is
absent or admits weaker than declared; axiom-parents resolving from
`base_rules` without a store row; a **false-premise pair** that merges
the e-graph yet fails site verification; and the audit's unit edges
(no cert, non-replayable, axioms-only).

## What stays open

- The conv+BN composite's `closure` refusal (6.00× > 2.0 cap) —
  expansion-composite policy, not derivability.
- `premise_anchored` does not reach the verdicts table: the gauntlet
  does not record verdict rows, so nothing persists a stale-anchored
  `derivable`; a future cached-verdict path for stored objects must
  carry the anchor scope in its key.

Gate account: `tests/test_discovery_gauntlet.py`,
`test_discovery_evidence.py`, `test_discovery_pipeline.py` green;
ruff/format/ty clean on touched files.
